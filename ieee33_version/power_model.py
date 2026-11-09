"""恢复优化模型：变量、目标函数、网络/SOC/启动/信息约束和求解器。

阅读顺序：第一部分通用稀疏矩阵与求解器；第二部分build_model逐块建模。
功率MW，能量MWh，时间h，费用千元。主程序通过analysis调用本模块。
"""
from __future__ import annotations
from data import build_profiles
from network import interface_coefficients
from repair_model import build_scenarios, schedule_policy

# ==================== 第一部分：稀疏模型与MILP/LP求解 ====================
from dataclasses import dataclass, field
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, linprog, Bounds, LinearConstraint


@dataclass
class LinearModel:
    c: np.ndarray
    integrality: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    A: sparse.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    var_indices: dict
    metadata: dict=field(default_factory=dict)
    objective_constant: float=0.0


class ModelBuilder:
    """按有名变量块及稀疏行组装模型，避免手工大矩阵下标混乱。"""
    def __init__(self):
        self.c=[];self.lb=[];self.ub=[];self.integrality=[];self.var_indices={}
        self.rr=[];self.cc=[];self.vv=[];self.row_lb=[];self.row_ub=[];self.row_names=[]
        self.metadata={};self.objective_constant=0.0

    def add_var(self,name,shape,lb=0.0,ub=np.inf,integer=False,cost=0.0):
        """广播变量界与费用；返回与shape同形的全局列号。"""
        shape=(shape,) if isinstance(shape,int) else tuple(shape)
        size=int(np.prod(shape));idx=np.arange(len(self.c),len(self.c)+size).reshape(shape)
        for target,value in ((self.c,cost),(self.lb,lb),(self.ub,ub)):
            target.extend(np.broadcast_to(value,shape).ravel().tolist())
        self.integrality.extend([int(integer)]*size);self.var_indices[name]=idx
        return idx

    def add_row(self,coeff,lb=-np.inf,ub=np.inf,name=""):
        """加入lb<=Σa_j x_j<=ub；coeff为{列号:系数}。"""
        row=len(self.row_lb)
        for col,val in coeff.items():
            if abs(val)>1e-14:
                self.rr.append(row);self.cc.append(int(col));self.vv.append(float(val))
        self.row_lb.append(lb);self.row_ub.append(ub);self.row_names.append(name)

    def build(self):
        """输出SciPy CSR，保留约束名称便于定位不可行或数值偏差。"""
        A=sparse.coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.row_lb),len(self.c))).tocsr()
        self.metadata["row_names"]=self.row_names
        return LinearModel(np.array(self.c),np.array(self.integrality),np.array(self.lb),np.array(self.ub),A,
                           np.array(self.row_lb),np.array(self.row_ub),self.var_indices,self.metadata,self.objective_constant)


def solve(model,solver_cfg=None):
    """求解MILP，限时但有incumbent仍返回可行解；未证明最优则保留间隙。"""
    cfg={"time_limit":25.0,"mip_rel_gap":0.005,"presolve":True,"disp":False}
    cfg.update(solver_cfg or {});start=time.perf_counter()
    result=milp(model.c,integrality=model.integrality,bounds=Bounds(model.lb,model.ub),
                constraints=LinearConstraint(model.A,model.row_lb,model.row_ub),options=cfg)
    x=result.x
    def finite(v,default):
        return float(v) if v is not None and np.isfinite(v) else default
    obj=finite(getattr(result,"fun",None),np.inf)+model.objective_constant
    lower=finite(getattr(result,"mip_dual_bound",None),-np.inf)+model.objective_constant
    # 纯LP或空分支有时未提供mip_dual_bound；已证最优时目标就是有效下界。
    if result.status==0 and not np.isfinite(lower):lower=obj
    output={"success":bool(result.success),"status":int(result.status),"message":result.message,"x":x,
            "feasible_plan_source":"MILP_incumbent" if x is not None else None,
            "objective":obj,"lower_bound":lower,"mip_gap":finite(getattr(result,"mip_gap",None),None),
            "solve_seconds":time.perf_counter()-start,"node_count":int(getattr(result,"mip_node_count",0) or 0),
            "absolute_objective_gap":float(obj-lower) if np.isfinite(obj-lower) else None,
            "relative_objective_gap_including_constant":float((obj-lower)/max(abs(obj),1e-9)) if np.isfinite(obj-lower) else None}
    # 限时无incumbent不等于物理不可行。用受限接线/启动/电池模式LP
    # 寻找一个保守可行计划，仅作上界；不修改原MILP的下界或最优性状态。
    # 滚动有历史锁定，继续使用rolling.py的原计划回退，不套用新初始计划。
    if x is None and result.status==1 and not model.metadata.get('rolling_fixed_prefix'):
        seed=restricted_feasible_plan(model)
        if seed is not None:
            output.update(x=seed['x'],objective=seed['objective'],
                message=str(result.message)+'；采用保留全部物理约束的固定离散方案LP可行回退；仅为上界',
                feasible_plan_source='restricted_discrete_LP',
                solve_seconds=time.perf_counter()-start)
            obj=seed['objective']
            output['absolute_objective_gap']=float(obj-lower) if np.isfinite(lower) else None
            output['relative_objective_gap_including_constant']=float((obj-lower)/max(abs(obj),1e-9)) if np.isfinite(lower) else None
    return output


def restricted_feasible_plan(model):
    """固定一个初始可用的径向接线、电池放电模式和共同启动时刻求LP。

    start/free取当前启动，wait取下一允许时刻。所有场景使用相同离散
    指令，原96时段、场景、SOC、网络、终端与信息约束完全保留。
    该额外限制只缩小可行集，因此得到的费用仅是原问题的可行上界。
    找不到此受限方案也不表示原问题不可行。
    """
    md=model.metadata
    if 'network' not in md:return None
    topologies=md['network']['topologies'];faults=list(md['network']['fault_edges'].values())
    candidates=[k for k,z in enumerate(topologies) if all(not z['statuses'][e] for e in faults)]
    if not candidates:return None
    # 优先选联络通道较多、仍保持径向的初始可用方案。
    edges=md['network']['edges']
    if md.get('fixed_switches'):
        candidates=[k for k in candidates if not any(z and edges[e]['kind']=='tie' for e,z in enumerate(topologies[k]['statuses']))]
    if not candidates:return None
    candidate=max(candidates,key=lambda k:sum(z and edges[e]['kind']=='tie' for e,z in enumerate(topologies[k]['statuses'])))
    lb=model.lb.copy();ub=model.ub.copy()
    for name in ('battery_mode','topology_choice','target_start'):
        ids=model.var_indices[name];values=np.zeros(ids.shape)
        if name=='topology_choice':values[:,candidate,:]=1
        if name=='target_start':
            slot=1 if md['branch']=='wait' else 0
            if not md.get('coordination',True) and md['branch']!='start':slot=ids.shape[-1]
            if slot<ids.shape[-1]:values[:,slot]=1
        if np.any(values<lb[ids]-1e-8) or np.any(values>ub[ids]+1e-8):return None
        lb[ids]=values;ub[ids]=values
    trial=LinearModel(model.c,np.zeros_like(model.integrality),lb,ub,model.A,model.row_lb,model.row_ub,
                      model.var_indices,model.metadata,model.objective_constant)
    solved=solve_lp(trial)
    if not solved.get('success') or solved.get('x') is None:return None
    xx=solved['x'];act=model.A@xx;mask=model.integrality!=0
    error=max(np.max(model.row_lb-act,initial=0),np.max(act-model.row_ub,initial=0),
              np.max(model.lb-xx,initial=0),np.max(xx-model.ub,initial=0),
              np.max(np.abs(xx[mask]-np.rint(xx[mask])),initial=0))
    return solved if error<=2e-5 else None


def freeze_discrete(model,x):
    """固定已校核解的整数变量得到条件LP；不改变连续跨时段约束。"""
    x=np.asarray(x);mask=model.integrality!=0
    if np.max(np.abs(x[mask]-np.rint(x[mask])),initial=0)>1e-5:
        raise ValueError("离散变量尚未满足整数性，不能固定生成能力域")
    lb=model.lb.copy();ub=model.ub.copy();lb[mask]=np.rint(x[mask]);ub[mask]=lb[mask]
    return LinearModel(model.c.copy(),np.zeros_like(model.integrality),lb,ub,model.A,
                       model.row_lb,model.row_ub,model.var_indices,{**model.metadata,"discrete_fixed":True},model.objective_constant)


def solve_lp(model,objective=None,extra_equalities=None):
    """解连续LP；extra_equalities=[({列:系数},右端),...]用于接口轨迹分解。"""
    if np.any(model.integrality):
        raise ValueError("solve_lp仅接受连续模型；请先freeze_discrete")
    c=model.c if objective is None else np.asarray(objective,dtype=float)
    finite_both=np.isfinite(model.row_lb)&np.isfinite(model.row_ub)
    eq=np.zeros(len(model.row_lb),dtype=bool)
    eq[finite_both]=np.abs(model.row_lb[finite_both]-model.row_ub[finite_both])<1e-10
    lo=np.isfinite(model.row_lb)&~eq;hi=np.isfinite(model.row_ub)&~eq
    Aeq=model.A[eq];beq=model.row_lb[eq]
    Aub=sparse.vstack((model.A[hi],-model.A[lo]),format="csr");bub=np.r_[model.row_ub[hi],-model.row_lb[lo]]
    if extra_equalities:
        rr=[];cc=[];vv=[];rhs=[]
        for i,(row,b) in enumerate(extra_equalities):
            rhs.append(b)
            for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
        new=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(c))).tocsr()
        Aeq=sparse.vstack((Aeq,new),format="csr");beq=np.r_[beq,rhs]
    start=time.perf_counter()
    res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=list(zip(model.lb,model.ub)),method="highs")
    constant=model.objective_constant if objective is None else 0.0
    obj=float(res.fun)+constant if res.fun is not None else np.inf
    return {"success":bool(res.success),"status":int(res.status),"message":res.message,"x":res.x,
            "objective":obj,"lower_bound":obj if res.success else -np.inf,"mip_gap":0.0 if res.success else None,
            "solve_seconds":time.perf_counter()-start,"node_count":0}

# ==================== 第二部分：完整24小时恢复模型 ====================
import json
from collections import defaultdict
import numpy as np



def _get(mapping, *keys, default=None):
    """读取同一参数的兼容字段；输出仍以配置文件中的显式值为准。"""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _history_key(history):
    """将观测历史转成稳定键；未观测的真实损伤不进入这个键。"""
    return json.dumps(history, ensure_ascii=False, sort_keys=True)


def build_model(cfg, policy, branch='free', information='causal',
                coordination=True, fixed_switches=False):
    """从参数和一条因果抢修策略生成完整随机 MILP。

    参数
    ----
    branch : {'free','start','wait'}
        free 自主选择；start 强制当前启动；wait 仅禁止当前启动，未来仍可启动。
        两个条件分支使用完全相同的24小时终点、资源和信息规则。
    information : {'causal','perfect_information'}
        causal 按实际完成的巡检绑定非预期性；完美信息仅是理想参照。
    coordination : bool
        False 时禁止配网向输电网反送，其他约束不变，形成嵌套的协同比较。
    fixed_switches : bool
        True 时保持五条联络线常开，用作联络重构动作消融。

    返回
    ----
    LinearModel
        含稀疏矩阵、变量索引和物理元数据。可交给 HiGHS MILP，或固定
        整数变量后用于全时域条件能力域投影和可行轨迹分解。
    """
    if branch not in {'free', 'start', 'wait'}:
        raise ValueError('branch 必须为 free、start 或 wait')
    if information not in {'causal', 'perfect_information'}:
        raise ValueError('information 必须为 causal 或 perfect_information')

    # 01 读取24小时连续调度曲线与四种真实损伤情景。
    prof = build_profiles(cfg)
    dt = float(prof['dt_hours'])
    n = int(cfg['n_steps'])
    scenarios = build_scenarios(cfg)
    schedules = schedule_policy(cfg, policy, scenarios)
    ns, nm, nb, ne = len(scenarios), len(cfg["microgrids"]), len(cfg["nodes"]), len(cfg["edges"])
    probs = np.asarray([s['probability'] for s in scenarios], dtype=float)
    mgs = cfg['microgrids']
    gen = cfg['generator']
    net = cfg.get('network', cfg.get('grid', {}))
    edges = cfg.get('edges', net.get('edges', []))
    grid = cfg.get('grid', {})
    costs = cfg.get('costs', {})
    base = float(cfg.get('base_mva', 10.0))
    vmin = float(cfg.get('voltage_min', .95)) ** 2
    vmax = float(cfg.get('voltage_max', 1.05)) ** 2
    facets = int(cfg.get('polygon_sides', 8))
    angles = (np.arange(facets) + .5) * 2 * np.pi / facets
    polygon_factor = float(np.cos(np.pi / facets))

    # 02 启动、开关、电池模式三种动作网格；连续出力和SOC仍保留96时段。
    start_step = max(1, int(round(float(_get(gen, 'start_interval_minutes', default=30)) / (dt*60))))
    aux_step = float(_get(gen, 'startup_profile_step_minutes', 'aux_step_minutes', default=15)) / 60
    aux_profile = np.asarray(_get(gen, 'startup_profile_mw', 'aux_profile_mw', default=[.45,.75,.55,.25]), float)
    aux_duration = len(aux_profile) * aux_step
    last_start = min(float(_get(gen, 'last_start_hour', 'start_window_hours', default=8)), n*dt-aux_duration)
    start_slots = np.arange(0, int(np.floor(last_start/dt+1e-8))+1, start_step, dtype=int)
    if len(start_slots) == 0:
        raise ValueError('恢复时域必须长于完整启动辅助功率持续时间')
    nk = len(start_slots)
    event_set = {0}
    # 滚动过滤情景时保留原始动作网格，避免追溯删去已经执行的开关时刻。
    event_set.update(int(t) for t in cfg.get("recovery_event_slots", []) if 0 <= int(t) < n)
    event_set.update(int(t) for t in start_slots)
    for sch in schedules:
        for key in ('inspection_hours', 'repair_hours'):
            event_set.update(int(round(h/dt)) for h in sch[key].values() if 0 <= h < n*dt)
    event_slots = np.asarray(sorted(event_set), dtype=int)
    evmap = np.searchsorted(event_slots, np.arange(n), side='right')-1
    mode_step = max(1, int(round(float(cfg.get('battery_mode_minutes',60))/(dt*60))))
    mode_slots = np.arange(0, n, mode_step, dtype=int)
    mode_map = np.arange(n)//mode_step
    ng, nev = len(mode_slots), len(event_slots)

    pmin = float(_get(gen,'min_mw', 'p_min_mw', default=.3))
    pmax = float(_get(gen,'max_mw', 'p_max_mw', default=3.0))
    ramp = float(_get(gen,'ramp_mw_per_hour', default=2.0)) * dt
    qmax = float(_get(gen,'q_max_mvar', default=1.5))
    aux_coeff = np.zeros((n,nk))
    online_coeff = np.zeros((n,nk))
    cap_coeff = np.zeros((n,nk))
    for t in range(n):
        for k, slot in enumerate(start_slots):
            elapsed = (t-slot)*dt
            if -1e-9 <= elapsed < aux_duration-1e-9:
                aux_coeff[t,k] = aux_profile[min(len(aux_profile)-1,int((elapsed+1e-8)/aux_step))]
            if elapsed >= aux_duration-1e-9:
                online_coeff[t,k] = 1
                # 首个并网时段出力从最小稳定出力开始，随后按斜率释放上限。
                cap_coeff[t,k] = min(pmax, pmin+float(_get(gen,'ramp_mw_per_hour',default=2.0))*(elapsed-aux_duration))

    # 03 目标函数：失供成本常数减去供电价值，再加实际燃料、启机、购电和电池成本。
    # 这保证所有分支采用同一量纲与同一经济目标。终端储备通过硬约束体现。
    b = ModelBuilder()
    demand_c = np.asarray(prof['load_critical_mw'],float)
    demand_g = np.asarray(prof['load_general_mw'],float)
    c_voll = np.asarray(prof.get('critical_voll',
                    np.full(n,float(_get(costs,'critical_voll_per_mwh',default=30.0)))),float)
    g_voll = np.asarray(prof.get('general_voll',
                    np.full(n,float(_get(costs,'general_voll_per_mwh',default=8.0)))),float)
    weight_c = probs[:,None,None]*dt*c_voll[None,None,:]
    weight_g = probs[:,None,None]*dt*g_voll[None,None,:]
    shape_s = (ns,nm,n)
    energy_cap = np.asarray([_get(m,'energy_mwh',default=1.6) for m in mgs],float)
    soc_min = np.asarray([_get(m,'soc_min',default=.15) for m in mgs],float)
    soc_max = np.asarray([_get(m,'soc_max',default=.95) for m in mgs],float)
    ch_cap = np.asarray([_get(m,'charge_mw',default=.55) for m in mgs],float)
    dis_cap = np.asarray([_get(m,'discharge_mw',default=.55) for m in mgs],float)
    dg_cap = np.asarray([_get(m,'dg_max_mw',default=.3) for m in mgs],float)
    dg_cost = np.asarray([_get(m,'dg_cost','dg_cost_per_mwh',default=.9) for m in mgs],float)
    bat_cost = np.asarray([_get(m,'throughput_cost','battery_cost_per_mwh',default=.05) for m in mgs],float)
    v = {}
    v['battery_charge'] = b.add_var('battery_charge',shape_s,ub=ch_cap[None,:,None],cost=probs[:,None,None]*dt*bat_cost[None,:,None])
    v['battery_discharge'] = b.add_var('battery_discharge',shape_s,ub=dis_cap[None,:,None],cost=probs[:,None,None]*dt*bat_cost[None,:,None])
    v['battery_mode'] = b.add_var('battery_mode',(ns,nm,ng),ub=1,integer=True)
    v['energy'] = b.add_var('energy',(ns,nm,n+1),lb=(energy_cap*soc_min)[None,:,None],ub=(energy_cap*soc_max)[None,:,None])
    v['pv_used'] = b.add_var('pv_used',shape_s,ub=np.asarray(prof['pv_mw'])[None,:,:])
    v['dg_power'] = b.add_var('dg_power',shape_s,ub=dg_cap[None,:,None],cost=probs[:,None,None]*dt*dg_cost[None,:,None])
    inv_caps = np.asarray([_get(m,'inverter_mva',default=.95) for m in mgs],float)
    v['inverter_q'] = b.add_var('inverter_q',shape_s,lb=-inv_caps[None,:,None],ub=inv_caps[None,:,None])
    v['served_critical'] = b.add_var('served_critical',(ns,nb,n),ub=demand_c[None,:,:],cost=-weight_c)
    v['served_general'] = b.add_var('served_general',(ns,nb,n),ub=demand_g[None,:,:],cost=-weight_g)
    v['target_power'] = b.add_var('target_power',(ns,n),ub=pmax,cost=probs[:,None]*dt*float(_get(gen,'fuel_cost','cost_per_mwh',default=.65)))
    v['target_q'] = b.add_var('target_q',(ns,n),lb=-qmax,ub=qmax)
    v['target_start'] = b.add_var('target_start',(ns,nk),ub=1,integer=True,cost=probs[:,None]*float(_get(gen,'start_cost',default=8.0)))
    # 每个事件时刻选择一个经检查的径向森林。状态是选择变量的线性映射，
    # 不能在未修复时关闭故障线；联络线闭合可伴随分段开关打开。
    topologies=cfg['topologies'];ntop=len(topologies)
    topo_matrix=np.asarray([z['statuses'] for z in topologies],float)
    v['topology_choice']=b.add_var('topology_choice',(ns,ntop,nev),ub=1,integer=True)
    v['line_status']=b.add_var('line_status',(ns,ne,nev),ub=1)
    v['switch_A'] = b.add_var('switch_A',(ns,nev),ub=1)
    v['switch_B'] = b.add_var('switch_B',(ns,nev),ub=1)
    caps = np.asarray([float(_get(e,'s_mva','capacity_mva','s_max_mva',default=3.0)) for e in edges],float)
    v['flow_p'] = b.add_var('flow_p',(ns,ne,n),lb=-caps[None,:,None],ub=caps[None,:,None])
    v['flow_q'] = b.add_var('flow_q',(ns,ne,n),lb=-caps[None,:,None],ub=caps[None,:,None])
    v['voltage_sq'] = b.add_var('voltage_sq',(ns,nb,n),lb=vmin,ub=vmax)
    alpha = float(cfg.get('alpha', 0.0))
    gpmax = alpha*float(cfg.get('grid_reference_mw',3.0))
    gqmax = gpmax*.5
    grid_price = np.asarray(prof.get('grid_price',np.full(n,float(_get(grid,'cost_per_mwh',default=.55)))),float)
    v['grid_p'] = b.add_var('grid_p',(ns,n),ub=gpmax,cost=probs[:,None]*dt*grid_price[None,:])
    v['grid_q'] = b.add_var('grid_q',(ns,n),lb=-gqmax,ub=gqmax)

    def row(coeff,lo=-np.inf,hi=np.inf,name=''):
        # 小工具仅压缩矩阵构造，不改变物理方程；重复项在此合并。
        b.add_row({int(k):float(x) for k,x in coeff.items() if abs(x)>1e-14},lb=lo,ub=hi,name=name)

    def add(coeff,col,val):
        col=int(col)
        coeff[col]=coeff.get(col,0.0)+float(val)

    # 04 微网物理模型：跨期能量守恒、充放互斥、逆变器P/Q耦合、保护负荷。
    for s in range(ns):
        for m, mg in enumerate(mgs):
            e0=energy_cap[m]*float(_get(mg,'soc_initial',default=.75))
            eterm=energy_cap[m]*float(_get(mg,'soc_terminal',default=.3))
            row({v['energy'][s,m,0]:1},e0,e0,'initial_soc')
            row({v['energy'][s,m,n]:1},eterm,np.inf,'terminal_reserve')
            eta_c=float(_get(mg,'eta_charge',default=.95))
            eta_d=float(_get(mg,'eta_discharge',default=.95))
            for t in range(n):
                ch,dis,e=v['battery_charge'][s,m,t],v['battery_discharge'][s,m,t],v['energy'][s,m,t]
                mode=v['battery_mode'][s,m,mode_map[t]]
                row({v['energy'][s,m,t+1]:1,e:-1,ch:-eta_c*dt,dis:dt/eta_d},0,0,'soc_balance')
                row({ch:1,mode:-ch_cap[m]},hi=0,name='charge_mode')
                row({dis:1,mode:dis_cap[m]},hi=dis_cap[m],name='discharge_mode')
                # PV 与电池共用逆变器，凸多边形为圆形额定容量的内近似。
                for ang in angles:
                    co,si=float(np.cos(ang)),float(np.sin(ang))
                    row({v['pv_used'][s,m,t]:co,dis:co,ch:-co,v['inverter_q'][s,m,t]:si},hi=inv_caps[m]*polygon_factor,name='inverter_polygon')
                if cfg['mg_nodes'][m] in cfg['load']['protected_nodes']:
                    node=cfg['mg_nodes'][m]
                    row({v['served_critical'][s,node,t]:1},demand_c[node,t],demand_c[node,t],'protected_load')

    for node in set(cfg['load']['protected_nodes'])-set(cfg['mg_nodes']):
        for s in range(ns):
            for t in range(n):row({v['served_critical'][s,node,t]:1},demand_c[node,t],demand_c[node,t],'protected_load')

    # 05 外部启动机组：一次启动、完整辅助负荷、在线状态、最小出力与爬坡。
    for s in range(ns):
        row({k:1 for k in v['target_start'][s]},hi=1,name='one_shot_start')
        if branch=='start':
            row({v['target_start'][s,0]:1},1,1,'force_current_start')
        elif branch=='wait':
            row({v['target_start'][s,0]:1},0,0,'forbid_current_start')
        for t in range(n):
            coeff={v['target_power'][s,t]:1}
            for k in range(nk): add(coeff,v['target_start'][s,k],-cap_coeff[t,k])
            row(coeff,hi=0,name='generation_after_start')
            coeff={v['target_power'][s,t]:1}
            for k in range(nk): add(coeff,v['target_start'][s,k],-pmin*online_coeff[t,k])
            row(coeff,lo=0,name='minimum_online_power')
            for sign in (-1,1):
                coeff={v['target_q'][s,t]:sign}
                for k in range(nk): add(coeff,v['target_start'][s,k],-qmax*online_coeff[t,k])
                row(coeff,hi=0,name='online_reactive_cap')
            # 发电机P/Q共享额定视在容量，Q独立上下限仍一并保留。
            target_s=float(_get(gen,'s_mva',default=np.hypot(pmax,qmax)))
            for ang in angles:
                coeff={v['target_power'][s,t]:float(np.cos(ang)),v['target_q'][s,t]:float(np.sin(ang))}
                for k in range(nk):
                    add(coeff,v['target_start'][s,k],-target_s*polygon_factor*online_coeff[t,k])
                row(coeff,hi=0,name='target_generator_polygon')
            # 无停机动作，因此全程在线后不会凭空切回零出力。
            if t:
                row({v['target_power'][s,t]:1,v['target_power'][s,t-1]:-1},-ramp,ramp,'target_ramp')

    # 06 故障可用性与可操作接线：修复后只是“可闭合”，不强迫合闸。
    # 接线菜单限定在有限个径向森林内，所有连通分量均含成网微网。
    for s,sch in enumerate(schedules):
        for j,t in enumerate(event_slots):
            row({col:1 for col in v['topology_choice'][s,:,j]},1,1,'choose_one_radial_forest')
            for e in range(ne):
                coeff={v['line_status'][s,e,j]:1}
                for k in range(ntop):add(coeff,v['topology_choice'][s,k,j],-topo_matrix[k,e])
                row(coeff,0,0,'line_status_from_forest')
                site=edges[e].get('fault_site')
                if site:
                    row({v['line_status'][s,e,j]:1},hi=float(sch['availability'][site][t]),name='repaired_before_close')
            for site in ('A','B'):
                e=cfg['fault_edges'][site]
                row({v['switch_'+site][s,j]:1,v['line_status'][s,e,j]:-1},0,0,'fault_switch_display')
            if fixed_switches:
                # 拓扑消融保留A/B随修复恢复，五个联络开关始终常开。
                for e,edge in enumerate(edges):
                    if edge['kind']=='tie':row({v['line_status'][s,e,j]:1},0,0,'fixed_tie_switch')

    # 07 每15分钟保留全部35节点功率平衡、39条线P/Q与电压。
    tanphi=np.asarray(cfg['load']['reactive_ratio'],float)
    auxpf=float(_get(gen,'aux_power_factor',default=.95))
    auxtan=float(np.tan(np.arccos(auxpf)))
    for s in range(ns):
        for t in range(n):
            pbal=[{} for _ in range(nb)]
            qbal=[{} for _ in range(nb)]
            for e,edge in enumerate(edges):
                fr=edge['from'];to=edge['to']
                ep,eq=v['flow_p'][s,e,t],v['flow_q'][s,e,t]
                add(pbal[fr],ep,-1); add(pbal[to],ep,1)
                add(qbal[fr],eq,-1); add(qbal[to],eq,1)
                status=v['line_status'][s,e,evmap[t]]
                for ang in angles:
                    row({ep:float(np.cos(ang)),eq:float(np.sin(ang)),status:-caps[e]*polygon_factor},hi=0,name='line_capacity_polygon')
                r=float(edge['r_pu'])/base;x=float(edge['x_pu'])/base
                drop={v['voltage_sq'][s,to,t]:1,v['voltage_sq'][s,fr,t]:-1,ep:2*r,eq:2*x}
                bigm=(vmax-vmin)+2*(abs(r)+abs(x))*caps[e]+1e-5
                upper=dict(drop);add(upper,status,bigm)
                lower=dict(drop);add(lower,status,-bigm)
                row(upper,hi=bigm,name='voltage_drop_active_upper')
                row(lower,lo=-bigm,name='voltage_drop_active_lower')
            add(pbal[0],v['grid_p'][s,t],1)
            add(pbal[0],v['target_power'][s,t],1)
            add(qbal[0],v['grid_q'][s,t],1)
            add(qbal[0],v['target_q'][s,t],1)
            for k in range(nk):
                add(pbal[0],v['target_start'][s,k],-aux_coeff[t,k])
                add(qbal[0],v['target_start'][s,k],-aux_coeff[t,k]*auxtan)
            for m in range(nm):
                node=cfg['mg_nodes'][m]
                for key,sign in [('dg_power',1),('pv_used',1),('battery_discharge',1),('battery_charge',-1)]:
                    add(pbal[node],v[key][s,m,t],sign)
                add(qbal[node],v['inverter_q'][s,m,t],1)
            for node in range(nb):
                for key in ('served_critical','served_general'):
                    add(pbal[node],v[key][s,node,t],-1)
                    add(qbal[node],v[key][s,node,t],-tanphi[node])
                row(pbal[node],0,0,'active_nodal_balance')
                row(qbal[node],0,0,'reactive_nodal_balance')
            # 每个已选连通分量仅设置一个电压参考；多个MG并列时不
            # 重复强制所有源母线都为1pu。所有节点已在成网森林内送电。
            for node in sorted({i for topo in topologies for i in topo['roots']}):
                roots=[k for k,topo in enumerate(topologies) if node in topo['roots']]
                upper={v['voltage_sq'][s,node,t]:1};lower=dict(upper)
                for k in roots:
                    add(upper,v['topology_choice'][s,k,evmap[t]],vmax-1)
                    add(lower,v['topology_choice'][s,k,evmap[t]],-(1-vmin))
                row(upper,hi=vmax,name='component_voltage_reference_upper')
                row(lower,lo=vmin,name='component_voltage_reference_lower')
            if not coordination:
                coeff={v['flow_p'][s,e,t]:sign for e,sign in cfg['upward_interface']}
                row(coeff,hi=0,name='no_upward_collaboration')

    # 08 条件非预期性：只有观测历史相同的情景当前动作必须相同。
    # 对所有可控连续资源逐时绑定；潮流由树状网络平衡决定，不额外泄漏信息。
    na_rows=0
    def bind_groups(indices, step, name):
        nonlocal na_rows
        if information!='causal': return
        groups=defaultdict(list)
        for s in range(ns):
            groups[_history_key(schedules[s]['observation_history_by_step'][int(step)])].append(s)
        for group in groups.values():
            for s in group[1:]:
                a=np.asarray(indices[group[0]]).ravel()
                z=np.asarray(indices[s]).ravel()
                for ia,iz in zip(a,z):
                    row({ia:1,iz:-1},0,0,'nonanticipativity_'+name)
                    na_rows+=1
    continuous_mg=['battery_charge','battery_discharge','pv_used','dg_power','inverter_q']
    continuous_bus=['served_critical','served_general']
    continuous_other=['target_power','target_q','grid_p','grid_q']
    for t in range(n):
        for key in continuous_mg+continuous_bus:
            bind_groups(v[key][:,:,t],t,key)
        for key in continuous_other:
            bind_groups(v[key][:,t],t,key)
    for g,t in enumerate(mode_slots): bind_groups(v['battery_mode'][:,:,g],t,'battery_mode')
    for j,t in enumerate(event_slots):
        bind_groups(v['switch_A'][:,j],t,'switch_A')
        bind_groups(v['switch_B'][:,j],t,'switch_B')
        bind_groups(v['topology_choice'][:,:,j],t,'topology_choice')
    for k,t in enumerate(start_slots): bind_groups(v['target_start'][:,k],t,'target_start')

    # 09 记录矩阵和经济常数，使校核、投影、分解和出图共享完全相同的模型。
    model=b.build()
    model.objective_constant=float(dt*np.sum(demand_c*c_voll[None,:]+demand_g*g_voll[None,:]))
    crew_cost=0.0
    crew=cfg.get('crew',{})
    for ps,sch in zip(probs,schedules):
        for task in sch['tasks']:
            kind=task.get('kind','')
            rate=float(crew.get('hourly_cost',0))
            crew_cost+=ps*rate*(float(task['end_hour'])-float(task['start_hour']))
    model.objective_constant+=crew_cost
    model.metadata.update({
        'profiles':prof,'scenarios':scenarios,'schedules':schedules,'policy':policy,
        'branch':branch,'information':information,'coordination':bool(coordination),
        'fixed_switches':bool(fixed_switches),'start_slots':start_slots,
        'event_slots':event_slots,'switch_index_for_t':evmap,
        'battery_mode_slots':mode_slots,'battery_mode_index_for_t':mode_map,
        'target_aux_coeff':aux_coeff,'online_coeff':online_coeff,
        'generation_cap_coeff':cap_coeff,'handover_threshold_mw':float(_get(gen,'handover_threshold_mw',default=.5)),
        'expected_crew_cost':float(crew_cost),'nonanticipativity_rows':na_rows,
        'unit_cost':cfg.get('currency_unit','千元'),'dt_hours':dt,
        'upward_interface':cfg['upward_interface'],'mg_interfaces':cfg['mg_interfaces'],
        'network':{k:cfg[k] for k in ('nodes','edges','mg_nodes','node_coordinates','topologies','fault_edges')},
        'scope':'35节点/39支路；有限径向森林及抢修策略菜单；24h随机MILP；线性无损网络；不含暂态',
    })
    # Builder 的标准索引与这里的形状保持一致，方便外部按名称取解。
    model.var_indices.update(v)
    return model
