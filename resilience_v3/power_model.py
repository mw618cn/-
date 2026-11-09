"""主配微统一优化模型：24h连续状态＋离散抢修/开关/启动。

三个研究点共用这一个物理模型；台风是外生场景输入。
网络采用虚拟源单商品流＋边数约束的通用带电森林，不枚举32个接线。
当前已报告抢修任务的前瞻日程跨预测场景共用；未知工时取重伤上界，
实际巡检揭示后再滚动。因此它是因果滚动原型，不是完整POMDP最优策略。
"""
from __future__ import annotations
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


from collections import defaultdict
from repair_model import job_options,baseline_schedule,travel_hours
from network import components
from capability import interface_envelopes
from information import history_key,option,validate_policy
from reachability import island_reserve_tube


def solve_lp(m,objective=None,equalities=None):
    """连续LP；保留所有时段和场景约束，用于回退、能力投影与轨迹分解。"""
    if np.any(m.integrality):raise ValueError('先固定离散变量')
    eq=np.isfinite(m.row_lb)&np.isfinite(m.row_ub)&(np.abs(m.row_lb-m.row_ub)<1e-10)
    lo=np.isfinite(m.row_lb)&~eq;hi=np.isfinite(m.row_ub)&~eq
    ae=m.A[eq];be=m.row_lb[eq]
    if equalities:
        rr=[];cc=[];vv=[];rhs=[]
        for i,(row,v) in enumerate(equalities):
            rhs.append(v)
            for j,a in row.items():rr.append(i);cc.append(j);vv.append(a)
        extra=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(m.c))).tocsr()
        ae=sparse.vstack([ae,extra],format='csr');be=np.r_[be,rhs]
    start=time.perf_counter();res=linprog(m.c if objective is None else objective,
        A_ub=sparse.vstack([m.A[hi],-m.A[lo]],format='csr'),b_ub=np.r_[m.row_ub[hi],-m.row_lb[lo]],
        A_eq=ae,b_eq=be,bounds=list(zip(m.lb,m.ub)),method='highs')
    return {'x':res.x,'status':int(res.status),'success':bool(res.success),'message':str(res.message),
            'objective':float(res.fun) if res.fun is not None else np.inf,'seconds':time.perf_counter()-start}


def freeze(m,x):
    lb=m.lb.copy();ub=m.ub.copy();mask=m.integrality!=0
    if np.max(np.abs(x[mask]-np.rint(x[mask])),initial=0)>2e-5:raise ValueError('整数解尚不可用')
    lb[mask]=np.rint(x[mask]);ub[mask]=lb[mask]
    return LinearModel(m.c.copy(),np.zeros_like(m.integrality),lb,ub,m.A,m.row_lb,m.row_ub,m.var_indices,m.metadata,m.objective_constant)


def residual(m,x):
    if x is None:return np.inf
    a=m.A@x;mask=m.integrality!=0
    return float(max(np.max(m.row_lb-a,initial=0),np.max(a-m.row_ub,initial=0),
        np.max(m.lb-x,initial=0),np.max(x-m.ub,initial=0),np.max(np.abs(x[mask]-np.rint(x[mask])),initial=0)))


def _add(row,col,value):
    row[int(col)]=row.get(int(col),0)+float(value)


def build_model(c,state,fc,branch='free',policy='joint'):
    """R优先随机前瞻模型；branch=start/wait仅约束当前动作，未来可启动。"""
    dt=c['dt'];now=state['step']*dt;hours=fc['hours'];n=len(hours);ns=len(fc['probabilities']);nb=len(c['nodes']);ne=len(c['edges']);nm=len(c['microgrids'])
    gen=c['generator'];probs=fc['probabilities'];b=ModelBuilder();v={}
    # 01 默认使用外部生成的有限因果作业策略树；旧的自由排序工具仍保留。
    # 信息到达后允许树分支，已经开始的实体作业由独立active状态锁定。
    external=fc.get('crew_plan')
    jobs,ctx=([],{'origin':state['crew']['position'],'ready_hour':now,'fixed_completion':{},'inspecting':None}) if external else job_options(c,state,fc)
    job_idx=[]
    for j,job in enumerate(jobs):
        idx=b.add_var('job_'+str(j),len(job['options']),ub=1,integer=True);job_idx.append(idx)
        b.add_row({k:1 for k in idx},ub=1,name='job_at_most_once')
    order={};big=4*c['horizon_hours']+20
    for i in range(len(jobs)):
        if not len(job_idx[i]):continue
        for j in range(i+1,len(jobs)):
            if not len(job_idx[j]):continue
            q=b.add_var(f'order_{i}_{j}',1,ub=1,integer=True)[0];order[i,j]=q
            dist=travel_hours(c,jobs[i]['edge'],jobs[j]['edge'])
            row={q:-big}
            for k,(_,end) in zip(job_idx[i],jobs[i]['options']):_add(row,k,-end-big)
            for k,(start,_) in zip(job_idx[j],jobs[j]['options']):_add(row,k,start-big)
            b.add_row(row,lb=dist-3*big,name='crew_i_before_j')
            row={q:big}
            for k,(_,end) in zip(job_idx[j],jobs[j]['options']):_add(row,k,-end-big)
            for k,(start,_) in zip(job_idx[i],jobs[i]['options']):_add(row,k,start-big)
            b.add_row(row,lb=dist-2*big,name='crew_j_before_i')
    if policy!='joint':
        chosen=baseline_schedule(c,state,fc,jobs,ctx,policy)
        for j,idx in enumerate(job_idx):
            for k,col in enumerate(idx):b.lb[col]=b.ub[col]=float(chosen.get(j)==k)
    end_crew=b.add_var('crew_finish',1,lb=min(c['horizon_hours'],max(now,ctx['ready_hour'])),ub=c['horizon_hours'],cost=0. if external else c['crew']['hourly_cost'])[0]
    for job,idx in zip(jobs,job_idx):
        row={end_crew:1}
        for col,(_,end) in zip(idx,job['options']):_add(row,col,-end)
        b.add_row(row,lb=0,name='crew_finish_after_jobs')
    # 02 拓扑决定网格和机组启动网格。功率与SOC始终保持15分钟。
    slots=np.unique(np.r_[0,np.flatnonzero(np.isclose(np.mod(hours,c['switch_minutes']/60),0,atol=1e-7))]).astype(int)
    if external:
        event_slots=[int(round((h-now)/dt)) for h in external['event_hours'] if now<=h<c['horizon_hours']]
        slots=np.unique(np.r_[slots,event_slots]).astype(int)
    block=np.searchsorted(slots,np.arange(n),side='right')-1;nk=len(slots)
    modes=np.unique(np.r_[0,np.flatnonzero(np.isclose(np.mod(hours,c['battery_mode_minutes']/60),0,atol=1e-7))]).astype(int)
    mmap=np.searchsorted(modes,np.arange(n),side='right')-1
    starts=[];start_interval=float(option(c,'generator.start_interval_hours',1.))
    if state['unit']['start_hour'] is None and state['unit']['attempts']<int(gen['max_attempts']):
        starts=[t for t,h in enumerate(hours) if h>=state['unit']['earliest_start_hour']-1e-8 and h<=gen['last_start_hour']+1e-8
                and (t==0 or np.isclose(h/start_interval,round(h/start_interval)))]
    starts=np.asarray(starts,int);ng=len(starts);roots=[0]+list(c['mg_nodes']);nr=len(roots)
    auxcoef=np.zeros((n,ng));oncoef=np.zeros((n,ng));capcoef=np.zeros((n,ng))
    fixed_aux=np.zeros(n);fixed_on=np.zeros(n);fixed_cap=np.zeros(n);auxprofile=gen['startup_profile_mw'];duration=len(auxprofile)*dt
    def generator_coeff(start):
        elapsed=hours-start;aux=np.zeros(n);on=(elapsed>=duration-1e-8).astype(float)
        for t,h in enumerate(elapsed):
            if -1e-8<=h<duration-1e-8:aux[t]=auxprofile[min(len(auxprofile)-1,int((h+1e-8)/dt))]
        cap=np.where(on>0,np.minimum(gen['max_mw'],gen['min_mw']+gen['ramp_mw_per_hour']*np.maximum(0,elapsed-duration)),0)
        return aux,on,cap
    for k,t in enumerate(starts):auxcoef[:,k],oncoef[:,k],capcoef[:,k]=generator_coeff(hours[t])
    if state['unit']['start_hour'] is not None:fixed_aux,fixed_on,fixed_cap=generator_coeff(state['unit']['start_hour'])
    v['target_start']=b.add_var('target_start',(ns,ng),ub=1,integer=True,cost=probs[:,None]*gen['start_cost'])
    # 本模型中T0仅有外部源、待启动G和TX；无外部源且禁止反送时，
    # 辅机阶段没有任何正有功来源。由节点平衡可严格排除全部启动时刻。
    # 这是物理必要条件预筛，不把超时或启发式失败当不可行。
    main_side=next(group for group in components(c,[e!=c['tx'] for e in range(ne)])[0] if 0 in group)
    if c['alpha']<=0 and not c.get('allow_upward',1) and np.any(auxprofile>0) and not (set(main_side)&set(c['mg_nodes'])):
        for col in v['target_start'].ravel():b.ub[col]=0.
    v['closed']=b.add_var('closed',(ns,ne,nk),ub=1,integer=True)
    v['energized_bus']=b.add_var('energized_bus',(ns,nb,nk),ub=1,integer=True)
    v['root']=b.add_var('root',(ns,nr,nk),ub=1,integer=True)
    v['energized_edge']=b.add_var('energized_edge',(ns,ne,nk),ub=1)
    v['fictitious_flow']=b.add_var('fictitious_flow',(ns,ne,nk),lb=-nb,ub=nb)
    v['root_supply']=b.add_var('root_supply',(ns,nr,nk),ub=nb)
    v['switch_change']=b.add_var('switch_change',(ns,ne,nk),ub=1,cost=probs[:,None,None]*c['switch_cost'])
    en=np.array([m['energy_mwh'] for m in c['microgrids']]);ch=np.array([m['charge_mw'] for m in c['microgrids']]);dis=np.array([m['discharge_mw'] for m in c['microgrids']])
    inv=np.array([m['inverter_mva'] for m in c['microgrids']]);dg=np.array([m['dg_max_mw'] for m in c['microgrids']])
    batcost=np.array([m['throughput_cost'] for m in c['microgrids']]);dgcost=np.array([m['dg_cost'] for m in c['microgrids']])
    v['charge']=b.add_var('charge',(ns,nm,n),ub=ch[None,:,None],cost=probs[:,None,None]*dt*batcost[None,:,None])
    v['discharge']=b.add_var('discharge',(ns,nm,n),ub=dis[None,:,None],cost=probs[:,None,None]*dt*batcost[None,:,None])
    # 显式MGO→DSO接口。跨期包络先进入模型，内部变量保留为分解见证。
    v['pcc_p']=b.add_var('pcc_p',(ns,nm,n),lb=-np.inf)
    v['pcc_q']=b.add_var('pcc_q',(ns,nm,n),lb=-np.inf)
    v['battery_mode']=b.add_var('battery_mode',(ns,nm,len(modes)),ub=1,integer=True)
    v['energy']=b.add_var('energy',(ns,nm,n+1),lb=(en*np.array([m['soc_min'] for m in c['microgrids']]))[None,:,None],ub=(en*np.array([m['soc_max'] for m in c['microgrids']]))[None,:,None])
    v['pv']=b.add_var('pv',(ns,nm,n),ub=fc['pv'])
    v['dg']=b.add_var('dg',(ns,nm,n),ub=dg[None,:,None],cost=probs[:,None,None]*dt*dgcost[None,:,None])
    v['mg_q']=b.add_var('mg_q',(ns,nm,n),lb=-inv[None,:,None],ub=inv[None,:,None])
    dc=c['profiles']['critical'][:,state['step']:];dd=c['profiles']['general'][:,state['step']:]
    v['served_c']=b.add_var('served_c',(ns,nb,n),ub=dc[None,:,:])
    v['served_g']=b.add_var('served_g',(ns,nb,n),ub=dd[None,:,:])
    v['target_p']=b.add_var('target_p',(ns,n),ub=gen['max_mw'],cost=probs[:,None]*dt*gen['fuel_cost'])
    v['target_q']=b.add_var('target_q',(ns,n),lb=-gen['q_max_mvar'],ub=gen['q_max_mvar'])
    v['grid_p']=b.add_var('grid_p',(ns,n),ub=c['alpha']*c['grid_reference_mw'],cost=probs[:,None]*dt*c['profiles']['price'][None,state['step']:])
    v['grid_q']=b.add_var('grid_q',(ns,n),lb=-c['alpha']*c['grid_reference_mw']*.5,ub=c['alpha']*c['grid_reference_mw']*.5)
    caps=np.array([e['s_mva'] for e in c['edges']]);vmin=c['voltage_min']**2;vmax=c['voltage_max']**2
    v['flow_p']=b.add_var('flow_p',(ns,ne,n),lb=-caps[None,:,None],ub=caps[None,:,None])
    v['flow_q']=b.add_var('flow_q',(ns,ne,n),lb=-caps[None,:,None],ub=caps[None,:,None])
    v['voltage']=b.add_var('voltage',(ns,nb,n),ub=vmax)
    # 当前线路损坏后的可用性取决于优化的维修完成，而不是事先给定恢复时刻。
    byedge={job['edge']:j for j,job in enumerate(jobs)}
    def available_expression(e,h):
        if external:return float(external['availability'][scenario,e,int(round((h-now)/dt))]),{}
        if fc['failure_hour'][scenario,e]<=h+1e-8:return 0.,{}
        if not state['damaged'][e]:return 1.,{}
        if e in ctx['fixed_completion']:return float(ctx['fixed_completion'][e]<=h+1e-8),{}
        if e not in byedge:return 0.,{}
        j=byedge[e];return 0.,{col:1. for col,(_,end) in zip(job_idx[j],jobs[j]['options']) if end<=h+1e-8}
    # 03 通用带电森林：每个带电分量一个可选虚拟根，允许无源区域失电。
    for scenario in range(ns):
        for k,t in enumerate(slots):
            h=hours[t];eb=v['energized_bus'][scenario,:,k];root=v['root'][scenario,:,k]
            forest={col:1 for col in v['energized_edge'][scenario,:,k]}
            for col in eb:_add(forest,col,-1)
            for col in root:_add(forest,col,1)
            b.add_row(forest,0,0,'forest_edge_count')
            for r,node in enumerate(roots):
                b.add_row({root[r]:1,eb[node]:-1},ub=0,name='root_is_energized')
                b.add_row({v['root_supply'][scenario,r,k]:1,root[r]:-nb},ub=0,name='virtual_root_capacity')
                if node==0 and c['alpha']<=0:
                    row={root[r]:1}
                    for q in range(ng):_add(row,v['target_start'][scenario,q],-oncoef[t,q])
                    b.add_row(row,ub=fixed_on[t],name='root_requires_available_source')
            balances=[{eb[i]:-1} for i in range(nb)]
            for r,node in enumerate(roots):_add(balances[node],v['root_supply'][scenario,r,k],1)
            for e,line in enumerate(c['edges']):
                z=v['closed'][scenario,e,k];y=v['energized_edge'][scenario,e,k];f=v['fictitious_flow'][scenario,e,k]
                fr,to=line['from'],line['to'];constant,expr=available_expression(e,h);row={z:1}
                for col,a in expr.items():_add(row,col,-a)
                b.add_row(row,ub=constant,name='close_only_if_available')
                if not line['switchable']:
                    fixedrow={z:1}
                    if line['initial_closed']:
                        for col,a in expr.items():_add(fixedrow,col,-a)
                        b.add_row(fixedrow,constant,constant,'fixed_healthy_switch')
                    else:b.add_row(fixedrow,0,0,'fixed_normally_open')
                b.add_row({eb[fr]:1,eb[to]:-1,z:1},ub=1,name='closed_equal_energization')
                b.add_row({eb[to]:1,eb[fr]:-1,z:1},ub=1,name='closed_equal_energization')
                b.add_row({y:1,z:-1},ub=0,name='active_edge_AND')
                b.add_row({y:1,eb[fr]:-1},ub=0,name='active_edge_AND')
                b.add_row({y:1,z:-1,eb[fr]:-1},lb=-1,name='active_edge_AND')
                b.add_row({f:1,y:-nb},ub=0,name='fictitious_link')
                b.add_row({f:-1,y:-nb},ub=0,name='fictitious_link')
                _add(balances[fr],f,-1);_add(balances[to],f,1)
                if not line['switchable']:
                    # 固定接线的修复包含现场恢复连接，不作为DSO可选开关操作。
                    # 常开固定支路始终不接入；不能借此模拟手动操作开关。
                    b.ub[v['switch_change'][scenario,e,k]]=0.
                    continue
                # 保护跳闸不计入主动操作；故障及修复事件已加入决定网格。
                forced=bool(np.isclose(fc['failure_hour'][scenario,e],h))
                delta=v['switch_change'][scenario,e,k]
                prev=v['closed'][scenario,e,k-1] if k else None;old=float(state['closed'][e])
                row={z:1,delta:-1}
                if k:_add(row,prev,-1);b.add_row(row,ub=0,name='switch_close_count')
                else:b.add_row(row,ub=old,name='switch_close_count')
                row={z:-1,delta:-1}
                if k:_add(row,prev,1);b.add_row(row,ub=float(forced),name='switch_open_count')
                else:b.add_row(row,ub=float(forced)-old,name='switch_open_count')
                if h-state['last_change'][e]<c['switch_min_hold_hours']-1e-8:
                    held=0. if fc['failure_hour'][scenario,e]<=h+1e-8 else old
                    b.add_row({z:1},held,held,'executed_switch_hold')
                hold=[j for j,u in enumerate(slots[:k+1]) if h-hours[u]<c['switch_min_hold_hours']-1e-8]
                b.add_row({v['switch_change'][scenario,e,j]:1 for j in hold},ub=1,name='switch_minimum_hold')
            for row in balances:b.add_row(row,0,0,'virtual_network_connectivity')
        for e,line in enumerate(c['edges']):
            b.add_row({col:1 for col in v['switch_change'][scenario,e]},ub=max(0,c['switch_max_per_line']-state['switch_count'][e]),name='line_operation_budget')
        b.add_row({col:1 for col in v['switch_change'][scenario].ravel()},ub=max(0,c['switch_max_total']-sum(state['switch_count'])),name='total_operation_budget')
    # 04 SOC、负荷、源出力与每15分钟的P/Q、电压。目标是供电量，不是仅末时刻恢复率。
    facets=int(c['polygon_sides']);angles=(np.arange(facets)+.5)*2*np.pi/facets;inner=np.cos(np.pi/facets)
    aux_tan=np.tan(np.arccos(gen['aux_power_factor']));ratio=c['load']['reactive_ratio']
    for s in range(ns):
        b.add_row({col:1 for col in v['target_start'][s]},ub=1,name='one_start_in_lookahead')
        if branch in ('start','wait'):
            at=np.flatnonzero(starts==0)
            if len(at):b.add_row({v['target_start'][s,at[0]]:1},float(branch=='start'),float(branch=='start'),'current_start_branch')
            elif branch=='start':b.add_row({},1,1,'current_start_not_allowed')
        for m,mg in enumerate(c['microgrids']):
            b.add_row({v['energy'][s,m,0]:1},state['energy'][m],state['energy'][m],'inherited_soc')
            b.add_row({v['energy'][s,m,n]:1},lb=en[m]*mg['soc_terminal'],name='terminal_reserve')
            for t in range(n):
                k=block[t];E=v['energized_bus'][s,c['mg_nodes'][m],k]
                chv=v['charge'][s,m,t];disv=v['discharge'][s,m,t];mode=v['battery_mode'][s,m,mmap[t]]
                b.add_row({v['energy'][s,m,t+1]:1,v['energy'][s,m,t]:-1,chv:-dt*mg['eta_charge'],disv:dt/mg['eta_discharge']},0,0,'soc_balance')
                b.add_row({chv:1,mode:-ch[m]},ub=0,name='charge_mode')
                b.add_row({disv:1,mode:dis[m]},ub=dis[m],name='discharge_mode')
                for key,cap in (('charge',ch[m]),('discharge',dis[m]),('dg',dg[m]),('pv',fc['pv'][s,m,t])):
                    b.add_row({v[key][s,m,t]:1,E:-cap},ub=0,name='source_requires_energization')
                for sign in (-1,1):b.add_row({v['mg_q'][s,m,t]:sign,E:-inv[m]},ub=0,name='source_requires_energization')
                for a in angles:
                    co,si=float(np.cos(a)),float(np.sin(a))
                    b.add_row({v['pv'][s,m,t]:co,disv:co,chv:-co,v['mg_q'][s,m,t]:si,E:-inv[m]*inner},ub=0,name='inverter_rating')
                node=c['mg_nodes'][m];sc=v['served_c'][s,node,t];sg=v['served_g'][s,node,t]
                b.add_row({v['pcc_p'][s,m,t]:1,v['pv'][s,m,t]:-1,v['dg'][s,m,t]:-1,disv:-1,chv:1,sc:1,sg:1},0,0,'MGO_interface_P')
                b.add_row({v['pcc_q'][s,m,t]:1,v['mg_q'][s,m,t]:-1,sc:ratio[node],sg:ratio[node]},0,0,'MGO_interface_Q')
        for t,h in enumerate(hours):
            k=block[t];eb=v['energized_bus'][s,:,k];pbal=[{} for _ in range(nb)];qbal=[{} for _ in range(nb)]
            # 外部电源注入及机组启动/出力必须处在已送电区域。
            for key,cap in (('grid_p',c['alpha']*c['grid_reference_mw']),('target_p',gen['max_mw'])):
                b.add_row({v[key][s,t]:1,eb[0]:-cap},ub=0,name='upstream_source_energized')
            row={v['target_p'][s,t]:1}
            for q in range(ng):_add(row,v['target_start'][s,q],-capcoef[t,q])
            b.add_row(row,ub=fixed_cap[t],name='generation_after_start')
            row={v['target_p'][s,t]:1}
            for q in range(ng):_add(row,v['target_start'][s,q],-gen['min_mw']*oncoef[t,q])
            b.add_row(row,lb=gen['min_mw']*fixed_on[t],name='online_minimum_power')
            for sign in (-1,1):
                row={v['target_q'][s,t]:sign}
                for q in range(ng):_add(row,v['target_start'][s,q],-gen['q_max_mvar']*oncoef[t,q])
                b.add_row(row,ub=gen['q_max_mvar']*fixed_on[t],name='online_q')
            for a in angles:
                row={v['target_p'][s,t]:float(np.cos(a)),v['target_q'][s,t]:float(np.sin(a))}
                for q in range(ng):_add(row,v['target_start'][s,q],-gen['s_mva']*inner*oncoef[t,q])
                b.add_row(row,ub=gen['s_mva']*inner*fixed_on[t],name='generator_rating')
            up=gen['ramp_mw_per_hour']*dt
            fault_event=any(np.isclose(fc['failure_hour'][s],h)) or (t==0 and state.get('new_fault',False))
            down=gen['emergency_ramp_down_mw'] if fault_event else up
            row={v['target_p'][s,t]:1}
            if t:_add(row,v['target_p'][s,t-1],-1);b.add_row(row,-down,up,'generator_ramp')
            else:b.add_row(row,state['previous_pg']-down,state['previous_pg']+up,'generator_ramp')
            for e,line in enumerate(c['edges']):
                fr,to=line['from'],line['to'];fp=v['flow_p'][s,e,t];fq=v['flow_q'][s,e,t];y=v['energized_edge'][s,e,k]
                _add(pbal[fr],fp,-1);_add(pbal[to],fp,1);_add(qbal[fr],fq,-1);_add(qbal[to],fq,1)
                for a in angles:b.add_row({fp:float(np.cos(a)),fq:float(np.sin(a)),y:-caps[e]*inner},ub=0,name='line_rating')
                rr=line['r_pu']/c['base_mva'];xx=line['x_pu']/c['base_mva'];M=2*vmax+2*(rr+xx)*caps[e]+1e-5
                drop={v['voltage'][s,to,t]:1,v['voltage'][s,fr,t]:-1,fp:2*rr,fq:2*xx};row=dict(drop);_add(row,y,M)
                b.add_row(row,ub=M,name='voltage_drop_upper');row=dict(drop);_add(row,y,-M);b.add_row(row,lb=-M,name='voltage_drop_lower')
            _add(pbal[0],v['grid_p'][s,t],1);_add(pbal[0],v['target_p'][s,t],1)
            _add(qbal[0],v['grid_q'][s,t],1);_add(qbal[0],v['target_q'][s,t],1)
            for q in range(ng):
                _add(pbal[0],v['target_start'][s,q],-auxcoef[t,q]);_add(qbal[0],v['target_start'][s,q],-aux_tan*auxcoef[t,q])
            for m,node in enumerate(c['mg_nodes']):
                _add(pbal[node],v['pcc_p'][s,m,t],1);_add(qbal[node],v['pcc_q'][s,m,t],1)
            for i in range(nb):
                for key,demand in (('served_c',dc),('served_g',dd)):
                    if i not in c['mg_nodes']:
                        _add(pbal[i],v[key][s,i,t],-1);_add(qbal[i],v[key][s,i,t],-ratio[i])
                    b.add_row({v[key][s,i,t]:1,eb[i]:-demand[i,t]},ub=0,name='load_energized')
                rhs=fixed_aux[t] if i==0 else 0.;b.add_row(pbal[i],rhs,rhs,'nodal_p')
                rhs=fixed_aux[t]*aux_tan if i==0 else 0.;b.add_row(qbal[i],rhs,rhs,'nodal_q')
                b.add_row({v['voltage'][s,i,t]:1,eb[i]:-vmax},ub=0,name='voltage_when_live')
                b.add_row({v['voltage'][s,i,t]:1,eb[i]:-vmin},lb=0,name='voltage_when_live')
                if i in c['protected_nodes']:b.add_row({v['served_c'][s,i,t]:1},dc[i,t],dc[i,t],'protected_local_critical')
            if not c.get('allow_upward',1):b.add_row({v['flow_p'][s,c['tx'],t]:1},lb=0,name='no_upward_coordination_baseline')
            for r,i in enumerate(roots):
                root=v['root'][s,r,k];vv=v['voltage'][s,i,t]
                b.add_row({vv:1,root:vmax},ub=1+vmax,name='root_voltage')
                b.add_row({vv:1,root:-vmax},lb=1-vmax,name='root_voltage')
    reserve=island_reserve_tube(c,state,fc) if option(c,'reachability.enforce_island_reserve',True) else None
    if reserve is not None:
        for scenario in range(ns):
            for j in range(nm):
                for t in range(n+1):
                    lo,hi=float(reserve['lower'][j,t]),float(reserve['upper'][j,t])
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo>hi+1e-8:b.add_row({},1,1,'empty_backward_reserve')
                    else:b.add_row({v['energy'][scenario,j,t]:1},lo,hi,'backward_island_continuation')
    envelopes=interface_envelopes(c,state,fc)
    for envelope in envelopes:
        s,j,stop=[envelope[k] for k in ('scenario','microgrid','stop')]
        b.add_row({v['pcc_p'][s,j,t]:dt for t in range(stop)},envelope['lower_mwh'],envelope['upper_mwh'],'MGO_multiperiod_envelope')
    # 05 条件非预期性：只按已发生的预测故障报告区分未来场景。
    # 未报告损伤的真实工时没有进入模型。当前抢修日程全场景共享。
    def bind(ids,h,name):
        groups=defaultdict(list)
        for s in range(ns):groups[history_key(fc,s,int(round((h-now)/dt)))].append(s)
        for group in groups.values():
            for s in group[1:]:
                for a,z in zip(np.asarray(ids[group[0]]).ravel(),np.asarray(ids[s]).ravel()):b.add_row({a:1,z:-1},0,0,'NA_'+name)
    for t,h in enumerate(hours):
        for key in ('charge','discharge','pv','dg','mg_q','served_c','served_g'):bind(v[key][:,:,t],h,key)
        for key in ('target_p','target_q','grid_p','grid_q'):bind(v[key][:,t],h,key)
    for k,t in enumerate(slots):
        for key in ('closed','energized_bus','root'):bind(v[key][:,:,k],hours[t],key)
    for k,t in enumerate(modes):bind(v['battery_mode'][:,:,k],hours[t],'battery_mode')
    for k,t in enumerate(starts):bind(v['target_start'][:,k],hours[t],'target_start')
    m=b.build();cost=m.c.copy();reward=np.zeros_like(m.c)
    reward[v['served_c']]=-probs[:,None,None]*dt*c['critical_weight'];reward[v['served_g']]=-probs[:,None,None]*dt*c['general_weight'];m.c=reward
    m.metadata.update({'cfg':c,'state':state,'forecast':fc,'hours':hours,'slots':slots,'block':block,'mode_slots':modes,'mode_map':mmap,
        'start_slots':starts,'auxcoef':auxcoef,'oncoef':oncoef,'capcoef':capcoef,'fixed_aux':fixed_aux,'fixed_on':fixed_on,'fixed_cap':fixed_cap,
        'roots':roots,'jobs':jobs,'job_indices':job_idx,'crew_context':ctx,'order_indices':order,'cost_objective':cost,
        'cost_constant':state['cost_accum']+float(probs@np.asarray(external['cost_by_scenario'])) if external else state['cost_accum']-now*c['crew']['hourly_cost'],'policy':policy,'branch':branch,
        'past_reward':state['reward_accum'],'denominator':c['denominator'],'interface_envelopes':envelopes,'reserve_tube':reserve,'crew_policy':None if external is None else external['name']})
    return m


def safe_seed(m,discharge=False,repair_rule=None,allow_start=True,battery_schedule=None):
    """构造并求解受限日程/径向接线的可行LP，作为MILP热启动。

    repair_rule仅生成初始候选，之后混合整数模型仍可优化作业/接线。
    每个候选都经过整时域校核；不把启发式候选当作最优解。
    """
    md=m.metadata;c=md['cfg'];state=md['state'];fc=md['forecast'];v=m.var_indices;ns=len(fc['probabilities'])
    external=fc.get('crew_plan')
    x=np.zeros(len(m.c));mask=m.integrality!=0
    for col in np.flatnonzero(mask):x[col]=m.lb[col] if m.lb[col]==m.ub[col] else 0
    if repair_rule and md['policy']=='joint':
        chosen=baseline_schedule(c,state,fc,md['jobs'],md['crew_context'],repair_rule)
        for j,idx in enumerate(md['job_indices']):
            for k,col in enumerate(idx):x[col]=float(chosen.get(j)==k)
    for (i,j),col in md['order_indices'].items():
        ai=[start for q,(start,end) in zip(md['job_indices'][i],md['jobs'][i]['options']) if x[q]>.5]
        aj=[start for q,(start,end) in zip(md['job_indices'][j],md['jobs'][j]['options']) if x[q]>.5]
        x[col]=float(bool(ai and aj and ai[0]<aj[0]))
    if md['branch']=='start':
        hit=np.flatnonzero(md['start_slots']==0)
        if not len(hit):return None
        x[v['target_start'][:,hit[0]]]=1
    elif allow_start and len(md['start_slots']) and np.any(m.ub[v['target_start']]>0):
        choices=np.flatnonzero(md['start_slots']>0) if md['branch']=='wait' else np.arange(len(md['start_slots']))
        if len(choices):x[v['target_start'][:,choices[0]]]=1
    x[v['battery_mode']]=0 if discharge else 1
    if battery_schedule is not None:
        x[v['battery_mode']]=np.broadcast_to(battery_schedule,v['battery_mode'].shape)
    completions=dict(md['crew_context']['fixed_completion'])
    for job,idx in zip(md['jobs'],md['job_indices']):
        for col,(start,end) in zip(idx,job['options']):
            if x[col]>.5:completions[job['edge']]=end
    for scenario in range(ns):
        online=md['fixed_on']+x[v['target_start']][scenario]@md['oncoef'].T
        closed=np.asarray(state['closed'],float).copy();last=np.asarray(state['last_change']).copy();counts=np.asarray(state['switch_count']).copy()
        for k,t in enumerate(md['slots']):
            h=md['hours'][t];available=np.ones(len(c['edges']),bool)
            for e in range(len(c['edges'])):
                if state['damaged'][e]:available[e]=completions.get(e,np.inf)<=h+1e-8
                if fc['failure_hour'][scenario,e]<=h+1e-8:available[e]=False
                if external is not None:available[e]=external['availability'][scenario,e,t]
                if not available[e]:closed[e]=0
                if not c['edges'][e]['switchable']:
                    closed[e]=float(c['edges'][e]['initial_closed'] and available[e])
            if repair_rule:
                # 合上健康可用支路前检查物理图，保持径向且遵守剩余操作预算。
                # 起点为上轮已执行森林，先恢复馈线，再尝试健康联络支路。
                candidates=sorted(range(len(c['edges'])),key=lambda e:c['edges'][e]['kind']=='tie')
                for e in candidates:
                    if closed[e] or not available[e] or not c['edges'][e]['switchable']:continue
                    if h-last[e]<c['switch_min_hold_hours']-1e-8 or counts[e]>=c['switch_max_per_line'] or counts.sum()>=c['switch_max_total']:continue
                    trial=closed.copy();trial[e]=1
                    _,cycle=components(c,trial)
                    if cycle:continue
                    closed=trial;last[e]=h;counts[e]+=1
            x[v['closed'][scenario,:,k]]=closed
            groups,_=components(c,closed)
            for group in groups:
                eligible=set(group)&set(c['mg_nodes'])
                if 0 in group and (c['alpha']>0 or online[t]>.5):eligible.add(0)
                if eligible:
                    root=min(eligible);x[v['energized_bus'][scenario,group,k]]=1
                    rr=md['roots'].index(root);x[v['root'][scenario,rr,k]]=1
    lb=m.lb.copy();ub=m.ub.copy()
    if np.any(x[mask]<lb[mask]-1e-8) or np.any(x[mask]>ub[mask]+1e-8):return None
    lb[mask]=ub[mask]=np.rint(x[mask])
    trial=LinearModel(m.c,np.zeros_like(m.integrality),lb,ub,m.A,m.row_lb,m.row_ub,m.var_indices,md)
    r=solve_lp(trial)
    if r['x'] is not None and residual(m,r['x'])<=2e-5:
        r['seed_rule']=(repair_rule or 'hold')+(':'+('mixed' if battery_schedule is not None else 'discharge' if discharge else 'charge'))
        return r
    return None


def solve_mip(m,opts,seed=None):
    """优先使用HiGHS接口热启动；没有highspy时保留SciPy兼容路径。"""
    from types import SimpleNamespace
    try:import highspy
    except ImportError:
        return milp(m.c,integrality=m.integrality,bounds=Bounds(m.lb,m.ub),constraints=LinearConstraint(m.A,m.row_lb,m.row_ub),options=opts)
    h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',1)
    h.setOptionValue('time_limit',float(opts['time_limit']));h.setOptionValue('mip_rel_gap',float(opts['mip_rel_gap']))
    lp=highspy.HighsLp();lp.num_col_=len(m.c);lp.num_row_=len(m.row_lb)
    lp.col_cost_=m.c;lp.col_lower_=m.lb;lp.col_upper_=m.ub;lp.row_lower_=m.row_lb;lp.row_upper_=m.row_ub
    A=m.A.tocsc();lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=A.indptr
    lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data
    lp.integrality_=[highspy.HighsVarType.kInteger if a else highspy.HighsVarType.kContinuous for a in m.integrality]
    h.passModel(lp)
    if seed is not None:
        sol=highspy.HighsSolution();sol.col_value=np.asarray(seed,dtype=float).tolist();sol.value_valid=True;h.setSolution(sol)
    h.run();status=h.getModelStatus();sol=h.getSolution();info=h.getInfo()
    code=0 if status==highspy.HighsModelStatus.kOptimal else 2 if status==highspy.HighsModelStatus.kInfeasible else 1
    xx=np.asarray(sol.col_value) if sol.value_valid else None
    return SimpleNamespace(x=xx,status=code,message=h.modelStatusToString(status),mip_dual_bound=info.mip_dual_bound)


def optimize(m,seconds=None,secondary_seconds=None,warm_x=None):
    """先最大化恢复量，再在已找到恢复量容差内降低成本。

    超时保留可行计划与真实界。未证明R最优时，第二阶段只承诺保持
    已找到恢复水平，不能声称全局词典序最优。R上下界用于D证据。
    """
    md=m.metadata;c=md['cfg'];cfg=c['solver'];start=time.perf_counter()
    seeds=[];seed_log=[]
    variants=[('critical_first',True)] if md['forecast'].get('crew_plan') else [('critical_first',True),('nearest',True),(None,True)]
    if len(md['start_slots']) and md['branch']!='start':variants.append(('critical_first',False))
    for rule,allow_start in variants:
        # 两种模式都评估。旧版“充电候选可行即跳过放电”会锁死SOC价值。
        for discharge in (True,False):
            seed=safe_seed(m,discharge,rule,allow_start)
            seed_log.append({'repair_rule':rule or 'hold','allow_start':allow_start,'battery':'discharge' if discharge else 'charge',
                             'feasible':seed is not None,'R':None if seed is None else float((md['past_reward']-seed['objective'])/md['denominator'])})
            if seed is not None:seeds.append(seed)
    if not seeds and md['forecast'].get('crew_plan'):
        for discharge in (True,False):
            fallback=safe_seed(m,discharge,repair_rule=None,allow_start=False)
            if fallback is not None:seeds.append(fallback)
    # 再评估白天允许充电、其他时段放电的混合候选；MILP仍可改全部模式。
    hours=md['hours'][md['mode_slots']]
    mixed=((hours>=9)&(hours<15)).astype(float)[None,None,:]
    trial=safe_seed(m,repair_rule='critical_first',battery_schedule=mixed)
    seed_log.append({'repair_rule':'critical_first','allow_start':True,'battery':'day_charge_other_discharge',
                     'feasible':trial is not None,'R':None if trial is None else float((md['past_reward']-trial['objective'])/md['denominator'])})
    if trial is not None:seeds.append(trial)
    if warm_x is not None and residual(m,warm_x)<=2e-5:
        seeds.append({'x':np.asarray(warm_x).copy(),'objective':float(m.c@warm_x),'seed_rule':'previous_verified'})
    seed=min(seeds,key=lambda a:a['objective']) if seeds else None
    opts={'time_limit':float(seconds if seconds is not None else cfg['primary_seconds']),
          'mip_rel_gap':cfg['rel_gap'],'presolve':True,'disp':False}
    # 保留当前路线/拓扑/启机，仅释放充放电模式作小邻域搜索；不能用其界
    # 充当原MILP的最优界。随后仍求完整MILP，且始终保留已校核最好候选。
    if seed is not None and opts['time_limit']>=1:
        mask=m.integrality!=0;mask[m.var_indices['battery_mode'].ravel()]=False
        lb=m.lb.copy();ub=m.ub.copy();lb[mask]=ub[mask]=np.rint(seed['x'][mask])
        neighborhood=LinearModel(m.c,m.integrality,lb,ub,m.A,m.row_lb,m.row_ub,m.var_indices,md)
        local=solve_mip(neighborhood,{**opts,'time_limit':min(2.,opts['time_limit']*.25)},seed['x'])
        if local.x is not None and residual(m,local.x)<=2e-5 and m.c@local.x<seed['objective']-1e-8:
            seed={'x':local.x,'objective':float(m.c@local.x),'seed_rule':'battery_neighborhood'}
            seed_log.append({'repair_rule':'best_verified','battery':'optimized_modes','feasible':True,
                             'R':float((md['past_reward']-seed['objective'])/md['denominator'])})
    out=solve_mip(m,opts,seed['x'] if seed is not None else None)
    candidates=[]
    if out.x is not None and residual(m,out.x)<=2e-5:candidates.append((out.x,'MILP'))
    if seed is not None:candidates.append((seed['x'],'verified_route_LP:'+seed['seed_rule']))
    if out.status==2 and candidates:raise RuntimeError('MILP不可行与校核可行计划冲突，停止输出')
    rawbound=getattr(out,'mip_dual_bound',None)
    bound=float(rawbound) if rawbound is not None and np.isfinite(rawbound) else -np.inf
    if not candidates:return {'x':None,'status':int(out.status),'message':str(out.message),'certified_infeasible':out.status==2,
                               'R_lower':None,'R_upper':None,'seconds':time.perf_counter()-start,'source':None,'seed_candidates':seed_log}
    x,source=min(candidates,key=lambda a:float(m.c@a[0]));best=float(m.c@x)
    if out.status==0 and not np.isfinite(bound):bound=best
    denom=md['denominator'];past=md['past_reward'];rl=(past-best)/denom
    remaining=md['cfg']['dt']*(c['critical_weight']*c['profiles']['critical'][:,md['state']['step']:].sum()+c['general_weight']*c['profiles']['general'][:,md['state']['step']:].sum())
    universal=min(1.,(past+remaining)/denom)
    ru=min(universal,(past-bound)/denom) if np.isfinite(bound) else universal
    # 已恢复历史固定；上界再结合剩余总需求的普适上限。
    cost=np.asarray(md['cost_objective']);row=sparse.csr_matrix(m.c.reshape(1,-1));eps=c['restore_tolerance']*denom
    secondary=LinearModel(cost,m.integrality,m.lb,m.ub,sparse.vstack([m.A,row],format='csr'),np.r_[m.row_lb,-np.inf],np.r_[m.row_ub,best+eps],m.var_indices,md)
    limit=float(secondary_seconds if secondary_seconds is not None else cfg['secondary_seconds'])
    if limit>0:
        # 固定当前整数后先求一个低成本LP，保证二阶段有可行候选。
        fixed=freeze(secondary,x);lp=solve_lp(fixed)
        if lp['x'] is not None and residual(secondary,lp['x'])<=2e-5:x=lp['x']
        rr=solve_mip(secondary,{'time_limit':limit,'mip_rel_gap':cfg['rel_gap'],'presolve':True,'disp':False},x)
        if rr.x is not None and residual(secondary,rr.x)<=2e-5 and cost@rr.x<cost@x:x=rr.x
    rl=max(rl,float((past-m.c@x)/denom))
    primary_rplan=float((past-best)/denom)
    return {'x':x,'status':int(out.status),'message':str(out.message),'source':source,
            'certified_infeasible':False,'R_lower':float(rl),'R_upper':float(max(rl,ru)),
            'R_plan':float((past-m.c@x)/denom),'cost_plan':float(cost@x+md['cost_constant']),
            'primary_objective_lower_bound':bound,'primary_objective_incumbent':best,
            'max_residual':residual(m,x),'seconds':time.perf_counter()-start,'seed_candidates':seed_log,
            'best_seed_R':None if seed is None else float((past-seed['objective'])/denom),
            'primary_R_plan':primary_rplan,'absolute_R_gap':float(max(0.,ru-rl)),
            'MILP_improvement_over_seed_R':None if seed is None else float((seed['objective']-best)/denom),
            'secondary_R_loss':float(max(0.,primary_rplan-(past-m.c@x)/denom)),
            'interface_envelope_count':len(md.get('interface_envelopes',[]))}


def extract(m,x,scenario=0):
    v=m.var_indices;md=m.metadata;d={k:np.asarray(x)[idx][scenario] for k,idx in v.items() if np.asarray(idx).ndim>=2 and k!='__none'}
    starts=d['target_start'];d['aux']=md['fixed_aux']+starts@md['auxcoef'].T;d['online']=md['fixed_on']+starts@md['oncoef'].T
    d['upward_p']=-d['flow_p'][md['cfg']['tx']];d['upward_q']=-d['flow_q'][md['cfg']['tx']]
    d['job_plan']=[]
    for j,(job,idx) in enumerate(zip(md['jobs'],md['job_indices'])):
        for col,(start,end) in zip(idx,job['options']):
            if x[col]>.5:d['job_plan'].append({'edge':job['edge'],'start':start,'end':end})
    d['job_plan'].sort(key=lambda r:r['start'])
    if md['forecast'].get('crew_plan'):d['job_plan']=md['forecast']['crew_plan']['tasks'][scenario]
    return d


def validate(m,x,tolerance=2e-5):
    """矩阵以外独立重建森林、节点平衡与能量平衡，核查非预期性。"""
    if x is None:return {'passed':False,'reason':'没有可行候选'}
    md=m.metadata;c=md['cfg'];fc=md['forecast'];v=m.var_indices;dt=c['dt'];checks={'matrix_and_bounds':residual(m,x)}
    def val(k):return x[v[k]]
    maxcycle=maxsource=maxbalance=maxenergy=maxna=0.;ns=len(fc['probabilities']);n=len(md['hours'])
    for s in range(ns):
        d=extract(m,x,s);block=md['block'];live=d['energized_bus'][:,block];active=d['energized_edge'][:,block]
        for k,t in enumerate(md['slots']):
            groups,cycle=components(c,d['energized_edge'][:,k]);maxcycle=max(maxcycle,float(cycle))
            for group in groups:
                if not any(d['energized_bus'][i,k]>.5 for i in group):continue
                eligible=set(group)&set(c['mg_nodes'])
                if 0 in group and (c['alpha']>0 or d['online'][t]>.5):eligible.add(0)
                if not eligible:maxsource=1.
        load=d['served_c']+d['served_g'];pr=-load.copy();qr=-load*c['load']['reactive_ratio'][:,None]
        pr[0]+=d['grid_p']+d['target_p']-d['aux'];qr[0]+=d['grid_q']+d['target_q']-d['aux']*np.tan(np.arccos(c['generator']['aux_power_factor']))
        for j,node in enumerate(c['mg_nodes']):
            pr[node]+=d['pv'][j]+d['dg'][j]+d['discharge'][j]-d['charge'][j];qr[node]+=d['mg_q'][j]
            mg=c['microgrids'][j];err=np.diff(d['energy'][j])-dt*(mg['eta_charge']*d['charge'][j]-d['discharge'][j]/mg['eta_discharge'])
            maxenergy=max(maxenergy,float(np.max(np.abs(err),initial=0)))
        for e,line in enumerate(c['edges']):
            pr[line['from']]-=d['flow_p'][e];pr[line['to']]+=d['flow_p'][e]
            qr[line['from']]-=d['flow_q'][e];qr[line['to']]+=d['flow_q'][e]
        maxbalance=max(maxbalance,float(np.max(np.abs(pr),initial=0)),float(np.max(np.abs(qr),initial=0)))
    for t,h in enumerate(md['hours']):
        groups=defaultdict(list)
        for s in range(ns):groups[history_key(fc,s,t)].append(s)
        for group in groups.values():
            for key in ('charge','discharge','pv','dg','served_c','served_g','target_p','grid_p'):
                a=val(key)[group,...,t];maxna=max(maxna,float(np.max(np.abs(a-a[0]),initial=0)))
    # 独立重建队伍实际顺序和起点时间，检查不存在隐含重叠。
    crewerr=0.
    if fc.get('crew_plan'):
        validate_policy(fc['crew_plan'],fc)
        current=md['state']['crew']['active']
        if current:
            for trace in fc['crew_plan']['trace']:
                if trace[0] is None or trace[0][0]!=current['edge'] or trace[0][1]!=current['kind']:crewerr=1.
    else:
        plan=extract(m,x,0)['job_plan'];origin=md['crew_context']['origin'];clock=md['crew_context']['ready_hour']
        for job in plan:
            travel=travel_hours(c,origin,job['edge']);crewerr=max(crewerr,clock+travel-job['start'])
            clock=job['end'];origin=job['edge']
    ops_error=0.
    for s in range(ns):
        closed=val('closed')[s];previous=np.asarray(md['state']['closed']);counts=np.asarray(md['state']['switch_count'],float).copy();last=np.asarray(md['state']['last_change']).copy()
        for k,t in enumerate(md['slots']):
            h=md['hours'][t];changed=np.abs(closed[:,k]-previous)>.5
            forced=np.isclose(fc['failure_hour'][s],h)&(closed[:,k]<.5)&(previous>.5)
            controlled=changed&~forced&np.array([edge['switchable'] for edge in c['edges']])
            if np.any(controlled):ops_error=max(ops_error,float(np.max(c['switch_min_hold_hours']-(h-last[controlled]),initial=0)))
            counts[controlled]+=1;last[controlled]=h;previous=closed[:,k]
        ops_error=max(ops_error,float(np.max(counts-c['switch_max_per_line'],initial=0)),float(counts.sum()-c['switch_max_total']))
    checks.update(crew_route_timing=max(0.,crewerr),switch_operations=max(0.,ops_error),forest_cycles=maxcycle,component_source=maxsource,nodal_PQ=maxbalance,SOC_balance=maxenergy,conditional_nonanticipativity=maxna)
    return {'passed':all(np.isfinite(a) and a<=tolerance for a in checks.values()),'checks':checks,'tolerance':tolerance,
            'scope':'预测场景下的线性无损准稳态模型；不代表频率/同期/涌流暂态验证'}
