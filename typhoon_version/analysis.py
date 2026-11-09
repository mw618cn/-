"""三个研究点的计算入口。

①同状态同预测样本下的启动/等待D_R及求解界；②固定全部离散变量后的
完整多时段条件域和整轨迹分解；③实际观测触发的抢修—重构—启动滚动。
真实台风轨迹只用于环境推进，模型用独立预测；实际执行收益单独统计。
"""
from pathlib import Path
import copy,time
import numpy as np
from scipy import sparse
from data import save_json,jsonable
from weather import forecast
from power_model import build_model,optimize,extract,validate,freeze,solve_lp,residual


def initial_state(c,world):
    ne=len(c['edges']);damaged=np.asarray(world['failure_hour'])<=0
    closed=np.array([e['initial_closed'] for e in c['edges']],float);closed[damaged]=0
    return {'step':0,'energy':np.array([m['energy_mwh']*m['soc_initial'] for m in c['microgrids']]),
        'damaged':damaged.copy(),'ever_failed':damaged.copy(),'observed_damage':{},
        'failure_observed_hour':{int(e):float(world['failure_hour'][e]) for e in np.flatnonzero(damaged)},
        'closed':closed,'last_change':np.full(ne,-1e9),'switch_count':np.zeros(ne,int),'previous_pg':0.,
        'crew':{'position':np.array([c['crew']['depot_x_km'],c['crew']['depot_y_km']]),'active':None},
        'unit':{'start_hour':None,'attempts':0,'earliest_start_hour':0.},'reward_accum':0.,'cost_accum':0.,
        'new_fault':bool(damaged.any())}


def decide(c,state,fc,policy='joint',seconds=None,secondary_seconds=None,compare=True):
    """D_R的界针对当前预测模型，非实际未知未来的全局POMDP价值界。"""
    now=state['step']*c['dt'];can_compare=compare and state['unit']['start_hour'] is None and now>=state['unit']['earliest_start_hour']-1e-8 and now<=c['generator']['last_start_hour']+1e-8 and state['unit']['attempts']<c['generator']['max_attempts']
    branches=('start','wait') if can_compare else ('free',);records={}
    for branch in branches:
        m=build_model(c,state,fc,branch,policy);r=optimize(m,seconds,secondary_seconds)
        if r['x'] is not None:
            check=validate(m,r['x'])
            if not check['passed']:raise RuntimeError(f'{branch}候选解未通过物理检查：{check}')
            r['validation']=check
        records[branch]=(m,r)
    feasible={k:v for k,v in records.items() if v[1]['x'] is not None}
    result={'hour':now,'policy':policy,'D_estimate':None,'D_lower':None,'D_upper':None,'certified':False,
            'decision':'机组承诺已固定或当前不在比较窗口','classification':'not_applicable'}
    if can_compare:
        a,b=records['start'][1],records['wait'][1];eps=c['phase_restore_tolerance']
        if a['x'] is not None and b['x'] is not None:
            lo=a['R_lower']-b['R_upper'];hi=a['R_upper']-b['R_lower'];estimate=a['R_plan']-b['R_plan']
            result.update(D_estimate=float(estimate),D_lower=float(lo),D_upper=float(hi))
            if lo>eps:result.update(decision='恢复率证据支持当前启动',classification='start',certified=True)
            elif hi<-eps:result.update(decision='恢复率证据支持等待',classification='wait',certified=True)
            elif lo>=-eps and hi<=eps:result.update(decision='恢复水平已证近似等效，按已找到的次级成本选择',classification='recovery_tie',certified=True)
            else:result.update(decision='恢复率界尚未分离，按已找到的可行计划执行',classification='unknown')
        elif a['certified_infeasible'] and b['x'] is not None:result.update(decision='当前预测模型已证当前启动不可行',classification='start_infeasible',certified=True)
        elif b['certified_infeasible'] and a['x'] is not None:result.update(decision='当前预测模型已证等待分支不可行',classification='wait_infeasible',certified=True)
        else:result.update(decision='至少一分支未取得可行计划，不以超时判不可行',classification='unknown')
    if feasible:
        if result['classification']=='start':chosen='start'
        elif result['classification']=='wait':chosen='wait'
        else:
            maxr=max(v[1]['R_plan'] for v in feasible.values())
            near={k:v for k,v in feasible.items() if v[1]['R_plan']>=maxr-c['restore_tolerance']}
            chosen=min(near,key=lambda k:near[k][1]['cost_plan'])
        result.update(action=chosen,job_count=len(records[chosen][0].metadata['jobs']))
    else:chosen=None;result.update(action='no_verified_plan',job_count=0)
    result['branches']={k:{a:b for a,b in r.items() if a!='x'} for k,(m,r) in records.items()}
    if chosen is not None:
        result['interface_envelope_count']=len(records[chosen][0].metadata['interface_envelopes'])
        result['interface_envelopes']=records[chosen][0].metadata['interface_envelopes']
        result['inherited_SOC']=[float(state['energy'][j]/mg['energy_mwh']) for j,mg in enumerate(c['microgrids'])]
    return result,records.get(chosen),records


def _event(events,h,kind,c,e=None,damage='',note=''):
    events.append({'hour':float(h),'kind':kind,'line':c['edges'][e]['name'] if e is not None else '',
                   'damage':damage,'note':note})


def observe_faults(c,state,world,events):
    h=state['step']*c['dt'];new=[]
    for e,t in enumerate(world['failure_hour']):
        if t<=h+1e-8 and not state['ever_failed'][e]:
            state['ever_failed'][e]=True;state['damaged'][e]=True;state['closed'][e]=0
            state['failure_observed_hour'][e]=h;new.append(e)
            _event(events,h,'保护跳闸/报告新故障',c,e,note='位置已知；轻重尚未巡检，不计主动开关次数')
    state['new_fault']=bool(new);return bool(new)


def advance_crew(c,state,world,events):
    """推进真实队伍一个15分钟；只有巡检实际完成后才读取隐藏损伤。"""
    active=state['crew']['active']
    if active is None:return False
    dt=c['dt'];h=state['step']*dt;e=active['edge'];remaining=dt
    travel=min(remaining,active['travel_remaining']);active['travel_remaining']-=travel;remaining-=travel
    changed=False
    if active['travel_remaining']<=1e-8:
        state['crew']['position']=c['edge_xy'][e].copy()
        if remaining>1e-8 and world['wind'][e,state['step']]<=c['crew']['safe_wind_ms']+1e-8:
            active['work_remaining']-=remaining
        if active['work_remaining']<=1e-8:
            when=h+dt
            if active['kind']=='inspect':
                severity=world['severity'][e];state['observed_damage'][e]=severity
                _event(events,when,'巡检完成/获知损伤',c,e,damage=severity,note='先获得信息；是否维修在下一次优化决定')
            else:
                state['damaged'][e]=False;_event(events,when,'维修完成/线路可用',c,e,note='未强制合闸，等待优化指令')
            state['crew']['active']=None;changed=True
    state['cost_accum']+=dt*c['crew']['hourly_cost'];return changed


def dispatch_crew(c,state,d,events,h):
    if state['crew']['active'] is not None:return
    for job in d['job_plan']:
        e=job['edge']
        if not state['damaged'][e]:continue
        from repair_model import travel_hours
        travel=travel_hours(c,state['crew']['position'],e)
        if job['start']-travel>h+1e-8:return
        known=state['observed_damage'].get(e);kind='repair' if known else 'inspect'
        work=c['edges'][e][f'repair_{known}_hours'] if known else c['edges'][e]['inspection_hours']
        state['crew']['active']={'edge':e,'kind':kind,'travel_remaining':float(travel),'work_remaining':float(work)}
        _event(events,h,'分派'+('维修' if known else '巡检'),c,e,damage=known or '',note='当前已报告任务；行程计入执行时间')
        return


def _compatible(md,state,h):
    start=md['hours'][0];actual={(int(e),float(t)) for e,t in state['failure_observed_hour'].items() if t>start+1e-8 and t<=h+1e-8}
    return [s for s in range(len(md['forecast']['probabilities'])) if {(int(e),float(md['forecast']['failure_hour'][s,e])) for e in np.flatnonzero(md['forecast']['failure_hour'][s]<=h+1e-8)}==actual]


def simulate(c,world,policy='joint',output=None,progress=print,capture_initial=False):
    """完整96步真实执行；事件或定期触发重新优化，不把一次预测当成全天结果。"""
    state=initial_state(c,world);initial=copy.deepcopy(state);dt=c['dt'];events=[];rounds=[]
    for e in np.flatnonzero(state['damaged']):_event(events,0,'初始已报告故障',c,int(e),note='由灾前风场历史生成；损伤轻重未知')
    result={k:[] for k in ('served_c','served_g','flow_p','flow_q','closed','energized_edges','energized_bus','voltage','target_p','aux','upward_p','unit_online','recovery','pv','dg','charge','discharge','pcc_p','pcc_q','damaged','crew_trace')}
    energy=[state['energy'].copy()];active_plan=None;selected=None;lastsolve=-1e9;trigger=True;initial_capture=None;emergencies=0
    for step in range(c['n_steps']):
        state['step']=step;h=step*dt;trigger=observe_faults(c,state,world,events) or trigger
        compatible=[] if active_plan is None else _compatible(active_plan[0].metadata,state,h)
        if trigger or h-lastsolve>=c['replan_hours']-1e-8 or not compatible:
            fc=forecast(c,state);decision,selected,allrecords=decide(c,copy.deepcopy(state),fc,policy)
            if selected is None:
                # 安全运行回退：尚未完成的启动可因缺少已校核调度而保守中止。
                # 不将“未找到计划”表述为已证明物理不可行。
                starth=state['unit']['start_hour'];duration=len(c['generator']['startup_profile_mw'])*dt
                if starth is not None and h<starth+duration-1e-8:
                    state['unit']['start_hour']=None;state['unit']['earliest_start_hour']=h+c['generator']['restart_delay_hours']
                    _event(events,h,'启动安全中止',c,note='未取得通过校核的连续支撑调度；已耗能量不退回')
                emergency=copy.deepcopy(c)
                for mg in emergency['microgrids']:mg['soc_terminal']=mg['soc_min']
                em=build_model(emergency,copy.deepcopy(state),fc,'free',policy);rr=optimize(em,seconds=.01,secondary_seconds=0)
                if rr['x'] is None or not validate(em,rr['x'])['passed']:
                    raise RuntimeError(f'{h:g}h无可执行的正常或应急计划，停止仿真：{decision}')
                selected=(em,rr);emergencies+=1
                decision.update(action='emergency_verified',decision='应急保供，终端目标储备放宽至物理下限',certified=False)
                _event(events,h,'应急保供',c,note='显式放宽终端目标，实际结果不可作为正常储备约束证书')
            m,r=selected;active_plan=selected;lastsolve=h;trigger=False;compatible=[0]
            decision['hour']=h;rounds.append(jsonable(decision))
            if output:save_json(Path(output)/'rounds'/f'{step:03d}.json',decision)
            if capture_initial and initial_capture is None:initial_capture=(m,r,initial)
            if progress:progress(f'[{policy}] t={h:g}h，已报告{int(state["ever_failed"].sum())}处，未修{int(state["damaged"].sum())}处；{decision["action"]}；{decision["classification"]}',flush=True)
        m,r=active_plan;s=compatible[0];t=step-int(round(m.metadata['hours'][0]/dt));d=extract(m,r['x'],s);k=m.metadata['block'][t]
        proposed=np.rint(d['closed'][:,k]);changed=np.flatnonzero(proposed!=state['closed'])
        for e in changed:
            if state['damaged'][e] and proposed[e]>.5:raise RuntimeError('未修复线路合闸')
            if h-state['last_change'][e]<c['switch_min_hold_hours']-1e-7:raise RuntimeError('主动开关保持时间违约')
            state['switch_count'][e]+=1;state['last_change'][e]=h
            _event(events,h,'主动合闸' if proposed[e]>.5 else '主动分闸',c,int(e))
        if (state['switch_count']>c['switch_max_per_line']+1e-7).any() or state['switch_count'].sum()>c['switch_max_total']+1e-7:raise RuntimeError('操作预算违约')
        state['closed']=proposed;state['cost_accum']+=len(changed)*c['switch_cost']
        if state['unit']['start_hour'] is None:
            at=np.flatnonzero(m.metadata['start_slots']==t)
            if len(at) and d['target_start'][at[0]]>.5:
                state['unit']['start_hour']=h;state['unit']['attempts']+=1;state['cost_accum']+=c['generator']['start_cost']
                _event(events,h,'机组开始启动',c)
        # 当前计划的第一步在共同信息下相同；实际PV取真实风场上限复核。
        realpv=c['profiles']['pv'][:,step]*np.clip(1-c['weather']['pv_wind_derating']*world['wind'][c['mg_edge_indices'],step]/50,.2,1)
        if np.max(d['pv'][:,t]-realpv,initial=0)>2e-5:raise RuntimeError('当前计划超出真实可用PV，需更新预测口径')
        if np.max(np.abs(d['energy'][:,t]-state['energy']),initial=0)>2e-5:raise RuntimeError('滚动SOC未继承')
        for key in ('served_c','served_g','flow_p','flow_q','target_p','aux','upward_p','pv','dg','charge','discharge','pcc_p','pcc_q'):
            result[key].append(np.asarray(d[key][...,t]).tolist())
        result['closed'].append(proposed.tolist());result['energized_edges'].append(np.rint(d['energized_edge'][:,k]).tolist())
        result['energized_bus'].append(np.rint(d['energized_bus'][:,k]).tolist());result['voltage'].append(np.sqrt(np.maximum(d['voltage'][:,t],0)).tolist())
        result['unit_online'].append(float(d['online'][t]))
        reward=dt*(c['critical_weight']*sum(d['served_c'][:,t])+c['general_weight']*sum(d['served_g'][:,t]));state['reward_accum']+=reward
        demand=c['critical_weight']*sum(c['profiles']['critical'][:,step])+c['general_weight']*sum(c['profiles']['general'][:,step])
        result['recovery'].append(float(reward/dt/demand) if demand else 1.)
        running=dt*(sum(d['dg'][:,t]*np.array([a['dg_cost'] for a in c['microgrids']]))+
            sum((d['charge'][:,t]+d['discharge'][:,t])*np.array([a['throughput_cost'] for a in c['microgrids']]))+
            d['target_p'][t]*c['generator']['fuel_cost']+d['grid_p'][t]*c['profiles']['price'][step])
        state['cost_accum']+=float(running);state['energy']=d['energy'][:,t+1].copy();energy.append(state['energy'].copy());state['previous_pg']=float(d['target_p'][t])
        result['damaged'].append(state['damaged'].astype(int).tolist())
        dispatch_crew(c,state,d,events,h)
        active=copy.deepcopy(state['crew']['active'])
        if active is not None:active['weather_wait']=bool(active['travel_remaining']<=1e-8 and world['wind'][active['edge'],step]>c['crew']['safe_wind_ms']+1e-8)
        result['crew_trace'].append(active)
        trigger=advance_crew(c,state,world,events)
    result.update(policy=policy,time_hours=(np.arange(c['n_steps'])*dt).tolist(),dt=dt,energy=np.asarray(energy).tolist(),
        energy_capacity=[m['energy_mwh'] for m in c['microgrids']],edge_names=[e['name'] for e in c['edges']],events=events,rounds=rounds)
    served=np.asarray(result['served_c'])+np.asarray(result['served_g']);demand=(c['profiles']['critical']+c['profiles']['general']).T
    eens=float(dt*(demand-served).sum());weighted=state['reward_accum']/c['denominator']
    result['summary']={'actual_weighted_recovery':float(weighted),'actual_EENS_MWh':eens,'actual_operation_cost_thousand':float(state['cost_accum']),
        'manual_switch_operations':int(state['switch_count'].sum()),'reported_fault_count':int(state['ever_failed'].sum()),
        'unrepaired_end_count':int(state['damaged'].sum()),'startup_attempts':state['unit']['attempts'],'reoptimizations':len(rounds),'emergency_rounds':emergencies,
        'terminal_reserve_satisfied':bool(all(state['energy'][j]>=mg['energy_mwh']*mg['soc_terminal']-2e-5 for j,mg in enumerate(c['microgrids']))),
        'scope':'当前策略沿一个合成风灾样本的实际执行，非全策略全场景最优韧性值'}
    if output:save_json(Path(output)/'executed_result.json',result)
    return result,initial_capture


def compute_capability(m,x,c,output,progress=print):
    """MGO局部资源域与网络可送能力分开；保留全时域及全部预测样本。

    图形局部LP只固定充放电模式；全局网络可送范围另算。逐时切片不能
    任意拼接。分解证书针对给定整轨迹，不代表所有接口指令都可分解。
    """
    from capability import build_resource_model
    state=m.metadata['state'];fc=m.metadata['forecast'];v=m.var_indices
    ns,n=v['flow_p'].shape[0],v['flow_p'].shape[-1]
    modes=np.rint(x[v['battery_mode']][:,:,m.metadata['mode_map']])
    local=build_resource_model(c,state,fc,modes);idx=local.var_indices
    stride=max(1,int(round(c['capability_interval_hours']/c['dt'])))
    slices=[];delivery=[];witnesses={}
    reference_all_p=x[v['pcc_p']];reference_all_q=x[v['pcc_q']]
    refp=reference_all_p.sum(axis=1);refq=reference_all_q.sum(axis=1)
    network=freeze(m,x);tx=c['tx']
    times=list(range(0,n,stride))
    for t in times:
        points=[]
        for j,a in enumerate(np.linspace(0,2*np.pi,c['settings']['capability_directions'],endpoint=False)):
            obj=np.zeros_like(local.c);obj[idx['pcc_p'][0,:,t]]=-np.cos(a);obj[idx['pcc_q'][0,:,t]]=-np.sin(a)
            r=solve_lp(local,obj)
            if r['x'] is not None and residual(local,r['x'])<=1e-6:
                z=r['x'];points.append([float(z[idx['pcc_p'][0,:,t]].sum()),float(z[idx['pcc_q'][0,:,t]].sum())]);witnesses[f't{t}_d{j}']=z
        slices.append({'hour':float(fc['hours'][t]),'points':points})
        bounds=[]
        for sign in (-1,1):
            obj=np.zeros_like(network.c);obj[v['flow_p'][0,tx,t]]=sign
            rr=solve_lp(network,obj)
            if rr['x'] is not None:bounds.append(-float(rr['x'][v['flow_p'][0,tx,t]]))
        delivery.append({'hour':float(fc['hours'][t]),'lower':min(bounds) if bounds else None,'upper':max(bounds) if bounds else None})
        if progress:progress(f'[能力域] {len(slices)}/{len(times)}，资源域与网络可送范围分开；完整{n}时段',flush=True)
    eq=[]
    for scenario in range(ns):
        for j in range(3):
            for t in range(n):
                eq.extend([({int(idx['pcc_p'][scenario,j,t]):1},float(reference_all_p[scenario,j,t])),
                           ({int(idx['pcc_q'][scenario,j,t]):1},float(reference_all_q[scenario,j,t]))])
    cert=solve_lp(local,equalities=eq)
    rp=np.inf if cert['x'] is None else float(np.max(np.abs(cert['x'][idx['pcc_p']]-reference_all_p),initial=0))
    rq=np.inf if cert['x'] is None else float(np.max(np.abs(cert['x'][idx['pcc_q']]-reference_all_q),initial=0))
    success=cert['x'] is not None and residual(local,cert['x'])<=1e-6 and max(rp,rq)<=1e-6
    decomposition=None
    if success:
        z=cert['x'];witnesses['decomposition']=z
        decomposition={k:z[idx[k]][0] for k in ('dg','pv','charge','discharge','energy','served_c','served_g')}
    result={'scope':'MGO局部全时域条件资源域：固定充放电模式，保留SOC/本地负荷/逆变器/同信息控制；网络可送范围另列',
            'horizon_hours':float(n*c['dt']),'n_steps':n,'scenario_count':ns,'slices':slices,'delivery_bounds':delivery,
            'reference_p':refp[0].tolist(),'reference_q':refq[0].tolist(),
            'reference_all_p':reference_all_p.tolist(),'reference_all_q':reference_all_q.tolist(),
            'tracking_p_max':rp,'tracking_q_max':rq,'time_hours':fc['hours'].tolist(),
            'decomposition_success':bool(success),'decomposition_residual':residual(local,cert['x']) if success else None,
            'decomposition':decomposition,'envelopes_used_in_dispatch':m.metadata['interface_envelopes'],
            'delivery_scope':'初始预测且固定全局离散方案下的主配接口范围；不是实际全天执行轨迹'}
    output=Path(output);output.mkdir(exist_ok=True,parents=True);save_json(output/'capability.json',result)
    np.savez_compressed(output/'full_horizon_witnesses.npz',**witnesses)
    return jsonable(result)


def phase_map(c,state,output,progress=print):
    """每格真正重优化start/wait，固定当前故障/信息与同一预测随机数。
    估计曲面和证据分类分别保存，不将未判点涂成已证最优分区。
    """
    records=[]
    for alpha in c['settings']['phase_alphas']:
        for soc in c['settings']['phase_socs']:
            trial=copy.deepcopy(c);trial['alpha']=float(alpha);st=copy.deepcopy(state)
            for j,mg in enumerate(trial['microgrids']):
                if not mg['soc_min']<=soc<=mg['soc_max']:raise ValueError('相图SOC超出物理范围')
                mg['soc_initial']=float(soc);st['energy'][j]=mg['energy_mwh']*soc
            fc=forecast(trial,st);r,chosen,_=decide(trial,st,fc,'joint',seconds=c['solver']['phase_seconds'],secondary_seconds=0)
            row={k:r.get(k) for k in ('D_estimate','D_lower','D_upper','classification','certified','branches')};row.update(alpha=alpha,soc=soc);records.append(row)
            if progress:progress(f'[D相图] alpha={alpha:g}, SOC={soc:g}：{r["classification"]}',flush=True)
    out={'scope':'初始状态下当前预测模型D_R；每格重优化，未判与不可行分开；不使用固定整数LP费用代替D',
         'alphas':c['settings']['phase_alphas'],'socs':c['settings']['phase_socs'],'records':records,'units':'恢复率之差（乘100为百分点）'}
    save_json(Path(output)/'phase_map.json',out);return out
