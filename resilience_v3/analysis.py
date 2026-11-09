"""三个研究点的计算入口。

①同状态同预测样本下的启动/等待D_R及求解界；②固定电池模式与信息树的
完整多时段条件资源域和整轨迹分解；③实际观测触发的抢修—重构—启动滚动。
真实台风轨迹只用于环境推进，模型用独立预测；实际执行收益单独统计。
"""
from pathlib import Path
import copy,time
import numpy as np
from scipy import sparse
from data import save_json,jsonable
from weather import forecast
from information import expand_hidden_types,make_crew_plan,policy_names,option,history_key
from reachability import certify_commands
from rolling import refresh_continuous
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
        'new_fault':bool(damaged.any()),'observed_wind':np.asarray(world['wind'])[:,0].copy()}


def decide(c,state,fc,policy='joint',seconds=None,secondary_seconds=None,compare=True):
    """D_R的界针对当前预测模型，非实际未知未来的全局POMDP价值界。"""
    now=state['step']*c['dt'];can_compare=compare and state['unit']['start_hour'] is None and now>=state['unit']['earliest_start_hour']-1e-8 and now<=c['generator']['last_start_hour']+1e-8 and state['unit']['attempts']<c['generator']['max_attempts']
    branches=('start','wait') if can_compare else ('free',);records={};evaluations={};refinement=[]
    base=fc if 'severity' in fc else expand_hidden_types(c,state,fc)
    names=policy_names(c,policy);prepared=[]
    for name in names:
        trial=copy.copy(base);trial['crew_plan']=make_crew_plan(c,state,base,name);prepared.append(trial)
    for branch in branches:
        for trial in prepared:
            m=build_model(c,state,trial,branch,policy)
            budget=(seconds if seconds is not None else c['solver']['primary_seconds'])/len(prepared)
            r=optimize(m,budget,secondary_seconds)
            if r['x'] is not None:
                check=validate(m,r['x'])
                if not check['passed']:raise RuntimeError(f'{branch}候选物理/信息检查失败：{check}')
                r['validation']=check
            evaluations[branch,trial['crew_plan']['name']]=(m,r)
    def bounds(branch):
        items=[r for (a,name),(m,r) in evaluations.items() if a==branch]
        feasible=[r for r in items if r['x'] is not None]
        lower=max((r['R_lower'] for r in feasible),default=None)
        possible=[r for r in items if not r['certified_infeasible']]
        upper=max((r.get('R_upper') if r.get('R_upper') is not None else 1. for r in possible),default=None)
        return lower,upper
    # D区间跨零时，只加算仍可能影响判据的当前候选；不改变场景/策略类。
    rounds=int(option(c,'solver.decision_refinement_rounds',1)) if can_compare else 0
    for iteration in range(rounds):
        ls,us=bounds('start');lw,uw=bounds('wait');eps=c['phase_restore_tolerance']
        if None not in (ls,us,lw,uw):
            lo,hi=ls-uw,us-lw
            if lo>eps or hi<-eps or (lo>=-eps and hi<=eps):break
        choices=[]
        for key,(m,r) in evaluations.items():
            if r['certified_infeasible']:continue
            lower,upper=bounds(key[0]);ru=r.get('R_upper') if r.get('R_upper') is not None else 1.
            if lower is not None and ru<lower-eps:continue
            gap=ru-(r.get('R_lower') if r.get('R_lower') is not None else 0.)
            choices.append((gap,key))
        if not choices:break
        _,key=max(choices);m,old=evaluations[key]
        extra=float(option(c,'solver.decision_refinement_seconds',2.))
        if extra<=0:break
        fresh=optimize(m,seconds=extra,secondary_seconds=0,warm_x=old.get('x'))
        if fresh['x'] is not None and not validate(m,fresh['x'])['passed']:raise RuntimeError('精化候选未通过物理/信息校核')
        old_feasible=old['x'] is not None;new_feasible=fresh['x'] is not None
        if old_feasible and new_feasible:
            best=old if old['R_plan']>fresh['R_plan']+c['restore_tolerance'] else fresh
            combined=dict(best);combined['R_lower']=max(old['R_lower'],fresh['R_lower']);combined['R_upper']=min(old['R_upper'],fresh['R_upper'])
            combined['absolute_R_gap']=max(0.,combined['R_upper']-combined['R_lower'])
        else:combined=fresh if new_feasible or fresh['certified_infeasible'] else old
        evaluations[key]=(m,combined)
        refinement.append({'round':iteration+1,'branch':key[0],'crew_policy':key[1],
            'R_lower_before':old.get('R_lower'),'R_upper_before':old.get('R_upper'),
            'R_lower_after':combined.get('R_lower'),'R_upper_after':combined.get('R_upper'),
            'scope':'同一模型的求解精化；不是扩大信息或策略类'})
    for branch in branches:
        items=[(m,r) for (a,name),(m,r) in evaluations.items() if a==branch]
        candidates=[(m,r) for m,r in items if r['x'] is not None]
        details=[{'crew_policy':m.metadata['crew_policy'],**{k:v for k,v in r.items() if k!='x'}} for m,r in items]
        if candidates:
            maxr=max(r['R_plan'] for _,r in candidates);near=[item for item in candidates if item[1]['R_plan']>=maxr-c['restore_tolerance']]
            m,r=min(near,key=lambda item:item[1]['cost_plan']);r=dict(r);r['R_lower'],r['R_upper']=bounds(branch)
            r['absolute_R_gap']=max(0.,r['R_upper']-r['R_lower']);r['crew_policy']=m.metadata['crew_policy']
            r['status']=0 if r['absolute_R_gap']<=c['restore_tolerance'] else 1
            r['message']='有限策略类的界已闭合至容差' if r['status']==0 else '有限策略类仍有未分离的最优值界'
            if option(c,'reachability.certify_commands',True):
                tube=certify_commands(m,r['x'])
                if not tube['passed']:raise RuntimeError(f'候选后向可达校核失败：{tube["max_tube_violation"]}')
                m.metadata['backward_certificate']=tube;r['backward_certificate']={k:v for k,v in tube.items() if k not in ('lower','upper')}
            r['policy_candidates']=details;r['bound_scope']='当前有限因果作业策略类的并集；不代表任意作业路径的全局界'
            records[branch]=(m,r)
        else:
            records[branch]=(None,{'x':None,'status':2 if all(q['certified_infeasible'] for q in details) else 1,
                'certified_infeasible':all(q['certified_infeasible'] for q in details),'R_lower':None,'R_upper':None,
                'policy_candidates':details,'source':None,'message':'因果作业候选均无已校核解'})
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
        result.update(action=chosen,job_count=int(state['damaged'].sum()))
    else:chosen=None;result.update(action='no_verified_plan',job_count=0)
    result['refinement_trace']=refinement
    result['branches']={k:{a:b for a,b in r.items() if a!='x'} for k,(m,r) in records.items()}
    if chosen is not None:
        result['interface_envelope_count']=len(records[chosen][0].metadata['interface_envelopes'])
        result['interface_envelopes']=records[chosen][0].metadata['interface_envelopes']
        result['inherited_SOC']=[float(state['energy'][j]/mg['energy_mwh']) for j,mg in enumerate(c['microgrids'])]
        result['crew_policy']=records[chosen][0].metadata['crew_policy']
        result['information_targets']=records[chosen][0].metadata['forecast']['hidden_targets']
        result['task_lock']=copy.deepcopy(state['crew']['active'])
        result['unit_start_commitment']=state['unit']['start_hour']
        result['posterior_information']={str(k):v for k,v in state['observed_damage'].items()}
        result['backward_certificate']=records[chosen][1].get('backward_certificate')
        result['bound_scope']='有限因果策略类；当前启动/等待为互斥当前动作，等待允许以后启动'
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
    travel=min(remaining,active['travel_remaining']) if world['wind'][e,state['step']]<=float(option(c,'crew.travel_wind_limit_ms',25.)) else 0.
    active['travel_remaining']-=travel;remaining-=travel
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
        if 'departure' in job:
            if job['departure']>h+1e-8:return
            if job['departure']<h-1e-8:continue
            known=state['observed_damage'].get(e);kind=job['kind']
            if kind=='repair' and known is None:raise RuntimeError('尚未检查获知损伤，不能直接维修')
            if kind=='inspect' and known is not None:continue
            work=c['edges'][e][f'repair_{known}_hours'] if kind=='repair' else c['edges'][e]['inspection_hours']
            trip=float(travel_hours(c,state['crew']['position'],e))
            limit=float(option(c,'crew.travel_wind_limit_ms',25.)) if trip>1e-8 else c['crew']['safe_wind_ms']
            if state['observed_wind'][e]>limit+1e-8:raise RuntimeError('当前观测风速不允许出发或施工，候选计划不适用')
            state['crew']['active']={'edge':e,'kind':kind,'travel_remaining':float(travel_hours(c,state['crew']['position'],e)),'work_remaining':float(work)}
            _event(events,h,'分派'+('维修' if kind=='repair' else '巡检'),c,e,damage=known or '',note='基于当前信息的因果策略首步；作业开始后单独锁定')
            return
        travel=travel_hours(c,state['crew']['position'],e)
        if job['start']-travel>h+1e-8:return
        known=state['observed_damage'].get(e);kind='repair' if known else 'inspect'
        work=c['edges'][e][f'repair_{known}_hours'] if known else c['edges'][e]['inspection_hours']
        state['crew']['active']={'edge':e,'kind':kind,'travel_remaining':float(travel),'work_remaining':float(work)}
        _event(events,h,'分派'+('维修' if known else '巡检'),c,e,damage=known or '',note='当前已报告任务；行程计入执行时间')
        return


def _compatible(md,state,h):
    """包络/旧计划适用性：完整观测历史与实际作业进度都必须匹配。

    不能只匹配故障位置：预测中已完成检查、实际尚未完成时，不得使用
    该预测样本的类型分支。这种情况下必须重新求解。
    """
    fc=md['forecast'];t=int(round((h-md['hours'][0])/md['cfg']['dt']))
    if t<0 or t>=len(md['hours']):return []
    reports=tuple(sorted((int(e),float(when)) for e,when in state['failure_observed_hour'].items()))
    diagnoses=tuple(sorted((int(e),value) for e,value in state['observed_damage'].items()))
    actual=(reports,diagnoses);valid=[]
    for s in range(len(fc['probabilities'])):
        if history_key(fc,s,t)!=actual:continue
        if fc.get('crew_plan'):
            predicted=fc['crew_plan']['trace'][s][t];active=state['crew']['active']
            command=fc['crew_plan']['dispatch'][s][t]
            if active is None:
                if predicted is not None and command is None:continue
                if command is not None:
                    from repair_model import travel_hours
                    e,kind=command;travel=travel_hours(md['cfg'],state['crew']['position'],e)
                    limit=float(option(md['cfg'],'crew.travel_wind_limit_ms',25.)) if travel>1e-8 else md['cfg']['crew']['safe_wind_ms']
                    if state['observed_wind'][e]>limit+1e-8:continue
                    if kind=='repair' and e not in state['observed_damage']:continue
            elif predicted is None or predicted[:2]!=(active['edge'],active['kind']):continue
            elif max(abs(predicted[2]-active['travel_remaining']),abs(predicted[3]-active['work_remaining']))>1e-7:continue
        valid.append(s)
    return valid


def simulate(c,world,policy='joint',output=None,progress=print,capture_initial=False):
    """完整96步真实执行；事件或定期触发重新优化，不把一次预测当成全天结果。"""
    state=initial_state(c,world);initial=copy.deepcopy(state);dt=c['dt'];events=[];rounds=[]
    for e in np.flatnonzero(state['damaged']):_event(events,0,'初始已报告故障',c,int(e),note='由灾前风场历史生成；损伤轻重未知')
    result={k:[] for k in ('served_c','served_g','flow_p','flow_q','closed','energized_edges','energized_bus','voltage','target_p','aux','upward_p','unit_online','recovery','pv','dg','charge','discharge','pcc_p','pcc_q','mg_q','grid_p','grid_q','target_q','damaged','crew_trace')}
    energy=[state['energy'].copy()];result['energy_history']=energy;controller_steps=[];active_plan=None;selected=None;lastsolve=-1e9;trigger=True;initial_capture=None;emergencies=0
    for step in range(c['n_steps']):
        state['step']=step;state['observed_wind']=np.asarray(world['wind'])[:,step].copy()
        h=step*dt;trigger=observe_faults(c,state,world,events) or trigger
        compatible=[] if active_plan is None else _compatible(active_plan[0].metadata,state,h)
        realpv=c['profiles']['pv'][:,step]*np.clip(1-c['weather']['pv_wind_derating']*world['wind'][c['mg_edge_indices'],step]/50,.2,1)
        start_interval=float(option(c,'generator.start_interval_hours',1.))
        launch=(state['unit']['start_hour'] is None and np.isclose(h/start_interval,round(h/start_interval)) and h<=c['generator']['last_start_hour'])
        from repair_model import travel_hours
        idle_job=state['crew']['active'] is None and any(state['observed_wind'][e]<=(float(option(c,'crew.travel_wind_limit_ms',25.)) if travel_hours(c,state['crew']['position'],int(e))>1e-8 else c['crew']['safe_wind_ms']) for e in np.flatnonzero(state['damaged']))
        full=option(c,'control.full_milp_each_step',False) or trigger or h-lastsolve>=c['replan_hours']-1e-8 or not compatible or launch or idle_job
        if not full:
            result['energy_history']=energy
            updated=refresh_continuous(c,state,result,*active_plan,compatible,realpv)
            if updated is None:full=True
            else:active_plan=updated
        if full:
            fc=forecast(c,state);decision,selected,allrecords=decide(c,copy.deepcopy(state),fc,policy)
            if selected is None:
                if not option(c,'control.allow_emergency_terminal_relaxation',False):
                    if output:save_json(Path(output)/'失败现场.json',{'state':state,'decision':decision,'scope':'未取得已校核计划；不把超时当作物理不可行，不自动撤销承诺或放松终端目标'})
                    raise RuntimeError(f'{h:g}h没有已校核的正常因果计划；已停止执行，详见失败现场。可增加求解预算或检查模型条件。')
                # 安全运行回退：尚未完成的启动可因缺少已校核调度而保守中止。
                # 不将“未找到计划”表述为已证明物理不可行。
                starth=state['unit']['start_hour'];duration=len(c['generator']['startup_profile_mw'])*dt
                if starth is not None and h<starth+duration-1e-8:
                    state['unit']['start_hour']=None;state['unit']['earliest_start_hour']=h+c['generator']['restart_delay_hours']
                    _event(events,h,'启动安全中止',c,note='未取得通过校核的连续支撑调度；已耗能量不退回')
                emergency=copy.deepcopy(c)
                for mg in emergency['microgrids']:mg['soc_terminal']=mg['soc_min']
                emergency_decision,emergency_selected,_=decide(emergency,copy.deepcopy(state),fc,policy,seconds=.1,secondary_seconds=0,compare=False)
                if emergency_selected is None:raise RuntimeError(f'{h:g}h无可执行的正常或应急因果计划，停止仿真：{decision}')
                em,rr=emergency_selected
                selected=(em,rr);emergencies+=1
                decision.update(action='emergency_verified',decision='应急保供，终端目标储备放宽至物理下限',certified=False)
                _event(events,h,'应急保供',c,note='显式放宽终端目标，实际结果不可作为正常储备约束证书')
            m,r=selected;active_plan=selected;lastsolve=h;trigger=False;compatible=[0]
            decision['hour']=h;decision['execution_steps']=1
            decision['horizon_end_hour']=float(c['horizon_hours']);decision['state_step']=step
            rounds.append(jsonable(decision))
            if output:save_json(Path(output)/'rounds'/f'{step:03d}.json',decision)
            if capture_initial and initial_capture is None:initial_capture=(m,r,initial)
            if progress:progress(f'[{policy}] t={h:g}h，已报告{int(state["ever_failed"].sum())}处，未修{int(state["damaged"].sum())}处；{decision["action"]}；{decision["classification"]}',flush=True)
        m,r=active_plan;s=compatible[0];t=step-int(round(m.metadata['hours'][0]/dt));d=extract(m,r['x'],s);k=m.metadata['block'][t]
        proposed=np.rint(d['closed'][:,k]);changed=np.flatnonzero(proposed!=state['closed'])
        manual_changed=[]
        for e in changed:
            if state['damaged'][e] and proposed[e]>.5:raise RuntimeError('未修复线路合闸')
            if not c['edges'][e]['switchable']:
                _event(events,h,'固定连接恢复' if proposed[e]>.5 else '固定连接断开',c,int(e),note='固定健康接线随可用性恢复；非DSO可选操作')
                continue
            if h-state['last_change'][e]<c['switch_min_hold_hours']-1e-7:raise RuntimeError('主动开关保持时间违约')
            manual_changed.append(e)
            state['switch_count'][e]+=1;state['last_change'][e]=h
            _event(events,h,'主动合闸' if proposed[e]>.5 else '主动分闸',c,int(e))
        if (state['switch_count']>c['switch_max_per_line']+1e-7).any() or state['switch_count'].sum()>c['switch_max_total']+1e-7:raise RuntimeError('操作预算违约')
        state['closed']=proposed;state['cost_accum']+=len(manual_changed)*c['switch_cost']
        if state['unit']['start_hour'] is None:
            at=np.flatnonzero(m.metadata['start_slots']==t)
            if len(at) and d['target_start'][at[0]]>.5:
                state['unit']['start_hour']=h;state['unit']['attempts']+=1;state['cost_accum']+=c['generator']['start_cost']
                _event(events,h,'机组开始启动',c)
        # 当前计划的第一步在共同信息下相同；实际PV取真实风场上限复核。
        realpv=c['profiles']['pv'][:,step]*np.clip(1-c['weather']['pv_wind_derating']*world['wind'][c['mg_edge_indices'],step]/50,.2,1)
        if np.max(d['pv'][:,t]-realpv,initial=0)>2e-5:raise RuntimeError('当前计划超出真实可用PV，需更新预测口径')
        if np.max(np.abs(d['energy'][:,t]-state['energy']),initial=0)>2e-5:raise RuntimeError('滚动SOC未继承')
        for key in ('served_c','served_g','flow_p','flow_q','target_p','aux','upward_p','pv','dg','charge','discharge','pcc_p','pcc_q','mg_q','grid_p','grid_q','target_q'):
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
        if active is not None:
            limit=float(option(c,'crew.travel_wind_limit_ms',25.)) if active['travel_remaining']>1e-8 else c['crew']['safe_wind_ms']
            active['weather_wait']=bool(world['wind'][active['edge'],step]>limit+1e-8)
        result['crew_trace'].append(active)
        controller_steps.append({'hour':h,'solve':'joint_MILP' if full else 'fixed_plan_LP','origin_hour':float(m.metadata['hours'][0]),
            'horizon_end_hour':float(c['horizon_hours']),'execute_step_minutes':c['dt']*60,
            'information':{str(e):v for e,v in state['observed_damage'].items()},'task_state':copy.deepcopy(state['crew']['active']),
            'unit_start_hour':state['unit']['start_hour'],'backward_passed':m.metadata.get('backward_certificate',{}).get('passed',False)})
        old_active=copy.deepcopy(state['crew']['active']);trigger=advance_crew(c,state,world,events)
        tube=m.metadata.get('backward_certificate')
        if tube is not None:
            if np.any(state['energy']<tube['lower'][s,:,t+1]-2e-5) or np.any(state['energy']>tube['upper'][s,:,t+1]+2e-5):
                raise RuntimeError('执行首步后的真实SOC不在下一阶段后向可达管内')
        if old_active is not None and not trigger:
            active=state['crew']['active']
            if active is None or (active['edge'],active['kind'])!=(old_active['edge'],old_active['kind']):raise RuntimeError('未完成作业被擅自取消或改派')
    result.pop('energy_history',None)
    result['controller_steps']=controller_steps
    result.update(policy=policy,time_hours=(np.arange(c['n_steps'])*dt).tolist(),dt=dt,energy=np.asarray(energy).tolist(),
        energy_capacity=[m['energy_mwh'] for m in c['microgrids']],terminal_crew=state['crew'],edge_names=[e['name'] for e in c['edges']],events=events,rounds=rounds)
    served=np.asarray(result['served_c'])+np.asarray(result['served_g']);demand=(c['profiles']['critical']+c['profiles']['general']).T
    eens=float(dt*(demand-served).sum());weighted=state['reward_accum']/c['denominator']
    result['summary']={'actual_weighted_recovery':float(weighted),'actual_EENS_MWh':eens,'actual_operation_cost_thousand':float(state['cost_accum']),
        'manual_switch_operations':int(state['switch_count'].sum()),'reported_fault_count':int(state['ever_failed'].sum()),
        'unrepaired_end_count':int(state['damaged'].sum()),'startup_attempts':state['unit']['attempts'],'reoptimizations':len(rounds),'emergency_rounds':emergencies,
        'terminal_reserve_satisfied':bool(all(state['energy'][j]>=mg['energy_mwh']*mg['soc_terminal']-2e-5 for j,mg in enumerate(c['microgrids']))),
        'scope':'有限因果作业策略类的首步滚动执行；检查信息与作业承诺分开；非完整POMDP最优值'}
    if output:save_json(Path(output)/'executed_result.json',result)
    return result,initial_capture


def compute_capability(m,x,c,output,progress=print):
    """MGO局部资源域与网络可送能力分开；保留全时域及全部预测样本。

    图形局部LP只固定充放电模式；全局网络可送范围另算。逐时切片不能
    任意拼接。分解证书针对给定整轨迹，不代表所有接口指令都可分解。
    """
    from capability import build_resource_model
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
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
        for j in range(len(c['microgrids'])):
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
    # 整轨迹内近似：同一组lambda组合所有时段/样本，不能逐时换lambda。
    vertices=list(witnesses.values());keys=list(witnesses)
    hull_summary={'vertices':len(vertices),'policy_vector_dimension':2*ns*n,'fixed_modes':True}
    splicing={'status':'not_tested'}
    if success:
        vertices.append(cert['x']);keys.append('reference')
        policy_p=np.array([z[idx['pcc_p']].sum(axis=1) for z in vertices])
        policy_q=np.array([z[idx['pcc_q']].sum(axis=1) for z in vertices])
        blend=.37*vertices[0]+.63*vertices[-1]
        hull_summary.update(vertices=len(vertices),convex_combination_residual=residual(local,blend),
            scope='固定模式和信息树下完整PCC策略向量的凸包内近似；共同lambda适用于全部时段与样本')
        # 每时段各选一个看似最好支持点，再检验其整轨迹，不预设反例一定失败。
        winner=np.argmax(policy_p[:,0,:],axis=0)
        joined_p=np.stack([policy_p[winner[t],:,t] for t in range(n)],axis=1)
        joined_q=np.stack([policy_q[winner[t],:,t] for t in range(n)],axis=1)
        eq_join=[]
        for scenario in range(ns):
            for t in range(n):
                eq_join.extend([({int(k):1. for k in idx['pcc_p'][scenario,:,t]},float(joined_p[scenario,t])),
                                ({int(k):1. for k in idx['pcc_q'][scenario,:,t]},float(joined_q[scenario,t]))])
        bad=solve_lp(local,equalities=eq_join)
        splicing={'status':'infeasible' if bad['status']==2 else 'feasible' if bad['x'] is not None else 'unknown',
                  'target_p':joined_p[0].tolist(),'target_q':joined_q[0].tolist(),
                  'scope':'逐时选择不同完整见证中的接口点，再做整轨迹可行性检验；未通过不能下发'}
        policy_cost=np.array([float(local.c@z) for z in vertices])
        np.savez_compressed(Path(output)/'trajectory_inner_hull.npz',P=policy_p,Q=policy_q,cost=policy_cost,labels=np.asarray(keys))
    result={'scope':'MGO局部全时域条件资源域：固定充放电模式，保留SOC/本地负荷/逆变器/同信息控制；网络可送范围另列',
            'horizon_hours':float(n*c['dt']),'n_steps':n,'scenario_count':ns,'slices':slices,'delivery_bounds':delivery,
            'reference_p':refp[0].tolist(),'reference_q':refq[0].tolist(),
            'reference_all_p':reference_all_p.tolist(),'reference_all_q':reference_all_q.tolist(),
            'tracking_p_max':rp,'tracking_q_max':rq,'time_hours':fc['hours'].tolist(),
            'decomposition_success':bool(success),'decomposition_residual':residual(local,cert['x']) if success else None,
            'decomposition':decomposition,'trajectory_inner_hull':hull_summary,'splicing_counterexample':splicing,'envelopes_used_in_dispatch':m.metadata['interface_envelopes'],
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
    out={'scope':'初始状态有限因果策略类的D_R；每格重优化，未判与不可行分开；不以固定整数LP冒充完整动作价值',
         'alphas':c['settings']['phase_alphas'],'socs':c['settings']['phase_socs'],'records':records,'units':'恢复率之差（乘100为百分点）'}
    save_json(Path(output)/'phase_map.json',out);return out
