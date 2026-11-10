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
        'unit':{'start_hour':None,'attempts':0,'earliest_start_hour':float(option(c,'generator.earliest_start_hour',0.))},'reward_accum':0.,'cost_accum':0.,
        'new_fault':bool(damaged.any()),'observed_wind':np.asarray(world['wind'])[:,0].copy()}


def decide(c,state,fc,policy='joint',seconds=None,secondary_seconds=None,compare=True,cache=None,progress=None):
    """同一有限策略类上的首动作认证；松弛外界与轨迹域可行下界分别保留。"""
    from bounds import class_bounds,relaxation_bound,scenario_relaxation_bound,apply_bound,model_fingerprint,reuse_cached_bounds
    from trajectory_domain import coordinate
    now=state['step']*c['dt'];can_compare=compare and state['unit']['start_hour'] is None and now>=state['unit']['earliest_start_hour']-1e-8 and now<=c['generator']['last_start_hour']+1e-8 and state['unit']['attempts']<c['generator']['max_attempts']
    branches=('start','wait') if can_compare else ('free',);evaluations={};records={};refinement=[]
    base=fc if 'severity' in fc else expand_hidden_types(c,state,fc)
    prepared=[]
    for name in policy_names(c,policy):
        trial=dict(base);trial['crew_plan']=make_crew_plan(c,state,base,name);prepared.append(trial)
    for branch in branches:
        for trial in prepared:
            if progress:progress(f'[候选] {branch}/{trial["crew_plan"]["name"]}，{len(trial["probabilities"])}样本×{len(trial["hours"])}时段',flush=True)
            m=build_model(c,state,trial,branch,policy)
            fingerprint=model_fingerprint(m);m.metadata['model_fingerprint']=fingerprint
            previous={} if cache is None else cache.get(fingerprint,{})
            budget=(seconds if seconds is not None else c['solver']['primary_seconds'])/len(prepared)
            r=optimize(m,budget,secondary_seconds,warm_x=previous.get('x'))
            reuse_cached_bounds(r,previous,fingerprint)
            r['model_fingerprint']=fingerprint
            if r['x'] is not None and not validate(m,r['x'])['passed']:raise RuntimeError('原始候选未通过物理/信息校核')
            evaluations[branch,trial['crew_plan']['name']]=(m,r)
    def bounds(branch):return class_bounds([r for (a,_),(m,r) in evaluations.items() if a==branch])
    def separated():
        for a,b in [('start','wait'),('wait','start')]:
            aa=[r for (name,_),(m,r) in evaluations.items() if name==a];bb=[r for (name,_),(m,r) in evaluations.items() if name==b]
            if all(r.get('certified_infeasible') for r in aa) and any(r.get('x') is not None for r in bb):return True
        ls,us=bounds('start');lw,uw=bounds('wait');eps=c['phase_restore_tolerance']
        return ((ls is not None and uw is not None and ls-uw>eps) or
                (lw is not None and us is not None and lw-us>eps) or
                (None not in (ls,us,lw,uw) and ls-uw>=-eps and us-lw<=eps))
    if can_compare and option(c,'solver.use_relaxation_bounds',True):
        # 先压低可能推翻当前最好可行首动作的竞争分支外界。
        total=float(option(c,'solver.decision_bound_seconds',240.));per=float(option(c,'solver.bound_lp_seconds',120.))
        close_all=bool(option(c,'solver.close_certified_interval',False));begin=time.perf_counter();attempted=set()
        while time.perf_counter()-begin<total:
            if separated():
                complete=all(r.get('certified_infeasible') or any(q.get('valid') for q in r.get('bound_certificates',[])) or (r.get('absolute_R_gap',1.)<=c['restore_tolerance']) for m,r in evaluations.values())
                if not close_all or complete:break
            ls,us=bounds('start');lw,uw=bounds('wait')
            preferred='start' if (ls if ls is not None else -1)>=(lw if lw is not None else -1) else 'wait'
            opposing='wait' if preferred=='start' else 'start';choices=[]
            for key,(m,r) in evaluations.items():
                method=str(option(c,'solver.bound_method','scenario_then_full'))
                has_bound=any(q.get('valid') for q in r.get('bound_certificates',[]))
                if separated() and close_all and has_bound:continue
                mode='scenario' if (key,'scenario') not in attempted and method!='full' else 'full'
                if mode=='scenario' and any(q.get('source')=='scenario_information_relaxation' for q in r.get('bound_certificates',[])):mode='full'
                if (key,mode) in attempted or r.get('certified_infeasible'):continue
                l,u=bounds(key[0]);upper=r.get('R_upper') if r.get('R_upper') is not None else 1.
                if l is not None and upper<l-c['restore_tolerance']:continue
                choices.append(((int(key[0]==opposing),int(mode=='scenario'),upper,r.get('R_lower') if r.get('R_lower') is not None else -1),key,mode))
            if not choices:break
            _,key,mode=max(choices);attempted.add((key,mode));m,r=evaluations[key]
            allowance=min(per,total-(time.perf_counter()-begin))
            if allowance<=0:break
            if progress:progress(f'[外界] {key[0]}/{key[1]}，{mode}，本次预算{allowance:.1f}s',flush=True)
            old_upper=r.get('R_upper');certificate=(scenario_relaxation_bound(m,allowance) if mode=='scenario' else relaxation_bound(m,allowance));apply_bound(r,certificate)
            if progress:progress(f'[外界完成] 有效={certificate.get("valid",False)}，R上界={r.get("R_upper")}，耗时{certificate["seconds"]:.1f}s',flush=True)
            refinement.append({'kind':'independent_LP_outer_bound','branch':key[0],'crew_policy':key[1],
                'R_upper_before':old_upper,'R_upper_after':r.get('R_upper'),'seconds':certificate['seconds'],'valid':certificate.get('valid',False),'bound_method':mode})
    # 外界仍未分离时，追加同一模型MIP求解；不改变场景或规则类。
    rounds=int(option(c,'solver.decision_refinement_rounds',1)) if can_compare else 0
    for iteration in range(rounds):
        if separated():break
        choices=[]
        for key,(m,r) in evaluations.items():
            if r.get('certified_infeasible'):continue
            lower,upper=bounds(key[0]);ru=r.get('R_upper') if r.get('R_upper') is not None else 1.
            if lower is not None and ru<lower-c['restore_tolerance']:continue
            choices.append((ru-(r.get('R_lower') if r.get('R_lower') is not None else 0.),key))
        if not choices:break
        _,key=max(choices);m,old=evaluations[key];extra=float(option(c,'solver.decision_refinement_seconds',2.))
        if extra<=0:break
        fresh=optimize(m,extra,0,warm_x=old.get('x'))
        if fresh['x'] is not None and not validate(m,fresh['x'])['passed']:raise RuntimeError('MIP精化候选校核失败')
        if old['x'] is not None and fresh['x'] is not None:
            combined=dict(old if old['R_plan']>fresh['R_plan']+c['restore_tolerance'] else fresh)
            combined['R_lower']=max(old['R_lower'],fresh['R_lower'])
            combined['R_upper']=min(old['R_upper'],fresh['R_upper'])
        else:combined=dict(fresh if fresh['x'] is not None or fresh.get('certified_infeasible') else old)
        combined['bound_certificates']=old.get('bound_certificates',[])
        for cert in combined['bound_certificates']:
            if cert.get('valid'):combined['R_upper']=min(combined.get('R_upper',1.),cert['R_upper'])
        if combined.get('R_lower') is not None:
            if combined['R_upper']<combined['R_lower']-2e-5:raise RuntimeError('精化后的上下界不一致')
            combined['R_upper']=max(combined['R_upper'],combined['R_lower'])
        evaluations[key]=(m,combined)
        refinement.append({'kind':'same_model_MIP','round':iteration+1,'branch':key[0],'crew_policy':key[1],
            'R_lower_before':old.get('R_lower'),'R_lower_after':combined.get('R_lower'),'R_upper_after':combined.get('R_upper')})
    evidence={}
    for branch in branches:
        items=[(m,r) for (a,_),(m,r) in evaluations.items() if a==branch];feasible=[(m,r) for m,r in items if r['x'] is not None]
        if feasible:
            bestR=max(r['R_plan'] for _,r in feasible)
            m,r=min((item for item in feasible if item[1]['R_plan']>=bestR-c['restore_tolerance']),key=lambda item:item[1]['cost_plan'])
            if option(c,'domain.coordination',True):
                if progress:progress(f'[域协调] {branch}/{m.metadata["crew_policy"]}，生成整轨迹列并消去内部变量',flush=True)
                r=coordinate(m,r);evaluations[branch,m.metadata['crew_policy']]=(m,r)
            r=dict(r);r['R_lower'],r['R_upper']=bounds(branch)
            if r['R_upper']<r['R_lower']-2e-5:raise RuntimeError('域协调后的可行下界超过有效外界')
            r['R_upper']=max(r['R_upper'],r['R_lower']);r['absolute_R_gap']=max(0.,r['R_upper']-r['R_lower']);r['crew_policy']=m.metadata['crew_policy']
            r['status']=int(r['absolute_R_gap']>c['restore_tolerance'])
            r['message']='有限策略类最优值已闭合至容差' if not r['status'] else '有限策略类仍有最优值间隙；首动作可单独被认证'
            if option(c,'reachability.certify_commands',True):
                tube=certify_commands(m,r['x'])
                if not tube['passed']:raise RuntimeError('域协调后候选未通过条件后向校核')
                m.metadata['backward_certificate']=tube;r['backward_certificate']={k:v for k,v in tube.items() if k not in ('lower','upper')}
            tube=m.metadata.get('backward_certificate',{})
            evidence[branch]={'branch':branch,'crew_policy':m.metadata['crew_policy'],'R_plan':r['R_plan'],'R_lower':r['R_lower'],'R_upper':r['R_upper'],
                'current_SOC':[float(state['energy'][j]/mg['energy_mwh']) for j,mg in enumerate(c['microgrids'])],
                'initial_lower_SOC':[float(z/mg['energy_mwh']) for z,mg in zip(tube.get('initial_lower',[]),c['microgrids'])],
                'initial_upper_SOC':[float(z/mg['energy_mwh']) for z,mg in zip(tube.get('initial_upper',[]),c['microgrids'])],
                'passed':tube.get('passed',False),'backward_certificate':tube,'domain':r.get('domain'),
                'scope':'给定可行计划的局部条件后向证据及完整网络见证；不是精确全系统动作可达集'}
            r['policy_candidates']=[{'crew_policy':mm.metadata['crew_policy'],**{k:v for k,v in rr.items() if k!='x'}} for (a,_),(mm,rr) in evaluations.items() if a==branch]
            r['bound_scope']='当前有限因果规则并集；松弛界覆盖原MILP，轨迹列主问题仅给可行下界'
            records[branch]=(m,r)
        else:
            all_infeasible=all(rr.get('certified_infeasible') for mm,rr in items);lower,upper=bounds(branch)
            records[branch]=(None,{'x':None,'status':2 if all_infeasible else 1,'certified_infeasible':all_infeasible,
                'R_lower':lower,'R_upper':upper,'source':None,'message':'无已校核计划；无解不等于已证不可行',
                'policy_candidates':[{'crew_policy':mm.metadata['crew_policy'],**{k:v for k,v in rr.items() if k!='x'}} for mm,rr in items]})
    result={'hour':now,'policy':policy,'D_estimate':None,'D_lower':None,'D_upper':None,'certified':False,
        'decision':'机组承诺已固定或当前不在比较窗口','classification':'not_applicable','refinement_trace':refinement}
    feasible={k:v for k,v in records.items() if v[1]['x'] is not None}
    if can_compare:
        a,b=records['start'][1],records['wait'][1];eps=c['phase_restore_tolerance']
        lo=None if a.get('R_lower') is None or b.get('R_upper') is None else a['R_lower']-b['R_upper']
        hi=None if a.get('R_upper') is None or b.get('R_lower') is None else a['R_upper']-b['R_lower']
        result.update(D_lower=lo,D_upper=hi)
        if a['x'] is not None and b['x'] is not None:result['D_estimate']=a['R_plan']-b['R_plan']
        if a.get('certified_infeasible') and b.get('certified_infeasible'):result.update(decision='两动作在当前条件模型均已证不可行',classification='both_infeasible',certified=True)
        elif a.get('certified_infeasible') and b['x'] is not None:result.update(decision='当前预测模型已证当前启动不可行',classification='start_infeasible',certified=True)
        elif b.get('certified_infeasible') and a['x'] is not None:result.update(decision='当前预测模型已证等待不可行',classification='wait_infeasible',certified=True)
        elif a['x'] is not None and lo is not None and lo>eps:result.update(decision='有效最优值界支持当前启动',classification='start',certified=True)
        elif b['x'] is not None and hi is not None and hi<-eps:result.update(decision='有效最优值界支持当前等待',classification='wait',certified=True)
        elif lo is not None and hi is not None and lo>=-eps and hi<=eps:result.update(decision='恢复水平已证近等效，按已找到成本选择',classification='recovery_tie',certified=True)
        else:result.update(decision='界未分离，按已校核可行计划执行',classification='unknown')
    chosen=None
    if feasible:
        if result['classification']=='start':chosen='start'
        elif result['classification']=='wait':chosen='wait'
        else:
            bestR=max(v[1]['R_plan'] for v in feasible.values());near={k:v for k,v in feasible.items() if v[1]['R_plan']>=bestR-c['restore_tolerance']}
            chosen=min(near,key=lambda k:near[k][1]['cost_plan'])
        result.update(action=chosen,job_count=int(state['damaged'].sum()))
    else:result.update(action='no_verified_plan',job_count=0)
    result['branches']={key:{k:v for k,v in rr.items() if k!='x'} for key,(mm,rr) in records.items()}
    result['action_evidence']={key:{k:v for k,v in item.items() if k!='backward_certificate'} for key,item in evidence.items()}
    if chosen is not None:
        m,r=records[chosen];m.metadata['action_evidence']=evidence
        result.update(interface_envelope_count=len(m.metadata['interface_envelopes']),interface_envelopes=m.metadata['interface_envelopes'],
            inherited_SOC=[float(state['energy'][j]/mg['energy_mwh']) for j,mg in enumerate(c['microgrids'])],
            crew_policy=m.metadata['crew_policy'],information_targets=m.metadata['forecast']['hidden_targets'],
            target_selection=m.metadata['forecast'].get('target_selection',[]),task_lock=copy.deepcopy(state['crew']['active']),
            unit_start_commitment=state['unit']['start_hour'],posterior_information={str(k):v for k,v in state['observed_damage'].items()},
            backward_certificate=r.get('backward_certificate'),domain=r.get('domain'),bound_scope='有限因果规则类、当前预测支持；非一般POMDP最优值')
    if cache is not None:
        for key,(mm,rr) in evaluations.items():
            cache[mm.metadata['model_fingerprint']]={k:rr[k] for k in ('x','R_lower','R_upper','bound_certificates') if k in rr}
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
    energy=[state['energy'].copy()];result['energy_history']=energy;controller_steps=[];active_plan=None;selected=None;lastsolve=-1e9;trigger=True;initial_capture=None;emergencies=0;state_snapshots=[]
    for step in range(c['n_steps']):
        state['step']=step;state['observed_wind']=np.asarray(world['wind'])[:,step].copy()
        h=step*dt;new_fault=observe_faults(c,state,world,events)
        reasons=[]
        if step==0:reasons.append('initial_state')
        elif trigger:reasons.append(pending_event_reason)
        if new_fault:reasons.append('new_fault_report')
        compatible=[] if active_plan is None else _compatible(active_plan[0].metadata,state,h)
        if active_plan is not None and not compatible:reasons.append('information_or_progress_changed')
        realpv=c['profiles']['pv'][:,step]*np.clip(1-c['weather']['pv_wind_derating']*world['wind'][c['mg_edge_indices'],step]/50,.2,1)
        start_interval=float(option(c,'generator.start_interval_hours',1.))
        launch=(state['unit']['start_hour'] is None and np.isclose(h/start_interval,round(h/start_interval)) and h<=c['generator']['last_start_hour'] and h>=state['unit']['earliest_start_hour']-1e-8)
        from repair_model import travel_hours
        idle_job=state['crew']['active'] is None and any(state['observed_wind'][e]<=(float(option(c,'crew.travel_wind_limit_ms',25.)) if travel_hours(c,state['crew']['position'],int(e))>1e-8 else c['crew']['safe_wind_ms']) for e in np.flatnonzero(state['damaged']))
        if option(c,'control.full_milp_each_step',False):reasons.append('configured_full_MILP')
        if h-lastsolve>=c['replan_hours']-1e-8:reasons.append('periodic_replan')
        if active_plan is None:reasons.append('no_previous_plan')
        if launch:reasons.append('startup_candidate_time')
        if idle_job:reasons.append('crew_ready_for_dispatch')
        full=bool(reasons)
        if not full:
            result['energy_history']=energy
            updated=refresh_continuous(c,state,result,*active_plan,compatible,realpv)
            if updated is None:full=True;reasons.append('continuous_or_domain_update_failed')
            else:active_plan=updated
        if full:
            state_snapshots.append(jsonable(copy.deepcopy(state)))
            fc=forecast(c,state);decision,selected,allrecords=decide(c,copy.deepcopy(state),fc,policy,progress=progress)
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
            decision['hour']=h;decision['execution_steps']=1;decision['trigger_reasons']=list(dict.fromkeys(reasons))
            decision['horizon_end_hour']=float(c['horizon_hours']);decision['state_step']=step
            rounds.append(jsonable(decision))
            if output:
                save_json(Path(output)/'rounds'/f'{step:03d}.json',decision)
                save_json(Path(output)/'action_evidence'/f'{step:03d}.json',m.metadata.get('action_evidence',{}))
            if capture_initial and initial_capture is None:
                from evidence import information_evidence
                m.metadata['information_evidence']=information_evidence(c,m,r['x'])
                initial_capture=(m,r,initial)
                if output:save_json(Path(output)/'initial_action_evidence.json',m.metadata.get('action_evidence',{}))
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
            'unit_start_hour':state['unit']['start_hour'],'trigger_reasons':list(dict.fromkeys(reasons)) if full else ['continuous_domain_refresh'],
            'domain_used':bool(m.metadata.get('domain_certificate',{}).get('used',False)),'domain':m.metadata.get('domain_certificate'),'backward_passed':m.metadata.get('backward_certificate',{}).get('passed',False)})
        old_active=copy.deepcopy(state['crew']['active']);trigger=advance_crew(c,state,world,events)
        pending_event_reason='inspection_information_arrived' if old_active and old_active['kind']=='inspect' else 'repair_completed'
        tube=m.metadata.get('backward_certificate')
        if tube is not None:
            if np.any(state['energy']<tube['lower'][s,:,t+1]-2e-5) or np.any(state['energy']>tube['upper'][s,:,t+1]+2e-5):
                raise RuntimeError('执行首步后的真实SOC不在下一阶段后向可达管内')
        if old_active is not None and not trigger:
            active=state['crew']['active']
            if active is None or (active['edge'],active['kind'])!=(old_active['edge'],old_active['kind']):raise RuntimeError('未完成作业被擅自取消或改派')
    result.pop('energy_history',None)
    result['controller_steps']=controller_steps;result['state_snapshots']=state_snapshots
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
    times=sorted(set([*range(0,n,stride),n-1]))
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
    if success and option(c,'analysis.splice_diagnostics',True):
        from evidence import splice_diagnostic
        splicing.update(splice_diagnostic(local,joined_p,joined_q,fc['probabilities'],c['dt'],n*c['dt']))
    result={'scope':'MGO局部全时域条件资源域：固定充放电模式，保留SOC/本地负荷/逆变器/同信息控制；网络可送范围另列',
            'horizon_hours':float(n*c['dt']),'n_steps':n,'scenario_count':ns,'slices':slices,'delivery_bounds':delivery,
            'reference_p':refp[0].tolist(),'reference_q':refq[0].tolist(),
            'reference_all_p':reference_all_p.tolist(),'reference_all_q':reference_all_q.tolist(),
            'tracking_p_max':rp,'tracking_q_max':rq,'time_hours':fc['hours'].tolist(),
            'decomposition_success':bool(success),'decomposition_residual':residual(local,cert['x']) if success else None,
            'online_domain':m.metadata.get('domain_certificate'),'decomposition':decomposition,'trajectory_inner_hull':hull_summary,'splicing_counterexample':splicing,'envelopes_used_in_dispatch':m.metadata['interface_envelopes'],
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
            fc=forecast(trial,st);point_cache={}
            r,chosen,_=decide(trial,st,fc,'joint',seconds=c['solver']['phase_seconds'],secondary_seconds=0,cache=point_cache,progress=progress)
            attempts=[{'classification':r['classification'],'D_lower':r['D_lower'],'D_upper':r['D_upper'],'primary_seconds':c['solver']['phase_seconds']}]
            for refinement in range(int(option(c,'analysis.phase_refinement_rounds',1))):
                if r['certified']:break
                refined=copy.deepcopy(trial);refined['solver']['rel_gap']=min(refined['solver']['rel_gap'],.001)
                refined['options']['solver.decision_bound_seconds']=float(option(c,'analysis.phase_extra_bound_seconds',0.))
                budget=float(option(c,'analysis.phase_refinement_seconds',20.))
                r,chosen,_=decide(refined,st,fc,'joint',seconds=budget,secondary_seconds=0,cache=point_cache,progress=progress)
                attempts.append({'classification':r['classification'],'D_lower':r['D_lower'],'D_upper':r['D_upper'],'primary_seconds':budget})
            row={k:r.get(k) for k in ('D_estimate','D_lower','D_upper','classification','certified','branches')};row.update(alpha=alpha,soc=soc,attempts=attempts);records.append(row)
            if progress:progress(f'[D相图] alpha={alpha:g}, SOC={soc:g}：{r["classification"]}',flush=True)
    out={'scope':'保存状态有限因果策略类的D_R；每格重优化，未判与不可行分开；不以固定整数LP冒充完整动作价值',
         'state_hour':float(state['step']*c['dt']),'alphas':c['settings']['phase_alphas'],'socs':c['settings']['phase_socs'],'records':records,'units':'恢复率之差（乘100为百分点）'}
    save_json(Path(output)/'phase_map.json',out);return out
