"""有限因果策略类：检查决定信息到达时间，作业状态单独继承。

不把真实隐蔽损伤传给调度器。天气路径与待检查类型作笛卡尔积，避免
一条天气路径的编号意外泄露全部损伤。策略只能读取已报告故障和已完成
检查的结果；已开始的行程/检查/维修则由active状态锁定。
"""
from itertools import product
import copy
import numpy as np
from repair_model import travel_hours


def option(c,name,default):
    return c.get('options',{}).get(name,default)


def estimated_reveal_hour(c,state,base,edge):
    """只用当前任务和预测风速估计单个候选的最早检查完成，不读取world。"""
    dt=c['dt'];hours=base['hours'];wind=base['crew_wind'];position=state['crew']['position'].copy()
    active=copy.deepcopy(state['crew']['active']);start=0
    def finish(job,begin):
        for t in range(begin,len(hours)):
            left=dt;e=job['edge'];trip=min(left,job['travel_remaining']) if wind[e,t]<=float(option(c,'crew.travel_wind_limit_ms',25.)) else 0.
            job['travel_remaining']-=trip;left-=trip
            if job['travel_remaining']<=1e-8 and wind[e,t]<=c['crew']['safe_wind_ms']+1e-8:job['work_remaining']-=left
            if job['work_remaining']<=1e-8:return t+1
        return None
    if active is not None:
        done=finish(active,0)
        if done is None:return np.inf
        if active['edge']==edge and active['kind']=='inspect':return float(hours[done-1]+dt)
        position=c['edge_xy'][active['edge']].copy();start=done
    job={'edge':edge,'travel_remaining':travel_hours(c,position,edge),'work_remaining':c['edges'][edge]['inspection_hours']}
    done=finish(job,start)
    return np.inf if done is None else float(hours[done-1]+dt)


def expand_hidden_types(c,state,base):
    """对当前最重要的K个未知故障显式展开2^K类；其余按重伤上界。

K是计算精度参数，不等同于所有线路损伤已被随机展开。记录target供核查。
"""
    unknown=[int(e) for e in np.flatnonzero(state['damaged']) if int(e) not in state['observed_damage']]
    selection=[];now=state['step']*c['dt'];method=str(option(c,'information.target_selection','timely'));pilot_times={}
    if method=='timely' and unknown:
        # 先用同一组因果规则作保守试排程，纳入其他任务及天气等待；
        # 不把“直奔某线的理论最早时刻”误当实际策略会及时揭示。
        pilot=dict(base);pilot['severity']=[['heavy']*len(c['edges']) for _ in base['probabilities']]
        for row in pilot['severity']:
            for edge,value in state['observed_damage'].items():row[int(edge)]=value
        for rule in policy_names(c,'joint'):
            plan=make_crew_plan(c,state,pilot,rule)
            for edge in unknown:
                times=[next((q['hour'] for q in reveals if q['edge']==edge),np.inf) for reveals in plan['reveals']]
                pilot_times.setdefault(edge,{})[rule]=max(times)

    for e in unknown:
        reveal=min(pilot_times.get(e,{}).values(),default=np.inf) if method=='timely' else estimated_reveal_hour(c,state,base,e);weight=c['repair_priority'][e]
        timely=now<c['generator']['last_start_hour'] and reveal<=c['generator']['last_start_hour']+1e-8
        bonus=2. if timely else .25 if now<c['generator']['last_start_hour'] else 1.
        score=weight if method=='priority' else (bonus*weight/max(c['dt'],reveal-now) if np.isfinite(reveal) else 0.)
        selection.append({'edge':e,'line':c['edges'][e]['name'],'estimated_reveal_hour':reveal,'before_start_deadline':bool(timely),'score':float(score),'pilot_reveal_hours':pilot_times.get(e,{})})
    ranking={r['edge']:r['score'] for r in selection}
    unknown.sort(key=lambda e:(-ranking[e],e))
    k=min(len(unknown),int(option(c,'information.max_hidden_targets',1)))
    targets=unknown[:k];patterns=list(product(('light','heavy'),repeat=k)) or [()]
    p=float(option(c,'information.heavy_prior',.5))
    if not 0<p<1:raise ValueError('隐蔽损伤先验须在(0,1)内')
    ids=[];prob=[];severity=[]
    for w,weight in enumerate(base['probabilities']):
        for pattern in patterns:
            row=['heavy']*len(c['edges'])
            for e,value in state['observed_damage'].items():row[int(e)]=value
            for e,value in zip(targets,pattern):row[e]=value
            ids.append(w);severity.append(row)
            prob.append(float(weight)*np.prod([p if value=='heavy' else 1-p for value in pattern]))
    fc={k:copy.deepcopy(v) for k,v in base.items()}
    for key in ('wind','pv','failure_hour'):fc[key]=np.asarray(base[key])[ids].copy()
    fc.update(probabilities=np.asarray(prob)/sum(prob),severity=severity,weather_path=np.asarray(ids),hidden_targets=targets,
              target_selection=selection,initial_reports=dict(state['failure_observed_hour']),initial_diagnoses=dict(state['observed_damage']))
    return fc


def history_key(fc,s,t):
    if 'crew_plan' in fc:return fc['crew_plan']['histories'][s][t]
    h=fc['hours'][t]
    return (tuple((int(e),float(fc['failure_hour'][s,e])) for e in np.flatnonzero(fc['failure_hour'][s]<=h+1e-8)),())


def policy_names(c,policy):
    if policy!='joint':return ['nearest' if policy=='nearest' else 'critical']
    values=[x.strip() for x in str(option(c,'information.crew_policies','critical,information')).split(',')]
    if not values or not set(values)<={'critical','nearest','information'}:raise ValueError('作业策略为critical,nearest,information')
    return list(dict.fromkeys(values))[:int(option(c,'information.max_policy_candidates',2))]


def make_crew_plan(c,state,fc,name):
    """在每个预测样本推进同一个因果派工规则，不预读未检查的类型。

未来故障只能在其报告后入队。天气服务采用各天气路径的共同安全上界，
保持当前策略的信息结构保守一致；真实更早完工会触发实际重算。
"""
    dt=c['dt'];ns=len(fc['probabilities']);n=len(fc['hours']);ne=len(c['edges'])
    availability=np.ones((ns,ne,n),bool);all_hist=[];all_dispatch=[];all_tasks=[];all_trace=[];cost=[];all_reveals=[]
    events=set(float(h) for h in fc['failure_hour'].ravel() if np.isfinite(h) and h<c['horizon_hours'])
    for s in range(ns):
        damaged=state['damaged'].copy();ever=state['ever_failed'].copy();known=dict(state['observed_damage']);reports=dict(state['failure_observed_hour'])
        active=copy.deepcopy(state['crew']['active']);position=state['crew']['position'].copy()
        histories=[];dispatch=[];tasks=[];trace=[];reveals=[];active_hours=0.
        if active:
            active['task_index']=0
            tasks.append({'edge':active['edge'],'kind':active['kind'],'departure':float(fc['hours'][0]),'end':None,'locked':True})
        for t,h in enumerate(fc['hours']):
            for e in np.flatnonzero((fc['failure_hour'][s]<=h+1e-8)&~ever):
                damaged[e]=True;ever[e]=True;reports[int(e)]=float(fc['failure_hour'][s,e])
            histories.append((tuple(sorted((int(e),float(v)) for e,v in reports.items())),tuple(sorted((int(e),v) for e,v in known.items()))))
            command=None
            if active is None:
                # 未实际出发/施工的计划不得变成不可撤销承诺。
                pending=[]
                for edge in np.flatnonzero(damaged):
                    e=int(edge);travel=travel_hours(c,position,e)
                    limit=float(option(c,'crew.travel_wind_limit_ms',25.)) if travel>1e-8 else c['crew']['safe_wind_ms']
                    if fc['crew_wind'][e,t]<=limit+1e-8:pending.append(e)
                def score(e):
                    travel=travel_hours(c,position,e);line=c['edges'][e];weight=c['repair_priority'][e]
                    if name=='nearest':return (-travel,weight,-e)
                    if name=='critical':return (weight,-travel,-e)
                    if e in known:duration=line[f'repair_{known[e]}_hours'];bonus=1.
                    else:
                        p=float(option(c,'information.heavy_prior',.5))
                        duration=line['inspection_hours']+(1-p)*line['repair_light_hours']+p*line['repair_heavy_hours']
                        bonus=1.+float(option(c,'information.priority_bonus',.5))*(-p*np.log2(p)-(1-p)*np.log2(1-p))
                    return (bonus*weight/max(dt,travel+duration),-travel,-e)
                if pending:
                    e=max(pending,key=score);kind='repair' if e in known else 'inspect'
                    work=c['edges'][e][f'repair_{known[e]}_hours'] if kind=='repair' else c['edges'][e]['inspection_hours']
                    active={'edge':e,'kind':kind,'travel_remaining':float(travel_hours(c,position,e)),
                            'work_remaining':float(work),'task_index':len(tasks)}
                    tasks.append({'edge':e,'kind':kind,'departure':float(h),'end':None,'locked':False})
                    command=(e,kind);events.add(float(h))
            dispatch.append(command);availability[s,:,t]=~damaged
            trace.append(None if active is None else (active['edge'],active['kind'],active['travel_remaining'],active['work_remaining']))
            if active:
                active_hours+=dt;e=active['edge'];remaining=dt
                travel=min(remaining,active['travel_remaining']) if fc['crew_wind'][e,t]<=float(option(c,'crew.travel_wind_limit_ms',25.)) else 0.
                active['travel_remaining']-=travel;remaining-=travel
                if active['travel_remaining']<=1e-8:
                    position=c['edge_xy'][e].copy()
                    if fc['crew_wind'][e,t]<=c['crew']['safe_wind_ms']+1e-8:active['work_remaining']-=remaining
                if active['work_remaining']<=1e-8:
                    end=float(h+dt);tasks[active['task_index']]['end']=end;events.add(end)
                    if active['kind']=='inspect':
                        # 唯一读取预测隐藏类型的位置：实体检查完成之后。
                        known[e]=fc['severity'][s][e];reveals.append({'edge':e,'hour':end,'severity':known[e]})
                    else:damaged[e]=False
                    active=None
        all_hist.append(histories);all_dispatch.append(dispatch);all_tasks.append(tasks);all_trace.append(trace)
        all_reveals.append(reveals);cost.append(active_hours*c['crew']['hourly_cost'])
    plan={'name':name,'availability':availability,'histories':all_hist,'dispatch':all_dispatch,'tasks':all_tasks,
          'trace':all_trace,'reveals':all_reveals,'cost_by_scenario':cost,'event_hours':sorted(events)}
    # 因果性与任务锁定分开检查；既不以计划名称代替检查，也不读取真实world。
    validate_policy(plan,fc)
    return plan


def validate_policy(plan,fc):
    for t in range(len(fc['hours'])):
        groups={}
        for s in range(len(fc['probabilities'])):groups.setdefault(plan['histories'][s][t],[]).append(s)
        for group in groups.values():
            actions={plan['dispatch'][s][t] for s in group}
            if len(actions)>1:raise RuntimeError('作业策略在相同观测历史下分派了不同作业')
    for tasks in plan['tasks']:
        for previous,current in zip(tasks,tasks[1:]):
            if previous['end'] is None or current['departure']<previous['end']-1e-8:raise RuntimeError('未完成作业被撤销或发生队伍重叠')
    return {'information_nonanticipativity':True,'task_commitment':True}
