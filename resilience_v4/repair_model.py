"""动态抢修：优化当前已报告故障的作业起点和先后次序，观测后滚动更新。

未知损伤的前瞻工时用重伤上界；执行时先巡检，揭示损伤后再优化是否维修。
未来尚未报告的故障不提前分派队伍。主流程使用information.py生成有限因果
作业计划，相同观测历史行动一致，检查后可分叉；本文件提供旅行/工时计算及
备用日程构件。支持天气暂停并保留进度；一支队伍。
"""
import numpy as np

def travel_hours(c,origin,target):
    if isinstance(origin,int):origin=c['edge_xy'][origin]
    if isinstance(target,int):target=c['edge_xy'][target]
    distance=np.linalg.norm(np.asarray(origin)-np.asarray(target))*c['crew']['road_detour_factor']
    raw=distance/c['crew']['speed_kmh'];return np.ceil((raw-1e-10)/c['dt'])*c['dt']

def service_end(c,fc,line,start,work):
    """按安全风速累积有效工时，坏天气期间暂停，不凭空完成维修。"""
    hours=fc['hours'];dt=c['dt'];remaining=work
    for t,h in enumerate(hours):
        if h+1e-8<start:continue
        if fc['crew_wind'][line,t]<=c['crew']['safe_wind_ms']+1e-8:remaining-=dt
        if remaining<=1e-8:return float(h+dt)
    return np.inf

def crew_context(c,state,fc):
    now=state['step']*c['dt'];active=state['crew']['active'];origin=np.asarray(state['crew']['position']);ready=now;fixed={};under_inspection=None
    if active:
        e=active['edge'];origin=c['edge_xy'][e];start=now+active.get('travel_remaining',0)
        end=service_end(c,fc,e,start,active['work_remaining'])
        ready=end
        if active['kind']=='repair':fixed[e]=end
        else:under_inspection=e
    return origin,ready,fixed,under_inspection

def job_options(c,state,fc):
    origin,ready,fixed,inspecting=crew_context(c,state,fc);now=state['step']*c['dt'];jobs=[]
    start_grid=np.unique(np.r_[now,np.arange(np.ceil(now),c['horizon_hours'],c['crew']['decision_hours'])])
    for e in np.flatnonzero(state['damaged']):
        if e in fixed:continue
        line=c['edges'][e];known=state['observed_damage'].get(int(e));inspection=0 if known or e==inspecting else line['inspection_hours']
        work=line[f'repair_{known or "heavy"}_hours']+inspection
        earliest=ready+travel_hours(c,origin,int(e));options=[]
        for start in start_grid:
            if start+1e-8<earliest:continue
            end=service_end(c,fc,int(e),start,work)
            if np.isfinite(end):options.append((float(start),float(end)))
        jobs.append({'edge':int(e),'known':known,'options':options,'work':work})
    return jobs,{'origin':origin,'ready_hour':ready,'fixed_completion':fixed,'inspecting':inspecting}

def baseline_schedule(c,state,fc,jobs,context,policy):
    """固定规则只约束抢修，功率和开关仍由同一物理优化模型计算。"""
    origin=context['origin'];clock=context['ready_hour'];pending=list(range(len(jobs)));chosen={}
    while pending and np.isfinite(clock):
        if policy=='nearest':j=min(pending,key=lambda j:travel_hours(c,origin,jobs[j]['edge']))
        else:j=max(pending,key=lambda j:c['repair_priority'][jobs[j]['edge']])
        e=jobs[j]['edge'];earliest=clock+travel_hours(c,origin,e)
        choices=[(k,v) for k,v in enumerate(jobs[j]['options']) if v[0]+1e-8>=earliest]
        if choices:
            k,(start,end)=choices[0];chosen[j]=k;clock=end;origin=e
        pending.remove(j)
    return chosen
