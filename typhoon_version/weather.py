"""台风输入模块（不作为第四个创新点）。

移动Rankine型简化涡旋风场→线路沿线三点最大风速→风速依赖危险率。
用指数阈值与累计危险率生成每条线路首次故障，常开联络线同样暴露。
本版每条线每次台风最多损坏一次；灾前历史用于产生t=0已知故障。
故障/实际损伤保存在world中，优化器只接受当前state和独立预测采样。
"""
import numpy as np

def wind_field(c,hours,multiplier=1.):
    hours=np.asarray(hours,float);tr=c['track'];tt=np.array([r['time_hour'] for r in tr])
    cx=np.interp(hours,tt,[r['center_x_km'] for r in tr]);cy=np.interp(hours,tt,[r['center_y_km'] for r in tr])
    peak=np.interp(hours,tt,[r['max_wind_ms'] for r in tr])*multiplier
    radius=np.interp(hours,tt,[r['radius_max_km'] for r in tr]);ne=len(c['edges']);out=np.zeros((ne,len(hours)))
    for e,line in enumerate(c['edges']):
        p=c['geo_xy'][line['from']];q=c['geo_xy'][line['to']]
        vv=[]
        for f in (0.,.5,1.):
            point=p*(1-f)+q*f;r=np.hypot(point[0]-cx,point[1]-cy);rr=np.maximum(r/np.maximum(radius,.1),1e-6)
            shape=np.where(rr<=1,rr,rr**(-c['weather']['outer_exponent']))
            vv.append(peak*shape*c['weather']['surface_factor'])
        out[e]=np.max(vv,axis=0)
    return out

def hazard(c,wind):
    """lambda单位1/h。参数是可替换合成值，不等同于已标定脆弱性曲线。"""
    out=np.zeros_like(wind)
    for e,line in enumerate(c['edges']):
        if line['exposed']:
            r=np.maximum(wind[e]/line['wind_threshold_ms']-1,0)
            out[e]=line['hazard_scale']*r**line['hazard_exponent']
    return out

def make_world(c):
    """生成真实但对控制器隐藏的完整风灾样本。一次采样保持时间一致性。"""
    dt=c['dt'];times=np.arange(-int(round(c['weather']['history_hours']/dt)),c['n_steps'])*dt
    wind=wind_field(c,times);rates=hazard(c,wind);rng=np.random.default_rng(int(c['weather']['seed']))
    threshold=rng.exponential(1,size=len(c['edges']));cumulative=np.cumsum(rates*dt,axis=1)
    failure=np.full(len(c['edges']),np.inf);severity=[]
    interval=c['weather']['fault_interval_hours']
    for e,line in enumerate(c['edges']):
        hit=np.flatnonzero(cumulative[e]>=threshold[e])
        if len(hit):
            raw=times[hit[0]]+dt;failure[e]=np.ceil((raw-1e-9)/interval)*interval
        peak=float(wind[e].max());prob=np.clip(.25+.4*(peak-25)/25,.15,.8)
        severity.append('heavy' if rng.random()<prob else 'light')
    return {'failure_hour':failure,'severity':severity,'wind':wind_field(c,np.arange(c['n_steps'])*dt),
            'hazard':hazard(c,wind_field(c,np.arange(c['n_steps'])*dt)),
            'threshold':threshold,'note':'真实潜变量只供环境推进和事后统计，不传给优化模型'}

def forecast(c,state,count=None):
    """给定已观测历史独立生成可能未来；不读取world中的实际未来。"""
    now=state['step'];n=c['n_steps']-now;hours=np.arange(now,c['n_steps'])*c['dt'];ne=len(c['edges'])
    count=int(count or c['settings']['forecast_scenarios']);rng=np.random.default_rng(int(c['weather']['seed'])+100003+now*997)
    failures=np.full((count,ne),np.inf);winds=[]
    # 共同随机数以时刻和线路寻址，策略或alpha/SOC改变不会另抽实际灾害。
    for s in range(count):
        multiplier=1 if count==1 else np.linspace(.9,1.1,count)[s]
        v=wind_field(c,hours,multiplier);winds.append(v)
        cum=np.cumsum(hazard(c,v)*c['dt'],axis=1);th=rng.exponential(1,size=ne)
        for e in range(ne):
            if state['ever_failed'][e] or not c['edges'][e]['exposed']:continue
            hit=np.flatnonzero(cum[e]>=th[e])
            if len(hit):
                raw=hours[hit[0]]+c['dt'];failures[s,e]=np.ceil((raw-1e-9)/c['weather']['fault_interval_hours'])*c['weather']['fault_interval_hours']
    wind=np.asarray(winds);pv=[]
    for s in range(count):
        local=wind[s,c['mg_edge_indices']]
        factor=np.clip(1-c['weather']['pv_wind_derating']*local/50,.2,1)
        pv.append(c['profiles']['pv'][:,now:]*factor)
    return {'hours':hours,'wind':wind,'pv':np.asarray(pv),'failure_hour':failures,
            'probabilities':np.full(count,1/count),'crew_wind':wind.max(axis=0)}
