"""台风输入模块（不作为第四个创新点）。

移动Rankine型简化涡旋风场→线路沿线三点最大风速→风速依赖危险率。
用指数阈值与累计危险率生成每条线路首次故障，常开联络线同样暴露。
本版每条线每次台风最多损坏一次；灾前历史用于产生t=0已知故障。
故障/实际损伤保存在world中，优化器只接受当前state和独立预测采样。
"""
import numpy as np
from scipy.special import ndtr
from information import option

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
    """离散时段末发生首次故障；存活概率与每时段概率分别保存。"""
    dt=c['dt'];history=int(round(c['weather']['history_hours']/dt))
    times=np.arange(-history,c['n_steps'])*dt;wind=wind_field(c,times)
    rng=np.random.default_rng(int(c['weather']['seed']));ne=len(c['edges'])
    model=str(option(c,'weather.probability_model','hazard'))
    if model=='hazard':
        rates=hazard(c,wind);cumulative=np.cumsum(rates*dt,axis=1);cdf=-np.expm1(-cumulative)
        threshold=rng.exponential(1,size=ne);hitmask=cumulative>=threshold[:,None]
    elif model=='peak_fragility':
        median=float(option(c,'weather.fragility_median_ms',35.));beta=float(option(c,'weather.fragility_beta',.25))
        if median<=0 or beta<=0:raise ValueError('对数正态脆弱性参数须正')
        peak=np.maximum.accumulate(wind,axis=1);cdf=ndtr((np.log(np.maximum(peak,1e-12))-np.log(median))/beta)
        cdf*=np.array([e['exposed'] for e in c['edges']])[:,None]
        threshold=rng.uniform(size=ne);hitmask=cdf>=threshold[:,None]
        before=np.c_[np.zeros(ne),cdf[:,:-1]];risk=np.clip((cdf-before)/np.maximum(1-before,1e-12),0,1-1e-12)
        rates=-np.log1p(-risk)/dt
    else:raise ValueError('概率模型为hazard或peak_fragility')
    failure=np.full(ne,np.inf);severity=[];interval=c['weather']['fault_interval_hours']
    for e,line in enumerate(c['edges']):
        hit=np.flatnonzero(hitmask[e])
        if len(hit):failure[e]=np.ceil((times[hit[0]]+dt-1e-9)/interval)*interval
        # 损伤类型使用外置先验；不使用未来台风峰值生成当前类型。
        severity.append('heavy' if rng.random()<float(option(c,'information.heavy_prior',.5)) else 'light')
    future_rates=rates[:,history:];p=-np.expm1(-future_rates*dt)
    survival_before=1-np.c_[np.zeros(ne),cdf[:,:-1]]
    return {'failure_hour':failure,'severity':severity,'wind':wind[:,history:],'hazard':future_rates,
            'interval_probability_if_intact':p,'cumulative_first_failure_probability':cdf[:,history:],
            'first_failure_mass':(survival_before*(-np.expm1(-rates*dt)))[:,history:],
            'threshold':threshold,'probability_model':model,'probability_time_hours':np.arange(1,c['n_steps']+1)*dt,
            'note':'合成参数；首次故障模型。累计概率含灾前历史，区间概率条件于尚未首损；真实类型仅环境持有'}


def forecast(c,state,count=None):
    """给定已观测历史独立生成可能未来；不读取world中的实际未来。"""
    now=state['step'];n=c['n_steps']-now;hours=np.arange(now,c['n_steps'])*c['dt'];ne=len(c['edges'])
    count=int(count or c['settings']['forecast_scenarios']);rng=np.random.default_rng(int(c['weather']['seed'])+100003+now*997)
    failures=np.full((count,ne),np.inf);winds=[]
    # 共同随机数以时刻和线路寻址，策略或alpha/SOC改变不会另抽实际灾害。
    for s in range(count):
        multiplier=1 if count==1 else np.linspace(.9,1.1,count)[s]
        v=wind_field(c,hours,multiplier);v[:,0]=state.get('observed_wind',wind_field(c,hours[:1])[:,0]);winds.append(v)
        if option(c,'weather.probability_model','hazard')=='hazard':
            cum=np.cumsum(hazard(c,v)*c['dt'],axis=1);th=rng.exponential(1,size=ne)
        else:
            past=np.arange(-c['weather']['history_hours'],hours[0]+c['dt']/2,c['dt'])
            peak0=wind_field(c,past).max(axis=1);peak=np.maximum.accumulate(np.c_[peak0,v],axis=1)[:,1:]
            med=float(option(c,'weather.fragility_median_ms',35.));beta=float(option(c,'weather.fragility_beta',.25))
            initial=ndtr((np.log(np.maximum(peak0,1e-12))-np.log(med))/beta)
            cum=ndtr((np.log(np.maximum(peak,1e-12))-np.log(med))/beta);th=initial+(1-initial)*rng.uniform(size=ne)
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
