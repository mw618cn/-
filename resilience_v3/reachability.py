"""显式后向递推：本地安全后备集，以及给定接口指令的条件可达管。

B_N=[E_terminal,E_max]；前驱要求“存在”可行控制增量。
B_t=X∩[lower(B_{t+1})−ΔE_max, upper(B_{t+1})−ΔE_min]。
这是B与负增量集合的Minkowski和，不是要求任意增量均成立的Pontryagin差。
当前观测组下一时刻可能分支时，先交各子分支的可延续集合再求前驱。
这不是把全系统混合整数可行域精确消元，也不声称覆盖样本外扰动。
"""
from functools import lru_cache
import numpy as np
from information import history_key,option


@lru_cache(maxsize=50000)
def _increment(pvmax,dgmax,chmax,dismax,eta_c,eta_d,inv,p,q,dt,sides,mode):
    """四变量单步LP的等价解析消元；状态仅在后向集合中约束。

    令y=PV−charge+discharge。给定q后，多边形给y一个区间；
    DG=p−y∈[0,DGmax]再给区间。固定模式下可直接消去PV/DG，
    得充/放电端点，避免后向每步重复调用小LP。独立测试与原LP对照。
    """
    angles=(np.arange(sides)+.5)*2*np.pi/sides;inner=np.cos(np.pi/sides)
    cos=np.cos(angles);rhs=inv*inner-np.sin(angles)*q;tol=1e-10
    if np.any(rhs[np.abs(cos)<=1e-12]<-tol):return np.inf,-np.inf
    ymin=float(np.max(rhs[cos<-1e-12]/cos[cos<-1e-12],initial=-np.inf))
    ymax=float(np.min(rhs[cos>1e-12]/cos[cos>1e-12],initial=np.inf))
    ymin=max(ymin,p-dgmax);ymax=min(ymax,p)
    if mode not in (-1,0,1):raise ValueError('充放电模式须为-1/0/1')
    limits=[]
    for selected in ((0,1) if mode==-1 else (mode,)):
        low=max(ymin,-chmax if selected else 0.)
        high=min(ymax,pvmax if selected else pvmax+dismax)
        if low>high+tol:continue
        if low>high:low=high=(low+high)/2
        if selected:
            cmin=max(0.,-high);cmax=min(chmax,pvmax-low)
            if cmin<=cmax+tol:limits.append((dt*eta_c*max(0.,cmin),dt*eta_c*max(cmin,cmax,0.)))
        else:
            dmin=max(0.,low-pvmax);dmax=min(dismax,high)
            if dmin<=dmax+tol:limits.append((-dt*max(dmin,dmax,0.)/eta_d,-dt*max(0.,dmin)/eta_d))
    if not limits:return np.inf,-np.inf
    if len(limits)==2 and max(z[0] for z in limits)>min(z[1] for z in limits)+1e-7:
        raise RuntimeError('单步充放电能量集合不连通，不能用区间代替其并集')
    return min(z[0] for z in limits),max(z[1] for z in limits)


def delta_interval(c,mg,pv,p,q,mode=-1):
    return _increment(float(max(0,pv)),float(mg['dg_max_mw']),float(mg['charge_mw']),float(mg['discharge_mw']),
                      float(mg['eta_charge']),float(mg['eta_discharge']),float(mg['inverter_mva']),float(p),float(q),
                      float(c['dt']),int(c['polygon_sides']),int(mode))


def island_reserve_tube(c,state,fc):
    """单微网岛内关键服务可延续至终端的安全后备集，使用共同PV下界。

采用固定终点；可在算法设置关闭此额外保守约束。不是主配启动域。
"""
    nm=len(c['microgrids']);n=len(fc['hours']);lower=np.empty((nm,n+1));upper=np.empty_like(lower)
    pv=np.min(fc['pv'],axis=0)
    for j,mg in enumerate(c['microgrids']):
        emin=mg['energy_mwh']*mg['soc_min'];emax=mg['energy_mwh']*mg['soc_max']
        lower[j,n]=mg['energy_mwh']*mg['soc_terminal'];upper[j,n]=emax;node=c['mg_nodes'][j]
        demand=c['profiles']['critical'][node,state['step']:];ratio=c['load']['reactive_ratio'][node]
        for t in range(n-1,-1,-1):
            dlo,dhi=delta_interval(c,mg,pv[j,t],demand[t],demand[t]*ratio)
            lower[j,t]=max(emin,lower[j,t+1]-dhi);upper[j,t]=min(emax,upper[j,t+1]-dlo)
    return {'lower':lower,'upper':upper,'scope':'单微网岛内关键服务及终端储备；共同PV下界，无外部输入'}


def certify_commands(m,x,tolerance=2e-5,start_index=0,active_scenarios=None):
    """对已选动作/网络/任务方案给出的全PCC轨迹做独立后向可达校核。"""
    from power_model import extract
    md=m.metadata;c=md['cfg'];fc=md['forecast'];ns=len(fc['probabilities']);n=len(fc['hours']);nm=len(c['microgrids'])
    data=[extract(m,x,s) for s in range(ns)]
    active=list(range(ns)) if active_scenarios is None else list(active_scenarios)
    lower=np.full((ns,nm,n+1),np.nan);upper=np.full_like(lower,np.nan)
    reserve=md.get('reserve_tube')
    for j,mg in enumerate(c['microgrids']):
        lower[active,j,n]=mg['energy_mwh']*mg['soc_terminal'];upper[active,j,n]=mg['energy_mwh']*mg['soc_max']
    for t in range(n-1,start_index-1,-1):
        groups={}
        for s in active:groups.setdefault(history_key(fc,s,t),[]).append(s)
        for group in groups.values():
            lead=group[0];d=data[lead]
            for j,mg in enumerate(c['microgrids']):
                node=c['mg_nodes'][j];loads=d['served_c'][node,t]+d['served_g'][node,t]
                p=d['pcc_p'][j,t]+loads;q=d['pcc_q'][j,t]+loads*c['load']['reactive_ratio'][node]
                mode=round(d['battery_mode'][j,md['mode_map'][t]])
                for s in group[1:]:
                    other=data[s];other_load=other['served_c'][node,t]+other['served_g'][node,t]
                    if max(abs(other['pcc_p'][j,t]+other_load-p),abs(other['pcc_q'][j,t]+other_load*c['load']['reactive_ratio'][node]-q))>tolerance:
                        raise RuntimeError('相同信息组的接口或本地指令不同，不能出具后向可达证书')
                dlo,dhi=delta_interval(c,mg,float(np.min(fc['pv'][group,j,t])),p,q,mode)
                # 当前控制在同一历史下共用；下一时刻的任一可观测子分支均须可续。
                next_lo=float(np.max(lower[group,j,t+1]));next_hi=float(np.min(upper[group,j,t+1]))
                lo=max(mg['energy_mwh']*mg['soc_min'],next_lo-dhi)
                hi=min(mg['energy_mwh']*mg['soc_max'],next_hi-dlo)
                if reserve is not None:
                    lo=max(lo,float(reserve['lower'][j,t]));hi=min(hi,float(reserve['upper'][j,t]))
                if next_lo>next_hi+tolerance:lo,hi=np.inf,-np.inf
                lower[group,j,t]=lo;upper[group,j,t]=hi
    energies=np.array([d['energy'] for d in data]);violation=max(float(np.max((lower-energies)[active,:,start_index:])),float(np.max((energies-upper)[active,:,start_index:])),0.)
    passed=bool(np.isfinite(violation) and violation<=tolerance)
    takeover=None
    for t in range(start_index,n):
        h=md['hours'][t]
        if all(np.all(data[s]['online'][t:]>.5) and np.all(data[s]['upward_p'][t:]<=tolerance) for s in active):
            takeover=float(h);break
    return {'passed':passed,'max_tube_violation':violation,'lower':lower,'upper':upper,
            'initial_lower':lower[active[0],:,start_index],'initial_upper':upper[active[0],:,start_index],
            'start_index':start_index,'active_scenarios':active,
            'takeover_hour':takeover,
            'takeover_definition':'各预测分支机组在线，且从该时刻至24h均不再需要主配接口向上支撑；受保护服务由全模型保持',
            'scope':'给定候选PCC/本地服务/充放电模式与因果信息树的局部后向可达管；不等同全系统精确动作可达集'}
