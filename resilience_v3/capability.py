"""微网接口的多时段能量包络及条件资源域。

包络在每次建模前由真实SOC、剩余时域、预测PV和本地负荷更新，并进入
全局模型。包络是必要条件；全局模型仍保留微网内部变量以保证可分解。
出图另用微网局部LP，不把主配通道断开误画成微网资源能力为零。
"""
from collections import defaultdict
from information import history_key,option
from reachability import island_reserve_tube
import numpy as np


def interface_envelopes(c,state,fc):
    """累计净输出电量界，单位MWh；非逐时P/Q截面的笛卡尔积。"""
    n=len(fc['hours']);dt=c['dt'];rows=[]
    # 约束当前一步及1/4/8/16小时、剩余全时域；终点仍是原24h。
    stops=sorted({1,n,*[min(n,max(1,round(h/dt))) for h in (1,4,8,16)]})
    for s in range(len(fc['probabilities'])):
        for j,mg in enumerate(c['microgrids']):
            node=c['mg_nodes'][j];dc=c['profiles']['critical'][node,state['step']:]
            dg=c['profiles']['general'][node,state['step']:]
            protected=dc if node in c['protected_nodes'] else np.zeros(n)
            initial=float(state['energy'][j]);maximum=mg['energy_mwh']*mg['soc_max']
            for stop in stops:
                reserve=mg['soc_terminal'] if stop==n else mg['soc_min']
                minimum=mg['energy_mwh']*reserve
                # sum(dis-charge) <= eta_dis*(E0-Eend)，充电损耗只会收紧上界。
                upper=dt*(fc['pv'][s,j,:stop].sum()+stop*mg['dg_max_mw']-protected[:stop].sum())+mg['eta_discharge']*(initial-minimum)
                lower=-dt*(dc[:stop]+dg[:stop]).sum()-(maximum-initial)/mg['eta_charge']
                rows.append({'scenario':s,'microgrid':j,'stop':stop,'lower_mwh':float(lower),'upper_mwh':float(upper)})
    return rows


def build_resource_model(c,state,fc,modes):
    """完整剩余时域、全部预测样本的微网资源LP；给定充放电模式。

只移除DSO/TSO网络，保留本地负荷、逆变器、SOC、终端储备和同信息控制。
这是固定模式下的条件域，不是所有模式的完整非凸能力域。
"""
    from power_model import ModelBuilder
    reserve=island_reserve_tube(c,state,fc) if option(c,'reachability.enforce_island_reserve',True) else None
    b=ModelBuilder();v={};ns=len(fc['probabilities']);n=len(fc['hours']);dt=c['dt'];nm=len(c['microgrids'])
    for key in ('pv','dg','charge','discharge','served_c','served_g'):
        v[key]=b.add_var(key,(ns,nm,n))
    for key in ('pcc_p','pcc_q','mg_q'):v[key]=b.add_var(key,(ns,nm,n),lb=-np.inf)
    v['energy']=b.add_var('energy',(ns,nm,n+1))
    angles=(np.arange(int(c['polygon_sides']))+.5)*2*np.pi/c['polygon_sides'];inner=np.cos(np.pi/c['polygon_sides'])
    for s in range(ns):
        for j,mg in enumerate(c['microgrids']):
            node=c['mg_nodes'][j];dc=c['profiles']['critical'][node,state['step']:];dd=c['profiles']['general'][node,state['step']:]
            ratio=c['load']['reactive_ratio'][node]
            b.add_row({v['energy'][s,j,0]:1},state['energy'][j],state['energy'][j],'inherited_soc')
            for t in range(n+1):
                col=v['energy'][s,j,t];b.lb[col]=mg['energy_mwh']*(mg['soc_terminal'] if t==n else mg['soc_min']);b.ub[col]=mg['energy_mwh']*mg['soc_max']
                if reserve is not None:
                    b.lb[col]=max(b.lb[col],float(reserve['lower'][j,t]));b.ub[col]=min(b.ub[col],float(reserve['upper'][j,t]))
            for t in range(n):
                pv,dg,ch,dis,sc,sg=[v[k][s,j,t] for k in ('pv','dg','charge','discharge','served_c','served_g')]
                for col,cap in ((pv,fc['pv'][s,j,t]),(dg,mg['dg_max_mw']),(ch,mg['charge_mw']*modes[s,j,t]),(dis,mg['discharge_mw']*(1-modes[s,j,t])),(sc,dc[t]),(sg,dd[t])):b.ub[col]=float(cap)
                if node in c['protected_nodes']:b.lb[sc]=dc[t]
                for col in (pv,dg,ch,dis):b.c[col]=fc['probabilities'][s]*dt*(mg['dg_cost'] if col==dg else mg['throughput_cost'] if col in (ch,dis) else 0.)
                b.add_row({v['energy'][s,j,t+1]:1,v['energy'][s,j,t]:-1,ch:-dt*mg['eta_charge'],dis:dt/mg['eta_discharge']},0,0,'soc_balance')
                b.add_row({v['pcc_p'][s,j,t]:1,pv:-1,dg:-1,dis:-1,ch:1,sc:1,sg:1},0,0,'local_PCC_P')
                b.add_row({v['pcc_q'][s,j,t]:1,v['mg_q'][s,j,t]:-1,sc:ratio,sg:ratio},0,0,'local_PCC_Q')
                for angle in angles:
                    co,si=float(np.cos(angle)),float(np.sin(angle))
                    b.add_row({pv:co,dis:co,ch:-co,v['mg_q'][s,j,t]:si},ub=mg['inverter_mva']*inner,name='inverter_rating')
    for t,h in enumerate(fc['hours']):
        groups=defaultdict(list)
        for s in range(ns):groups[history_key(fc,s,t)].append(s)
        for group in groups.values():
            for s in group[1:]:
                for key in ('pv','dg','charge','discharge','served_c','served_g','mg_q'):
                    for j in range(nm):b.add_row({v[key][s,j,t]:1,v[key][group[0],j,t]:-1},0,0,'conditional_NA')
    m=b.build();m.metadata.update(cfg=c,state=state,forecast=fc,hours=fc['hours'])
    return m
