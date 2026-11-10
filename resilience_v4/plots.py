"""R4证据图：实际执行、预测树、条件域和最优值界各自标识。

所有图片仍强制原版宋体。绘图只消费已保存证据，不补造临界面、违规时刻
或其他策略曲线；图注单独输出，保持固定单/双栏宽度。
"""
from pathlib import Path
import json
import numpy as np
from plot_style import setup,figure,save,panel,timeaxis,stairs,COLORS,series
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import ConvexHull,QhullError


def _line(c,e):return c['edges'][e]['name'].replace('_','-')


def _network(ax,c,closed=None,damaged=None,live=None,fraction=None,show_switches=False):
    xy=c['plot_xy'];closed=np.array([e['initial_closed'] for e in c['edges']]) if closed is None else np.asarray(closed)
    damaged=np.zeros(len(c['edges'])) if damaged is None else np.asarray(damaged);live=np.ones(len(c['nodes'])) if live is None else np.asarray(live)
    for e,line in enumerate(c['edges']):
        a,b=xy[line['from']],xy[line['to']];route=np.asarray(line.get('plot_route',[]),float).reshape(-1,2);path=np.vstack([a,route,b]);on=closed[e]>.5;bad=damaged[e]>.5
        color='#B2182B' if bad else '.15' if on else '.65'
        ax.plot(path[:,0],path[:,1],color=color,ls='-' if on and not bad else '--',lw=1.15 if bad else .7,zorder=1)
        mid=path[len(path)//2] if len(path)>2 else (a+b)/2
        if bad:ax.scatter(*mid,marker='x',s=31,color='#B2182B',linewidths=1.2,zorder=8)
        if line['kind']=='tie':ax.annotate(line['name'],mid,xytext=(3,5),textcoords='offset points',fontsize=6.5,color=color)
        if show_switches and line['switchable']:
            ax.scatter(*mid,marker='s',s=12,facecolor='.15' if on else 'white',edgecolor='.15',linewidths=.55,zorder=4)
            if not on:ax.annotate(line['switch_id'],mid,xytext=(0,-9),textcoords='offset points',ha='center',fontsize=6.5)
    if fraction is None:ax.scatter(xy[:,0],xy[:,1],s=24,fc='white',ec='.1',lw=.6,zorder=4)
    else:
        f=np.asarray(fraction);states=[(live<.5,'s','#BDBDBD'),((live>.5)&~np.isfinite(f),'h','white'),
            ((live>.5)&np.isfinite(f)&(f>=.999),'o',COLORS[2]),((live>.5)&np.isfinite(f)&(f>1e-5)&(f<.999),'^','#E69F00'),
            ((live>.5)&np.isfinite(f)&(f<=1e-5),'o','white')]
        for mask,marker,color in states:ax.scatter(xy[mask,0],xy[mask,1],s=28,marker=marker,c=color,edgecolor='.1',linewidths=.65,zorder=4)
    for i,(x,y) in enumerate(xy):ax.annotate(c['nodes'][i].removeprefix('D'),(x,y),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=6.5,zorder=6)
    for j,node in enumerate(c['mg_nodes']):
        ax.scatter(*xy[node],s=95,fc='none',ec=COLORS[j%len(COLORS)],marker='D',lw=1,zorder=7)
        ax.annotate(c['microgrids'][j]['name'],xy[node],xytext=(0,-13),textcoords='offset points',ha='center',fontsize=6.5)
    ax.annotate('TSO / G',xy[0],xytext=(0,15),textcoords='offset points',ha='center',fontsize=7)
    ax.text(np.mean(xy[:,0]),np.max(xy[:,1])+1.1,'DSO配网',fontsize=7,ha='center')
    ax.set_aspect('equal');ax.axis('off');ax.margins(x=.07,y=.27)


def plot_layout(c,w,out):
    fig=figure(3.1);ax=fig.add_subplot();bad=np.asarray(w['failure_hour'])<=0;z=np.array([e['initial_closed'] for e in c['edges']]);z[bad]=0
    _network(ax,c,z,bad,show_switches=True)
    fig.legend(handles=[Line2D([],[],color='.1',label='闭合电气支路'),Line2D([],[],color='.6',ls='--',label='断开电气支路'),Line2D([],[],marker='x',color='#B2182B',ls='',label='初始故障'),Line2D([],[],marker='D',color='.1',mfc='none',ls='',label='微网接入点')],ncol=4,loc='outside lower center')
    return save(fig,out,'00_主配微接线与开关位置')


def plot_weather(c,w,out):
    H=c['horizon_hours'];t=np.arange(c['n_steps'])*c['dt'];ends=np.asarray(w['probability_time_hours']);failed=np.flatnonzero(np.asarray(w['failure_hour'])<H)
    selected=sorted(failed,key=lambda e:-c['repair_priority'][e])[:3] or [j for j,e in enumerate(c['edges']) if e['exposed']][:3]
    fig=figure(4.7);axes=fig.subplots(2,2);geo=c['geo_xy'];ax=axes[0,0]
    for e,line in enumerate(c['edges']):
        a,b=geo[line['from']],geo[line['to']];ax.plot([a[0],b[0]],[a[1],b[1]],c='.7',lw=.6,ls='--' if line['kind']=='tie' else '-')
        if e in failed:ax.scatter(*((a+b)/2),marker='x',s=20,c='#B2182B')
    tr=c['track'];ax.plot([r['center_x_km'] for r in tr],[r['center_y_km'] for r in tr],c=COLORS[0],ls='-.',lw=1,label='局部台风中心路径')
    ax.scatter(geo[:,0],geo[:,1],s=6,c='.2');ax.set(xlim=(geo[:,0].min()-8,geo[:,0].max()+8),ylim=(geo[:,1].min()-6,geo[:,1].max()+6),xlabel='合成坐标 x / km',ylabel='合成坐标 y / km');ax.legend();panel(ax,'a','合成风场与首损位置')
    for j,e in enumerate(selected):
        axes[0,1].plot(t,w['wind'][e],label=_line(c,e),**series(j,len(t)))
        axes[1,0].plot(ends,100*np.asarray(w['interval_probability_if_intact'])[e],label=_line(c,e),**series(j,len(t)))
        if 0<=w['failure_hour'][e]<H:axes[0,1].axvline(w['failure_hour'][e],color=COLORS[j],ls=':',lw=.7)
    axes[0,1].axhline(c['crew']['safe_wind_ms'],c='.2',ls='--',lw=.7,label='作业风速上限');axes[0,1].set_ylabel('风速 / (m/s)');axes[0,1].legend(ncol=2);panel(axes[0,1],'b','代表线路风速与首损时刻')
    axes[1,0].set_ylabel(f'{c["step_minutes"]:g}分钟条件首损概率 / %');axes[1,0].legend();panel(axes[1,0],'c','条件于尚未首次损坏')
    F=np.asarray(w['cumulative_first_failure_probability']);mean=F.sum(axis=0);sd=np.sqrt((F*(1-F)).sum(axis=0))
    axes[1,1].fill_between(ends,np.maximum(0,mean-2*sd),np.minimum(len(c['edges']),mean+2*sd),color='.9',hatch='..',label='均值±2σ')
    axes[1,1].plot(ends,mean,c=COLORS[0],label='模型均值');axes[1,1].step(ends,[np.sum(np.asarray(w['failure_hour'])<=h+1e-8) for h in ends],where='post',c=COLORS[1],ls='--',label='单条实际样本路径')
    axes[1,1].set_ylabel('首损线路累计数');axes[1,1].legend();panel(axes[1,1],'d','首损计数模型与单条样本')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:timeaxis(ax,H)
    return save(fig,out,'01_台风输入与首次故障概率')


def plot_faults(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];bad=np.asarray(r['damaged']);z=np.asarray(r['closed']);live=np.asarray(r['energized_edges'])
    ids=np.flatnonzero(np.any(bad,axis=0)|np.any(np.abs(np.diff(z,axis=0))>.5,axis=0));fig=figure(max(2.6,len(ids)*.18+1.1));ax=fig.add_subplot()
    for row,e in enumerate(ids):
        positions=np.flatnonzero(bad[:,e]>.5)
        for block in np.split(positions,np.flatnonzero(np.diff(positions)>1)+1):
            if len(block):ax.broken_barh([(t[block[0]],len(block)*dt)],(row-.28,.56),fc='#F4B6A6',ec='#B2182B',hatch='////',lw=.5)
        on=np.flatnonzero((live[:,e]>.5)&(bad[:,e]<.5))
        for block in np.split(on,np.flatnonzero(np.diff(on)>1)+1):
            if len(block):ax.plot([t[block[0]],t[block[-1]]+dt],[row,row],c=COLORS[0],lw=2)
        for event in r['events']:
            if event['line']!=c['edges'][e]['name']:continue
            if '维修完成' in event['kind']:ax.scatter(event['hour'],row+.12,s=21,marker='o',c=COLORS[2],ec='.1',lw=.4,zorder=5)
            if event['kind'] in ('主动合闸','主动分闸'):ax.scatter(event['hour'],row-.12,s=22,marker='^' if event['kind']=='主动合闸' else 'v',fc='white',ec='.1',lw=.8,zorder=6)
        if bad[-1,e]>.5:ax.annotate('未修复',(c['horizon_hours'],row),xytext=(3,0),textcoords='offset points',va='center',fontsize=6.5)
    ax.set_yticks(range(len(ids)),[_line(c,e) for e in ids]);ax.invert_yaxis();timeaxis(ax,c['horizon_hours']);ax.set_ylabel('故障或操作线路')
    fig.legend(handles=[Patch(fc='#F4B6A6',ec='#B2182B',hatch='////',label='故障未修复'),Line2D([],[],c=COLORS[0],lw=2,label='实际带电'),Line2D([],[],marker='o',c=COLORS[2],ls='',label='维修完成'),Line2D([],[],marker='^',mfc='white',c='.1',ls='',label='主动合闸'),Line2D([],[],marker='v',mfc='white',c='.1',ls='',label='主动分闸')],ncol=5,loc='outside upper center')
    return save(fig,out,'02_故障修复开关与带电时序')


def plot_dispatch(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];fig=figure(3.8);axes=fig.subplots(2,1,sharex=True)
    for j,(key,label) in enumerate([('upward_p','DSO向TSO净输出'),('target_p','机组有功'),('aux','启动辅机')]):stairs(axes[0],t,np.asarray(r[key]),dt,label=label,**series(j,len(t)))
    axes[0].axhline(0,c='.2',lw=.6);axes[0].set_ylabel('有功 / MW');axes[0].legend(ncol=3);panel(axes[0],'a','主配功率及启动支撑')
    if np.max(r['aux'])>1e-8:
        inset=axes[0].inset_axes([.56,.5,.38,.44]);limit=min(c['horizon_hours'],max(2.,len(c['generator']['startup_profile_mw'])*dt+dt))
        stairs(inset,t,np.asarray(r['aux']),dt,color=COLORS[2]);inset.set(xlim=(0,limit),ylim=(0,np.max(r['aux'])*1.2),title='辅机阶段放大');inset.tick_params(labelsize=6.5)
    for j,mg in enumerate(c['microgrids']):stairs(axes[1],t,np.asarray(r['pcc_p'])[:,j],dt,label=mg['name'],**series(j,len(t)))
    axes[1].axhline(0,c='.2',lw=.6);axes[1].set_ylabel('微网PCC净输出 / MW');axes[1].legend(ncol=len(c['microgrids']));panel(axes[1],'b','扣除微网本地已供负荷后的输出')
    for ax in axes:timeaxis(ax,c['horizon_hours'])
    files=save(fig,out,'03_主配微功率与辅机支撑')
    fig=figure(2.35);ax=fig.add_subplot();soc=np.asarray(r['energy'])/np.asarray(r['energy_capacity'])
    for j,mg in enumerate(c['microgrids']):
        ax.plot(np.arange(len(soc))*dt,soc[:,j],label=mg['name'],**series(j,len(soc)))
        ax.axhline(mg['soc_min'],color=COLORS[j],ls=':',lw=.55);ax.axhline(mg['soc_max'],color=COLORS[j],ls=':',lw=.55)
    ax.set(ylim=(max(0,min(mg['soc_min'] for mg in c['microgrids'])-.04),min(1,max(mg['soc_max'] for mg in c['microgrids'])+.04)),ylabel='SOC / p.u.');timeaxis(ax,c['horizon_hours']);ax.legend(ncol=len(c['microgrids']));panel(ax,'a','实际SOC与物理上下界')
    return files+save(fig,out,'03b_微网SOC与状态边界')


def plot_recovery(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];sc=np.asarray(r['served_c']).sum(axis=1);sg=np.asarray(r['served_g']).sum(axis=1);dc=c['profiles']['critical'].sum(axis=0);dg=c['profiles']['general'].sum(axis=0)
    fig=figure(4.1);axes=fig.subplots(2,2)
    for j,(values,label) in enumerate([(dc+dg,'总需求'),(sc+sg,'实际已供')]):stairs(axes[0,0],t,values,dt,label=label,**series(j,len(t)))
    axes[0,0].set_ylabel('有功 / MW');axes[0,0].legend();panel(axes[0,0],'a','需求和供电')
    for j,(values,label) in enumerate([(np.asarray(r['recovery']),'加权'),(sc/np.maximum(dc,1e-9),'关键'),(sg/np.maximum(dg,1e-9),'一般')]):stairs(axes[0,1],t,100*values,dt,label=label,**series(j,len(t)))
    axes[0,1].set(ylim=(0,103),ylabel='恢复率 / %');axes[0,1].legend(ncol=3);panel(axes[0,1],'b','三种恢复率分别统计')
    for j,(values,label) in enumerate([(dc-sc,'关键负荷'),(dg-sg,'一般负荷')]):axes[1,0].plot(np.r_[t,c['horizon_hours']],np.r_[0,np.cumsum(values)*dt],label=label,**series(j,len(t)))
    axes[1,0].set_ylabel('累计失供电量 / MWh');axes[1,0].legend();panel(axes[1,0],'c','关键与一般失供电量')
    V=np.asarray(r['voltage']);on=np.asarray(r['energized_bus'])>.5;vmin=np.min(np.where(on,V,np.inf),axis=1);vmax=np.max(np.where(on,V,-np.inf),axis=1);vmin[~np.isfinite(vmin)]=np.nan;vmax[~np.isfinite(vmax)]=np.nan
    stairs(axes[1,1],t,vmin,dt,label='带电节点最小',**series(0,len(t)));stairs(axes[1,1],t,vmax,dt,label='带电节点最大',**series(1,len(t)))
    for value in (c['voltage_min'],c['voltage_max']):axes[1,1].axhline(value,c='.2',ls=':',lw=.7)
    axes[1,1].set(ylim=(min(c['voltage_min'],np.nanmin(vmin))-.005,max(c['voltage_max'],np.nanmax(vmax))+.005),ylabel='电压 / p.u.');axes[1,1].legend();panel(axes[1,1],'d','带电节点电压及模型限值')
    for ax in axes.flat:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'04_全过程恢复与韧性指标')


def plot_crew(c,r,out):
    trace=r['crew_trace'];dt=r['dt'];ids=list(dict.fromkeys(q['edge'] for q in trace if q is not None));fig=figure(max(3.,len(ids)*.23+1.3));axes=fig.subplots(2,1,height_ratios=[max(1.,len(ids)*.24),1],sharex=True)
    colors={'travel':'.65','inspect':COLORS[0],'repair':COLORS[1],'wait':'white'}
    def key(q):return None if q is None else (q['edge'],'wait' if q.get('weather_wait') else 'travel' if q['travel_remaining']>1e-8 else q['kind'])
    start=0
    for t in range(1,len(trace)+1):
        if t<len(trace) and key(trace[t])==key(trace[start]):continue
        value=key(trace[start])
        if value:
            e,kind=value;axes[0].broken_barh([(start*dt,(t-start)*dt)],(ids.index(e)-.3,.6),fc=colors[kind],ec='.25',hatch='////' if kind=='wait' else None,lw=.45)
        start=t
    for ev in r['events']:
        if '巡检完成' in ev['kind']:
            e=next(j for j,line in enumerate(c['edges']) if line['name']==ev['line'])
            if e in ids:axes[0].scatter(ev['hour'],ids.index(e),marker='D',s=18,c='black',zorder=6);axes[0].annotate('H' if ev['damage']=='heavy' else 'L',(ev['hour'],ids.index(e)),xytext=(3,4),textcoords='offset points',fontsize=6.5)
    axes[0].set_yticks(range(len(ids)),[f'{j+1}. {_line(c,e)}' for j,e in enumerate(ids)]);axes[0].invert_yaxis();axes[0].set_ylabel('线路（按首次访问排序）');panel(axes[0],'a','行程、巡检、维修与天气等待')
    fig.legend(handles=[Patch(fc=colors[k],ec='.3',hatch='////' if k=='wait' else None,label=label) for k,label in [('travel','行程'),('inspect','巡检'),('repair','维修'),('wait','天气等待')]]+[Line2D([],[],marker='D',c='black',ls='',label='类型揭示L/H')],ncol=5,loc='outside upper center')
    for j,(kind,label) in enumerate([('巡检完成','已获知类型'),('维修完成','已修复')]):stairs(axes[1],np.asarray(r['time_hours']),np.array([sum(kind in ev['kind'] and ev['hour']<=h+1e-8 for ev in r['events']) for h in r['time_hours']]),dt,label=label,**series(j,len(trace)))
    axes[1].set_ylabel('线路数');axes[1].legend(ncol=2);panel(axes[1],'b','信息到达和物理修复分离')
    for ax in axes:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'05_巡检揭示与维修作业甘特图')


def plot_D(r,out):
    rows=[q for q in r['rounds'] if q.get('D_lower') is not None or q.get('D_upper') is not None]
    if not rows:return []
    if len(rows)==1:
        row=rows[0];fig=figure(2.05);axes=fig.subplots(1,2,width_ratios=[1.5,1]);ax=axes[0]
        for j,(branch,label) in enumerate([('start','当前启动'),('wait','当前等待')]):
            q=row['branches'][branch];lo=q.get('R_lower');hi=q.get('R_upper')
            if lo is not None and hi is not None:ax.plot([100*lo,100*hi],[j,j],c=COLORS[j],lw=1.5);ax.scatter(100*q['R_plan'],j,marker=['^','v'][j],c=COLORS[j],s=28)
        ax.set(yticks=[0,1],yticklabels=['当前启动','当前等待'],xlabel='预测加权恢复面积 / %',ylim=(-.6,1.6));ax.grid(axis='x',color='.9');panel(ax,'a',f't={row["hour"]:g}h，两动作的可行值与外界')
        axes[1].axis('off');lo=row.get('D_lower');hi=row.get('D_upper');values=[['项目','当前证据'],['D估计/百分点','—' if row.get('D_estimate') is None else f'{100*row["D_estimate"]:.4f}'],['D下界/百分点','—' if lo is None else f'{100*lo:.4f}'],['D上界/百分点','—' if hi is None else f'{100*hi:.4f}'],['判定',row['classification']]]
        table=axes[1].table(cellText=values,cellLoc='center',loc='center',colWidths=[.55,.45]);table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.35)
        panel(axes[1],'b','有限策略类首动作认证')
    else:
        fig=figure(max(2.1,.22*len(rows)+.8),width='single');ax=fig.add_subplot()
        for j,q in enumerate(rows):
            if q.get('D_lower') is not None and q.get('D_upper') is not None:ax.plot([100*q['D_lower'],100*q['D_upper']],[j,j],c=COLORS[0])
            if q.get('D_estimate') is not None:ax.scatter(100*q['D_estimate'],j,c=COLORS[1],marker='o',s=18)
        ax.axvline(0,c='.2',ls='--',lw=.7);ax.set(yticks=range(len(rows)),yticklabels=[f'{q["hour"]:g}h' for q in rows],xlabel='D_R / 百分点');panel(ax,'a','仅显示存在自由首动作选择的时刻')
    return save(fig,out,'06_首动作恢复值与D认证证据')


def plot_capability(cap,out):
    if not cap:return []
    slices=cap['slices'];H=cap['horizon_hours'];fig=figure(4.4);gs=fig.add_gridspec(2,6);axes=[fig.add_subplot(gs[0,:3]),fig.add_subplot(gs[0,3:])]
    for coord,(ax,label) in enumerate(zip(axes,['P / MW','Q / Mvar'])):
        t=[q['hour'] for q in slices];lo=[min((z[coord] for z in q['points']),default=np.nan) for q in slices];hi=[max((z[coord] for z in q['points']),default=np.nan) for q in slices]
        ax.fill_between(t,lo,hi,color='.9',hatch='..',label='采样截面范围');ax.plot(t,lo,c='.35',ls='--');ax.plot(t,hi,c='.35',ls='--')
        ax.plot(cap['time_hours'],cap['reference_p' if coord==0 else 'reference_q'],c=COLORS[0],label='同一完整轨迹');ax.set_ylabel(label);timeaxis(ax,H);ax.legend();panel(ax,chr(97+coord),'完整时域条件截面与参考轨迹')
    chosen=np.unique(np.linspace(0,len(slices)-1,3).astype(int))
    for j,k in enumerate(chosen):
        ax=fig.add_subplot(gs[1,2*j:2*j+2]);q=slices[k];pts=np.asarray(q['points'])
        if len(pts):
            ax.scatter(pts[:,0],pts[:,1],s=18,c=COLORS[0],marker='o',label='方向支持点')
            if len(pts)>=3:
                try:
                    ids=ConvexHull(pts).vertices;poly=pts[np.r_[ids,ids[0]]];ax.plot(poly[:,0],poly[:,1],c='.2',lw=.9)
                except QhullError:pass
        ax.set(xlabel='P / MW',ylabel='Q / Mvar');ax.grid(color='.9',lw=.4);panel(ax,chr(99+j),f't={q["hour"]:g}h，{len(pts)}个支持点')
    return save(fig,out,'07_24小时条件截面与采样内近似')


def plot_phase(phase,out):
    if not phase:return []
    rows=phase['records'];finite=[q for q in rows if q.get('D_estimate') is not None];fig=figure(2.8);axes=fig.subplots(1,2)
    if finite:
        z=np.array([100*q['D_estimate'] for q in finite]);lim=max(.01,float(np.abs(z).max()));im=axes[0].scatter([q['alpha'] for q in finite],[q['soc'] for q in finite],c=z,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=150,edgecolors='.2',linewidths=.5)
        fig.colorbar(im,ax=axes[0],label='可行计划差值 / 百分点',shrink=.75)
        for q in finite:axes[0].annotate(f'{100*q["D_estimate"]:+.3f}',(q['alpha'],q['soc']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=6.5)
    markers={'unknown':('x','.4','未判'),'start':('^',COLORS[2],'启动'),'wait':('v',COLORS[0],'等待'),'recovery_tie':('D','#E69F00','近等效'),'start_infeasible':('s','.1','启动不可行'),'wait_infeasible':('P',COLORS[1],'等待不可行'),'both_infeasible':('X','#B2182B','两者不可行')}
    seen=set()
    for q in rows:
        marker,color,label=markers.get(q['classification'],markers['unknown']);axes[1].scatter(q['alpha'],q['soc'],s=65,marker=marker,c=color,label=label if label not in seen else None);seen.add(label)
    axes[1].legend(ncol=2,loc='best');panel(axes[0],'a','实际采样点的计划差值');panel(axes[1],'b','由有效最优值界给出的分类')
    for ax in axes:
        ax.set(xlabel='主网剩余能力 alpha / p.u.',ylabel=('初始SOC / p.u.' if phase.get('state_hour',0)==0 else '当前SOC / p.u.'),xticks=phase['alphas'],yticks=phase['socs']);ax.margins(x=.2,y=.25);ax.grid(color='.9',lw=.4)
    return save(fig,out,'08_D采样点与认证决策相图')


def plot_comparison(results,out):
    if len(results)<2:return []
    fig=figure(2.9);axes=fig.subplots(1,2);labels={'joint':'联合有限规则类','critical_first':'关键负荷优先','nearest':'就近优先','no_upward':'禁止主配反送'}
    for j,r in enumerate(results):stairs(axes[0],np.asarray(r['time_hours']),100*np.asarray(r['recovery']),r['dt'],label=labels.get(r['policy'],r['policy']),**series(j,len(r['time_hours'])))
    timeaxis(axes[0],results[0]['time_hours'][-1]+results[0]['dt']);axes[0].set_ylabel('实际加权恢复率 / %');axes[0].legend();panel(axes[0],'a','同一风灾样本的实际首步执行')
    bars=axes[1].bar(range(len(results)),[r['summary']['actual_EENS_MWh'] for r in results],color=COLORS[:len(results)],ec='.2',width=.6)
    for j,b in enumerate(bars):b.set_hatch(['','///','..','xx'][j%4]);axes[1].text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height():.2f}',ha='center',va='bottom',fontsize=7)
    axes[1].set(xticks=range(len(results)),xticklabels=[labels.get(r['policy'],r['policy']) for r in results],ylabel='实际EENS / MWh');axes[1].tick_params(axis='x',labelrotation=15);panel(axes[1],'b','实际失供电量，不预设策略优势')
    return save(fig,out,'09_多策略实际恢复与失供电量')


def plot_network(c,r,out):
    hours=[float(v) for v in str(c.get('options',{}).get('plots.snapshot_hours','0,4,12,23.75')).split(',')];fig=figure(4.8);axes=fig.subplots(int(np.ceil(len(hours)/2)),2,squeeze=False)
    served=np.asarray(r['served_c'])+np.asarray(r['served_g']);demand=(c['profiles']['critical']+c['profiles']['general']).T
    for ax,h in zip(axes.flat,hours):
        t=min(c['n_steps']-1,max(0,int(round(h/c['dt']))));f=np.divide(served[t],demand[t],out=np.full_like(served[t],np.nan),where=demand[t]>1e-9)
        _network(ax,c,r['closed'][t],r['damaged'][t],r['energized_bus'][t],f,show_switches=True)
        dc=c['profiles']['critical'][:,t].sum();rate=np.asarray(r['served_c'])[t].sum()/dc if dc>0 else 1.;time=f'{r["time_hours"][t]:g}h' if t<c['n_steps']-1 else f'{r["time_hours"][t]:g}—{c["horizon_hours"]:g}h'
        ax.set_title(f'{time}，关键负荷 {100*rate:.1f}%',loc='left',fontsize=8)
    for ax in list(axes.flat)[len(hours):]:ax.axis('off')
    handles=[Line2D([],[],marker=mk,c='.1',mfc=col,ls='',label=label) for mk,col,label in [('o',COLORS[2],'全部恢复'),('^','#E69F00','部分恢复'),('o','white','带电未供'),('s','#BDBDBD','失电'),('h','white','无本地负荷')]]+[Line2D([],[],c='#B2182B',ls='--',marker='x',label='故障支路')]
    fig.legend(handles=handles,ncol=3,loc='outside lower center')
    return save(fig,out,'10_节点服务状态与开关拓扑快照')


def plot_information(info,out):
    if not info or not info.get('information_evidence'):return []
    ev=info['information_evidence'];hours=np.asarray(ev['hours']);H=info['horizon_hours'];fig=figure(3.1);axes=fig.subplots(2,1,height_ratios=[1,1.6])
    axes[0].step(hours,ev['group_count'],where='post',c=COLORS[0],marker='o',markevery=max(1,len(hours)//12),ms=3)
    axes[0].set_ylabel('观测等价类数');timeaxis(axes[0],H);panel(axes[0],'a',f'{len(ev["scenario_probabilities"])}个样本的可区分性演化')
    nodes=ev['nodes'];selected=[];seen=set()
    for node in nodes:
        key=tuple((d['line'],d['type']) for d in node['diagnoses'])
        if (node['history_id'],key) not in seen:
            if not selected or node['diagnoses'] or node['new_dispatch']:selected.append(node)
            seen.add((node['history_id'],key))
    # 展示有限节选，全部96步节点保存在JSON/Excel；不凑不同策略为样本。
    selected=selected[:8];rows=[]
    for node in selected:
        known='；'.join(d['line'].replace('L','').replace('_','-')+('=H' if d['type']=='heavy' else '=L') for d in node['diagnoses'][-2:]) or '尚未获知类型'
        rows.append([f'{node["hour"]:g}',known,f'{node["probability"]:.3g}',node['crew_action'].replace('L','').replace('_','-'),'启动' if node['start_command'] else '无新启动'])
    axes[1].axis('off');table=axes[1].table(cellText=rows,colLabels=['时刻/h','类型历史摘要（节选）','组概率','组内队伍动作','机组动作'],cellLoc='center',loc='center',colWidths=[.09,.34,.09,.3,.18]);table.auto_set_font_size(False);table.set_fontsize(6.5);table.scale(1,1.28);panel(axes[1],'b','同信息组的历史—动作映射节选')
    files=save(fig,out,'11_观测等价类与因果动作证据')
    post=np.asarray(ev['posterior_heavy']);k=len(ev['hidden_targets'])
    if k:
        fig=figure(max(2.2,k*1.25));axs=fig.subplots(k,1,squeeze=False,sharex=True)
        for j,ax in enumerate(axs.flat):
            for s in range(post.shape[0]):stairs(ax,hours,post[s,j],H/len(hours),label=f'样本{s+1}',**series(s,len(hours)))
            ax.set(ylim=(-.05,1.05),yticks=[0,.5,1],ylabel='P(重伤)');timeaxis(ax,H);panel(ax,chr(97+j),ev['hidden_targets'][j]+'的有限支持后验')
        axs[0,0].legend(ncol=min(4,post.shape[0]));files+=save(fig,out,'11b_类型揭示与后验变化')
    return files


def plot_backward(c,info,out):
    if not info or not info.get('backward_certificate'):return []
    cert=info['backward_certificate'];low=np.asarray(cert['lower'])[0];high=np.asarray(cert['upper'])[0];reserve=info.get('reserve_tube');nm=len(c['microgrids']);fig=figure(max(3.,nm*1.1));axes=fig.subplots(nm,1,squeeze=False,sharex=True);times=np.linspace(0,c['horizon_hours'],low.shape[1])
    for j,ax in enumerate(axes.flat):
        cap=c['microgrids'][j]['energy_mwh'];lo=low[j]/cap;hi=high[j]/cap
        ax.fill_between(times,lo,hi,color=COLORS[0],alpha=.12);ax.plot(times,lo,c=COLORS[0],ls='-',lw=1,label='条件管下界');ax.plot(times,hi,c=COLORS[0],ls='--',lw=1,label='条件管上界')
        if reserve:ax.plot(times,np.asarray(reserve['lower'])[j]/cap,c='.15',ls=':',lw=1,label='岛内后备下界')
        if info.get('reference_energy') is not None:ax.plot(times,np.asarray(info['reference_energy'])[0,j]/cap,c=COLORS[1],lw=.9,label='选定候选轨迹')
        ax.scatter(0,c['microgrids'][j]['soc_initial'],s=24,marker='o',c='black',zorder=5)
        if cert.get('takeover_hour') is not None:ax.axvline(cert['takeover_hour'],c=COLORS[2],ls='-.',lw=.8,label='预测接管时刻')
        ax.set(ylim=(max(0,float(np.nanmin(lo))-.05),min(1,float(np.nanmax(hi))+.05)),ylabel=c['microgrids'][j]['name']+' SOC');timeaxis(ax,c['horizon_hours']);panel(ax,chr(97+j),'样本1条件分支')
    axes[0,0].annotate('从终端向前驱递推',xy=(.55,.73),xytext=(.98,.73),xycoords='axes fraction',textcoords='axes fraction',arrowprops={'arrowstyle':'->','lw':.8},ha='right',fontsize=7)
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=3,loc='outside upper center');files=save(fig,out,'12_条件后向管上下界与接管标志')
    actions=info.get('action_evidence',{})
    if any(key in actions for key in ('start','wait')):
        fig=figure(2.6,width='single');ax=fig.add_subplot();row=0;labels=[]
        for branch,label in [('start','启动'),('wait','等待')]:
            item=actions.get(branch)
            if not item:continue
            for j,(lo,hi,current) in enumerate(zip(item['initial_lower_SOC'],item['initial_upper_SOC'],item['current_SOC'])):
                ax.plot([lo,hi],[row,row],c=COLORS[0 if branch=='start' else 1],lw=2);ax.scatter(current,row,c='black',marker='^' if branch=='start' else 'v',s=22);labels.append(c['microgrids'][j]['name']+'/'+label);row+=1
        ax.set(yticks=range(row),yticklabels=labels,xlabel='当前SOC / p.u.',xlim=(0,1));ax.invert_yaxis();ax.grid(axis='x',color='.9');panel(ax,'a','各动作已找到计划的初始可实施区间')
        files+=save(fig,out,'12b_首动作条件SOC证据')
    return files


def plot_rolling(c,r,out):
    steps=r.get('controller_steps',[])
    if not steps:return []
    chosen=np.unique(np.linspace(0,len(steps)-1,min(8,len(steps))).astype(int));fig=figure(4.2);axes=fig.subplots(3,1,height_ratios=[2,1,1],sharex=True);H=c['horizon_hours']
    for row,k in enumerate(chosen):
        h=steps[k]['hour'];axes[0].broken_barh([(0,h)],(row-.3,.6),fc='.7',ec='.4',hatch='///',lw=.4);axes[0].broken_barh([(h,H-h)],(row-.3,.6),fc='white',ec=COLORS[0],hatch='..',lw=.5);axes[0].broken_barh([(h,c['dt'])],(row-.3,.6),fc=COLORS[1],ec=COLORS[1]);axes[0].scatter(h,row,s=20,marker='>',c=COLORS[1],zorder=6)
    axes[0].set_yticks(range(len(chosen)),[f'{steps[k]["hour"]:g}' for k in chosen]);axes[0].invert_yaxis();axes[0].set_ylabel('决策时刻 / h');panel(axes[0],'a',f'从{len(steps)}步等距选{len(chosen)}步显示，只下发当前一步')
    styles={'new_fault_report':('^','#B2182B','新故障'),'inspection_information_arrived':('D',COLORS[0],'信息揭示'),'repair_completed':('s',COLORS[2],'修复完成'),'periodic_replan':('o','.2','周期/其他联合重算')};used=set()
    for q in steps:
        if q['solve']=='fixed_plan_LP':axes[1].scatter(q['hour'],0,marker='|',s=25,c='.4');continue
        reason=next((key for key in styles if key in q.get('trigger_reasons',[])),'periodic_replan');mk,col,label=styles[reason];axes[1].scatter(q['hour'],1,marker=mk,s=23,c=col,label=label if label not in used else None);used.add(label)
    axes[1].set(yticks=[0,1],yticklabels=['连续域LP','联合MILP'],ylim=(-.5,1.5));axes[1].legend(ncol=2,loc='upper right');panel(axes[1],'b','完整96步求解事件轨道与主要触发源')
    aux=np.asarray(r['aux']);on=np.asarray(r['unit_online']);state=np.where(aux>1e-8,1,np.where(on>.5,2,0));names=['未启动','辅机支撑','机组在线'];colors=['.8',COLORS[1],COLORS[2]];start=0
    for t in range(1,len(state)+1):
        if t<len(state) and state[t]==state[start]:continue
        axes[2].broken_barh([(start*c['dt'],(t-start)*c['dt'])],(.2,.5),fc=colors[int(state[start])],ec='.3',hatch=['///','..',''][int(state[start])],lw=.4);start=t
    axes[2].set(yticks=[],ylim=(0,1));axes[2].legend(handles=[Patch(fc=colors[j],ec='.3',hatch=['///','..',''][j],label=names[j]) for j in sorted(set(state.tolist()))],ncol=3,loc='upper right');panel(axes[2],'c','实际机组阶段，承诺后不重新开放启动选择')
    for ax in axes:timeaxis(ax,H)
    return save(fig,out,'13_首步执行触发原因与启动承诺')


def plot_trajectory_hull(cap,out):
    if not cap or not cap.get('splicing_counterexample'):return []
    test=cap['splicing_counterexample'];t=np.asarray(cap['time_hours']);H=cap['horizon_hours'];fig=figure(2.8);axes=fig.subplots(1,3)
    axes[0].plot(t,cap['reference_p'],label='可分解参考',**series(0,len(t)))
    if test.get('target_p') is not None:axes[0].plot(t,test['target_p'],label='逐时拼接请求',**series(1,len(t)))
    first=test.get('first_nonextendable_prefix_end_hour')
    if first is not None:axes[0].axvline(first,c='#B2182B',ls=':',lw=.8)
    axes[0].set_ylabel('PCC有功总和 / MW');axes[0].legend();panel(axes[0],'a','接口请求及参考轨迹')
    prefix=test.get('prefix_extendable')
    if prefix is not None:
        axes[1].step(np.arange(len(prefix))*H/(len(prefix)-1),[np.nan if z is None else z for z in prefix],where='post',c=COLORS[0])
        axes[1].set(yticks=[0,1],yticklabels=['不可延续','可延续'],ylim=(-.2,1.2))
        if first is not None:axes[1].axvline(first,c='#B2182B',ls=':');axes[1].text(.03,.1,f'首个不可延续前缀末端\n{first:g} h',transform=axes[1].transAxes,fontsize=7)
    else:axes[1].text(.5,.5,'未保存前缀证据',transform=axes[1].transAxes,ha='center')
    panel(axes[1],'b','能否延续到原终端目标')
    if test.get('slack_scenario_0') is not None:
        slack=np.asarray(test['slack_scenario_0']);pfix=slack[:,:2].sum(axis=1);stairs(axes[2],t,pfix,H/len(t),color=COLORS[1]);axes[2].set_ylabel('样本1有功绝对修正 / MW')
        axes[2].text(.03,.93,f'加权累计修正\n{test["P_correction_MWh"]:.3f} MWh',transform=axes[2].transAxes,va='top',fontsize=7)
    else:axes[2].text(.5,.5,'未保存最小修正证据',transform=axes[2].transAxes,ha='center')
    panel(axes[2],'c','保持硬约束的最小跟踪修正')
    for ax in axes:timeaxis(ax,H)
    return save(fig,out,'14_拼接前缀可延续性与最小修正')


def plot_domain(r,info,out):
    rounds=[q for q in r['rounds'] if q.get('domain')]
    if not rounds:return []
    fig=figure(2.7);axes=fig.subplots(1,2);q=rounds[0]['domain'];weights=q['lambda'];bars=axes[0].bar(range(len(weights)),weights,color=COLORS[0],ec='.2')
    for j,b in enumerate(bars):b.set_hatch(['','//','..','xx'][j%4])
    axes[0].set(xlabel='完整轨迹列编号',ylabel='共同组合系数lambda',ylim=(0,1.05));panel(axes[0],'a','同一lambda用于全部时段和样本')
    t=[q['hour'] for q in rounds]
    axes[1].plot(t,[q['domain']['original_variables'] for q in rounds],label='原连续扩展模型',**series(0,len(t)))
    axes[1].plot(t,[q['domain']['master_variables'] for q in rounds],label='消去微网内部列后',**series(1,len(t)))
    axes[1].set_ylabel('变量数');timeaxis(axes[1],r['time_hours'][-1]+r['dt']);axes[1].legend();panel(axes[1],'b','真实求解规模，不预设速度优势')
    return save(fig,out,'15_在线完整轨迹列协调证据')


def plot_all(c,world,result,out,cap=None,phase=None,comparisons=None,information=None):
    setup(c);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # 只清理由旧清单登记、且位于当前图目录的生成图；不删除用户其他文件。
    manifest=out/'figures_manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        for name in previous.get('files',[]):
            base=Path(str(name).replace('\\','/')).name;p=out/base
            if p.suffix.lower() in ('.png','.pdf','.svg') and base[:2].isdigit() and p.is_file():p.unlink()
    files=[]
    for fn,args in [(plot_layout,(c,world,out)),(plot_weather,(c,world,out)),(plot_faults,(c,result,out)),(plot_dispatch,(c,result,out)),(plot_recovery,(c,result,out)),(plot_crew,(c,result,out)),(plot_D,(result,out)),(plot_capability,(cap,out)),(plot_phase,(phase,out)),(plot_comparison,(comparisons or [result],out)),(plot_network,(c,result,out)),(plot_information,(information,out)),(plot_backward,(c,information,out)),(plot_rolling,(c,result,out)),(plot_trajectory_hull,(cap,out)),(plot_domain,(result,information,out))]:files.extend(fn(*args))
    manifest.write_text(json.dumps({'files':files,'run_id':None if information is None else information.get('run_id'),'scope':'实际执行、条件资源域、有限策略类外界分别标识；没有源证据不补造图'},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'图注与数据口径.md').write_text('所有模型保留完整24h/96步。07每个支持点来自完整时域LP，轮廓是条件内近似；08为实际格点，未画连续临界面；11为单一候选预测树；12为给定候选的局部条件管；14前缀不可延续含终端/后备约束，不是瞬时超载，跟踪修正不等于EENS；15为固定离散方案的轨迹列主问题，不是原MILP最优上界。图中不烘焙总图题，正式caption在论文排版中添加。',encoding='utf-8')
    return files
