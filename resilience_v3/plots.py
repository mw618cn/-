"""按实际字段作图；期刊图中不放宣传式标题，不补造临界线或观测。"""
from pathlib import Path
import json
import numpy as np
# 先初始化可写缓存目录及无界面后端，再导入pyplot。
from plot_style import setup,figure,save,panel,timeaxis,stairs,COLORS,CONFIG
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull,QhullError


def style(c=None):
    if c is not None:return setup(c)


def _line(c,e):return c['edges'][e]['name'].replace('L','').replace('_','-')


def _network(ax,c,closed=None,damaged=None,live=None,fraction=None,show_switches=False):
    xy=c['plot_xy'];closed=np.array([e['initial_closed'] for e in c['edges']]) if closed is None else np.asarray(closed)
    damaged=np.zeros(len(c['edges'])) if damaged is None else damaged
    live=np.ones(len(c['nodes'])) if live is None else np.asarray(live)
    for e,line in enumerate(c['edges']):
        a,b=xy[line['from']],xy[line['to']];route=np.asarray(line.get('plot_route',[]),float).reshape(-1,2)
        path=np.vstack([a,route,b]);fault=bool(damaged[e]);on=closed[e]>.5
        ax.plot(path[:,0],path[:,1],color=COLORS[1] if fault else '.12' if on else '.65',
                linestyle='-' if on and not fault else '--',linewidth=1.05 if fault else .7,zorder=1)
        mid=path[len(path)//2] if len(path)>2 else (a+b)/2
        if fault:ax.scatter(*mid,marker='x',s=24,color=COLORS[1],linewidths=1,zorder=5)
        if show_switches and line['switchable']:
            ax.scatter(*mid,marker='s',s=8,facecolor='white' if not on else '.2',edgecolor='.2',linewidths=.4,zorder=4)
            ax.annotate(line['switch_id'],mid,xytext=(0,5 if e%2 else -8),textcoords='offset points',ha='center',fontsize=5)
    if fraction is None:
        ax.scatter(xy[:,0],xy[:,1],s=37,facecolor='white',edgecolor='.1',linewidths=.55,zorder=4)
    else:
        # 三种离散服务状态：颜色与边框同时区分；带电和恢复比例分别定义。
        f=np.asarray(fraction);colors=np.where(f>=.999,COLORS[2],np.where(f>1e-5,'#E69F00','white')).astype(object)
        colors[live<.5]='#BDBDBD'
        colors[(~np.isfinite(f))&(live>.5)]='#C5D5E3'
        ax.scatter(xy[:,0],xy[:,1],s=39,c=colors,edgecolor='.1',linewidths=.6,zorder=4)
    for i,(x,y) in enumerate(xy):ax.text(x,y,c['nodes'][i].removeprefix('D'),ha='center',va='center',fontsize=5.5,zorder=6)
    for j,node in enumerate(c['mg_nodes']):
        ax.scatter(*xy[node],s=105,facecolors='none',edgecolors=COLORS[j%len(COLORS)],marker='D',linewidths=1,zorder=7)
        ax.annotate(c['microgrids'][j]['name'],xy[node],xytext=(0,-14),textcoords='offset points',ha='center',fontsize=6,color='black')
    ax.text(xy[0,0],xy[0,1]+.65,'TSO / G',fontsize=6,ha='center')
    ax.text(np.mean(xy[:,0]),np.max(xy[:,1])+1.,'DSO',fontsize=7,ha='center')
    ax.set_aspect('equal');ax.axis('off');ax.margins(x=.03,y=.2)


def plot_layout(c,world,out):
    setup(c);fig=figure(3.0);ax=fig.add_subplot();dm=np.asarray(world['failure_hour'])<=0
    closed=np.array([e['initial_closed'] for e in c['edges']]);closed[dm]=0
    _network(ax,c,closed,dm,show_switches=True)
    ax.legend(handles=[Line2D([],[],color='.1',label='闭合线路'),Line2D([],[],color='.6',ls='--',label='断开线路'),
        Line2D([],[],color=COLORS[1],marker='x',ls='--',label='初始故障'),Line2D([],[],marker='D',mfc='none',color=COLORS[0],ls='',label='MGO接入点')],loc='upper center',ncol=4)
    return save(fig,out,'00_节点接线与开关位置')


def plot_weather(c,w,out):
    horizon=c['horizon_hours'];t=np.arange(c['n_steps'])*c['dt'];end=np.asarray(w['probability_time_hours'])
    failed=np.flatnonzero(np.asarray(w['failure_hour'])<horizon)
    selected=sorted(failed,key=lambda e:-c['repair_priority'][e])[:3]
    if not selected:selected=[e for e,line in enumerate(c['edges']) if line['exposed']][:3]
    fig=figure(4.8);axes=fig.subplots(2,2);xy=c['geo_xy']
    ax=axes[0,0]
    for e,line in enumerate(c['edges']):
        a,b=xy[line['from']],xy[line['to']];ax.plot([a[0],b[0]],[a[1],b[1]],color='.75',lw=.6,ls='--' if line['kind']=='tie' else '-')
        if e in failed:ax.scatter(*((a+b)/2),s=17,marker='x',color=COLORS[1],zorder=4)
        if e in selected:ax.annotate(_line(c,e),(a+b)/2,xytext=(3,4),textcoords='offset points',fontsize=6)
    tr=c['track'];ax.plot([r['center_x_km'] for r in tr],[r['center_y_km'] for r in tr],color=COLORS[0],ls='-.',lw=.9,label='台风中心路径')
    ax.scatter(xy[:,0],xy[:,1],s=5,c='.25');ax.set(xlim=(xy[:,0].min()-8,xy[:,0].max()+8),ylim=(xy[:,1].min()-6,xy[:,1].max()+6),xlabel='地理坐标 x / km',ylabel='地理坐标 y / km');ax.legend(loc='lower right',fontsize=6);panel(ax,'a','路径与首次故障位置')
    for j,e in enumerate(selected):
        color=COLORS[j];label=c['edges'][e]['name']
        axes[0,1].plot(t,w['wind'][e],color=color,label=label)
        fail=w['failure_hour'][e]
        if 0<=fail<horizon:axes[0,1].axvline(fail,color=color,ls=':',lw=.6)
        p=np.asarray(w['interval_probability_if_intact'])[e]
        axes[1,0].plot(end,p,color=color,label=label)
    axes[0,1].axhline(c['crew']['safe_wind_ms'],color='.3',ls='--',lw=.65,label='作业安全上限')
    axes[0,1].set_ylabel('风速 / (m/s)');axes[0,1].legend(fontsize=6,ncol=2);panel(axes[0,1],'b','代表线路风速与故障时刻')
    axes[1,0].set_ylabel('单时段故障概率');axes[1,0].legend(fontsize=6);panel(axes[1,0],'c','条件于尚未首损的区间概率')
    f=np.asarray(w['cumulative_first_failure_probability']);mean=f.sum(axis=0);sd=np.sqrt((f*(1-f)).sum(axis=0))
    axes[1,1].fill_between(end,np.maximum(0,mean-2*sd),np.minimum(len(c['edges']),mean+2*sd),color='.9',label='条件均值±2标准差')
    axes[1,1].plot(end,mean,color=COLORS[0],label='模型均值')
    count=[np.sum(np.asarray(w['failure_hour'])<=h+1e-8) for h in end]
    axes[1,1].step(end,count,where='post',color=COLORS[1],label='单次实际样本')
    axes[1,1].set_ylabel('首次故障线路数');axes[1,1].legend(fontsize=6);panel(axes[1,1],'d','概率模型与样本计数分开')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:timeaxis(ax,horizon)
    return save(fig,out,'01_台风路径风速与故障概率')


def plot_faults(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];dm=np.asarray(r['damaged']);closed=np.asarray(r['closed']);energy=np.asarray(r['energized_edges'])
    changed=np.any(dm,axis=0)|np.any(np.abs(np.diff(closed,axis=0))>.5,axis=0)
    idx=np.flatnonzero(changed);fig=figure(max(2.6,.18*len(idx)+1.2));ax=fig.add_subplot()
    for row,e in enumerate(idx):
        bad=np.flatnonzero(dm[:,e]>.5)
        if len(bad):
            begin=bad[0];last=bad[-1]+1;ax.broken_barh([(t[begin],(last-begin)*dt)],(row-.25,.5),facecolors='#F4B6A6',edgecolors=COLORS[1],hatch='////',linewidth=.45)
            ax.scatter(t[begin],row,marker='x',s=21,c=COLORS[1],zorder=4)
            done=next((ev['hour'] for ev in r['events'] if ev['line']==c['edges'][e]['name'] and '维修完成' in ev['kind']),None)
            if done is not None:ax.scatter(done,row,marker='o',s=17,c=COLORS[2],zorder=5)
        on=np.flatnonzero((energy[:,e]>.5)&(dm[:,e]<.5))
        if len(on):
            # 由连续真值片段画带电区段，不能用首尾连线跨过中间故障。
            blocks=np.split(on,np.flatnonzero(np.diff(on)>1)+1)
            for block in blocks:ax.plot([t[block[0]],t[block[-1]]+dt],[row,row],color=COLORS[0],lw=2.0)
        for ev in r['events']:
            if ev['line']==c['edges'][e]['name'] and ev['kind']=='主动合闸':ax.scatter(ev['hour'],row,marker='^',s=17,facecolor='white',edgecolor='.1',zorder=6)
    ax.set_yticks(range(len(idx)),[c['edges'][e]['switch_id']+' / '+_line(c,e) for e in idx]);ax.invert_yaxis();timeaxis(ax,c['horizon_hours']);ax.set_ylabel('发生故障或操作的线路')
    ax.legend(handles=[Patch(fc='#F4B6A6',ec=COLORS[1],hatch='////',label='故障未修复'),Line2D([],[],marker='o',color=COLORS[2],ls='',label='维修完成'),Line2D([],[],marker='^',mfc='white',color='.1',ls='',label='主动合闸'),Line2D([],[],color=COLORS[0],lw=2,label='实际带电区段')],ncol=4,loc='upper center',bbox_to_anchor=(.5,1.13),fontsize=6)
    files=save(fig,out,'02_线路故障修复与投入时序')
    operations=[e for e in r['events'] if e['kind'] in ('主动合闸','主动分闸')]
    fig=figure(max(2.,.14*len(operations)+1.));ax=fig.add_subplot()
    for j,ev in enumerate(operations):ax.scatter(ev['hour'],j,marker='^' if ev['kind']=='主动合闸' else 'v',s=24,c=COLORS[0] if ev['kind']=='主动合闸' else COLORS[1])
    ax.set_yticks(range(len(operations)),[e['line']+' '+('合' if e['kind']=='主动合闸' else '分') for e in operations]);ax.invert_yaxis();timeaxis(ax,c['horizon_hours']);panel(ax,'a','主动开关操作事件，不含保护跳闸')
    return files+save(fig,out,'02b_开关操作事件表')


def plot_dispatch(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];fig=figure(5.0);axes=fig.subplots(3,1,sharex=True)
    for key,label,col in [('upward_p','DSO向TSO净输出',COLORS[0]),('aux','启动辅机功率',COLORS[1]),('target_p','目标机组出力',COLORS[2])]:stairs(axes[0],t,np.asarray(r[key]),dt,label=label,color=col)
    axes[0].axhline(0,color='.3',lw=.4);axes[0].set_ylabel('有功 / MW');axes[0].legend(ncol=3,fontsize=6);panel(axes[0],'a','主配接口与机组')
    for j,node in enumerate(c['mg_nodes']):
        net=np.asarray(r['pcc_p'])[:,j];stairs(axes[1],t,net,dt,label=c['microgrids'][j]['name'],color=COLORS[j%len(COLORS)])
        axes[2].plot(np.arange(len(r['energy']))*dt,np.asarray(r['energy'])[:,j]/r['energy_capacity'][j],label=c['microgrids'][j]['name'],color=COLORS[j%len(COLORS)])
    axes[1].set_ylabel('PCC有功 / MW');panel(axes[1],'b','扣除本地供电负荷后的微网净输出');axes[1].legend(ncol=len(c['microgrids']),fontsize=6)
    axes[2].set_ylabel('SOC');panel(axes[2],'c','实际储能状态，非独立时段拼接');axes[2].legend(ncol=len(c['microgrids']),fontsize=6)
    for ax in axes:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'03_主配微功率与SOC')


def plot_recovery(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];sc=np.asarray(r['served_c']).sum(axis=1);sg=np.asarray(r['served_g']).sum(axis=1)
    dc=c['profiles']['critical'].sum(axis=0);dg=c['profiles']['general'].sum(axis=0)
    fig=figure(4.1);axes=fig.subplots(2,2)
    stairs(axes[0,0],t,dc+dg,dt,label='总需求',color='.45',ls='--');stairs(axes[0,0],t,sc+sg,dt,label='实际供电',color=COLORS[0]);axes[0,0].set_ylabel('有功 / MW');axes[0,0].legend();panel(axes[0,0],'a','供电与需求')
    stairs(axes[0,1],t,np.asarray(r['recovery'])*100,dt,label='加权恢复率',color=COLORS[0]);stairs(axes[0,1],t,sc/np.maximum(dc,1e-9)*100,dt,label='关键负荷恢复率',color=COLORS[1]);axes[0,1].set(ylim=(0,103),ylabel='恢复率 / %');axes[0,1].legend(fontsize=6);panel(axes[0,1],'b','全过程关键服务')
    for loss,label,color in [(dc-sc,'关键负荷',COLORS[1]),(dg-sg,'一般负荷',COLORS[0])]:axes[1,0].plot(np.arange(len(t)+1)*dt,np.r_[0,np.cumsum(loss*dt)],label=label,color=color)
    axes[1,0].set_ylabel('失供电量 / MWh');axes[1,0].legend();panel(axes[1,0],'c','失供电量分解')
    vv=np.asarray(r['voltage']);live=np.asarray(r['energized_bus'])>.5
    stairs(axes[1,1],t,np.min(np.where(live,vv,np.inf),axis=1),dt,label='最小',color=COLORS[0]);stairs(axes[1,1],t,np.max(np.where(live,vv,-np.inf),axis=1),dt,label='最大',color=COLORS[1]);axes[1,1].axhline(c['voltage_min'],ls=':',lw=.6,color='.3');axes[1,1].axhline(c['voltage_max'],ls=':',lw=.6,color='.3');axes[1,1].set_ylabel('电压 / p.u.');axes[1,1].legend();panel(axes[1,1],'d','仅统计带电节点电压')
    for ax in axes.flat:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'04_关键服务与韧性指标')


def plot_crew(c,r,out):
    trace=r['crew_trace'];dt=r['dt'];ids=list(dict.fromkeys(a['edge'] for a in trace if a is not None))
    fig=figure(max(3.,len(ids)*.24+1.3));axes=fig.subplots(2,1,height_ratios=[max(1,len(ids)*.24),1],sharex=True)
    palette={'travel':'.65','inspect':COLORS[0],'repair':COLORS[1],'wait':'white'}
    def key(a):
        if a is None:return None
        return (a['edge'],'wait' if a.get('weather_wait') else 'travel' if a['travel_remaining']>1e-8 else a['kind'])
    begin=0
    for i in range(1,len(trace)+1):
        if i<len(trace) and key(trace[i])==key(trace[begin]):continue
        value=key(trace[begin])
        if value:
            e,kind=value;row=ids.index(e);axes[0].broken_barh([(begin*dt,(i-begin)*dt)],(row-.3,.6),facecolors=palette[kind],edgecolors='.3',hatch='////' if kind=='wait' else None,linewidth=.4)
        begin=i
    for event in r['events']:
        if '巡检完成' in event['kind']:
            e=next(i for i,line in enumerate(c['edges']) if line['name']==event['line'])
            if e in ids:axes[0].scatter(event['hour'],ids.index(e),marker='D',s=18,c='black',zorder=6);axes[0].annotate('H' if event['damage']=='heavy' else 'L',(event['hour'],ids.index(e)),xytext=(3,4),textcoords='offset points',fontsize=6)
    axes[0].set_yticks(range(len(ids)),[_line(c,e) for e in ids]);axes[0].invert_yaxis();axes[0].set_ylabel('作业线路');panel(axes[0],'a','检查与修复甘特图，菱形表示类型获知')
    axes[0].legend(handles=[Patch(fc='.65',label='行程'),Patch(fc=COLORS[0],label='有效检查'),Patch(fc=COLORS[1],label='有效维修'),Patch(fc='white',ec='.3',hatch='////',label='天气等待')],ncol=4,fontsize=6,loc='upper center',bbox_to_anchor=(.5,1.25))
    known=np.array([sum('巡检完成' in e['kind'] and e['hour']<=h+1e-8 for e in r['events']) for h in r['time_hours']]);repaired=np.array([sum('维修完成' in e['kind'] and e['hour']<=h+1e-8 for e in r['events']) for h in r['time_hours']])
    stairs(axes[1],np.asarray(r['time_hours']),known,dt,label='已获知类型数',color=COLORS[0]);stairs(axes[1],np.asarray(r['time_hours']),repaired,dt,label='已修复数',color=COLORS[1]);axes[1].set_ylabel('线路数');axes[1].legend(ncol=2);panel(axes[1],'b','信息到达与物理修复分别统计')
    for ax in axes:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'05_检查信息与修复作业时序')


def plot_D(r,out):
    rows=[a for a in r['rounds'] if a.get('D_estimate') is not None]
    if not rows:return []
    fig=figure(3.4);axes=fig.subplots(2,1,height_ratios=[1.8,1])
    positions=np.arange(len(rows))
    for j,a in enumerate(rows):
        axes[0].plot([100*a['D_lower'],100*a['D_upper']],[j,j],color=COLORS[0],lw=1.1)
        axes[0].scatter(100*a['D_estimate'],j,s=16,c=COLORS[1],zorder=4)
    axes[0].axvline(0,color='.2',ls='--',lw=.6);axes[0].set_yticks(positions,[f"t={a['hour']:g} h" for a in rows]);axes[0].set_xlabel('D_R / 百分点');panel(axes[0],'a','有自由启动选择时的差值与有效区间')
    t=np.asarray(r['time_hours']);dt=r['dt'];on=np.asarray(r['unit_online']);aux=np.asarray(r['aux']);pg=np.asarray(r['target_p'])
    stage=np.where(aux>1e-8,1,np.where(on>.5,2,0));colors=['.85',COLORS[1],COLORS[2]]
    start=0
    for i in range(1,len(t)+1):
        if i<len(t) and stage[i]==stage[start]:continue
        axes[1].broken_barh([(t[start],(i-start)*dt)],(.2,.5),facecolors=colors[int(stage[start])],edgecolors='white',lw=.4);start=i
    axes[1].set(yticks=[],ylim=(0,1));timeaxis(axes[1],t[-1]+dt);panel(axes[1],'b','启动成为承诺后，不再比较当前启动与等待')
    axes[1].legend(handles=[Patch(fc=colors[j],label=label) for j,label in enumerate(['未启动','启动辅机阶段','已在线'])],ncol=3,fontsize=6,loc='lower center',bbox_to_anchor=(.5,-.1))
    return save(fig,out,'06_启动判据与机组承诺阶段')


def plot_capability(cap,out):
    if not cap:return []
    fig=figure(3.7);ax=fig.add_subplot(121,projection='3d');other=fig.add_subplot(122)
    allpoints=[]
    for j,sl in enumerate(cap['slices']):
        a=np.asarray(sl['points']);color=plt.cm.viridis(j/max(1,len(cap['slices'])-1))
        if len(a)>=3:
            try:
                hull=ConvexHull(a);q=a[hull.vertices];verts=np.c_[np.full(len(q),sl['hour']),q];allpoints.extend(verts)
                ax.add_collection3d(Poly3DCollection([verts],facecolor=color,edgecolor=color,alpha=.12,linewidths=.5));continue
            except QhullError:pass
        if len(a):ax.plot(np.full(len(a),sl['hour']),a[:,0],a[:,1],color=color,lw=.6)
    ax.plot(cap['time_hours'],cap['reference_p'],cap['reference_q'],color=COLORS[1],lw=1,label='同一完整轨迹')
    if allpoints:
        p=np.asarray(allpoints);ax.auto_scale_xyz(p[:,0],p[:,1],p[:,2])
    ax.set(xlabel='时间 / h',ylabel='P / MW',zlabel='Q / Mvar',xlim=(0,cap['horizon_hours']));ax.tick_params(labelsize=6);ax.legend(fontsize=6);panel(ax,'a','全时域条件资源域的截面内近似')
    energy=np.asarray(cap['decomposition']['energy']) if cap.get('decomposition') else None
    if energy is not None:
        for j in range(len(energy)):other.plot(np.linspace(0,cap['horizon_hours'],energy.shape[-1]),energy[j],label=f'MG{j+1}',color=COLORS[j%len(COLORS)])
    other.set(ylabel='储能 / MWh');other.yaxis.tick_right();other.yaxis.set_label_position('right')
    timeaxis(other,cap['horizon_hours']);other.legend(fontsize=6);panel(other,'b','整条接口轨迹的可实施分解')
    return save(fig,out,'07_多时段资源域与轨迹分解')


def plot_phase(phase,out):
    if not phase:return []
    xs=phase['alphas'];ys=phase['socs'];z=np.full((len(ys),len(xs)),np.nan);classes=np.zeros_like(z)
    codes={'unknown':0,'start':1,'wait':2,'recovery_tie':3,'start_infeasible':4,'wait_infeasible':5}
    for r in phase['records']:
        i=ys.index(r['soc']);j=xs.index(r['alpha']);classes[i,j]=codes.get(r['classification'],0)
        if r['D_estimate'] is not None:z[i,j]=r['D_estimate']*100
    fig=figure(3.0);axes=fig.subplots(1,2)
    finite=z[np.isfinite(z)];lim=max(.01,float(np.max(np.abs(finite)))) if len(finite) else 1
    im=axes[0].imshow(z,origin='lower',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
    fig.colorbar(im,ax=axes[0],label='计划差值 / 百分点',shrink=.8)
    cmap=ListedColormap(['#e0e0e0',COLORS[2],COLORS[0],'#f0d380','#484848','#b66a50']);axes[1].imshow(classes,origin='lower',cmap=cmap,vmin=-.5,vmax=5.5,aspect='auto')
    names=['未判','启动','等待','近等效','启动不可行','等待不可行']
    for i in range(len(ys)):
        for j in range(len(xs)):
            axes[0].text(j,i,'NA' if not np.isfinite(z[i,j]) else f'{z[i,j]:.3g}',ha='center',va='center',fontsize=6.5,color='white' if np.isfinite(z[i,j]) and abs(z[i,j])>lim*.65 else 'black')
            axes[1].text(j,i,names[int(classes[i,j])],ha='center',va='center',fontsize=6.5,color='white' if classes[i,j]==4 else 'black')
    for ax in axes:ax.set(xticks=range(len(xs)),xticklabels=[f'{x:g}' for x in xs],yticks=range(len(ys)),yticklabels=[f'{y:g}' for y in ys],xlabel='主网剩余比例 alpha',ylabel='初始SOC')
    panel(axes[0],'a','真实采样点的计划差值，不补画曲率');panel(axes[1],'b','有限策略类最优值界给出的判据')
    return save(fig,out,'08_D值网格与决策证据')


def plot_comparison(results,out):
    if len(results)<2:return []
    fig=figure(2.7);axes=fig.subplots(1,2);labels={'joint':'联合策略类','critical_first':'关键负荷规则','nearest':'就近规则','no_upward':'禁止反送'}
    for j,r in enumerate(results):stairs(axes[0],np.asarray(r['time_hours']),np.asarray(r['recovery'])*100,r['dt'],label=labels[r['policy']],color=COLORS[j])
    timeaxis(axes[0],results[0]['time_hours'][-1]+results[0]['dt']);axes[0].set_ylabel('加权恢复率 / %');axes[0].legend(fontsize=6);panel(axes[0],'a','同一灾害与相同资源')
    x=np.arange(len(results));bars=axes[1].bar(x,[r['summary']['actual_EENS_MWh'] for r in results],color=COLORS[:len(results)],width=.6)
    for j,b in enumerate(bars):axes[1].text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height():.2f}',ha='center',va='bottom',fontsize=7)
    axes[1].set(xticks=x,xticklabels=[labels[r['policy']] for r in results],ylabel='失供电量 / MWh');panel(axes[1],'b','实际失供电量，非预设协同增益')
    return save(fig,out,'09_策略恢复与失供电对照')


def plot_network(c,r,out):
    hours=[float(x) for x in str(c.get('options',{}).get('plots.snapshot_hours','0,4,12,23.75')).split(',')]
    fig=figure(4.6);axes=fig.subplots(int(np.ceil(len(hours)/2)),2,squeeze=False)
    served=np.asarray(r['served_c'])+np.asarray(r['served_g']);demand=(c['profiles']['critical']+c['profiles']['general']).T
    for ax,h in zip(axes.flat,hours):
        t=min(c['n_steps']-1,max(0,int(round(h/c['dt']))));fraction=np.divide(served[t],demand[t],out=np.full_like(served[t],np.nan),where=demand[t]>1e-9)
        _network(ax,c,r['closed'][t],r['damaged'][t],r['energized_bus'][t],fraction)
        ax.set_title(f't={r["time_hours"][t]:g} h',loc='left')
    for ax in list(axes.flat)[len(hours):]:ax.axis('off')
    handles=[Line2D([],[],marker='o',color='.1',mfc=col,ls='',label=lab) for col,lab in [(COLORS[2],'负荷全部恢复'),('#E69F00','部分恢复'),('white','带电但未恢复'),('#BDBDBD','失电')]]
    handles +=[Line2D([],[],marker='o',mfc='#C5D5E3',color='.1',ls='',label='无本地负荷'),Line2D([],[],color=COLORS[1],marker='x',ls='--',label='故障线路')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=6)
    return save(fig,out,'10_节点服务状态与故障拓扑')


def plot_information(info,out):
    if not info:return []
    plan=info['crew_plan'];hist=plan['histories'];ns=len(hist);n=len(hist[0]);ids=np.zeros((ns,n));nextid=0;mapping={}
    for t in range(n):
        local={}
        for s in range(ns):
            key=json.dumps(hist[s][t],sort_keys=True)
            if key not in local:local[key]=len(local)
            ids[s,t]=local[key]
    fig=figure(2.7);ax=fig.add_subplot();horizon=info.get('horizon_hours',24)
    ax.imshow(ids,aspect='auto',origin='lower',interpolation='nearest',extent=(0,horizon,-.5,ns-.5),cmap='tab20')
    ax.set_yticks(range(ns),[f'样本{s+1}' for s in range(ns)]);timeaxis(ax,horizon)
    panel(ax,'a','同一时刻同色表示相同观测历史，组内控制一致')
    for s,events in enumerate(plan['reveals']):
        for ev in events:ax.scatter(ev['hour'],s,marker='D',s=14,c='black',edgecolors='white',linewidths=.3)
    return save(fig,out,'11_检查触发的信息分组')


def plot_backward(c,info,out):
    if not info or not info.get('backward_certificate'):return []
    cert=info['backward_certificate'];low=np.asarray(cert['lower'])[0];high=np.asarray(cert['upper'])[0];reserve=info.get('reserve_tube')
    nm=len(c['microgrids']);fig=figure(max(2.5,.9*nm));axes=fig.subplots(nm,1,squeeze=False)
    times=np.linspace(0,c['horizon_hours'],low.shape[1])
    for j,ax in enumerate(axes.flat):
        cap=c['microgrids'][j]['energy_mwh'];ax.fill_between(times,low[j]/cap,high[j]/cap,color=COLORS[0],alpha=.2,label='条件后向可达管')
        if reserve:ax.plot(times,np.asarray(reserve['lower'])[j]/cap,color='.25',ls='--',label='岛内安全后备下界')
        ax.scatter(0,c['microgrids'][j]['soc_initial'],s=18,c=COLORS[1],zorder=5,label='当前状态')
        if info.get('reference_energy') is not None:ax.plot(times,np.asarray(info['reference_energy'])[0,j]/cap,color=COLORS[0],lw=.8,label='同一候选轨迹')
        if cert.get('takeover_hour') is not None:ax.axvline(cert['takeover_hour'],color=COLORS[2],ls=':',lw=.8)
        ax.set_ylabel(c['microgrids'][j]['name']+' SOC');timeaxis(ax,c['horizon_hours'])
    axes[0,0].legend(ncol=3,fontsize=6);panel(axes[0,0],'a','从终端集合向前驱状态递推；不反转物理时间')
    return save(fig,out,'12_条件后向可达管与安全后备集')


def plot_rolling(c,r,out):
    steps=r.get('controller_steps',[])
    if not steps:return []
    chosen=np.unique(np.linspace(0,len(steps)-1,min(6,len(steps))).astype(int));fig=figure(3.1);axes=fig.subplots(2,1,height_ratios=[2,1],sharex=True)
    for row,k in enumerate(chosen):
        h=steps[k]['hour'];axes[0].broken_barh([(0,h)],(row-.3,.6),facecolors='.85');axes[0].broken_barh([(h,c['horizon_hours']-h)],(row-.3,.6),facecolors='#d5e8f2');axes[0].broken_barh([(h,c['dt'])],(row-.3,.6),facecolors=COLORS[1])
    axes[0].set_yticks(range(len(chosen)),[f"t={steps[k]['hour']:g}" for k in chosen]);axes[0].invert_yaxis();panel(axes[0],'a','固定24h终点：已执行前缀不变，只下发当前一步')
    for step in steps:axes[1].scatter(step['hour'],1 if step['solve']=='joint_MILP' else 0,marker='|' if step['solve']=='fixed_plan_LP' else 'o',s=14,c=COLORS[0] if step['solve']=='joint_MILP' else '.4')
    axes[1].set(yticks=[0,1],yticklabels=['连续LP更新','联合MILP重算'],ylim=(-.5,1.5));panel(axes[1],'b','每一步均求解与校核，离散周期及观测事件触发联合重算')
    for ax in axes:timeaxis(ax,c['horizon_hours'])
    return save(fig,out,'13_实际首步执行与固定终点滚动')


def plot_trajectory_hull(cap,out):
    if not cap or 'trajectory_inner_hull' not in cap:return []
    fig=figure(2.6);ax=fig.add_subplot();t=np.asarray(cap['time_hours'])
    ax.plot(t,cap['reference_p'],color=COLORS[0],label='完整可分解参考轨迹')
    test=cap.get('splicing_counterexample',{})
    if test.get('target_p') is not None:ax.plot(t,test['target_p'],color=COLORS[1],ls='--',label='逐时选择接口点后的拼接轨迹')
    text={'infeasible':'拼接结果：不可行','feasible':'拼接结果：本次可行','unknown':'拼接结果：未判','not_tested':'尚未检验'}[test.get('status','not_tested')]
    ax.text(.02,.95,text,transform=ax.transAxes,va='top',fontsize=7)
    timeaxis(ax,cap['horizon_hours']);ax.set_ylabel('微网PCC有功总和 / MW');ax.legend(loc='lower right',fontsize=6)
    panel(ax,'a','共同凸组合系数保持全时域可行；逐时拼接另作检验')
    return save(fig,out,'14_完整轨迹内近似与逐时拼接检验')


def plot_phase_wireframe(c,phase,out):
    if not phase or not c.get('options',{}).get('plots.phase_surface',False):return []
    x,y=np.meshgrid(phase['alphas'],phase['socs']);z=np.full_like(x,np.nan,dtype=float)
    for row in phase['records']:
        if row['D_estimate'] is not None:z[phase['socs'].index(row['soc']),phase['alphas'].index(row['alpha'])]=100*row['D_estimate']
    fig=figure(3.2);ax=fig.add_subplot(projection='3d');ax.scatter(x,y,z,s=15,color=COLORS[0]);ax.plot_wireframe(x,y,z,color='.4',linewidth=.6)
    ax.set(xlabel='主网剩余比例 alpha',ylabel='初始SOC',zlabel='D_R / 百分点');panel(ax,'a','实际采样点及线框，不补造曲率或临界面')
    return save(fig,out,'08b_D采样点与分段线框')


def plot_all(c,world,result,out,cap=None,phase=None,comparisons=None,information=None):
    setup(c);files=[]
    calls=[(plot_layout,(c,world,out)),(plot_weather,(c,world,out)),(plot_faults,(c,result,out)),(plot_dispatch,(c,result,out)),(plot_recovery,(c,result,out)),(plot_crew,(c,result,out)),(plot_D,(result,out)),(plot_capability,(cap,out)),(plot_phase,(phase,out)),(plot_comparison,(comparisons or [result],out)),(plot_network,(c,result,out)),(plot_information,(information,out)),(plot_backward,(c,information,out)),(plot_rolling,(c,result,out)),(plot_trajectory_hull,(cap,out)),(plot_phase_wireframe,(c,phase,out))]
    for fn,args in calls:files.extend(fn(*args))
    Path(out,'figures_manifest.json').write_text(json.dumps({'files':files,'scope':'实际执行、预测条件域和有限策略类判据分别标识；不补造曲率或临界线'},ensure_ascii=False,indent=2),encoding='utf-8')
    return files
