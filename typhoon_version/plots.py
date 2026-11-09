"""只根据真实输入、优化结果和实际执行记录出图；没有预画的D边界。"""
from pathlib import Path
import os,tempfile,json
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tdm_matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'tdm_font_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch,FancyArrowPatch,Circle
from matplotlib.colors import ListedColormap,TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull,QhullError

COLORS=['#1767a5','#149b89','#dd922f','#8a61a6'];INK='#17374d'

def style():
    p=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if p.exists():font_manager.fontManager.addfont(str(p));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(p)).get_name()
    else:plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'axes.titleweight':'bold','legend.frameon':False})

def save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in ('png','svg','pdf'):
        p=out/f'{name}.{suffix}';fig.savefig(p,dpi=180,bbox_inches='tight');paths.append(str(p))
    plt.close(fig);return paths

def stairs(ax,t,y,dt,**kw):return ax.step(np.r_[t,t[-1]+dt],np.r_[y,y[-1]],where='post',**kw)

def timeaxis(ax):ax.set(xlim=(0,24),xticks=np.arange(0,25,4),xlabel='灾后时间 / h');ax.grid(alpha=.18)

def plot_weather(c,world,out):
    style();fig,axes=plt.subplots(2,2,figsize=(15,9));xy=c['geo_xy']
    for ax,h in zip(axes.flat,[0,4,8,12]):
        t=min(c['n_steps']-1,round(h/c['dt']));wind=world['wind'][:,t]
        for e,line in enumerate(c['edges']):
            a,b=xy[line['from']],xy[line['to']];col=plt.cm.YlOrRd(np.clip(wind[e]/50,0,1))
            ax.plot([a[0],b[0]],[a[1],b[1]],color=col,lw=2,ls='--' if line['kind']=='tie' else '-')
            if world['failure_hour'][e]<=h:ax.scatter(*(a+b)/2,s=34,marker='x',color='#a22a36',zorder=5)
        ax.scatter(xy[:,0],xy[:,1],c='#527385',s=12,zorder=3)
        tr=c['track'];tt=[r['time_hour'] for r in tr];cx=np.interp(h,tt,[r['center_x_km'] for r in tr]);cy=np.interp(h,tt,[r['center_y_km'] for r in tr])
        ax.scatter(cx,cy,s=70,marker='*',color='#1b4a76');ax.add_patch(Circle((cx,cy),np.interp(h,tt,[r['radius_max_km'] for r in tr]),fill=False,ls=':',color='#597d9a',alpha=.6))
        ax.set(title=f't={h:g} h｜已发生故障{sum(world["failure_hour"]<=h)}处',xlabel='独立合成地理X / km',ylabel='独立合成地理Y / km');ax.grid(alpha=.15)
    fig.suptitle('台风移动风场与动态故障发生｜红×表示曾经故障，修复状态另见图02',fontsize=16,fontweight='bold')
    fig.text(.5,.015,'输入为移动Rankine型简化风场及合成地理/危险率参数；不是实测台风。虚线为联络线，线色越暖风速越大。',ha='center',fontsize=10,color='#667d8d')
    fig.tight_layout(rect=(0,.04,1,.95));return save(fig,out,'01_台风移动与动态故障')

def plot_faults(c,r,out):
    t=np.asarray(r['time_hours']);dm=np.asarray(r['damaged']).T;closed=np.asarray(r['closed']).T;idx=[e for e,line in enumerate(c['edges']) if line['exposed']]
    fig,axes=plt.subplots(1,2,figsize=(16,13),sharey=True)
    im=axes[0].imshow(dm[idx],aspect='auto',interpolation='nearest',extent=(0,24,len(idx)-.5,-.5),cmap=ListedColormap(['#e1edf2','#c94c56']),vmin=0,vmax=1)
    axes[1].imshow(closed[idx],aspect='auto',interpolation='nearest',extent=(0,24,len(idx)-.5,-.5),cmap=ListedColormap(['#f0f3f5','#245f8c']),vmin=0,vmax=1)
    axes[0].set_yticks(range(len(idx)),[c['edges'][e]['switch_id']+'｜'+c['edges'][e]['name'] for e in idx],fontsize=8)
    axes[0].set_title('物理状态：红色故障未修复，浅色可用');axes[1].set_title('开关状态：深色闭合，浅色断开')
    for ax in axes:timeaxis(ax)
    fig.suptitle('故障可用性与开关决定分开记录｜修好不等于自动合闸',fontsize=17,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.96));return save(fig,out,'02_故障修复与开关96时段')

def plot_dispatch(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True)
    for key,label,col in [('upward_p','DSO向TSO净上送',COLORS[0]),('aux','目标机组辅机需求',COLORS[2]),('target_p','目标机组发电',COLORS[1])]:stairs(axes[0],t,np.asarray(r[key]),dt,label=label,color=col,lw=1.8)
    axes[0].axhline(0,color='#718697',lw=.7);axes[0].set(ylabel='功率 / MW',title='主配接口与机组：正值向上送，负值由主网向配网送电');axes[0].legend(ncol=3)
    mg=np.asarray(r['pv'])+np.asarray(r['dg'])+np.asarray(r['discharge'])-np.asarray(r['charge'])
    for j,node in enumerate(c['mg_nodes']):
        net=mg[:,j]-np.asarray(r['served_c'])[:,node]-np.asarray(r['served_g'])[:,node]
        stairs(axes[1],t,net,dt,label=f'MG{j+1} PCC净输出',color=COLORS[j],lw=1.6)
        axes[2].plot(np.arange(len(r['energy']))*dt,np.asarray(r['energy'])[:,j]/r['energy_capacity'][j],label=f'MG{j+1}',color=COLORS[j],lw=1.6)
    axes[1].set(ylabel='功率 / MW',title='微网PCC净输出：资源出力扣除本地供电负荷');axes[1].legend(ncol=3)
    axes[2].set(ylabel='SOC',title='继承真实SOC，终点保持原24小时');axes[2].legend(ncol=3)
    for ax in axes:timeaxis(ax)
    fig.suptitle('24小时主配微协同轨迹与储能状态｜实际执行',fontsize=17,fontweight='bold');fig.tight_layout(rect=(0,0,1,.96));return save(fig,out,'03_主配微24小时功率与SOC')

def plot_recovery(c,r,out):
    t=np.asarray(r['time_hours']);dt=r['dt'];sc=np.asarray(r['served_c']).sum(axis=1);sg=np.asarray(r['served_g']).sum(axis=1)
    dc=c['profiles']['critical'].sum(axis=0);dg=c['profiles']['general'].sum(axis=0);fig,axes=plt.subplots(2,2,figsize=(14,9))
    stairs(axes[0,0],t,dc+dg,dt,label='总需求',color='#8b9ba6',ls='--');stairs(axes[0,0],t,sc+sg,dt,label='实际供电',color=COLORS[0],lw=2);axes[0,0].set(ylabel='MW',title='恢复供电与需求');axes[0,0].legend()
    stairs(axes[0,1],t,np.asarray(r['recovery'])*100,dt,label='加权恢复率',color=COLORS[1],lw=2)
    stairs(axes[0,1],t,sc/np.maximum(dc,1e-9)*100,dt,label='关键负荷恢复率',color=COLORS[2]);axes[0,1].set(ylim=(0,103),ylabel='%',title='主目标覆盖全过程，不仅末时刻');axes[0,1].legend()
    for a,label,col in [(dc-sc,'关键失供',COLORS[2]),(dg-sg,'一般失供',COLORS[0])]:axes[1,0].plot(np.arange(len(t)+1)*dt,np.r_[0,np.cumsum(a*dt)],label=label,color=col,lw=2)
    axes[1,0].set(title='累计失供电量',ylabel='MWh');axes[1,0].legend()
    vv=np.asarray(r['voltage']);live=np.asarray(r['energized_bus'])>.5;low=np.min(np.where(live,vv,np.inf),axis=1);high=np.max(np.where(live,vv,-np.inf),axis=1)
    stairs(axes[1,1],t,low,dt,label='带电节点最小电压');stairs(axes[1,1],t,high,dt,label='带电节点最大电压');axes[1,1].set(title='仅统计带电节点电压',ylabel='pu');axes[1,1].legend()
    for ax in axes.flat:timeaxis(ax)
    fig.suptitle(f'韧性实际结果：加权恢复面积比 {100*r["summary"]["actual_weighted_recovery"]:.2f}%｜EENS {r["summary"]["actual_EENS_MWh"]:.3f} MWh',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.95));return save(fig,out,'04_恢复率优先与韧性指标')

def plot_crew(c,r,out):
    fig,axes=plt.subplots(2,1,figsize=(15,7),sharex=True,gridspec_kw={'height_ratios':[1.2,1]});trace=r['crew_trace'];dt=r['dt'];start=0
    colors={'travel':'#879eae','inspect':'#dc9c42','repair':'#228b82','weather_wait':'#c7c9d3','idle':'#eff2f4'}
    def key(item):return ('idle',-1) if item is None else ('travel' if item['travel_remaining']>1e-8 else 'weather_wait' if item.get('weather_wait') else item['kind'],item['edge'])
    for t in range(1,len(trace)+1):
        if t<len(trace) and key(trace[t])==key(trace[start]):continue
        kind,e=key(trace[start]);axes[0].broken_barh([(start*dt,(t-start)*dt)],(.15,.6),facecolors=colors[kind],edgecolor='white')
        if e>=0 and t-start>=2:axes[0].text((start+t)*dt/2,.45,c['edges'][e]['name'],ha='center',va='center',rotation=45,fontsize=8)
        start=t
    axes[0].set(ylim=(0,1),yticks=[],title='真实行程—巡检—维修；不安全风速下保留进度并等待')
    axes[0].legend(handles=[Patch(color=v,label=k) for k,v in [('行程',colors['travel']),('有效巡检',colors['inspect']),('有效维修',colors['repair']),('天气等待',colors['weather_wait'])]],ncol=4,loc='upper center',bbox_to_anchor=(.5,1.38))
    for ev in r['events']:
        if '巡检完成' in ev['kind']:axes[1].scatter(ev['hour'],1,marker='v',color=COLORS[2],s=45)
        if '维修完成' in ev['kind']:axes[1].scatter(ev['hour'],.5,marker='o',color=COLORS[1],s=35)
    axes[1].set(ylim=(.1,1.4),yticks=[.5,1],yticklabels=['维修完成/线路可用','巡检完成/损伤揭示'],title='信息揭示与物理修复是不同事件')
    for ax in axes:timeaxis(ax)
    fig.suptitle('信息驱动的抢修执行与任务优先顺序',fontsize=17,fontweight='bold');fig.tight_layout(rect=(0,0,1,.91));return save(fig,out,'05_巡检抢修与观测时间线')

def plot_D(r,out):
    rows=[a for a in r['rounds'] if a.get('D_estimate') is not None]
    if not rows:return []
    fig,ax=plt.subplots(figsize=(13,5))
    for a in rows:
        h=a['hour'];lo=a['D_lower']*100;hi=a['D_upper']*100;d=a['D_estimate']*100;col=COLORS[1] if a['classification']=='start' else COLORS[0] if a['classification']=='wait' else '#cb9a3f'
        ax.plot([h,h],[lo,hi],color=col,lw=2,alpha=.6);ax.scatter(h,d,color=col,s=35)
    ax.axhline(0,ls='--',color='#768a98');ax.set(ylabel='D_R / 百分点',title='点为两个可行计划恢复率差；竖线为当前预测模型D_R界。跨零仍未判定。');timeaxis(ax)
    fig.suptitle('启动／等待边界：实际滚动时刻的证据',fontsize=16,fontweight='bold');fig.tight_layout(rect=(0,0,1,.94));return save(fig,out,'06_滚动D_R与求解界')

def plot_capability(cap,out):
    if not cap:return []
    fig=plt.figure(figsize=(16,8));ax=fig.add_subplot(121,projection='3d');energyax=fig.add_subplot(222);deliveryax=fig.add_subplot(224)
    widths=[];hours=[]
    for j,sl in enumerate(cap['slices']):
        a=np.asarray(sl['points']);color=plt.cm.viridis(j/max(1,len(cap['slices'])-1));hours.append(sl['hour'])
        widths.append([a[:,0].min(),a[:,0].max()] if len(a) else [np.nan,np.nan])
        if len(a)>=3:
            try:
                hull=ConvexHull(a);q=a[hull.vertices];poly=np.c_[np.full(len(q),sl['hour']),q]
                ax.add_collection3d(Poly3DCollection([poly],facecolor=color,edgecolor=color,alpha=.22));continue
            except QhullError:pass
        if len(a):ax.plot(np.full(len(a),sl['hour']),a[:,0],a[:,1],color=color,lw=1)
    ax.plot(cap['time_hours'],cap['reference_p'],cap['reference_q'],color='#bd4657',lw=2,label='同一整轨迹的MG净输出总和')
    ax.set(xlabel='时间 / h',ylabel='Σ微网PCC有功 / MW',zlabel='Σ微网PCC无功 / Mvar',xlim=(0,24));ax.legend(fontsize=8)
    if cap['decomposition']:
        d=cap['decomposition'];energy=np.asarray(d['energy'])
        for j in range(3):energyax.plot(np.linspace(0,cap['horizon_hours'],energy.shape[-1]),energy[j],label=f'MG{j+1}',color=COLORS[j])
        energyax.set(xlabel='时间 / h',ylabel='储能 / MWh',title='给定整条PCC轨迹的本地分解');energyax.legend(ncol=3,fontsize=8);energyax.grid(alpha=.2)
    widths=np.asarray(widths);deliveryax.fill_between(hours,widths[:,0],widths[:,1],color=COLORS[1],alpha=.18,label='Σ微网PCC净输出范围')
    bounds=cap.get('delivery_bounds',[])
    if bounds:
        hh=[q['hour'] for q in bounds];lo=[np.nan if q['lower'] is None else q['lower'] for q in bounds];hi=[np.nan if q['upper'] is None else q['upper'] for q in bounds]
        deliveryax.plot(hh,lo,color='#b84955',ls='--',label='主配接口净上送范围');deliveryax.plot(hh,hi,color='#b84955',ls='--')
    deliveryax.set(xlim=(0,24),xlabel='时间 / h',ylabel='净有功 / MW',title='不同接口：微网资源与主配可送范围');deliveryax.legend(fontsize=8);deliveryax.grid(alpha=.2)
    fig.suptitle(f'多时段微网条件资源域内近似｜{cap["n_steps"]}时段、{cap["scenario_count"]}预测样本；固定充放电模式',fontsize=14,fontweight='bold')
    fig.text(.5,.025,'各截面不能任意拼接。右下绿色为Σ微网PCC，红线为主配接口（另含配网负荷与网络约束），并非同一接口的包含关系。实际执行见图03。',ha='center',fontsize=9,color='#687f91')
    fig.tight_layout(rect=(0,.07,1,.93));return save(fig,out,'07_24小时能力域与分解证书')

def plot_phase(phase,out):
    if not phase:return []
    xs=phase['alphas'];ys=phase['socs'];z=np.full((len(ys),len(xs)),np.nan);cls=np.full_like(z,3);codes={'start':0,'wait':1,'recovery_tie':2,'unknown':3,'start_infeasible':4,'wait_infeasible':5}
    for r in phase['records']:
        i=ys.index(r['soc']);j=xs.index(r['alpha']);cls[i,j]=codes.get(r['classification'],3)
        if r['D_estimate'] is not None:z[i,j]=100*r['D_estimate']
    x,y=np.meshgrid(xs,ys);fig=plt.figure(figsize=(18,6.5));ax=fig.add_subplot(131,projection='3d');top=fig.add_subplot(132);cert=fig.add_subplot(133)
    finite=z[np.isfinite(z)];lim=max(.01,float(np.max(np.abs(finite))) if len(finite) else 1.);norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    if np.isfinite(z).sum()>=3:ax.plot_surface(x,y,z,cmap='BrBG',norm=norm,edgecolor='#ffffff77',linewidth=.4)
    else:ax.text2D(.1,.5,'有效网格不足，未补画曲面',transform=ax.transAxes)
    ax.set(xlabel='主网剩余α',ylabel='初始SOC',zlabel='可行计划D_R差 / 百分点',title='两个可行计划的差值曲面')
    mesh=top.pcolormesh(x,y,np.ma.masked_invalid(z),shading='nearest',cmap='BrBG',norm=norm);fig.colorbar(mesh,ax=top,fraction=.05,pad=.02)
    # 未判网格不画临界线；插值过零不构成临界判据。
    if np.any(cls==0) and np.any(cls==1) and len(finite) and finite.min()<0<finite.max():
        top.contour(x,y,np.ma.masked_invalid(z),levels=[0],colors=INK,linewidths=1.5)
    top.set(xlabel='主网剩余α',ylabel='初始SOC',title='同一数值网格俯视图（不是认证分区）')
    cmap=ListedColormap(['#2c9677','#3971a4','#a587b8','#e8c563','#bdc6cc','#de8c7e']);cert.pcolormesh(x,y,cls,shading='nearest',cmap=cmap,vmin=-.5,vmax=5.5,edgecolor='white')
    for i in range(len(ys)):
        for j in range(len(xs)):
            xx=xs[j]+(.06 if j==0 else -.06 if j==len(xs)-1 else 0)*(max(xs)-min(xs))
            yy=ys[i]+(.04 if i==0 else -.04 if i==len(ys)-1 else 0)*(max(ys)-min(ys))
            cert.text(xx,yy,['启','等','平','未判','启不可行','等不可行'][int(cls[i,j])],ha='center',va='center',fontsize=9)
    cert.set(xlabel='主网剩余α',ylabel='初始SOC',title='由上下界给出的决策证据')
    for pane in (top,cert):pane.set(xlim=(min(xs),max(xs)),ylim=(min(ys),max(ys)),xticks=xs,yticks=ys)
    count=sum(bool(q.get('certified')) for q in phase['records'])
    fig.suptitle(f'D_R相图：{count}/{len(phase["records"])}格已判定｜曲面为可行计划差值，非已证临界面',fontsize=16,fontweight='bold')
    fig.text(.5,.03,'固定当前观测、真实初始故障和同一预测随机数。黄色是求解界未分离；灰色仅表示当前预测模型已证启动不可行。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.93));return save(fig,out,'08_D_R曲面与认证决策相图')

def plot_comparison(results,out):
    if len(results)<2:return []
    fig,axes=plt.subplots(1,3,figsize=(16,5));labels={'joint':'联合滚动','critical_first':'重要负荷规则','nearest':'就近规则','no_upward':'禁止向上支撑'}
    for j,r in enumerate(results):stairs(axes[0],np.asarray(r['time_hours']),np.asarray(r['recovery'])*100,r['dt'],label=labels[r['policy']],color=COLORS[j],lw=1.8)
    timeaxis(axes[0]);axes[0].set(ylabel='加权恢复率%',title='同一真实台风样本');axes[0].legend()
    names=[labels[r['policy']] for r in results]
    for ax,key,title in [(axes[1],'actual_EENS_MWh','实际EENS / MWh'),(axes[2],'manual_switch_operations','主动开关操作次数')]:
        bars=ax.bar(names,[r['summary'][key] for r in results],color=COLORS[:len(results)])
        for b,r in zip(bars,results):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{r["summary"][key]:.2f}',ha='center',va='bottom')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('抢修规则与向上支撑权限的对照｜资源与实际灾害相同',fontsize=16,fontweight='bold');fig.tight_layout(rect=(0,0,1,.93));return save(fig,out,'09_策略对照与韧性收益')

def plot_network(c,r,out):
    xy=c['plot_xy'];fig,axes=plt.subplots(2,2,figsize=(21,11));served=np.asarray(r['served_c'])+np.asarray(r['served_g']);demand=(c['profiles']['critical']+c['profiles']['general']).T
    for ax,h in zip(axes.flat,[0,4,12,23.75]):
        t=min(c['n_steps']-1,round(h/c['dt']));closed=np.asarray(r['closed'])[t];damaged=np.asarray(r['damaged'])[t];flows=np.asarray(r['flow_p'])[t]
        for e,line in enumerate(c['edges']):
            p,q=xy[line['from']],xy[line['to']];color='#c74a57' if damaged[e] else '#3b687e' if closed[e] else '#b7c5ce';ls='-' if closed[e] and not damaged[e] else '--'
            if line['kind']=='tie':
                route=np.asarray(line.get('plot_route',[]),float).reshape(-1,2)
                path=np.vstack([p,route,q]);ax.plot(path[:,0],path[:,1],color=color,ls=ls,lw=1.1,zorder=1)
            else:ax.plot([p[0],q[0]],[p[1],q[1]],color=color,ls=ls,lw=1.4,zorder=1)
            if abs(flows[e])>.25 and line['kind']!='tie':
                a=p*.65+q*.35;b=p*.35+q*.65
                if flows[e]<0:a,b=b,a
                ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#d38b36',lw=1.4))
        fraction=np.divide(served[t],demand[t],out=np.zeros_like(served[t]),where=demand[t]>1e-9)
        col=plt.cm.YlGn(.2+.75*fraction);col[demand[t]<=1e-9]=matplotlib.colors.to_rgba('#c6d0d7')
        ax.scatter(xy[:,0],xy[:,1],s=90,c=col,edgecolor=INK,lw=.6,zorder=3)
        for i,(x,y) in enumerate(xy):ax.text(x,y,c['nodes'][i][1:] if i>=2 else c['nodes'][i],ha='center',va='center',fontsize=7,zorder=4)
        for j,node in enumerate(c['mg_nodes']):ax.scatter(*xy[node],s=260,facecolor='none',edgecolor=COLORS[j],lw=2,zorder=5);ax.text(xy[node,0],xy[node,1]-.65,f'MG{j+1}',color=COLORS[j],fontsize=9,ha='center')
        ax.set_title(f't={h:g}h｜G'+('在线' if r['unit_online'][t]>.5 else '未在线')+f'｜未修{int(damaged.sum())}处',loc='left');ax.set_aspect('equal');ax.axis('off')
    fig.suptitle('33节点主配微的实际恢复过程｜箭头为实际有功流向',fontsize=17,fontweight='bold')
    fig.text(.5,.02,'T0/T1为主网等值，MG接18/25/33。绿色仅表示负荷恢复比例；无负荷节点灰色。红虚线故障，其他虚线为打开线路。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95));return save(fig,out,'10_33节点主配微实际拓扑快照')

def plot_layout(c,world,out):
    """动作前的参数接线：独立标明三层位置、全部开关和初始风灾故障。"""
    from matplotlib.patches import FancyBboxPatch
    style();fig,ax=plt.subplots(figsize=(20,7.5));xy=c['plot_xy'];failed=world['failure_hour']<=0
    ax.add_patch(FancyBboxPatch((-3.3,-.9),3.3,2.1,boxstyle='round,pad=.15',facecolor='#edf3fa',edgecolor='#aec1d6',zorder=0))
    ax.text(-1.7,.8,'TSO / 主网等值',ha='center',fontsize=10,color=INK)
    ax.text(-2.8,-.65,'G＋剩余电源α',ha='left',fontsize=9);ax.text(-.2,-.65,'TX主配接口',ha='center',fontsize=9)
    ax.text(7,4.0,'DSO / 33节点配电网',ha='center',fontsize=14,fontweight='bold')
    for e,line in enumerate(c['edges']):
        p,q=xy[line['from']],xy[line['to']];mid=(p+q)/2;col='#c54b59' if failed[e] else '#8c6dac' if line['kind']=='tie' else '#4b6d82'
        opened=failed[e] or line['kind']=='tie'
        if line['kind']=='tie':
            route=np.asarray(line.get('plot_route',[]),float).reshape(-1,2);path=np.vstack([p,route,q])
            ax.plot(path[:,0],path[:,1],color=col,lw=1.4,ls='--');mid=path[len(path)//2]
        else:ax.plot([p[0],q[0]],[p[1],q[1]],color=col,lw=1.5,ls='--' if opened else '-')
        if line['switchable']:
            ax.scatter(*mid,marker='s',s=22,facecolor='white' if opened else col,edgecolor=col,lw=.8,zorder=3)
            ax.text(mid[0],mid[1]-.25,line['switch_id'],ha='center',va='top',fontsize=7,color=col,
                bbox=dict(facecolor='white',edgecolor='none',pad=.15),zorder=4)
    ax.scatter(xy[:,0],xy[:,1],s=100,facecolor='#f0f5f8',edgecolor=INK,lw=.8,zorder=5)
    for i,(x,y) in enumerate(xy):ax.text(x,y,c['nodes'][i][1:] if i>=2 else c['nodes'][i],ha='center',va='center',fontsize=8,zorder=6)
    for j,node in enumerate(c['mg_nodes']):
        ax.scatter(*xy[node],s=300,facecolor='none',edgecolor=COLORS[j],lw=2.2,zorder=6)
        ax.text(xy[node,0],xy[node,1]+1.65 if j==2 else xy[node,1]-.85,f'MGO{j+1} / MG{j+1}\nPV＋储能＋DG',color=COLORS[j],ha='center',fontsize=10)
    ax.set(xlim=(-3.6,18.2),ylim=(-3.5,4.6));ax.set_aspect('equal');ax.axis('off')
    fig.suptitle('33节点主配微位置与全部开关｜动作前0−的风灾输入',fontsize=19,fontweight='bold')
    fig.text(.5,.075,'S01—S32：原馈线分段开关；S33—S37：原常开联络开关。红虚线为风场产生的初始故障，紫虚线为健康常开联络线。',ha='center',fontsize=10)
    fig.text(.5,.025,'微网在18/25/33；另加T0/T1两个主网等值，总母线35。图为接线排版，不是地理坐标；实际0+接线见图10。',ha='center',fontsize=10,color='#6e8290')
    fig.tight_layout(rect=(0,.13,1,.92));return save(fig,out,'00_33节点主配微与全部开关位置')


def plot_all(c,world,result,out,cap=None,phase=None,comparisons=None):
    style();files=[]
    for function,args in [(plot_layout,(c,world,out)),(plot_weather,(c,world,out)),(plot_faults,(c,result,out)),(plot_dispatch,(c,result,out)),(plot_recovery,(c,result,out)),(plot_crew,(c,result,out)),(plot_D,(result,out)),(plot_capability,(cap,out)),(plot_phase,(phase,out)),(plot_network,(c,result,out)),(plot_comparison,(comparisons or [result],out))]:files+=function(*args)
    Path(out).mkdir(exist_ok=True,parents=True);Path(out,'figures_manifest.json').write_text(json.dumps({'files':files,'source':'输入风场、真实执行记录、明确标识的预测条件证书；不含预设D相图'},ensure_ascii=False,indent=2),encoding='utf-8')
    return files
