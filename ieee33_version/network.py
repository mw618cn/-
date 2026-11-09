"""33节点网络工具：母线/PCC映射、径向森林检查、参数图与实际开关图。

33个配电母线之外另加T0/T1两个主网等值节点。微网为单母线聚合
模型，接在D18/D25/D33；其PCC交换量取该母线全部DSO支路的有向和。
重构在Excel显式给出的有限径向接线菜单内优化，不声称遍历全部生成树。
"""
from pathlib import Path
import numpy as np


def forest_components(nodes, edges, statuses):
    """用并查集检查整张接线是否无环，并返回每个电气连通分量。"""
    parent=list(range(len(nodes)))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    for edge,closed in zip(edges,statuses):
        if not closed:continue
        a,b=find(edge['from']),find(edge['to'])
        if a==b:raise ValueError(f"接线成环：{edge['name']}；闭合联络线时须打开环内分段开关")
        parent[a]=b
    groups={}
    for i in range(len(nodes)):groups.setdefault(find(i),[]).append(i)
    return list(groups.values())


def prepare_network(cfg):
    """把用户表格中的节点名映射为索引，核查每个候选森林的成网电源。"""
    nodes=cfg['nodes'];edges=cfg['edges'];lookup={v:i for i,v in enumerate(nodes)}
    mg_nodes=[lookup[m['bus']] for m in cfg['microgrids']]
    if len(set(mg_nodes))!=len(mg_nodes):raise ValueError('三个微网需接在不同母线')
    cfg['mg_nodes']=mg_nodes
    if any(e.get('fault_site') and e['initial_closed'] for e in edges):raise ValueError('初始故障线路必须先隔离，initial_closed应为0')
    edge_lookup={e['name']:i for i,e in enumerate(edges)}
    fault_edges={s:[] for s in ('A','B')}
    for i,e in enumerate(edges):
        if e.get('fault_site'):fault_edges[e['fault_site']].append(i)
    if any(len(v)!=1 for v in fault_edges.values()):raise ValueError('A/B各须对应一条故障线路')
    cfg['fault_edges']={k:v[0] for k,v in fault_edges.items()}
    signatures=set()
    for topo in cfg['topologies']:
        missing=set(topo['open_edges'])-set(edge_lookup)
        if missing:raise ValueError(f"接线{topo['id']}含未知线路{sorted(missing)}")
        statuses=[int(e['name'] not in topo['open_edges']) for e in edges]
        if tuple(statuses) in signatures:raise ValueError('接线菜单中存在重复接线')
        signatures.add(tuple(statuses))
        for e,z in zip(edges,statuses):
            if not e['switchable'] and not z:raise ValueError(f"{e['name']}没有可操作开关，不能打开")
        groups=forest_components(nodes,edges,statuses)
        roots=[]
        for group in groups:
            forming=sorted(set(group)&set(mg_nodes))
            # 本版所有候选分量均保留至少一个黑启动成网微网；不让一个
            # 未启动机组或alpha=0的上级电网凭空成为成网根节点。
            if not forming:raise ValueError(f"接线{topo['id']}的分量{[nodes[k] for k in group]}没有成网微网")
            roots.append(0 if 0 in group else forming[0])
        topo.update(statuses=statuses,roots=roots,components=groups,
                    required_sites=[s for s,e in cfg['fault_edges'].items() if statuses[e]])
    cfg['upward_interface']=[(edge_lookup['TX'],-1.0)]
    cfg['mg_interfaces']=[[(i,1.0 if e['from']==node else -1.0)
                          for i,e in enumerate(edges) if node in (e['from'],e['to'])]
                         for node in mg_nodes]
    return cfg


def interface_coefficients(model):
    """返回(P_up,Q_up)的有向支路系数，正值表示DSO向TSO反送。"""
    return model.metadata['upward_interface']


def mg_exchanges(model, x, reactive=False):
    """微网PCC净输出=所在母线向所有相邻DSO支路的净注入。"""
    values=np.asarray(x)[model.var_indices['flow_q' if reactive else 'flow_p']]
    return np.stack([sum(sign*values[:,edge,:] for edge,sign in interface)
                     for interface in model.metadata['mg_interfaces']],axis=1)


def draw_network(cfg, outdir, solution=None):
    """先画不依赖求解的参数接线图，再按真实结果画四个时刻及开关热图。"""
    # plots先设置可写字体缓存和无窗口后端，兼容Windows IDE和云环境。
    from plots import _set_style, _save
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    _set_style();outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    coords=np.asarray(cfg['node_coordinates'],float)
    nodes=cfg['nodes'];edges=cfg['edges']
    mg_nodes=cfg['mg_nodes'];colors=['#177d88','#ce8a29','#7765ad']
    def panel(ax,statuses,title,served=None):
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=10)
        ax.add_patch(FancyBboxPatch((-3.2,-1.05),3.0,2.6,boxstyle='round,pad=.15',
                    facecolor='#edf3fb',edgecolor='#adc1da',lw=1.1,zorder=0))
        ax.text(-1.7,1.25,'TSO｜主网等值',ha='center',color='#284e73',fontsize=10)
        for i,e in enumerate(edges):
            p,q=coords[e['from']],coords[e['to']];mid=(p+q)/2
            closed=bool(statuses[i]);fault=bool(e.get('fault_site'))
            col='#d05256' if fault and not closed else '#af83cb' if e['kind']=='tie' else '#557183'
            if e['kind']=='tie':
                # 折线路径将五条联络线与原馈线分离；位置仅为示意。
                route=e.get('plot_route',[])
                path=np.vstack([p,np.asarray(route).reshape(-1,2),q]) if route else np.vstack([p,q])
                ax.plot(path[:,0],path[:,1],color=col,lw=1.45,ls='-' if closed else '--',alpha=.85,zorder=1)
                mid=path[len(path)//2] if route else mid
            else:ax.plot([p[0],q[0]],[p[1],q[1]],color=col,lw=1.6,ls='-' if closed else '--',zorder=1)
            if e['switchable']:
                ax.scatter(*mid,s=23,marker='s',facecolor=col if closed else 'white',edgecolor=col,lw=.9,zorder=3)
            if e['switchable'] and e['kind']!='tie':
                ax.text(mid[0],mid[1]-.26,e['switch_id'],ha='center',fontsize=7,color=col,
                        bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=4)
            if fault:
                ax.text(mid[0],mid[1]+.27,f"故障{e['fault_site']}",ha='center',fontsize=8,color='#bd3e48',
                        bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=4)
            if e['kind']=='tie':
                ax.text(mid[0],mid[1]-.29,e['switch_id'],ha='center',fontsize=8,color='#8558a3',
                        bbox=dict(facecolor='white',edgecolor='none',pad=.3),zorder=4)
        face=['#e8f0f6']*len(nodes)
        if served is not None:face=plt.cm.YlGn(.25+.7*np.asarray(served))
        ax.scatter(coords[:,0],coords[:,1],s=100,c=face,edgecolor='#315870',lw=.8,zorder=5)
        for i,(x,y) in enumerate(coords):ax.text(x,y,nodes[i][1:] if nodes[i].startswith('D') else nodes[i],ha='center',va='center',fontsize=7.5,zorder=6)
        for j,node in enumerate(mg_nodes):
            x,y=coords[node];ax.scatter(x,y,s=300,facecolor='none',edgecolor=colors[j],lw=2,zorder=6)
            ax.annotate(f'MGO{j+1} / MG{j+1}\nPV＋储能＋DG',xy=(x,y),xytext=(x+.2,y+1.65 if j==2 else y-.9),fontsize=8.5,
                        ha='center',color=colors[j],arrowprops=dict(arrowstyle='-',color=colors[j]),zorder=7)
        ax.text(-2.9,-.7,'T0：待启动机组G\n     ＋剩余电源 α',fontsize=8,color='#284e73')
        ax.text(-.1,-.7,'TX：主配接口\n理想变比＋阻抗',fontsize=8,color='#284e73',ha='center')
        ax.text(5,4.75,'DSO｜33节点配电网（12.66 kV）',ha='center',fontsize=11,color='#315870',fontweight='bold')
        ax.set(xlim=(-3.5,18.3),ylim=(-3.7,5.3));ax.set_aspect('equal');ax.axis('off')
    initial=[e['initial_closed'] for e in edges]
    fig,ax=plt.subplots(figsize=(18,9));panel(ax,initial,'33节点主配微算例：位置、故障与开关')
    handles=[Line2D([0],[0],color='#557183',lw=2,label='初始闭合线路'),
             Line2D([0],[0],color='#af83cb',ls='--',lw=2,label='常开联络线 S33—S37'),
             Line2D([0],[0],marker='s',color='#557183',ls='none',label='可操作开关（方框）'),
             Line2D([0],[0],color='#d05256',ls='--',lw=2,label='初始故障隔离线路 A/B')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,.09),fontsize=11)
    fig.text(.5,.055,'配电节点33个；另加主网等值T0/T1。接线/阻抗/原始P、Q基于MATPOWER case33bw；电源、容量、故障、抢修和地理布局为自设研究参数。',ha='center',fontsize=9,color='#6b7c8c')
    fig.text(.5,.025,'图为参数接线示意，不是恢复计算结果；初始时各孤岛由所在成网微网维持，忽略送电暂态与励磁涌流。',ha='center',fontsize=9,color='#6b7c8c')
    paths=_save(fig,outdir,'00_33节点主配微接线与开关位置')
    if not solution or not solution.get('series'):return paths
    sid=solution['scenario_ids'].index('HH') if 'HH' in solution['scenario_ids'] else 0
    series=solution['series'];status=np.asarray(series['line_status'])[sid]
    power=np.asarray(series['served_critical_mw'])[sid]+np.asarray(series['served_general_mw'])[sid]
    demand=np.asarray(series['demand_critical_mw'])+np.asarray(series['demand_general_mw'])
    fraction=np.divide(power,demand,out=np.ones_like(power),where=demand>1e-9)
    fig,axes=plt.subplots(2,2,figsize=(22,12))
    for ax,h in zip(axes.flat,[0,4,12,23.75]):
        t=min(status.shape[-1]-1,int(round(h/solution['dt_hours'])))
        panel(ax,status[:,t]>.5,f"{solution['scenario_names'][sid]}｜t={h:g} h",fraction[:,t])
    fig.suptitle('33节点恢复接线随时间变化｜取自实际求解路径',fontsize=19,fontweight='bold')
    fig.text(.5,.025,'绿色深浅表示节点恢复比例；实/虚线为当时闭合/断开状态。每个时刻均单独验证径向性与成网微网连通性。',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.05,1,.94));paths+=_save(fig,outdir,'14_33节点实际恢复接线快照')
    indices=[i for i,e in enumerate(edges) if e['switchable']]
    fig,ax=plt.subplots(figsize=(13,max(5,.32*len(indices))))
    ax.imshow(status[indices],aspect='auto',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,
              extent=(0,24,len(indices)-.5,-.5))
    ax.set_yticks(range(len(indices)),[edges[i]['switch_id']+'｜'+edges[i]['name'] for i in indices])
    ax.set(xlabel='灾后时间 / h',ylabel='开关及所在支路',xticks=range(0,25,2),title='实际开关状态：浅色断开 / 深色闭合（同一场景全96时段）')
    fig.tight_layout();paths+=_save(fig,outdir,'15_33节点开关状态96时段')
    return paths
