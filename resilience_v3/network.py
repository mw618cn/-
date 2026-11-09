"""通用网络工具：不使用A/B标签和固定接线菜单。"""
import numpy as np

def components(c,closed):
    parent=list(range(len(c['nodes'])))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    cyclic=False
    for e,z in enumerate(closed):
        if z<.5:continue
        line=c['edges'][e];a,b=find(line['from']),find(line['to'])
        if a==b:cyclic=True
        else:parent[a]=b
    groups={}
    for i in range(len(parent)):groups.setdefault(find(i),[]).append(i)
    return list(groups.values()),cyclic

def prepare(c):
    c['tx']=next(i for i,e in enumerate(c['edges']) if e['name']=='TX')
    c['mg_edge_indices']=[next(i for i,e in enumerate(c['edges']) if m in (e['from'],e['to'])) for m in c['mg_nodes']]
    c['edge_xy']=np.array([(c['geo_xy'][e['from']]+c['geo_xy'][e['to']])/2 for e in c['edges']])
    c['incident']=[[(j,1 if e['from']==i else -1) for j,e in enumerate(c['edges']) if i in (e['from'],e['to'])] for i in range(len(c['nodes']))]
    # 仅用于“重要负荷优先”基线的静态下游权重，联合策略不以此代替优化。
    weights=c['critical_weight']*c['load']['critical_base_mw']+c['general_weight']*c['load']['general_base_mw']
    c['repair_priority']=[]
    for j,e in enumerate(c['edges']):
        closed=[line['kind']!='tie' and k!=j for k,line in enumerate(c['edges'])]
        groups,_=components(c,closed)
        downstream=next((g for g in groups if e['to'] in g),[])
        c['repair_priority'].append(float(weights[downstream].sum()) if 0 not in downstream else float(weights[[e['from'],e['to']]].sum()))
    c['denominator']=float(c['dt']*(c['critical_weight']*c['profiles']['critical'].sum()+c['general_weight']*c['profiles']['general'].sum()))
    return c
