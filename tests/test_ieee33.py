"""隔离33节点模块导入，避免与保留的6节点data/analysis同名模块串用。"""
from pathlib import Path
import subprocess,sys,unittest

ROOT=Path(__file__).resolve().parents[1]

class IEEE33Tests(unittest.TestCase):
 def test_ieee33_network_and_information_model(self):
    code=r'''
import sys,copy,numpy as np
sys.path.insert(0,'ieee33_version')
from data import load_parameters,build_profiles
from network import prepare_network,forest_components
from power_model import build_model
from repair_model import enumerate_policies
c,settings=load_parameters('ieee33_version/参数数据.xlsx')
assert len(c['nodes'])==35 and len(c['edges'])==39
assert [c['nodes'][i] for i in c['mg_nodes']]==['D18','D25','D33']
assert c['n_steps']==96 and len(c['scenarios'])==4
pd=np.array(c['load']['critical_base_mw'])+np.array(c['load']['general_base_mw'])
assert np.isclose(pd[2:].sum(),3.715)
assert np.isclose((pd*np.array(c['load']['reactive_ratio']))[2:].sum(),2.3)
assert np.isclose(c['load']['reactive_ratio'][31],3.0) # D30:200kW/600kvar
assert len(c['topologies'])==32
for e,edge in enumerate(c['edges']):
    if edge['kind']=='tie':assert any(z['statuses'][e] for z in c['topologies'])
for topo in c['topologies']:
    components=forest_components(c['nodes'],c['edges'],topo['statuses'])
    assert all(set(comp)&set(c['mg_nodes']) for comp in components)
    assert len(topo['roots'])==len(components)
# 没有故障时闭合原骨架+一条联络线会成环，必须被拒绝。
bad=copy.deepcopy(c)
bad['topologies']=[{'id':'bad','label':'bad','open_edges':['TIE1','TIE3','TIE4','TIE5']}]
try:prepare_network(bad)
except ValueError as ex:assert '成环' in str(ex)
else:raise AssertionError('带环接线未被拒绝')
p=next(v for v in enumerate_policies(c) if v['id']=='B_11')
m=build_model(c,p,branch='start')
assert m.var_indices['flow_p'].shape==(4,39,96)
assert m.var_indices['served_critical'].shape==(4,35,96)
assert m.var_indices['energy'].shape==(4,3,97)
assert m.var_indices['topology_choice'].shape[1]==32
assert m.metadata['upward_interface']==[(1,-1.0)]
names=m.metadata['row_names']
assert 'nonanticipativity_topology_choice' in names
assert 'repaired_before_close' in names
assert 'force_current_start' in names
assert m.metadata['profiles']['load_critical_mw'].shape==(35,96)
# 每个未修复故障支路在相应事件时刻的合闸上界必须为0。
for scene,schedule in enumerate(m.metadata['schedules']):
    assert schedule['availability']['A'][0]==0 and schedule['availability']['B'][0]==0
    assert schedule['observation_history_by_step'][0]==()
print('33节点、标准P/Q、五联络候选、成网森林、拒环、全96时段与拓扑非预期性：通过')
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
