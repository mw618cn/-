"""动态风灾版的物理/信息边界，独立子进程防止同名模块互相污染。"""
from pathlib import Path
import subprocess,sys,unittest
ROOT=Path(__file__).resolve().parents[1]

class TyphoonVersionTests(unittest.TestCase):
 def test_weather_information_crew_and_forests(self):
    code=r'''
import sys,copy,numpy as np
sys.path.insert(0,'typhoon_version')
from data import load_parameters
from network import prepare
from weather import make_world,forecast,hazard
from analysis import initial_state,advance_crew
from repair_model import job_options
from power_model import build_model,safe_seed,validate,LinearModel,solve_lp
c,_=load_parameters('typhoon_version/参数数据.xlsx');prepare(c);w=make_world(c);st=initial_state(c,w)
assert c['n_steps']==96 and len(c['nodes'])==35 and len(c['edges'])==39
assert sum(w['failure_hour']<24)>2 and sum(w['failure_hour']<=0)>0
assert all(e['exposed'] for e in c['edges'] if e['kind']=='tie')
assert np.max(hazard(c,np.zeros((39,2))))==0
assert np.allclose(make_world(c)['failure_hour'],w['failure_hour'])
assert not st['observed_damage'] # 故障位置已报告，潜在轻重未提前揭示。
fc=forecast(c,st);jobs,_=job_options(c,st,fc)
assert {j['edge'] for j in jobs}==set(np.flatnonzero(st['damaged']))
for j in jobs:assert j['work']==c['edges'][j['edge']]['inspection_hours']+c['edges'][j['edge']]['repair_heavy_hours']
# 修改尚未发生的真实未来和潜在损伤，不改变当前预测。优化器根本不接收world。
f=copy.deepcopy(w);f['failure_hour'][f['failure_hour']>0]=99;f['severity']=['heavy']*39
other=forecast(c,initial_state(c,f))
np.testing.assert_allclose(fc['failure_hour'],other['failure_hour']);np.testing.assert_allclose(fc['pv'],other['pv'])
# 巡检完成只揭示损伤；维修完成改变可用性，均不自动合闸。
e=int(np.flatnonzero(st['damaged'])[0]);fake=copy.deepcopy(w);fake['wind'][:]=0;fake['severity'][e]='heavy'
a=copy.deepcopy(st);a['crew']['active']={'edge':e,'kind':'inspect','travel_remaining':0.,'work_remaining':c['dt']};events=[]
assert advance_crew(c,a,fake,events);assert a['observed_damage'][e]=='heavy' and a['damaged'][e] and a['closed'][e]==0
work=c['edges'][e]['repair_heavy_hours'];a['crew']['active']={'edge':e,'kind':'repair','travel_remaining':0.,'work_remaining':work}
for k in range(int(round(work/c['dt']))):a['step']=k;advance_crew(c,a,fake,events)
assert not a['damaged'][e] and a['closed'][e]==0
# 用完整96时段LP候选复核径向、源、SOC、P/Q和信息约束。
m=build_model(c,st,fc,'wait');seed=safe_seed(m,repair_rule='critical_first')
if seed is None:seed=safe_seed(m,True,'critical_first')
if seed is None:seed=safe_seed(m,allow_start=False)
assert seed is not None
assert validate(m,seed['x'])['passed']
assert m.var_indices['flow_p'].shape==(2,39,96) and m.var_indices['energy'].shape==(2,3,97)
# 中途重算也必须继承之前的开关保持时间，覆盖后续小时网格而非仅当前格。
hold=copy.deepcopy(st);hold['step']=21;hold['last_change'][3]=5.25
hold['energy']=np.array([m['energy_mwh']*m['soc_initial'] for m in c['microgrids']])
hm=build_model(c,hold,forecast(c,hold),'wait')
# now=5.25，保持期到6.25；6h的未来网格也必须锁定，而非只锁当前格。
kk=int(np.flatnonzero(np.isclose(hm.metadata['hours'][hm.metadata['slots']],6.0))[0])
col=int(hm.var_indices['closed'][0,3,kk])
locked=[i for i,name in enumerate(hm.metadata['row_names']) if name=='executed_switch_hold']
assert hm.A[locked,col].nnz>0
# 强制断开唯一主配馈线入口，alpha=0时T0/T1/D1允许失电；当前启动不可行。
bad=copy.deepcopy(st);entry=next(i for i,e in enumerate(c['edges']) if e['name']=='L01_02')
bad['damaged'][entry]=True;bad['ever_failed'][entry]=True;bad['closed'][entry]=0
bf=forecast(c,bad);bm=build_model(c,bad,bf,'start')
relax=LinearModel(bm.c,np.zeros_like(bm.integrality),bm.lb,bm.ub,bm.A,bm.row_lb,bm.row_ub,bm.var_indices,bm.metadata)
assert solve_lp(relax)['status']==2 # LP松弛已不可行，可据此证明该预测模型启动不可行。
print('多故障、危险率、无未来泄漏、巡检/维修/合闸区分、全96时段森林及启动不可行证明：通过')
'''
    r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=150)
    self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)
