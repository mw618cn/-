"""后向递推、内生观测与实体作业承诺的独立机制测试。"""
from pathlib import Path
import subprocess,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
CODE_DIR=ROOT/'resilience_v4' if (ROOT/'resilience_v4').is_dir() else ROOT


class V4RegressionTests(unittest.TestCase):
    def run_example(self,code):
        r=subprocess.run([sys.executable,'-c',code],cwd=CODE_DIR,capture_output=True,text=True,timeout=240)
        self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)

    def test_backward_predecessor_matches_analytic_storage_case(self):
        self.run_example(r'''
import sys,copy,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from reachability import island_reserve_tube,delta_interval
c,_=load_parameters('参数数据.xlsx');prepare(c)
c['microgrids']=copy.deepcopy(c['microgrids'][:1]);c['mg_nodes']=c['mg_nodes'][:1]
mg=c['microgrids'][0];mg.update(energy_mwh=1.,soc_min=.1,soc_max=1.,soc_terminal=.2,dg_max_mw=0.,eta_charge=1.,eta_discharge=1.,inverter_mva=2.)
node=c['mg_nodes'][0];c['profiles']['critical'][node]=.2;c['load']['reactive_ratio'][node]=0
fc={'hours':np.arange(4)*.25,'pv':np.zeros((1,1,4))}
tube=island_reserve_tube(c,{'step':0},fc)
np.testing.assert_allclose(tube['lower'][0],[.4,.35,.3,.25,.2],atol=1e-8)
assert .39<tube['lower'][0,0] and .41>tube['lower'][0,0]
# 同一瞬时功率需求不能在能量不足时被“投影可行”掩盖。
assert abs(delta_interval(c,mg,0,.2,0,mode=0)[0]+.05)<1e-9
print('后向前驱区间与解析解一致，24h代码使用相同递推。')
''')

    def test_revelation_changes_actions_only_after_inspection_and_locks_are_separate(self):
        self.run_example(r'''
import sys,copy,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from information import expand_hidden_types,make_crew_plan
c,_=load_parameters('参数数据.xlsx');prepare(c);c['horizon_hours']=1.
c['options']['information.priority_bonus']=0.;c['options']['information.max_hidden_targets']=1
e0,e1=2,3;ne=len(c['edges']);nm=len(c['microgrids']);c['edge_xy'][:]=0
for e in [e0,e1]:c['edges'][e].update(inspection_hours=.25,repair_light_hours=.5,repair_heavy_hours=5.)
c['repair_priority']=[0.]*ne;c['repair_priority'][e0]=6;c['repair_priority'][e1]=5
damaged=np.zeros(ne,bool);damaged[[e0,e1]]=True
state={'step':0,'damaged':damaged,'ever_failed':damaged.copy(),'observed_damage':{},'failure_observed_hour':{e0:0.,e1:0.},'crew':{'position':np.zeros(2),'active':None}}
base={'hours':np.arange(4)*.25,'pv':np.zeros((1,nm,4)),'wind':np.zeros((1,ne,4)),'failure_hour':np.full((1,ne),np.inf),'probabilities':np.ones(1),'crew_wind':np.zeros((ne,4))}
fc=expand_hidden_types(c,state,base);plan=make_crew_plan(c,state,fc,'information')
assert plan['dispatch'][0][0]==plan['dispatch'][1][0]==(e0,'inspect')
assert plan['histories'][0][0]==plan['histories'][1][0]
assert plan['histories'][0][1]!=plan['histories'][1][1]
assert plan['dispatch'][0][1]==(e0,'repair') and plan['dispatch'][1][1]==(e1,'inspect')
# 这是信息到达导致的合法分支；另检验开始维修后不能改派。
locked=copy.deepcopy(state);locked['observed_damage']={e0:'light'};locked['crew']['active']={'edge':e0,'kind':'repair','travel_remaining':0.,'work_remaining':.5}
lf=expand_hidden_types(c,locked,base);lp=make_crew_plan(c,locked,lf,'information')
for path in lp['trace']:
 assert path[0][:2]==path[1][:2]==(e0,'repair')
for actions in lp['dispatch']:assert actions[0] is None and actions[1] is None
# 未实际出发时的天气禁行不能提前产生实体任务锁。
storm=copy.deepcopy(base);storm['crew_wind'][:]=30.
sf=expand_hidden_types(c,state,storm);sp=make_crew_plan(c,state,sf,'critical')
assert all(item is None for actions in sp['dispatch'] for item in actions)
assert all(item is None for path in sp['trace'] for item in path)
print('检查前相同动作、检查后按类型分支、已开始维修独立保持。')
''')

    def test_hazard_probability_and_survival_have_distinct_meanings(self):
        self.run_example(r'''
import sys,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from weather import make_world
c,_=load_parameters('参数数据.xlsx');prepare(c);w=make_world(c)
p=np.asarray(w['interval_probability_if_intact']);f=np.asarray(w['cumulative_first_failure_probability'])
assert np.all((p>=0)&(p<=1)) and np.all((f>=0)&(f<=1))
assert np.min(np.diff(f,axis=1))>=-1e-12
np.testing.assert_allclose(p,-np.expm1(-np.asarray(w['hazard'])*c['dt']),atol=1e-12)
assert np.asarray(w['first_failure_mass']).sum(axis=1).max()<=1+1e-10
# 指数生存律不能被每15分钟重新套一次事件脆弱性概率代替。
lam=.2
assert abs((1-(-np.expm1(-lam*.25)))**4-np.exp(-lam))<1e-12
c['options']['weather.probability_model']='peak_fragility';q=make_world(c)
assert np.min(np.diff(np.asarray(q['cumulative_first_failure_probability']),axis=1))>=-1e-12
print('危险率、区间条件概率、累计首损概率与峰值脆弱性分别自洽。')
''')

    def test_delayed_real_observation_invalidates_old_information_branch(self):
        self.run_example(r'''
import sys,copy,numpy as np
sys.path.insert(0,'.')
from analysis import _compatible
c={'dt':.25,'crew':{'safe_wind_ms':18},'options':{}}
key0=(((2,0.),),());revealed=(((2,0.),),((2,'light'),))
fc={'probabilities':np.ones(1),'crew_plan':{'histories':[[key0,revealed]],'trace':[[(2,'inspect',0.,.25),(2,'repair',0.,1.)]],'dispatch':[[None,None]]}}
md={'hours':np.array([0.,.25]),'forecast':fc,'cfg':c}
state={'failure_observed_hour':{2:0.},'observed_damage':{},'observed_wind':np.zeros(3),'crew':{'active':{'edge':2,'kind':'inspect','travel_remaining':0.,'work_remaining':.25}}}
assert _compatible(md,state,0.)==[0]
# 预测检查已结束，真实检查尚未结束，必须重算，不能选“light”分支继续。
assert _compatible(md,state,.25)==[]
print('旧信息树的适用性包含实际检查完成情况，而非仅故障位置。')
''')

    def test_backward_information_tree_intersects_children_before_observation(self):
        self.run_example(r'''
import sys,copy,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from power_model import ModelBuilder
from reachability import certify_commands
c,_=load_parameters('参数数据.xlsx');prepare(c);c['dt']=1.;c['microgrids']=copy.deepcopy(c['microgrids'][:1]);c['mg_nodes']=c['mg_nodes'][:1]
c['microgrids'][0].update(energy_mwh=1.,soc_min=.1,soc_max=1.,soc_terminal=.2,dg_max_mw=0.,eta_charge=1.,eta_discharge=1.,inverter_mva=2.)
b=ModelBuilder();v={};ns,n,nm=2,2,1
for k in ('pv','dg','charge','discharge','pcc_p','pcc_q','mg_q','battery_mode'):v[k]=b.add_var(k,(2,1,2))
v['energy']=b.add_var('energy',(2,1,3));v['served_c']=b.add_var('served_c',(2,len(c['nodes']),2));v['served_g']=b.add_var('served_g',(2,len(c['nodes']),2))
v['flow_p']=b.add_var('flow_p',(2,len(c['edges']),2));v['flow_q']=b.add_var('flow_q',(2,len(c['edges']),2));v['target_start']=b.add_var('target_start',(2,0))
common=((),());ha=((),((2,'light'),));hb=((),((2,'heavy'),))
fc={'hours':np.array([0.,1.]),'probabilities':np.array([.5,.5]),'pv':np.zeros((2,1,2)),
 'crew_plan':{'histories':[[common,ha],[common,hb]],'tasks':[[],[]]}}
m=b.build();m.metadata.update(cfg=c,forecast=fc,hours=fc['hours'],mode_map=np.array([0,1]),fixed_aux=np.zeros(2),fixed_on=np.zeros(2),fixed_cap=np.zeros(2),auxcoef=np.zeros((2,0)),oncoef=np.zeros((2,0)),capcoef=np.zeros((2,0)),jobs=[],job_indices=[])
x=np.zeros(len(m.c));x[v['pcc_p'][:,0,1]]=[.1,.2];x[v['discharge'][:,0,1]]=[.1,.2];x[v['energy'][:,0,:]]=[[.5,.5,.4],[.5,.5,.3]]
cert=certify_commands(m,x);assert cert['passed'];np.testing.assert_allclose(cert['initial_lower'],[.4],atol=1e-8)
# 已观察到light分支后，只对剩余兼容树求前驱，不能继续交不可能的heavy。
conditional=certify_commands(m,x,start_index=1,active_scenarios=[0]);np.testing.assert_allclose(conditional['initial_lower'],[.3],atol=1e-8)
print('未观察先交子分支；观察后只保留兼容后继，解析区间通过。')
''')

    def test_original_simsun_is_required_and_not_silently_replaced(self):
        self.run_example(r'''
import sys
sys.path.insert(0,'.')
from plot_style import setup
import os,matplotlib
from pathlib import Path
os.environ.pop('TDM_SIMSUN_PATH',None)
other_font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
try:setup({'options':{'plots.font_file':str(other_font)}})
except ValueError as error:assert '不是宋体' in str(error)
else:raise AssertionError('不能把其他字体冒充原版宋体')
print('字体硬校验通过。')
''')

    def test_fixed_connection_restoration_is_not_a_controlled_switch_operation(self):
        self.run_example(r'''
import sys,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from weather import make_world
from analysis import initial_state
from information import expand_hidden_types,make_crew_plan
from power_model import build_model,optimize,extract,validate
c,_=load_parameters('参数数据.xlsx');prepare(c)
c['horizon_hours']=1.;c['n_steps']=4
for key in ('pv','critical','general'):c['profiles'][key]=c['profiles'][key][...,:4]
c['profiles']['price']=c['profiles']['price'][:4];c['alpha']=1.;c['generator']['last_start_hour']=0.
c['switch_max_total']=0.;c['switch_max_per_line']=0.;e=2;c['edges'][e]['switchable']=False
world=make_world(c);world['failure_hour'][:]=np.inf;world['failure_hour'][e]=0.;world['wind'][:]=0.
state=initial_state(c,world);state['observed_damage']={e:'light'};state['crew']['position']=c['edge_xy'][e].copy()
state['crew']['active']={'edge':e,'kind':'repair','travel_remaining':0.,'work_remaining':.25}
ne=len(c['edges']);nm=len(c['microgrids'])
base={'hours':np.arange(4)*.25,'pv':np.zeros((1,nm,4)),'wind':np.zeros((1,ne,4)),'failure_hour':np.full((1,ne),np.inf),'probabilities':np.ones(1),'crew_wind':np.zeros((ne,4))}
fc=expand_hidden_types(c,state,base);fc['crew_plan']=make_crew_plan(c,state,fc,'critical')
m=build_model(c,state,fc,'wait');r=optimize(m,.05,0)
assert r['x'] is not None and validate(m,r['x'])['passed']
d=extract(m,r['x']);assert d['closed'][e,0]<.5 and d['closed'][e,1]>.5
assert np.max(r['x'][m.var_indices['switch_change']])<1e-8
print('固定接线随维修恢复，DSO操作预算为0仍保持正确，不伪造主动合闸。')
''')

    def test_no_initial_source_and_no_upward_supply_excludes_start_by_balance(self):
        self.run_example(r'''
import sys,numpy as np
sys.path.insert(0,'.')
from data import load_parameters
from network import prepare
from weather import make_world,forecast
from analysis import initial_state
from information import expand_hidden_types,make_crew_plan
from power_model import build_model,optimize
c,_=load_parameters('参数数据.xlsx');prepare(c);c['horizon_hours']=1.;c['n_steps']=4;c['alpha']=0.;c['allow_upward']=0
for key in ('pv','critical','general'):c['profiles'][key]=c['profiles'][key][...,:4]
c['profiles']['price']=c['profiles']['price'][:4];c['generator']['last_start_hour']=0.
w=make_world(c);state=initial_state(c,w);fc=expand_hidden_types(c,state,forecast(c,state));fc['crew_plan']=make_crew_plan(c,state,fc,'critical')
m=build_model(c,state,fc,'start');assert np.all(m.ub[m.var_indices['target_start']]==0)
r=optimize(m,.2,0);assert r['certified_infeasible']
print('无上级初始源且禁止TX反送，正辅机功率由节点平衡严格排除，不依赖启发式失败。')
''')

    def test_analytic_increment_projection_matches_independent_four_variable_lp(self):
        self.run_example(r'''
import sys,numpy as np
sys.path.insert(0,'.')
from scipy.optimize import linprog
from reachability import _increment
rng=np.random.default_rng(20261110)
for case in range(120):
 pv,dg=rng.uniform(0,1.5),rng.uniform(0,.8);p,q=rng.uniform(-1.2,2.8),rng.uniform(-2.,2.)
 ch,dis,ec,ed,inv,dt=.7,.9,.94,.93,1.6,.25;sides=[6,8,12][case%3]
 angles=(np.arange(sides)+.5)*2*np.pi/sides;inner=np.cos(np.pi/sides)
 A=np.array([[np.cos(a),0.,-np.cos(a),np.cos(a)] for a in angles]);rhs=inv*inner-np.sin(angles)*q
 energy=np.array([0.,0.,dt*ec,-dt/ed]);eq=np.array([[1.,1.,-1.,1.]])
 for mode in (0,1):
  bounds=[(0,pv),(0,dg),(0,ch*mode),(0,dis*(1-mode))]
  lo=linprog(energy,A_ub=A,b_ub=rhs,A_eq=eq,b_eq=[p],bounds=bounds,method='highs')
  hi=linprog(-energy,A_ub=A,b_ub=rhs,A_eq=eq,b_eq=[p],bounds=bounds,method='highs')
  actual=_increment(pv,dg,ch,dis,ec,ed,inv,p,q,dt,sides,mode)
  if lo.success and hi.success:np.testing.assert_allclose(actual,[lo.fun,-hi.fun],atol=1e-8,rtol=0)
  else:assert actual==(np.inf,-np.inf)
print('120组P/Q/PV/DG与三种多边形、两种模式，共240个问题，与独立LP端点/不可行状态一致。')
''')
