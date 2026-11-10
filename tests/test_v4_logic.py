"""R4真正外界、轨迹主问题、更新后前缀约束和及时信息选择测试。"""
from pathlib import Path
import subprocess,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'resilience_v4' if (ROOT/'resilience_v4').is_dir() else ROOT


class V4LogicTests(unittest.TestCase):
    def run_code(self,s):
        r=subprocess.run([sys.executable,'-c',s],cwd=CODE,text=True,capture_output=True,timeout=300)
        self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)

    def test_lp_relaxation_is_outer_bound_not_fixed_integer_value(self):
        self.run_code(r'''
import numpy as np
from scipy import sparse
from power_model import LinearModel
from bounds import relaxation_bound
m=LinearModel(np.array([-1.,-.2]),np.array([1,0]),np.zeros(2),np.ones(2),sparse.csr_matrix([[2.,1.]]),np.array([-np.inf]),np.array([1.5]),{}, {'past_reward':0.,'denominator':1.,'remaining_reward_max':1.})
r=relaxation_bound(m,3)
assert r['valid'] and .75<=r['R_upper']<.75001
# 固定整数x=0的最优R只有.2，不能拿.2充当该MILP上界。
assert r['R_upper']>.2 and r['cost_lower_bound']<=-.75
print(r)
''')

    def test_class_upper_is_max_and_unknown_candidates_are_kept(self):
        self.run_code(r'''
from bounds import class_bounds
L,U=class_bounds([{'R_lower':.7,'R_upper':.75},{'R_lower':.8,'R_upper':.9}]);assert L==.8 and U==.9
L,U=class_bounds([{'R_lower':.7,'R_upper':.75},{'R_lower':None,'R_upper':None}]);assert L==.7 and U==1.
L,U=class_bounds([{'R_lower':.7,'R_upper':.75},{'R_lower':None,'R_upper':None,'certified_infeasible':True}]);assert U==.75
''')

    def test_lagrangian_certificate_keeps_stationarity_remainder(self):
        self.run_code(r'''
import numpy as np
from scipy import sparse
from bounds import lagrangian_lower
# 任意不精确对偶都必须保守；保留盒余项后仍不高于真实最小值-1。
z=lagrangian_lower(np.array([-1.]),sparse.csr_matrix([[1.]]),np.array([.5]),sparse.csr_matrix((0,1)),np.array([]),np.array([0.]),np.array([1.]),np.array([0.]),np.array([]))
assert z['cost_lower_bound']<=-1.
''')

    def test_master_uses_common_lambda_and_changes_dispatch(self):
        self.run_code(r'''
import numpy as np
from scipy import sparse
from power_model import LinearModel,residual
from trajectory_domain import solve_with_domain
m=LinearModel(np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),np.ones(3),sparse.csr_matrix([[1.,0.,1.],[0.,1.,1.],[1.,1.,0.]]),np.array([1.,1.,-np.inf]),np.array([np.inf,np.inf,1.]),{})
context={'mapping':np.array([0,1]),'V':np.array([[1.,0.],[0.,1.]])}
r=solve_with_domain(m,context);np.testing.assert_allclose(r['x'],[.5,.5,.5],atol=1e-8)
assert residual(m,r['x'])<1e-8 and abs(sum(r['domain']['lambda'])-1)<1e-8
assert r['objective']<1. # 参考列(1,0)需g=1，主问题选共同凸组合并降低g。
''')

    def test_updated_prefix_bounds_are_not_lost_during_elimination(self):
        self.run_code(r'''
import numpy as np
from scipy import sparse
from power_model import LinearModel
from trajectory_domain import solve_with_domain
m=LinearModel(np.zeros(3),np.zeros(3),np.array([1.,1.,0.]),np.ones(3),sparse.csr_matrix((0,3)),np.array([]),np.array([]),{})
ctx={'mapping':np.array([0,1]),'V':np.array([[1.,0.],[0.,1.]])}
r=solve_with_domain(m,ctx);assert r['x'] is None and r['status']==2
''')

    def test_timely_target_selection_does_not_read_hidden_damage(self):
        self.run_code(r'''
import numpy as np
from data import load_parameters
from network import prepare
from information import expand_hidden_types
c,_=load_parameters('参数数据.xlsx');prepare(c);e0,e1=2,3;ne=len(c['edges']);nm=len(c['microgrids']);c['edge_xy'][:]=0
c['options'].update({'information.max_hidden_targets':1,'information.target_selection':'timely'});c['generator']['last_start_hour']=.5
c['repair_priority']=[0.]*ne;c['repair_priority'][e0]=10;c['repair_priority'][e1]=1
for e in (e0,e1):c['edges'][e]['inspection_hours']=.25
bad=np.zeros(ne,bool);bad[[e0,e1]]=1
st={'step':0,'damaged':bad,'ever_failed':bad.copy(),'observed_damage':{},'failure_observed_hour':{e0:0.,e1:0.},'crew':{'active':None,'position':np.zeros(2)}}
w=np.zeros((ne,4));w[e0]=35.
f={'hours':np.arange(4)*.25,'wind':w[None],'crew_wind':w,'pv':np.zeros((1,nm,4)),'failure_hour':np.full((1,ne),np.inf),'probabilities':np.ones(1)}
r=expand_hidden_types(c,st,f);assert r['hidden_targets']==[e1]
assert any(v['before_start_deadline'] for v in r['target_selection'] if v['edge']==e1)
''')

    def test_physical_domain_coordinates_and_decomposes_complete_horizon(self):
        self.run_code(r'''
import numpy as np
from data import load_parameters
from network import prepare
from weather import make_world,forecast
from analysis import initial_state
from information import expand_hidden_types,make_crew_plan
from power_model import build_model,optimize,validate,residual
from trajectory_domain import coordinate
c,_=load_parameters('参数数据.xlsx');prepare(c);c['horizon_hours']=1.;c['n_steps']=4;c['alpha']=1.;c['generator']['last_start_hour']=0.
for k in ('pv','critical','general'):c['profiles'][k]=c['profiles'][k][...,:4]
c['profiles']['price']=c['profiles']['price'][:4];c['options']['domain.max_columns']=4
w=make_world(c);w['failure_hour'][:]=np.inf;w['wind'][:]=0.;st=initial_state(c,w)
f=expand_hidden_types(c,st,forecast(c,st));f['crew_plan']=make_crew_plan(c,st,f,'critical')
m=build_model(c,st,f,'wait');r=optimize(m,.1,0);assert r['x'] is not None
new=coordinate(m,r);assert validate(m,new['x'])['passed'] and new['domain']['used']
ctx=m.metadata['domain_context'];assert residual(ctx['local_model'],new['x'][ctx['mapping']])<2e-5
assert new['domain']['master_variables']<new['domain']['original_variables']
print(new['domain'])
''')

    def test_splicing_failure_is_cross_time_not_instantaneous_limit(self):
        self.run_code(r'''
import numpy as np
from power_model import ModelBuilder
from evidence import splice_diagnostic
b=ModelBuilder();p=b.add_var('pcc_p',(1,1,2),ub=1);q=b.add_var('pcc_q',(1,1,2),ub=0)
b.add_row({int(p[0,0,0]):1.,int(p[0,0,1]):1.},ub=1.,name='whole_horizon_energy');m=b.build()
r=splice_diagnostic(m,np.array([[1.,1.]]),np.zeros((1,2)),np.ones(1),1.,2.)
assert r['status']=='infeasible' and r['first_nonextendable_prefix_end_hour']==2.
assert abs(r['P_correction_MWh']-1)<1e-8 and r['Q_correction_Mvarh']==0
assert r['prefix_extendable']==[1,1,0]
''')

    def test_scenario_information_relaxation_is_an_outer_bound(self):
        self.run_code(r'''
import numpy as np
from scipy import sparse
from power_model import LinearModel
from bounds import scenario_relaxation_bound
# 同信息x0=x1时R*=.4；信息松弛分别取.4和1，外界为.7。
m=LinearModel(np.array([-.5,-.5]),np.zeros(2),np.zeros(2),np.ones(2),sparse.csr_matrix([[1.,0.],[0.,1.],[1.,-1.]]),np.array([-np.inf,-np.inf,0.]),np.array([.4,1.,0.]),{'served_c':np.array([[0],[1]])},
 {'forecast':{'probabilities':np.array([.5,.5]),'crew_plan':{'present':True}},'row_names':['physical0','physical1','NA_control'],'past_reward':0.,'denominator':1.,'remaining_reward_max':1.})
r=scenario_relaxation_bound(m,5.);assert r['valid'] and .7<=r['R_upper']<.70001
assert r['relaxed_NA_rows']==1 and r['R_upper']>=.4
''')

    def test_four_step_rollout_preserves_domain_and_executed_prefix(self):
        self.run_code(r'''
import numpy as np
from data import load_parameters
from network import prepare
from weather import make_world
from analysis import simulate
c,_=load_parameters('参数数据.xlsx');c['horizon_hours']=1.;c['n_steps']=4;c['alpha']=1.;c['generator']['last_start_hour']=0.
for k in ('pv','critical','general'):c['profiles'][k]=c['profiles'][k][...,:4]
c['profiles']['price']=c['profiles']['price'][:4];prepare(c)
for edge in c['edges']:edge['hazard_scale']=0.
c['options'].update({'domain.max_columns':3,'solver.use_relaxation_bounds':False,'solver.decision_refinement_rounds':0})
w=make_world(c);r,_=simulate(c,w,progress=None)
assert len(r['controller_steps'])==4 and all(s['domain_used'] and s['backward_passed'] for s in r['controller_steps'])
assert all(s['horizon_end_hour']==1. and s['execute_step_minutes']==15 for s in r['controller_steps'])
assert any(s['solve']=='fixed_plan_LP' for s in r['controller_steps'])
for j,mg in enumerate(c['microgrids']):
 np.testing.assert_allclose(np.diff(np.asarray(r['energy'])[:,j]),c['dt']*(mg['eta_charge']*np.asarray(r['charge'])[:,j]-np.asarray(r['discharge'])[:,j]/mg['eta_discharge']),atol=2e-5)
print(r['summary'])
''')

    def test_bound_cache_key_changes_with_problem_not_solver_budget(self):
        self.run_code(r'''
import numpy as np,copy
from scipy import sparse
from power_model import LinearModel
from bounds import model_fingerprint,reuse_cached_bounds
m=LinearModel(np.array([-1.]),np.ones(1),np.zeros(1),np.ones(1),sparse.csr_matrix([[1.]]),np.array([0.]),np.array([1.]),{}, {'solver_seconds':1})
key=model_fingerprint(m);n=copy.deepcopy(m);n.metadata['solver_seconds']=60;assert model_fingerprint(n)==key
n.lb[0]=.5;assert model_fingerprint(n)!=key
n=copy.deepcopy(m);n.integrality[0]=0;assert model_fingerprint(n)!=key
n=copy.deepcopy(m);n.metadata['past_reward']=.2;assert model_fingerprint(n)!=key
n=copy.deepcopy(m);n.metadata['denominator']=2.;assert model_fingerprint(n)!=key
r={'R_lower':.5,'R_upper':.95};reuse_cached_bounds(r,{'R_lower':.6,'R_upper':.8},key)
assert r['R_lower']==.6 and r['R_upper']==.8 and abs(r['absolute_R_gap']-.2)<1e-9
assert r['bound_certificates'][-1]['model_fingerprint']==key
''')
