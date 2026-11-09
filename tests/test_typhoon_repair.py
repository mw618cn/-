"""针对用户运行包中暴露的退化结果做实算回归。"""
from pathlib import Path
import subprocess,sys,unittest
ROOT=Path(__file__).resolve().parents[1]


class TyphoonRepairTests(unittest.TestCase):
    def run_code(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_battery_candidates_and_explicit_interface(self):
        self.run_code(r'''
import sys,copy,numpy as np
sys.path.insert(0,'typhoon_version')
from data import load_parameters
from network import prepare
from weather import make_world,forecast
from analysis import initial_state
from power_model import build_model,safe_seed,optimize,validate,extract
c,_=load_parameters('typhoon_version/参数数据.xlsx');prepare(c);state=initial_state(c,make_world(c))
for j,mg in enumerate(c['microgrids']):state['energy'][j]=mg['energy_mwh']*.85
fc=forecast(c,state);m=build_model(c,state,fc,'start')
charge=safe_seed(m,False,'critical_first');discharge=safe_seed(m,True,'critical_first')
assert charge is not None and discharge is not None
assert charge['objective']-discharge['objective']>.02*c['denominator']
# 即使只给0.01秒MILP，也不能丢掉已校核的较好放电候选。
r=optimize(m,seconds=.01,secondary_seconds=0)
assert r['R_plan']>=-discharge['objective']/c['denominator']-1e-7
assert {'charge','discharge'} <= {q['battery'] for q in r['seed_candidates']}
assert validate(m,r['x'])['passed']
assert m.metadata['interface_envelopes'] and r['interface_envelope_count']>0
d=extract(m,r['x']);node=c['mg_nodes'][0]
np.testing.assert_allclose(d['pcc_p'][0],d['pv'][0]+d['dg'][0]+d['discharge'][0]-d['charge'][0]-d['served_c'][node]-d['served_g'][node],atol=1e-6)
# 真实SOC变化必须改变重新上报的累计能量包络。
other=copy.deepcopy(state);other['energy'][0]-=.2
m2=build_model(c,other,forecast(c,other),'start')
a=next(q for q in m.metadata['interface_envelopes'] if q['scenario']==0 and q['microgrid']==0)
b=next(q for q in m2.metadata['interface_envelopes'] if q['scenario']==0 and q['microgrid']==0)
assert abs(a['upper_mwh']-b['upper_mwh']-.2*c['microgrids'][0]['eta_discharge'])<1e-8
print('放电候选不丢失、显式接口守恒、SOC更新包络：通过')
''')

    def test_local_resource_domain_and_impossible_trajectory(self):
        self.run_code(r'''
import sys,numpy as np
sys.path.insert(0,'typhoon_version')
from data import load_parameters
from network import prepare
from weather import make_world,forecast
from analysis import initial_state
from power_model import solve_lp,residual
from capability import build_resource_model
c,_=load_parameters('typhoon_version/参数数据.xlsx');prepare(c);st=initial_state(c,make_world(c));fc=forecast(c,st)
m=build_resource_model(c,st,fc,np.zeros((2,3,96)));v=m.var_indices
assert v['energy'].shape==(2,3,97)
# 初始预测的DSO入口故障不应把微网内部资源域画成零。
assert fc['failure_hour'][0,next(i for i,e in enumerate(c['edges']) if e['name']=='L01_02')]==4
obj=np.zeros_like(m.c);obj[v['pcc_p'][0,:,48]]=-1
r=solve_lp(m,obj);assert r['x'] is not None and residual(m,r['x'])<1e-6
assert r['x'][v['pcc_p'][0,:,48]].sum()>.1
# 逐时点不能任意拼接：整天持续10MW/微网的指令必须被拒绝。
eq=[({int(v['pcc_p'][0,0,t]):1},10.) for t in range(96)]
assert solve_lp(m,equalities=eq)['status']==2
# 低功率但持续24h、累计能量超额的轨迹同样拒绝。
pv_energy=fc['pv'][0,0].sum()*c['dt'];mg=c['microgrids'][0]
upper=pv_energy+24*mg['dg_max_mw']+mg['eta_discharge']*(st['energy'][0]-mg['energy_mwh']*mg['soc_terminal'])
eq=[({int(col):c['dt'] for col in v['pcc_p'][0,0]},upper+1.)]
assert solve_lp(m,equalities=eq)['status']==2
print('完整96时段局部资源域、网络/资源区分、持续能量超额拒绝：通过')
''')
