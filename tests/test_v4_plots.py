"""只检查图形数据接口/artist，不渲染或交付任何替代字体图片。"""
from pathlib import Path
import subprocess,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'resilience_v4' if (ROOT/'resilience_v4').is_dir() else ROOT


class V4PlotTests(unittest.TestCase):
    def test_explicit_single_column_width(self):
        code="""
import plot_style
plot_style.CONFIG={'plots.width_mm':180,'plots.single_width_mm':88.9}
f=plot_style.figure(height_in=2.,width='single')
assert abs(f.get_figwidth()*25.4-88.9)<1e-9 and f.get_figheight()==2.
plot_style.plt.close(f)
"""
        r=subprocess.run([sys.executable,'-c',code],cwd=CODE,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_saved_data_plot_interfaces_without_rendering(self):
        fixture=CODE/'示例结果_24小时'
        if not fixture.exists():fixture=ROOT/'resilience_v3/示例结果_24小时'
        if not fixture.exists():self.skipTest('没有随包数值示例')
        code=r'''
import json,numpy as np,tempfile
from pathlib import Path
from unittest.mock import patch
import plots
from data import load_parameters
from network import prepare
root=Path(FIXTURE);read=lambda p:json.loads((root/p).read_text())
c,_=load_parameters(root/'本次参数.xlsx');prepare(c);r=read('joint/executed_result.json');w=read('world.json')
w['failure_hour']=np.array([np.inf if x is None else x for x in w['failure_hour']])
for key in ('wind','hazard','interval_probability_if_intact','cumulative_first_failure_probability'):w[key]=np.asarray(w[key])
cap=read('capability/capability.json');phase=read('phase/phase_map.json') if (root/'phase/phase_map.json').exists() else None
info=read('initial_information.json');seen=[]
comparisons=[r]+[read(name+'/executed_result.json') for name in ('critical_first','nearest','no_upward') if (root/name/'executed_result.json').exists()]
def close_without_rendering(fig,out,stem):
 assert fig.axes
 if stem.startswith('07_'):assert all(ax.name!='3d' for ax in fig.axes)
 seen.append(stem);plots.plt.close(fig);return []
with tempfile.TemporaryDirectory() as folder,patch.object(plots,'setup',lambda c:None),patch.object(plots,'save',close_without_rendering):
 plots.plot_all(c,w,r,folder,cap,phase,comparisons,information=info)
assert any(s.startswith('07_') for s in seen) and any(s.startswith('14_') for s in seen)
if len(comparisons)>1:assert any(s.startswith('09_') for s in seen)
assert any(s.startswith('15_') for s in seen)
print('检查图接口',len(seen),'；未渲染任何字体图片')
'''.replace('FIXTURE',repr(str(fixture)))
        r=subprocess.run([sys.executable,'-c',code],cwd=CODE,capture_output=True,text=True,timeout=90)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_saved_json_can_be_exported_to_editable_excel(self):
        fixture=CODE/'示例结果_24小时'
        if not fixture.exists():self.skipTest('没有随包R4数值示例')
        code=r'''
import json,tempfile
from pathlib import Path
from data import write_workbook
from main import check_provenance
from openpyxl import load_workbook
root=Path(FIXTURE);read=lambda p:json.loads((root/p).read_text())
r=read('joint/executed_result.json');w=read('world.json');cap=read('capability/capability.json')
fingerprint=read('来源指纹.json');check_provenance(fingerprint,[('实际',r),('域',cap)])
try:check_provenance(fingerprint,[('混入另一运行',dict(r,run_id='another_run'))])
except ValueError:pass
else:raise AssertionError('跨运行实际轨迹未被拒绝')
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'result.xlsx';write_workbook(r,None,p,world=w,capability=cap)
 b=load_workbook(p,read_only=True)
 assert b['24小时功率'].max_row==97 and b['SOC'].max_row==98
 assert b['在线轨迹域协调'].max_row==97 and b['故障概率定义'].max_row>96
 assert b['拼接最小修正'].max_row>1
 b.close()
'''.replace('FIXTURE',repr(str(fixture)))
        r=subprocess.run([sys.executable,'-c',code],cwd=CODE,capture_output=True,text=True,timeout=90)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
