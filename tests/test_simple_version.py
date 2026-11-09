"""简明版的Excel接口和模型等价性：验证简化使用方式没有改变物理模型。"""
import unittest
from pathlib import Path
import sys
import tempfile
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simple_version'))
from data import load_parameters,build_profiles
from power_model import build_model
from repair_model import enumerate_policies
from tdm_restore.config import load_config,build_profiles as old_profiles
from tdm_restore.model import build_model as old_model

BOOK = ROOT/'simple_version/参数数据.xlsx'

class SimpleVersionTests(unittest.TestCase):
    def test_excel_reconstructs_same_24h_model(self):
        """同一参数经Excel后，全部线性行、整数性、边界、目标与原模型相同。"""
        cfg,settings=load_parameters(BOOK);original=load_config(ROOT/'configs/demo_24h.json')
        self.assertEqual(cfg['n_steps'],96);self.assertEqual(settings['policy_ids'],'all')
        a,b=build_profiles(cfg),old_profiles(original)
        for k in ['pv_mw','load_critical_mw','load_general_mw','grid_price','critical_voll','general_voll']:
            np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=0)
        policy=enumerate_policies(cfg)[0];m=build_model(cfg,policy);n=old_model(original,policy)
        self.assertEqual(m.A.shape,n.A.shape);self.assertEqual((m.A!=n.A).nnz,0)
        for k in ['c','lb','ub','row_lb','row_ub','integrality']:
            np.testing.assert_allclose(getattr(m,k),getattr(n,k),atol=1e-12,rtol=0)
        self.assertAlmostEqual(m.objective_constant,n.objective_constant,places=9)

    def test_editing_capacity_load_price_in_excel_takes_effect(self):
        """修改用户能看到的三类数值，实际模型曲线相应变化，无隐藏覆盖。"""
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'case.xlsx';book=load_workbook(BOOK)
            for row in book['微网参数'].iter_rows(min_row=4):
                if row[0].value=='pv_peak_mw':row[1].value=0.425
            for row in book['基础参数'].iter_rows(min_row=4):
                if row[0].value=='grid_price':row[1].value=0.9
            for row in book['节点负荷'].iter_rows(min_row=4):
                if row[0].value=='D0':row[2].value=2.5
            book.save(path);book.close()
            changed,_=load_parameters(path);base,_=load_parameters(BOOK)
            a,b=build_profiles(changed),build_profiles(base)
            np.testing.assert_allclose(a['pv_mw'][0],b['pv_mw'][0]*0.5)
            np.testing.assert_allclose(a['load_general_mw'][2],b['load_general_mw'][2]*2)
            np.testing.assert_allclose(a['grid_price'],np.full(96,0.9))

    def test_absolute_curves_are_used_as_mw(self):
        """实测MW模式直接读数，不再乘装机容量或负荷基值。"""
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'case.xlsx';book=load_workbook(BOOK)
            for row in book['运行设置'].iter_rows(min_row=4):
                if row[0].value=='curve_mode':row[1].value='实际功率'
            book['24小时曲线']['B4']=0.123
            book.save(path);book.close();cfg,_=load_parameters(path)
            self.assertAlmostEqual(build_profiles(cfg)['pv_mw'][0,0],0.123)

    def test_invalid_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'case.xlsx';book=load_workbook(BOOK)
            book['损伤场景']['C4']=0.99;book.save(path);book.close()
            with self.assertRaisesRegex(ValueError,'概率'):load_parameters(path)

    def test_missing_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'case.xlsx';book=load_workbook(BOOK)
            book['基础参数'].delete_rows(4);book.save(path);book.close()
            with self.assertRaisesRegex(ValueError,'缺失'):load_parameters(path)


    def test_default_projection_covers_every_hour(self):
        from types import SimpleNamespace
        from analysis import _projection_steps
        model=SimpleNamespace(var_indices={"flow_p":np.zeros((4,6,96),dtype=int)})
        steps=_projection_steps(model,np.array([]),{"step_minutes":15})
        self.assertEqual(steps,list(range(0,96,4)))
        dense=_projection_steps(model,np.array([]),{"step_minutes":15,"capability":{"interval_hours":.25}})
        self.assertEqual(dense,list(range(96)))

    def test_old_workbook_keeps_new_plot_defaults(self):
        from data import OPTIONAL_RUN_SETTINGS
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.xlsx';book=load_workbook(BOOK);ws=book['运行设置']
            for row in range(ws.max_row,3,-1):
                if ws.cell(row,1).value in OPTIONAL_RUN_SETTINGS:ws.delete_rows(row)
            book.save(path);book.close();cfg,settings=load_parameters(path)
            self.assertEqual(cfg['capability']['interval_hours'],1.0)
            self.assertTrue(settings['compute_cost_surface'])

if __name__=='__main__':unittest.main()
