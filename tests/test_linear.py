"""连续可分解性和整数固定的反例测试。

两时段只用于构造最小数学反例；交付算例仍完整求解24小时96时段。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tdm_restore.linear import ModelBuilder, freeze_discrete, solve, solve_lp
from tdm_restore.capability import decompose_trajectory


class ProjectionAndDecompositionTests(unittest.TestCase):
    """验证“各时段分别可行”不能被错误地解释为“整条轨迹可行”。"""

    def test_cross_time_coupling_is_kept_when_fixing_interface(self):
        # 电池仅有1MWh：两个1小时的放电功率分别可达1MW，但不能同时达到。
        builder = ModelBuilder()
        power = builder.add_var("export", 2, ub=1)
        builder.add_row({power[0]: 1, power[1]: 1}, ub=1, name="总能量预算")
        model = builder.build()
        first = solve_lp(model, objective=np.array([-1.0, 0.0]))
        second = solve_lp(model, objective=np.array([0.0, -1.0]))
        self.assertTrue(first["success"] and second["success"])
        self.assertAlmostEqual(first["x"][power[0]], 1)
        self.assertAlmostEqual(second["x"][power[1]], 1)
        impossible = solve_lp(model, extra_equalities=[({power[0]: 1}, 1), ({power[1]: 1}, 1)])
        self.assertEqual(impossible["status"], 2)
        feasible = solve_lp(model, extra_equalities=[({power[0]: 1}, 0.4), ({power[1]: 1}, 0.6)])
        self.assertTrue(feasible["success"])
        np.testing.assert_allclose(feasible["x"], [0.4, 0.6])

    def test_fixed_binary_does_not_become_fractional_capacity(self):
        builder = ModelBuilder()
        power = builder.add_var("power", 1, ub=2)
        closed = builder.add_var("closed", 1, ub=1, integer=True)
        builder.add_row({power[0]: 1, closed[0]: -2}, ub=0)
        model = builder.build()
        with self.assertRaises(ValueError):
            solve_lp(model)
        fixed = freeze_discrete(model, np.array([0.0, 0.0]))
        result = solve_lp(fixed, objective=np.array([-1.0, 0.0]))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][power[0]], 0)
        with self.assertRaises(ValueError):
            freeze_discrete(model, np.array([1.0, 0.5]))

    def test_gap_tolerance_does_not_erase_valid_dual_bound(self):
        """状态0可能仅达到允许间隙；D 判据仍须保留真正的下界。"""
        builder = ModelBuilder()
        builder.add_var("binary", 1, ub=1, integer=True)
        builder.objective_constant = 100
        model = builder.build()
        solver_reply = SimpleNamespace(x=np.array([1.0]), fun=10.0,
                                       mip_dual_bound=9.98, mip_gap=0.002,
                                       status=0, success=True,
                                       message="达到设置的相对间隙", mip_node_count=1)
        with patch("tdm_restore.linear.milp", return_value=solver_reply):
            result = solve(model)
        self.assertAlmostEqual(result["objective"], 110)
        self.assertAlmostEqual(result["lower_bound"], 109.98)

    def test_decomposition_rejects_stitched_projection_points(self):
        """直接调用交付分解入口，确认没有用逐时拟合掩盖跨期能量不足。"""
        builder = ModelBuilder()
        fp = builder.add_var("flow_p", (1, 6, 2), lb=0, ub=0)
        builder.add_var("flow_q", (1, 6, 2), lb=0, ub=0)
        model = builder.build()
        # 仅L0允许向上1MW；使用负方向表示微网向主网净输出。
        model.lb[fp[0, 1]] = -1
        # 总能量预算 -P0-P1 <= 1。
        from scipy import sparse
        cols = fp[0, 1]
        model.A = sparse.coo_matrix(([-1., -1.], ([0, 0], cols)),
                                    shape=(1, len(model.c))).tocsr()
        model.row_lb = np.array([-np.inf])
        model.row_ub = np.array([1.])
        bad = decompose_trajectory(model, np.array([[1., 1.]]))
        self.assertFalse(bad["success"])
        good = decompose_trajectory(model, np.array([[0.4, 0.6]]))
        self.assertTrue(good["success"])
        self.assertLessEqual(good["p_tracking_max_mw"], 1e-6)


if __name__ == "__main__":
    unittest.main()
