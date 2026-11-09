"""校核层的反例测试：用错误信息时序和整数值确认审计不会漏报。"""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from tdm_restore.validation import validate_solution


class ValidationTests(unittest.TestCase):
    """这些反例不依赖求解器最优性，专门验证校核是否能识别坏方案。"""

    def make_case(self):
        model = SimpleNamespace(
            c=np.zeros(4), integrality=np.zeros(4), lb=np.zeros(4), ub=np.full(4, 10.0),
            A=csr_matrix((0, 4)), row_lb=np.array([]), row_ub=np.array([]),
            objective_constant=7.0,
            var_indices={"grid_p": np.arange(4).reshape(2, 2)},
            metadata={"profiles": {"time_hours": [0, 0.25], "dt_hours": 0.25},
                      "information": "causal",
                      "schedules": [
                          {"observation_history_by_step": [(), (("A", "light"),)]},
                          {"observation_history_by_step": [(), (("A", "heavy"),)]}]} )
        return model, {"n_steps": 2, "step_minutes": 15}

    def test_cannot_act_on_unobserved_damage(self):
        """两个尚未区分的场景在 t=0 采取不同进口，必须判为信息泄漏。"""
        model, cfg = self.make_case()
        report = validate_solution(model, {"x": np.array([1, 2, 3, 4]), "objective": 7}, cfg)
        check = next(c for c in report["checks"] if "非预期性" in c["name"])
        self.assertFalse(check["passed"])
        self.assertAlmostEqual(check["max_violation"], 2)

    def test_can_adapt_after_observation(self):
        """完成巡检后允许不同动作，不能把合法适应性误判为不一致。"""
        model, cfg = self.make_case()
        report = validate_solution(model, {"x": np.array([1, 2, 1, 4]), "objective": 7}, cfg)
        self.assertTrue(report["passed"])

    def test_fractional_binary_is_rejected(self):
        model, cfg = self.make_case()
        model.integrality[1] = 1
        report = validate_solution(model, {"x": np.array([1, 0.4, 1, 4]), "objective": 7}, cfg)
        self.assertFalse(report["passed"])
        check = next(c for c in report["checks"] if "整数性" in c["name"])
        self.assertFalse(check["passed"])

    def test_no_incumbent_never_passes(self):
        model, cfg = self.make_case()
        self.assertFalse(validate_solution(model, {"x": None}, cfg)["passed"])


if __name__ == "__main__":
    unittest.main()
