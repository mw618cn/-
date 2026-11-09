"""实验比较的数学边界检查，避免只报告“最佳候选”的局部求解界。"""

import unittest

import numpy as np

from tdm_restore.experiments import _aggregate


class MenuBoundsTests(unittest.TestCase):
    def test_menu_lower_bound_includes_every_candidate(self):
        # 策略B目前incumbent更好，但A的未闭合间隙仍可能容纳真正最优解。
        records = [
            {"policy": {"id": "A"}, "result": {"x": np.zeros(1), "status": 1,
                                                  "objective": 10., "lower_bound": 1.}},
            {"policy": {"id": "B"}, "result": {"x": np.zeros(1), "status": 0,
                                                  "objective": 9., "lower_bound": 8.9}}]
        bound = _aggregate(records)
        self.assertEqual(bound["policy_id"], "B")
        self.assertAlmostEqual(bound["objective"], 9.)
        self.assertAlmostEqual(bound["lower_bound"], 1.)

    def test_certified_infeasible_policy_does_not_destroy_menu_bound(self):
        records = [
            {"policy": {"id": "A"}, "result": {"x": None, "status": 2,
                                                  "objective": np.inf, "lower_bound": -np.inf}},
            {"policy": {"id": "B"}, "result": {"x": np.zeros(1), "status": 0,
                                                  "objective": 9., "lower_bound": 8.9}}]
        bound = _aggregate(records)
        self.assertAlmostEqual(bound["lower_bound"], 8.9)
        self.assertFalse(bound["certified_infeasible"])


if __name__ == "__main__":
    unittest.main()
