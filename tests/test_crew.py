"""巡检信息与非抢占维修的因果性测试。"""

import unittest

import numpy as np

from tdm_restore.config import load_config
from tdm_restore.crew import build_scenarios, enumerate_policies, schedule_policy


class CrewCausalityTests(unittest.TestCase):
    """不是检查某条固定时间表，而是检查整个策略菜单共同满足的物理性质。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.scenarios = build_scenarios(cls.cfg)
        cls.policies = enumerate_policies(cls.cfg)

    def test_joint_prior_and_policy_menu(self):
        self.assertEqual(len(self.scenarios), 4)
        self.assertAlmostEqual(sum(s["probability"] for s in self.scenarios), 1.0)
        self.assertTrue(all(s["probability"] > 0 for s in self.scenarios))
        self.assertEqual(len(self.policies), 8)
        self.assertEqual(len({p["id"] for p in self.policies}), 8)
        self.assertEqual({p["first"] for p in self.policies}, {"A", "B"})

    def test_one_crew_and_inspect_before_repair(self):
        for policy in self.policies:
            schedules = schedule_policy(self.cfg, policy, self.scenarios)
            for schedule in schedules:
                tasks = sorted(schedule["tasks"], key=lambda task: task["start_hour"])
                for before, after in zip(tasks, tasks[1:]):
                    self.assertLessEqual(before["end_hour"], after["start_hour"] + 1e-9)
                for task in tasks:
                    self.assertGreaterEqual(task["start_hour"], 0)
                    self.assertGreaterEqual(task["end_hour"], task["start_hour"])
                    if task["kind"] == "repair":
                        self.assertGreaterEqual(task["start_hour"],
                                                schedule["inspection_hours"][task["site"]] - 1e-9)
                for site in ("A", "B"):
                    self.assertGreaterEqual(schedule["repair_hours"][site],
                                            schedule["inspection_hours"][site])
                    available = np.asarray(schedule["availability"][site])
                    self.assertTrue(np.isin(available, [0, 1]).all())
                    self.assertTrue((np.diff(available) >= 0).all())
                    times = np.arange(self.cfg["n_steps"]) * self.cfg["step_minutes"] / 60
                    np.testing.assert_array_equal(available,
                                                  times >= schedule["repair_hours"][site] - 1e-9)

    def test_same_history_implies_same_current_crew_action(self):
        """未观察到不同灾损前，不能在某个隐藏场景中提前去另一个站点。"""
        dt = self.cfg["step_minutes"] / 60
        for policy in self.policies:
            schedules = schedule_policy(self.cfg, policy, self.scenarios)
            for t in range(self.cfg["n_steps"]):
                observed_groups = {}
                for s, schedule in enumerate(schedules):
                    history = repr(schedule["observation_history_by_step"][t])
                    active = [task for task in schedule["tasks"]
                              if task["start_hour"] <= t * dt + 1e-9
                              and task["end_hour"] > t * dt + 1e-9]
                    action = tuple((task["kind"], task.get("site")) for task in active)
                    if history in observed_groups:
                        self.assertEqual(action, observed_groups[history],
                                         f"策略{policy['id']}在 t={t*dt} 偷看未观测损伤")
                    observed_groups[history] = action


if __name__ == "__main__":
    unittest.main()
