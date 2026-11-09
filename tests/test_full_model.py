"""完整24小时四场景 MILP 的集成复核；不是两时段替代算例。"""

import unittest

import numpy as np

from tdm_restore.config import load_config
from tdm_restore.crew import enumerate_policies
from tdm_restore.model import build_model
from tdm_restore.linear import solve
from tdm_restore.solution import extract_solution
from tdm_restore.validation import validate_solution
from tdm_restore.rolling import run_rolling_replay


class FullHorizonIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """集成检查共享同一个已解24小时基线，避免为每条断言重复优化。"""
        cls.cfg = load_config()
        cls.policy = enumerate_policies(cls.cfg)[0]
        cls.model = build_model(cls.cfg, cls.policy)
        cls.result = solve(cls.model, {**cls.cfg["solver"], "time_limit": 30.0})

    def test_default_24h_stochastic_model_has_physical_incumbent(self):
        """默认96时段下复核信息约束、完整启动、SOC、功率、成本及韧性面积。"""
        cfg, model, result = self.cfg, self.model, self.result
        self.assertIsNotNone(result["x"], result["message"])
        report = validate_solution(model, result, cfg)
        failures = [c for c in report["checks"] if not c["passed"]]
        self.assertTrue(report["passed"], failures)
        solution = extract_solution(model, result, cfg)
        self.assertEqual(np.asarray(solution["series"]["mg_soc"]).shape, (4, 3, 97))
        self.assertEqual(len(solution["time_hours"]), 96)
        self.assertAlmostEqual(solution["summary"]["protected_critical_served_fraction"], 1.0, places=6)
        self.assertAlmostEqual(solution["diagnostics"]["cost_reconstruction_error"], 0.0, places=5)
        self.assertLessEqual(result["lower_bound"], result["objective"] + 1e-5)
        summary = solution["summary"]
        self.assertAlmostEqual(summary["expected_eens_mwh"] + summary["expected_restored_energy_mwh"],
                               summary["total_demand_energy_mwh"], places=6)

    def test_observation_reoptimization_keeps_history_and_boundary_state(self):
        """2.25h重算：只知道A重伤；B类型不能泄漏，当前SOC和小时模式不得重置。"""
        cfg, model, result = self.cfg, self.model, self.result
        replay = run_rolling_replay(cfg, self.policy, model, result,
                                    realized_scenario_id="HH", decision_hours=(2.25,))
        step = 9
        original_s = next(i for i, scene in enumerate(cfg["scenarios"]) if scene["id"] == "HH")
        round_ = replay["rounds"][0]
        self.assertEqual(set(round_["posterior"]), {"HL", "HH"})
        self.assertAlmostEqual(round_["posterior"]["HL"], 0.6)
        self.assertEqual({o["site"] for o in round_["observations"]}, {"A"})
        self.assertFalse(round_["fallback"])
        self.assertTrue(replay["executed_validation"]["passed"])
        self.assertEqual(round_["original_terminal_hour"], 24)
        capacities = np.array([m["energy_mwh"] for m in cfg["microgrids"]])
        energy = result["x"][model.var_indices["energy"]][original_s, :, step]
        np.testing.assert_allclose(round_["current_soc"], energy / capacities, atol=2e-5)
        self.assertEqual(round_["fixed_prefix"]["by_block"]["battery_mode"], 18)
        executed = replay["executed_solution"]["series"]
        for internal, external in (("battery_charge", "mg_charge_mw"),
                                   ("battery_discharge", "mg_discharge_mw"),
                                   ("dg_power", "mg_dg_mw")):
            old = result["x"][model.var_indices[internal]][original_s, :, :step]
            np.testing.assert_allclose(np.array(executed[external])[0, :, :step], old, atol=2e-5)


    def test_conditioned_cost_grid_distinguishes_infeasible_and_invalid(self):
        """真实96时段LP：大启动需求超通道极限为不可行，越界SOC为参数错误。"""
        import copy
        from pathlib import Path
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"simple_version"))
        from analysis import compute_conditional_cost_surface
        cfg=copy.deepcopy(self.cfg)
        cfg["experiment"].update(surface_startup_multipliers=[.4,2.0],surface_soc_levels=[.1,.7,.9])
        self.assertTrue(np.all(self.result["x"][self.model.var_indices["target_start"][:,0]]>.5))
        surface=compute_conditional_cost_surface(self.model,self.result["x"],cfg,progress=None)
        self.assertEqual(surface["n_steps"],96)
        rows=surface["records"]
        self.assertTrue(all(r["status"]=="invalid_parameter" for r in rows if r["soc_initial"]==.1))
        self.assertTrue(all(r["status"]=="infeasible" for r in rows if r["soc_initial"]>=.7 and r["startup_multiplier"]==2.0))
        self.assertTrue(all(r["status"]=="optimal" and r["validation_passed"] for r in rows if r["soc_initial"]>=.7 and r["startup_multiplier"]==.4))


if __name__ == "__main__":
    unittest.main()
