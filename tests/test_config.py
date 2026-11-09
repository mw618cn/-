"""可替换参数的物理合法性与完整24小时网格检查。"""

import unittest

import numpy as np

from tdm_restore.config import default_config, load_config, build_profiles


class ParameterTests(unittest.TestCase):
    def test_default_is_24h_with_terminal_state(self):
        cfg = load_config()
        profiles = build_profiles(cfg)
        self.assertEqual(cfg["n_steps"], 96)
        self.assertEqual(cfg["horizon_hours"], 24)
        self.assertEqual(profiles["pv_mw"].shape, (3, 96))
        self.assertEqual(profiles["load_critical_mw"].shape, (6, 96))
        self.assertAlmostEqual(profiles["time_hours"][-1] + profiles["dt_hours"], 24)
        self.assertTrue((profiles["pv_mw"] >= 0).all())

    def test_bad_probability_and_efficiency_rejected(self):
        scenarios = default_config()["scenarios"]
        scenarios[0]["probability"] += 0.1
        with self.assertRaises(ValueError):
            load_config(overrides={"scenarios": scenarios})
        mg = default_config()["microgrids"]
        mg[0]["eta_charge"] = 1.05
        with self.assertRaises(ValueError):
            load_config(overrides={"microgrids": mg})

    def test_bad_soc_reserve_rejected(self):
        mg = default_config()["microgrids"]
        mg[1]["soc_terminal"] = 1.1
        with self.assertRaises(ValueError):
            load_config(overrides={"microgrids": mg})

    def test_parameter_objects_do_not_share_state(self):
        one = load_config()
        one["microgrids"][0]["soc_initial"] = 0.2
        two = load_config()
        self.assertAlmostEqual(two["microgrids"][0]["soc_initial"], 0.7)

    def test_invalid_decision_and_travel_grids_rejected(self):
        for patch in ({"step_minutes": 0}, {"battery_mode_minutes": 0},
                      {"generator": {"start_interval_minutes": 0}},
                      {"crew": {"travel_hours": {"depot": {"A": -0.25}}}},
                      {"crew": {"travel_hours": {"depot": {"A": 0.3}}}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                load_config(overrides=patch)

    def test_cannot_skip_startup_peak_or_use_invalid_power_factor(self):
        for patch in ({"generator": {"startup_profile_step_minutes": 7.5}},
                      {"generator": {"startup_profile_mw": []}},
                      {"load": {"power_factor": 0}},
                      {"generator": {"aux_power_factor": 1.1}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                load_config(overrides=patch)

    def test_scenario_ids_cannot_collide(self):
        scenarios = default_config()["scenarios"]
        scenarios[1]["id"] = scenarios[0]["id"]
        with self.assertRaises(ValueError):
            load_config(overrides={"scenarios": scenarios})


if __name__ == "__main__":
    unittest.main()
