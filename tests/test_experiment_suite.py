import unittest

from ftm.experiment_suite import (bootstrap_interval, paired_bootstrap,
                                  paired_sign_flip_pvalue)
from ftm.config import ScenarioConfig
from ftm.surrogate import raw_to_scenario, scenario_to_raw


class ExperimentSuiteTests(unittest.TestCase):
    def test_bootstrap_is_seed_reproducible(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(bootstrap_interval(values, seed=17, resamples=500),
                         bootstrap_interval(values, seed=17, resamples=500))

    def test_paired_bootstrap_uses_common_seeds(self):
        rows = []
        for seed, baseline, policy in ((1, 2.0, 3.0), (2, 1.0, 3.0), (3, 4.0, 5.0)):
            rows.append({"scenario": "normal", "policy": "no_migration", "seed": seed,
                         "task_completion_rate": baseline})
            rows.append({"scenario": "normal", "policy": "time_min", "seed": seed,
                         "task_completion_rate": policy})
        result = paired_bootstrap(rows, resamples=500)
        completion = next(row for row in result
                          if row["policy"] == "time_min"
                          and row["metric"] == "task_completion_rate")
        self.assertEqual(completion["n_pairs"], 3)
        self.assertAlmostEqual(completion["mean_difference"], 4.0 / 3.0)

    def test_single_policy_has_no_paired_comparison(self):
        rows = [{"scenario": "normal", "policy": "no_migration", "seed": 1,
                 "task_completion_rate": 1.0}]
        self.assertEqual(paired_bootstrap(rows), [])

    def test_small_paired_sign_flip_test_is_exact(self):
        self.assertAlmostEqual(paired_sign_flip_pvalue([1.0, 1.0, 1.0]), 0.25)
        self.assertEqual(paired_sign_flip_pvalue([0.0, 0.0]), 1.0)

    def test_unconstrained_surrogate_parameters_decode_to_legal_scenario(self):
        template = ScenarioConfig()
        round_trip = raw_to_scenario(scenario_to_raw(template), template)
        for field in ("congestion_enter", "congestion_leave",
                      "common_shock_probability", "load_burst_probability",
                      "load_burst_cpu"):
            self.assertGreaterEqual(getattr(round_trip, field), 0.0)
            self.assertLessEqual(getattr(round_trip, field), 1.0)
        self.assertGreater(round_trip.failure_multiplier, 0.0)


if __name__ == "__main__":
    unittest.main()
