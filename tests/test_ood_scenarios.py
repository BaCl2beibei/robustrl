import unittest
from importlib.util import find_spec
import random

from ftm.config import SimulationConfig
from ftm.environment import scenario_by_name
from ftm.experiment_suite import (_episode_row, build_evaluation_cases,
                                  paired_bootstrap, summarize)
from ftm.main import run_episode
from ftm.ood_scenarios import (OOD_FAMILIES, make_test_scenario,
                               raw_coverage_diagnostics, scenario_family)
from ftm.surrogate import (PARAMETER_NAMES, perturb_scenario, scenario_to_raw,
                           training_raw_bounds)
from ftm.trained_policy import run_trained_episode


class OODScenarioTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                       max_time_seconds=1.0)

    def test_base_cases_preserve_training_scenario_configs(self):
        for family in OOD_FAMILIES:
            self.assertEqual(make_test_scenario(family, "base", self.config),
                             scenario_by_name(family, self.config))

    def test_ood_levels_raise_the_documented_disturbances(self):
        expected = {
            "normal": {"failure_multiplier": (0.4, 0.6),
                       "congestion_enter": (0.02, 0.04)},
            "failure": {"failure_multiplier": (12.0, 16.0),
                        "common_shock_probability": (0.003, 0.005)},
            "network": {"congestion_enter": (0.50, 0.65),
                        "congestion_leave": (0.08, 0.05)},
            "load": {"load_burst_probability": (0.22, 0.30),
                     "load_burst_cpu": (0.65, 0.75)},
            "mixed": {"failure_multiplier": (5.0, 7.0),
                      "congestion_enter": (0.25, 0.35),
                      "common_shock_probability": (0.003, 0.005),
                      "load_burst_probability": (0.10, 0.15),
                      "load_burst_cpu": (0.65, 0.75),
                      "congestion_leave": (0.15, 0.08)},
        }
        for family, fields in expected.items():
            base = make_test_scenario(family, "base", self.config)
            l1 = make_test_scenario(family, "l1", self.config)
            l2 = make_test_scenario(family, "l2", self.config)
            self.assertEqual(l1.name, f"{family}__ood_l1")
            self.assertEqual(l2.name, f"{family}__ood_l2")
            for field, (l1_value, l2_value) in fields.items():
                self.assertEqual(getattr(l1, field), l1_value)
                self.assertEqual(getattr(l2, field), l2_value)
                self.assertNotEqual(getattr(base, field), getattr(l1, field))

    def test_load_duration_variants_map_to_load_family(self):
        self.assertEqual(scenario_family("load_short"), "load")
        self.assertEqual(make_test_scenario("load_short", "base", self.config).name,
                         "load_short")

    def test_training_noise_bounds_are_centered_on_each_base_scenario(self):
        base = scenario_by_name("mixed", self.config)
        lower, upper = training_raw_bounds(base, noise_std=0.25)
        center = scenario_to_raw(base)
        self.assertEqual(len(center), len(PARAMETER_NAMES))
        for value, low, high in zip(center, lower, upper):
            self.assertAlmostEqual(value - low, 0.5)
            self.assertAlmostEqual(high - value, 0.5)
        self.assertEqual(perturb_scenario(base, random.Random(44)),
                         perturb_scenario(base, random.Random(44)))

    def test_evaluation_cases_have_unique_ids_and_expected_splits(self):
        id_cases = build_evaluation_cases("id", "base", "all", self.config)
        ood_cases = build_evaluation_cases("ood", "all", "all", self.config)
        all_cases = build_evaluation_cases("all", "all", "all", self.config)
        self.assertEqual(len(id_cases), 8)
        self.assertEqual(len(ood_cases), 10)
        self.assertEqual(len(all_cases), len(id_cases) + len(ood_cases))
        self.assertEqual(len({case.scenario_id for case in all_cases}), len(all_cases))
        self.assertTrue(all(case.split == "id" and case.severity == "base"
                            for case in id_cases))
        self.assertTrue(all(case.split == "ood" and case.severity in ("l1", "l2")
                            for case in ood_cases))
        diagnostics = raw_coverage_diagnostics(ood_cases, self.config)
        self.assertTrue(all(item["coverage_source"] == "theoretical_gaussian_interval"
                            for item in diagnostics))
        self.assertTrue(all(item["ood_by_main_coverage"] for item in diagnostics))

    def test_coverage_diagnostic_uses_unique_empirical_parameter_samples(self):
        case = build_evaluation_cases("ood", "l1", "load", self.config)[0]
        raw = scenario_to_raw(scenario_by_name("load", self.config))
        nearby = list(raw)
        nearby[0] += 0.01
        records = [
            {"scenario": "load_short", "parameters": list(raw)},
            {"scenario": "load_short", "parameters": list(raw)},
            {"scenario": "load_medium", "parameters": nearby},
        ]
        diagnostic = raw_coverage_diagnostics([case], self.config, records)[0]
        self.assertEqual(diagnostic["coverage_source"], "empirical_training_records")
        self.assertEqual(diagnostic["training_samples"], 2)

    def test_episode_and_trained_episode_accept_frozen_scenario_objects(self):
        scenario = make_test_scenario("network", "l1", self.config)
        metrics, env = run_episode("no_migration", scenario, 3, self.config)
        self.assertEqual(env.scenario, scenario)
        case = next(case for case in build_evaluation_cases(
            "ood", "l1", "network", self.config
        ))
        row = _episode_row(metrics, case)
        self.assertEqual(row["scenario_family"], "network")
        self.assertEqual(row["severity"], "l1")
        self.assertEqual(row["split"], "ood")
        self.assertEqual(row["scenario_id"], "network__ood_l1")
        self.assertEqual(row["disturbance_congestion_enter"], 0.50)

        class WaitController:
            def act(self, env, observation):
                return env.config.n_nodes

        _, trained_env = run_trained_episode("trained_surrogate_ppo", scenario, 3,
                                             self.config, WaitController())
        self.assertEqual(trained_env.scenario, scenario)

    def test_summaries_keep_ood_scenario_ids_separate(self):
        rows = []
        for scenario_id, split, severity, score in (
                ("load", "id", "base", 1.0),
                ("load__ood_l1", "ood", "l1", 0.2)):
            for policy, value in (("no_migration", score), ("greedy", score + 0.1)):
                rows.append({
                    "scenario": scenario_id,
                    "scenario_id": scenario_id,
                    "scenario_family": "load",
                    "split": split,
                    "severity": severity,
                    "policy": policy,
                    "seed": 7,
                    "task_completion_rate": value,
                })
        summary = summarize(rows, resamples=100)
        comparison = paired_bootstrap(rows, resamples=100, permutation_resamples=100)
        self.assertEqual({row["scenario_id"] for row in summary},
                         {"load", "load__ood_l1"})
        self.assertEqual({row["scenario_id"] for row in comparison},
                         {"load", "load__ood_l1"})

    @unittest.skipUnless(find_spec("torch"), "PyTorch learning extra is not installed")
    def test_pgd_projection_stays_inside_ellipsoid_box_intersection(self):
        import torch

        from ftm.torch_ops import projected_adversarial_search

        center = torch.tensor([0.0, 0.0])
        axes = torch.tensor([1.0, 1.0])
        point, _ = projected_adversarial_search(
            center, axes, 1.0, lambda raw: -raw.sum(), steps=8,
            learning_rate=0.2, starts=1,
            lower_bounds=(-0.3, -0.1), upper_bounds=(0.3, 0.1),
        )
        self.assertTrue(torch.all(point >= torch.tensor([-0.3, -0.1]) - 1e-6).item())
        self.assertTrue(torch.all(point <= torch.tensor([0.3, 0.1]) + 1e-6).item())
        self.assertLessEqual(float(torch.linalg.vector_norm(point)), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
