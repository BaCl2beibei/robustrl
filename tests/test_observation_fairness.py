"""Policy inputs must be causal snapshots shared by all decision makers."""

import unittest
from dataclasses import FrozenInstanceError

from ftm.config import MonitoringConfig, ScenarioConfig, SimulationConfig
from ftm.environment import MigrationEnvironment
from ftm.observation import SCHEMA_VERSION, encode_observation
from ftm.policies import (GreedyPolicy, LoadBasedPolicy, NoMigrationPolicy,
                          RandomPolicy, RobustHeuristicPolicy)


class ObservationFairnessTests(unittest.TestCase):
    def make_env(self, monitoring=MonitoringConfig()):
        config = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                  max_time_seconds=20, monitoring=monitoring)
        env = MigrationEnvironment(config, ScenarioConfig(name="quiet"))
        env.reset(11)
        return env

    def test_context_is_immutable_and_encodes_every_public_field(self):
        env = self.make_env()
        context = env.get_decision_context()
        self.assertEqual(context.schema_version, SCHEMA_VERSION)
        self.assertEqual(encode_observation(context), env.observation())
        self.assertEqual(context.action_mask, env.valid_action_mask())
        self.assertEqual(context.action_mask[:-1],
                         env.manager.valid_targets(env.focus_task, env.cluster))
        with self.assertRaises(FrozenInstanceError):
            context.nodes[0].online = False
        for policy in (NoMigrationPolicy(), RandomPolicy(1), GreedyPolicy(),
                       LoadBasedPolicy(), RobustHeuristicPolicy()):
            self.assertTrue(context.action_mask[policy.act(context)])

    def test_hidden_scenario_change_cannot_change_current_observation(self):
        env = self.make_env()
        before = env.get_decision_context()
        env.scenario = ScenarioConfig(name="future", failure_multiplier=100,
                                      load_burst_probability=1)
        env.disturbances.scenario = env.scenario
        self.assertEqual(env.get_decision_context(), before)

    def test_monitor_noise_does_not_change_physical_random_sequence(self):
        config = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                  max_time_seconds=4)
        noisy = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                 max_time_seconds=4,
                                 monitoring=MonitoringConfig(cpu_noise_std=0.2,
                                                             memory_noise_std=0.5,
                                                             network_rtt_noise_std=0.1))
        left = MigrationEnvironment(config, ScenarioConfig(name="mixed"))
        right = MigrationEnvironment(noisy, ScenarioConfig(name="mixed"))
        left.reset(9)
        right.reset(9)
        for _ in range(4):
            _, left_reward, left_end, left_truncated, left_info = left.step(2)
            _, right_reward, right_end, right_truncated, right_info = right.step(2)
            self.assertEqual(left_reward, right_reward)
            self.assertEqual(left_info["events"], right_info["events"])
            if left_end or left_truncated:
                self.assertEqual((left_end, left_truncated),
                                 (right_end, right_truncated))
                break

    def test_observed_mask_can_disagree_with_physical_execution(self):
        env = self.make_env(MonitoringConfig(reporting_delay_seconds=2))
        context = env.get_decision_context()
        target = next(i for i, valid in enumerate(context.action_mask[:-1]) if valid)
        env.cluster.nodes[target].up = False
        self.assertTrue(env.get_decision_context().action_mask[target])
        _, _, _, _, info = env.step(target)
        self.assertTrue(any(event["kind"] == "invalid_action"
                            for event in info["events"]))


if __name__ == "__main__":
    unittest.main()
