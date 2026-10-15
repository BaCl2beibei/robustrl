"""Checks for the auxiliary PPO observation and PGD projection."""

import math
import unittest
from dataclasses import replace
from importlib.util import find_spec

from ftm.config import SimulationConfig
from ftm.decision_features import AUX_FEATURES_PER_NODE, encode_candidate_features
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.observation import encode_observation
from ftm.policies import (estimate_completion_time, estimate_execution_time,
                          estimate_migration_time, estimate_wait_time)
from ftm.reliability import transition_probabilities
from ftm.surrogate import _infer_state_layout, project_predicted_next_state


class DecisionFeaturesTests(unittest.TestCase):
    def make_context(self, n_nodes=4, n_tasks=6):
        config = SimulationConfig(n_nodes=n_nodes, n_tasks=n_tasks,
                                  k_required=min(3, n_nodes))
        env = MigrationEnvironment(config, scenario_by_name("normal", config))
        env.reset(42)
        return env, env.get_decision_context()

    def test_values_match_baselines_and_invalid_targets_are_zero(self):
        env, context = self.make_context()
        task = context.get_focus_task()
        aux = encode_candidate_features(context)
        observation = encode_observation(context)
        self.assertEqual(len(observation), 325 + 8 * env.config.n_nodes)
        self.assertEqual(observation, env.observation())
        self.assertEqual(observation[-len(aux):], aux)
        self.assertTrue(all(math.isfinite(value) for value in observation))
        time_ref = max(1.0, context.public_config.max_time_seconds)
        stay = estimate_execution_time(context, task, task.node_id)
        for index, allowed in enumerate(context.action_mask[:-1]):
            row = aux[index * AUX_FEATURES_PER_NODE:(index + 1) * AUX_FEATURES_PER_NODE]
            if not allowed:
                self.assertEqual(row, (0.0,) * AUX_FEATURES_PER_NODE)
                continue
            migration = estimate_migration_time(context, task, index)
            execution = estimate_execution_time(context, task, index, migration)
            total = estimate_completion_time(context, task, index)
            self.assertAlmostEqual(row[0], math.asinh(migration / time_ref))
            self.assertAlmostEqual(row[1], math.asinh(execution / time_ref))
            self.assertAlmostEqual(row[2], math.asinh(total / time_ref))
            self.assertAlmostEqual(row[3], math.asinh((stay - total) / time_ref))
            node = context.nodes[index]
            self.assertAlmostEqual(row[4], node.cpu_utilization
                                   + task.cpu_need / node.cpu_capacity)
            self.assertAlmostEqual(row[5], node.memory_utilization
                                   + task.memory_need / node.memory_capacity)
            self.assertAlmostEqual(row[6], math.asinh(stay / time_ref))
            self.assertEqual(row[7], 1.0)

    def test_missing_focus_interrupted_and_no_targets_keep_fixed_length(self):
        _, context = self.make_context(2, 1)
        expected_length = len(encode_observation(context))
        no_focus = replace(context, focus_task_id=None)
        self.assertEqual(encode_candidate_features(no_focus), (0.0,) * 16)
        interrupted = replace(context, tasks=tuple(
            replace(task, status="interrupted") if task.task_id == context.focus_task_id
            else task for task in context.tasks))
        running_state = encode_observation(context)
        interrupted_state = encode_observation(interrupted)
        self.assertNotEqual(running_state, interrupted_state)
        self.assertIn((1.0, 0.0), zip(running_state, interrupted_state))
        self.assertIn((0.0, 1.0), zip(running_state, interrupted_state))
        interrupted_task = interrupted.get_focus_task()
        wait = estimate_wait_time(interrupted, interrupted_task)
        restore = math.ceil(context.public_config.restore_delay_seconds
                            / context.public_config.dt_seconds) * context.public_config.dt_seconds
        self.assertAlmostEqual(wait, restore
                               + estimate_execution_time(
                                   interrupted, interrupted_task,
                                   interrupted_task.node_id,
                                   restore))
        self.assertTrue(any(encode_candidate_features(interrupted)[i * 8 + 3] != 0.0
                            for i, legal in enumerate(interrupted.action_mask[:-1]) if legal))
        no_targets = replace(context, action_mask=(False, False, True))
        self.assertEqual(encode_candidate_features(no_targets), (0.0,) * 16)
        for variant in (no_focus, interrupted, no_targets):
            observation = encode_observation(variant)
            self.assertEqual(len(observation), expected_length)
            self.assertTrue(all(math.isfinite(value) for value in observation))

    def test_down_source_wait_and_impossible_recovery(self):
        _, context = self.make_context(2, 1)
        task = context.get_focus_task()
        source = context.nodes[task.node_id]
        interrupted = replace(context,
            nodes=tuple(replace(node, online=False, repair_rate_estimate=0.25)
                        if node.node_id == source.node_id else node
                        for node in context.nodes),
            tasks=(replace(task, status="interrupted"),))
        wait = estimate_wait_time(interrupted, interrupted.get_focus_task())
        probability = transition_probabilities(
            interrupted.nodes[source.node_id].failure_rate_estimate,
            0.25, context.public_config.dt_seconds)[1]
        recovery = context.public_config.dt_seconds / probability + math.ceil(
            context.public_config.restore_delay_seconds
            / context.public_config.dt_seconds) * context.public_config.dt_seconds
        self.assertAlmostEqual(wait, recovery + estimate_execution_time(
            interrupted, interrupted.get_focus_task(), source.node_id, recovery))
        aux = encode_candidate_features(interrupted)
        target = next(i for i, legal in enumerate(interrupted.action_mask[:-1]) if legal)
        self.assertAlmostEqual(aux[target * 8 + 3], math.asinh(
            (wait - estimate_completion_time(interrupted,
                interrupted.get_focus_task(), target))
            / context.public_config.max_time_seconds))
        impossible = replace(interrupted, nodes=tuple(
            replace(node, repair_rate_estimate=0.0) if node.node_id == source.node_id
            else node for node in interrupted.nodes))
        self.assertTrue(math.isinf(estimate_wait_time(impossible,
                                                     impossible.get_focus_task())))
        aux = encode_candidate_features(impossible)
        self.assertEqual(aux[target * 8 + 6:target * 8 + 8], (0.0, 0.0))
        self.assertTrue(all(math.isfinite(value) for value in encode_observation(impossible)))

    @unittest.skipUnless(find_spec("torch"), "PyTorch learning extra is not installed")
    def test_surrogate_keeps_mask_position_and_aux_gradient(self):
        import torch

        _, context = self.make_context(2, 1)
        observation = encode_observation(context)
        self.assertEqual(_infer_state_layout(len(observation), 3), (2, 1))
        base_dim = len(observation) - 16
        prediction = torch.tensor(observation, dtype=torch.float32)
        prediction[base_dim:base_dim + 8] = torch.tensor(
            [-1.0, 0.5, 0.7, -0.25, -2.0, 0.8, 0.4, 2.0])
        prediction.requires_grad_()
        projected = project_predicted_next_state(prediction.unsqueeze(0), 3)
        self.assertEqual(projected.shape[-1], len(observation))
        self.assertTrue(torch.all((projected[0, base_dim - 3:base_dim] == 0.0)
                                  | (projected[0, base_dim - 3:base_dim] == 1.0)))
        for index, allowed in enumerate(context.action_mask[:-1]):
            row = projected[0, base_dim + index * 8:base_dim + (index + 1) * 8]
            if not allowed:
                self.assertTrue(torch.all(row == 0.0))
        self.assertTrue(torch.isfinite(projected).all().item())
        projected.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all().item())
        self.assertGreater(float(prediction.grad[base_dim:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
