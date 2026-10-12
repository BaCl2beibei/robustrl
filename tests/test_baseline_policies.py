"""The heuristic baselines optimize their documented public objectives."""

import unittest
from dataclasses import replace

from ftm.observation import (DecisionContext, LinkObservation, NodeObservation,
                             PublicConfig, TaskObservation)
from ftm.main import POLICIES, make_policy
from ftm.policies import (ResourceMinPolicy, TimeMinPolicy,
                          TimeResourceMinPolicy, estimate_completion_time,
                          estimate_resource_score, valid_targets)


def make_context(status: str = "running",
                 action_mask: tuple[bool, ...] = (False, True, True, True)
                 ) -> DecisionContext:
    nodes = (
        NodeObservation(0, True, 4.0, 16.0, 2.0, 8.0, 0.8, 0.8,
                        0.0, 0.0, 0.0, 0.5, 0.5, 1.0),
        NodeObservation(1, True, 4.0, 16.0, 2.0, 8.0, 0.7, 0.7,
                        0.0, 0.0, 0.0, 0.0, 0.5, 4.0),
        NodeObservation(2, True, 4.0, 16.0, 2.0, 8.0, 0.1, 0.1,
                        0.0, 0.0, 0.0, 0.5, 0.0, 0.5),
    )
    links = tuple(
        LinkObservation(i, j, True,
                        0.1 if j == 1 else 0.2,
                        10_000_000.0)
        for i in range(3) for j in range(3)
    )
    task = TaskObservation(
        7, 1.0, 4.0, 20.0, 10.0, 100.0, 0, 0.0, 0, 0.0,
        status, 0, None, 0, False,
    )
    public = PublicConfig(600.0, 8, 0.1, 0.1, 0.18, 0.08,
                          25_000_000.0)
    return DecisionContext(5.0, 7, nodes, (task,), links, action_mask, public)


class BaselinePolicyTests(unittest.TestCase):
    def test_new_baselines_are_registered(self):
        self.assertEqual(POLICIES[1:], (
            "no_migration", "random", "time_min", "resource_min",
            "time_resource_min",
        ))
        self.assertIsInstance(make_policy("time_min", 1), TimeMinPolicy)
        self.assertIsInstance(make_policy("resource_min", 1), ResourceMinPolicy)
        self.assertIsInstance(make_policy("time_resource_min", 1),
                              TimeResourceMinPolicy)

    def test_time_min_uses_completion_time_not_resource_score(self):
        context = make_context()
        task = context.get_focus_task()
        self.assertGreater(estimate_resource_score(context, task, 1),
                           estimate_resource_score(context, task, 2))
        self.assertLess(estimate_completion_time(context, task, 1),
                        estimate_completion_time(context, task, 2))
        self.assertEqual(TimeMinPolicy().act(context), 1)

    def test_resource_min_uses_post_migration_utilization_only(self):
        context = make_context()
        task = context.get_focus_task()
        self.assertLess(estimate_resource_score(context, task, 2),
                        estimate_resource_score(context, task, 1))
        # Node 2 is slower than staying on node 0, but resource score governs B4.
        self.assertGreater(estimate_completion_time(context, task, 2), 10.0)
        self.assertEqual(ResourceMinPolicy().act(context), 2)

    def test_joint_score_uses_fixed_normalization_and_alpha(self):
        context = make_context()
        task = context.get_focus_task()
        policy = TimeResourceMinPolicy(alpha=0.25, time_reference_seconds=10.0)
        expected = (0.25 * estimate_completion_time(context, task, 2) / 10.0
                    + 0.75 * estimate_resource_score(context, task, 2) / 2.0)
        self.assertAlmostEqual(policy.score(context, task, 2), expected)
        self.assertEqual(policy.act(context), 2)

    def test_joint_score_does_not_use_failure_probability_estimates(self):
        context = make_context()
        changed_nodes = tuple(
            replace(node,
                    failure_rate_estimate=10_000.0 if node.node_id == 1 else 0.0,
                    repair_rate_estimate=0.0 if node.node_id == 1 else 10_000.0)
            for node in context.nodes
        )
        changed = replace(context, nodes=changed_nodes)
        task = context.get_focus_task()
        changed_task = changed.get_focus_task()
        policy = TimeResourceMinPolicy()
        for target in (1, 2):
            self.assertEqual(policy.score(context, task, target),
                             policy.score(changed, changed_task, target))

    def test_running_task_waits_when_no_objective_improves(self):
        context = make_context()
        source = replace(context.nodes[0], cpu_utilization=0.1,
                         memory_utilization=0.1, cpu_rate=10.0)
        slow_loaded_targets = (
            replace(context.nodes[1], cpu_utilization=0.7,
                    memory_utilization=0.7, cpu_rate=0.1),
            replace(context.nodes[2], cpu_utilization=0.8,
                    memory_utilization=0.8, cpu_rate=0.1),
        )
        context = replace(context, nodes=(source, *slow_loaded_targets))
        for policy in (TimeMinPolicy(), ResourceMinPolicy(), TimeResourceMinPolicy()):
            self.assertEqual(policy.act(context), len(context.nodes))

    def test_interrupted_task_uses_legal_target_and_waits_if_none(self):
        interrupted = make_context(status="interrupted")
        self.assertEqual(valid_targets(interrupted), [1, 2])
        for policy in (TimeMinPolicy(), ResourceMinPolicy(), TimeResourceMinPolicy()):
            self.assertIn(policy.act(interrupted), (1, 2))

        no_target = replace(interrupted, action_mask=(False, False, False, True))
        self.assertEqual(valid_targets(no_target), [])
        for policy in (TimeMinPolicy(), ResourceMinPolicy(),
                       TimeResourceMinPolicy()):
            self.assertEqual(policy.act(no_target), len(no_target.nodes))

    def test_scores_accept_only_the_public_decision_context(self):
        context = make_context()
        task = context.get_focus_task()
        for policy in (TimeMinPolicy(), ResourceMinPolicy(),
                       TimeResourceMinPolicy()):
            self.assertIsInstance(context, DecisionContext)
            self.assertTrue(context.action_mask[policy.act(context)])
            self.assertGreaterEqual(policy.score(context, task, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
