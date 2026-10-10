import random
import unittest
from dataclasses import replace
from importlib.util import find_spec

from ftm.cluster import Cluster, task_token
from ftm.config import ScenarioConfig, SimulationConfig
from ftm.disturbances import DisturbanceModel, DisturbanceSample
from ftm.entities import Event, Node, Task, TaskStatus
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.metrics import compute_episode_metrics
from ftm.network import Network
from ftm.policies import predict_burst_remaining, predict_execution_time
from ftm.surrogate import _infer_state_layout, project_predicted_next_state


class ResearchLogicUpdateTests(unittest.TestCase):
    def test_duration_prediction_uses_only_long_enough_history(self):
        self.assertEqual(predict_burst_remaining(6.0, [4.0, 5.0, 8.0, 10.0]), 3.0)
        self.assertEqual(predict_burst_remaining(6.0, [], default_duration=10.0), 4.0)

    def test_execution_time_splits_burst_and_normal_load(self):
        predicted = predict_execution_time(50.0, 10.0, 0.5, 0.1, 4.0)
        self.assertAlmostEqual(predicted, 4.0 + 30.0 / 9.0)

    def test_load_scenarios_use_the_documented_duration_ranges(self):
        config = SimulationConfig()
        expected = {
            "load_short": (2.0, 5.0),
            "load_medium": (10.0, 30.0),
            "load_sustained": (30.0, 120.0),
        }
        for name, bounds in expected.items():
            self.assertEqual(scenario_by_name(name, config).load_burst_duration_bounds(),
                             bounds)

    def test_burst_duration_is_sampled_and_recorded_in_seconds(self):
        config = SimulationConfig(n_nodes=2, k_required=2, base_failure_rate=0.0,
                                  base_repair_rate=0.0, dt_seconds=1.0)
        scenario = ScenarioConfig(name="fixed_burst", load_burst_probability=1.0,
                                  load_burst_duration_min_seconds=2.0,
                                  load_burst_duration_max_seconds=2.0)
        nodes = [Node(i, 8.0, 32.0, 1.0, 0.0, 0.0) for i in range(2)]
        disturbances = DisturbanceModel(config, scenario, random.Random(10))

        first = disturbances.sample(nodes, 1.0)
        self.assertTrue(first.load_burst)
        self.assertEqual(disturbances.burst_remaining, [1.0, 1.0])
        self.assertEqual(disturbances.predict_burst_remaining_seconds(0), 4.0)
        disturbances.sample(nodes, 1.0)
        self.assertEqual(disturbances.burst_history_seconds, [[2.0], [2.0]])

    def test_burst_durations_vary_within_scenario_bounds(self):
        config = SimulationConfig(n_nodes=2, k_required=2, base_failure_rate=0.0,
                                  base_repair_rate=0.0)
        scenario = ScenarioConfig(name="variable_burst", load_burst_probability=1.0,
                                  load_burst_duration_min_seconds=2.0,
                                  load_burst_duration_max_seconds=5.0)
        sampled = []
        for seed in range(8):
            model = DisturbanceModel(config, scenario, random.Random(seed))
            nodes = [Node(i, 8.0, 32.0, 1.0, 0.0, 0.0) for i in range(2)]
            model.sample(nodes, 1.0)
            sampled.append(model.burst_total_seconds[0])
        self.assertTrue(all(2.0 <= duration <= 5.0 for duration in sampled))
        self.assertGreater(len(set(sampled)), 1)

    def test_network_scenario_updates_symmetric_link_reachability(self):
        config = SimulationConfig(n_nodes=2, k_required=2, base_failure_rate=0.0,
                                  base_repair_rate=0.0)
        scenario = ScenarioConfig(name="partition", congestion_enter=0.0,
                                  disconnect_enter=1.0, disconnect_leave=0.0)
        model = DisturbanceModel(config, scenario, random.Random(2))
        network = Network(config)
        model._advance_network_regimes(False)
        network.update(model)
        self.assertFalse(network.links[0][1].reachable)
        self.assertFalse(network.links[1][0].reachable)

        model.scenario = replace(scenario, disconnect_enter=0.0, disconnect_leave=1.0)
        model._advance_network_regimes(False)
        network.update(model)
        self.assertTrue(network.links[0][1].reachable)
        self.assertTrue(network.links[1][0].reachable)

    def test_cpu_background_load_is_soft_but_memory_remains_hard(self):
        node = Node(0, 8.0, 32.0, 1.0, 0.0, 0.0,
                    external_cpu_fraction=0.95)
        self.assertTrue(node.can_host(1.0, 1.0))
        node.external_memory = 32.0
        self.assertFalse(node.can_host(1.0, 1.0))

    def test_node_task_count_observation_is_normalized(self):
        config = SimulationConfig(n_nodes=2, n_tasks=3, k_required=1)
        nodes = [Node(i, 8.0, 32.0, 1.0, 0.0, 0.0) for i in range(2)]
        tasks = [Task(i, 1.0, 1.0, 10.0, 20.0, 10, 5.0,
                      status=TaskStatus.RUNNING, node_id=0) for i in range(3)]
        for task in tasks:
            nodes[0].reserve(task.task_id, task.cpu_need, task.memory_need,
                             task_token(task.task_id))
        cluster = Cluster(config, nodes, tasks, Network(config))

        state = cluster.state_vector(False, config.max_time_seconds)
        self.assertAlmostEqual(state[4], 1.0)
        self.assertEqual(state[12], 0.0)

    def test_observation_preserves_unbounded_values_and_focus(self):
        config = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                  max_time_seconds=1.0,
                                  task_cpu_min=1.0, task_cpu_max=4.0)
        env = MigrationEnvironment(config, ScenarioConfig(name="quiet"))
        observation, _ = env.reset(seed=29)
        context = env.get_decision_context()
        self.assertEqual(context.focus_task_id, env.focus_task.task_id)
        self.assertEqual(context.action_mask, env.valid_action_mask())
        self.assertEqual(observation[1:2], (1.0,))
        self.assertIn(context.tasks[0].checkpoint_size_bytes / 1_000_000,
                      observation)
        self.assertEqual(_infer_state_layout(len(observation), config.n_nodes + 1),
                         (config.n_nodes, config.n_tasks))

    @unittest.skipUnless(find_spec("torch"), "PyTorch learning extra is not installed")
    def test_surrogate_projection_keeps_categories_legal_and_gradient_live(self):
        import torch

        config = SimulationConfig(n_nodes=2, n_tasks=2, k_required=1)
        env = MigrationEnvironment(config, ScenarioConfig(name="quiet"))
        observation, _ = env.reset(seed=23)
        n_nodes = config.n_nodes
        task_offset = 9 + config.n_tasks + 14 * n_nodes
        status_start = task_offset + 10
        predicted = torch.tensor(observation, dtype=torch.float32).clone()
        predicted[4] = 2.5  # unbounded measured values are retained
        predicted[0] = -2.0
        predicted[status_start:status_start + 6] = torch.tensor(
            [0.1, 0.6, 0.2, 0.1, 0.0, 0.0]
        )
        predicted.requires_grad_()

        projected = project_predicted_next_state(predicted.unsqueeze(0),
                                                   config.n_nodes + 1)
        status = projected[0, status_start:status_start + 6]
        self.assertAlmostEqual(float(status.sum()), 1.0)
        self.assertTrue(torch.all((status == 0.0) | (status == 1.0)).item())
        self.assertTrue(torch.all(projected >= 0.0).item())
        self.assertAlmostEqual(float(projected[0, 4]), 2.5)
        self.assertAlmostEqual(float(projected[0, 0]), 0.0)

        weights = torch.arange(projected.shape[-1], dtype=torch.float32)
        (projected * weights).sum().backward()
        self.assertGreater(float(predicted.grad[status_start:status_start + 6].abs().sum()),
                           0.0)
        self.assertAlmostEqual(float(predicted.grad[4]), 4.0)

    def test_sla_uses_strict_deadline_comparison(self):
        task = Task(0, 1.0, 1.0, 10.0, 5.0, 10, 5.0)
        self.assertFalse(task.has_missed_deadline(5.0))
        self.assertTrue(task.has_missed_deadline(5.01))
        task.completion_time = 5.0
        self.assertFalse(task.has_missed_deadline(10.0))
        task.completion_time = 5.01
        self.assertTrue(task.has_missed_deadline(10.0))

    def test_cluster_checks_sla_for_running_and_nonrunning_tasks(self):
        config = SimulationConfig(n_nodes=2, n_tasks=4, k_required=1,
                                  base_failure_rate=0.0, base_repair_rate=0.0)
        nodes = [Node(i, 8.0, 32.0, 1.0, 0.0, 0.0) for i in range(2)]
        statuses = (TaskStatus.RUNNING, TaskStatus.MIGRATING,
                    TaskStatus.INTERRUPTED, TaskStatus.PENDING)
        tasks = [Task(i, 1.0, 1.0, 10.0, 5.0, 10, 5.0, status=status,
                      node_id=0 if status == TaskStatus.RUNNING else None)
                 for i, status in enumerate(statuses)]
        nodes[0].reserve(tasks[0].task_id, tasks[0].cpu_need, tasks[0].memory_need,
                         task_token(tasks[0].task_id))
        cluster = Cluster(config, nodes, tasks, Network(config))

        events = cluster.advance(1.0, DisturbanceSample(False, (), False), 5.01)
        violated = {event.task_id for event in events if event.kind == "sla_violated"}
        self.assertEqual(violated, set(range(len(statuses))))

    def test_terminal_failure_and_horizon_incomplete_have_equal_base_cost(self):
        task_failure = MigrationEnvironment._event_reward(Event(1.0, "task_failed", 0))
        horizon = MigrationEnvironment._event_reward(Event(1.0, "horizon_incomplete", 0))
        self.assertEqual(task_failure, horizon)
        self.assertEqual(task_failure, -1.0)

    def test_reward_includes_dense_service_loss_and_progress(self):
        interrupted_config = SimulationConfig(
            n_nodes=2, n_tasks=1, k_required=1, max_time_seconds=10.0,
            base_failure_rate=0.0, base_repair_rate=0.0,
            task_work_min=100.0, task_work_max=100.0,
            task_cpu_min=1.0, task_cpu_max=1.0,
        )
        quiet = ScenarioConfig(name="quiet", failure_multiplier=0.0,
                               congestion_enter=0.0, load_burst_probability=0.0)
        interrupted_env = MigrationEnvironment(interrupted_config, quiet)
        interrupted_env.reset(seed=41)
        task = next(iter(interrupted_env.cluster.tasks.values()))
        interrupted_env.cluster.nodes[task.node_id].up = False
        task.status = TaskStatus.INTERRUPTED
        _, reward, _, _, _ = interrupted_env.step(interrupted_config.n_nodes)
        self.assertAlmostEqual(reward, -0.15)
        task.status = TaskStatus.MIGRATING
        self.assertEqual(interrupted_env._service_loss_ratio(), 0.0)

        progress_config = SimulationConfig(
            n_nodes=2, n_tasks=1, k_required=1, max_time_seconds=10.0,
            cpu_rate=100.0, base_failure_rate=0.0, base_repair_rate=0.0,
            task_work_min=1.0, task_work_max=1.0,
            task_cpu_min=1.0, task_cpu_max=1.0,
        )
        progress_env = MigrationEnvironment(progress_config, quiet)
        progress_env.reset(seed=43)
        _, reward, terminated, _, _ = progress_env.step(progress_config.n_nodes)
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, 1.2)

    def test_repaired_node_resumes_interrupted_task_with_existing_reservation(self):
        config = SimulationConfig(n_nodes=2, n_tasks=1, k_required=1,
                                  restore_delay_seconds=1.0)
        nodes = [Node(i, 8.0, 32.0, 1.0, 0.0, 0.0) for i in range(2)]
        task = Task(0, 1.0, 1.0, 10.0, 20.0, 10, 5.0,
                    status=TaskStatus.INTERRUPTED, node_id=0)
        nodes[0].reserve(task.task_id, task.cpu_need, task.memory_need, task_token(task.task_id))
        cluster = Cluster(config, nodes, [task], Network(config))

        cluster.advance(1.0, DisturbanceSample(False, ((0, "node_repair"),), False), 1.0)
        self.assertEqual(task.status, TaskStatus.INTERRUPTED)
        self.assertEqual(len(nodes[0].reservations), 1)
        cluster.advance(1.0, DisturbanceSample(False, (), False), 2.0)

        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertEqual(len(nodes[0].reservations), 1)

    def test_episode_metrics_use_fixed_window_and_separate_task_service(self):
        config = SimulationConfig(
            n_nodes=2, n_tasks=1, k_required=1, max_time_seconds=3.0,
            cpu_rate=100.0, base_failure_rate=0.0, base_repair_rate=0.0,
            task_work_min=1.0, task_work_max=1.0, task_cpu_min=1.0,
            task_cpu_max=1.0, task_memory_min=1.0, task_memory_max=1.0,
            delay_cost_weight=0.0,
        )
        scenario = ScenarioConfig(name="quiet", failure_multiplier=0.0,
                                  congestion_enter=0.0, load_burst_probability=0.0)
        env = MigrationEnvironment(config, scenario)
        env.reset(seed=7)
        _, reward, terminated, truncated, _ = env.step(config.n_nodes)

        metrics = compute_episode_metrics(env, "no_migration", 7, reward)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(metrics.elapsed_seconds, config.max_time_seconds)
        self.assertEqual(metrics.cluster_availability, 1.0)
        self.assertEqual(metrics.task_service_availability, 1.0)
        self.assertGreater(metrics.task_cpu_utilization, 0.0)
        self.assertGreaterEqual(metrics.mean_completion_seconds_horizon_capped, 0.0)

    def test_episode_metrics_report_migrations_cut_off_at_horizon(self):
        config = SimulationConfig(
            n_nodes=2, n_tasks=1, k_required=1, dt_seconds=0.1,
            max_time_seconds=0.2, base_failure_rate=0.0, base_repair_rate=0.0,
            checkpoint_fixed_delay_seconds=1.0,
        )
        scenario = ScenarioConfig(name="quiet", failure_multiplier=0.0,
                                  congestion_enter=0.0, load_burst_probability=0.0)
        env = MigrationEnvironment(config, scenario)
        env.reset(seed=13)
        _, reward, terminated, truncated, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, second_reward, terminated, truncated, _ = env.step(config.n_nodes)
        reward += second_reward

        metrics = compute_episode_metrics(env, "greedy", 13, reward)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(metrics.migration_attempt_count, 1)
        self.assertEqual(metrics.migration_count, 0)
        self.assertEqual(metrics.migration_failure_count, 0)
        self.assertEqual(metrics.migration_incomplete_count, 1)
        self.assertAlmostEqual(metrics.total_migration_seconds, 0.2)


if __name__ == "__main__":
    unittest.main()
