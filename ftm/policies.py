"""Simple migration baselines and a conservative worst-case heuristic."""

from __future__ import annotations

import random
from typing import Protocol

from .environment import MigrationEnvironment
from .entities import Task, TaskStatus
from .reliability import node_up_probability


class Policy(Protocol):
    name: str

    def act(self, env: MigrationEnvironment) -> int: ...


def _wait_action(env: MigrationEnvironment) -> int:
    return env.config.n_nodes


class NoMigrationPolicy:
    name = "no_migration"

    def act(self, env: MigrationEnvironment) -> int:
        return _wait_action(env)


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, env: MigrationEnvironment) -> int:
        valid = [i for i, ok in enumerate(env.valid_action_mask()) if ok]
        return self.rng.choice(valid) if valid else _wait_action(env)


class GreedyPolicy:
    """Choose the target with the lowest estimated transfer + execution time."""

    name = "greedy"

    def score(self, env: MigrationEnvironment, task: Task, target_id: int) -> float:
        node = env.cluster.nodes[target_id]
        transfer = env.cluster.network.estimate_transfer_seconds(
            task.node_id, target_id, task.checkpoint_size_bytes
        )
        compute_rate = max(0.05, node.cpu_rate * task.cpu_need
                           * (1.0 - node.external_cpu_fraction))
        return transfer + env.config.restore_delay_seconds + task.remaining_work / compute_rate

    def act(self, env: MigrationEnvironment) -> int:
        task = env.focus_task
        if task is None:
            return _wait_action(env)
        candidates = [i for i, ok in enumerate(env.manager.valid_targets(task, env.cluster)) if ok]
        if not candidates:
            return _wait_action(env)
        target = min(candidates, key=lambda i: (self.score(env, task, i), i))
        best = self.score(env, task, target)
        if task.status == TaskStatus.RUNNING:
            source = env.cluster.nodes[task.node_id]
            source_rate = max(0.05, source.cpu_rate * task.cpu_need
                              * (1.0 - source.external_cpu_fraction))
            stay = task.remaining_work / source_rate
            if best >= stay:
                return _wait_action(env)
        return target


class LoadBasedPolicy(GreedyPolicy):
    name = "load_based"

    def act(self, env: MigrationEnvironment) -> int:
        task = env.focus_task
        if task is None:
            return _wait_action(env)
        candidates = [i for i, ok in enumerate(env.manager.valid_targets(task, env.cluster)) if ok]
        if not candidates:
            return _wait_action(env)
        target = min(candidates, key=lambda i: (
            env.cluster.nodes[i].cpu_utilization + env.cluster.nodes[i].memory_utilization,
            self.score(env, task, i), i
        ))
        if task.status == TaskStatus.RUNNING:
            source = env.cluster.nodes[task.node_id]
            source_rate = max(0.05, source.cpu_rate * task.cpu_need
                              * (1.0 - source.external_cpu_fraction))
            stay_seconds = task.remaining_work / source_rate
            if self.score(env, task, target) >= stay_seconds * 0.9:
                return _wait_action(env)
        return target


class RobustHeuristicPolicy(GreedyPolicy):
    """Finite conservative baseline; it is not the learned robust-MDP policy."""

    name = "robust_heuristic"

    def score(self, env: MigrationEnvironment, task: Task, target_id: int) -> float:
        node = env.cluster.nodes[target_id]
        link = (None if task.node_id is None
                else env.cluster.network.links[task.node_id][target_id])
        bandwidth = env.config.link_bandwidth_bytes_per_second * 0.35
        delay = env.config.congested_delay_mean_seconds + 2 * env.config.congested_delay_std_seconds
        transfer = env.config.checkpoint_fixed_delay_seconds + env.config.restore_delay_seconds
        transfer += task.checkpoint_size_bytes / bandwidth
        transfer += delay if link is not None else 0.0
        time_to_complete = task.remaining_work / max(
            0.05, node.cpu_rate * task.cpu_need * (1 - env.scenario.load_burst_cpu)
        )
        target_up_prob = node_up_probability(True,
                                             node.failure_rate * max(1.0,
                                                                     env.scenario.failure_multiplier),
                                             node.repair_rate,
                                             transfer + time_to_complete)
        risk_penalty = (1.0 - target_up_prob) * max(1.0, time_to_complete)
        return transfer + time_to_complete + risk_penalty

    def act(self, env: MigrationEnvironment) -> int:
        task = env.focus_task
        if task is None:
            return _wait_action(env)
        candidates = [i for i, ok in enumerate(env.manager.valid_targets(task, env.cluster)) if ok]
        if not candidates:
            return _wait_action(env)
        target = min(candidates, key=lambda i: (self.score(env, task, i), i))
        if task.status == TaskStatus.RUNNING:
            source = env.cluster.nodes[task.node_id]
            stay_seconds = task.remaining_work / max(
                0.05, source.cpu_rate * task.cpu_need * (1.0 - source.external_cpu_fraction)
            )
            if self.score(env, task, target) >= 0.8 * stay_seconds:
                return _wait_action(env)
        return target
