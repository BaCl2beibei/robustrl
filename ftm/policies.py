"""Simple migration baselines and a conservative worst-case heuristic."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Protocol

from .entities import Task, TaskStatus
from .reliability import node_up_probability

if TYPE_CHECKING:
    from .environment import MigrationEnvironment


def predict_burst_remaining(elapsed_seconds: float,
                            historical_durations: tuple[float, ...] | list[float],
                            default_duration: float = 5.0) -> float:
    """Estimate remaining time using completed burst durations conditional on age."""
    if elapsed_seconds < 0 or default_duration < 0:
        raise ValueError("burst times must be non-negative")
    remaining_samples = [duration - elapsed_seconds for duration in historical_durations
                         if duration > elapsed_seconds]
    if remaining_samples:
        return sum(remaining_samples) / len(remaining_samples)
    return max(0.0, default_duration - elapsed_seconds)


def predict_execution_time(remaining_work: float, base_rate: float,
                           current_external_load: float, normal_external_load: float,
                           burst_remaining_seconds: float) -> float:
    """Estimate task time across the predicted end of the current load burst."""
    if remaining_work < 0 or burst_remaining_seconds < 0:
        raise ValueError("work and burst duration must be non-negative")
    rate_burst = max(0.05, base_rate * (1.0 - current_external_load))
    rate_normal = max(0.05, base_rate * (1.0 - normal_external_load))
    burst_work = rate_burst * burst_remaining_seconds
    if remaining_work <= burst_work:
        return remaining_work / rate_burst
    return burst_remaining_seconds + (remaining_work - burst_work) / rate_normal


def _predicted_execution_time(env: MigrationEnvironment, task: Task,
                              node_id: int, migration_seconds: float = 0.0) -> float:
    node = env.cluster.nodes[node_id]
    burst_remaining = max(
        0.0, env.disturbances.predict_burst_remaining_seconds(node_id) - migration_seconds
    )
    current_load = (node.external_cpu_fraction if burst_remaining > 0
                    else node.normal_external_cpu_fraction)
    return predict_execution_time(
        task.remaining_work, node.cpu_rate * task.cpu_need, current_load,
        node.normal_external_cpu_fraction, burst_remaining,
    )


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
        migration_seconds = transfer + env.config.restore_delay_seconds
        execution_seconds = _predicted_execution_time(
            env, task, target_id, migration_seconds
        )
        return migration_seconds + execution_seconds

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
            stay = _predicted_execution_time(env, task, task.node_id)
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
            stay_seconds = _predicted_execution_time(env, task, task.node_id)
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
        migration_seconds = transfer
        time_to_complete = _predicted_execution_time(
            env, task, target_id, migration_seconds
        )
        target_up_prob = node_up_probability(True,
                                             node.failure_rate,
                                             node.repair_rate,
                                             migration_seconds + time_to_complete)
        risk_penalty = (1.0 - target_up_prob) * max(1.0, time_to_complete)
        return migration_seconds + time_to_complete + risk_penalty

    def act(self, env: MigrationEnvironment) -> int:
        task = env.focus_task
        if task is None:
            return _wait_action(env)
        candidates = [i for i, ok in enumerate(env.manager.valid_targets(task, env.cluster)) if ok]
        if not candidates:
            return _wait_action(env)
        target = min(candidates, key=lambda i: (self.score(env, task, i), i))
        if task.status == TaskStatus.RUNNING:
            stay_seconds = _predicted_execution_time(env, task, task.node_id)
            if self.score(env, task, target) >= 0.8 * stay_seconds:
                return _wait_action(env)
        return target
