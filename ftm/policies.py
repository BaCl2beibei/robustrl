"""Simple migration baselines and a conservative worst-case heuristic."""

from __future__ import annotations

import random
from typing import Protocol

from .observation import DecisionContext, TaskObservation
from .reliability import node_up_probability


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


def _predicted_execution_time(context: DecisionContext, task: TaskObservation,
                              node_id: int, migration_seconds: float = 0.0) -> float:
    node = context.nodes[node_id]
    burst_remaining = max(
        0.0, node.burst_remaining_estimate - migration_seconds
    )
    current_load = (node.external_cpu_fraction if burst_remaining > 0
                    else node.background_cpu_estimate)
    return predict_execution_time(
        task.remaining_work_estimate, node.cpu_rate * task.cpu_need, current_load,
        node.background_cpu_estimate, burst_remaining,
    )


class Policy(Protocol):
    name: str

    def act(self, context: DecisionContext) -> int: ...


def _wait_action(context: DecisionContext) -> int:
    return len(context.nodes)


class NoMigrationPolicy:
    name = "no_migration"

    def act(self, context: DecisionContext) -> int:
        return _wait_action(context)


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, context: DecisionContext) -> int:
        valid = [i for i, ok in enumerate(context.action_mask) if ok]
        return self.rng.choice(valid) if valid else _wait_action(context)


class GreedyPolicy:
    """Choose the target with the lowest estimated transfer + execution time."""

    name = "greedy"

    def score(self, context: DecisionContext, task: TaskObservation,
              target_id: int) -> float:
        if task.node_id is None:
            transfer = (task.checkpoint_size_bytes
                        / context.public_config.link_bandwidth_bytes_per_second)
        else:
            link = context.link(task.node_id, target_id)
            transfer = (link.rtt_seconds + task.checkpoint_size_bytes
                        / max(1e-9, link.effective_bandwidth))
        migration_seconds = (transfer
                             + context.public_config.checkpoint_fixed_delay_seconds
                             + context.public_config.restore_delay_seconds)
        execution_seconds = _predicted_execution_time(
            context, task, target_id, migration_seconds
        )
        return migration_seconds + execution_seconds

    def act(self, context: DecisionContext) -> int:
        task = context.get_focus_task()
        if task is None:
            return _wait_action(context)
        candidates = [i for i, ok in enumerate(context.action_mask[:-1]) if ok]
        if not candidates:
            return _wait_action(context)
        target = min(candidates, key=lambda i: (self.score(context, task, i), i))
        best = self.score(context, task, target)
        if task.status == "running" and task.node_id is not None:
            stay = _predicted_execution_time(context, task, task.node_id)
            if best >= stay:
                return _wait_action(context)
        return target


class LoadBasedPolicy(GreedyPolicy):
    name = "load_based"

    def act(self, context: DecisionContext) -> int:
        task = context.get_focus_task()
        if task is None:
            return _wait_action(context)
        candidates = [i for i, ok in enumerate(context.action_mask[:-1]) if ok]
        if not candidates:
            return _wait_action(context)
        target = min(candidates, key=lambda i: (
            context.nodes[i].cpu_utilization + context.nodes[i].memory_utilization,
            self.score(context, task, i), i
        ))
        if task.status == "running" and task.node_id is not None:
            stay_seconds = _predicted_execution_time(context, task, task.node_id)
            if self.score(context, task, target) >= stay_seconds * 0.9:
                return _wait_action(context)
        return target


class RobustHeuristicPolicy(GreedyPolicy):
    """Finite conservative baseline; it is not the learned robust-MDP policy."""

    name = "robust_heuristic"

    def score(self, context: DecisionContext, task: TaskObservation,
              target_id: int) -> float:
        node = context.nodes[target_id]
        link = (None if task.node_id is None else context.link(task.node_id, target_id))
        bandwidth = context.public_config.link_bandwidth_bytes_per_second * 0.35
        delay = (context.public_config.congested_delay_mean_seconds
                 + 2 * context.public_config.congested_delay_std_seconds)
        transfer = (context.public_config.checkpoint_fixed_delay_seconds
                    + context.public_config.restore_delay_seconds)
        transfer += task.checkpoint_size_bytes / bandwidth
        transfer += delay if link is not None else 0.0
        migration_seconds = transfer
        time_to_complete = _predicted_execution_time(
            context, task, target_id, migration_seconds
        )
        target_up_prob = node_up_probability(True,
                                             node.failure_rate_estimate,
                                             node.repair_rate_estimate,
                                             migration_seconds + time_to_complete)
        risk_penalty = (1.0 - target_up_prob) * max(1.0, time_to_complete)
        return migration_seconds + time_to_complete + risk_penalty

    def act(self, context: DecisionContext) -> int:
        task = context.get_focus_task()
        if task is None:
            return _wait_action(context)
        candidates = [i for i, ok in enumerate(context.action_mask[:-1]) if ok]
        if not candidates:
            return _wait_action(context)
        target = min(candidates, key=lambda i: (self.score(context, task, i), i))
        if task.status == "running" and task.node_id is not None:
            stay_seconds = _predicted_execution_time(context, task, task.node_id)
            if self.score(context, task, target) >= 0.8 * stay_seconds:
                return _wait_action(context)
        return target
