"""Simple migration baselines with time and resource objectives."""

from __future__ import annotations

import random
import math
from collections.abc import Callable
from typing import Protocol

from .observation import DecisionContext, TaskObservation
from .reliability import transition_probabilities


# Fixed from the project's default training horizon; keep it unchanged for OOD.
TIME_REFERENCE_SECONDS = 600.0
MIN_RELATIVE_IMPROVEMENT = 1e-3


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


def estimate_execution_time(context: DecisionContext, task: TaskObservation,
                            node_id: int, migration_seconds: float = 0.0) -> float:
    """Estimate remaining execution time from the shared public observation."""
    return _predicted_execution_time(context, task, node_id, migration_seconds)


def estimate_wait_time(context: DecisionContext, task: TaskObservation) -> float:
    """Expected time to finish on the source, including interrupted recovery.

    A down node repairs at its observed rate. The simulator then starts a
    restore delay before execution resumes. A zero repair rate means waiting
    cannot complete; an already repaired task may still be restoring.
    """
    if task.node_id is None or not 0 <= task.node_id < len(context.nodes):
        return math.inf
    dt = context.public_config.dt_seconds
    restore = context.public_config.restore_delay_seconds
    restore_ticks = math.ceil(restore / dt)
    source = context.nodes[task.node_id]
    if task.status == "running":
        if not source.online:
            return math.inf
        recovery = 0.0
    elif task.status == "interrupted":
        if source.online:
            # The public snapshot does not expose the current restore timer.
            recovery = restore_ticks * dt
        else:
            repair_probability = transition_probabilities(
                max(0.0, source.failure_rate_estimate),
                max(0.0, source.repair_rate_estimate), dt)[1]
            if repair_probability <= 0.0:
                return math.inf
            recovery = dt / repair_probability + restore_ticks * dt
    else:
        return math.inf
    return recovery + _predicted_execution_time(context, task, task.node_id, recovery)


def estimate_migration_time(context: DecisionContext, task: TaskObservation,
                            target_id: int) -> float:
    """Estimate checkpoint transfer, network, and restore time from public data."""
    if task.node_id is None:
        transfer_seconds = (task.checkpoint_size_bytes
                            / max(1e-9, context.public_config.link_bandwidth_bytes_per_second))
    else:
        link = context.link(task.node_id, target_id)
        transfer_seconds = (link.rtt_seconds + task.checkpoint_size_bytes
                            / max(1e-9, link.effective_bandwidth))
    dt = context.public_config.dt_seconds
    transfer_ticks = max(1, math.ceil(
        (transfer_seconds + context.public_config.checkpoint_fixed_delay_seconds) / dt))
    # MigrationManager enters RESTORING after transfer and confirms on a later tick.
    restore_ticks = max(1, math.ceil(context.public_config.restore_delay_seconds / dt))
    return (transfer_ticks + restore_ticks) * dt


def estimate_completion_time(context: DecisionContext, task: TaskObservation,
                             target_id: int) -> float:
    """Estimate migration time plus remaining execution time on a target."""
    migration_seconds = estimate_migration_time(context, task, target_id)
    execution_seconds = _predicted_execution_time(
        context, task, target_id, migration_seconds
    )
    return migration_seconds + execution_seconds


def estimate_resource_score(context: DecisionContext, task: TaskObservation,
                            target_id: int) -> float:
    """Return post-migration CPU plus memory utilization using public observations."""
    node = context.nodes[target_id]
    cpu_after = node.cpu_utilization + task.cpu_need / max(1e-9, node.cpu_capacity)
    memory_after = (node.memory_utilization
                    + task.memory_need / max(1e-9, node.memory_capacity))
    return cpu_after + memory_after


def valid_targets(context: DecisionContext) -> list[int]:
    """Return migration targets allowed by the shared public action mask."""
    return [index for index, allowed in enumerate(context.action_mask[:-1]) if allowed]


class Policy(Protocol):
    name: str

    def act(self, context: DecisionContext) -> int: ...


def _wait_action(context: DecisionContext) -> int:
    return len(context.nodes)


def _score_improves(candidate: float, stay: float) -> bool:
    """Require a small relative gain before moving an already-running task."""
    margin = MIN_RELATIVE_IMPROVEMENT * max(1.0, abs(stay))
    return candidate < stay - margin


def _act_by_score(context: DecisionContext, score: Callable[[DecisionContext,
                                                              TaskObservation, int], float],
                  stay_score: Callable[[DecisionContext, TaskObservation], float]) -> int:
    task = context.get_focus_task()
    if task is None:
        return _wait_action(context)
    candidates = valid_targets(context)
    if not candidates:
        return _wait_action(context)

    best_score, target = min((score(context, task, index), index)
                             for index in candidates)
    if (task.status == "running" and task.node_id is not None
            and not _score_improves(best_score, stay_score(context, task))):
        return _wait_action(context)
    return target


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


class TimeMinPolicy:
    """Choose the target with minimum estimated migration plus execution time."""

    name = "time_min"

    def score(self, context: DecisionContext, task: TaskObservation,
              target_id: int) -> float:
        return estimate_completion_time(context, task, target_id)

    def act(self, context: DecisionContext) -> int:
        return _act_by_score(
            context, self.score,
            lambda ctx, task: _predicted_execution_time(ctx, task, task.node_id),
        )


class ResourceMinPolicy:
    """Choose the target with minimum post-migration CPU and memory utilization."""

    name = "resource_min"

    def score(self, context: DecisionContext, task: TaskObservation,
              target_id: int) -> float:
        return estimate_resource_score(context, task, target_id)

    def act(self, context: DecisionContext) -> int:
        return _act_by_score(
            context, self.score,
            lambda ctx, task: (ctx.nodes[task.node_id].cpu_utilization
                               + ctx.nodes[task.node_id].memory_utilization),
        )


class TimeResourceMinPolicy:
    """Minimize a normalized weighted sum of time and target resource use."""

    name = "time_resource_min"

    def __init__(self, alpha: float = 0.5,
                 time_reference_seconds: float = TIME_REFERENCE_SECONDS) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if time_reference_seconds <= 0:
            raise ValueError("time_reference_seconds must be positive")
        self.alpha = alpha
        self.time_reference_seconds = time_reference_seconds

    def score(self, context: DecisionContext, task: TaskObservation,
              target_id: int) -> float:
        normalized_time = (estimate_completion_time(context, task, target_id)
                           / self.time_reference_seconds)
        normalized_resource = estimate_resource_score(context, task, target_id) / 2.0
        return (self.alpha * normalized_time
                + (1.0 - self.alpha) * normalized_resource)

    def act(self, context: DecisionContext) -> int:
        def stay_score(ctx: DecisionContext, task: TaskObservation) -> float:
            execution_seconds = _predicted_execution_time(ctx, task, task.node_id)
            node = ctx.nodes[task.node_id]
            resource_score = node.cpu_utilization + node.memory_utilization
            return (self.alpha * execution_seconds / self.time_reference_seconds
                    + (1.0 - self.alpha) * resource_score / 2.0)

        return _act_by_score(context, self.score, stay_score)
