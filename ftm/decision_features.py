"""Public, per-target estimates appended to the PPO observation."""

from __future__ import annotations

import math

from .observation import DecisionContext
from .policies import estimate_execution_time, estimate_migration_time, estimate_wait_time


AUX_FEATURE_ORDER = (
    "migration_time", "execution_time", "move_time", "time_saving",
    "cpu_after", "memory_after", "wait_time", "wait_recoverable",
)
AUX_FEATURES_PER_NODE = len(AUX_FEATURE_ORDER)


def encode_candidate_features(context: DecisionContext) -> tuple[float, ...]:
    """Encode legal migration targets using only the public decision snapshot."""
    n_nodes = len(context.nodes)
    if len(context.action_mask) != n_nodes + 1:
        raise ValueError("unexpected action mask length")
    task = context.get_focus_task()
    if task is None:
        return (0.0,) * (n_nodes * AUX_FEATURES_PER_NODE)

    time_ref = max(1.0, context.public_config.max_time_seconds)
    stay_time = estimate_wait_time(context, task)
    recoverable = math.isfinite(stay_time)

    features: list[float] = []
    for index, node in enumerate(context.nodes):
        if not context.action_mask[index]:
            features.extend((0.0,) * AUX_FEATURES_PER_NODE)
            continue
        migration_time = estimate_migration_time(context, task, index)
        execution_time = estimate_execution_time(context, task, index, migration_time)
        move_time = migration_time + execution_time
        time_saving = stay_time - move_time if recoverable else 0.0
        cpu_after = node.cpu_utilization + task.cpu_need / max(1e-9, node.cpu_capacity)
        memory_after = (node.memory_utilization
                        + task.memory_need / max(1e-9, node.memory_capacity))
        features.extend((
            math.asinh(migration_time / time_ref),
            math.asinh(execution_time / time_ref),
            math.asinh(move_time / time_ref),
            math.asinh(time_saving / time_ref),
            cpu_after, memory_after,
            math.asinh(stay_time / time_ref) if recoverable else 0.0,
            float(recoverable),
        ))
    return tuple(features)
