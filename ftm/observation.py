"""The only information supplied to migration decision makers."""

from __future__ import annotations

from dataclasses import dataclass, fields


SCHEMA_VERSION = "decision-context-v3-recovery-aux8"


@dataclass(frozen=True)
class PublicConfig:
    max_time_seconds: float
    max_migration_attempts: int
    checkpoint_fixed_delay_seconds: float
    restore_delay_seconds: float
    congested_delay_mean_seconds: float
    congested_delay_std_seconds: float
    link_bandwidth_bytes_per_second: float
    dt_seconds: float = 1.0


@dataclass(frozen=True)
class NodeObservation:
    node_id: int
    online: bool
    cpu_capacity: float
    memory_capacity: float
    cpu_available: float
    memory_available: float
    cpu_utilization: float
    memory_utilization: float
    external_cpu_fraction: float
    background_cpu_estimate: float
    burst_remaining_estimate: float
    failure_rate_estimate: float
    repair_rate_estimate: float
    cpu_rate: float


@dataclass(frozen=True)
class TaskObservation:
    task_id: int
    cpu_need: float
    memory_need: float
    total_work: float
    remaining_work_estimate: float
    deadline_seconds: float
    checkpoint_size_bytes: int
    checkpoint_work: float
    checkpoint_version: int
    last_checkpoint_time: float
    status: str
    node_id: int | None
    migration_target: int | None
    migration_attempts: int
    sla_violated: bool


@dataclass(frozen=True)
class LinkObservation:
    source_id: int
    target_id: int
    reachable: bool
    rtt_seconds: float
    effective_bandwidth: float


@dataclass(frozen=True)
class DecisionContext:
    time_seconds: float
    focus_task_id: int | None
    nodes: tuple[NodeObservation, ...]
    tasks: tuple[TaskObservation, ...]
    links: tuple[LinkObservation, ...]
    action_mask: tuple[bool, ...]
    public_config: PublicConfig
    schema_version: str = SCHEMA_VERSION

    def get_focus_task(self) -> TaskObservation | None:
        return next((task for task in self.tasks
                     if task.task_id == self.focus_task_id), None)

    def link(self, source_id: int, target_id: int) -> LinkObservation:
        return self.links[source_id * len(self.nodes) + target_id]


FEATURE_ORDER = (
    "time_seconds", "focus_task_id", "public_config", "nodes", "tasks",
    "links", "action_mask", "candidate_node_aux8",
)
FEATURE_SCALES = {
    "time_seconds": 100.0, "max_time_seconds": 100.0,
    "deadline_seconds": 100.0, "last_checkpoint_time": 100.0,
    "total_work": 100.0, "remaining_work_estimate": 100.0,
    "checkpoint_work": 100.0, "checkpoint_size_bytes": 1_000_000.0,
    "link_bandwidth_bytes_per_second": 10_000_000.0,
    "effective_bandwidth": 10_000_000.0,
    "cpu_capacity": 8.0, "cpu_available": 8.0,
    "memory_capacity": 32.0, "memory_available": 32.0,
    "failure_rate_estimate": 0.001, "repair_rate_estimate": 0.001,
}


def _scaled(name: str, value: float | int | bool) -> float:
    return float(value) / FEATURE_SCALES.get(name, 1.0)


def encode_observation(context: DecisionContext) -> tuple[float, ...]:
    """Lossless, fixed-order numeric encoding; no clipping or test-time fitting."""
    if context.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported observation schema")
    values: list[float] = [_scaled("time_seconds", context.time_seconds)]
    values.extend(float(context.focus_task_id == task.task_id)
                  for task in context.tasks)
    values.append(float(context.focus_task_id is None))
    values.extend(_scaled(field.name, getattr(context.public_config, field.name))
                  for field in fields(PublicConfig))
    for node in context.nodes:
        values.extend(_scaled(field.name, getattr(node, field.name))
                      for field in fields(NodeObservation))
    statuses = ("pending", "running", "migrating", "interrupted",
                "completed", "failed")
    for task in context.tasks:
        for field in fields(TaskObservation):
            if field.name == "status":
                values.extend(float(task.status == status) for status in statuses)
            elif field.name in ("node_id", "migration_target"):
                node_id = getattr(task, field.name)
                values.extend(float(node_id == node.node_id) for node in context.nodes)
                values.append(float(node_id is None))
            else:
                values.append(_scaled(field.name, getattr(task, field.name)))
    for link in context.links:
        values.extend(_scaled(field.name, getattr(link, field.name))
                      for field in fields(LinkObservation))
    values.extend(float(ok) for ok in context.action_mask)
    from .decision_features import encode_candidate_features
    values.extend(encode_candidate_features(context))
    return tuple(values)
