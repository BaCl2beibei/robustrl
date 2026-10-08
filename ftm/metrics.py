"""Episode metrics derived from task state and the event log."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .environment import MigrationEnvironment
from .entities import TaskStatus


@dataclass
class EpisodeMetrics:
    scenario: str
    policy: str
    seed: int
    elapsed_seconds: float
    task_completion_rate: float
    continuation_success_rate: float | None
    cluster_availability: float
    mean_completion_seconds: float | None
    sla_violation_rate: float
    migration_count: int
    migration_attempt_count: int
    mean_migration_seconds: float | None
    cpu_utilization: float
    memory_utilization: float
    total_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_episode_metrics(env: MigrationEnvironment, policy_name: str,
                            seed: int, total_reward: float) -> EpisodeMetrics:
    tasks = list(env.cluster.tasks.values())
    events = env.episode_events
    completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
    interrupted_ids = {event.task_id for event in events
                       if event.kind == "task_interrupted" and event.task_id is not None}
    continued_ids = {task.task_id for task in completed} & interrupted_ids
    completion_times = [task.completion_time - task.arrival_time for task in completed
                        if task.completion_time is not None]
    confirmed = [event for event in events if event.kind == "migration_confirmed"]
    mean_migration = (sum(float(event.value or 0.0) for event in confirmed) / len(confirmed)
                      if confirmed else None)
    elapsed = max(env.time_seconds, 1e-9)
    nodes = len(env.cluster.nodes)
    cpu_denominator = nodes * env.config.cpu_capacity * elapsed
    memory_denominator = nodes * env.config.memory_capacity * elapsed
    return EpisodeMetrics(
        scenario=env.scenario.name,
        policy=policy_name,
        seed=seed,
        elapsed_seconds=env.time_seconds,
        task_completion_rate=len(completed) / max(1, len(tasks)),
        continuation_success_rate=(len(continued_ids) / len(interrupted_ids)
                                   if interrupted_ids else None),
        cluster_availability=min(1.0, env.cluster.available_node_seconds / elapsed),
        mean_completion_seconds=(sum(completion_times) / len(completion_times)
                                 if completion_times else None),
        sla_violation_rate=sum(task.sla_violated for task in tasks) / max(1, len(tasks)),
        migration_count=len(confirmed),
        migration_attempt_count=sum(event.kind == "migration_started" for event in events),
        mean_migration_seconds=mean_migration,
        cpu_utilization=min(1.0, env.cluster.cpu_used_seconds / max(cpu_denominator, 1e-9)),
        memory_utilization=min(1.0, env.cluster.memory_used_seconds
                               / max(memory_denominator, 1e-9)),
        total_reward=total_reward,
    )
