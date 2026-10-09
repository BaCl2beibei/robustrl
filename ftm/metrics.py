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
    node_failure_continuation_success_rate: float | None
    task_service_availability: float | None
    cluster_availability: float
    mean_completion_seconds: float | None
    mean_completion_seconds_horizon_capped: float
    sla_violation_rate: float
    migration_count: int
    migration_attempt_count: int
    migration_failure_count: int
    migration_incomplete_count: int
    mean_migration_seconds: float | None
    total_migration_seconds: float
    cpu_utilization: float
    task_cpu_utilization: float
    reserved_cpu_utilization: float
    external_cpu_utilization: float
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
    capped_completion_times = [
        (task.completion_time - task.arrival_time
         if task.completion_time is not None
         else max(0.0, env.config.max_time_seconds - task.arrival_time))
        for task in tasks
    ]
    confirmed = [event for event in events if event.kind == "migration_confirmed"]
    migrations = [event for event in events
                  if event.kind in ("migration_confirmed", "migration_aborted")]
    successful_migration_times = [float(event.duration_seconds or 0.0)
                                  for event in confirmed]
    mean_migration = (sum(successful_migration_times) / len(successful_migration_times)
                      if successful_migration_times else None)
    total_migration_seconds = sum(float(event.duration_seconds or 0.0)
                                  for event in migrations)
    in_flight_migrations = list(env.manager.active.values())
    total_migration_seconds += sum(migration.elapsed_seconds
                                   for migration in in_flight_migrations)
    nodes = len(env.cluster.nodes)
    observation_window = env.config.max_time_seconds
    cpu_denominator = nodes * env.config.cpu_capacity * observation_window
    memory_denominator = nodes * env.config.memory_capacity * observation_window
    return EpisodeMetrics(
        scenario=env.scenario.name,
        policy=policy_name,
        seed=seed,
        elapsed_seconds=env.time_seconds,
        task_completion_rate=len(completed) / max(1, len(tasks)),
        node_failure_continuation_success_rate=(len(continued_ids) / len(interrupted_ids)
                                                if interrupted_ids else None),
        task_service_availability=(env.cluster.task_served_seconds
                                   / env.cluster.task_demand_seconds
                                   if env.cluster.task_demand_seconds > 0 else None),
        cluster_availability=min(1.0, env.cluster.available_node_seconds
                                 / observation_window),
        mean_completion_seconds=(sum(completion_times) / len(completion_times)
                                 if completion_times else None),
        mean_completion_seconds_horizon_capped=(
            sum(capped_completion_times) / max(1, len(capped_completion_times))
        ),
        sla_violation_rate=sum(task.has_missed_deadline(env.time_seconds)
                               for task in tasks) / max(1, len(tasks)),
        migration_count=len(confirmed),
        migration_attempt_count=sum(event.kind == "migration_started" for event in events),
        migration_failure_count=sum(event.kind == "migration_aborted" for event in events),
        migration_incomplete_count=len(in_flight_migrations),
        mean_migration_seconds=mean_migration,
        total_migration_seconds=total_migration_seconds,
        cpu_utilization=min(1.0, env.cluster.cpu_used_seconds / max(cpu_denominator, 1e-9)),
        task_cpu_utilization=min(1.0, env.cluster.task_cpu_used_seconds
                                 / max(cpu_denominator, 1e-9)),
        reserved_cpu_utilization=min(1.0, env.cluster.task_cpu_reserved_seconds
                                     / max(cpu_denominator, 1e-9)),
        external_cpu_utilization=min(1.0, env.cluster.external_cpu_used_seconds
                                     / max(cpu_denominator, 1e-9)),
        memory_utilization=min(1.0, env.cluster.memory_used_seconds
                               / max(memory_denominator, 1e-9)),
        total_reward=total_reward,
    )
