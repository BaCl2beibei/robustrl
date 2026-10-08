"""Checkpoint-based migration state machine with reservation and rollback."""

from __future__ import annotations

from .cluster import Cluster, task_token
from .config import SimulationConfig
from .entities import Event, Migration, MigrationStage, Task, TaskStatus


class MigrationManager:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.active: dict[int, Migration] = {}

    def valid_targets(self, task: Task, cluster: Cluster) -> tuple[bool, ...]:
        valid = []
        for node_id, node in cluster.nodes.items():
            source = task.node_id
            link_ok = (source is None or cluster.network.links[source][node_id].reachable)
            valid.append(node_id != source and node.can_host(task.cpu_need, task.memory_need)
                         and link_ok and task.task_id not in self.active)
        return tuple(valid)

    def start(self, task: Task, target_id: int, cluster: Cluster,
              now: float) -> tuple[bool, str]:
        if task.task_id in self.active:
            return False, "task_already_migrating"
        if task.status not in (TaskStatus.RUNNING, TaskStatus.INTERRUPTED):
            return False, "task_not_migratable"
        valid = self.valid_targets(task, cluster)
        if not 0 <= target_id < len(valid) or not valid[target_id]:
            return False, "invalid_target"

        source_id = task.node_id
        if task.status == TaskStatus.RUNNING:
            task.commit_checkpoint(now)
        else:
            task.restore_latest_checkpoint()
        token = f"migration:{task.task_id}:{task.migration_attempts + 1}"
        target = cluster.nodes[target_id]
        if not target.reserve(task.task_id, task.cpu_need, task.memory_need, token):
            return False, "reservation_failed"

        task.migration_attempts += 1
        task.migration_target = target_id
        task.status = TaskStatus.MIGRATING
        latency = cluster.network.transfer_delay(source_id, target_id)
        self.active[task.task_id] = Migration(
            task_id=task.task_id,
            source_id=source_id,
            target_id=target_id,
            reservation_token=token,
            remaining_bytes=float(task.checkpoint_size_bytes),
            latency_remaining=latency + self.config.checkpoint_fixed_delay_seconds,
            restore_remaining=self.config.restore_delay_seconds,
        )
        return True, "started"

    def advance(self, dt_seconds: float, cluster: Cluster, now: float) -> list[Event]:
        emitted: list[Event] = []
        for task_id, migration in list(self.active.items()):
            task = cluster.tasks[task_id]
            target = cluster.nodes[migration.target_id]
            migration.elapsed_seconds += dt_seconds
            if not target.up:
                emitted.extend(self._abort(task, migration, cluster, now, "target_failed"))
                continue
            if migration.stage == MigrationStage.TRANSFERRING:
                time_for_transfer = dt_seconds
                if migration.latency_remaining > 0:
                    consumed = min(dt_seconds, migration.latency_remaining)
                    migration.latency_remaining -= consumed
                    time_for_transfer -= consumed
                if time_for_transfer > 0 and migration.latency_remaining <= 0:
                    rate = cluster.network.transfer_rate(migration.source_id,
                                                         migration.target_id)
                    if rate <= 0:
                        emitted.extend(self._abort(task, migration, cluster, now,
                                                   "network_unreachable"))
                        continue
                    migration.remaining_bytes = max(
                        0.0, migration.remaining_bytes - rate * time_for_transfer
                    )
                if migration.remaining_bytes <= 0 and migration.latency_remaining <= 0:
                    migration.stage = MigrationStage.RESTORING
                    emitted.append(Event(now, "checkpoint_transferred", task_id,
                                         migration.source_id, migration.target_id,
                                         task.checkpoint_version))
            elif migration.stage == MigrationStage.RESTORING:
                migration.restore_remaining = max(0.0,
                                                  migration.restore_remaining - dt_seconds)
                if migration.restore_remaining <= 0:
                    if not target.promote_reservation(migration.reservation_token,
                                                      task_token(task_id)):
                        emitted.extend(self._abort(task, migration, cluster, now,
                                                   "reservation_promotion_failed"))
                        continue
                    if migration.source_id is not None:
                        cluster.nodes[migration.source_id].release(task_token(task_id))
                    task.node_id = migration.target_id
                    task.migration_target = None
                    task.restore_latest_checkpoint()
                    task.status = TaskStatus.RUNNING
                    migration.stage = MigrationStage.CONFIRMED
                    emitted.append(Event(now, "migration_confirmed", task_id,
                                         migration.source_id, migration.target_id,
                                         migration.elapsed_seconds))
                    del self.active[task_id]
        cluster.events.extend(emitted)
        return emitted

    def _abort(self, task: Task, migration: Migration, cluster: Cluster,
               now: float, reason: str) -> list[Event]:
        cluster.nodes[migration.target_id].release(migration.reservation_token)
        source_up = (migration.source_id is not None
                     and cluster.nodes[migration.source_id].up)
        task.migration_target = None
        task.restore_latest_checkpoint()
        if source_up:
            task.status = TaskStatus.RUNNING
        else:
            task.status = TaskStatus.INTERRUPTED
        if task.migration_attempts >= self.config.max_migration_attempts:
            task.status = TaskStatus.FAILED
            cluster.release_task(task)
            reason = "max_attempts:" + reason
            failure_event = Event(now, "task_failed", task.task_id,
                                  migration.source_id, migration.target_id, reason)
        else:
            failure_event = None
        self.active.pop(task.task_id, None)
        events = [Event(now, "migration_aborted", task.task_id, migration.source_id,
                        migration.target_id, reason)]
        if failure_event is not None:
            events.append(failure_event)
        return events
