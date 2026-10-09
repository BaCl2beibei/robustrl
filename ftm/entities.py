"""Cluster entities and migration/checkpoint state types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    MIGRATING = "migrating"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class MigrationStage(str, Enum):
    TRANSFERRING = "transferring"
    RESTORING = "restoring"
    CONFIRMED = "confirmed"
    ABORTED = "aborted"


@dataclass
class Reservation:
    task_id: int
    cpu: float
    memory: float
    token: str


@dataclass
class Node:
    node_id: int
    cpu_capacity: float
    memory_capacity: float
    cpu_rate: float
    failure_rate: float
    repair_rate: float
    up: bool = True
    external_cpu_fraction: float = 0.0
    external_memory: float = 0.0
    reservations: dict[str, Reservation] = field(default_factory=dict)
    normal_external_cpu_fraction: float = 0.105

    @property
    def cpu_used(self) -> float:
        return sum(item.cpu for item in self.reservations.values())

    @property
    def memory_used(self) -> float:
        return sum(item.memory for item in self.reservations.values()) + self.external_memory

    @property
    def cpu_utilization(self) -> float:
        return min(1.0, (self.cpu_used / self.cpu_capacity) + self.external_cpu_fraction)

    @property
    def memory_utilization(self) -> float:
        return min(1.0, self.memory_used / self.memory_capacity)

    @property
    def estimated_burst_cpu_fraction(self) -> float:
        return max(0.0, self.external_cpu_fraction - self.normal_external_cpu_fraction)

    def can_host(self, cpu: float, memory: float) -> bool:
        # CPU reservations are the hard limit; external CPU load remains a soft load.
        return (self.up and self.cpu_used + cpu <= self.cpu_capacity + 1e-9
                and self.memory_used + memory <= self.memory_capacity + 1e-9)

    def reserve(self, task_id: int, cpu: float, memory: float, token: str) -> bool:
        if token in self.reservations or not self.can_host(cpu, memory):
            return False
        self.reservations[token] = Reservation(task_id, cpu, memory, token)
        return True

    def release(self, token: str) -> bool:
        return self.reservations.pop(token, None) is not None

    def promote_reservation(self, old_token: str, new_token: str) -> bool:
        if new_token in self.reservations or old_token not in self.reservations:
            return False
        reservation = self.reservations.pop(old_token)
        self.reservations[new_token] = Reservation(
            reservation.task_id, reservation.cpu, reservation.memory, new_token
        )
        return True

    def promote_reservation(self, old_token: str, new_token: str) -> bool:
        if new_token in self.reservations or old_token not in self.reservations:
            return False
        reservation = self.reservations.pop(old_token)
        self.reservations[new_token] = Reservation(
            reservation.task_id, reservation.cpu, reservation.memory, new_token
        )
        return True


@dataclass
class Task:
    task_id: int
    cpu_need: float
    memory_need: float
    total_work: float
    deadline_seconds: float
    checkpoint_size_bytes: int
    checkpoint_interval_work: float
    arrival_time: float = 0.0
    remaining_work: float = 0.0
    checkpoint_work: float = 0.0
    checkpoint_version: int = 0
    status: TaskStatus = TaskStatus.PENDING
    node_id: int | None = None
    migration_target: int | None = None
    last_checkpoint_time: float = 0.0
    completion_time: float | None = None
    migration_attempts: int = 0
    sla_violated: bool = False

    def __post_init__(self) -> None:
        if self.remaining_work == 0.0:
            self.remaining_work = self.total_work

    @property
    def progress(self) -> float:
        return max(0.0, self.total_work - self.remaining_work)

    def advance(self, work: float, now: float) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False
        self.remaining_work = max(0.0, self.remaining_work - max(0.0, work))
        if self.progress - self.checkpoint_work >= self.checkpoint_interval_work:
            self.commit_checkpoint(now)
        if self.remaining_work <= 1e-9:
            self.remaining_work = 0.0
            self.status = TaskStatus.COMPLETED
            self.completion_time = now
            return True
        return False

    def commit_checkpoint(self, now: float) -> None:
        self.checkpoint_work = self.progress
        self.checkpoint_version += 1
        self.last_checkpoint_time = now

    def restore_latest_checkpoint(self) -> None:
        self.remaining_work = max(0.0, self.total_work - self.checkpoint_work)

    def has_missed_deadline(self, now: float) -> bool:
        """Return whether this task finished late or remains unfinished past its deadline."""
        if self.completion_time is not None:
            return self.completion_time > self.deadline_seconds
        return now > self.deadline_seconds


@dataclass
class Migration:
    task_id: int
    source_id: int | None
    target_id: int
    reservation_token: str
    remaining_bytes: float
    latency_remaining: float
    restore_remaining: float
    stage: MigrationStage = MigrationStage.TRANSFERRING
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    task_id: int | None = None
    node_id: int | None = None
    target_id: int | None = None
    value: float | str | None = None
    duration_seconds: float | None = None
