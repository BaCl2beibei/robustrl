"""Interfaces for replacing the simulator with real distributed components.

This module defines contracts only. It does not connect to Kubernetes, a batch
scheduler, shared storage, or a real network. A production backend must provide
durable checkpoint storage, fenced leases, authenticated transport, and an
idempotency strategy for external task side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CheckpointManifest:
    task_id: str
    version: int
    uri: str
    size_bytes: int
    checksum: str
    completed_work: float


@dataclass(frozen=True)
class Lease:
    task_id: str
    node_id: str
    fencing_epoch: int
    owner_token: str


@dataclass(frozen=True)
class ExecutionHandle:
    task_id: str
    node_id: str
    process_token: str


class CheckpointStore(Protocol):
    def latest(self, task_id: str) -> CheckpointManifest: ...

    def copy_to(self, checkpoint: CheckpointManifest, target_node: str) -> CheckpointManifest: ...


class NodeExecutor(Protocol):
    def reserve(self, task_id: str, target_node: str, fencing_epoch: int) -> Lease: ...

    def restore(self, lease: Lease, checkpoint: CheckpointManifest) -> ExecutionHandle: ...

    def ready(self, handle: ExecutionHandle) -> bool: ...

    def stop(self, handle: ExecutionHandle, fencing_epoch: int) -> None: ...

    def release(self, lease: Lease) -> None: ...


class LeaseRegistry(Protocol):
    def current(self, task_id: str) -> Lease | None: ...

    def compare_and_swap(self, task_id: str, expected: Lease | None,
                         replacement: Lease) -> bool: ...


class MigrationTransport(Protocol):
    def transfer(self, source: CheckpointManifest, target_node: str
                 ) -> CheckpointManifest: ...


@dataclass(frozen=True)
class RealMigrationResult:
    committed: bool
    lease: Lease | None
    reason: str


def execute_two_phase_migration(task_id: str, target_node: str,
                                expected_source_lease: Lease | None,
                                new_fencing_epoch: int,
                                checkpoints: CheckpointStore,
                                executor: NodeExecutor,
                                leases: LeaseRegistry,
                                transport: MigrationTransport,
                                source_handle: ExecutionHandle | None = None
                                ) -> RealMigrationResult:
    """Orchestrate reserve→copy→restore→fence-switch→release.

    The old executor is stopped only after the target has reported ready and the
    atomic lease compare-and-swap succeeds. A failed step releases target-side
    resources. This control-plane order prevents two unfenced owners; it cannot
    make arbitrary external side effects exactly-once without task cooperation.
    """
    target_lease = executor.reserve(task_id, target_node, new_fencing_epoch)
    target_handle: ExecutionHandle | None = None
    switched = False
    try:
        checkpoint = checkpoints.latest(task_id)
        copied = transport.transfer(checkpoint, target_node)
        if copied.checksum != checkpoint.checksum or copied.version != checkpoint.version:
            raise RuntimeError("checkpoint integrity/version mismatch")
        target_handle = executor.restore(target_lease, copied)
        if not executor.ready(target_handle):
            raise RuntimeError("target restore was not ready")
        if not leases.compare_and_swap(task_id, expected_source_lease, target_lease):
            raise RuntimeError("source lease changed during migration")
        switched = True
        if source_handle is not None and expected_source_lease is not None:
            try:
                executor.stop(source_handle, new_fencing_epoch)
                executor.release(expected_source_lease)
            except Exception as exc:
                return RealMigrationResult(True, target_lease,
                                           "committed; source cleanup pending: " + str(exc))
        return RealMigrationResult(True, target_lease, "committed")
    except Exception as exc:
        if switched:
            return RealMigrationResult(True, target_lease,
                                       "committed; cleanup pending: " + str(exc))
        cleanup_errors = []
        if target_handle is not None:
            try:
                executor.stop(target_handle, new_fencing_epoch)
            except Exception as cleanup_exc:
                cleanup_errors.append("stop target: " + str(cleanup_exc))
        try:
            executor.release(target_lease)
        except Exception as cleanup_exc:
            cleanup_errors.append("release target: " + str(cleanup_exc))
        reason = str(exc)
        if cleanup_errors:
            reason += "; cleanup pending: " + "; ".join(cleanup_errors)
        return RealMigrationResult(False, None, reason)
