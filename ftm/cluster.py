"""Resource accounting, task execution and node failure handling."""

from __future__ import annotations

from .config import SimulationConfig
from .disturbances import DisturbanceSample
from .entities import Event, Node, Task, TaskStatus
from .network import Network


def task_token(task_id: int) -> str:
    return f"task:{task_id}"


class Cluster:
    def __init__(self, config: SimulationConfig, nodes: list[Node],
                 tasks: list[Task], network: Network) -> None:
        self.config = config
        self.nodes = {node.node_id: node for node in nodes}
        self.tasks = {task.task_id: task for task in tasks}
        self.network = network
        self.events: list[Event] = []
        self.time_seconds = 0.0
        self.available_node_seconds = 0.0
        self.cpu_used_seconds = 0.0
        self.memory_used_seconds = 0.0

    def place(self, task: Task, node_id: int) -> bool:
        node = self.nodes[node_id]
        if task.status not in (TaskStatus.PENDING, TaskStatus.INTERRUPTED):
            return False
        if not node.reserve(task.task_id, task.cpu_need, task.memory_need,
                            task_token(task.task_id)):
            return False
        task.node_id = node_id
        task.status = TaskStatus.RUNNING
        return True

    def place_pending(self) -> None:
        pending = sorted((t for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                         key=lambda t: (t.deadline_seconds, t.task_id))
        for task in pending:
            candidates = [n for n in self.nodes.values() if n.can_host(task.cpu_need, task.memory_need)]
            if candidates:
                node = min(candidates, key=lambda n: (n.cpu_utilization + n.memory_utilization,
                                                       n.node_id))
                self.place(task, node.node_id)

    def advance(self, dt_seconds: float, disturbance: DisturbanceSample,
                now: float) -> list[Event]:
        self.time_seconds = now
        emitted: list[Event] = []
        for node_id, kind in disturbance.failed_or_repaired:
            emitted.append(Event(now, kind, node_id=node_id))
            if kind == "node_failure":
                for task in self.tasks.values():
                    if task.node_id == node_id and task.status == TaskStatus.RUNNING:
                        task.restore_latest_checkpoint()
                        task.status = TaskStatus.INTERRUPTED
                        emitted.append(Event(now, "task_interrupted", task.task_id, node_id,
                                             value=task.checkpoint_version))

        up_count = sum(1 for node in self.nodes.values() if node.up)
        if up_count >= self.config.k_required:
            self.available_node_seconds += dt_seconds
        self.cpu_used_seconds += dt_seconds * sum(
            min(node.cpu_capacity, node.cpu_used + node.external_cpu_fraction * node.cpu_capacity)
            for node in self.nodes.values() if node.up
        )
        self.memory_used_seconds += dt_seconds * sum(
            min(node.memory_capacity, node.memory_used)
            for node in self.nodes.values() if node.up
        )

        for task in self.tasks.values():
            if task.status != TaskStatus.RUNNING or task.node_id is None:
                continue
            node = self.nodes[task.node_id]
            if not node.up:
                continue
            effective_rate = node.cpu_rate * task.cpu_need * max(
                0.05, 1.0 - node.external_cpu_fraction
            )
            completed = task.advance(effective_rate * dt_seconds, now)
            if not task.sla_violated and now >= task.deadline_seconds:
                task.sla_violated = True
                emitted.append(Event(now, "sla_violated", task.task_id, node.node_id))
            if completed:
                node.release(task_token(task.task_id))
                emitted.append(Event(now, "task_completed", task.task_id, node.node_id))
        self.events.extend(emitted)
        return emitted

    def release_task(self, task: Task) -> None:
        if task.node_id is not None:
            self.nodes[task.node_id].release(task_token(task.task_id))

    def focus_task(self) -> Task | None:
        interrupted = [t for t in self.tasks.values() if t.status == TaskStatus.INTERRUPTED]
        if interrupted:
            return min(interrupted, key=lambda t: (t.deadline_seconds, t.task_id))
        running = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
        if running:
            return min(running, key=lambda t: (t.deadline_seconds, t.task_id))
        return None

    def all_terminal(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                   for t in self.tasks.values())

    def state_vector(self, common_shock: bool, time_limit: float) -> tuple[float, ...]:
        values: list[float] = []
        n = len(self.nodes)
        for node in self.nodes.values():
            values.extend((float(node.up), node.cpu_utilization, node.memory_utilization,
                           node.external_cpu_fraction,
                           sum(1 for t in self.tasks.values()
                               if t.node_id == node.node_id and t.status in
                               (TaskStatus.RUNNING, TaskStatus.MIGRATING)),
                           min(1.0, node.failure_rate * 1000.0),
                           min(1.0, node.repair_rate * 100.0),
                           min(1.0, len(node.reservations) / max(1, len(self.tasks)))))
        for i in range(n):
            for j in range(n):
                link = self.network.links[i][j]
                values.extend((min(1.0, link.delay_seconds),
                               min(1.0, link.bandwidth_bytes_per_second / 100_000_000.0),
                               float(link.congested), float(link.reachable)))
        status_order = (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.MIGRATING,
                        TaskStatus.INTERRUPTED, TaskStatus.COMPLETED, TaskStatus.FAILED)
        for task in self.tasks.values():
            values.extend((task.cpu_need / self.config.cpu_capacity,
                           task.memory_need / self.config.memory_capacity,
                           task.remaining_work / task.total_work,
                           task.checkpoint_work / task.total_work,
                           max(0.0, task.deadline_seconds - self.time_seconds) / time_limit,
                           max(0.0, self.time_seconds - task.last_checkpoint_time) / time_limit,
                           float(task.sla_violated),
                           min(1.0, task.migration_attempts / self.config.max_migration_attempts)))
            values.extend(float(task.status == status) for status in status_order)
            values.extend(float(task.node_id == i) for i in range(n))
            values.append(float(task.node_id is None))
            values.extend(float(task.migration_target == i) for i in range(n))
            values.append(float(task.migration_target is None))
        values.extend((self.time_seconds / time_limit,
                       self.config.k_required / self.config.n_nodes,
                       float(common_shock)))
        return tuple(values)
