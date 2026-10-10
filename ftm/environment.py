"""Single-machine fixed-step environment for checkpoint-based migration."""

from __future__ import annotations

import random
from dataclasses import asdict

from .cluster import Cluster
from .config import ScenarioConfig, SimulationConfig
from .disturbances import DisturbanceModel
from .entities import Event, Node, Task, TaskStatus
from .migration import MigrationManager
from .network import Network
from .observation import (DecisionContext, LinkObservation, NodeObservation,
                          PublicConfig, TaskObservation, encode_observation)
from .policies import predict_burst_remaining


def scenario_by_name(name: str, config: SimulationConfig) -> ScenarioConfig:
    if name == "normal":
        return ScenarioConfig(name, failure_multiplier=0.2, congestion_enter=0.01,
                              congestion_leave=0.6, load_burst_probability=0.002)
    if name == "failure":
        return ScenarioConfig(name, failure_multiplier=8.0, congestion_enter=0.05,
                              common_shock_probability=0.002,
                              common_shock_failure_multiplier=15.0,
                              load_burst_probability=0.01)
    if name == "network":
        return ScenarioConfig(name, failure_multiplier=0.5, congestion_enter=0.35,
                              congestion_leave=0.12, load_burst_probability=0.005,
                              disconnect_enter=0.08, disconnect_leave=0.30)
    if name in ("load", "load_short", "load_medium", "load_sustained"):
        duration_range = {
            "load": (5.0, 60.0),
            "load_short": (2.0, 5.0),
            "load_medium": (10.0, 30.0),
            "load_sustained": (30.0, 120.0),
        }[name]
        return ScenarioConfig(name, failure_multiplier=0.5, congestion_enter=0.08,
                              load_burst_probability=0.15, load_burst_cpu=0.55,
                              load_burst_duration_min_seconds=duration_range[0],
                              load_burst_duration_max_seconds=duration_range[1])
    if name == "mixed":
        return ScenarioConfig(name, failure_multiplier=3.0, congestion_enter=0.16,
                              congestion_leave=0.22, common_shock_probability=0.001,
                              common_shock_failure_multiplier=12.0,
                              common_shock_congestion_boost=0.55,
                              load_burst_probability=0.06, load_burst_cpu=0.45)
    raise ValueError(f"unknown scenario: {name}")


class MigrationEnvironment:
    """Gymnasium-like API without a mandatory Gymnasium dependency.

    Action values 0..N-1 select a migration destination; N means wait/no migration.
    The focus task is selected by the environment and exposed through ``focus_task``.
    """

    def __init__(self, config: SimulationConfig | None = None,
                 scenario: ScenarioConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.scenario = scenario or scenario_by_name("mixed", self.config)
        self.rng = random.Random()
        self.cluster: Cluster
        self.manager: MigrationManager
        self.disturbances: DisturbanceModel
        self.time_seconds = 0.0
        self.last_common_shock = False
        self.episode_events: list[Event] = []
        self._has_reset = False
        self._finished = False
        self._monitor_history: list[DecisionContext] = []

    @property
    def focus_task(self) -> Task | None:
        return self.cluster.focus_task()

    def reset(self, seed: int | None = None) -> tuple[tuple[float, ...], dict]:
        self.rng = random.Random(seed)
        self.time_seconds = 0.0
        self.last_common_shock = False
        self.episode_events = []
        self._finished = False
        self._monitor_rng = random.Random((seed or 0) ^ 0x4D4F4E49)
        self._monitor_history = []
        self._last_monitor_time = float("-inf")
        self._node_up_seconds = [0.0] * self.config.n_nodes
        self._node_down_seconds = [0.0] * self.config.n_nodes
        self._node_failures = [0] * self.config.n_nodes
        self._node_repairs = [0] * self.config.n_nodes
        self._last_heartbeat = [0.0] * self.config.n_nodes
        self._background_cpu = [0.105] * self.config.n_nodes
        self._burst_start: list[float | None] = [None] * self.config.n_nodes
        self._burst_durations: list[list[float]] = [[] for _ in range(self.config.n_nodes)]

        nodes = [Node(i, self.config.cpu_capacity, self.config.memory_capacity,
                      self.config.cpu_rate,
                      self.config.base_failure_rate * (0.8 + 0.4 * self.rng.random()),
                      self.config.base_repair_rate * (0.8 + 0.4 * self.rng.random()))
                 for i in range(self.config.n_nodes)]
        tasks = []
        for task_id in range(self.config.n_tasks):
            work = self.rng.uniform(self.config.task_work_min, self.config.task_work_max)
            cpu_need = self.rng.uniform(self.config.task_cpu_min, self.config.task_cpu_max)
            memory = self.rng.uniform(
                self.config.task_memory_min,
                min(self.config.task_memory_max, self.config.memory_capacity * 0.8),
            )
            expected_seconds = work / max(0.1, cpu_need * self.config.cpu_rate * 0.85)
            tasks.append(Task(
                task_id=task_id,
                cpu_need=cpu_need,
                memory_need=memory,
                total_work=work,
                deadline_seconds=max(10.0, expected_seconds * self.config.task_deadline_multiplier),
                checkpoint_size_bytes=self.config.checkpoint_size_bytes,
                checkpoint_interval_work=self.config.checkpoint_interval_work,
            ))

        network = Network(self.config)
        self.cluster = Cluster(self.config, nodes, tasks, network)
        self.manager = MigrationManager(self.config)
        self.disturbances = DisturbanceModel(self.config, self.scenario, self.rng)
        self.cluster.place_pending()
        self.cluster.network.update(self.disturbances)
        for task in self.cluster.tasks.values():
            if task.status == TaskStatus.RUNNING:
                event = Event(0.0, "task_scheduled", task.task_id, task.node_id)
                self.cluster.events.append(event)
                self.episode_events.append(event)
        self._has_reset = True
        self._record_monitor(force=True)
        return self.observation(), {"seed": seed, "scenario": self.scenario.name}

    def valid_action_mask(self) -> tuple[bool, ...]:
        return self.get_decision_context().action_mask

    def observation(self) -> tuple[float, ...]:
        return encode_observation(self.get_decision_context())

    def get_decision_context(self) -> DecisionContext:
        if not self._has_reset:
            raise RuntimeError("call reset() before requesting a decision context")
        cutoff = self.time_seconds - self.config.monitoring.reporting_delay_seconds
        snapshot = next((item for item in reversed(self._monitor_history)
                         if item.time_seconds <= cutoff + 1e-9),
                        self._monitor_history[0])
        # The focus is scheduler metadata at the decision instant. The sampled
        # task and resource data below may be older when reporting is delayed.
        focus = self.focus_task
        focus_id = None if focus is None else focus.task_id
        mask = self._observed_action_mask(snapshot, focus_id)
        return DecisionContext(self.time_seconds, focus_id, snapshot.nodes,
                               snapshot.tasks, snapshot.links, mask,
                               snapshot.public_config)

    @staticmethod
    def _observed_action_mask(snapshot: DecisionContext,
                              focus_id: int | None) -> tuple[bool, ...]:
        task = next((item for item in snapshot.tasks
                     if item.task_id == focus_id), None)
        mask = []
        for node in snapshot.nodes:
            link_ok = (task is None or task.node_id is None
                       or snapshot.link(task.node_id, node.node_id).reachable)
            mask.append(bool(task is not None
                             and task.status in ("running", "interrupted")
                             and task.node_id != node.node_id
                             and node.online and link_ok
                             and node.cpu_available + 1e-9 >= task.cpu_need
                             and node.memory_available + 1e-9 >= task.memory_need))
        return tuple(mask) + (True,)

    def _record_monitor(self, force: bool = False) -> None:
        cfg = self.config.monitoring
        if (not force and self.time_seconds - self._last_monitor_time
                < cfg.sample_interval_seconds - 1e-9):
            return
        dt = 0.0 if not self._monitor_history else self.time_seconds - self._last_monitor_time
        previous = self._monitor_history[-1] if self._monitor_history else None
        nodes = []
        for node_id, node in self.cluster.nodes.items():
            if node.up:
                self._last_heartbeat[node_id] = self.time_seconds
            observed_online = (self.time_seconds - self._last_heartbeat[node_id]
                               < cfg.heartbeat_timeout_seconds)
            if previous is not None:
                was_up = previous.nodes[node_id].online
                if was_up:
                    self._node_up_seconds[node_id] += dt
                else:
                    self._node_down_seconds[node_id] += dt
                self._node_failures[node_id] += int(was_up and not observed_online)
                self._node_repairs[node_id] += int(not was_up and observed_online)
            external = max(0.0, node.external_cpu_fraction
                           + self._monitor_rng.gauss(0.0, cfg.cpu_noise_std))
            baseline = self._background_cpu[node_id]
            is_burst = external > baseline + 0.2
            start = self._burst_start[node_id]
            if is_burst and start is None:
                self._burst_start[node_id] = self.time_seconds
            elif not is_burst and start is not None:
                self._burst_durations[node_id].append(self.time_seconds - start)
                self._burst_start[node_id] = None
            if not is_burst:
                self._background_cpu[node_id] = 0.9 * baseline + 0.1 * external
            age = (0.0 if self._burst_start[node_id] is None else
                   self.time_seconds - self._burst_start[node_id])
            burst_remaining = (0.0 if not is_burst else predict_burst_remaining(
                age, self._burst_durations[node_id], default_duration=5.0))
            cpu_used = node.cpu_used
            memory_used = max(0.0, node.memory_used
                              + self._monitor_rng.gauss(0.0, cfg.memory_noise_std))
            nodes.append(NodeObservation(
                node_id, observed_online, node.cpu_capacity, node.memory_capacity,
                max(0.0, node.cpu_capacity - cpu_used),
                max(0.0, node.memory_capacity - memory_used),
                cpu_used / node.cpu_capacity + external,
                memory_used / node.memory_capacity, external,
                self._background_cpu[node_id], burst_remaining,
                (self._node_failures[node_id] / self._node_up_seconds[node_id]
                 if self._node_up_seconds[node_id] else self.config.base_failure_rate),
                (self._node_repairs[node_id] / self._node_down_seconds[node_id]
                 if self._node_down_seconds[node_id] else self.config.base_repair_rate),
                node.cpu_rate))
        tasks = tuple(TaskObservation(
            task.task_id, task.cpu_need, task.memory_need, task.total_work,
            task.remaining_work, task.deadline_seconds, task.checkpoint_size_bytes,
            task.checkpoint_work, task.checkpoint_version,
            task.last_checkpoint_time, task.status.value, task.node_id,
            task.migration_target, task.migration_attempts, task.sla_violated)
            for task in self.cluster.tasks.values())
        links = tuple(LinkObservation(
            i, j, link.reachable,
            max(0.0, link.delay_seconds
                + self._monitor_rng.gauss(0.0, cfg.network_rtt_noise_std)),
            self.cluster.network.transfer_rate(i, j))
            for i, row in enumerate(self.cluster.network.links)
            for j, link in enumerate(row))
        public = PublicConfig(
            self.config.max_time_seconds, self.config.max_migration_attempts,
            self.config.checkpoint_fixed_delay_seconds,
            self.config.restore_delay_seconds,
            self.config.congested_delay_mean_seconds,
            self.config.congested_delay_std_seconds,
            self.config.link_bandwidth_bytes_per_second)
        self._monitor_history.append(DecisionContext(
            self.time_seconds, None, tuple(nodes), tasks, links, (), public))
        self._last_monitor_time = self.time_seconds

    def _service_loss_ratio(self) -> float:
        """Fraction of active demand awaiting placement or node recovery.

        MIGRATING is accounted for by its duration-based migration cost when the
        transfer completes, so it does not also accrue the interruption cost.
        """
        active = [task for task in self.cluster.tasks.values()
                  if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]
        if not active:
            return 0.0
        unavailable = sum(task.status in (TaskStatus.PENDING, TaskStatus.INTERRUPTED)
                          for task in active)
        return unavailable / len(active)

    def step(self, action: int) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        if self._finished:
            raise RuntimeError("episode finished; call reset()")
        if not 0 <= action <= self.config.n_nodes:
            raise ValueError("action must be a target node id or n_nodes for wait")

        reward = 0.0
        emitted: list[Event] = []
        task = self.focus_task
        if action < self.config.n_nodes and task is not None:
            source_id = task.node_id
            ok, reason = self.manager.start(task, action, self.cluster, self.time_seconds)
            if not ok:
                reward -= 0.05
                emitted.append(Event(self.time_seconds, "invalid_action", task.task_id,
                                     target_id=action, value=reason))
            else:
                emitted.append(Event(self.time_seconds, "migration_started", task.task_id,
                                     source_id, action, task.checkpoint_version))

        for _ in range(self.config.decision_period_steps):
            if self.time_seconds >= self.config.max_time_seconds:
                break
            dt_seconds = min(self.config.dt_seconds,
                             self.config.max_time_seconds - self.time_seconds)
            unfinished_before = sum(
                task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                for task in self.cluster.tasks.values()
            )
            service_loss_before = self._service_loss_ratio()
            work_before = {task.task_id: task.remaining_work
                           for task in self.cluster.tasks.values()}
            sample = self.disturbances.sample(list(self.cluster.nodes.values()),
                                              dt_seconds)
            self.last_common_shock = sample.common_shock
            self.cluster.network.update(self.disturbances)
            self.time_seconds += dt_seconds
            tick_events = self.cluster.advance(dt_seconds, sample,
                                               self.time_seconds)
            migration_events = self.manager.advance(dt_seconds,
                                                     self.cluster, self.time_seconds)
            schedule_events = self._schedule_pending()
            self._record_monitor()
            for event in tick_events + migration_events + schedule_events:
                reward += self._event_reward(event)
            reward -= self._resource_overload_cost() * dt_seconds
            reward -= (self.config.delay_cost_weight * dt_seconds
                       / self.config.max_time_seconds * unfinished_before)
            service_loss_after = self._service_loss_ratio()
            reward -= (self.config.service_loss_cost_weight
                       * 0.5 * (service_loss_before + service_loss_after)
                       * dt_seconds / self.config.max_time_seconds)
            work_done = sum(work_before[task.task_id] - task.remaining_work
                            for task in self.cluster.tasks.values())
            total_work = sum(task.total_work for task in self.cluster.tasks.values())
            if total_work > 0.0:
                reward += (self.config.progress_reward_weight * work_done / total_work)
            emitted.extend(tick_events)
            emitted.extend(migration_events)
            emitted.extend(schedule_events)

        terminated = self.cluster.all_terminal()
        if terminated and self.time_seconds < self.config.max_time_seconds:
            fixed_window_events = self._advance_fixed_window()
            emitted.extend(fixed_window_events)
            reward += sum(self._event_reward(event) for event in fixed_window_events)
        truncated = self.time_seconds >= self.config.max_time_seconds and not terminated
        if truncated:
            for task in self.cluster.tasks.values():
                if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    event = Event(self.time_seconds, "horizon_incomplete", task.task_id,
                                  task.node_id)
                    emitted.append(event)
                    reward += self._event_reward(event)
        self.episode_events.extend(emitted)
        self._finished = terminated or truncated
        info = {
            "time_seconds": self.time_seconds,
            "events": [asdict(event) for event in emitted],
            "action_mask": self.valid_action_mask(),
            "focus_task_id": None if self.focus_task is None else self.focus_task.task_id,
        }
        return self.observation(), reward, terminated, truncated, info

    def _advance_fixed_window(self) -> list[Event]:
        """Continue only physical sampling after tasks terminate, through the horizon."""
        emitted: list[Event] = []
        while self.time_seconds < self.config.max_time_seconds:
            dt_seconds = min(self.config.dt_seconds,
                             self.config.max_time_seconds - self.time_seconds)
            sample = self.disturbances.sample(list(self.cluster.nodes.values()), dt_seconds)
            self.last_common_shock = sample.common_shock
            self.cluster.network.update(self.disturbances)
            self.time_seconds += dt_seconds
            emitted.extend(self.cluster.advance(dt_seconds, sample, self.time_seconds))
            self._record_monitor()
        return emitted

    def _schedule_pending(self) -> list[Event]:
        before = {t.task_id: t.status for t in self.cluster.tasks.values()}
        self.cluster.place_pending()
        events = []
        for task in self.cluster.tasks.values():
            if before[task.task_id] == TaskStatus.PENDING and task.status == TaskStatus.RUNNING:
                events.append(Event(self.time_seconds, "task_scheduled", task.task_id,
                                    task.node_id))
        self.cluster.events.extend(events)
        return events

    def _resource_overload_cost(self) -> float:
        cost = 0.0
        for node in self.cluster.nodes.values():
            if node.up:
                cost += max(0.0, node.cpu_utilization - 0.85) ** 2
                cost += max(0.0, node.memory_utilization - 0.90) ** 2
        return 0.02 * cost / max(1, self.config.n_nodes)

    @staticmethod
    def _event_reward(event: Event) -> float:
        if event.kind == "task_completed":
            return 1.0
        if event.kind == "task_failed":
            return -1.0
        if event.kind == "horizon_incomplete":
            return -1.0
        if event.kind == "sla_violated":
            return -0.5
        if event.kind == "migration_aborted":
            return -0.02
        if event.kind == "migration_confirmed":
            duration = float(event.value or 0.0)
            return -0.1 * duration / 60.0
        return 0.0
