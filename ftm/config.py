"""Configuration dataclasses. All rates use seconds as the base time unit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "mixed"
    failure_multiplier: float = 1.0
    congestion_enter: float = 0.08
    congestion_leave: float = 0.35
    common_shock_probability: float = 0.0
    common_shock_failure_multiplier: float = 8.0
    common_shock_congestion_boost: float = 0.35
    load_burst_probability: float = 0.02
    load_burst_cpu: float = 0.35
    load_burst_duration_steps: int | None = None
    disconnect_enter: float = 0.0
    disconnect_leave: float = 0.3
    common_shock_disconnect_boost: float = 0.0
    load_burst_duration_min_seconds: float = 5.0
    load_burst_duration_max_seconds: float = 60.0

    def load_burst_duration_bounds(self, dt_seconds: float = 1.0) -> tuple[float, float]:
        if self.load_burst_duration_steps is not None:
            duration = float(self.load_burst_duration_steps) * dt_seconds
            return duration, duration
        return self.load_burst_duration_min_seconds, self.load_burst_duration_max_seconds

    @property
    def bgmm_normal_weight(self) -> float:
        """Stationary normal-mode probability implied by the regime CTMC."""
        total = self.congestion_enter + self.congestion_leave
        return self.congestion_leave / total if total > 0 else 1.0

    def __post_init__(self) -> None:
        if self.failure_multiplier < 0:
            raise ValueError("failure_multiplier must be non-negative")
        for name in ("congestion_enter", "congestion_leave", "common_shock_probability",
                     "load_burst_probability", "load_burst_cpu", "disconnect_enter",
                     "disconnect_leave", "common_shock_disconnect_boost"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (self.load_burst_duration_steps is not None
                and self.load_burst_duration_steps < 1):
            raise ValueError("load_burst_duration_steps must be positive")
        if (self.load_burst_duration_min_seconds <= 0
                or self.load_burst_duration_max_seconds < self.load_burst_duration_min_seconds):
            raise ValueError("invalid load burst duration range")


@dataclass(frozen=True)
class MonitoringConfig:
    sample_interval_seconds: float = 1.0
    reporting_delay_seconds: float = 0.0
    cpu_noise_std: float = 0.0
    memory_noise_std: float = 0.0
    network_rtt_noise_std: float = 0.0
    heartbeat_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("monitoring intervals must be positive")
        if any(value < 0 for value in (self.reporting_delay_seconds,
                                      self.cpu_noise_std, self.memory_noise_std,
                                      self.network_rtt_noise_std)):
            raise ValueError("monitoring delay and noise must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    n_nodes: int = 4
    n_tasks: int = 6
    k_required: int = 3
    dt_seconds: float = 1.0
    max_time_seconds: float = 600.0
    decision_period_steps: int = 1
    cpu_capacity: float = 8.0
    memory_capacity: float = 32.0
    cpu_rate: float = 1.0
    base_failure_rate: float = 1.0e-4
    base_repair_rate: float = 2.0e-3
    checkpoint_size_bytes: int = 2_000_000
    checkpoint_interval_work: float = 20.0
    checkpoint_fixed_delay_seconds: float = 0.15
    restore_delay_seconds: float = 0.5
    link_bandwidth_bytes_per_second: float = 25_000_000.0
    normal_delay_mean_seconds: float = 0.015
    normal_delay_std_seconds: float = 0.004
    congested_delay_mean_seconds: float = 0.18
    congested_delay_std_seconds: float = 0.08
    congestion_enter_probability: float = 0.08
    congestion_leave_probability: float = 0.35
    max_migration_attempts: int = 8
    task_work_min: float = 40.0
    task_work_max: float = 160.0
    task_cpu_min: float = 1.0
    task_cpu_max: float = 4.0
    task_memory_min: float = 2.0
    task_memory_max: float = 12.0
    task_deadline_multiplier: float = 1.8
    delay_cost_weight: float = 0.5
    overdue_cost_weight: float = 0.5
    service_loss_cost_weight: float = 1.0
    progress_reward_weight: float = 0.25
    monitoring: MonitoringConfig = MonitoringConfig()

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if not 1 <= self.k_required <= self.n_nodes:
            raise ValueError("k_required must be in [1, n_nodes]")
        if self.dt_seconds <= 0 or self.max_time_seconds <= 0:
            raise ValueError("time values must be positive")
        if self.decision_period_steps < 1:
            raise ValueError("decision_period_steps must be positive")
        if self.cpu_capacity <= 0 or self.memory_capacity <= 0:
            raise ValueError("node capacities must be positive")
        if self.link_bandwidth_bytes_per_second <= 0:
            raise ValueError("link bandwidth must be positive")
        if self.task_work_min <= 0 or self.task_work_max < self.task_work_min:
            raise ValueError("invalid task work interval")
        if self.delay_cost_weight < 0:
            raise ValueError("delay_cost_weight must be non-negative")
        if self.overdue_cost_weight < 0:
            raise ValueError("overdue_cost_weight must be non-negative")
        if self.service_loss_cost_weight < 0:
            raise ValueError("service_loss_cost_weight must be non-negative")
        if self.progress_reward_weight < 0:
            raise ValueError("progress_reward_weight must be non-negative")
