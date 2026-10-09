"""Joint node, network and load disturbance sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ScenarioConfig, SimulationConfig
from .entities import Node
from .policies import predict_burst_remaining
from .reliability import sample_node_state


@dataclass(frozen=True)
class DisturbanceSample:
    common_shock: bool
    failed_or_repaired: tuple[tuple[int, str], ...]
    load_burst: bool


class DisturbanceModel:
    """Samples a persistent network regime and CTMC node health per step."""

    def __init__(self, config: SimulationConfig, scenario: ScenarioConfig,
                 rng: random.Random) -> None:
        self.config = config
        self.scenario = scenario
        self.rng = rng
        self.network_congested = [[False] * config.n_nodes for _ in range(config.n_nodes)]
        self.network_disconnected = [[False] * config.n_nodes for _ in range(config.n_nodes)]
        self.burst_remaining = [0.0] * config.n_nodes
        self.burst_elapsed_seconds = [0.0] * config.n_nodes
        self.burst_total_seconds = [0.0] * config.n_nodes
        self.burst_history_seconds: list[list[float]] = [
            [] for _ in range(config.n_nodes)
        ]

    def predict_burst_remaining_seconds(self, node_id: int) -> float:
        if self.burst_remaining[node_id] <= 0.0:
            return 0.0
        duration_min, duration_max = self.scenario.load_burst_duration_bounds(
            self.config.dt_seconds
        )
        default_duration = (duration_min + duration_max) / 2.0
        return predict_burst_remaining(
            self.burst_elapsed_seconds[node_id], self.burst_history_seconds[node_id],
            default_duration,
        )

    def sample(self, nodes: list[Node], dt_seconds: float) -> DisturbanceSample:
        common = self.rng.random() < self.scenario.common_shock_probability
        rate_multiplier = self.scenario.failure_multiplier
        if common:
            rate_multiplier *= self.scenario.common_shock_failure_multiplier
        events: list[tuple[int, str]] = []
        any_burst_active = False
        for node in nodes:
            was_up = node.up
            node.up, event = sample_node_state(
                node.up, node.failure_rate * rate_multiplier,
                node.repair_rate, dt_seconds, self.rng
            )
            if event:
                events.append((node.node_id, event))
            baseline = 0.08 + 0.05 * self.rng.random()
            node_id = node.node_id
            if (self.burst_remaining[node_id] <= 0.0
                    and self.rng.random() < self.scenario.load_burst_probability):
                duration_min, duration_max = self.scenario.load_burst_duration_bounds(
                    self.config.dt_seconds
                )
                duration = self.rng.uniform(duration_min, duration_max)
                self.burst_remaining[node_id] = duration
                self.burst_total_seconds[node_id] = duration
                self.burst_elapsed_seconds[node_id] = 0.0

            active_burst_seconds = min(dt_seconds, self.burst_remaining[node_id])
            burst_fraction = active_burst_seconds / dt_seconds
            burst = active_burst_seconds > 0.0
            any_burst_active = any_burst_active or burst
            node.external_cpu_fraction = min(
                0.95, baseline + self.scenario.load_burst_cpu * burst_fraction
            )
            node.external_memory = node.memory_capacity * (0.04 + 0.04 * self.rng.random())
            if burst:
                self.burst_elapsed_seconds[node_id] += active_burst_seconds
                self.burst_remaining[node_id] = max(
                    0.0, self.burst_remaining[node_id] - dt_seconds
                )
                if self.burst_remaining[node_id] <= 1e-9:
                    self.burst_history_seconds[node_id].append(
                        self.burst_total_seconds[node_id]
                    )
                    self.burst_total_seconds[node_id] = 0.0
                    self.burst_elapsed_seconds[node_id] = 0.0
            else:
                node.normal_external_cpu_fraction = (
                    0.9 * node.normal_external_cpu_fraction + 0.1 * baseline
                )
        self._advance_network_regimes(common)
        return DisturbanceSample(common, tuple(events), any_burst_active)

    def _advance_network_regimes(self, common_shock: bool) -> None:
        n = self.config.n_nodes
        p_enter = self.scenario.congestion_enter
        p_disconnect = self.scenario.disconnect_enter
        if common_shock:
            p_enter = min(1.0, p_enter + self.scenario.common_shock_congestion_boost)
            p_disconnect = min(
                1.0, p_disconnect + self.scenario.common_shock_disconnect_boost
            )
        for i in range(n):
            self.network_congested[i][i] = False
            self.network_disconnected[i][i] = False
            for j in range(i + 1, n):
                disconnected = self.network_disconnected[i][j]
                congested = self.network_congested[i][j]
                if disconnected:
                    if self.rng.random() < self.scenario.disconnect_leave:
                        disconnected = False
                        congested = self.rng.random() < p_enter
                elif self.rng.random() < p_disconnect:
                    disconnected = True
                    congested = False
                elif congested:
                    if self.rng.random() < self.scenario.congestion_leave:
                        congested = False
                elif self.rng.random() < p_enter:
                    congested = True
                self.network_disconnected[i][j] = disconnected
                self.network_disconnected[j][i] = disconnected
                self.network_congested[i][j] = congested
                self.network_congested[j][i] = congested

    def link_delay_seconds(self, congested: bool) -> float:
        if congested:
            mean = self.config.congested_delay_mean_seconds
            std = self.config.congested_delay_std_seconds
        else:
            mean = self.config.normal_delay_mean_seconds
            std = self.config.normal_delay_std_seconds
        for _ in range(100):
            value = self.rng.gauss(mean, std)
            if value > 1e-6:
                return value
        return max(1e-6, mean)
