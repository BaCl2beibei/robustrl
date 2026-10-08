"""Joint node, network and load disturbance sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ScenarioConfig, SimulationConfig
from .entities import Node
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
        self.burst_remaining = [0] * config.n_nodes

    def sample(self, nodes: list[Node], dt_seconds: float) -> DisturbanceSample:
        common = self.rng.random() < self.scenario.common_shock_probability
        rate_multiplier = self.scenario.failure_multiplier
        if common:
            rate_multiplier *= self.scenario.common_shock_failure_multiplier
        events: list[tuple[int, str]] = []
        for node in nodes:
            was_up = node.up
            node.up, event = sample_node_state(
                node.up, node.failure_rate * rate_multiplier,
                node.repair_rate, dt_seconds, self.rng
            )
            if event:
                events.append((node.node_id, event))
            if self.burst_remaining[node.node_id] > 0:
                self.burst_remaining[node.node_id] -= 1
            elif self.rng.random() < self.scenario.load_burst_probability:
                self.burst_remaining[node.node_id] = self.scenario.load_burst_duration_steps
            burst = self.burst_remaining[node.node_id] > 0
            baseline = 0.08 + 0.05 * self.rng.random()
            node.external_cpu_fraction = min(
                0.95, baseline + (self.scenario.load_burst_cpu if burst else 0.0)
            )
            node.external_memory = node.memory_capacity * (0.04 + 0.04 * self.rng.random())
        self._advance_network_regimes(common)
        return DisturbanceSample(common, tuple(events), any(self.burst_remaining))

    def _advance_network_regimes(self, common_shock: bool) -> None:
        n = self.config.n_nodes
        p_enter = self.scenario.congestion_enter
        if common_shock:
            p_enter = min(1.0, p_enter + self.scenario.common_shock_congestion_boost)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.network_congested[i][j] = False
                elif self.network_congested[i][j]:
                    if self.rng.random() < self.scenario.congestion_leave:
                        self.network_congested[i][j] = False
                elif self.rng.random() < p_enter:
                    self.network_congested[i][j] = True

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
