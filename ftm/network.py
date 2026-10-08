"""Directed links and bandwidth-limited checkpoint transfer."""

from __future__ import annotations

from dataclasses import dataclass

from .config import SimulationConfig
from .disturbances import DisturbanceModel


@dataclass
class Link:
    bandwidth_bytes_per_second: float
    delay_seconds: float = 0.01
    congested: bool = False
    reachable: bool = True


class Network:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        n = config.n_nodes
        self.links = [[Link(config.link_bandwidth_bytes_per_second) for _ in range(n)]
                      for _ in range(n)]

    def update(self, disturbance: DisturbanceModel) -> None:
        for i, row in enumerate(self.links):
            for j, link in enumerate(row):
                if i == j:
                    link.delay_seconds = 0.0
                    link.congested = False
                    continue
                link.congested = disturbance.network_congested[i][j]
                link.delay_seconds = disturbance.link_delay_seconds(link.congested)

    def transfer_rate(self, source: int | None, target: int) -> float:
        if source is None:
            return self.config.link_bandwidth_bytes_per_second
        link = self.links[source][target]
        if not link.reachable:
            return 0.0
        reduction = 0.65 if link.congested else 0.0
        return link.bandwidth_bytes_per_second * (1.0 - reduction)

    def transfer_delay(self, source: int | None, target: int) -> float:
        return 0.0 if source is None else self.links[source][target].delay_seconds

    def estimate_transfer_seconds(self, source: int | None, target: int,
                                  nbytes: int) -> float:
        rate = self.transfer_rate(source, target)
        if rate <= 0:
            return float("inf")
        return self.transfer_delay(source, target) + nbytes / rate
