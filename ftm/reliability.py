"""Two-state CTMC reliability calculations and sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def node_up_probability(initially_up: bool, failure_rate: float,
                        repair_rate: float, time_seconds: float) -> float:
    if min(failure_rate, repair_rate, time_seconds) < 0:
        raise ValueError("rates and time must be non-negative")
    total = failure_rate + repair_rate
    if total == 0:
        return float(initially_up)
    stationary_up = repair_rate / total
    initial = 1.0 if initially_up else 0.0
    return stationary_up + (initial - stationary_up) * math.exp(-total * time_seconds)


def transition_probabilities(failure_rate: float, repair_rate: float,
                             dt_seconds: float) -> tuple[float, float]:
    """Return (up_to_down, down_to_up) probabilities over a fixed interval."""
    if min(failure_rate, repair_rate, dt_seconds) < 0:
        raise ValueError("rates and dt must be non-negative")
    total = failure_rate + repair_rate
    if total == 0 or dt_seconds == 0:
        return 0.0, 0.0
    changed = -math.expm1(-total * dt_seconds)
    return failure_rate / total * changed, repair_rate / total * changed


def sample_node_state(is_up: bool, failure_rate: float, repair_rate: float,
                      dt_seconds: float, rng: random.Random) -> tuple[bool, str | None]:
    p_fail, p_repair = transition_probabilities(failure_rate, repair_rate, dt_seconds)
    if is_up and rng.random() < p_fail:
        return False, "node_failure"
    if not is_up and rng.random() < p_repair:
        return True, "node_repair"
    return is_up, None


def k_of_n_availability(up_probabilities: Sequence[float], k_required: int) -> float:
    """Poisson-binomial tail P(number of available nodes >= k)."""
    n = len(up_probabilities)
    if not 1 <= k_required <= n:
        raise ValueError("k_required must be in [1, number of nodes]")
    mass = [0.0] * (n + 1)
    mass[0] = 1.0
    processed = 0
    for p in up_probabilities:
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        nxt = [0.0] * (n + 1)
        for successes in range(processed + 1):
            nxt[successes] += mass[successes] * (1.0 - p)
            nxt[successes + 1] += mass[successes] * p
        mass = nxt
        processed += 1
    return sum(mass[k_required:])
