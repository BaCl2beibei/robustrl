"""Sensitivity-shaped ellipsoid and projected-gradient inner minimization."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence


def sensitivity_axes(gradient: Sequence[float], base_scales: Sequence[float],
                     min_fraction: float = 0.05) -> tuple[float, ...]:
    if len(gradient) != len(base_scales) or not gradient:
        raise ValueError("gradient and base_scales must have equal non-zero lengths")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must be in (0, 1]")
    maximum = max(abs(value) for value in gradient)
    normalized = ([0.0] * len(gradient) if maximum <= 1e-15 else
                  [abs(value) / maximum for value in gradient])
    return tuple(scale * (min_fraction + (1.0 - min_fraction) * value)
                 for scale, value in zip(base_scales, normalized))


@dataclass(frozen=True)
class Ellipsoid:
    center: tuple[float, ...]
    axes: tuple[float, ...]
    alpha: float

    def __post_init__(self) -> None:
        if len(self.center) != len(self.axes) or not self.center:
            raise ValueError("center and axes must have equal non-zero lengths")
        if self.alpha < 0 or any(axis <= 0 for axis in self.axes):
            raise ValueError("alpha must be non-negative and axes must be positive")

    def point(self, z: Sequence[float]) -> tuple[float, ...]:
        if len(z) != len(self.center):
            raise ValueError("dimension mismatch")
        return tuple(center + self.alpha * axis * zi
                     for center, axis, zi in zip(self.center, self.axes, z))

    def contains(self, point: Sequence[float], tolerance: float = 1e-9) -> bool:
        if len(point) != len(self.center):
            return False
        if self.alpha == 0:
            return all(abs(x - c) <= tolerance for x, c in zip(point, self.center))
        norm2 = sum(((x - c) / (self.alpha * axis)) ** 2
                    for x, c, axis in zip(point, self.center, self.axes))
        return norm2 <= 1.0 + tolerance


def project_unit_ball(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def projected_gradient_minimize(ellipsoid: Ellipsoid,
                                objective: Callable[[tuple[float, ...]], float],
                                gradient: Callable[[tuple[float, ...]], Sequence[float]],
                                steps: int = 20, learning_rate: float = 0.1
                                ) -> tuple[tuple[float, ...], float]:
    """PGD in unit-ball coordinates; result is a local worst-case candidate."""
    if steps < 0 or learning_rate <= 0:
        raise ValueError("steps must be non-negative and learning_rate positive")
    dimension = len(ellipsoid.center)
    starts = [tuple(0.0 for _ in range(dimension))]
    for index in range(dimension):
        for sign in (-1.0, 1.0):
            starts.append(tuple(sign if j == index else 0.0 for j in range(dimension)))
    best_point = ellipsoid.center
    best_value = objective(best_point)
    for z in starts:
        for _ in range(steps):
            point = ellipsoid.point(z)
            grad_point = gradient(point)
            if len(grad_point) != dimension:
                raise ValueError("gradient dimension mismatch")
            grad_z = tuple(g * ellipsoid.alpha * axis
                           for g, axis in zip(grad_point, ellipsoid.axes))
            z = project_unit_ball(tuple(zi - learning_rate * gi
                                        for zi, gi in zip(z, grad_z)))
        point = ellipsoid.point(z)
        value = objective(point)
        if value < best_value:
            best_point, best_value = point, value
    return best_point, best_value


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def decode_robust_parameters(raw: Sequence[float], epsilon: float = 1e-8) -> dict[str, float]:
    """Map ten unconstrained coordinates to legal physical parameters.

    Order: failure rate, repair rate, congestion-enter probability,
    congestion-leave probability, normal-delay mean, congested-delay increment,
    normal-delay std, congested-delay std, bandwidth loss, burst probability.
    """
    if len(raw) != 10:
        raise ValueError("raw robust parameter vector must have length 10")
    failure_rate = softplus(raw[0]) + epsilon
    repair_rate = softplus(raw[1]) + epsilon
    normal_mean = softplus(raw[4]) + epsilon
    return {
        "failure_rate": failure_rate,
        "repair_rate": repair_rate,
        "congestion_enter": sigmoid(raw[2]),
        "congestion_leave": sigmoid(raw[3]),
        "normal_delay_mean": normal_mean,
        "congested_delay_mean": normal_mean + softplus(raw[5]) + epsilon,
        "normal_delay_std": softplus(raw[6]) + epsilon,
        "congested_delay_std": softplus(raw[7]) + epsilon,
        "bandwidth_loss": 0.95 * sigmoid(raw[8]),
        "load_burst_probability": sigmoid(raw[9]),
    }


class RollingUncertaintyUpdater:
    """Update center, empirical parameter scales and sensitivity axes by window."""

    def __init__(self, dimension: int, window_size: int = 256,
                 update_every: int = 32, alpha: float = 1.0,
                 gradient_ema: float = 0.8, minimum_scale: float = 1e-4,
                 min_axis_fraction: float = 0.05) -> None:
        if dimension < 1 or window_size < 2 or update_every < 1:
            raise ValueError("invalid rolling uncertainty dimensions")
        if not 0.0 <= gradient_ema < 1.0:
            raise ValueError("gradient_ema must be in [0, 1)")
        self.dimension = dimension
        self.records: deque[tuple[float, ...]] = deque(maxlen=window_size)
        self.update_every = update_every
        self.alpha = alpha
        self.gradient_ema = gradient_ema
        self.minimum_scale = minimum_scale
        self.min_axis_fraction = min_axis_fraction
        self.gradient_average = [0.0] * dimension
        self.updates = 0
        self.current: Ellipsoid | None = None

    def add(self, parameter_estimate: Sequence[float],
            sensitivity: Sequence[float]) -> Ellipsoid | None:
        if len(parameter_estimate) != self.dimension or len(sensitivity) != self.dimension:
            raise ValueError("parameter and sensitivity dimensions must match")
        self.records.append(tuple(float(x) for x in parameter_estimate))
        beta = self.gradient_ema
        self.gradient_average = [beta * old + (1.0 - beta) * float(new)
                                 for old, new in zip(self.gradient_average, sensitivity)]
        self.updates += 1
        if self.updates % self.update_every != 0 or len(self.records) < 2:
            return self.current
        center = tuple(statistics.mean(row[i] for row in self.records)
                       for i in range(self.dimension))
        scales = tuple(max(self.minimum_scale,
                           statistics.stdev(row[i] for row in self.records))
                       for i in range(self.dimension))
        axes = sensitivity_axes(self.gradient_average, scales,
                                self.min_axis_fraction)
        self.current = Ellipsoid(center, axes, self.alpha)
        return self.current
