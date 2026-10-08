"""Simple estimators for nominal reliability and a one-dimensional BGMM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RateEstimate:
    failure_rate: float
    repair_rate: float


@dataclass(frozen=True)
class BGMMFit:
    normal_weight: float
    normal_mean: float
    normal_std: float
    congested_mean: float
    congested_std: float
    log_likelihood: float


def estimate_ctmc_rates(failures: int, repairs: int,
                        up_exposure_seconds: float,
                        down_exposure_seconds: float,
                        prior_events: float = 0.0,
                        prior_exposure_seconds: float = 0.0) -> RateEstimate:
    """MLE with optional Gamma-style event/exposure smoothing."""
    if min(failures, repairs, up_exposure_seconds, down_exposure_seconds,
           prior_events, prior_exposure_seconds) < 0:
        raise ValueError("counts and exposure times must be non-negative")
    lam = ((failures + prior_events)
           / max(up_exposure_seconds + prior_exposure_seconds, 1e-12))
    mu = ((repairs + prior_events)
          / max(down_exposure_seconds + prior_exposure_seconds, 1e-12))
    return RateEstimate(lam, mu)


def _normal_density(value: float, mean: float, std: float) -> float:
    variance = max(std * std, 1e-18)
    return math.exp(-0.5 * (value - mean) ** 2 / variance) / math.sqrt(
        2.0 * math.pi * variance
    )


def fit_bgmm(delays_seconds: Sequence[float], iterations: int = 200,
             min_std: float = 1e-5, min_weight: float = 1e-4) -> BGMMFit:
    """Fit a two-Gaussian mixture with EM and label the lower mean as normal.

    This dependency-free estimator is intended for synthetic/clean measurements.
    For large or contaminated datasets, use a tested mixture implementation and
    preserve train/calibration/test separation.
    """
    data = [float(value) for value in delays_seconds]
    if len(data) < 4 or any(value <= 0 or not math.isfinite(value) for value in data):
        raise ValueError("BGMM fitting needs at least four finite positive delays")
    ordered = sorted(data)
    mean1, mean2 = ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4]
    overall = sum(data) / len(data)
    variance = sum((x - overall) ** 2 for x in data) / len(data)
    std1 = std2 = max(min_std, math.sqrt(variance))
    weight1 = 0.5
    responsibilities = [0.5] * len(data)
    for _ in range(iterations):
        for index, value in enumerate(data):
            p1 = weight1 * _normal_density(value, mean1, std1)
            p2 = (1.0 - weight1) * _normal_density(value, mean2, std2)
            responsibilities[index] = p1 / max(p1 + p2, 1e-300)
        mass1 = sum(responsibilities)
        mass2 = len(data) - mass1
        weight1 = min(1.0 - min_weight,
                      max(min_weight, mass1 / len(data)))
        mean1 = sum(r * x for r, x in zip(responsibilities, data)) / max(mass1, 1e-12)
        mean2 = sum((1.0 - r) * x for r, x in zip(responsibilities, data)) / max(mass2, 1e-12)
        std1 = max(min_std, math.sqrt(sum(r * (x - mean1) ** 2
                                          for r, x in zip(responsibilities, data))
                                       / max(mass1, 1e-12)))
        std2 = max(min_std, math.sqrt(sum((1.0 - r) * (x - mean2) ** 2
                                          for r, x in zip(responsibilities, data))
                                       / max(mass2, 1e-12)))
    log_likelihood = sum(math.log(max(
        weight1 * _normal_density(x, mean1, std1)
        + (1.0 - weight1) * _normal_density(x, mean2, std2), 1e-300
    )) for x in data)
    if mean1 <= mean2:
        return BGMMFit(weight1, mean1, std1, mean2, std2, log_likelihood)
    return BGMMFit(1.0 - weight1, mean2, std2, mean1, std1, log_likelihood)
