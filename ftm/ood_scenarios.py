"""Held-out, higher-severity evaluation scenarios.

These values are initial experimental settings and should be calibrated with
pilot runs; the training scenario definitions remain in ``environment.py``.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist
from typing import Sequence

from .config import ScenarioConfig, SimulationConfig
from .environment import scenario_by_name
from .surrogate import PARAMETER_NAMES, scenario_to_raw


OOD_FAMILIES = ("normal", "failure", "network", "load", "mixed")
OOD_SEVERITIES = ("base", "l1", "l2", "l3")


def scenario_family(scenario_name: str) -> str:
    if scenario_name in OOD_FAMILIES:
        return scenario_name
    if scenario_name.startswith("load_"):
        return "load"
    raise ValueError(f"unknown OOD scenario family: {scenario_name}")


def make_test_scenario(scenario_name: str, severity: str,
                       config: SimulationConfig) -> ScenarioConfig:
    """Return an ID or stronger OOD scenario without changing training presets."""
    if severity not in OOD_SEVERITIES:
        raise ValueError(f"unknown severity: {severity}")
    family = scenario_family(scenario_name)
    if severity == "base":
        return scenario_by_name(scenario_name, config)

    base = scenario_by_name(family, config)
    if family == "normal":
        changes = ({"failure_multiplier": 0.4, "congestion_enter": 0.02}
                   if severity == "l1" else
                   {"failure_multiplier": 0.6, "congestion_enter": 0.04}
                   if severity == "l2" else
                   {"failure_multiplier": 0.8, "congestion_enter": 0.06})
    elif family == "failure":
        changes = ({"failure_multiplier": 12.0, "common_shock_probability": 0.003}
                   if severity == "l1" else
                   {"failure_multiplier": 16.0, "common_shock_probability": 0.005}
                   if severity == "l2" else
                   {"failure_multiplier": 20.0, "common_shock_probability": 0.008})
    elif family == "network":
        changes = ({"congestion_enter": 0.50, "congestion_leave": 0.08}
                   if severity == "l1" else
                   {"congestion_enter": 0.65, "congestion_leave": 0.05}
                   if severity == "l2" else
                   {"congestion_enter": 0.80, "congestion_leave": 0.03})
    elif family == "load":
        changes = ({"load_burst_probability": 0.22, "load_burst_cpu": 0.65}
                   if severity == "l1" else
                   {"load_burst_probability": 0.30, "load_burst_cpu": 0.75}
                   if severity == "l2" else
                   {"load_burst_probability": 0.40, "load_burst_cpu": 0.85})
    else:
        changes = ({
            "failure_multiplier": 5.0,
            "congestion_enter": 0.25,
            "congestion_leave": 0.15,
            "common_shock_probability": 0.003,
            "load_burst_probability": 0.10,
            "load_burst_cpu": 0.65,
        } if severity == "l1" else {
            "failure_multiplier": 7.0,
            "congestion_enter": 0.35,
            "congestion_leave": 0.08,
            "common_shock_probability": 0.005,
            "load_burst_probability": 0.15,
            "load_burst_cpu": 0.75,
        } if severity == "l2" else {
            "failure_multiplier": 9.0,
            "congestion_enter": 0.45,
            "congestion_leave": 0.05,
            "common_shock_probability": 0.008,
            "load_burst_probability": 0.20,
            "load_burst_cpu": 0.85,
        })
    return replace(base, name=f"{family}__ood_{severity}", **changes)


def raw_coverage_diagnostics(cases: Sequence, config: SimulationConfig,
                             training_records: Sequence[dict] | None = None,
                             central_quantile: float = 0.90,
                             noise_std: float = 0.25) -> list[dict]:
    """Compare OOD raw parameters with empirical or theoretical training coverage.

    Empirical ranges deduplicate repeated transitions from the same sampled
    episode parameter vector. Without records, the central Gaussian interval is
    reported as a theoretical reference, not as observed dataset coverage.
    """
    if not 0.0 < central_quantile < 1.0 or noise_std < 0:
        raise ValueError("invalid coverage quantile or noise standard deviation")
    samples_by_family: dict[str, set[tuple[float, ...]]] = {}
    if training_records is not None:
        for record in training_records:
            name = record.get("scenario")
            parameters = record.get("parameters")
            if not isinstance(name, str) or not isinstance(parameters, (list, tuple)):
                continue
            try:
                family = scenario_family(name)
            except ValueError:
                continue
            row = tuple(float(value) for value in parameters)
            if len(row) != len(PARAMETER_NAMES):
                raise ValueError("surrogate coverage records must use eight parameters")
            samples_by_family.setdefault(family, set()).add(row)

    tail = (1.0 - central_quantile) / 2.0
    normal_z = NormalDist().inv_cdf(1.0 - tail)
    output = []
    for case in cases:
        if case.split != "ood":
            continue
        family = case.scenario_family
        template = scenario_by_name(family, config)
        test_raw = scenario_to_raw(case.scenario)
        family_samples = samples_by_family.get(family, set())
        if training_records is None:
            center = scenario_to_raw(template)
            radius = normal_z * noise_std
            lower = tuple(value - radius for value in center)
            upper = tuple(value + radius for value in center)
            source = "theoretical_gaussian_interval"
        elif not family_samples:
            output.append({
                "scenario_id": case.scenario_id,
                "scenario_family": family,
                "severity": case.severity,
                "coverage_source": "unavailable",
                "central_quantile": central_quantile,
                "training_samples": 0,
                "outside_parameters": [],
                "ood_by_main_coverage": None,
            })
            continue
        else:
            samples = list(family_samples)
            quantile_values = {}
            for index, name in enumerate(PARAMETER_NAMES):
                ordered = sorted(row[index] for row in samples)
                quantile_values[name] = (
                    _empirical_quantile(ordered, tail),
                    _empirical_quantile(ordered, 1.0 - tail),
                )
            lower = tuple(quantile_values[name][0]
                          for name in PARAMETER_NAMES)
            upper = tuple(quantile_values[name][1]
                          for name in PARAMETER_NAMES)
            source = "empirical_training_records"

        names = PARAMETER_NAMES
        parameter_report = {
            name: {
                "training_low": lower[index],
                "training_high": upper[index],
                "ood_raw": test_raw[index],
                "outside_main_coverage": not lower[index] <= test_raw[index] <= upper[index],
            }
            for index, name in enumerate(names)
        }
        outside = [name for name, values in parameter_report.items()
                   if values["outside_main_coverage"]]
        output.append({
            "scenario_id": case.scenario_id,
            "scenario_family": family,
            "severity": case.severity,
            "coverage_source": source,
            "central_quantile": central_quantile,
            "training_samples": (len(family_samples) if training_records is not None else None),
            "parameters": parameter_report,
            "outside_parameters": outside,
            "ood_by_main_coverage": bool(outside),
        })
    return output


def _empirical_quantile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        raise ValueError("cannot compute a quantile from no training values")
    position = quantile * (len(ordered_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    return (ordered_values[lower_index] * (1.0 - fraction)
            + ordered_values[upper_index] * fraction)
