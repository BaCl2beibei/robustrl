"""Multi-scenario, multi-seed evaluation with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ScenarioConfig, SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .main import POLICIES, SCENARIOS, run_episode
from .ood_scenarios import (OOD_FAMILIES, make_test_scenario,
                            raw_coverage_diagnostics, scenario_family)
from .surrogate import PARAMETER_NAMES
from .trained_policy import TrainedPolicy, run_trained_episode


METRICS = ("task_completion_rate", "node_failure_continuation_success_rate",
           "task_service_availability", "cluster_availability", "mean_completion_seconds",
           "mean_completion_seconds_horizon_capped", "sla_violation_rate",
           "migration_count", "migration_attempt_count", "migration_failure_count",
           "migration_incomplete_count",
           "mean_migration_seconds", "total_migration_seconds", "cpu_utilization",
           "task_cpu_utilization", "reserved_cpu_utilization", "external_cpu_utilization",
           "memory_utilization", "total_reward")
TRAINED_POLICIES = ("trained_ppo", "trained_surrogate_ppo")


@dataclass(frozen=True)
class EvaluationCase:
    scenario: ScenarioConfig
    scenario_family: str
    severity: str
    split: str

    @property
    def scenario_id(self) -> str:
        return self.scenario.name


def build_evaluation_cases(split: str, severity: str, scenario_name: str,
                           config: SimulationConfig) -> list[EvaluationCase]:
    """Build uniquely named ID/OOD cases while leaving training presets untouched."""
    if split not in ("id", "ood", "all"):
        raise ValueError("split must be id, ood, or all")
    if severity not in ("base", "l1", "l2", "l3", "all"):
        raise ValueError("severity must be base, l1, l2, l3, or all")
    if scenario_name != "all" and scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")

    include_id = split in ("id", "all") and severity in ("base", "all")
    include_ood = split in ("ood", "all") and severity in ("l1", "l2", "l3", "all")
    if split == "id" and severity in ("l1", "l2", "l3"):
        raise ValueError("ID split only supports severity=base or all")
    if split == "ood" and severity == "base":
        raise ValueError("OOD split requires severity=l1, l2, l3, or all")

    selected_names = SCENARIOS if scenario_name == "all" else (scenario_name,)
    cases: list[EvaluationCase] = []
    if include_id:
        for name in selected_names:
            cases.append(EvaluationCase(make_test_scenario(name, "base", config),
                                        scenario_family(name), "base", "id"))
    if include_ood:
        if scenario_name == "all":
            families = OOD_FAMILIES
        else:
            family = scenario_family(scenario_name)
            if family not in OOD_FAMILIES:
                raise ValueError(f"OOD evaluation is unavailable for {scenario_name}")
            families = (family,)
        # Keep the existing 'all' suite stable; L3 is an explicit held-out test.
        severities = (("l1", "l2") if severity == "all" else (severity,))
        for family in families:
            for level in severities:
                cases.append(EvaluationCase(make_test_scenario(family, level, config),
                                            family, level, "ood"))
    return cases


def _scenario_key(row: dict) -> str:
    return str(row.get("scenario_id", row.get("scenario", "")))


def _metadata(row: dict) -> dict:
    return {
        "scenario": row.get("scenario", _scenario_key(row)),
        "scenario_id": _scenario_key(row),
        "scenario_family": row.get("scenario_family"),
        "severity": row.get("severity"),
        "split": row.get("split"),
    }


def _episode_row(metrics, case: EvaluationCase) -> dict:
    row = metrics.to_dict()
    row.update({
        "scenario_family": case.scenario_family,
        "severity": case.severity,
        "split": case.split,
        "scenario_id": case.scenario_id,
    })
    for name in PARAMETER_NAMES:
        row[f"disturbance_{name}"] = float(getattr(case.scenario, name))
    return row


def bootstrap_interval(values: list[float], seed: int = 0,
                       resamples: int = 4000,
                       confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for a sample mean (descriptive, not a test)."""
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    if resamples < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("resamples must be >=100 and confidence must be in (0,1)")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * resamples))
    high_index = min(resamples - 1, int((1.0 - tail) * resamples) - 1)
    return means[low_index], means[high_index]


def summarize(rows: list[dict], seed: int = 0,
              resamples: int = 4000) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(_scenario_key(row), row["policy"])].append(row)
    summary = []
    for (scenario_id, policy), group in sorted(groups.items()):
        metadata = _metadata(group[0])
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            low, high = bootstrap_interval(values, seed + len(summary), resamples)
            summary.append({
                **metadata, "policy": policy, "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
                "ci95_low": low, "ci95_high": high,
            })
    return summary


def paired_bootstrap(rows: list[dict], reference: str = "no_migration",
                     seed: int = 0, resamples: int = 4000,
                     permutation_resamples: int = 10_000) -> list[dict]:
    by_key = {(_scenario_key(row), row["policy"], row["seed"]): row for row in rows}
    scenarios = sorted({_scenario_key(row) for row in rows})
    policies = sorted({row["policy"] for row in rows if row["policy"] != reference})
    output = []
    for scenario in scenarios:
        for policy in policies:
            common_seeds = sorted({row["seed"] for row in rows
                                   if _scenario_key(row) == scenario and row["policy"] == policy}
                                  & {row["seed"] for row in rows
                                     if _scenario_key(row) == scenario
                                     and row["policy"] == reference})
            for metric in METRICS:
                differences = []
                for run_seed in common_seeds:
                    value = by_key[(scenario, policy, run_seed)].get(metric)
                    baseline = by_key[(scenario, reference, run_seed)].get(metric)
                    if value is not None and baseline is not None:
                        differences.append(float(value) - float(baseline))
                low, high = bootstrap_interval(differences, seed + len(output), resamples)
                metadata = _metadata(by_key[(scenario, policy, common_seeds[0])]) \
                    if common_seeds else {"scenario": scenario, "scenario_id": scenario,
                                          "scenario_family": None, "severity": None,
                                          "split": None}
                output.append({
                    **metadata, "policy": policy, "reference": reference,
                    "metric": metric, "n_pairs": len(differences),
                    "mean_difference": statistics.mean(differences) if differences else None,
                    "ci95_low": low, "ci95_high": high,
                    "p_value_two_sided": paired_sign_flip_pvalue(
                        differences, seed + len(output), permutation_resamples
                    ),
                    "p_value_holm_scenario": None,
                })
    # Holm adjustment controls family-wise error across policies/metrics within each scenario.
    for scenario in scenarios:
        family = [(index, row["p_value_two_sided"]) for index, row in enumerate(output)
                  if row["scenario_id"] == scenario
                  and row["p_value_two_sided"] is not None]
        family.sort(key=lambda pair: pair[1])
        running_max = 0.0
        family_size = len(family)
        for rank, (index, p_value) in enumerate(family):
            adjusted = min(1.0, (family_size - rank) * p_value)
            running_max = max(running_max, adjusted)
            output[index]["p_value_holm_scenario"] = running_max
    return output


def paired_sign_flip_pvalue(differences: list[float], seed: int = 0,
                            resamples: int = 10_000,
                            exact_limit: int = 12) -> float | None:
    """Two-sided paired sign-flip p-value for a mean difference.

    Exact enumeration is used for small n; otherwise a Monte Carlo randomization
    test is used. Its null requires paired differences to be sign-exchangeable.
    """
    if not differences:
        return None
    observed = abs(statistics.mean(differences))
    tolerance = 1e-12
    if len(differences) <= exact_limit:
        extreme = total = 0
        for mask in range(1 << len(differences)):
            value = sum(item if mask & (1 << index) else -item
                        for index, item in enumerate(differences)) / len(differences)
            total += 1
            if abs(value) >= observed - tolerance:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        value = sum(item if rng.random() < 0.5 else -item for item in differences)
        if abs(value / len(differences)) >= observed - tolerance:
            extreme += 1
    return (extreme + 1) / (resamples + 1)


def _write_csv(path: Path, rows: list[dict],
               fieldnames: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = list(rows[0]) if rows else list(fieldnames or ())
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_training_parameter_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record.get("scenario"), str) or not isinstance(
                    record.get("parameters"), list):
                raise ValueError(f"missing scenario/parameters at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError("training dataset contains no parameter records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30,
                        help="number of paired seeds per scenario and policy")
    parser.add_argument("--policy", choices=("all",) + POLICIES[1:] + TRAINED_POLICIES,
                        default="all")
    parser.add_argument("--checkpoint", type=Path,
                        help="PPO checkpoint for a trained_* policy")
    parser.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all")
    parser.add_argument("--split", choices=("id", "ood", "all"), default="id",
                        help="evaluate base ID conditions, stronger OOD conditions, or both")
    parser.add_argument("--severity", choices=("base", "l1", "l2", "l3", "all"),
                        default="base", help="severity level for the requested split")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("runs/research"))
    parser.add_argument("--training-dataset", type=Path,
                        help="optional JSONL file for empirical OOD coverage diagnostics")
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--permutation-resamples", type=int, default=10_000)
    parser.add_argument("--max-time", type=float, default=600.0)
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("use at least 2 paired seeds to report an interval")
    if args.bootstrap_resamples < 100 or args.permutation_resamples < 100:
        parser.error("bootstrap/permutation resamples must be >=100")
    config = SimulationConfig(max_time_seconds=args.max_time)
    try:
        cases = build_evaluation_cases(args.split, args.severity, args.scenario, config)
    except ValueError as exc:
        parser.error(str(exc))
    training_records = None
    if args.training_dataset is not None:
        try:
            training_records = read_training_parameter_records(args.training_dataset)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    coverage_diagnostics = raw_coverage_diagnostics(
        cases, config, training_records=training_records
    )
    for diagnostic in coverage_diagnostics:
        status = diagnostic["ood_by_main_coverage"]
        if status is None:
            label = "unavailable"
        else:
            label = "outside" if status else "inside"
        print(f"OOD coverage {diagnostic['scenario_id']}: {label} "
              f"({diagnostic['coverage_source']})")
    if args.policy == "all":
        policies = POLICIES[1:]
    elif args.policy == "no_migration":
        policies = ("no_migration",)
    else:
        # Always retain the paired reference when a single comparison policy is selected.
        policies = ("no_migration", args.policy)
    controller = None
    trained = any(policy in TRAINED_POLICIES for policy in policies)
    if trained and args.checkpoint is None:
        parser.error("--checkpoint is required for a trained PPO policy")
    if args.checkpoint is not None:
        probe = MigrationEnvironment(config, scenario_by_name("normal", config))
        observation_dim = len(probe.reset(args.seed)[0])
        controller = TrainedPolicy(args.checkpoint, observation_dim,
                                   config.n_nodes + 1)
        if "trained_ppo" in policies and controller.is_adversarial:
            parser.error("trained_ppo expects a standard PPO checkpoint")
        if "trained_surrogate_ppo" in policies and not controller.is_adversarial:
            parser.error("trained_surrogate_ppo expects a surrogate-PGD checkpoint")
    raw_rows = []
    total = len(policies) * len(cases) * args.episodes
    completed = 0
    for case in cases:
        for episode in range(args.episodes):
            run_seed = args.seed + episode
            for policy_name in policies:
                if policy_name in TRAINED_POLICIES:
                    metrics, _ = run_trained_episode(
                        policy_name, case.scenario, run_seed, config, controller
                    )
                else:
                    metrics, _ = run_episode(policy_name, case.scenario, run_seed, config)
                raw_rows.append(_episode_row(metrics, case))
                completed += 1
                print(f"{completed}/{total} {case.scenario_id}/{policy_name} seed={run_seed}")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "episodes.csv", raw_rows)
    _write_csv(args.output / "summary_bootstrap_ci.csv",
               summarize(raw_rows, args.seed, args.bootstrap_resamples))
    _write_csv(args.output / "paired_vs_no_migration.csv",
               paired_bootstrap(raw_rows, seed=args.seed,
                                resamples=args.bootstrap_resamples,
                                permutation_resamples=args.permutation_resamples),
               fieldnames=("scenario", "scenario_id", "scenario_family", "severity",
                           "split", "policy", "reference", "metric", "n_pairs",
                           "mean_difference", "ci95_low", "ci95_high",
                           "p_value_two_sided", "p_value_holm_scenario"))
    metadata = {
        "seed": args.seed,
        "episodes_per_scenario_policy": args.episodes,
        "split": args.split,
        "severity": args.severity,
        "scenarios": [case.scenario_id for case in cases],
        "evaluation_cases": [{
            "scenario_id": case.scenario_id,
            "scenario_family": case.scenario_family,
            "severity": case.severity,
            "split": case.split,
            "parameters": {name: getattr(case.scenario, name)
                           for name in PARAMETER_NAMES},
        } for case in cases],
        "training_dataset": (None if args.training_dataset is None
                             else str(args.training_dataset.resolve())),
        "training_coverage_diagnostics": coverage_diagnostics,
        "policies": list(policies),
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutation_resamples": args.permutation_resamples,
        "confidence": 0.95,
        "config": asdict(config),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"raw, summary and paired comparison tables written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
