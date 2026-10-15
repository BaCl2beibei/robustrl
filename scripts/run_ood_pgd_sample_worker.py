"""Test selected PPO/PGD-PPO models by sampling from masked policy logits.

Run one training seed per process. The original argmax test files are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ftm.config import MonitoringConfig, SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import _episode_row, bootstrap_interval, build_evaluation_cases
from ftm.main import run_episode
from ftm.observation import encode_observation
from ftm.trained_policy import TrainedPolicy, run_trained_episode
from run_ood_pgd_experiment import POLICIES, SCENARIOS


METRICS = (
    "task_completion_rate", "task_service_availability", "sla_violation_rate",
    "mean_completion_seconds_horizon_capped", "migration_count",
    "mean_migration_seconds", "total_reward",
)


class SampledPolicy(TrainedPolicy):
    def act(self, context) -> int:
        state = torch.tensor(encode_observation(context), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        mask = torch.tensor(context.action_mask, dtype=torch.bool,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state, mask)
            masked_logits = logits.masked_fill(~mask, -1.0e9)
            return int(torch.distributions.Categorical(
                logits=masked_logits).sample().item())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], cases: list, seed: int) -> tuple[list[dict], list[dict]]:
    summary, paired = [], []
    for case in cases:
        by_policy = {policy: [row for row in rows
                              if row["scenario_id"] == case.scenario_id
                              and row["policy"] == policy]
                     for policy in POLICIES}
        for policy, group in by_policy.items():
            for metric in METRICS:
                values = [float(row[metric]) for row in group
                          if row[metric] is not None]
                summary.append({
                    "scenario_id": case.scenario_id, "method": policy,
                    "metric": metric, "n": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "stdev": statistics.stdev(values) if len(values) > 1 else None,
                })
        for policy in ("time_min", "ppo", "pgd_ppo"):
            candidate = {row["seed"]: row for row in by_policy[policy]}
            for reference in ("no_migration", "ppo"):
                if policy == reference:
                    continue
                baseline = {row["seed"]: row for row in by_policy[reference]}
                for metric in ("task_completion_rate", "task_service_availability",
                               "sla_violation_rate", "total_reward"):
                    differences = [float(candidate[run_seed][metric])
                                   - float(baseline[run_seed][metric])
                                   for run_seed in sorted(candidate.keys() & baseline.keys())
                                   if candidate[run_seed][metric] is not None
                                   and baseline[run_seed][metric] is not None]
                    low, high = bootstrap_interval(differences, seed=seed,
                                                   resamples=1000)
                    paired.append({
                        "scenario_id": case.scenario_id, "method": policy,
                        "reference": reference, "metric": metric,
                        "n_pairs": len(differences),
                        "mean_difference": statistics.mean(differences)
                        if differences else None,
                        "ci95_low": low, "ci95_high": high,
                    })
    return summary, paired


def evaluate(source_root: Path, output_root: Path, seed: int, device: str) -> None:
    run = source_root / f"main_seed{seed}"
    for marker in ("TRAINING_COMPLETE.txt", "TEST_COMPLETE.txt"):
        if not (run / marker).is_file():
            raise RuntimeError(f"source run is incomplete: {run / marker}")
    source_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    values = source_config["simulation_config"]
    config = SimulationConfig(**(values | {
        "monitoring": MonitoringConfig(**values["monitoring"]),
    }))
    cases = build_evaluation_cases("all", "all", "all", config)
    cases = [case for case in cases if case.scenario_family in SCENARIOS
             and (case.split == "ood" or case.scenario_id in SCENARIOS)]
    if len(cases) != 15:
        raise RuntimeError(f"expected 15 ID/OOD conditions; got {len(cases)}")
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(1)
    controllers, selections = {}, {}
    for method in ("ppo", "pgd_ppo"):
        selected = json.loads((run / method / "selected.json").read_text(
            encoding="utf-8"))
        checkpoint = run / selected["checkpoint"].replace("\\", "/")
        controllers[method] = SampledPolicy(
            checkpoint, len(state), config.n_nodes + 1, device)
        selections[method] = selected | {"resolved_checkpoint": str(checkpoint.resolve())}

    output = output_root / f"main_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        count = (source_config["args"]["id_episodes"] if case.split == "id"
                 else source_config["args"]["ood_episodes"])
        for index in range(count):
            test_seed = seed * 1_000_000 + 800_000 + index
            for method in POLICIES:
                if method in controllers:
                    torch.manual_seed(test_seed)
                    metrics, _ = run_trained_episode(
                        method, case.scenario, test_seed, config, controllers[method])
                else:
                    metrics, _ = run_episode(method, case.scenario, test_seed, config)
                row = _episode_row(metrics, case)
                row["train_seed"] = seed
                row["test_episode"] = index + 1
                row["action_selection"] = ("categorical_sample" if method in controllers
                                           else "baseline")
                rows.append(row)
        print(f"TEST_CONDITION_COMPLETE seed={seed} scenario={case.scenario_id}",
              flush=True)
    summary, paired = summarize(rows, cases, seed)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    (output / "selected_models.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "source_run": str(run.resolve()), "train_seed": seed,
        "id_episodes_per_case_policy": source_config["args"]["id_episodes"],
        "ood_episodes_per_case_policy": source_config["args"]["ood_episodes"],
        "test_seed_rule": "train_seed * 1000000 + 800000 + episode_index",
        "action_selection": "masked-logit categorical sampling for PPO and PGD-PPO",
        "simulation_config": values,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "TEST_COMPLETE.txt").write_text(
        f"episodes={len(rows)}\n", encoding="utf-8")
    print(f"TEST_COMPLETE seed={seed} episodes={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    evaluate(args.source_root.resolve(), args.output_root.resolve(),
             args.seed, args.device)


if __name__ == "__main__":
    main()
