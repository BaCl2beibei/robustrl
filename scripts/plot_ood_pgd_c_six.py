"""Validate six completed C-PPO/PGD-PPO seeds and plot four paired metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = tuple(range(1, 7))
FAMILIES = ("normal", "failure", "network", "load", "mixed")
SCENARIOS = FAMILIES + tuple(
    f"{family}__ood_{level}" for family in FAMILIES for level in ("l1", "l2"))
METHODS = ("no_migration", "time_min", "ppo", "pgd_ppo")
LABELS = ("No migration", "Time-Min", "PPO (C)", "PGD-PPO (C)")
COLORS = ("#80868e", "#b45c87", "#2877ad", "#659d62")
METRICS = (
    ("task_completion_rate", "Task completion rate"),
    ("mean_completion_seconds_horizon_capped", "Mean completion time (s)"),
    ("sla_violation_rate", "SLA violation rate"),
    ("migration_count", "Migrations per episode"),
)
SCHEDULE_FIELDS = (
    "episodes", "warmup", "block_size", "rollout_steps",
    "validation_interval", "validation_per_scenario", "id_episodes",
    "ood_episodes", "surrogate_episodes", "surrogate_epochs",
    "pretrain_episodes", "aux_beta", "pgd_steps", "pgd_restarts", "max_time",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect(root: Path) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    reference = None
    for seed in SEEDS:
        run = root / f"main_seed{seed}"
        for marker in ("PREPARE_COMPLETE.txt", "TRAINING_COMPLETE.txt",
                       "TEST_COMPLETE.txt"):
            if not (run / marker).is_file():
                raise RuntimeError(f"missing completion marker: {run / marker}")
        if not (run / "gate_pretraining.json").is_file():
            raise RuntimeError(f"C Gate pretraining is missing for seed {seed}")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        signature = (config["observation_schema_version"],
                     tuple(config["args"][name] for name in SCHEDULE_FIELDS))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise RuntimeError("training or test settings differ across seeds")
        for method in ("ppo", "pgd_ppo"):
            branch = run / method
            if not (branch / "BRANCH_COMPLETE.txt").is_file():
                raise RuntimeError(f"incomplete branch: {branch}")
            selected = json.loads((branch / "selected.json").read_text(encoding="utf-8"))
            if not (run / selected["checkpoint"]).is_file():
                raise RuntimeError(f"missing selected checkpoint: {branch}")
        for row in read_csv(run / "test" / "episodes.csv"):
            scenario, method = row["scenario_id"], row["policy"]
            if scenario not in SCENARIOS or method not in METHODS:
                raise RuntimeError(f"unexpected test condition: {scenario}/{method}")
            expected_split = "id" if scenario in FAMILIES else "ood"
            if row["split"] != expected_split:
                raise RuntimeError(f"wrong split for {scenario}")
            expected_seed = seed * 1_000_000 + 800_000
            if not expected_seed <= int(row["seed"]) < expected_seed + (
                    config["args"]["id_episodes"] if expected_split == "id"
                    else config["args"]["ood_episodes"]):
                raise RuntimeError(f"test seed outside paired schedule: seed {seed}")
            grouped[scenario, method, seed].append(row)
    expected_keys = {(scenario, method, seed)
                     for scenario in SCENARIOS for method in METHODS for seed in SEEDS}
    if set(grouped) != expected_keys:
        raise RuntimeError("six-seed test conditions are incomplete")
    args = config["args"]
    for (scenario, method, seed), rows in grouped.items():
        expected = args["id_episodes"] if scenario in FAMILIES else args["ood_episodes"]
        if len(rows) != expected or len({row["seed"] for row in rows}) != expected:
            raise RuntimeError(f"missing or duplicate tests: {scenario}/{method}/seed{seed}")
    return grouped, args


def summarize(grouped) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for metric, _ in METRICS:
                seed_means = []
                for seed in SEEDS:
                    values = [float(row[metric])
                              for row in grouped[scenario, method, seed]
                              if row[metric] not in (None, "")]
                    if len(values) != len(grouped[scenario, method, seed]):
                        raise RuntimeError(f"missing {metric}: {scenario}/{method}/seed{seed}")
                    seed_means.append(statistics.mean(values))
                rows.append({
                    "scenario_id": scenario,
                    "split": "id" if scenario in FAMILIES else "ood",
                    "method": method, "metric": metric, "n_seeds": len(SEEDS),
                    "mean": statistics.mean(seed_means),
                    "seed_sd": statistics.stdev(seed_means),
                    "episodes_per_seed": len(grouped[scenario, method, SEEDS[0]]),
                })
    return rows


def plot(root: Path) -> None:
    output = root / "figures"
    if output.exists():
        raise FileExistsError(f"figure output already exists: {output}")
    grouped, args = collect(root)
    summary = summarize(grouped)
    output.mkdir(parents=True)
    with (output / "four_metric_summary.csv").open(
            "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    by_key = {(row["scenario_id"], row["method"], row["metric"]): row
              for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(19, 11))
    fig.subplots_adjust(top=0.83, bottom=0.13, hspace=0.48, wspace=0.18)
    xs = np.arange(len(SCENARIOS))
    width = 0.19
    for ax, (metric, title) in zip(axes.flat, METRICS):
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            values = [by_key[scenario, method, metric]["mean"] for scenario in SCENARIOS]
            ax.bar(xs + (index - 1.5) * width, values, width,
                   label=label, color=color)
        ax.axvline(4.5, color="#444444", linewidth=0.9, alpha=0.5)
        ax.set_title(title)
        ax.set_xticks(xs, SCENARIOS, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        if metric in ("task_completion_rate", "sla_violation_rate"):
            ax.set_ylim(0, 1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Six-seed PPO (C) and PGD-PPO (C): paired ID/OOD comparison",
                 y=0.995)
    fig.text(0.5, 0.025, "Unfinished tasks count as the simulation horizon in mean completion time. "
             f"ID: {args['id_episodes']} episodes/seed; OOD: {args['ood_episodes']} episodes/seed.",
             ha="center", fontsize=10)
    image = output / "four_metric_comparison.png"
    fig.savefig(image, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (output / "FOUR_METRIC_PLOT_COMPLETE.txt").write_text(
        f"seeds={len(SEEDS)}\nsummary={len(summary)}\nfigure={image.name}\n",
        encoding="utf-8")
    print(f"FOUR_METRIC_PLOT_COMPLETE {image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)
