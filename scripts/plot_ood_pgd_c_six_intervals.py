"""Plot six-seed training curves and test metrics with between-seed 95% CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ood_pgd_c_six import (COLORS, LABELS, METHODS, METRICS,
                                SCENARIOS, SEEDS, collect, summarize)

T_95_DF5 = 2.5705818366147395
TRAIN_METHODS = ("ppo", "pgd_ppo")
TRAIN_LABELS = ("PPO (C)", "PGD-PPO (C)")
TRAIN_COLORS = (COLORS[2], COLORS[3])


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def interval(values) -> tuple[float, float, float]:
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("a complete set of six finite seed values is required")
    mean = statistics.mean(values)
    half_width = T_95_DF5 * statistics.stdev(values) / math.sqrt(len(SEEDS))
    return mean, mean - half_width, mean + half_width


def training_values(path: Path, first: int, last: int) -> dict[int, float]:
    values = {}
    for row in read_rows(path):
        episode = int(row["episode"])
        value = float(row["discounted_return"])
        if episode in values or not math.isfinite(value):
            raise ValueError(f"duplicate or nonfinite training episode: {path} {episode}")
        values[episode] = value
    if set(values) != set(range(first, last + 1)):
        raise ValueError(f"training episodes are incomplete: {path}")
    return values


def validation_values(path: Path, episodes: list[int]) -> dict[int, float]:
    values = {}
    for row in read_rows(path):
        episode = int(row["episode"])
        value = float(row["score"])
        if episode in values or not math.isfinite(value):
            raise ValueError(f"duplicate or nonfinite validation episode: {path} {episode}")
        values[episode] = value
    if set(values) != set(episodes):
        raise ValueError(f"validation checkpoints are incomplete: {path}")
    return values


def training_summary(root: Path, args: dict) -> list[dict]:
    warmup, episodes = args["warmup"], args["episodes"]
    cadence = args["validation_interval"]
    bin_size = 200 if warmup % 200 == 0 and episodes % 200 == 0 else 5
    if warmup % bin_size or (episodes - warmup) % bin_size:
        raise ValueError("the episode schedule cannot be divided into common bins")
    validation_episodes = list(range(warmup + cadence, episodes + 1, cadence))
    if not validation_episodes or validation_episodes[-1] != episodes:
        validation_episodes.append(episodes)
    returns = {}
    scores = {}
    for seed in SEEDS:
        run = root / f"main_seed{seed}"
        returns["warmup", seed] = training_values(
            run / "warmup_training.csv", 1, warmup)
        warmup_score = float(json.loads((run / "warmup_score.json").read_text(
            encoding="utf-8"))["score"])
        if not math.isfinite(warmup_score):
            raise ValueError(f"nonfinite warmup score: seed {seed}")
        scores["warmup", seed] = {warmup: warmup_score}
        for method in TRAIN_METHODS:
            returns[method, seed] = training_values(
                run / method / "training.csv", warmup + 1, episodes)
            scores[method, seed] = validation_values(
                run / method / "validation.csv", validation_episodes)
    summary = []
    for method, first, last in (("warmup", 1, warmup),
                                ("ppo", warmup + 1, episodes),
                                ("pgd_ppo", warmup + 1, episodes)):
        for end in range(first + bin_size - 1, last + 1, bin_size):
            seed_means = [statistics.mean(returns[method, seed][episode]
                           for episode in range(end - bin_size + 1, end + 1))
                          for seed in SEEDS]
            mean, lower, upper = interval(seed_means)
            summary.append(dict(kind="discounted_return", method=method,
                                episode=end, n_seeds=len(SEEDS), mean=mean,
                                ci95_lower=lower, ci95_upper=upper,
                                episodes_per_seed=bin_size))
    for method, checkpoints in (("warmup", [warmup]),
                                ("ppo", validation_episodes),
                                ("pgd_ppo", validation_episodes)):
        for episode in checkpoints:
            mean, lower, upper = interval(
                [scores[method, seed][episode] for seed in SEEDS])
            summary.append(dict(kind="validation_score", method=method,
                                episode=episode, n_seeds=len(SEEDS), mean=mean,
                                ci95_lower=lower, ci95_upper=upper,
                                episodes_per_seed=args["validation_per_scenario"]))
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(path: Path, rows: list[dict], warmup: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    bin_size = next(row["episodes_per_seed"] for row in rows
                    if row["kind"] == "discounted_return")
    for ax, kind, title in zip(
            axes, ("discounted_return", "validation_score"),
            (f"Training discounted return ({bin_size}-episode bins)",
             "ID validation score")):
        warm = sorted((row for row in rows if row["kind"] == kind
                       and row["method"] == "warmup"),
                      key=lambda row: row["episode"])
        if kind == "discounted_return":
            xs = [row["episode"] for row in warm]
            ax.plot(xs, [row["mean"] for row in warm], color="#80868e",
                    label="Shared warmup", linewidth=1.8)
            ax.fill_between(xs, [row["ci95_lower"] for row in warm],
                            [row["ci95_upper"] for row in warm],
                            color="#80868e", alpha=0.15)
        for method, label, color in zip(TRAIN_METHODS, TRAIN_LABELS, TRAIN_COLORS):
            selected = sorted((row for row in rows if row["kind"] == kind
                               and row["method"] == method),
                              key=lambda row: row["episode"])
            xs = [row["episode"] for row in selected]
            ax.plot(xs, [row["mean"] for row in selected], label=label,
                    color=color, linewidth=2)
            ax.fill_between(xs, [row["ci95_lower"] for row in selected],
                            [row["ci95_upper"] for row in selected],
                            color=color, alpha=0.18)
        ax.axvline(warmup, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", ncol=3)
    axes[-1].set_xlabel("Training episode")
    fig.suptitle("Six-seed PPO (C) vs PGD-PPO (C) training", y=0.99)
    fig.text(0.5, 0.015,
             "Lines: mean across six seeds; shaded bands: 95% Student-t confidence intervals. "
             "The two methods share the warmup model.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metrics(path: Path, rows: list[dict]) -> None:
    by_key = {(row["scenario_id"], row["method"], row["metric"]): row
              for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(19, 11))
    fig.subplots_adjust(top=0.83, bottom=0.13, hspace=0.48, wspace=0.18)
    xs = np.arange(len(SCENARIOS))
    for ax, (metric, title) in zip(axes.flat, METRICS):
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            selected = [by_key[scenario, method, metric] for scenario in SCENARIOS]
            means = np.array([row["mean"] for row in selected])
            lower = np.array([row["ci95_lower"] for row in selected])
            upper = np.array([row["ci95_upper"] for row in selected])
            ax.errorbar(xs + (index - 1.5) * 0.19, means,
                        yerr=np.vstack((means - lower, upper - means)),
                        fmt="o", markersize=3.5, color=color, ecolor=color,
                        capsize=2, elinewidth=1, label=label)
        ax.axvline(4.5, color="#444444", linewidth=0.9, alpha=0.5)
        ax.set_title(title)
        ax.set_xticks(xs, SCENARIOS, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        if metric in ("task_completion_rate", "sla_violation_rate"):
            ax.set_ylim(-0.04, 1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Six-seed paired ID/OOD test metrics: mean and 95% CI", y=0.995)
    fig.text(0.5, 0.025,
             "Intervals use the six per-seed episode means (Student-t, df=5). "
             "Unfinished tasks count as the simulation horizon in completion time.",
             ha="center", fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(root: Path) -> None:
    output = root / "figures"
    if not output.is_dir():
        raise FileNotFoundError(f"existing figure directory is missing: {output}")
    files = [output / name for name in (
        "training_curves_95ci.csv", "training_curves_95ci.png",
        "four_metric_95ci.csv", "four_metric_95ci.png",
        "SIX_SEED_INTERVAL_PLOTS_COMPLETE.txt")]
    existing = [str(path) for path in files if path.exists()]
    if existing:
        raise FileExistsError(f"interval output already exists: {existing}")
    grouped, args = collect(root)
    training = training_summary(root, args)
    metrics = []
    for row in summarize(grouped):
        half_width = T_95_DF5 * row["seed_sd"] / math.sqrt(len(SEEDS))
        metrics.append(row | {"ci95_lower": row["mean"] - half_width,
                              "ci95_upper": row["mean"] + half_width})
    write_csv(files[0], training)
    plot_training(files[1], training, args["warmup"])
    write_csv(files[2], metrics)
    plot_metrics(files[3], metrics)
    files[4].write_text("seeds=6\ninterval=Student-t 95% CI, df=5\n",
                        encoding="utf-8")
    print(f"SIX_SEED_INTERVAL_PLOTS_COMPLETE {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir)
