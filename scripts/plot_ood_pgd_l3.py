"""Plot separate four-metric OOD L3 comparisons from six paired test seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ood_pgd_c_six import COLORS, FAMILIES, LABELS, METHODS, METRICS, SEEDS
from plot_ood_pgd_c_six_intervals import interval

SCENARIOS = tuple(f"{family}__ood_l3" for family in FAMILIES)


def collect(root: Path) -> tuple[dict, int]:
    grouped = defaultdict(list)
    count = None
    for seed in SEEDS:
        run = root / f"main_seed{seed}"
        if not (run / "L3_TEST_COMPLETE.txt").is_file():
            raise RuntimeError(f"incomplete L3 test: {run}")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        local_count = config["episodes_per_scenario_per_policy"]
        if count is None:
            count = local_count
        elif count != local_count:
            raise ValueError("test episode counts differ across seeds")
        with (run / "episodes.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            scenario, method = row["scenario_id"], row["policy"]
            if (scenario not in SCENARIOS or method not in METHODS
                    or row["split"] != "ood" or row["severity"] != "l3"
                    or int(row["train_seed"]) != seed):
                raise ValueError(f"unexpected test row in {run}: {scenario}/{method}")
            test_seed = int(row["seed"])
            if not seed * 1_000_000 + 900_000 <= test_seed < (
                    seed * 1_000_000 + 900_000 + count):
                raise ValueError(f"test seed outside fresh L3 range: {run}")
            grouped[scenario, method, seed].append(row)
    expected = {(scenario, method, seed) for scenario in SCENARIOS
                for method in METHODS for seed in SEEDS}
    if set(grouped) != expected:
        raise ValueError("L3 scenario/method/seed coverage is incomplete")
    for key, rows in grouped.items():
        if len(rows) != count or len({row["seed"] for row in rows}) != count:
            raise ValueError(f"missing or duplicate L3 test episodes: {key}")
    return grouped, count


def summarize(grouped: dict, count: int) -> list[dict]:
    result = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for metric, _ in METRICS:
                seed_means = []
                for seed in SEEDS:
                    values = [float(row[metric])
                              for row in grouped[scenario, method, seed]]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"nonfinite metric: {scenario}/{method}/{metric}")
                    seed_means.append(statistics.mean(values))
                mean, lower, upper = interval(seed_means)
                result.append({
                    "scenario_id": scenario, "method": method, "metric": metric,
                    "n_seeds": len(SEEDS), "episodes_per_seed": count,
                    "mean": mean, "seed_sd": statistics.stdev(seed_means),
                    "ci95_lower": lower, "ci95_upper": upper,
                })
    return result


def plot(root: Path) -> None:
    grouped, count = collect(root)
    summary = summarize(grouped, count)
    output = root / "figures"
    output.mkdir(exist_ok=False)
    with (output / "l3_four_metric_summary.csv").open(
            "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    by_key = {(row["scenario_id"], row["method"], row["metric"]): row
              for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(top=0.82, bottom=0.14, hspace=0.35, wspace=0.22)
    xs = np.arange(len(SCENARIOS))
    width = 0.19
    for ax, (metric, title) in zip(axes.flat, METRICS):
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            selected = [by_key[scenario, method, metric] for scenario in SCENARIOS]
            means = np.array([row["mean"] for row in selected])
            lows = np.array([row["ci95_lower"] for row in selected])
            highs = np.array([row["ci95_upper"] for row in selected])
            ax.bar(xs + (index - 1.5) * width, means, width, color=color,
                   label=label, yerr=np.vstack((means - lows, highs - means)),
                   capsize=2.5, error_kw={"elinewidth": 1})
        ax.set_title(title)
        ax.set_xticks(xs, FAMILIES)
        ax.grid(axis="y", alpha=0.2)
        if metric in ("task_completion_rate", "sla_violation_rate"):
            ax.set_ylim(0, 1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("OOD L3: selected PPO (C), PGD-PPO (C), and baselines", y=0.995)
    fig.text(0.5, 0.035,
             f"{len(SEEDS)} training seeds; {count} fresh paired test episodes per seed and family. "
             "Bars: seed means; whiskers: between-seed 95% Student-t CIs. "
             "Unfinished tasks count as the simulation horizon in completion time.",
             ha="center", fontsize=9)
    image = output / "l3_four_metric_comparison.png"
    fig.savefig(image, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (output / "L3_PLOT_COMPLETE.txt").write_text(
        f"seeds={len(SEEDS)}\nepisodes_per_seed={count}\nfigure={image.name}\n",
        encoding="utf-8")
    print(f"L3_PLOT_COMPLETE {image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)
