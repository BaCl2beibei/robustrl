"""Plot C hierarchical PPO against No-Migration and Time-Min."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINES = ("no_migration", "time_min")
BASELINE_LABELS = ("No migration", "Time-Min")
BASELINE_COLORS = ("#787e87", "#ad5982")
C_METHOD = "c_hierarchical_aux"
C_COMPARISON_METRICS = (
    ("task_completion_rate", "Task completion rate"),
    ("mean_completion_seconds_horizon_capped",
     "Mean completion time (unfinished tasks capped at 600 s)"),
    ("task_service_availability", "Task service availability"),
    ("sla_violation_rate", "SLA violation rate"),
    ("migration_count", "Migrations per episode"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_c_vs_baselines(root: Path):
    test_dir = root / "test"
    if not (test_dir / "TEST_COMPLETE.txt").is_file():
        raise RuntimeError("paired test has not completed")
    summary = read_csv(test_dir / "summary.csv")
    methods = BASELINES + (C_METHOD,)
    if not set(methods).issubset({row["method"] for row in summary}):
        # Historical A/B/C runs stored the two baselines in test_five.
        test_dir = root / "test_five"
        if not (test_dir / "FIVE_METHOD_TEST_COMPLETE.txt").is_file():
            raise RuntimeError("C and both baselines are missing from the test")
        summary = read_csv(test_dir / "summary.csv")
    if not set(methods).issubset({row["method"] for row in summary}):
        raise RuntimeError("C and both baselines are missing from the summary")
    labels = BASELINE_LABELS + ("C Hierarchical + aux",)
    colors = BASELINE_COLORS + ("#609b63",)
    scenarios = list(dict.fromkeys(row["scenario_id"] for row in summary))
    by_key = {(row["scenario_id"], row["method"], row["metric"]): row
              for row in summary}
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.55, wspace=0.18)
    xs = np.arange(len(scenarios))
    width = 0.25
    for ax, (metric, title) in zip(axes.flat, C_COMPARISON_METRICS):
        for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = [float(by_key[scenario, method, metric]["mean"])
                      for scenario in scenarios]
            ax.bar(xs + (index - 1) * width, values, width,
                   label=label, color=color)
        ax.set_title(title)
        ax.set_xticks(xs, scenarios, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        if metric in ("task_completion_rate", "task_service_availability",
                      "sla_violation_rate"):
            ax.set_ylim(0, 1)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, _ = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=13)
    legend_ax.text(0.5, 0.28, "Paired ID/OOD test seeds; one PPO training seed",
                   transform=legend_ax.transAxes, ha="center", fontsize=11)
    fig.suptitle("No-Migration, Time-Min and C: paired test comparison", y=0.985)
    out = test_dir / "c_vs_baselines_with_completion.png"
    if out.exists():
        raise FileExistsError(f"plot already exists: {out}")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    marker = test_dir / "C_VS_BASELINES_PLOT_COMPLETE.txt"
    marker.write_text(out.name + "\n", encoding="utf-8")
    print(f"C_VS_BASELINES_PLOT_COMPLETE {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    plot_c_vs_baselines(args.run_dir)
