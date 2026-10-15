"""Plot the 10 OOD conditions from a greedy-action evaluation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = ("normal", "failure", "network", "load", "mixed")
LEVELS = ("l1", "l2")
SEEDS = (1, 2, 3)
POLICIES = ("no_migration", "time_min", "ppo", "pgd_ppo")
LABELS = {"no_migration": "No-Migration", "time_min": "Time-Min",
          "ppo": "PPO (argmax)", "pgd_ppo": "PGD-PPO (argmax)"}
COLORS = {"no_migration": "#707988", "time_min": "#e68624",
          "ppo": "#2369a8", "pgd_ppo": "#7d4bb0"}
MARKERS = {"no_migration": "o", "time_min": "s", "ppo": "^", "pgd_ppo": "D"}
METRICS = (
    ("task_completion_rate", "Task completion", "%", True),
    ("task_service_availability", "Task service availability", "%", True),
    ("sla_violation_rate", "SLA violation", "%", True),
    ("mean_completion_seconds_horizon_capped", "Completion time (horizon capped)", "s", False),
    ("migration_count", "Migrations per episode", "count", False),
    ("total_reward", "Total reward", "reward", False),
)


def read_and_summarize(root: Path, episodes_per_condition: int) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        run = root / f"main_seed{seed}"
        marker = run / "TEST_COMPLETE.txt"
        if not marker.is_file():
            raise RuntimeError(f"worker is incomplete: {run}")
        frames.append(pd.read_csv(run / "episodes.csv"))
    data = pd.concat(frames, ignore_index=True)
    expected_cases = {f"{family}__ood_{level}" for family in FAMILIES
                      for level in LEVELS}
    expected = {(case, policy, seed) for case in expected_cases
                for policy in POLICIES for seed in SEEDS}
    counts = data.groupby(["scenario_id", "policy", "train_seed"]).size().to_dict()
    if set(counts) != expected or any(
            count != episodes_per_condition for count in counts.values()):
        raise RuntimeError("episode counts do not cover all OOD/policy/seed conditions")
    if len(data) != len(expected) * episodes_per_condition:
        raise RuntimeError("unexpected total number of OOD evaluation rows")
    for field, _, _, _ in METRICS:
        if data[field].isna().any():
            raise RuntimeError(f"missing values in metric {field}")

    fields = [metric[0] for metric in METRICS]
    seed_means = data.groupby(
        ["scenario_id", "scenario_family", "severity", "policy", "train_seed"],
        as_index=False,
    )[fields].mean()
    summary_rows = []
    for key, group in seed_means.groupby(
            ["scenario_id", "scenario_family", "severity", "policy"], sort=False):
        row = dict(zip(("scenario_id", "scenario_family", "severity", "policy"), key))
        row.update(n_seeds=len(group), episodes_per_seed=episodes_per_condition)
        for field in fields:
            row[f"{field}_mean"] = group[field].mean()
            row[f"{field}_seed_sd"] = group[field].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if len(summary) != 40:
        raise RuntimeError(f"expected 40 OOD summaries; got {len(summary)}")
    return summary


def plot(root: Path, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(19, 13.5), layout="constrained")
    case_ids = [f"{family}__ood_{level}" for family in FAMILIES for level in LEVELS]
    ticks = np.arange(len(case_ids), dtype=float)
    tick_labels = [f"{family.title()}\n{level.upper()}"
                   for family in FAMILIES for level in LEVELS]
    offsets = dict(zip(POLICIES, (-0.27, -0.09, 0.09, 0.27)))

    for ax, (field, title, unit, percent) in zip(axes.flat, METRICS):
        values_for_limits = []
        for policy in POLICIES:
            means, deviations, positions = [], [], []
            for index, case_id in enumerate(case_ids):
                row = summary[(summary["scenario_id"] == case_id)
                              & (summary["policy"] == policy)].iloc[0]
                scale = 100 if percent else 1
                mean = row[f"{field}_mean"] * scale
                deviation = row[f"{field}_seed_sd"] * scale
                x = ticks[index] + offsets[policy]
                means.append(mean)
                deviations.append(deviation)
                positions.append(x)
                values_for_limits.extend((mean - deviation, mean + deviation))
            ax.errorbar(positions, means, yerr=deviations, color=COLORS[policy],
                        marker=MARKERS[policy], markersize=5.5, capsize=2.3,
                        elinewidth=1.1, linestyle="none", markeredgecolor="white",
                        markeredgewidth=0.4, label=LABELS[policy])

        for index in range(1, len(FAMILIES)):
            ax.axvline(index * 2 - 0.5, color="#dce2e8", linewidth=1)
        ax.set_xticks(ticks, tick_labels, fontsize=8.5)
        ax.set_xlim(-0.65, len(case_ids) - 0.35)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#e8ebef", linewidth=0.8)
        ax.set_axisbelow(True)
        low, high = min(values_for_limits), max(values_for_limits)
        padding = max(0.02 * max(1, abs(high)), 0.08 * (high - low))
        if percent:
            ax.set_ylim(max(0, low - padding), min(100, high + padding))
        else:
            ax.set_ylim(low - padding, high + padding)

    handles = [plt.Line2D([], [], color=COLORS[policy], marker=MARKERS[policy],
                          linestyle="none", markersize=8, label=LABELS[policy])
               for policy in POLICIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=11)
    fig.suptitle("PPO and PGD-PPO with greedy actions across 10 OOD conditions",
                 fontsize=17, fontweight="bold")
    fig.text(0.5, 0.025,
             "Points: mean; whiskers: SD across three training seeds. "
             "Each seed has 100 paired episodes per condition and policy.",
             ha="center", fontsize=9, color="#4b5563")
    for suffix in ("png", "svg"):
        output = root / f"ood_argmax_comparison.{suffix}"
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/ood_pgd_20261012/argmax_test"))
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    root = args.run_dir.resolve()
    summary_path = root / "summary.csv"
    if summary_path.exists():
        parser.error(f"summary already exists: {summary_path}")
    summary = read_and_summarize(root, args.episodes)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary_path)
    plot(root, summary)


if __name__ == "__main__":
    main()
