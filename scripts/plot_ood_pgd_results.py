"""Plot the completed three-seed PPO/PGD-PPO experiment.

Usage: python scripts/plot_ood_pgd_results.py --run-dir runs/ood_pgd_20261012
Outputs are written directly to the run directory. Existing outputs are kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = ("normal", "failure", "network", "load", "mixed")
LEVELS = ("l1", "l2")
POLICIES = ("no_migration", "time_min", "ppo", "pgd_ppo")
LABELS = {"no_migration": "No-Migration", "time_min": "Time-Min",
          "ppo": "PPO", "pgd_ppo": "PGD-PPO"}
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


def load_data(root: Path):
    if "QUEUE_COMPLETE" not in (root / "six_parallel_queue.log").read_text(
            encoding="utf-8"):
        raise RuntimeError("six-process experiment is not complete")
    tests, training, validation, selections = [], [], [], {}
    for seed in (1, 2, 3):
        run = root / f"main_seed{seed}"
        for marker in ("TRAINING_COMPLETE.txt", "TEST_COMPLETE.txt"):
            if not (run / marker).is_file():
                raise RuntimeError(f"missing {run / marker}")
        tests.append(pd.read_csv(run / "test" / "episodes.csv").assign(train_seed=seed))
        training.append(pd.read_csv(run / "training.csv").assign(train_seed=seed))
        validation.append(pd.read_csv(run / "validation.csv").assign(train_seed=seed))
        for method in ("ppo", "pgd_ppo"):
            selections[seed, method] = json.loads(
                (run / method / "selected.json").read_text(encoding="utf-8"))
    test = pd.concat(tests, ignore_index=True)
    train = pd.concat(training, ignore_index=True)
    valid = pd.concat(validation, ignore_index=True)
    ood = test[test["split"] == "ood"].copy()
    expected = {(f"{family}__ood_{level}", policy, seed)
                for family in FAMILIES for level in LEVELS
                for policy in POLICIES for seed in (1, 2, 3)}
    counts = ood.groupby(["scenario_id", "policy", "train_seed"]).size().to_dict()
    if set(counts) != expected or any(count != 100 for count in counts.values()):
        raise RuntimeError("expected exactly 100 OOD episodes for every condition/policy/seed")
    if len(ood) != 12_000:
        raise RuntimeError(f"expected 12000 OOD rows; got {len(ood)}")
    for field, _, _, _ in METRICS:
        if ood[field].isna().any():
            raise RuntimeError(f"missing OOD values for {field}")
    return ood, train, valid, selections


def summarize_ood(ood: pd.DataFrame) -> pd.DataFrame:
    fields = [metric[0] for metric in METRICS]
    by_seed = (ood.groupby(["scenario_id", "scenario_family", "severity",
                            "policy", "train_seed"], as_index=False)[fields].mean())
    rows = []
    for key, group in by_seed.groupby(["scenario_id", "scenario_family",
                                       "severity", "policy"], sort=False):
        row = dict(zip(("scenario_id", "scenario_family", "severity", "policy"), key))
        row.update(n_seeds=3, episodes_per_seed=100)
        for field in fields:
            row[f"{field}_mean"] = group[field].mean()
            row[f"{field}_seed_sd"] = group[field].std(ddof=1)
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 40:
        raise RuntimeError("expected 40 OOD condition/policy summaries")
    return result


def save_figure(fig, root: Path, stem: str):
    for suffix in ("png", "svg"):
        path = root / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(f"figure already exists: {path}")
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white")
        print(path)
    plt.close(fig)


def ood_overview(root: Path, summary: pd.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(19, 13.5), layout="constrained")
    positions = {(family, level): index * 2.65 + level_index * 0.9
                 for index, family in enumerate(FAMILIES)
                 for level_index, level in enumerate(LEVELS)}
    ticks = [positions[family, level] for family in FAMILIES for level in LEVELS]
    tick_labels = [f"{family.title()}\n{level.upper()}"
                   for family in FAMILIES for level in LEVELS]
    offsets = dict(zip(POLICIES, (-0.24, -0.08, 0.08, 0.24)))
    for ax, (field, title, unit, percent) in zip(axes.flat, METRICS):
        plotted = []
        for policy in POLICIES:
            points = []
            for family in FAMILIES:
                for level in LEVELS:
                    scenario_id = f"{family}__ood_{level}"
                    row = summary[(summary["scenario_id"] == scenario_id)
                                  & (summary["policy"] == policy)].iloc[0]
                    scale = 100 if percent else 1
                    x = positions[family, level] + offsets[policy]
                    y = row[f"{field}_mean"] * scale
                    sd = row[f"{field}_seed_sd"] * scale
                    points.append((x, y, sd))
                    ax.errorbar(x, y, yerr=sd, color=COLORS[policy],
                                marker=MARKERS[policy], markersize=5.5,
                                capsize=2.3, elinewidth=1.1, linestyle="none",
                                markeredgecolor="white", markeredgewidth=0.4)
            plotted.extend(points)
        for index in range(4):
            ax.axvline((ticks[index * 2 + 1] + ticks[index * 2 + 2]) / 2,
                       color="#dce2e8", linewidth=1)
        ax.set_xticks(ticks, tick_labels, fontsize=8.5)
        ax.set_xlim(min(ticks) - 0.65, max(ticks) + 0.65)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#e8ebef", linewidth=0.8)
        ax.set_axisbelow(True)
        low = min(y - sd for _, y, sd in plotted)
        high = max(y + sd for _, y, sd in plotted)
        padding = max(0.02 * max(1, abs(high)), 0.08 * (high - low))
        if percent:
            ax.set_ylim(max(0, low - padding), min(100, high + padding))
        else:
            ax.set_ylim(low - padding, high + padding)
    handles = [plt.Line2D([], [], color=COLORS[p], marker=MARKERS[p],
                          linestyle="none", markersize=8, label=LABELS[p])
               for p in POLICIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=11)
    fig.suptitle("Four-policy comparison across 10 OOD conditions", fontsize=17,
                 fontweight="bold")
    fig.text(0.5, 0.025,
             "Points: mean; whiskers: SD of three experiment-seed means. "
             "Each seed has 100 paired test episodes per condition.",
             ha="center", fontsize=9, color="#4b5563")
    save_figure(fig, root, "ood_four_policy_comparison")


def training_curves(root: Path, train: pd.DataFrame, valid: pd.DataFrame,
                    selections: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             layout="constrained")
    window = 500
    x = np.arange(100, 20_001, 100)
    for method in ("ppo", "pgd_ppo"):
        curves = []
        for seed in (1, 2, 3):
            seed_rows = train[train["train_seed"] == seed]
            warmup = seed_rows[seed_rows["method"] == "warmup"]
            branch = seed_rows[seed_rows["method"] == method]
            joined = pd.concat((warmup, branch)).sort_values("episode")
            if len(joined) != 20_000 or not np.array_equal(
                    joined["episode"].to_numpy(), np.arange(1, 20_001)):
                raise RuntimeError(f"incomplete training episodes: seed={seed} {method}")
            rolling = joined["discounted_return"].rolling(
                window=window, min_periods=100).mean().to_numpy()
            curves.append(rolling[x - 1])
        array = np.stack(curves)
        mean = array.mean(axis=0)
        sd = array.std(axis=0, ddof=1)
        axes[0].plot(x, mean, color=COLORS[method], linewidth=2,
                     label=LABELS[method])
        axes[0].fill_between(x, mean - sd, mean + sd,
                             color=COLORS[method], alpha=0.16)
    axes[0].axvline(2000, color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title("Training discounted return (500-episode rolling mean)",
                      loc="left", fontweight="bold")
    axes[0].set_ylabel("Discounted return")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(color="#e8ebef", linewidth=0.8)

    for method in ("ppo", "pgd_ppo"):
        series = []
        for seed in (1, 2, 3):
            warmup = json.loads((root / f"main_seed{seed}" / "warmup_score.json")
                                .read_text(encoding="utf-8"))["score"]
            branch = valid[(valid["train_seed"] == seed)
                           & (valid["method"] == method)].sort_values("episode")
            episodes = np.r_[2000, branch["episode"].to_numpy()]
            scores = np.r_[warmup, branch["score"].to_numpy()]
            if len(episodes) != 19 or not np.array_equal(
                    episodes, np.arange(2000, 20_001, 1000)):
                raise RuntimeError(f"incomplete ID validation: seed={seed} {method}")
            series.append(scores)
            selected_name = Path(selections[seed, method]["checkpoint"].replace("\\", "/"))
            selected_episode = 2000 if selected_name.name == "warmup.pt" else int(
                selected_name.stem.split("_")[-1])
            index = (selected_episode - 2000) // 1000
            axes[1].scatter(selected_episode, scores[index], marker="*", s=75,
                            color=COLORS[method], edgecolor="white", linewidth=0.5,
                            zorder=5)
        array = np.stack(series)
        mean = array.mean(axis=0)
        sd = array.std(axis=0, ddof=1)
        xx = np.arange(2000, 20_001, 1000)
        axes[1].plot(xx, mean, marker=MARKERS[method], markersize=4.5,
                     color=COLORS[method], linewidth=2, label=LABELS[method])
        axes[1].fill_between(xx, mean - sd, mean + sd,
                             color=COLORS[method], alpha=0.16)
    axes[1].set_title("Independent ID validation score (stars: selected checkpoints)",
                      loc="left", fontweight="bold")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Composite score")
    axes[1].grid(color="#e8ebef", linewidth=0.8)
    axes[1].set_xlim(0, 20_000)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("PPO and PGD-PPO learning curves", fontsize=16,
                 fontweight="bold")
    fig.text(0.5, 0.015,
             "Line: mean over three seeds; band: ±1 seed SD. "
             "0–2,000 episodes use the same warmup model for both methods.",
             ha="center", fontsize=9, color="#4b5563")
    save_figure(fig, root, "training_curves")


def write_notes(root: Path, summary: pd.DataFrame):
    rates = ("task_completion_rate", "task_service_availability",
             "sla_violation_rate")
    macro = (summary.groupby("policy")[[f"{name}_mean" for name in rates]]
             .mean().reindex(POLICIES) * 100)
    lines = ["# OOD 绘图说明", "",
             "数据来自 `main_seed1`～`main_seed3/test/episodes.csv`；每种 OOD 条件、每种策略、每个种子有 100 个测试回合。",
             "先计算每个种子的均值，再对三个种子取均值；图中的误差线是三个种子均值的标准差，不代表置信区间。",
             "同一测试种子用于四种策略的配对比较。训练曲线为 500 回合滚动均值；验证曲线使用独立 ID 种子，星号表示每个种子的入选 checkpoint。",
             "", "## 10 种 OOD 条件的宏平均", "",
             "| 策略 | 任务完成率 | 服务可用率 | SLA 违约率 |",
             "| --- | ---: | ---: | ---: |"]
    for policy in POLICIES:
        values = macro.loc[policy]
        lines.append(f"| {LABELS[policy]} | {values.iloc[0]:.2f}% | "
                     f"{values.iloc[1]:.2f}% | {values.iloc[2]:.2f}% |")
    lines.extend(["", "宏平均给 10 种条件相同权重；完整的逐条件结果见 `ood_four_policy_summary.csv`。",
                  "这些图展示观测结果，不据此宣称算法差异具有统计显著性。", ""])
    path = root / "plot_notes.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/ood_pgd_20261012"))
    args = parser.parse_args()
    root = args.run_dir.resolve()
    outputs = [root / f"{stem}.{suffix}"
               for stem in ("ood_four_policy_comparison", "training_curves")
               for suffix in ("png", "svg")]
    outputs += [root / "ood_four_policy_summary.csv", root / "plot_notes.md"]
    existing = [path for path in outputs if path.exists()]
    if existing:
        parser.error("output already exists: " + ", ".join(map(str, existing)))
    ood, train, valid, selections = load_data(root)
    summary = summarize_ood(ood)
    summary.to_csv(root / "ood_four_policy_summary.csv", index=False,
                   encoding="utf-8-sig")
    ood_overview(root, summary)
    training_curves(root, train, valid, selections)
    write_notes(root, summary)


if __name__ == "__main__":
    main()
