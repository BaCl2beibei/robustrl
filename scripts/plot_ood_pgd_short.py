"""Plot OOD comparison and learning curves for the three-seed short run.

Usage: python scripts/plot_ood_pgd_short.py --run-dir runs/ood_pgd_short_20261013
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

import plot_ood_pgd_argmax as reference


def load_data(root: Path):
    tests, training, validation, selections = [], [], [], {}
    schedule = None
    for seed in reference.SEEDS:
        run = root / f"main_seed{seed}"
        for marker in ("TRAINING_COMPLETE.txt", "TEST_COMPLETE.txt"):
            if not (run / marker).is_file():
                raise RuntimeError(f"missing completion marker: {run / marker}")
        args = json.loads((run / "config.json").read_text(encoding="utf-8"))["args"]
        current = (args["episodes"], args["warmup"], args["validation_interval"],
                   args["id_episodes"], args["ood_episodes"])
        if schedule is None:
            schedule = current
        elif current != schedule:
            raise RuntimeError("training or test schedules differ across seeds")
        tests.append(pd.read_csv(run / "test" / "episodes.csv").assign(train_seed=seed))
        training.append(pd.read_csv(run / "training.csv").assign(train_seed=seed))
        validation.append(pd.read_csv(run / "validation.csv").assign(train_seed=seed))
        for method in ("ppo", "pgd_ppo"):
            selections[seed, method] = json.loads(
                (run / method / "selected.json").read_text(encoding="utf-8"))

    episodes, warmup, interval, id_count, ood_count = schedule
    if (episodes <= warmup or warmup < 1 or interval < 1
            or (episodes - warmup) % interval):
        raise RuntimeError("invalid episode and validation schedule")
    test = pd.concat(tests, ignore_index=True)
    expected = {
        (f"{family}__ood_{level}", policy, seed)
        for family in reference.FAMILIES for level in reference.LEVELS
        for policy in reference.POLICIES for seed in reference.SEEDS
    }
    ood = test[test["split"] == "ood"].copy()
    counts = ood.groupby(["scenario_id", "policy", "train_seed"]).size().to_dict()
    if set(counts) != expected or any(count != ood_count for count in counts.values()):
        raise RuntimeError("OOD test rows are incomplete")
    expected_total = len(reference.SEEDS) * len(reference.POLICIES) * (
        len(reference.FAMILIES) * id_count + len(reference.FAMILIES)
        * len(reference.LEVELS) * ood_count)
    if len(test) != expected_total:
        raise RuntimeError(f"expected {expected_total} test rows; got {len(test)}")
    for field, _, _, _ in reference.METRICS:
        if ood[field].isna().any():
            raise RuntimeError(f"missing OOD values for {field}")
    return (ood, pd.concat(training, ignore_index=True),
            pd.concat(validation, ignore_index=True), selections,
            (episodes, warmup, interval, ood_count))


def summarize_ood(ood: pd.DataFrame, ood_count: int) -> pd.DataFrame:
    fields = [item[0] for item in reference.METRICS]
    keys = ["scenario_id", "scenario_family", "severity", "policy"]
    by_seed = ood.groupby(keys + ["train_seed"], as_index=False)[fields].mean()
    rows = []
    for key, group in by_seed.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row.update(n_seeds=len(group), episodes_per_seed=ood_count)
        for field in fields:
            row[f"{field}_mean"] = group[field].mean()
            row[f"{field}_seed_sd"] = group[field].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary) != len(reference.FAMILIES) * len(reference.LEVELS) * len(reference.POLICIES):
        raise RuntimeError("incomplete OOD summary")
    return summary


def plot_comparison(output_dir: Path, summary: pd.DataFrame, ood_count: int,
                    labels: dict[str, str] | None = None,
                    title: str = "PPO and PGD-PPO with greedy actions across 10 OOD conditions",
                    output_stem: str = "ood_argmax_comparison") -> None:
    labels = labels or reference.LABELS
    figure_title = title
    fig, axes = plt.subplots(3, 2, figsize=(19, 13.5), layout="constrained")
    case_ids = [f"{family}__ood_{level}" for family in reference.FAMILIES
                for level in reference.LEVELS]
    ticks = np.arange(len(case_ids), dtype=float)
    tick_labels = [f"{family.title()}\n{level.upper()}" for family in reference.FAMILIES
                   for level in reference.LEVELS]
    offsets = dict(zip(reference.POLICIES, (-0.27, -0.09, 0.09, 0.27)))
    for ax, (field, metric_title, unit, percent) in zip(axes.flat, reference.METRICS):
        limits = []
        for policy in reference.POLICIES:
            means, deviations, positions = [], [], []
            for index, case_id in enumerate(case_ids):
                row = summary[(summary["scenario_id"] == case_id)
                              & (summary["policy"] == policy)].iloc[0]
                scale = 100 if percent else 1
                mean = row[f"{field}_mean"] * scale
                deviation = row[f"{field}_seed_sd"] * scale
                means.append(mean)
                deviations.append(deviation)
                positions.append(ticks[index] + offsets[policy])
                limits.extend((mean - deviation, mean + deviation))
            ax.errorbar(positions, means, yerr=deviations,
                        color=reference.COLORS[policy], marker=reference.MARKERS[policy],
                        markersize=5.5, capsize=2.3, elinewidth=1.1,
                        linestyle="none", markeredgecolor="white", markeredgewidth=0.4)
        for index in range(1, len(reference.FAMILIES)):
            ax.axvline(index * 2 - 0.5, color="#dce2e8", linewidth=1)
        ax.set_xticks(ticks, tick_labels, fontsize=8.5)
        ax.set_xlim(-0.65, len(case_ids) - 0.35)
        ax.set_title(metric_title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#e8ebef", linewidth=0.8)
        ax.set_axisbelow(True)
        low, high = min(limits), max(limits)
        padding = max(0.02 * max(1, abs(high)), 0.08 * (high - low))
        ax.set_ylim((max(0, low - padding), min(100, high + padding)) if percent
                    else (low - padding, high + padding))
    handles = [plt.Line2D([], [], color=reference.COLORS[policy],
                          marker=reference.MARKERS[policy], linestyle="none",
                          markersize=8, label=labels[policy])
               for policy in reference.POLICIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle(figure_title, fontsize=17, fontweight="bold")
    fig.text(0.5, -0.065,
             f"Points: mean; whiskers: SD across three training seeds. "
             f"Each seed has {ood_count} paired episodes per condition and policy.",
             ha="center", fontsize=9, color="#4b5563")
    for suffix in ("png", "svg"):
        output = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


def plot_training(root: Path, output_dir: Path, train: pd.DataFrame, valid: pd.DataFrame,
                  selections: dict, episodes: int, warmup: int, interval: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             layout="constrained")
    window = max(20, episodes // 40)
    stride = max(1, episodes // 200)
    x = np.arange(stride, episodes + 1, stride)
    x = np.unique(np.r_[x, episodes])
    validation_x = np.arange(warmup, episodes + 1, interval)

    for method in ("ppo", "pgd_ppo"):
        curves, scores_by_seed = [], []
        for seed in reference.SEEDS:
            seed_rows = train[train["train_seed"] == seed]
            joined = pd.concat((seed_rows[seed_rows["method"] == "warmup"],
                                seed_rows[seed_rows["method"] == method]))
            joined = joined.sort_values("episode")
            if len(joined) != episodes or not np.array_equal(
                    joined["episode"].to_numpy(), np.arange(1, episodes + 1)):
                raise RuntimeError(f"incomplete training: seed={seed} method={method}")
            rolling = joined["discounted_return"].rolling(
                window=window, min_periods=1).mean().to_numpy()
            curves.append(rolling[x - 1])

            warmup_score = json.loads((root / f"main_seed{seed}" / "warmup_score.json")
                                      .read_text(encoding="utf-8"))["score"]
            branch = valid[(valid["train_seed"] == seed)
                           & (valid["method"] == method)].sort_values("episode")
            if not np.array_equal(branch["episode"].to_numpy(), validation_x[1:]):
                raise RuntimeError(f"incomplete validation: seed={seed} method={method}")
            scores = np.r_[warmup_score, branch["score"].to_numpy()]
            scores_by_seed.append(scores)
            selected = Path(selections[seed, method]["checkpoint"].replace("\\", "/"))
            chosen_episode = (warmup if selected.name == "warmup.pt"
                              else int(selected.stem.split("_")[-1]))
            match = np.flatnonzero(validation_x == chosen_episode)
            if len(match) != 1:
                raise RuntimeError(f"invalid selected checkpoint: seed={seed} method={method}")
            axes[1].scatter(chosen_episode, scores[match[0]], marker="*", s=75,
                            color=reference.COLORS[method], edgecolor="white",
                            linewidth=0.5, zorder=5)

        curve_array = np.stack(curves)
        curve_mean = curve_array.mean(axis=0)
        curve_sd = curve_array.std(axis=0, ddof=1)
        axes[0].plot(x, curve_mean, color=reference.COLORS[method], linewidth=2,
                     label=reference.LABELS[method])
        axes[0].fill_between(x, curve_mean - curve_sd, curve_mean + curve_sd,
                             color=reference.COLORS[method], alpha=0.16)

        score_array = np.stack(scores_by_seed)
        score_mean = score_array.mean(axis=0)
        score_sd = score_array.std(axis=0, ddof=1)
        axes[1].plot(validation_x, score_mean, color=reference.COLORS[method],
                     marker=reference.MARKERS[method], markersize=4.5,
                     linewidth=2, label=reference.LABELS[method])
        axes[1].fill_between(validation_x, score_mean - score_sd,
                             score_mean + score_sd,
                             color=reference.COLORS[method], alpha=0.16)

    axes[0].axvline(warmup, color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title(f"Training discounted return ({window}-episode rolling mean)",
                      loc="left", fontweight="bold")
    axes[0].set_ylabel("Discounted return")
    axes[1].set_title("Independent ID validation score (stars: selected checkpoints)",
                      loc="left", fontweight="bold")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Composite score")
    axes[1].set_xlim(0, episodes)
    for ax in axes:
        ax.grid(color="#e8ebef", linewidth=0.8)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("PPO and PGD-PPO learning curves", fontsize=16, fontweight="bold")
    fig.text(0.5, -0.045,
             f"Line: mean over three seeds; band: ±1 seed SD. "
             f"The first {warmup:,} episodes share the same warmup model.",
             ha="center", fontsize=9, color="#4b5563")
    for suffix in ("png", "svg"):
        output = output_dir / f"training_curves.{suffix}"
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/ood_pgd_short_20261013"))
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output_dir = root / "plots"
    outputs = [output_dir / "ood_argmax_summary.csv"]
    outputs += [output_dir / f"{stem}.{suffix}"
                for stem in ("ood_argmax_comparison", "training_curves")
                for suffix in ("png", "svg")]
    existing = [path for path in outputs if path.exists()]
    if existing:
        parser.error("output already exists: " + ", ".join(map(str, existing)))
    ood, train, valid, selections, schedule = load_data(root)
    episodes, warmup, interval, ood_count = schedule
    summary = summarize_ood(ood, ood_count)
    output_dir.mkdir(exist_ok=False)
    summary.to_csv(outputs[0], index=False, encoding="utf-8-sig")
    print(outputs[0])
    plot_comparison(output_dir, summary, ood_count)
    plot_training(root, output_dir, train, valid, selections, episodes, warmup, interval)


if __name__ == "__main__":
    main()
