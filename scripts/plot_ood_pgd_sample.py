"""Plot sampled-action OOD results and paired differences from argmax testing.

Usage: python scripts/plot_ood_pgd_sample.py \
    --sample-root runs/ood_pgd_short_aux6_20261013/sample_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_ood_pgd_argmax as reference
from plot_ood_pgd_short import plot_comparison, summarize_ood


TRAINED = ("ppo", "pgd_ppo")
CASE_IDS = [f"{family}__ood_{level}" for family in reference.FAMILIES
            for level in reference.LEVELS]
ID_CASE_IDS = list(reference.FAMILIES)
ALL_CASE_IDS = [case for family in reference.FAMILIES
                for case in (family, f"{family}__ood_l1", f"{family}__ood_l2")]


def read_tests(sample_root: Path, argmax_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    sampled, greedy = [], []
    counts_by_split = {}
    for seed in reference.SEEDS:
        sample_run = sample_root / f"main_seed{seed}"
        argmax_run = argmax_root / f"main_seed{seed}"
        for run, episode_path in ((sample_run, sample_run / "episodes.csv"),
                                  (argmax_run, argmax_run / "test" / "episodes.csv")):
            if not (run / "TEST_COMPLETE.txt").is_file() or not episode_path.is_file():
                raise RuntimeError(f"incomplete test result: {run}")
        sample = pd.read_csv(sample_run / "episodes.csv")
        argmax = pd.read_csv(argmax_run / "test" / "episodes.csv")
        if not (sample["train_seed"] == seed).all():
            raise RuntimeError(f"incorrect sample train seed: {seed}")
        if not sample.loc[sample["policy"].isin(TRAINED),
                          "action_selection"].eq("categorical_sample").all():
            raise RuntimeError(f"sampled policy rows are mislabeled: {seed}")
        sampled.append(sample)
        greedy.append(argmax.assign(train_seed=seed))
    sampled = pd.concat(sampled, ignore_index=True)
    greedy = pd.concat(greedy, ignore_index=True)
    for name, frame in (("sample", sampled), ("argmax", greedy)):
        for split, cases in (("id", ID_CASE_IDS), ("ood", CASE_IDS)):
            subset = frame[frame["split"] == split]
            expected = {(case, policy, seed) for case in cases
                        for policy in reference.POLICIES for seed in reference.SEEDS}
            counts = subset.groupby(["scenario_id", "policy", "train_seed"]).size().to_dict()
            if set(counts) != expected or len(set(counts.values())) != 1:
                raise RuntimeError(f"incomplete {name} {split.upper()} conditions")
            count = next(iter(counts.values()))
            if split in counts_by_split and count != counts_by_split[split]:
                raise RuntimeError(f"sample and argmax {split.upper()} test counts differ")
            counts_by_split[split] = count
        for field, _, _, _ in reference.METRICS:
            if frame[field].isna().any():
                raise RuntimeError(f"missing {field} in {name} results")
    return sampled, greedy, counts_by_split["id"], counts_by_split["ood"]


def summarize_all(sampled: pd.DataFrame, id_count: int, ood_count: int) -> pd.DataFrame:
    fields = [item[0] for item in reference.METRICS]
    keys = ["scenario_id", "scenario_family", "severity", "policy"]
    by_seed = sampled.groupby(keys + ["train_seed"], as_index=False)[fields].mean()
    rows = []
    for key, group in by_seed.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row.update(n_seeds=len(group),
                   episodes_per_seed=id_count if row["scenario_id"] in ID_CASE_IDS else ood_count)
        for field in fields:
            row[f"{field}_mean"] = group[field].mean()
            row[f"{field}_seed_sd"] = group[field].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all_comparison(output_dir: Path, summary: pd.DataFrame,
                        id_count: int, ood_count: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(23, 13.5), layout="constrained")
    ticks = np.arange(len(ALL_CASE_IDS), dtype=float)
    tick_labels = [f"{family.title()}\n{level}" for family in reference.FAMILIES
                   for level in ("ID", "L1", "L2")]
    offsets = dict(zip(reference.POLICIES, (-0.27, -0.09, 0.09, 0.27)))
    indexed = summary.set_index(["scenario_id", "policy"])
    for ax, (field, title, unit, percent) in zip(axes.flat, reference.METRICS):
        limits = []
        for policy in reference.POLICIES:
            rows = indexed.loc[[(case, policy) for case in ALL_CASE_IDS]]
            scale = 100 if percent else 1
            means = rows[f"{field}_mean"].to_numpy() * scale
            deviations = rows[f"{field}_seed_sd"].to_numpy() * scale
            ax.errorbar(ticks + offsets[policy], means, yerr=deviations,
                        color=reference.COLORS[policy], marker=reference.MARKERS[policy],
                        markersize=5.5, capsize=2.3, elinewidth=1.1,
                        linestyle="none", markeredgecolor="white", markeredgewidth=0.4)
            limits.extend(means - deviations)
            limits.extend(means + deviations)
        for index in range(1, len(reference.FAMILIES)):
            ax.axvline(index * 3 - 0.5, color="#dce2e8", linewidth=1)
        ax.set_xticks(ticks, tick_labels, fontsize=8.5)
        ax.set_xlim(-0.65, len(ALL_CASE_IDS) - 0.35)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#e8ebef", linewidth=0.8)
        ax.set_axisbelow(True)
        low, high = min(limits), max(limits)
        padding = max(0.02 * max(1, abs(high)), 0.08 * (high - low))
        ax.set_ylim((max(0, low - padding), min(100, high + padding)) if percent
                    else (low - padding, high + padding))
    labels = reference.LABELS | {"ppo": "PPO (sample)",
                                 "pgd_ppo": "PGD-PPO (sample)"}
    handles = [plt.Line2D([], [], color=reference.COLORS[policy],
                          marker=reference.MARKERS[policy], linestyle="none",
                          markersize=8, label=labels[policy])
               for policy in reference.POLICIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Four policies across five ID and ten OOD conditions",
                 fontsize=17, fontweight="bold")
    fig.text(0.5, -0.065,
             f"Points: mean; whiskers: SD across three training seeds. "
             f"Each seed has {id_count} ID and {ood_count} OOD episodes per condition.",
             ha="center", fontsize=9, color="#4b5563")
    for suffix in ("png", "svg"):
        output = output_dir / f"id_ood_sample_comparison.{suffix}"
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


def paired_differences(sampled: pd.DataFrame, greedy: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario_id", "policy", "train_seed", "seed"]
    fields = [item[0] for item in reference.METRICS]
    sampled = sampled[sampled["policy"].isin(TRAINED)]
    greedy = greedy[greedy["policy"].isin(TRAINED)]
    joined = sampled[keys + fields].merge(greedy[keys + fields], on=keys,
                                          suffixes=("_sample", "_argmax"),
                                          validate="one_to_one")
    if len(joined) != len(sampled) or len(joined) != len(greedy):
        raise RuntimeError("sample and argmax episode seeds do not match")
    for field in fields:
        joined[field] = joined[f"{field}_sample"] - joined[f"{field}_argmax"]
    by_seed = joined.groupby(["scenario_id", "policy", "train_seed"],
                             as_index=False)[fields].mean()
    rows = []
    for (case_id, policy), group in by_seed.groupby(["scenario_id", "policy"]):
        if len(group) != len(reference.SEEDS):
            raise RuntimeError(f"missing training seed for {case_id} {policy}")
        row = {"scenario_id": case_id, "policy": policy,
               "n_training_seeds": len(group)}
        for field in fields:
            row[f"{field}_mean_delta"] = group[field].mean()
            row[f"{field}_seed_sd"] = group[field].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_delta(output_dir: Path, delta: pd.DataFrame, count: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(19, 13.5), layout="constrained")
    ticks = np.arange(len(CASE_IDS), dtype=float)
    labels = [f"{family.title()}\n{level.upper()}" for family in reference.FAMILIES
              for level in reference.LEVELS]
    for ax, (field, title, unit, percent) in zip(axes.flat, reference.METRICS):
        values = []
        for policy, offset in (("ppo", -0.08), ("pgd_ppo", 0.08)):
            rows = delta.set_index(["scenario_id", "policy"])
            scale = 100 if percent else 1
            means = np.array([rows.loc[(case, policy), f"{field}_mean_delta"]
                              for case in CASE_IDS]) * scale
            deviations = np.array([rows.loc[(case, policy), f"{field}_seed_sd"]
                                   for case in CASE_IDS]) * scale
            ax.errorbar(ticks + offset, means, yerr=deviations,
                        color=reference.COLORS[policy], marker=reference.MARKERS[policy],
                        markersize=5.5, capsize=2.3, linestyle="none",
                        label=policy)
            values.extend(means - deviations)
            values.extend(means + deviations)
        ax.axhline(0, color="#6b7280", linewidth=1)
        for index in range(1, len(reference.FAMILIES)):
            ax.axvline(index * 2 - 0.5, color="#dce2e8", linewidth=1)
        ax.set_xticks(ticks, labels, fontsize=8.5)
        ax.set_xlim(-0.65, len(CASE_IDS) - 0.35)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("percentage points" if percent else unit)
        ax.grid(axis="y", color="#e8ebef", linewidth=0.8)
        ax.set_axisbelow(True)
        low, high = min(values), max(values)
        padding = max(0.02 * max(1, abs(low), abs(high)), 0.08 * (high - low))
        ax.set_ylim(min(0, low) - padding, max(0, high) + padding)
    handles = [plt.Line2D([], [], color=reference.COLORS[policy],
                          marker=reference.MARKERS[policy], linestyle="none",
                          markersize=8, label=reference.LABELS[policy].split(" (")[0])
               for policy in TRAINED]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Sampled action minus argmax across 10 OOD conditions",
                 fontsize=17, fontweight="bold")
    fig.text(0.5, -0.065,
             f"Paired by test seed. Points: mean over three training seeds; "
             f"whiskers: seed SD. Each seed has {count} episodes per condition.",
             ha="center", fontsize=9, color="#4b5563")
    for suffix in ("png", "svg"):
        output = output_dir / f"sample_minus_argmax.{suffix}"
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path,
                        default=Path("runs/ood_pgd_short_aux6_20261013/sample_test"))
    parser.add_argument("--argmax-root", type=Path,
                        help="training run directory; defaults to sample-root parent")
    args = parser.parse_args()
    sample_root = args.sample_root.resolve()
    argmax_root = (args.argmax_root or sample_root.parent).resolve()
    output_dir = sample_root / "plots"
    outputs = [output_dir / "ood_sample_summary.csv",
               output_dir / "sample_minus_argmax_summary.csv",
               output_dir / "id_ood_sample_summary.csv"]
    outputs += [output_dir / f"{stem}.{suffix}"
                for stem in ("ood_sample_comparison", "sample_minus_argmax",
                             "id_ood_sample_comparison")
                for suffix in ("png", "svg")]
    for stem in ("ood_sample_comparison", "sample_minus_argmax",
                 "id_ood_sample_comparison"):
        figures = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "svg")]
        if any(path.exists() for path in figures) and not all(path.is_file() for path in figures):
            parser.error("incomplete existing figures: " + ", ".join(map(str, figures)))
    sampled, greedy, id_count, ood_count = read_tests(sample_root, argmax_root)
    summary = summarize_ood(sampled[sampled["split"] == "ood"], ood_count)
    delta = paired_differences(sampled[sampled["split"] == "ood"],
                               greedy[greedy["split"] == "ood"])
    all_summary = summarize_all(sampled, id_count, ood_count)
    if len(delta) != len(CASE_IDS) * len(TRAINED):
        raise RuntimeError("incomplete sampled-versus-argmax summary")
    if len(all_summary) != len(ALL_CASE_IDS) * len(reference.POLICIES):
        raise RuntimeError("incomplete ID/OOD summary")
    if output_dir.exists():
        if not all(path.is_file() for path in outputs[:2]):
            parser.error(f"incomplete existing summary files: {output_dir}")
        pd.testing.assert_frame_equal(pd.read_csv(outputs[0]), summary,
                                      check_exact=False, rtol=1e-12)
        pd.testing.assert_frame_equal(pd.read_csv(outputs[1]), delta,
                                      check_exact=False, rtol=1e-12)
    else:
        output_dir.mkdir()
        summary.to_csv(outputs[0], index=False, encoding="utf-8-sig")
        delta.to_csv(outputs[1], index=False, encoding="utf-8-sig")
    if outputs[2].exists():
        pd.testing.assert_frame_equal(pd.read_csv(outputs[2]), all_summary,
                                      check_exact=False, rtol=1e-12)
    else:
        all_summary.to_csv(outputs[2], index=False, encoding="utf-8-sig")
    for output in outputs[:3]:
        print(output)
    labels = reference.LABELS | {"ppo": "PPO (sample)",
                                 "pgd_ppo": "PGD-PPO (sample)"}
    if not (output_dir / "ood_sample_comparison.png").exists():
        plot_comparison(output_dir, summary, ood_count, labels=labels,
                        title="Four policies with sampled PPO actions across 10 OOD conditions",
                        output_stem="ood_sample_comparison")
    if not (output_dir / "sample_minus_argmax.png").exists():
        plot_delta(output_dir, delta, ood_count)
    if not (output_dir / "id_ood_sample_comparison.png").exists():
        plot_all_comparison(output_dir, all_summary, id_count, ood_count)


if __name__ == "__main__":
    main()
