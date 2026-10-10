"""Train mixed PPO, select on held-out mixed seeds, then run paired OOD tests.

Run with the project's Anaconda Python, for example:
  D:\\Anaconda_envs\\envs\\migrationproject\\python.exe \
    scripts/run_mixed_ppo_earlystop_ood.py --output-dir runs/mixed_v1_new
The output directory must be new. Use --phase resume after an interrupted
training run, or --phase ood after training completed without OOD evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from xml.sax.saxutils import escape

# A file launched from scripts/ must still import the local ftm checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.config import MonitoringConfig, SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import (_episode_row, bootstrap_interval,
                                  build_evaluation_cases)
from ftm.main import run_episode
from ftm.observation import FEATURE_ORDER, FEATURE_SCALES, SCHEMA_VERSION
from ftm.robust_ppo import (FiniteScenarioRobustPPO, PPOConfig,
                            _build_actor_critic)
from ftm.trained_policy import TrainedPolicy, run_trained_episode


BASELINES = ("no_migration", "random", "greedy", "load_based",
             "robust_heuristic")
METRICS = ("task_completion_rate", "task_service_availability",
           "sla_violation_rate", "migration_count", "total_reward")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no records for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _all_training_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "training").glob("episodes_*.csv")):
        rows.extend(_read_csv(path))
    return rows


def _all_validation_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "validation").glob("checkpoint_*.csv")):
        rows.extend(_read_csv(path))
    return rows


def _plot_curve(path: Path, training: list[dict], validation: list[dict],
                selected_episode: int | None, max_episodes: int) -> None:
    """Standalone SVG, so the experiment needs no plotting dependency."""
    train = [(int(row["episode"]), float(row["discounted_return"]))
             for row in training]
    valid = [(int(row["episode"]), float(row["mean_discounted_return"]))
             for row in validation]
    if not train:
        return
    width, height = 1080, 600
    left, right, top, bottom = 80, 30, 65, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [value for _, value in train + valid]
    low, high = min(all_y), max(all_y)
    pad = max(1.0, (high - low) * 0.08)
    low, high = low - pad, high + pad

    def x_pos(x: int) -> float:
        return left + plot_w * x / max_episodes

    def y_pos(y: float) -> float:
        return top + plot_h * (high - y) / (high - low)

    def polyline(points: list[tuple[int, float]], color: str,
                 stroke_width: int) -> str:
        coords = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in points)
        return (f'<polyline fill="none" stroke="{color}" '
                f'stroke-width="{stroke_width}" points="{coords}"/>')

    rolling = []
    for index, (episode, _) in enumerate(train):
        rolling.append((episode, statistics.mean(
            value for _, value in train[max(0, index - 99):index + 1])))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="35" font-size="22" fill="#172033">'
             f'{escape("Mixed PPO training and held-out validation")}</text>']
    for index in range(6):
        value = low + (high - low) * index / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
                     f'y2="{y:.1f}" stroke="#e5eaf0"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="12" fill="#4b5563">{value:.2f}</text>')
    for episode in range(0, max_episodes + 1,
                         max(1, math.ceil(max_episodes / 5))):
        x = x_pos(episode)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" '
                     f'text-anchor="middle" font-size="12" fill="#4b5563">'
                     f'{episode}</text>')
    parts.extend((polyline(train, "#ccd5df", 1),
                  polyline(rolling, "#2563eb", 2),
                  polyline(valid, "#ea580c", 3)))
    for episode, value in valid:
        fill = "#16a34a" if episode == selected_episode else "#ea580c"
        parts.append(f'<circle cx="{x_pos(episode):.1f}" '
                     f'cy="{y_pos(value):.1f}" r="5" fill="{fill}"/>')
    parts.append(f'<text x="{width/2}" y="{height-20}" '
                 'text-anchor="middle" font-size="14">Training episode</text>')
    parts.append(f'<text x="{left}" y="{height-47}" font-size="13" '
                 'fill="#2563eb">Train: rolling 100 episodes</text>')
    parts.append(f'<text x="{left+265}" y="{height-47}" font-size="13" '
                 'fill="#ea580c">Validation: fixed held-out seeds</text>')
    parts.append(f'<text x="{left+585}" y="{height-47}" font-size="13" '
                 'fill="#16a34a">Selected checkpoint</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(parts))


def _init_trainer(config: SimulationConfig, ppo_config: PPOConfig,
                  device: str, seed: int) -> FiniteScenarioRobustPPO:
    import torch

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    trainer = FiniteScenarioRobustPPO(config, ppo_config, ("mixed",), device, seed)
    probe = MigrationEnvironment(config, scenario_by_name("mixed", config))
    state, _ = probe.reset(seed)
    trainer.model = _build_actor_critic(len(state), config.n_nodes + 1).to(device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(),
                                         lr=ppo_config.learning_rate)
    return trainer


def train(root: Path, args: argparse.Namespace, config: SimulationConfig,
          ppo_config: PPOConfig, resume: bool = False) -> dict:
    import torch

    trainer = _init_trainer(config, ppo_config, args.device, args.train_seed)
    best_score = float("-inf")
    patience_score = float("-inf")
    best_episode = None
    stale = 0
    segment = []
    start_episode = 1
    if resume:
        states = sorted((root / "training_state").glob("state_ep*.pt"))
        if not states:
            raise ValueError("no complete checkpoint state is available to resume")
        state = torch.load(states[-1], map_location="cpu", weights_only=False)
        checkpoint_data = torch.load(root / state["checkpoint"],
                                     map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(checkpoint_data["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        best_score = state["best_score"]
        patience_score = state["patience_score"]
        best_episode = state["best_episode"]
        stale = state["stale"]
        start_episode = state["episode"] + 1
        print(f"TRAINING_RESUMED after_episode={state['episode']}", flush=True)
    completed_episode = start_episode - 1
    for episode in range(start_episode, args.episodes + 1):
        result, rollout = trainer._collect_episode(
            "mixed", args.train_seed + episode - 1, episode)
        trainer._ppo_update(rollout)
        segment.append(asdict(result))
        completed_episode = episode
        if episode % args.interval and episode != args.episodes:
            continue
        checkpoint = root / "checkpoints" / f"mixed_ppo_ep{episode:05d}.pt"
        trainer.save(checkpoint, episode)
        _write_csv(root / "training" / f"episodes_{episode:05d}.csv", segment)
        segment = []
        trainer.model.eval()
        try:
            scores = [trainer._evaluate_episode("mixed", args.validation_seed + i)
                      for i in range(args.validation_episodes)]
        finally:
            trainer.model.train()
        mean_score = statistics.mean(scores)
        improved = best_episode is None or mean_score > best_score
        if improved:
            best_score, best_episode = mean_score, episode
        meaningful_improvement = (patience_score == float("-inf")
                                  or mean_score > patience_score + args.min_delta)
        if meaningful_improvement:
            patience_score, stale = mean_score, 0
        else:
            stale += 1
        row = {"episode": episode, "mean_discounted_return": mean_score,
               "std_discounted_return": statistics.stdev(scores)
               if len(scores) > 1 else 0.0,
               "validation_episodes": len(scores), "improved": improved,
               "meaningful_improvement": meaningful_improvement,
               "consecutive_nonimprovements": stale,
               "checkpoint": str(checkpoint.relative_to(root))}
        _write_csv(root / "validation" / f"checkpoint_{episode:05d}.csv", [row])
        state_path = root / "training_state" / f"state_ep{episode:05d}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("xb") as stream:
            torch.save({
                "episode": episode,
                "checkpoint": str(checkpoint.relative_to(root)),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if args.device == "cuda" else [],
                "best_score": best_score, "patience_score": patience_score,
                "best_episode": best_episode, "stale": stale,
            }, stream)
        _plot_curve(root / "curves" / f"curve_ep{episode:05d}.svg",
                    _all_training_rows(root), _all_validation_rows(root),
                    best_episode, args.episodes)
        print(f"CHECKPOINT_VALIDATED episode={episode} score={mean_score:.6f} "
              f"best_episode={best_episode} stale={stale}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"EARLY_STOP episode={episode}", flush=True)
            break
    selected = {
        "best_episode": best_episode, "best_score": best_score,
        "checkpoint": f"checkpoints/mixed_ppo_ep{best_episode:05d}.pt",
        "completed_episodes": completed_episode,
        "stopped_early": completed_episode < args.episodes,
        "selection_metric": "held-out mixed mean discounted return",
        "validation_seed_start": args.validation_seed,
        "validation_seed_end": args.validation_seed + args.validation_episodes - 1,
    }
    _write_json(root / "selected_checkpoint.json", selected)
    _write_csv(root / "training.csv", _all_training_rows(root))
    _write_csv(root / "validation.csv", _all_validation_rows(root))
    _plot_curve(root / "training_curve.svg", _all_training_rows(root),
                _all_validation_rows(root), best_episode, args.episodes)
    with (root / "TRAINING_COMPLETE.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"episodes={completed_episode}\nbest_episode={best_episode}\n"
                     f"device={args.device}\n")
    return selected


def _summarize_ood(root: Path, rows: list[dict], resamples: int,
                   seed: int) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario_id"], row["policy"]), []).append(row)
    summaries = []
    for (scenario, policy), group in sorted(groups.items()):
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            low, high = bootstrap_interval(values, seed, resamples)
            summaries.append({"scenario_id": scenario, "policy": policy,
                              "metric": metric, "n": len(values),
                              "mean": statistics.mean(values),
                              "ci95_low": low, "ci95_high": high})
    _write_csv(root / "ood" / "summary.csv", summaries)
    by_key = {(row["scenario_id"], row["seed"], row["policy"]): row
              for row in rows}
    paired = []
    for scenario in sorted({row["scenario_id"] for row in rows}):
        seeds = sorted({row["seed"] for row in rows
                        if row["scenario_id"] == scenario})
        for baseline in BASELINES:
            for metric in METRICS:
                differences = [
                    float(by_key[(scenario, run_seed, "ordinary_ppo")][metric])
                    - float(by_key[(scenario, run_seed, baseline)][metric])
                    for run_seed in seeds]
                low, high = bootstrap_interval(differences, seed, resamples)
                paired.append({"scenario_id": scenario, "baseline": baseline,
                               "metric": metric, "n_pairs": len(differences),
                               "ppo_minus_baseline": statistics.mean(differences),
                               "ci95_low": low, "ci95_high": high})
    _write_csv(root / "ood" / "paired_vs_baselines.csv", paired)
    macro = []
    for policy in (*BASELINES, "ordinary_ppo"):
        for metric in METRICS:
            values = [row["mean"] for row in summaries
                      if row["policy"] == policy and row["metric"] == metric]
            macro.append({"policy": policy, "metric": metric,
                          "ood_conditions": len(values),
                          "macro_mean": statistics.mean(values)})
    _write_csv(root / "ood" / "macro_summary.csv", macro)


def evaluate_ood(root: Path, args: argparse.Namespace,
                 config: SimulationConfig, selected: dict) -> None:
    if (root / "ood" / "episodes.csv").exists():
        raise FileExistsError("OOD results already exist; choose a fresh output directory")
    cases = build_evaluation_cases("ood", "all", "all", config)
    if len(cases) != 10:
        raise RuntimeError("expected the existing five OOD families at L1 and L2")
    probe = MigrationEnvironment(config, scenario_by_name("mixed", config))
    state, _ = probe.reset(args.ood_seed)
    controller = TrainedPolicy(root / selected["checkpoint"], len(state),
                               config.n_nodes + 1, device=args.device)
    rows = []
    for case in cases:
        for index in range(args.ood_episodes):
            run_seed = args.ood_seed + index
            for name in (*BASELINES, "ordinary_ppo"):
                if name == "ordinary_ppo":
                    metrics, _ = run_trained_episode(name, case.scenario,
                                                     run_seed, config, controller)
                else:
                    metrics, _ = run_episode(name, case.scenario, run_seed, config)
                rows.append(_episode_row(metrics, case))
        print(f"OOD_CONDITION_COMPLETE scenario={case.scenario_id} "
              f"paired_seeds={args.ood_episodes}", flush=True)
    _write_csv(root / "ood" / "episodes.csv", rows)
    _summarize_ood(root, rows, args.bootstrap_resamples, args.ood_seed)
    _write_json(root / "ood" / "evaluation_config.json", {
        "checkpoint": selected["checkpoint"],
        "selected_on": selected["selection_metric"],
        "ood_seed_start": args.ood_seed,
        "ood_seed_end": args.ood_seed + args.ood_episodes - 1,
        "episodes_per_condition_per_policy": args.ood_episodes,
        "cases": [case.scenario_id for case in cases],
        "policies": [*BASELINES, "ordinary_ppo"],
        "device": args.device, "observation_schema_version": SCHEMA_VERSION,
    })
    with (root / "ood" / "OOD_COMPLETE.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"episodes={len(rows)}\nconditions={len(cases)}\n")


def _check_disjoint(args: argparse.Namespace) -> None:
    spans = (("train", args.train_seed, args.episodes),
             ("validation", args.validation_seed, args.validation_episodes),
             ("ood", args.ood_seed, args.ood_episodes))
    for index, (name, start, count) in enumerate(spans):
        for other, other_start, other_count in spans[index + 1:]:
            if max(start, other_start) < min(start + count, other_start + other_count):
                raise ValueError(f"{name} and {other} seed ranges overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "train", "resume", "ood"),
                        default="all")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--interval", type=int, default=1_000)
    parser.add_argument("--patience", type=int, default=3,
                        help="checkpoint validations without improvement; 0 trains all episodes")
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--ood-episodes", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int, default=20_026)
    parser.add_argument("--ood-seed", type=int, default=30_026)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.episodes < 1 or args.interval < 1 or args.patience < 0
            or args.validation_episodes < 2 or args.ood_episodes < 2
            or args.bootstrap_resamples < 100 or args.min_delta < 0
            or args.max_time <= 0):
        parser.error("invalid episode, interval, patience, resample or min-delta value")
    try:
        _check_disjoint(args)
    except ValueError as exc:
        parser.error(str(exc))
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a GPU")
    root = args.output_dir.resolve()
    if args.phase in ("all", "train") and root.exists():
        parser.error("output directory already exists; choose a new one")
    if args.phase == "resume" and ((root / "TRAINING_COMPLETE.txt").exists()
                                   or not (root / "training_state").is_dir()):
        parser.error("resume requires an interrupted run with checkpoint state")
    if args.phase == "ood" and not (root / "TRAINING_COMPLETE.txt").is_file():
        parser.error("OOD phase requires a completed training run")
    config = SimulationConfig(max_time_seconds=args.max_time)
    ppo_config = PPOConfig(gamma=0.995, entropy_coefficient=0.03,
                           learning_rate=3e-4, clip_ratio=0.2,
                           update_epochs=8, minibatch_size=128,
                           gae_lambda=0.95, scenario_eval_interval=args.interval)
    if args.dry_run:
        print(json.dumps({"output_dir": str(root), "phase": args.phase,
                          "training_max_episodes": args.episodes,
                          "checkpoint_interval": args.interval,
                          "patience": args.patience,
                          "validation_episodes_per_checkpoint": args.validation_episodes,
                          "ood_cases": 10, "ood_episodes_per_policy_per_case":
                          args.ood_episodes, "ood_total_episodes":
                          10 * 6 * args.ood_episodes,
                          "device": args.device,
                          "cuda_device": torch.cuda.get_device_name(0)
                          if args.device == "cuda" else None}, indent=2))
        return
    if args.phase in ("all", "train"):
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "experiment_config.json", {
            "arguments": vars(args) | {"output_dir": str(root)},
            "env_config": asdict(config), "ppo_config": asdict(ppo_config),
            "observation_schema_version": SCHEMA_VERSION,
            "feature_order": FEATURE_ORDER, "feature_scales": FEATURE_SCALES,
            "ood_design": "five existing OOD families at L1/L2; never used for selection",
            "ood_primary_metric": "task_completion_rate",
            "ood_secondary_metrics": [metric for metric in METRICS
                                      if metric != "task_completion_rate"],
            "comparison": "paired seeds within each OOD condition; descriptive bootstrap CI",
        })
        selected = train(root, args, config, ppo_config)
    else:
        saved = json.loads((root / "experiment_config.json").read_text(
            encoding="utf-8"))
        saved_args = saved["arguments"]
        _check_disjoint(argparse.Namespace(
            train_seed=saved_args["train_seed"], episodes=saved_args["episodes"],
            validation_seed=saved_args["validation_seed"],
            validation_episodes=saved_args["validation_episodes"],
            ood_seed=args.ood_seed, ood_episodes=args.ood_episodes))
        config_data = dict(saved["env_config"])
        config_data["monitoring"] = MonitoringConfig(**config_data["monitoring"])
        config = SimulationConfig(**config_data)
        if saved["observation_schema_version"] != SCHEMA_VERSION:
            raise ValueError("saved run observation schema does not match current code")
        if args.phase == "resume":
            resumed_args = argparse.Namespace(**saved_args)
            ppo_config = PPOConfig(**saved["ppo_config"])
            _check_disjoint(resumed_args)
            selected = train(root, resumed_args, config, ppo_config, resume=True)
            args = resumed_args
        else:
            selected = json.loads((root / "selected_checkpoint.json").read_text(
                encoding="utf-8"))
    if args.phase in ("all", "ood"):
        evaluate_ood(root, args, config, selected)


if __name__ == "__main__":
    main()
