"""Prepare one seed, train one branch, or test a completed PPO/PGD-PPO pair.

The prepare phase owns the shared warmup and ID surrogate data. The branch
phase writes only to its own method directory, so all six branches can run
as separate CUDA processes after the three prepare phases finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from run_ood_pgd_experiment import (
    SCENARIOS, balanced_schedule, build_surrogate, collect_and_update,
    evaluate, flush, make_trainer, save_checkpoint, search_block,
    set_auxiliary_beta, validate,
    write_csv,
)
from run_ppo_abc import pretrain_gate
from ftm.config import SimulationConfig
from ftm.observation import FEATURE_ORDER, FEATURE_SCALES, SCHEMA_VERSION
from ftm.robust_ppo import PPOConfig
from ftm.surrogate import build_transition_surrogate


def paths_and_seeds(args):
    root = args.output_dir.resolve()
    schedule = balanced_schedule(args.episodes, args.seed * 1_000_000 + 10_000)
    env_seeds = [args.seed * 1_000_000 + 100_000 + i
                 for i in range(args.episodes)]
    val_seeds = [args.seed * 1_000_000 + 700_000 + i
                 for i in range(args.validation_per_scenario)]
    return root, schedule, env_seeds, val_seeds


def prepare(args, config):
    root, schedule, env_seeds, val_seeds = paths_and_seeds(args)
    ppo = PPOConfig(update_epochs=4, minibatch_size=256)
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps({
        "args": vars(args) | {"output_dir": str(root)},
        "scenarios": SCENARIOS, "ppo_config": asdict(ppo),
        "simulation_config": asdict(config),
        "observation_schema_version": SCHEMA_VERSION,
        "feature_order": FEATURE_ORDER, "feature_scales": FEATURE_SCALES,
        "validation_seeds": val_seeds,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_csv(root / "schedule.csv", [
        {"episode": i + 1, "scenario": scenario, "env_seed": env_seeds[i]}
        for i, scenario in enumerate(schedule)])

    trainer = make_trainer(config, ppo, args.seed * 1_000_000 + 1, args.device)
    pretrain_info = pretrain_gate(trainer, config, args.seed, args.pretrain_episodes)
    (root / "gate_pretraining.json").write_text(
        json.dumps(pretrain_info, indent=2), encoding="utf-8")
    buffer, training = [], []
    for i in range(args.warmup):
        set_auxiliary_beta(trainer, i + 1, args)
        result = collect_and_update(trainer, schedule[i], env_seeds[i], i + 1,
                                    buffer, args.rollout_steps)
        training.append(asdict(result) | {"method": "warmup", "adversarial": False})
    flush(trainer, buffer)
    save_checkpoint(trainer, root / "warmup.pt", args.warmup,
                    {"method": "warmup"})
    write_csv(root / "warmup_training.csv", training)
    score, rows = validate(trainer, config, val_seeds)
    write_csv(root / "validation" / "warmup.csv", rows)
    (root / "warmup_score.json").write_text(
        json.dumps({"score": score}, indent=2), encoding="utf-8")
    print(f"WARMUP_COMPLETE seed={args.seed} score={score:.6f}", flush=True)

    surrogate, records = build_surrogate(
        root, config, args, trainer.model.gate[0].in_features)
    with (root / "surrogate.pt").open("xb") as stream:
        torch.save(surrogate.state_dict(), stream)
    (root / "PREPARE_COMPLETE.txt").write_text(
        f"seed={args.seed}\nepisodes={args.warmup}\n"
        f"surrogate_records={len(records)}\n", encoding="utf-8")
    print(f"PREPARE_COMPLETE seed={args.seed}", flush=True)


def branch(args, config):
    root, schedule, env_seeds, val_seeds = paths_and_seeds(args)
    if not (root / "PREPARE_COMPLETE.txt").exists():
        raise RuntimeError("shared warmup and surrogate are incomplete")
    method = args.method
    method_dir = root / method
    method_dir.mkdir(exist_ok=False)
    warmup = torch.load(root / "warmup.pt", map_location="cpu",
                        weights_only=False)
    if warmup.get("model_type") != "c_hierarchical_aux":
        raise ValueError("shared warmup must use the C hierarchical PPO model")
    ppo = PPOConfig(**warmup["ppo_config"])
    trainer = make_trainer(config, ppo, args.seed * 1_000_000 + 2, args.device)
    trainer.model.load_state_dict(warmup["model_state_dict"])
    trainer.optimizer.load_state_dict(warmup["optimizer_state_dict"])
    warmup_score = json.loads((root / "warmup_score.json").read_text(
        encoding="utf-8"))["score"]
    best_score = warmup_score
    best = root / "warmup.pt"
    buffer, train_rows, val_rows = [], [], []

    surrogate, records = None, None
    if method == "pgd_ppo":
        surrogate = build_transition_surrogate(
            trainer.model.gate[0].in_features, config.n_nodes + 1).to(args.device)
        surrogate.load_state_dict(torch.load(root / "surrogate.pt",
                                             map_location="cpu", weights_only=True))
        surrogate.eval()
        records = [json.loads(line) for line in (root / "surrogate_id.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
    block_choices = None
    for i in range(args.warmup, args.episodes):
        set_auxiliary_beta(trainer, i + 1, args)
        block = (i - args.warmup) // args.block_size
        if method == "pgd_ppo" and (i - args.warmup) % args.block_size == 0:
            block_choices, pgd_rows = search_block(trainer, surrogate, records,
                                                   config, args, block + 1)
            write_csv(method_dir / "pgd" / f"block_{block + 1:04d}.csv", pgd_rows)
            print(f"PGD_BLOCK_COMPLETE seed={args.seed} block={block + 1}",
                  flush=True)
        adv_ratio = 0.3 if block < 10 else 0.5
        adversarial = (method == "pgd_ppo" and
                       random.Random(args.seed * 1_000_000 + 600_000 + i)
                       .random() < adv_ratio)
        override = block_choices[schedule[i]] if adversarial else None
        result = collect_and_update(trainer, schedule[i], env_seeds[i], i + 1,
                                    buffer, args.rollout_steps, override)
        train_rows.append(asdict(result) | {"method": method,
                                            "adversarial": adversarial})
        episode = i + 1
        if episode % args.validation_interval == 0 or episode == args.episodes:
            flush(trainer, buffer)
            score, rows = validate(trainer, config, val_seeds)
            checkpoint = method_dir / f"episode_{episode:05d}.pt"
            extra = {"method": method, "validation_score": score,
                     "pgd_steps": args.pgd_steps if method == "pgd_ppo" else 0}
            if method == "pgd_ppo":
                extra["surrogate_state_dict"] = surrogate.state_dict()
            save_checkpoint(trainer, checkpoint, episode, extra)
            if score > best_score:
                best_score, best = score, checkpoint
            val_rows.append({"method": method, "episode": episode,
                             "score": score,
                             "best_episode": args.warmup if best.name == "warmup.pt"
                             else int(best.stem.split("_")[-1]),
                             "checkpoint": str(checkpoint.relative_to(root))})
            write_csv(method_dir / "validation" /
                      f"episode_{episode:05d}.csv", rows)
            print(f"CHECKPOINT_VALIDATED seed={args.seed} method={method} "
                  f"episode={episode} score={score:.6f}", flush=True)
    write_csv(method_dir / "training.csv", train_rows)
    write_csv(method_dir / "validation.csv", val_rows)
    (method_dir / "selected.json").write_text(json.dumps({
        "checkpoint": str(best.relative_to(root)), "score": best_score,
        "selection": "fixed independent ID seeds; completion + availability + (1-SLA)"
    }, indent=2), encoding="utf-8")
    (method_dir / "BRANCH_COMPLETE.txt").write_text(
        f"seed={args.seed}\nmethod={method}\nepisodes={args.episodes}\n",
        encoding="utf-8")
    print(f"BRANCH_COMPLETE seed={args.seed} method={method}", flush=True)


def test_pair(args, config):
    root = args.output_dir.resolve()
    for method in ("ppo", "pgd_ppo"):
        if not (root / method / "BRANCH_COMPLETE.txt").exists():
            raise RuntimeError(f"{method} branch completion marker is missing")
    training_rows = []
    for path in (root / "warmup_training.csv", root / "ppo" / "training.csv",
                 root / "pgd_ppo" / "training.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            training_rows.extend(csv.DictReader(stream))
    write_csv(root / "training.csv", training_rows)
    validation_rows = []
    for path in (root / "ppo" / "validation.csv",
                 root / "pgd_ppo" / "validation.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            validation_rows.extend(csv.DictReader(stream))
    write_csv(root / "validation.csv", validation_rows)
    (root / "TRAINING_COMPLETE.txt").write_text(
        f"episodes={args.episodes}\nseed={args.seed}\n",
        encoding="utf-8")
    print(f"TRAINING_COMPLETE seed={args.seed}", flush=True)
    evaluate(root, config, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True,
                        choices=("prepare", "branch", "test"))
    parser.add_argument("--method", choices=("ppo", "pgd_ppo"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--block-size", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--validation-per-scenario", type=int, default=20)
    parser.add_argument("--id-episodes", type=int, default=50)
    parser.add_argument("--ood-episodes", type=int, default=100)
    parser.add_argument("--surrogate-episodes", type=int, default=10)
    parser.add_argument("--pretrain-episodes", type=int, default=50)
    parser.add_argument("--aux-beta", type=float, default=0.1)
    parser.add_argument("--surrogate-epochs", type=int, default=20)
    parser.add_argument("--pgd-steps", type=int, default=4)
    parser.add_argument("--pgd-rate", type=float, default=0.05)
    parser.add_argument("--pgd-restarts", type=int, default=2)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if (args.phase == "branch" and args.method is None):
        parser.error("--method is required for branch phase")
    if (args.seed < 1 or args.episodes <= args.warmup or args.warmup < 1
            or args.episodes % 5 or args.warmup % 5
            or (args.episodes - args.warmup) % args.block_size
            or args.block_size % 5 or args.validation_interval < 1
            or args.pretrain_episodes < 1 or args.aux_beta < 0):
        parser.error("invalid episode schedule")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    config = SimulationConfig(max_time_seconds=args.max_time)
    if args.phase == "prepare":
        prepare(args, config)
    elif args.phase == "branch":
        branch(args, config)
    else:
        test_pair(args, config)


if __name__ == "__main__":
    main()
