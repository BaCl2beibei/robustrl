"""Five-scenario PPO/PGD-PPO experiment described in 实验.md.

Run one independent seed per invocation. A fresh output directory is required.
The training and test phases are separate so OOD cannot affect model selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.config import SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import (_episode_row, bootstrap_interval,
                                  build_evaluation_cases)
from ftm.main import run_episode
from ftm.observation import FEATURE_ORDER, FEATURE_SCALES, SCHEMA_VERSION
from ftm.robust_ppo import FiniteScenarioRobustPPO, PPOConfig, _build_actor_critic
from ftm.surrogate import (fit_surrogate, raw_to_scenario, scenario_to_raw,
                           training_raw_bounds, worst_case_scenario)
from ftm.surrogate_dataset import collect as collect_surrogate_data
from ftm.trained_policy import TrainedPolicy, run_trained_episode

SCENARIOS = ("normal", "failure", "network", "load", "mixed")
POLICIES = ("no_migration", "time_min", "ppo", "pgd_ppo")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def balanced_schedule(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    order: list[str] = []
    while len(order) < count:
        block = list(SCENARIOS)
        rng.shuffle(block)
        while order and block[0] == order[-1]:
            rng.shuffle(block)
        order.extend(block)
    return order[:count]


def make_trainer(config: SimulationConfig, ppo: PPOConfig, seed: int,
                 device: str) -> FiniteScenarioRobustPPO:
    import torch

    torch.manual_seed(seed)
    trainer = FiniteScenarioRobustPPO(config, ppo, SCENARIOS, device, seed)
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(seed)
    trainer.model = _build_actor_critic(len(state), config.n_nodes + 1).to(device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=ppo.learning_rate)
    return trainer


def save_checkpoint(trainer: FiniteScenarioRobustPPO, path: Path, episode: int,
                    extra: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_state_dict": trainer.model.state_dict(),
        "observation_schema_version": SCHEMA_VERSION,
        "feature_order": FEATURE_ORDER, "feature_scales": FEATURE_SCALES,
        "state_dim": trainer.model.body[0].in_features,
        "action_dim": trainer.model.actor.out_features,
        "env_config": asdict(trainer.env_config),
        "ppo_config": asdict(trainer.ppo_config),
        "scenario_names": trainer.scenario_names,
        "seed": trainer.seed, "episodes": episode,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        **extra,
    }
    with path.open("xb") as stream:
        torch.save(data, stream)


def collect_and_update(trainer: FiniteScenarioRobustPPO, scenario: str,
                       env_seed: int, episode: int, buffer: list[dict],
                       rollout_steps: int, override=None):
    result, rollout = trainer._collect_episode(scenario, env_seed, episode, override)
    buffer.append(rollout)
    if sum(len(item["rewards"]) for item in buffer) >= rollout_steps:
        trainer._ppo_update_many(buffer)
        buffer.clear()
    return result


def flush(trainer: FiniteScenarioRobustPPO, buffer: list[dict]) -> None:
    if buffer:
        trainer._ppo_update_many(buffer)
        buffer.clear()


def validate(trainer: FiniteScenarioRobustPPO, config: SimulationConfig,
             seeds: list[int]) -> tuple[float, list[dict]]:
    from ftm.metrics import compute_episode_metrics
    import torch

    rows = []
    trainer.model.eval()
    try:
        for scenario in SCENARIOS:
            for seed in seeds:
                env = MigrationEnvironment(config, scenario_by_name(scenario, config))
                env.reset(seed)
                total_reward = 0.0
                while True:
                    context = env.get_decision_context()
                    state = torch.tensor(env.observation(), dtype=torch.float32,
                                         device=trainer.device).unsqueeze(0)
                    mask = torch.tensor(context.action_mask, dtype=torch.bool,
                                        device=trainer.device).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = trainer.model(state)
                        action = int(logits.masked_fill(~mask, -1e9).argmax(-1).item())
                    _, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                metric = compute_episode_metrics(env, "validation", seed, total_reward)
                row = metric.to_dict()
                row["scenario"] = scenario
                rows.append(row)
    finally:
        trainer.model.train()
    score = statistics.mean((row["task_completion_rate"]
                             + row["task_service_availability"]
                             + 1 - row["sla_violation_rate"]) / 3 for row in rows)
    return score, rows


def build_surrogate(root: Path, config: SimulationConfig, args, state_dim: int):
    import torch

    dataset = root / "surrogate_id.jsonl"
    collect_surrogate_data(dataset, config, args.surrogate_episodes, SCENARIOS,
                           ("no_migration", "time_min"),
                           args.seed * 1_000_000 + 400_000)
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    if not records or any(row["observation_schema_version"] != SCHEMA_VERSION
                          for row in records):
        raise RuntimeError("invalid ID surrogate data")
    torch.manual_seed(args.seed * 1_000_000 + 400_001)
    surrogate = fit_surrogate(records, config.n_nodes + 1,
                              epochs=args.surrogate_epochs, device=args.device)
    return surrogate, records


def search_block(trainer, surrogate, records, config, args, block: int):
    import torch

    choices = {}
    log = []
    rng = random.Random(args.seed * 1_000_000 + 500_000 + block)
    for scenario in SCENARIOS:
        template = scenario_by_name(scenario, config)
        center = scenario_to_raw(template)
        lower, upper = training_raw_bounds(template)
        axes = tuple(upper_i - center_i for center_i, upper_i in zip(center, upper))
        family_records = [row for row in records if row["scenario"] == scenario]
        if not family_records:
            raise RuntimeError(f"no surrogate records for {scenario}")
        candidates = [center]
        values = []
        for row in rng.sample(family_records, min(args.pgd_restarts, len(family_records))):
            _, raw, value = worst_case_scenario(
                surrogate, lambda states: trainer.model(states)[1],
                row["state"], row["action"], center, axes, 1.0, template,
                gamma=trainer.ppo_config.gamma, pgd_steps=args.pgd_steps,
                learning_rate=args.pgd_rate, device=args.device,
                lower_bounds=lower, upper_bounds=upper)
            if not all(lo - 1e-6 <= x <= hi + 1e-6
                       for x, lo, hi in zip(raw, lower, upper)):
                raise RuntimeError("PGD escaped ID bounds")
            candidates.append(raw)
            values.append(value)
        # One frozen physical scenario for this family and block.
        selected = candidates[1 + min(range(len(values)), key=values.__getitem__)]
        choices[scenario] = raw_to_scenario(selected, template,
                                            name=f"{scenario}_pgd_block_{block}")
        log.append({"block": block, "scenario": scenario,
                    "raw_parameters": json.dumps(selected),
                    "predicted_value": min(values)})
    return choices, log


def train(root: Path, config: SimulationConfig, args) -> None:
    import torch

    ppo = PPOConfig(update_epochs=4, minibatch_size=256)
    schedule = balanced_schedule(args.episodes, args.seed * 1_000_000 + 10_000)
    env_seeds = [args.seed * 1_000_000 + 100_000 + i for i in range(args.episodes)]
    val_seeds = [args.seed * 1_000_000 + 700_000 + i
                 for i in range(args.validation_per_scenario)]
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
    buffer, train_rows, val_rows = [], [], []
    for i in range(args.warmup):
        result = collect_and_update(trainer, schedule[i], env_seeds[i], i + 1,
                                    buffer, args.rollout_steps)
        train_rows.append(asdict(result) | {"method": "warmup", "adversarial": False})
    flush(trainer, buffer)
    warmup_path = root / "warmup.pt"
    save_checkpoint(trainer, warmup_path, args.warmup, {"method": "warmup"})
    print(f"WARMUP_COMPLETE episodes={args.warmup}", flush=True)
    warmup_score, warmup_validation = validate(trainer, config, val_seeds)
    write_csv(root / "validation" / "warmup.csv", warmup_validation)
    surrogate, records = build_surrogate(root, config, args,
                                         trainer.model.body[0].in_features)
    state = copy.deepcopy(trainer.model.state_dict())
    optim_state = copy.deepcopy(trainer.optimizer.state_dict())
    for method in ("ppo", "pgd_ppo"):
        branch = make_trainer(config, ppo, args.seed * 1_000_000 + 2, args.device)
        branch.model.load_state_dict(state)
        branch.optimizer.load_state_dict(optim_state)
        best_score = warmup_score
        best = warmup_path
        block_choices = None
        for i in range(args.warmup, args.episodes):
            block = (i - args.warmup) // args.block_size
            if method == "pgd_ppo" and (i - args.warmup) % args.block_size == 0:
                block_choices, pgd_rows = search_block(branch, surrogate, records,
                                                       config, args, block + 1)
                write_csv(root / "pgd" / f"block_{block + 1:04d}.csv", pgd_rows)
                print(f"PGD_BLOCK_COMPLETE block={block + 1}", flush=True)
            adv_ratio = 0.3 if block < 10 else 0.5
            adversarial = (method == "pgd_ppo" and
                           random.Random(args.seed * 1_000_000 + 600_000 + i)
                           .random() < adv_ratio)
            override = block_choices[schedule[i]] if adversarial else None
            result = collect_and_update(branch, schedule[i], env_seeds[i], i + 1,
                                        buffer, args.rollout_steps, override)
            train_rows.append(asdict(result) | {"method": method,
                                                "adversarial": adversarial})
            episode = i + 1
            if episode % args.validation_interval == 0 or episode == args.episodes:
                flush(branch, buffer)
                score, rows = validate(branch, config, val_seeds)
                checkpoint = root / method / f"episode_{episode:05d}.pt"
                extra = {"method": method, "validation_score": score,
                         "pgd_steps": args.pgd_steps if method == "pgd_ppo" else 0}
                if method == "pgd_ppo":
                    extra["surrogate_state_dict"] = surrogate.state_dict()
                save_checkpoint(branch, checkpoint, episode, extra)
                if score > best_score:
                    best_score = score
                    best = checkpoint
                val_rows.append({"method": method, "episode": episode,
                                 "score": score,
                                 "best_episode": args.warmup if best == warmup_path
                                 else int(best.stem.split("_")[-1]),
                                 "checkpoint": str(checkpoint.relative_to(root))})
                write_csv(root / "validation" /
                          f"{method}_{episode:05d}.csv",
                          [dict(row, method=method, episode=episode) for row in rows])
                print(f"CHECKPOINT_VALIDATED method={method} episode={episode} "
                      f"score={score:.6f} best={best.name}", flush=True)
        (root / method / "selected.json").write_text(json.dumps({
            "checkpoint": str(best.relative_to(root)), "score": best_score,
            "selection": "fixed independent ID seeds; completion + availability + (1-SLA)"
        }, indent=2), encoding="utf-8")
        print(f"BRANCH_COMPLETE method={method}", flush=True)
    write_csv(root / "training.csv", train_rows)
    write_csv(root / "validation.csv", val_rows)
    (root / "TRAINING_COMPLETE.txt").write_text(
        f"episodes={args.episodes}\nseed={args.seed}\n", encoding="utf-8")
    print("TRAINING_COMPLETE", flush=True)


def evaluate(root: Path, config: SimulationConfig, args) -> None:
    import torch

    if not (root / "TRAINING_COMPLETE.txt").exists():
        raise RuntimeError("training completion marker is missing")
    cases = build_evaluation_cases("all", "all", "all", config)
    cases = [case for case in cases if case.scenario_family in SCENARIOS
             and (case.split == "ood" or case.scenario_id in SCENARIOS)]
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(1)
    controllers = {}
    for method in ("ppo", "pgd_ppo"):
        selected = json.loads((root / method / "selected.json").read_text(encoding="utf-8"))
        controllers[method] = TrainedPolicy(root / selected["checkpoint"],
                                            len(state), config.n_nodes + 1, args.device)
    rows = []
    for case in cases:
        count = args.id_episodes if case.split == "id" else args.ood_episodes
        for i in range(count):
            seed = args.seed * 1_000_000 + 800_000 + i
            for method in POLICIES:
                if method in controllers:
                    torch.manual_seed(seed)
                    metrics, _ = run_trained_episode(method, case.scenario, seed,
                                                     config, controllers[method])
                else:
                    metrics, _ = run_episode(method, case.scenario, seed, config)
                rows.append(_episode_row(metrics, case))
        print(f"TEST_CONDITION_COMPLETE scenario={case.scenario_id}", flush=True)
    write_csv(root / "test" / "episodes.csv", rows)
    summary = []
    for case in cases:
        for method in POLICIES:
            group = [row for row in rows if row["scenario_id"] == case.scenario_id
                     and row["policy"] == method]
            for metric in ("task_completion_rate", "task_service_availability",
                           "sla_violation_rate", "mean_completion_seconds_horizon_capped",
                           "migration_count", "mean_migration_seconds", "total_reward"):
                values = [float(row[metric]) for row in group if row[metric] is not None]
                summary.append({"scenario_id": case.scenario_id, "method": method,
                                "metric": metric, "n": len(values),
                                "mean": statistics.mean(values) if values else None,
                                "stdev": statistics.stdev(values) if len(values) > 1 else None})
    write_csv(root / "test" / "summary.csv", summary)
    paired = []
    for case in cases:
        for method in ("time_min", "ppo", "pgd_ppo"):
            for reference in ("no_migration", "ppo"):
                if method == reference:
                    continue
                for metric in ("task_completion_rate", "task_service_availability",
                               "sla_violation_rate", "total_reward"):
                    candidate = {row["seed"]: row for row in rows
                                 if row["scenario_id"] == case.scenario_id
                                 and row["policy"] == method}
                    baseline = {row["seed"]: row for row in rows
                                if row["scenario_id"] == case.scenario_id
                                and row["policy"] == reference}
                    differences = [float(candidate[seed][metric])
                                   - float(baseline[seed][metric])
                                   for seed in sorted(candidate.keys() & baseline.keys())
                                   if candidate[seed][metric] is not None
                                   and baseline[seed][metric] is not None]
                    low, high = bootstrap_interval(differences, seed=args.seed,
                                                   resamples=1000)
                    paired.append({"scenario_id": case.scenario_id,
                                   "method": method, "reference": reference,
                                   "metric": metric, "n_pairs": len(differences),
                                   "mean_difference": statistics.mean(differences)
                                   if differences else None,
                                   "ci95_low": low, "ci95_high": high})
    write_csv(root / "test" / "paired.csv", paired)
    (root / "TEST_COMPLETE.txt").write_text(f"episodes={len(rows)}\n", encoding="utf-8")
    print("TEST_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
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
    parser.add_argument("--surrogate-epochs", type=int, default=20)
    parser.add_argument("--pgd-steps", type=int, default=4)
    parser.add_argument("--pgd-rate", type=float, default=0.05)
    parser.add_argument("--pgd-restarts", type=int, default=2)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if (args.seed < 1 or args.episodes <= args.warmup or args.warmup < 1
            or args.episodes % 5 or args.warmup % 5
            or (args.episodes - args.warmup) % args.block_size
            or args.block_size % 5 or args.validation_interval < 1
            or min(args.rollout_steps, args.validation_per_scenario,
                   args.id_episodes, args.ood_episodes, args.surrogate_episodes,
                   args.surrogate_epochs, args.pgd_steps, args.pgd_restarts) < 1):
        parser.error("invalid episode counts or schedule")
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    config = SimulationConfig(max_time_seconds=args.max_time)
    root = args.output_dir.resolve()
    if args.phase in ("all", "train"):
        if root.exists():
            parser.error("output directory already exists")
        train(root, config, args)
    if args.phase in ("all", "test"):
        evaluate(root, config, args)


if __name__ == "__main__":
    main()
