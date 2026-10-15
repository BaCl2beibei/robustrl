"""Train C hierarchical PPO and compare it with two baselines on paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.config import SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import _episode_row, build_evaluation_cases
from ftm.hierarchical_ppo import HierarchicalPPO, build_hierarchical_actor_critic, gate_labels
from ftm.metrics import compute_episode_metrics
from ftm.observation import FEATURE_ORDER, FEATURE_SCALES, SCHEMA_VERSION, encode_observation
from ftm.policies import NoMigrationPolicy, TimeMinPolicy
from ftm.robust_ppo import PPOConfig

SCENARIOS = ("normal", "failure", "network", "load", "mixed")
METHOD = "c_hierarchical_aux"
BASELINES = ("no_migration", "time_min")
TEST_METHODS = BASELINES + (METHOD,)
METRICS = ("task_completion_rate", "task_service_availability",
           "sla_violation_rate", "mean_completion_seconds_horizon_capped",
           "migration_count", "total_reward")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def schedule(episodes: int, seed: int):
    rng = random.Random(seed * 1_000_000 + 10_000)
    result = []
    while len(result) < episodes:
        block = list(SCENARIOS)
        rng.shuffle(block)
        result.extend(block)
    return [(result[i], seed * 1_000_000 + 100_000 + i)
            for i in range(episodes)]


def build_trainer(config: SimulationConfig, ppo: PPOConfig,
                  seed: int, device: str):
    import torch

    model_seed = seed * 1_000_000 + 1
    torch.manual_seed(model_seed)
    trainer = HierarchicalPPO(config, ppo, SCENARIOS, device, model_seed,
                              auxiliary=True)
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(seed)
    trainer.model = build_hierarchical_actor_critic(
        len(state), config.n_nodes + 1).to(device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=ppo.learning_rate)
    return trainer


def run_argmax(model, config: SimulationConfig, scenario, seed: int,
               device: str, method: str):
    import torch

    env = MigrationEnvironment(config, scenario)
    env.reset(seed)
    reward_total = 0.0
    wait_count = move_count = interrupted_count = 0
    model.eval()
    while True:
        context = env.get_decision_context()
        state = torch.tensor(encode_observation(context), dtype=torch.float32,
                             device=device).unsqueeze(0)
        mask = torch.tensor(context.action_mask, dtype=torch.bool,
                            device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state, mask)
            action = int(logits.masked_fill(~mask, -1.0e9).argmax(-1).item())
        wait_count += action == config.n_nodes
        move_count += action != config.n_nodes
        task = context.get_focus_task()
        interrupted_count += task is not None and task.status == "interrupted"
        _, reward, terminated, truncated, _ = env.step(action)
        reward_total += reward
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(env, method, seed, reward_total)
    return metrics, {"wait_decisions": wait_count, "move_decisions": move_count,
                     "interrupted_decisions": interrupted_count}


def run_baseline(method, config, scenario, seed):
    env = MigrationEnvironment(config, scenario)
    env.reset(seed)
    policy = NoMigrationPolicy() if method == "no_migration" else TimeMinPolicy()
    reward_total = 0.0
    wait_count = move_count = interrupted_count = 0
    while True:
        context = env.get_decision_context()
        action = policy.act(context)
        wait_count += action == config.n_nodes
        move_count += action != config.n_nodes
        task = context.get_focus_task()
        interrupted_count += task is not None and task.status == "interrupted"
        _, reward, terminated, truncated, _ = env.step(action)
        reward_total += reward
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(env, method, seed, reward_total)
    return metrics, {"wait_decisions": wait_count, "move_decisions": move_count,
                     "interrupted_decisions": interrupted_count}


def validation_score(model, config, seeds, device):
    values = []
    for name in SCENARIOS:
        for seed in seeds:
            metrics, _ = run_argmax(model, config, scenario_by_name(name, config),
                                    seed, device, "validation")
            values.append((metrics.task_completion_rate
                           + metrics.task_service_availability
                           + 1 - metrics.sla_violation_rate) / 3)
    model.train()
    return statistics.mean(values)


def pretrain_gate(trainer, config, seed: int, episodes: int):
    """Use separate ID heuristic episodes; record their extra interaction cost."""
    import torch

    states, masks = [], []
    policy = TimeMinPolicy()
    for i, (name, env_seed) in enumerate(schedule(episodes, seed)):
        env = MigrationEnvironment(config, scenario_by_name(name, config))
        env.reset(seed * 1_000_000 + 300_000 + i)
        while True:
            context = env.get_decision_context()
            states.append(encode_observation(context))
            masks.append(context.action_mask)
            _, _, terminated, truncated, _ = env.step(policy.act(context))
            if terminated or truncated:
                break
    state_tensor = torch.tensor(states, dtype=torch.float32, device=trainer.device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=trainer.device)
    labels, selected = gate_labels(state_tensor, mask_tensor, config.n_nodes)
    positive = torch.where(selected & (labels > 0.5))[0]
    negative = torch.where(selected & (labels < 0.5))[0]
    if len(positive) == 0 or len(negative) == 0:
        raise RuntimeError("high-confidence Gate pretraining requires both labels")
    generator = torch.Generator(device=trainer.device).manual_seed(seed * 1_000_000 + 4)
    sample_size = min(len(positive), len(negative))
    chosen = torch.cat((positive[torch.randperm(len(positive), generator=generator,
                                               device=trainer.device)[:sample_size]],
                        negative[torch.randperm(len(negative), generator=generator,
                                               device=trainer.device)[:sample_size]]))
    optimizer = torch.optim.Adam(trainer.model.gate.parameters(), lr=trainer.ppo_config.learning_rate)
    for _ in range(5):
        for batch in chosen[torch.randperm(len(chosen), generator=generator,
                                          device=trainer.device)].split(256):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                trainer.model.gate_logits(state_tensor[batch]), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {"extra_episodes": episodes, "extra_decisions": len(states),
            "positive_labels": len(positive), "negative_labels": len(negative),
            "balanced_samples_per_class": sample_size}


def save_model(path, trainer, method, episode, score):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save({"model_state_dict": trainer.model.state_dict(),
                    "model_type": method, "state_dim": next(trainer.model.parameters()).shape[1],
                    "action_dim": trainer.env_config.n_nodes + 1,
                    "observation_schema_version": SCHEMA_VERSION,
                    "feature_order": FEATURE_ORDER, "feature_scales": FEATURE_SCALES,
                    "episode": episode, "validation_score": score}, stream)


def train(root, args):
    config = SimulationConfig()
    ppo = PPOConfig(update_epochs=4, minibatch_size=256)
    trainer = build_trainer(config, ppo, args.seed, args.device)
    target = root / METHOD
    target.mkdir(parents=True, exist_ok=False)
    info = pretrain_gate(trainer, config, args.seed, args.pretrain_episodes)
    (target / "pretraining.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    train_rows, validation_rows, buffer = [], [], []
    best_score, best_path = -float("inf"), None
    validation_seeds = [args.seed * 1_000_000 + 700_000 + i
                        for i in range(args.validation_per_scenario)]
    for episode, (name, env_seed) in enumerate(schedule(args.episodes, args.seed), 1):
        trainer.auxiliary_beta = args.aux_beta * max(0.0, 1 - episode / args.episodes)
        result, rollout = trainer._collect_episode(name, env_seed, episode)
        buffer.append(rollout)
        train_rows.append(asdict(result) | {"method": METHOD})
        if sum(len(item["rewards"]) for item in buffer) >= args.rollout_steps:
            trainer._ppo_update_many(buffer)
            buffer.clear()
        if episode % args.validation_interval == 0 or episode == args.episodes:
            if buffer:
                trainer._ppo_update_many(buffer)
                buffer.clear()
            score = validation_score(trainer.model, config, validation_seeds, args.device)
            path = target / f"episode_{episode:05d}.pt"
            save_model(path, trainer, METHOD, episode, score)
            if score > best_score:
                best_score, best_path = score, path
            validation_rows.append({"episode": episode, "score": score,
                                    "best_episode": int(best_path.stem.split("_")[-1])})
            print(f"CHECKPOINT_VALIDATED method={METHOD} episode={episode} score={score:.6f}",
                  flush=True)
    write_csv(target / "training.csv", train_rows)
    write_csv(target / "validation.csv", validation_rows)
    (target / "selected.json").write_text(json.dumps({
        "checkpoint": best_path.name, "score": best_score,
        "selection": "independent ID seeds; completion, service, and SLA mean"
    }, indent=2), encoding="utf-8")
    (target / "BRANCH_COMPLETE.txt").write_text(
        f"method={METHOD}\nepisodes={args.episodes}\nseed={args.seed}\n", encoding="utf-8")
    print(f"BRANCH_COMPLETE method={METHOD}", flush=True)


def load_selected(root, config, device):
    import torch

    selected = json.loads((root / METHOD / "selected.json").read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(root / METHOD / selected["checkpoint"],
                                map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(root / METHOD / selected["checkpoint"], map_location="cpu")
    if checkpoint["observation_schema_version"] != SCHEMA_VERSION:
        raise ValueError("observation schema changed; retrain this branch")
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(1)
    if checkpoint.get("model_type") != METHOD:
        raise ValueError("checkpoint is not the C hierarchical PPO model")
    model = build_hierarchical_actor_critic(len(state), config.n_nodes + 1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def test(root, args):
    config = SimulationConfig()
    if not (root / METHOD / "BRANCH_COMPLETE.txt").is_file():
        raise RuntimeError(f"missing completed training branch: {METHOD}")
    model = load_selected(root, config, args.device)
    cases = build_evaluation_cases("all", "all", "all", config)
    cases = [case for case in cases if case.scenario_family in SCENARIOS
             and (case.split == "ood" or case.scenario_id in SCENARIOS)]
    rows = []
    for case in cases:
        count = args.id_episodes if case.split == "id" else args.ood_episodes
        for i in range(count):
            env_seed = args.seed * 1_000_000 + 800_000 + i
            for method in TEST_METHODS:
                if method == METHOD:
                    metrics, decisions = run_argmax(model, config, case.scenario,
                                                    env_seed, args.device, method)
                else:
                    metrics, decisions = run_baseline(method, config,
                                                      case.scenario, env_seed)
                rows.append(_episode_row(metrics, case) | decisions)
        print(f"TEST_CONDITION_COMPLETE scenario={case.scenario_id}", flush=True)
    write_csv(root / "test" / "episodes.csv", rows)
    summary = []
    for case in cases:
        for method in TEST_METHODS:
            group = [row for row in rows if row["scenario_id"] == case.scenario_id
                     and row["policy"] == method]
            for metric in METRICS + ("wait_decisions", "move_decisions",
                                     "interrupted_decisions"):
                values = [float(row[metric]) for row in group if row[metric] is not None]
                summary.append({"scenario_id": case.scenario_id, "split": case.split,
                                "method": method, "metric": metric, "n": len(values),
                                "mean": statistics.mean(values) if values else None,
                                "stdev": statistics.stdev(values) if len(values) > 1 else None})
    write_csv(root / "test" / "summary.csv", summary)
    (root / "test" / "TEST_COMPLETE.txt").write_text(
        f"episodes={len(rows)}\nselection=ID-only\nppo_actions=argmax\n"
        f"methods={','.join(TEST_METHODS)}\n", encoding="utf-8")
    print("TEST_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-per-scenario", type=int, default=10)
    parser.add_argument("--pretrain-episodes", type=int, default=50)
    parser.add_argument("--aux-beta", type=float, default=0.1)
    parser.add_argument("--id-episodes", type=int, default=50)
    parser.add_argument("--ood-episodes", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.phase == "train":
        train(args.output_dir, args)
    else:
        test(args.output_dir, args)


if __name__ == "__main__":
    main()
