"""Command line entry point for simulations and the robust-MDP toy example."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from .config import SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .metrics import compute_episode_metrics
from .policies import (GreedyPolicy, LoadBasedPolicy, NoMigrationPolicy, RandomPolicy,
                       RobustHeuristicPolicy)
from .robust_mdp import two_state_example


POLICIES = ("all", "no_migration", "random", "greedy", "load_based", "robust_heuristic")
SCENARIOS = ("normal", "failure", "network", "load", "mixed")


def make_policy(name: str, seed: int):
    if name == "no_migration":
        return NoMigrationPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "greedy":
        return GreedyPolicy()
    if name == "load_based":
        return LoadBasedPolicy()
    if name == "robust_heuristic":
        return RobustHeuristicPolicy()
    raise ValueError(f"unknown policy: {name}")


def run_episode(policy_name: str, scenario_name: str, seed: int,
                config: SimulationConfig):
    scenario = scenario_by_name(scenario_name, config)
    env = MigrationEnvironment(config, scenario)
    env.reset(seed)
    policy = make_policy(policy_name, seed + 10_000)
    total_reward = 0.0
    while True:
        action = policy.act(env)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return compute_episode_metrics(env, policy_name, seed, total_reward), env


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def run_robust_demo() -> None:
    mdp = two_state_example()
    nominal = mdp.value_iteration(gamma=0.9, robust=False)
    robust = mdp.value_iteration(gamma=0.9, robust=True)
    print("Nominal policy:", nominal.policy)
    print("Robust policy: ", robust.policy)
    print("Nominal values:", nominal.values)
    print("Robust values: ", robust.values)
    print("Note: the inner minimum is exact over the finite rows in this example.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("episodes", "robust-demo"), default="episodes")
    parser.add_argument("--policy", choices=POLICIES, default="robust_heuristic")
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()

    if args.mode == "robust-demo":
        run_robust_demo()
        return
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    config = SimulationConfig(n_nodes=args.nodes, n_tasks=args.tasks,
                              k_required=args.k, max_time_seconds=args.max_time)
    rows = []
    selected_policies = POLICIES[1:] if args.policy == "all" else (args.policy,)
    for episode in range(args.episodes):
        seed = args.seed + episode
        for policy_name in selected_policies:
            metrics, _ = run_episode(policy_name, args.scenario, seed, config)
            rows.append(metrics.to_dict())
            print(f"policy={policy_name} episode={episode + 1}/{args.episodes} seed={seed} "
                  f"completion={metrics.task_completion_rate:.3f} "
                  f"continuation={_fmt(metrics.continuation_success_rate)} "
                  f"availability={metrics.cluster_availability:.3f} "
                  f"migrations={metrics.migration_count}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"{args.scenario}_{args.policy}_{args.seed}.csv"
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for policy_name in selected_policies:
        group = [row for row in rows if row["policy"] == policy_name]
        completion = statistics.mean(row["task_completion_rate"] for row in group)
        print(f"{policy_name} mean task completion rate: {completion:.3f}")
    print(f"episode metrics: {output_file.resolve()}")


if __name__ == "__main__":
    main()
