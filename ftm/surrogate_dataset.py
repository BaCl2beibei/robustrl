"""Collect transition/value records for differentiable surrogate fitting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .main import POLICIES, SCENARIOS, make_policy
from .surrogate import perturb_scenario, scenario_to_raw


def collect(output: Path, config: SimulationConfig, episodes_per_scenario: int,
            scenario_names: tuple[str, ...], policy_names: tuple[str, ...],
            seed: int, parameter_noise: float = 0.25) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for scenario_index, scenario_name in enumerate(scenario_names):
            base_scenario = scenario_by_name(scenario_name, config)
            for episode in range(episodes_per_scenario):
                run_seed = seed + scenario_index * episodes_per_scenario + episode
                rng = random.Random(run_seed + 70_000)
                scenario = perturb_scenario(base_scenario, rng, parameter_noise)
                policy_name = policy_names[episode % len(policy_names)]
                env = MigrationEnvironment(config, scenario)
                env.reset(run_seed)
                policy = make_policy(policy_name, run_seed + 30_000)
                episode_records = []
                while True:
                    state = env.observation()
                    action = policy.act(env)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_records.append({
                        "state": list(state),
                        "action": int(action),
                        "parameters": list(scenario_to_raw(scenario)),
                        "next_state": list(next_state),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "return_to_go": 0.0,
                        "scenario": scenario_name,
                        "policy": policy_name,
                        "seed": run_seed,
                    })
                    if terminated or truncated:
                        break
                return_to_go = 0.0
                for record in reversed(episode_records):
                    return_to_go = float(record["reward"]) + 0.99 * return_to_go
                    record["return_to_go"] = return_to_go
                for record in episode_records:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += len(episode_records)
                print(f"scenario={scenario_name} episode={episode + 1}/"
                      f"{episodes_per_scenario} transitions={len(episode_records)}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/surrogate_transitions.jsonl"))
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all")
    parser.add_argument("--policy", choices=("all",) + POLICIES[1:], default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--parameter-noise", type=float, default=0.25)
    args = parser.parse_args()
    if args.episodes_per_scenario < 1 or args.parameter_noise < 0:
        parser.error("episode count must be positive and parameter-noise non-negative")
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    policies = POLICIES[1:] if args.policy == "all" else (args.policy,)
    count = collect(args.output,
                    SimulationConfig(max_time_seconds=args.max_time),
                    args.episodes_per_scenario, scenarios, policies,
                    args.seed, args.parameter_noise)
    print(f"collected {count} transitions in {args.output.resolve()}")


if __name__ == "__main__":
    main()
