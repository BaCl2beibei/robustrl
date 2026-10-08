"""Train the optional finite-scenario robust PPO policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from .config import SimulationConfig
from .main import SCENARIOS
from .robust_ppo import FiniteScenarioRobustPPO, PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all",
                        help="all enables finite-scenario adversarial PPO; one value trains a fixed-scenario PPO")
    parser.add_argument("--output", type=Path, default=Path("models/robust_ppo.pt"))
    parser.add_argument("--log", type=Path, default=Path("runs/robust_ppo_training.csv"))
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = SimulationConfig(max_time_seconds=args.max_time)
    scenario_names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    trainer = FiniteScenarioRobustPPO(config, PPOConfig(),
                                      scenario_names=scenario_names,
                                      device=args.device,
                                      seed=args.seed)
    results = trainer.train(args.episodes, model_path=args.output)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(f"model saved: {args.output.resolve()}")
    print(f"training log saved: {args.log.resolve()}")


if __name__ == "__main__":
    main()
