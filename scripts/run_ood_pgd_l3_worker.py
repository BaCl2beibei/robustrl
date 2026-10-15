"""Evaluate one selected PPO/PGD-PPO seed and both baselines on fresh OOD L3 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ftm.config import MonitoringConfig, SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import _episode_row, build_evaluation_cases
from ftm.main import run_episode
from ftm.trained_policy import TrainedPolicy, run_trained_episode

METHODS = ("no_migration", "time_min", "ppo", "pgd_ppo")


def evaluate(source_root: Path, output_root: Path, seed: int,
             episodes: int | None, device: str) -> None:
    run = source_root / f"main_seed{seed}"
    if not (run / "TEST_COMPLETE.txt").is_file():
        raise RuntimeError(f"source run is incomplete: {run}")
    source = json.loads((run / "config.json").read_text(encoding="utf-8"))
    values = source["simulation_config"]
    config = SimulationConfig(**(values | {
        "monitoring": MonitoringConfig(**values["monitoring"]),
    }))
    count = source["args"]["ood_episodes"] if episodes is None else episodes
    if count < 1:
        raise ValueError("episodes per scenario must be positive")
    cases = build_evaluation_cases("ood", "l3", "all", config)
    if len(cases) != 5:
        raise RuntimeError("expected five L3 scenario families")
    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    state, _ = probe.reset(1)
    controllers, selections = {}, {}
    for method in ("ppo", "pgd_ppo"):
        selected = json.loads((run / method / "selected.json").read_text(
            encoding="utf-8"))
        checkpoint = run / selected["checkpoint"].replace("\\", "/")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        controller = TrainedPolicy(checkpoint, len(state), config.n_nodes + 1,
                                   device=device)
        if controller.model.__class__.__name__ != "HierarchicalActorCritic":
            raise ValueError(f"selected model is not the C actor: {checkpoint}")
        controllers[method] = controller
        selections[method] = selected | {"resolved_checkpoint": str(checkpoint.resolve())}

    output = output_root / f"main_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        for index in range(count):
            test_seed = seed * 1_000_000 + 900_000 + index
            for method in METHODS:
                if method in controllers:
                    metrics, _ = run_trained_episode(
                        method, case.scenario, test_seed, config, controllers[method])
                else:
                    metrics, _ = run_episode(method, case.scenario, test_seed, config)
                row = _episode_row(metrics, case)
                row["train_seed"] = seed
                row["test_episode"] = index + 1
                row["action_selection"] = ("argmax" if method in controllers
                                           else "baseline")
                rows.append(row)
        print(f"L3_CONDITION_COMPLETE seed={seed} scenario={case.scenario_id}",
              flush=True)
    with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "selected_models.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "source_run": str(run.resolve()), "train_seed": seed,
        "ood_level": "l3", "episodes_per_scenario_per_policy": count,
        "test_seed_rule": "train_seed * 1000000 + 900000 + episode_index",
        "action_selection": "masked argmax for PPO and PGD-PPO",
        "scenarios": {case.scenario_id: asdict(case.scenario) for case in cases},
        "simulation_config": values,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "L3_TEST_COMPLETE.txt").write_text(
        f"seed={seed}\nepisodes={len(rows)}\n", encoding="utf-8")
    print(f"L3_TEST_COMPLETE seed={seed} episodes={len(rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--episodes-per-scenario", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    evaluate(args.source_root.resolve(), args.output_root.resolve(), args.seed,
             args.episodes_per_scenario, args.device)
