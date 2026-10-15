"""Record selected PPO/PGD-PPO action distributions on fresh ID episodes.

Usage: python scripts/diagnose_id_policy_probabilities.py
The seed range (train_seed * 1_000_000 + 900_000 + episode_index) is disjoint
from training, checkpoint selection, and the existing ID/OOD tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ftm.config import MonitoringConfig, SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.metrics import compute_episode_metrics
from ftm.observation import encode_observation
from ftm.policies import estimate_completion_time, estimate_wait_time
from ftm.trained_policy import TrainedPolicy
from run_ood_pgd_experiment import SCENARIOS


METHODS = ("ppo", "pgd_ppo")
RISK_PROBABILITY = 0.05
RISK_REDUCTION = 0.02
TIME_GAIN_FRACTION = 0.05
LARGE_TIME_GAIN_FRACTION = 0.15
MIN_TIME_GAIN_SECONDS = 1.0
SLA_SLACK_SECONDS = 10.0
SLA_SLACK_FRACTION = 0.10


def decision_estimates(context) -> dict:
    """Use only the public snapshot available to the actor at this instant."""
    task = context.get_focus_task()
    targets = [i for i, legal in enumerate(context.action_mask[:-1]) if legal]
    result = {
        "focus_task_id": context.focus_task_id,
        "task_status": None if task is None else task.status,
        "source_node": None if task is None else task.node_id,
        "legal_target_count": len(targets),
        "stay_time_seconds": None,
        "wait_time_seconds": None,
        "recovery_possible": None,
        "best_target": None,
        "best_move_time_seconds": None,
        "time_saving_seconds": None,
        "sla_slack_seconds": None,
        "source_failure_probability": None,
        "target_failure_probability": None,
        "time_favorable": False,
        "high_risk": False,
        "low_sla_slack": False,
        "urgent": False,
        "worthwhile_opportunity": False,
        "urgent_time_favorable": False,
    }
    if task is None or task.status not in ("running", "interrupted"):
        return result
    stay = estimate_wait_time(context, task)
    recovery_possible = math.isfinite(stay)
    result["recovery_possible"] = recovery_possible
    if task.status == "interrupted" and recovery_possible:
        result["wait_time_seconds"] = stay
    if not targets:
        return result
    move, target = min((estimate_completion_time(context, task, i), i)
                       for i in targets)
    result["best_target"] = target
    result["best_move_time_seconds"] = move
    if not recovery_possible:
        return result
    saving = stay - move
    slack = task.deadline_seconds - context.time_seconds - stay
    source_rate = max(0.0, context.nodes[task.node_id].failure_rate_estimate)
    target_rate = max(0.0, context.nodes[target].failure_rate_estimate)
    source_risk = (-math.expm1(-source_rate * stay)
                   if task.status == "running" else None)
    target_risk = -math.expm1(-target_rate * move)
    favorable = saving >= max(MIN_TIME_GAIN_SECONDS, TIME_GAIN_FRACTION * stay)
    high_risk = source_risk is not None and source_risk >= RISK_PROBABILITY
    low_slack = slack <= max(SLA_SLACK_SECONDS, SLA_SLACK_FRACTION * stay)
    urgent = high_risk or low_slack
    result.update({
        "stay_time_seconds": stay,
        "best_target": target,
        "best_move_time_seconds": move,
        "time_saving_seconds": saving,
        "sla_slack_seconds": slack,
        "source_failure_probability": source_risk,
        "target_failure_probability": target_risk,
        "time_favorable": favorable,
        "high_risk": high_risk,
        "low_sla_slack": low_slack,
        "urgent": urgent,
        "worthwhile_opportunity": (favorable and (low_slack
                                  or (source_risk is not None
                                      and source_risk - target_risk >= RISK_REDUCTION)
                                  or saving >= LARGE_TIME_GAIN_FRACTION * stay)),
        "urgent_time_favorable": urgent and favorable,
    })
    return result


def action_distribution(controller: TrainedPolicy, context) -> tuple[list[float], int]:
    state = torch.tensor(encode_observation(context), dtype=torch.float32,
                         device=controller.device).unsqueeze(0)
    mask = torch.tensor(context.action_mask, dtype=torch.bool,
                        device=controller.device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = controller.model(state, mask)
        masked = logits.masked_fill(~mask, -1.0e9)
        probabilities = torch.softmax(masked, dim=-1).squeeze(0).cpu().tolist()
        action = int(masked.argmax(dim=-1).item())
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise RuntimeError("masked action probabilities do not sum to one")
    return probabilities, action


def update_summary(totals: dict, row: dict) -> None:
    key = (row["train_seed"], row["scenario"], row["method"])
    item = totals[key]
    item["decisions"] += 1
    for field in ("p_wait", "p_move", "argmax_wait"):
        item[field] += row[field]
    if row["legal_target_count"]:
        item["p_best_target"] += row["p_best_target"]
    for label, present in (("actionable", row["legal_target_count"] > 0),
                           ("time_favorable", row["time_favorable"]),
                           ("urgent", row["urgent"]),
                           ("worthwhile_opportunity", row["worthwhile_opportunity"]),
                           ("urgent_time_favorable", row["urgent_time_favorable"])):
        if present:
            item[f"{label}_count"] += 1
            for field in ("p_wait", "p_move", "argmax_wait"):
                item[f"{label}_{field}"] += row[field]


def summarize(totals: dict) -> list[dict]:
    rows = []
    for (train_seed, scenario, method), item in totals.items():
        decisions = item["decisions"]
        actionable = item["actionable_count"]
        row = {"train_seed": train_seed, "scenario": scenario, "method": method,
               "decisions": decisions, "actionable_decisions": actionable,
               "p_wait_mean": item["p_wait"] / decisions,
               "p_move_mean": item["p_move"] / decisions,
               "best_target_probability_mean": (item["p_best_target"] / actionable
                                                if actionable else None),
               "argmax_wait_rate_all": item["argmax_wait"] / decisions}
        for label in ("actionable", "time_favorable", "urgent",
                      "worthwhile_opportunity", "urgent_time_favorable"):
            count = item[f"{label}_count"]
            row[f"{label}_count"] = count
            row[f"argmax_wait_rate_{label}"] = (item[f"{label}_argmax_wait"] / count
                                                if count else None)
            row[f"p_wait_mean_{label}"] = (item[f"{label}_p_wait"] / count
                                          if count else None)
            row[f"p_move_mean_{label}"] = (item[f"{label}_p_move"] / count
                                          if count else None)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(source_root: Path, output_root: Path, episodes: int, device: str) -> None:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    prepared = []
    for train_seed in (1, 2, 3):
        run = source_root / f"main_seed{train_seed}"
        if not (run / "TRAINING_COMPLETE.txt").is_file():
            raise RuntimeError(f"training is incomplete: {run}")
        source = json.loads((run / "config.json").read_text(encoding="utf-8"))
        values = source["simulation_config"]
        config = SimulationConfig(**(values | {
            "monitoring": MonitoringConfig(**values["monitoring"])}))
        probe = MigrationEnvironment(config, scenario_by_name("normal", config))
        state, _ = probe.reset(train_seed)
        controllers = {}
        for method in METHODS:
            selected = json.loads((run / method / "selected.json").read_text(
                encoding="utf-8"))
            checkpoint = run / selected["checkpoint"].replace("\\", "/")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            controllers[method] = TrainedPolicy(
                checkpoint, len(state), config.n_nodes + 1, device)
        prepared.append((train_seed, config, controllers))

    output_root.mkdir(parents=True)
    detail_path = output_root / "decisions.csv"
    episode_rows = []
    totals = defaultdict(lambda: defaultdict(float))
    decision_count = 0
    field_probe = MigrationEnvironment(prepared[0][1],
                                       scenario_by_name("normal", prepared[0][1]))
    field_probe.reset(1)
    fields = (["train_seed", "scenario", "method", "episode", "env_seed",
               "decision_index", "time_seconds", "action_mask", "argmax_action",
               "argmax_wait", "p_wait", "p_move", "p_best_target"]
              + [f"p_node{i}" for i in range(prepared[0][1].n_nodes)]
              + list(decision_estimates(field_probe.get_decision_context()).keys()))
    for train_seed, config, controllers in prepared:
        if config.n_nodes != prepared[0][1].n_nodes:
            raise RuntimeError("node counts differ across training seeds")
    with detail_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for train_seed, config, controllers in prepared:
            for scenario in SCENARIOS:
                for episode in range(episodes):
                    env_seed = train_seed * 1_000_000 + 900_000 + episode
                    for method, controller in controllers.items():
                        env = MigrationEnvironment(config, scenario_by_name(scenario, config))
                        env.reset(env_seed)
                        total_reward = 0.0
                        decision_index = 0
                        while True:
                            context = env.get_decision_context()
                            probs, action = action_distribution(controller, context)
                            estimates = decision_estimates(context)
                            node_probs = probs[:-1]
                            row = {
                                "train_seed": train_seed, "scenario": scenario,
                                "method": method, "episode": episode + 1,
                                "env_seed": env_seed, "decision_index": decision_index,
                                "time_seconds": context.time_seconds,
                                "action_mask": json.dumps(context.action_mask),
                                "argmax_action": action,
                                "argmax_wait": action == config.n_nodes,
                                "p_wait": probs[-1], "p_move": sum(node_probs),
                                "p_best_target": max(node_probs),
                                **{f"p_node{i}": p for i, p in enumerate(node_probs)},
                                **estimates,
                            }
                            writer.writerow(row)
                            update_summary(totals, row)
                            decision_count += 1
                            _, reward, terminated, truncated, _ = env.step(action)
                            total_reward += reward
                            decision_index += 1
                            if terminated or truncated:
                                break
                        metrics = compute_episode_metrics(
                            env, method, env_seed, total_reward).to_dict()
                        episode_rows.append({"train_seed": train_seed,
                                             "scenario": scenario, "method": method,
                                             "episode": episode + 1, **metrics})
                print(f"ID_DIAGNOSTIC_CONDITION_COMPLETE seed={train_seed} "
                      f"scenario={scenario}", flush=True)
    expected_groups = len(prepared) * len(SCENARIOS) * len(METHODS)
    summary = summarize(totals)
    if len(summary) != expected_groups:
        raise RuntimeError("diagnostic summary is incomplete")
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "episodes.csv", episode_rows)
    (output_root / "definitions.json").write_text(json.dumps({
        "seed_rule": "train_seed * 1000000 + 900000 + episode_index",
        "episodes_per_scenario_method_seed": episodes,
        "action_selection": "masked-logit argmax; probabilities saved before action",
        "probability_definition": "softmax over legal actions; p_move=sum(p_node*); "
                                  "p_best_target=max(p_node*)",
        "time_favorable": "running or recoverable interrupted task with legal target; "
                          "stay_time - best_move_time "
                          ">= max(1 second, 0.05 * stay_time)",
        "wait_time": "interrupted task: estimated node repair wait plus restore delay "
                     "plus remaining execution on source; null if recovery is impossible",
        "high_risk": "estimated source-node failure probability during stay_time >= 0.05",
        "low_sla_slack": "deadline - now - stay_time <= max(10 seconds, 0.10 * stay_time)",
        "urgent": "high_risk or low_sla_slack",
        "worthwhile_opportunity": "time_favorable and (low_sla_slack or "
                                 "source_risk - target_risk >= 0.02 or "
                                 "time_saving >= 0.15 * stay_time)",
        "caveat": "All opportunity labels are public-snapshot heuristics, not proven "
                  "benefit; verify with paired counterfactual simulation.",
        "source_root": str(source_root),
        "simulation_configs": {str(seed): asdict(config)
                               for seed, config, _ in prepared},
    }, indent=2), encoding="utf-8")
    (output_root / "DIAGNOSTIC_COMPLETE.txt").write_text(
        f"decisions={decision_count}\nepisodes={len(episode_rows)}\n",
        encoding="utf-8")
    print(f"ID_DIAGNOSTIC_COMPLETE decisions={decision_count} "
          f"episodes={len(episode_rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("runs/ood_pgd_short_aux6_20261013"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("runs/ood_pgd_short_aux6_20261013/id_policy_diagnostics"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    evaluate(args.source_root.resolve(), args.output_root.resolve(),
             args.episodes, args.device)


if __name__ == "__main__":
    main()
