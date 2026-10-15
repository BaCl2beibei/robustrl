"""Log every selected C-hierarchical PPO decision on fresh ID and OOD episodes.

The saved joint action distribution is P(move) * P(target | move), plus P(wait).
Seeds use train_seed * 1_000_000 + 900_000 + episode_index, disjoint from
training, model selection, and the existing ID/OOD test episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from diagnose_id_policy_probabilities import decision_estimates, summarize, update_summary, write_csv
from ftm.config import SimulationConfig
from ftm.environment import MigrationEnvironment, scenario_by_name
from ftm.experiment_suite import build_evaluation_cases
from ftm.metrics import compute_episode_metrics
from ftm.observation import encode_observation
from run_ppo_abc import METHOD, SCENARIOS, load_selected


TRAIN_SEED = 1


def distribution(model, context, device: str) -> dict:
    state = torch.tensor(encode_observation(context), dtype=torch.float32,
                         device=device).unsqueeze(0)
    mask = torch.tensor(context.action_mask, dtype=torch.bool,
                        device=device).unsqueeze(0)
    legal = mask[0, :-1]
    with torch.no_grad():
        logits, _ = model(state, mask)
        joint = torch.softmax(logits.masked_fill(~mask, -1.0e9),
                              dim=-1).squeeze(0)
        action = int(logits.masked_fill(~mask, -1.0e9).argmax(dim=-1).item())
        raw_gate = torch.sigmoid(model.gate_logits(state))[0]
        if legal.any():
            target_logits = model.target(state)[0].masked_fill(~legal, -1.0e9)
            conditional = torch.softmax(target_logits, dim=-1)
            effective_gate = raw_gate
        else:
            conditional = torch.zeros_like(joint[:-1])
            effective_gate = torch.zeros_like(raw_gate)
    probs = joint.cpu().tolist()
    target_probs = conditional.cpu().tolist()
    gate = float(effective_gate.item())
    if (not math.isclose(sum(probs), 1.0, abs_tol=1e-6)
            or not math.isclose(sum(probs[:-1]), gate, abs_tol=1e-6)
            or any(not math.isclose(p, gate * q, abs_tol=1e-6)
                   for p, q in zip(probs[:-1], target_probs))):
        raise RuntimeError("hierarchical action probabilities are inconsistent")
    return {"probabilities": probs, "target_conditional": target_probs,
            "gate_move_raw": float(raw_gate.item()),
            "gate_move_effective": gate, "action": action}


def evaluate(run_root: Path, output_root: Path, id_episodes: int,
             ood_episodes: int, device: str) -> None:
    branch = run_root / METHOD
    if not (branch / "BRANCH_COMPLETE.txt").is_file():
        raise RuntimeError(f"training branch is incomplete: {branch}")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    selected = json.loads((branch / "selected.json").read_text(encoding="utf-8"))
    checkpoint = branch / selected["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = SimulationConfig()
    model = load_selected(run_root, config, device)
    model.eval()
    cases = build_evaluation_cases("all", "all", "all", config)
    cases = [case for case in cases if case.scenario_family in SCENARIOS
             and (case.split == "ood" or case.scenario_id in SCENARIOS)]
    if len(cases) != 15:
        raise RuntimeError(f"expected 5 ID and 10 OOD cases; got {len(cases)}")

    probe = MigrationEnvironment(config, scenario_by_name("normal", config))
    probe.reset(1)
    fields = (["train_seed", "split", "scenario_id", "scenario", "method",
               "episode", "env_seed", "decision_index", "time_seconds",
               "action_mask", "argmax_action", "argmax_wait", "p_wait", "p_move",
               "p_best_target", "p_gate_move_raw", "p_gate_move_effective",
               "p_target_cond_best"]
              + [f"p_node{i}" for i in range(config.n_nodes)]
              + [f"p_target_cond_node{i}" for i in range(config.n_nodes)]
              + list(decision_estimates(probe.get_decision_context())))
    output_root.mkdir(parents=True)
    totals = defaultdict(lambda: defaultdict(float))
    gate_totals = defaultdict(lambda: defaultdict(float))
    episode_rows = []
    decision_count = 0
    with (output_root / "decisions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            count = id_episodes if case.split == "id" else ood_episodes
            for episode in range(count):
                env_seed = TRAIN_SEED * 1_000_000 + 900_000 + episode
                env = MigrationEnvironment(config, case.scenario)
                env.reset(env_seed)
                total_reward = 0.0
                decision_index = 0
                while True:
                    context = env.get_decision_context()
                    result = distribution(model, context, device)
                    probs = result["probabilities"]
                    target_conditional = result["target_conditional"]
                    action = result["action"]
                    estimates = decision_estimates(context)
                    row = {
                        "train_seed": TRAIN_SEED, "split": case.split,
                        "scenario_id": case.scenario_id,
                        "scenario": case.scenario_id, "method": METHOD,
                        "episode": episode + 1, "env_seed": env_seed,
                        "decision_index": decision_index,
                        "time_seconds": context.time_seconds,
                        "action_mask": json.dumps(context.action_mask),
                        "argmax_action": action,
                        "argmax_wait": action == config.n_nodes,
                        "p_wait": probs[-1], "p_move": sum(probs[:-1]),
                        "p_best_target": max(probs[:-1]),
                        "p_gate_move_raw": result["gate_move_raw"],
                        "p_gate_move_effective": result["gate_move_effective"],
                        "p_target_cond_best": max(target_conditional),
                        **{f"p_node{i}": p for i, p in enumerate(probs[:-1])},
                        **{f"p_target_cond_node{i}": p
                           for i, p in enumerate(target_conditional)},
                        **estimates,
                    }
                    writer.writerow(row)
                    update_summary(totals, row)
                    if estimates["legal_target_count"]:
                        key = (TRAIN_SEED, case.scenario_id, METHOD)
                        gate_totals[key]["gate"] += row["p_gate_move_effective"]
                        gate_totals[key]["conditional_best"] += row["p_target_cond_best"]
                    decision_count += 1
                    _, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    decision_index += 1
                    if terminated or truncated:
                        break
                metrics = compute_episode_metrics(
                    env, METHOD, env_seed, total_reward).to_dict()
                episode_rows.append({"split": case.split,
                                     "scenario_id": case.scenario_id,
                                     "episode": episode + 1, **metrics})
            print(f"DIAGNOSTIC_CONDITION_COMPLETE split={case.split} "
                  f"scenario={case.scenario_id} episodes={count}", flush=True)

    summary = summarize(totals)
    if len(summary) != len(cases):
        raise RuntimeError("diagnostic summary is incomplete")
    for row in summary:
        row["scenario_id"] = row["scenario"]
        row["split"] = "id" if row["scenario"] in SCENARIOS else "ood"
        key = (TRAIN_SEED, row["scenario"], METHOD)
        count = row["actionable_count"]
        row["p_gate_move_mean_actionable"] = (
            gate_totals[key]["gate"] / count if count else None)
        row["p_target_cond_best_mean_actionable"] = (
            gate_totals[key]["conditional_best"] / count if count else None)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "episodes.csv", episode_rows)
    (output_root / "definitions.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()), "selection": selected,
        "seed_rule": "train_seed * 1000000 + 900000 + episode_index",
        "id_episodes_per_condition": id_episodes,
        "ood_episodes_per_condition": ood_episodes,
        "action_selection": "joint-probability argmax, as in run_ppo_abc.py",
        "p_gate_move_raw": "sigmoid(gate logit) before legal-target handling",
        "p_gate_move_effective": "gate probability, zero if no legal target",
        "p_target_cond_node": "legal-target conditional softmax; zero if no legal target",
        "p_node": "joint probability = effective gate * target conditional probability",
        "p_move": "sum of p_node; equal to effective gate",
        "p_best_target": "max joint p_node among legal targets",
        "opportunity_definition": "shared with diagnose_id_policy_probabilities.py; "
                                  "public-snapshot heuristic, not proven benefit",
    }, indent=2), encoding="utf-8")
    (output_root / "DIAGNOSTIC_COMPLETE.txt").write_text(
        f"decisions={decision_count}\nepisodes={len(episode_rows)}\n",
        encoding="utf-8")
    print(f"DIAGNOSTIC_COMPLETE decisions={decision_count} "
          f"episodes={len(episode_rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("runs/ppo_abc_seed1_3000_20261014"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--id-episodes", type=int, default=20)
    parser.add_argument("--ood-episodes", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.id_episodes, args.ood_episodes) < 1:
        parser.error("episode counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    root = args.run_root.resolve()
    output = (args.output_root or root / METHOD / "diagnostics_id_ood").resolve()
    evaluate(root, output, args.id_episodes, args.ood_episodes, args.device)


if __name__ == "__main__":
    main()
