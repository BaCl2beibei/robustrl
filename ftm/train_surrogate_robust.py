"""Fit a transition surrogate and run PPO with per-step surrogate PGD."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from .config import SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .ood_scenarios import scenario_family
from .observation import FEATURE_ORDER, FEATURE_SCALES, SCHEMA_VERSION
from .robust_ppo import PPOConfig
from .surrogate import (PARAMETER_NAMES, estimate_sensitivity_ellipsoid,
                        fit_surrogate, fit_value_model, scenario_to_raw,
                        training_raw_bounds)
from .surrogate_robust_ppo import DEFAULT_PGD_SCENARIOS, SurrogatePGDRobustPPO


def read_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not all(key in record for key in
                           ("state", "action", "parameters", "next_state", "reward",
                            "return_to_go")):
                    raise ValueError("missing required fields")
                records.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    if len(records) < 2:
        raise ValueError("dataset must contain at least two valid transitions")
    return records


def build_scenario_pgd_constraints(records: list[dict], surrogate, value_model,
                                   action_dim: int, config: SimulationConfig,
                                   scenario_names: tuple[str, ...], alpha: float,
                                   device: str) -> dict[str, dict]:
    if any(len(record["parameters"]) != len(PARAMETER_NAMES) for record in records):
        raise ValueError("training records must have exactly eight disturbance parameters")

    constraints_by_family = {}
    required_families = {scenario_family(name) for name in scenario_names}
    for family in sorted(required_families):
        family_records = []
        for record in records:
            name = record.get("scenario")
            if not isinstance(name, str):
                continue
            try:
                record_family = scenario_family(name)
            except ValueError:
                continue
            if record_family == family:
                family_records.append(record)
        if len(family_records) < 2:
            raise ValueError(
                f"need at least two transition records for PGD family {family!r}"
            )

        template = scenario_by_name(family, config)
        center = scenario_to_raw(template)
        _, axes, _, gradient = estimate_sensitivity_ellipsoid(
            family_records, surrogate, value_model, action_dim,
            alpha=alpha, device=device, max_records=512,
            center_override=center,
        )
        lower, upper = training_raw_bounds(template)
        constraints_by_family[family] = {
            "center": center,
            "axes": axes,
            "alpha": alpha,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "sensitivity_gradient": gradient,
            "parameter_names": PARAMETER_NAMES,
        }
    return {
        name: constraints_by_family[scenario_family(name)]
        for name in scenario_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/surrogate_transitions.jsonl"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--surrogate-epochs", type=int, default=40)
    parser.add_argument("--value-epochs", type=int, default=40)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--warmup-fraction", type=float, default=0.2,
                        help="fraction of episodes trained with ordinary PPO before PGD")
    parser.add_argument("--pgd-step-fraction", type=float, default=0.2,
                        help="maximum per-coordinate change as a fraction of its raw range")
    parser.add_argument("--surrogate-refresh-interval", type=int, default=10,
                        help="refresh the surrogate after this many PPO updates")
    parser.add_argument("--surrogate-refresh-epochs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("models/surrogate_robust_ppo.pt"))
    parser.add_argument("--log", type=Path, default=Path("runs/surrogate_robust_ppo.csv"))
    parser.add_argument("--adversary-log", type=Path,
                        default=Path("runs/surrogate_pgd_summary.json"))
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = parser.parse_args()
    if (args.episodes < 1 or args.alpha < 0 or args.surrogate_epochs < 1
            or args.value_epochs < 1 or args.pgd_steps < 1
            or not 0.0 <= args.warmup_fraction < 1.0
            or not 0.0 < args.pgd_step_fraction <= 1.0
            or args.surrogate_refresh_interval < 1
            or args.surrogate_refresh_epochs < 1):
        parser.error("episodes/epochs/PGD steps must be positive and alpha non-negative")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("This command needs the optional learning extra: "
                         "python -m pip install -e '.[learning]'") from exc
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = read_records(args.dataset)
    if {len(record["parameters"]) for record in records} != {len(PARAMETER_NAMES)}:
        raise ValueError("surrogate dataset must contain exactly eight parameters per record")
    if any(record.get("observation_schema_version") != SCHEMA_VERSION
           for record in records):
        raise ValueError("surrogate dataset observation schema is incompatible; regenerate it")
    config = SimulationConfig(max_time_seconds=args.max_time)
    probe_env = MigrationEnvironment(config, scenario_by_name("normal", config))
    state_dim = len(probe_env.reset(args.seed)[0])
    action_dim = config.n_nodes + 1
    if any(len(record["state"]) != state_dim
           or len(record["next_state"]) != state_dim
           or len(record.get("action_mask", ())) != action_dim
           or not 0 <= record["action"] < action_dim
           or not record["action_mask"][record["action"]]
           for record in records):
        raise ValueError("surrogate dataset state, action or mask is incompatible")
    surrogate = fit_surrogate(records, action_dim, epochs=args.surrogate_epochs, device=device)
    value_model = fit_value_model(records, action_dim, epochs=args.value_epochs, device=device)
    center, axes, _, gradient = estimate_sensitivity_ellipsoid(
        records, surrogate, value_model, action_dim, alpha=args.alpha, device=device
    )
    scenario_constraints = build_scenario_pgd_constraints(
        records, surrogate, value_model, action_dim, config,
        DEFAULT_PGD_SCENARIOS, args.alpha, device,
    )
    warmup_episodes = 0
    if args.episodes > 1 and args.warmup_fraction > 0:
        warmup_episodes = min(
            args.episodes - 1,
            max(1, round(args.episodes * args.warmup_fraction)),
        )
    trainer = SurrogatePGDRobustPPO(
        surrogate, center, axes, args.alpha, env_config=config, ppo_config=PPOConfig(),
        device=device, seed=args.seed, pgd_steps=args.pgd_steps,
        scenario_constraints=scenario_constraints,
        warmup_episodes=warmup_episodes,
        max_pgd_step_fraction=args.pgd_step_fraction,
        surrogate_training_records=records,
        surrogate_refresh_interval=args.surrogate_refresh_interval,
        surrogate_refresh_epochs=args.surrogate_refresh_epochs,
    )
    results = trainer.train(args.episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": trainer.model.state_dict(),
        "observation_schema_version": SCHEMA_VERSION,
        "feature_order": FEATURE_ORDER,
        "feature_scales": FEATURE_SCALES,
        "surrogate_state_dict": trainer.surrogate.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pgd_steps": args.pgd_steps,
        "center": center, "axes": axes, "alpha": args.alpha,
        "parameter_names": PARAMETER_NAMES,
        "scenario_pgd_constraints": scenario_constraints,
        "sensitivity_gradient": gradient,
        "env_config": asdict(config), "ppo_config": asdict(trainer.ppo_config),
        "seed": args.seed, "episodes": args.episodes,
        "surrogate_dataset": str(args.dataset),
        "warmup_episodes": trainer.warmup_episodes,
        "pgd_episode_fraction_after_warmup": 0.5,
        "max_pgd_step_fraction": args.pgd_step_fraction,
        "surrogate_refresh_interval": args.surrogate_refresh_interval,
        "surrogate_refresh_count": trainer.surrogate_refresh_count,
    }, args.output)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    args.adversary_log.parent.mkdir(parents=True, exist_ok=True)
    args.adversary_log.write_text(json.dumps(trainer.adversary_summary, indent=2),
                                  encoding="utf-8")
    print(f"model saved: {args.output.resolve()}")
    print(f"training log saved: {args.log.resolve()}")
    print(f"PGD summary saved: {args.adversary_log.resolve()}")


if __name__ == "__main__":
    main()
