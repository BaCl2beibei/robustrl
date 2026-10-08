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
from .robust_ppo import PPOConfig
from .surrogate import estimate_sensitivity_ellipsoid, fit_surrogate, fit_value_model
from .surrogate_robust_ppo import SurrogatePGDRobustPPO


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
    parser.add_argument("--output", type=Path, default=Path("models/surrogate_robust_ppo.pt"))
    parser.add_argument("--log", type=Path, default=Path("runs/surrogate_robust_ppo.csv"))
    parser.add_argument("--adversary-log", type=Path,
                        default=Path("runs/surrogate_pgd_summary.json"))
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = parser.parse_args()
    if args.episodes < 1 or args.alpha < 0 or args.surrogate_epochs < 1 \
            or args.value_epochs < 1 or args.pgd_steps < 1:
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
    config = SimulationConfig(max_time_seconds=args.max_time)
    probe_env = MigrationEnvironment(config, scenario_by_name("normal", config))
    state_dim = len(probe_env.reset(args.seed)[0])
    action_dim = config.n_nodes + 1
    surrogate = fit_surrogate(records, action_dim, epochs=args.surrogate_epochs, device=device)
    value_model = fit_value_model(records, action_dim, epochs=args.value_epochs, device=device)
    center, axes, _, gradient = estimate_sensitivity_ellipsoid(
        records, surrogate, value_model, action_dim, alpha=args.alpha, device=device
    )
    trainer = SurrogatePGDRobustPPO(
        surrogate, center, axes, args.alpha, env_config=config, ppo_config=PPOConfig(),
        device=device, seed=args.seed, pgd_steps=args.pgd_steps,
    )
    results = trainer.train(args.episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": trainer.model.state_dict(),
        "surrogate_state_dict": surrogate.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pgd_steps": args.pgd_steps,
        "center": center, "axes": axes, "alpha": args.alpha,
        "sensitivity_gradient": gradient,
        "env_config": asdict(config), "ppo_config": asdict(trainer.ppo_config),
        "seed": args.seed, "episodes": args.episodes,
        "surrogate_dataset": str(args.dataset),
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
