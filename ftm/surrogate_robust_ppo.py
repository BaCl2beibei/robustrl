"""PPO with per-decision PGD against a differentiable transition surrogate.

PGD chooses legal scenario parameters from the learned surrogate's one-step
Bellman objective. The actual environment then samples hard CTMC/network/load
events under those parameters. This connects PGD to PPO training while keeping
the discrete simulator authoritative for collected rewards and transitions.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .config import ScenarioConfig, SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .robust_ppo import EpisodeResult, FiniteScenarioRobustPPO, PPOConfig
from .surrogate import (PARAMETER_NAMES, scenario_to_raw, training_raw_bounds,
                        worst_case_scenario)


DEFAULT_PGD_SCENARIOS = ("normal", "failure", "network", "load_short",
                         "load_medium", "load_sustained", "mixed")


class SurrogatePGDRobustPPO(FiniteScenarioRobustPPO):
    def __init__(self, surrogate, center: Sequence[float], axes: Sequence[float],
                 alpha: float, env_config: SimulationConfig | None = None,
                 ppo_config: PPOConfig | None = None,
                 scenario_names: tuple[str, ...] = DEFAULT_PGD_SCENARIOS,
                 device: str | None = None, seed: int = 2026,
                 pgd_steps: int = 8, pgd_learning_rate: float = 0.05,
                 scenario_constraints: Mapping[str, Mapping] | None = None) -> None:
        super().__init__(env_config, ppo_config, scenario_names, device, seed)
        if len(center) != len(axes) or len(center) not in (6, len(PARAMETER_NAMES)):
            raise ValueError("PGD center and axes must match the simulator parameters")
        if alpha < 0 or pgd_steps < 1 or pgd_learning_rate <= 0:
            raise ValueError("invalid PGD configuration")
        self.surrogate = surrogate.to(self.device).eval()
        self.center = tuple(float(x) for x in center)
        self.axes = tuple(max(1e-6, float(x)) for x in axes)
        self.alpha = float(alpha)
        self.pgd_steps = pgd_steps
        self.pgd_learning_rate = pgd_learning_rate
        self.scenario_constraints = dict(scenario_constraints or {})
        self.adversary_summary: list[dict] = []

    def _collect_episode(self, scenario_name: str, seed: int, episode_number: int):
        torch = self.torch
        template = scenario_by_name(scenario_name, self.env_config)
        env = MigrationEnvironment(self.env_config, template)
        observation, _ = env.reset(seed)
        states, masks, actions, log_probs, values, rewards = [], [], [], [], [], []
        terminated_flags, boundaries = [], []
        total_reward = discounted_return = 0.0
        discount = 1.0
        pgd_values: list[float] = []
        raw_parameters: list[tuple[float, ...]] = []
        physical_parameters: list[tuple[float, ...]] = []
        constraint_checks: list[dict] = []
        constraint = self.scenario_constraints.get(scenario_name)
        if constraint is None:
            local_center = scenario_to_raw(template)[:len(self.center)]
            local_axes = self.axes
            local_alpha = self.alpha
            lower_bounds, upper_bounds = training_raw_bounds(
                template, parameter_dim=len(local_center)
            )
        else:
            local_center = tuple(float(value) for value in constraint["center"])
            local_axes = tuple(float(value) for value in constraint["axes"])
            local_alpha = float(constraint.get("alpha", self.alpha))
            lower_bounds = tuple(float(value) for value in constraint["lower_bounds"])
            upper_bounds = tuple(float(value) for value in constraint["upper_bounds"])

        while True:
            state_tensor = torch.tensor(observation, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
            mask_tuple = env.valid_action_mask()
            mask_tensor = torch.tensor(mask_tuple, dtype=torch.bool,
                                       device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.model(state_tensor)
                logits = logits.masked_fill(~mask_tensor, -1.0e9)
                distribution = torch.distributions.Categorical(logits=logits)
                action_tensor = distribution.sample()
                log_prob = distribution.log_prob(action_tensor)
            action = int(action_tensor.item())

            adversarial, raw, predicted_value = worst_case_scenario(
                self.surrogate,
                lambda next_states: self.model(next_states)[1],
                observation, action, local_center, local_axes, local_alpha, template,
                gamma=self.ppo_config.gamma, pgd_steps=self.pgd_steps,
                learning_rate=self.pgd_learning_rate, device=self.device,
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
            )
            # The physical parameter transform guarantees legal probabilities/rates.
            env.scenario = adversarial
            env.disturbances.scenario = adversarial
            pgd_values.append(predicted_value)
            raw_parameters.append(raw)
            physical_parameters.append(tuple(
                float(getattr(adversarial, name)) for name in PARAMETER_NAMES[:len(raw)]
            ))
            if local_alpha == 0:
                ellipsoid_distance = (0.0 if all(
                    abs(value - center) <= 1e-7
                    for value, center in zip(raw, local_center)
                ) else math.inf)
            else:
                ellipsoid_distance = math.sqrt(sum(
                    ((value - center) / (local_alpha * axis)) ** 2
                    for value, center, axis in zip(raw, local_center, local_axes)
                ))
            in_bounds = all(lower - 1e-7 <= value <= upper + 1e-7
                            for value, lower, upper in
                            zip(raw, lower_bounds, upper_bounds))
            in_ellipsoid = ellipsoid_distance <= 1.0 + 1e-6
            if not in_bounds or not in_ellipsoid:
                raise RuntimeError("PGD returned parameters outside its scenario constraints")
            constraint_checks.append({
                "in_raw_bounds": in_bounds,
                "in_ellipsoid": in_ellipsoid,
                "feasible": in_bounds and in_ellipsoid,
                "ellipsoid_l2": ellipsoid_distance,
            })

            next_observation, reward, terminated, truncated, _ = env.step(action)
            states.append(tuple(observation))
            masks.append(tuple(mask_tuple))
            actions.append(action)
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            boundaries.append(bool(terminated or truncated))
            total_reward += reward
            discounted_return += discount * reward
            discount *= self.ppo_config.gamma
            observation = next_observation
            if terminated or truncated:
                final_value = 0.0
                if truncated and not terminated:
                    with torch.no_grad():
                        state = torch.tensor(observation, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                        _, final = self.model(state)
                        final_value = float(final.item())
                break

        self.adversary_summary.append({
            "episode": episode_number,
            "base_scenario": scenario_name,
            "pgd_calls": len(pgd_values),
            "mean_surrogate_worst_value": sum(pgd_values) / max(1, len(pgd_values)),
            "parameter_names": list(PARAMETER_NAMES[:len(local_center)]),
            "ellipsoid": {"center": list(local_center), "axes": list(local_axes),
                          "alpha": local_alpha},
            "raw_bounds": {"lower": list(lower_bounds), "upper": list(upper_bounds)},
            "selected_raw_parameters": [list(row) for row in raw_parameters],
            "selected_physical_parameters": [list(row) for row in physical_parameters],
            "constraint_checks": constraint_checks,
            "all_constraints_satisfied": all(item["feasible"]
                                              for item in constraint_checks),
            "mean_raw_parameters": [
                sum(row[index] for row in raw_parameters) / max(1, len(raw_parameters))
                for index in range(len(local_center))
            ],
        })
        rollout = {
            "states": states, "masks": masks, "actions": actions,
            "log_probs": log_probs, "values": values, "rewards": rewards,
            "terminated": terminated_flags, "boundaries": boundaries,
            "final_value": final_value,
        }
        return EpisodeResult(episode_number, seed, scenario_name, discounted_return,
                             total_reward, len(actions)), rollout
