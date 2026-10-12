"""PPO with per-decision PGD against a differentiable transition surrogate.

PGD chooses legal scenario parameters from the learned surrogate's one-step
Bellman objective. The actual environment then samples hard CTMC/network/load
events under those parameters. This connects PGD to PPO training while keeping
the discrete simulator authoritative for collected rewards and transitions.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .config import ScenarioConfig, SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .observation import encode_observation
from .robust_ppo import EpisodeResult, FiniteScenarioRobustPPO, PPOConfig
from .surrogate import (PARAMETER_NAMES, scenario_to_raw, training_raw_bounds,
                        fit_surrogate, worst_case_scenario)


DEFAULT_PGD_SCENARIOS = ("normal", "failure", "network", "load_short",
                         "load_medium", "load_sustained", "mixed")


class SurrogatePGDRobustPPO(FiniteScenarioRobustPPO):
    def __init__(self, surrogate, center: Sequence[float], axes: Sequence[float],
                 alpha: float, env_config: SimulationConfig | None = None,
                 ppo_config: PPOConfig | None = None,
                 scenario_names: tuple[str, ...] = DEFAULT_PGD_SCENARIOS,
                 device: str | None = None, seed: int = 2026,
                 pgd_steps: int = 8, pgd_learning_rate: float = 0.05,
                 scenario_constraints: Mapping[str, Mapping] | None = None,
                 warmup_episodes: int = 0,
                 max_pgd_step_fraction: float = 0.2,
                 surrogate_training_records: Sequence[Mapping] = (),
                 surrogate_refresh_interval: int = 10,
                 surrogate_refresh_epochs: int = 5,
                 surrogate_replay_capacity: int = 20_000) -> None:
        super().__init__(env_config, ppo_config, scenario_names, device, seed)
        if len(center) != len(PARAMETER_NAMES) or len(axes) != len(PARAMETER_NAMES):
            raise ValueError("PGD center and axes must each have exactly eight entries")
        if alpha < 0 or pgd_steps < 1 or pgd_learning_rate <= 0:
            raise ValueError("invalid PGD configuration")
        if warmup_episodes < 0 or not 0.0 < max_pgd_step_fraction <= 1.0:
            raise ValueError("invalid warmup or per-step PGD constraint")
        if surrogate_refresh_interval < 1 or surrogate_refresh_epochs < 1 \
                or surrogate_replay_capacity < 2:
            raise ValueError("invalid surrogate refresh configuration")
        self.surrogate = surrogate.to(self.device).eval()
        self.center = tuple(float(x) for x in center)
        self.axes = tuple(max(1e-6, float(x)) for x in axes)
        self.alpha = float(alpha)
        self.pgd_steps = pgd_steps
        self.pgd_learning_rate = pgd_learning_rate
        self.scenario_constraints = dict(scenario_constraints or {})
        self.warmup_episodes = warmup_episodes
        self.max_pgd_step_fraction = max_pgd_step_fraction
        self.initial_surrogate_records = [dict(row) for row in surrogate_training_records]
        if self.initial_surrogate_records and {
                len(row.get("parameters", ())) for row in self.initial_surrogate_records
        } != {len(PARAMETER_NAMES)}:
            raise ValueError("surrogate refresh records must have exactly eight parameters")
        self.surrogate_refresh_interval = surrogate_refresh_interval
        self.surrogate_refresh_epochs = surrogate_refresh_epochs
        self.surrogate_replay_capacity = surrogate_replay_capacity
        self.online_surrogate_records: list[dict] = []
        self.surrogate_refresh_count = 0
        self.ppo_update_count = 0
        self._refresh_rng = random.Random(seed ^ 0x8D1A5EED)
        self.adversary_summary: list[dict] = []

    def _collect_episode(self, scenario_name: str, seed: int, episode_number: int):
        if (episode_number <= self.warmup_episodes
                or (episode_number - self.warmup_episodes - 1) % 2 == 1):
            result, rollout = super()._collect_episode(
                scenario_name, seed, episode_number
            )
            parameters = list(scenario_to_raw(
                scenario_by_name(scenario_name, self.env_config)
            ))
            records = [None] * len(rollout["rewards"])
            return_to_go = 0.0
            for index in reversed(range(len(records))):
                reward = float(rollout["rewards"][index])
                return_to_go = reward + self.ppo_config.gamma * return_to_go
                records[index] = {
                    "state": list(rollout["states"][index]),
                    "action": int(rollout["actions"][index]),
                    "parameters": parameters,
                    "next_state": list(rollout["next_states"][index]),
                    "reward": reward,
                    "terminated": bool(rollout["terminated"][index]),
                    "return_to_go": return_to_go,
                    "scenario": scenario_name,
                }
            rollout["surrogate_records"] = records
            return result, rollout

        torch = self.torch
        template = scenario_by_name(scenario_name, self.env_config)
        env = MigrationEnvironment(self.env_config, template)
        env.reset(seed)
        states, masks, actions, log_probs, values, rewards = [], [], [], [], [], []
        terminated_flags, boundaries = [], []
        total_reward = discounted_return = 0.0
        discount = 1.0
        pgd_values: list[float] = []
        raw_parameters: list[tuple[float, ...]] = []
        physical_parameters: list[tuple[float, ...]] = []
        constraint_checks: list[dict] = []
        transition_records: list[dict] = []
        constraint = self.scenario_constraints.get(scenario_name)
        if constraint is None:
            local_center = scenario_to_raw(template)
            local_axes = self.axes
            local_alpha = self.alpha
            lower_bounds, upper_bounds = training_raw_bounds(template)
        else:
            local_center = tuple(float(value) for value in constraint["center"])
            local_axes = tuple(float(value) for value in constraint["axes"])
            local_alpha = float(constraint.get("alpha", self.alpha))
            lower_bounds = tuple(float(value) for value in constraint["lower_bounds"])
            upper_bounds = tuple(float(value) for value in constraint["upper_bounds"])
        if any(len(vector) != len(PARAMETER_NAMES) for vector in
               (local_center, local_axes, lower_bounds, upper_bounds)):
            raise ValueError("scenario PGD constraints must use eight dimensions")
        step_limits = tuple((upper - lower) * self.max_pgd_step_fraction
                            for lower, upper in zip(lower_bounds, upper_bounds))
        previous_raw = local_center

        while True:
            context = env.get_decision_context()
            observation = encode_observation(context)
            state_tensor = torch.tensor(observation, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
            mask_tuple = context.action_mask
            mask_tensor = torch.tensor(mask_tuple, dtype=torch.bool,
                                       device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.model(state_tensor)
                logits = logits.masked_fill(~mask_tensor, -1.0e9)
                distribution = torch.distributions.Categorical(logits=logits)
                action_tensor = distribution.sample()
                log_prob = distribution.log_prob(action_tensor)
            action = int(action_tensor.item())

            step_lower = tuple(max(lower, previous - limit)
                               for lower, previous, limit in
                               zip(lower_bounds, previous_raw, step_limits))
            step_upper = tuple(min(upper, previous + limit)
                               for upper, previous, limit in
                               zip(upper_bounds, previous_raw, step_limits))
            adversarial, raw, predicted_value = worst_case_scenario(
                self.surrogate,
                lambda next_states: self.model(next_states)[1],
                observation, action, local_center, local_axes, local_alpha, template,
                gamma=self.ppo_config.gamma, pgd_steps=self.pgd_steps,
                learning_rate=self.pgd_learning_rate, device=self.device,
                lower_bounds=step_lower, upper_bounds=step_upper,
            )
            # The physical parameter transform guarantees legal probabilities/rates.
            env.scenario = adversarial
            env.disturbances.scenario = adversarial
            pgd_values.append(predicted_value)
            raw_parameters.append(raw)
            physical_parameters.append(tuple(
                float(getattr(adversarial, name)) for name in PARAMETER_NAMES
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
            step_change = tuple(abs(value - previous)
                                for value, previous in zip(raw, previous_raw))
            within_step = all(change <= limit + 1e-6
                              for change, limit in zip(step_change, step_limits))
            in_ellipsoid = ellipsoid_distance <= 1.0 + 1e-6
            if not in_bounds or not in_ellipsoid or not within_step:
                raise RuntimeError("PGD returned parameters outside its scenario constraints")
            constraint_checks.append({
                "in_raw_bounds": in_bounds,
                "in_ellipsoid": in_ellipsoid,
                "within_step_change": within_step,
                "max_step_change": max(step_change, default=0.0),
                "feasible": in_bounds and in_ellipsoid and within_step,
                "ellipsoid_l2": ellipsoid_distance,
            })

            next_observation, reward, terminated, truncated, _ = env.step(action)
            transition_records.append({
                "state": list(observation),
                "action": action,
                "parameters": list(raw),
                "next_state": list(next_observation),
                "reward": float(reward),
                "terminated": bool(terminated),
                "return_to_go": 0.0,
                "scenario": scenario_name,
            })
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
            previous_raw = raw
            if terminated or truncated:
                final_value = 0.0
                if truncated and not terminated:
                    with torch.no_grad():
                        state = torch.tensor(observation, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                        _, final = self.model(state)
                        final_value = float(final.item())
                break

        return_to_go = 0.0
        for record in reversed(transition_records):
            return_to_go = (float(record["reward"])
                            + self.ppo_config.gamma * return_to_go)
            record["return_to_go"] = return_to_go

        self.adversary_summary.append({
            "episode": episode_number,
            "base_scenario": scenario_name,
            "pgd_calls": len(pgd_values),
            "mean_surrogate_worst_value": sum(pgd_values) / max(1, len(pgd_values)),
            "parameter_names": list(PARAMETER_NAMES),
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
            "surrogate_records": transition_records,
        }
        return EpisodeResult(episode_number, seed, scenario_name, discounted_return,
                             total_reward, len(actions)), rollout

    def _ppo_update(self, rollout: dict) -> None:
        super()._ppo_update(rollout)
        self.ppo_update_count += 1
        new_records = rollout.get("surrogate_records", ())
        if new_records:
            self.online_surrogate_records.extend(dict(record) for record in new_records)
            overflow = len(self.online_surrogate_records) - self.surrogate_replay_capacity
            if overflow > 0:
                del self.online_surrogate_records[:overflow]
        if (self.ppo_update_count % self.surrogate_refresh_interval == 0
                and self.online_surrogate_records):
            self._refresh_surrogate()

    def _refresh_surrogate(self) -> None:
        """Refit between episodes using a mix of initial and recent real transitions."""
        capacity = self.surrogate_replay_capacity
        base_limit = capacity // 2
        base_count = min(len(self.initial_surrogate_records), base_limit)
        if base_count == len(self.initial_surrogate_records):
            base_records = list(self.initial_surrogate_records)
        else:
            base_records = self._refresh_rng.sample(self.initial_surrogate_records,
                                                    base_count)
        recent_limit = capacity - len(base_records)
        recent_records = self.online_surrogate_records[-recent_limit:]
        if len(recent_records) < 2:
            return
        refreshed = fit_surrogate(
            base_records + recent_records, self.env_config.n_nodes + 1,
            epochs=self.surrogate_refresh_epochs, device=self.device,
        )
        self.surrogate = refreshed.to(self.device).eval()
        self.surrogate_refresh_count += 1
