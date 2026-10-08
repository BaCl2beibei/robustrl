"""PPO with per-decision PGD against a differentiable transition surrogate.

PGD chooses legal scenario parameters from the learned surrogate's one-step
Bellman objective. The actual environment then samples hard CTMC/network/load
events under those parameters. This connects PGD to PPO training while keeping
the discrete simulator authoritative for collected rewards and transitions.
"""

from __future__ import annotations

from typing import Sequence

from .config import ScenarioConfig, SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name
from .robust_ppo import EpisodeResult, FiniteScenarioRobustPPO, PPOConfig
from .surrogate import worst_case_scenario


class SurrogatePGDRobustPPO(FiniteScenarioRobustPPO):
    def __init__(self, surrogate, center: Sequence[float], axes: Sequence[float],
                 alpha: float, env_config: SimulationConfig | None = None,
                 ppo_config: PPOConfig | None = None,
                 scenario_names: tuple[str, ...] =
                 ("normal", "failure", "network", "load", "mixed"),
                 device: str | None = None, seed: int = 2026,
                 pgd_steps: int = 8, pgd_learning_rate: float = 0.05) -> None:
        super().__init__(env_config, ppo_config, scenario_names, device, seed)
        if len(center) != len(axes) or len(center) != 6:
            raise ValueError("PGD center and axes must each have six parameters")
        if alpha < 0 or pgd_steps < 1 or pgd_learning_rate <= 0:
            raise ValueError("invalid PGD configuration")
        self.surrogate = surrogate.to(self.device).eval()
        self.center = tuple(float(x) for x in center)
        self.axes = tuple(max(1e-6, float(x)) for x in axes)
        self.alpha = float(alpha)
        self.pgd_steps = pgd_steps
        self.pgd_learning_rate = pgd_learning_rate
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
                observation, action, self.center, self.axes, self.alpha, template,
                gamma=self.ppo_config.gamma, pgd_steps=self.pgd_steps,
                learning_rate=self.pgd_learning_rate, device=self.device,
            )
            # The physical parameter transform guarantees legal probabilities/rates.
            env.scenario = adversarial
            env.disturbances.scenario = adversarial
            pgd_values.append(predicted_value)
            raw_parameters.append(raw)

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
            "mean_raw_parameters": [
                sum(row[index] for row in raw_parameters) / max(1, len(raw_parameters))
                for index in range(6)
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
