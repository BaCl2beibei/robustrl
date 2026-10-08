"""Optional PPO trainer with a finite-scenario adversary.

The adversary selects the currently worst-scoring scenario from a fixed suite.
This is a practical robust-training approximation; it is not the exact solution
to the continuous parameter-ellipsoid min-max problem.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SimulationConfig
from .environment import MigrationEnvironment, scenario_by_name


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'learning' extra to train the PPO policy") from exc
    return torch


def _build_actor_critic(state_dim: int, action_dim: int):
    torch = _torch()
    nn = torch.nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(),
            )
            self.actor = nn.Linear(128, action_dim)
            self.critic = nn.Linear(128, 1)

        def forward(self, states):
            hidden = self.body(states)
            return self.actor(hidden), self.critic(hidden).squeeze(-1)

    return ActorCritic()


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 128
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    scenario_history_size: int = 12

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("clip_ratio and learning_rate must be positive")
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("update_epochs and minibatch_size must be positive")


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    scenario: str
    discounted_return: float
    total_reward: float
    decisions: int


class FiniteScenarioRobustPPO:
    """PPO with action masks and a worst-performing scenario selector."""

    def __init__(self, env_config: SimulationConfig | None = None,
                 ppo_config: PPOConfig | None = None,
                 scenario_names: tuple[str, ...] =
                 ("normal", "failure", "network", "load", "mixed"),
                 device: str | None = None, seed: int = 2026) -> None:
        if not scenario_names:
            raise ValueError("scenario_names must not be empty")
        self.env_config = env_config or SimulationConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.scenario_names = scenario_names
        self.seed = seed
        self.torch = _torch()
        self.torch.manual_seed(seed)
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scenario_returns: dict[str, list[float]] = {name: [] for name in scenario_names}

    def train(self, episodes: int, model_path: str | Path | None = None,
              log_every: int = 10) -> list[EpisodeResult]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        torch = self.torch
        initial_env = MigrationEnvironment(
            self.env_config, scenario_by_name(self.scenario_names[0], self.env_config)
        )
        initial_obs, _ = initial_env.reset(self.seed)
        self.model = _build_actor_critic(len(initial_obs), self.env_config.n_nodes + 1)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.ppo_config.learning_rate)
        results: list[EpisodeResult] = []
        for episode in range(episodes):
            scenario_name = self._select_scenario(episode)
            seed = self.seed + episode
            result, rollout = self._collect_episode(scenario_name, seed, episode + 1)
            self._ppo_update(rollout)
            results.append(result)
            self.scenario_returns[scenario_name].append(result.discounted_return)
            if log_every > 0 and ((episode + 1) % log_every == 0 or episode + 1 == episodes):
                print(f"episode={episode + 1}/{episodes} scenario={scenario_name} "
                      f"discounted_return={result.discounted_return:.3f} "
                      f"reward={result.total_reward:.3f}")
        if model_path is not None:
            self.save(model_path, episodes)
        return results

    def _select_scenario(self, episode_index: int) -> str:
        # Initial coverage gives the adversary a score for each scenario.
        if episode_index < len(self.scenario_names):
            return self.scenario_names[episode_index]
        scores = {}
        for name, returns in self.scenario_returns.items():
            recent = returns[-self.ppo_config.scenario_history_size:]
            if recent:
                scores[name] = statistics.mean(recent)
        return min(scores, key=scores.get) if scores else self.scenario_names[0]

    def _collect_episode(self, scenario_name: str, seed: int, episode_number: int):
        torch = self.torch
        env = MigrationEnvironment(
            self.env_config, scenario_by_name(scenario_name, self.env_config)
        )
        observation, _ = env.reset(seed)
        states, masks, actions, log_probs, values, rewards = [], [], [], [], [], []
        terminated_flags, boundaries = [], []
        total_reward = 0.0
        discounted_return = 0.0
        discount = 1.0
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
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
            next_observation, reward, terminated, truncated, _ = env.step(int(action.item()))
            states.append(tuple(observation))
            masks.append(tuple(mask_tuple))
            actions.append(int(action.item()))
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
                        _, value = self.model(state)
                        final_value = float(value.item())
                break
        rollout = {
            "states": states, "masks": masks, "actions": actions,
            "log_probs": log_probs, "values": values, "rewards": rewards,
            "terminated": terminated_flags, "boundaries": boundaries,
            "final_value": final_value,
        }
        return EpisodeResult(episode_number, seed, scenario_name, discounted_return,
                             total_reward, len(actions)), rollout

    def _ppo_update(self, rollout: dict) -> None:
        torch = self.torch
        cfg = self.ppo_config
        rewards = rollout["rewards"]
        values = rollout["values"]
        boundaries = rollout["boundaries"]
        terminated = rollout["terminated"]
        advantages = [0.0] * len(rewards)
        gae = 0.0
        next_value = rollout["final_value"]
        for index in reversed(range(len(rewards))):
            bootstrap = 0.0 if terminated[index] else next_value
            delta = rewards[index] + cfg.gamma * bootstrap - values[index]
            continuation = 0.0 if boundaries[index] else 1.0
            gae = delta + cfg.gamma * cfg.gae_lambda * continuation * gae
            advantages[index] = gae
            next_value = values[index]
        returns = [adv + value for adv, value in zip(advantages, values)]

        states = torch.tensor(rollout["states"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(rollout["masks"], dtype=torch.bool, device=self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32,
                                     device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        target_values = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        count = states.shape[0]
        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                indices = permutation[start:start + cfg.minibatch_size]
                logits, predicted_values = self.model(states[indices])
                masked_logits = logits.masked_fill(~masks[indices], -1.0e9)
                distribution = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratio * adv[indices]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio,
                                      1.0 + cfg.clip_ratio) * adv[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(
                    predicted_values, target_values[indices]
                )
                loss = (policy_loss + cfg.value_coefficient * value_loss
                        - cfg.entropy_coefficient * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.max_gradient_norm)
                self.optimizer.step()

    def save(self, path: str | Path, episodes: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({
            "model_state_dict": self.model.state_dict(),
            "env_config": asdict(self.env_config),
            "ppo_config": asdict(self.ppo_config),
            "scenario_names": self.scenario_names,
            "seed": self.seed,
            "episodes": episodes,
            "scenario_returns": self.scenario_returns,
        }, path)

