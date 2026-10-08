"""Load and evaluate saved PPO or surrogate-PGD PPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import ScenarioConfig, SimulationConfig
from .robust_ppo import _build_actor_critic
from .surrogate import build_transition_surrogate, worst_case_scenario


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the optional 'learning' extra to evaluate PPO") from exc
    return torch


class TrainedPolicy:
    """Inference-only checkpoint adapter for paired-seed evaluation."""

    def __init__(self, checkpoint_path: str | Path, state_dim: int, action_dim: int,
                 device: str | None = None) -> None:
        torch = _torch()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                         weights_only=False)
        except TypeError:  # compatibility with older supported torch releases
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        policy_key = ("policy_state_dict" if "policy_state_dict" in self.checkpoint
                      else "model_state_dict")
        if policy_key not in self.checkpoint:
            raise ValueError("checkpoint has no recognized PPO state dictionary")
        if (self.checkpoint.get("state_dim", state_dim) != state_dim
                or self.checkpoint.get("action_dim", action_dim) != action_dim):
            raise ValueError("checkpoint dimensions do not match the requested environment")
        self.model = _build_actor_critic(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(self.checkpoint[policy_key])
        self.model.eval()
        self.surrogate = None
        self.center: Sequence[float] | None = None
        self.axes: Sequence[float] | None = None
        self.alpha = 0.0
        if "surrogate_state_dict" in self.checkpoint:
            self.surrogate = build_transition_surrogate(state_dim, action_dim).to(self.device)
            self.surrogate.load_state_dict(self.checkpoint["surrogate_state_dict"])
            self.surrogate.eval()
            self.center = self.checkpoint["center"]
            self.axes = self.checkpoint["axes"]
            self.alpha = float(self.checkpoint["alpha"])

    @property
    def is_adversarial(self) -> bool:
        return self.surrogate is not None

    def act(self, env, observation: Sequence[float], template: ScenarioConfig) -> int:
        torch = self.torch
        state = torch.tensor(observation, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state)
            action = int(logits.masked_fill(~mask, -1.0e9).argmax(dim=-1).item())
        if self.surrogate is not None:
            scenario, _, _ = worst_case_scenario(
                self.surrogate, lambda next_states: self.model(next_states)[1],
                observation, action, self.center, self.axes, self.alpha, template,
                gamma=float(self.checkpoint.get("ppo_config", {}).get("gamma", 0.99)),
                pgd_steps=int(self.checkpoint.get("pgd_steps", 8)), device=self.device,
            )
            env.scenario = scenario
            env.disturbances.scenario = scenario
        return action


def run_trained_episode(policy_name: str, scenario_name: str, seed: int,
                        config: SimulationConfig, controller: TrainedPolicy):
    """Run and score one checkpoint episode using the hard-event simulator."""
    from .environment import MigrationEnvironment, scenario_by_name
    from .metrics import compute_episode_metrics

    template = scenario_by_name(scenario_name, config)
    env = MigrationEnvironment(config, template)
    observation, _ = env.reset(seed)
    total_reward = 0.0
    while True:
        action = controller.act(env, observation, template)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(env, policy_name, seed, total_reward)
    # The surrogate-PGD controller may replace env.scenario during the episode.
    # Keep the requested outer evaluation scenario in the report so matched
    # scenario comparisons remain possible.
    metrics.scenario = template.name
    return metrics, env
