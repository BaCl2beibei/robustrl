"""Load and evaluate saved PPO or surrogate-PGD PPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import ScenarioConfig, SimulationConfig
from .robust_ppo import _build_actor_critic


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
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=False)
        except TypeError:  # compatibility with older supported torch releases
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy_key = ("policy_state_dict" if "policy_state_dict" in checkpoint
                      else "model_state_dict")
        if policy_key not in checkpoint:
            raise ValueError("checkpoint has no recognized PPO state dictionary")
        if (checkpoint.get("state_dim", state_dim) != state_dim
                or checkpoint.get("action_dim", action_dim) != action_dim):
            raise ValueError("checkpoint dimensions do not match the requested environment")
        self._is_adversarial = "surrogate_state_dict" in checkpoint
        self.model = _build_actor_critic(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(checkpoint[policy_key])
        self.model.eval()
        del checkpoint

    @property
    def is_adversarial(self) -> bool:
        return self._is_adversarial

    def act(self, env, observation: Sequence[float], template=None) -> int:
        torch = self.torch
        state = torch.tensor(observation, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state)
            action = int(logits.masked_fill(~mask, -1.0e9).argmax(dim=-1).item())
        return action


def run_trained_episode(policy_name: str, scenario_name: str | ScenarioConfig, seed: int,
                        config: SimulationConfig, controller: TrainedPolicy):
    """Run and score one checkpoint episode using the hard-event simulator."""
    from .environment import MigrationEnvironment, scenario_by_name
    from .metrics import compute_episode_metrics

    scenario = (scenario_name if isinstance(scenario_name, ScenarioConfig)
                else scenario_by_name(scenario_name, config))
    env = MigrationEnvironment(config, scenario)
    observation, _ = env.reset(seed)
    total_reward = 0.0
    while True:
        action = controller.act(env, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(env, policy_name, seed, total_reward)
    return metrics, env
