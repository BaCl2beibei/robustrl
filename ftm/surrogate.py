"""Differentiable transition surrogate and PGD interface to the simulator.

PGD cannot differentiate through sampled CTMC transitions, hard task statuses,
or discrete migration events. This module learns a conditional expected
one-step model from simulator traces, differentiates through that surrogate,
then applies the selected physical disturbance parameters to subsequent simulator
steps. The resulting min-max training is approximate and must be validated on
the original hard-event simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Callable, Mapping, Sequence

from .config import ScenarioConfig
from .uncertainty import sensitivity_axes, sigmoid, softplus


PARAMETER_NAMES = ("failure_multiplier", "congestion_enter", "congestion_leave",
                   "common_shock_probability", "load_burst_probability",
                   "load_burst_cpu")


def _inverse_softplus(value: float) -> float:
    value = max(value, 1e-8)
    return value if value > 30 else math.log(math.expm1(value))


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(probability / (1.0 - probability))


def scenario_to_raw(scenario: ScenarioConfig) -> tuple[float, ...]:
    """Encode the six simulator controls in unconstrained coordinates."""
    return (
        _inverse_softplus(scenario.failure_multiplier),
        _logit(scenario.congestion_enter),
        _logit(scenario.congestion_leave),
        _logit(scenario.common_shock_probability),
        _logit(scenario.load_burst_probability),
        _logit(scenario.load_burst_cpu),
    )


def raw_to_scenario(raw: Sequence[float], template: ScenarioConfig,
                    name: str = "pgd_adversarial") -> ScenarioConfig:
    if len(raw) != len(PARAMETER_NAMES):
        raise ValueError(f"raw parameter vector must have {len(PARAMETER_NAMES)} entries")
    values = [float(x) for x in raw]
    return replace(
        template,
        name=name,
        failure_multiplier=softplus(values[0]) + 1e-8,
        congestion_enter=sigmoid(values[1]),
        congestion_leave=sigmoid(values[2]),
        common_shock_probability=sigmoid(values[3]),
        load_burst_probability=sigmoid(values[4]),
        load_burst_cpu=sigmoid(values[5]),
    )


def perturb_scenario(template: ScenarioConfig, rng: random.Random,
                     raw_noise_std: float = 0.25) -> ScenarioConfig:
    if raw_noise_std < 0:
        raise ValueError("raw_noise_std must be non-negative")
    raw = scenario_to_raw(template)
    perturbed = [value + rng.gauss(0.0, raw_noise_std) for value in raw]
    return raw_to_scenario(perturbed, template, name=template.name + "_sample")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'learning' extra to use the surrogate") from exc
    return torch


def build_transition_surrogate(state_dim: int, action_dim: int,
                                parameter_dim: int = 6):
    torch = _torch()
    nn = torch.nn
    input_dim = state_dim + action_dim + parameter_dim

    class TransitionSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.body = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
            )
            self.delta_head = nn.Linear(128, state_dim)
            self.reward_head = nn.Linear(128, 1)
            self.terminal_head = nn.Linear(128, 1)

        def forward(self, states, action_one_hot, raw_parameters):
            features = torch.cat((states, action_one_hot, raw_parameters), dim=-1)
            hidden = self.body(features)
            return (self.delta_head(hidden), self.reward_head(hidden).squeeze(-1),
                    self.terminal_head(hidden).squeeze(-1))

    return TransitionSurrogate()


def build_value_model(state_dim: int):
    torch = _torch()
    nn = torch.nn

    class ValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1),
            )

        def forward(self, states):
            return self.network(states).squeeze(-1)

    return ValueModel()


def _records_to_tensors(records: Sequence[Mapping], action_dim: int, device: str):
    torch = _torch()
    if not records:
        raise ValueError("surrogate training needs transition records")
    states = torch.tensor([r["state"] for r in records], dtype=torch.float32, device=device)
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long, device=device)
    one_hot = torch.nn.functional.one_hot(actions, num_classes=action_dim).float()
    parameters = torch.tensor([r["parameters"] for r in records], dtype=torch.float32,
                              device=device)
    next_states = torch.tensor([r["next_state"] for r in records], dtype=torch.float32,
                               device=device)
    rewards = torch.tensor([r["reward"] for r in records], dtype=torch.float32,
                           device=device)
    returns = torch.tensor([r["return_to_go"] for r in records], dtype=torch.float32,
                           device=device)
    terminals = torch.tensor([float(r.get("terminated", False)) for r in records],
                             dtype=torch.float32, device=device)
    return states, one_hot, parameters, next_states, rewards, returns, terminals


def fit_surrogate(records: Sequence[Mapping], action_dim: int,
                  epochs: int = 30, batch_size: int = 256,
                  learning_rate: float = 1e-3, device: str = "cpu"):
    torch = _torch()
    states, one_hot, parameters, next_states, rewards, _, terminals = _records_to_tensors(
        records, action_dim, device
    )
    model = build_transition_surrogate(states.shape[-1], action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = states.shape[0]
    for _ in range(epochs):
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            delta_hat, reward_hat, terminal_logits = model(
                states[idx], one_hot[idx], parameters[idx]
            )
            delta_loss = torch.nn.functional.mse_loss(delta_hat,
                                                      next_states[idx] - states[idx])
            reward_loss = torch.nn.functional.mse_loss(reward_hat, rewards[idx])
            terminal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                terminal_logits, terminals[idx]
            )
            loss = delta_loss + reward_loss + terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval()


def fit_value_model(records: Sequence[Mapping], action_dim: int,
                    epochs: int = 30, batch_size: int = 256,
                    learning_rate: float = 1e-3, device: str = "cpu"):
    torch = _torch()
    states, _, _, _, _, returns, _ = _records_to_tensors(records, action_dim, device)
    model = build_value_model(states.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = states.shape[0]
    for _ in range(epochs):
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            prediction = model(states[idx])
            loss = torch.nn.functional.mse_loss(prediction, returns[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval()


def estimate_sensitivity_ellipsoid(records: Sequence[Mapping], surrogate,
                                   value_model, action_dim: int, gamma: float = 0.99,
                                   alpha: float = 1.0, min_scale: float = 0.02,
                                   min_axis_fraction: float = 0.05,
                                   max_records: int = 2048,
                                   device: str = "cpu") -> tuple[tuple[float, ...],
                                                                  tuple[float, ...], float,
                                                                  tuple[float, ...]]:
    """Estimate center, axes and mean |d Bellman target / d raw parameter|.

    Gradients pass through the learned conditional transition surrogate and the
    fitted value model. Hard event sampling is not differentiated.
    """
    torch = _torch()
    from .uncertainty import sensitivity_axes

    selected = list(records[:max_records])
    if len(selected) < 2:
        raise ValueError("at least two records are required to calibrate uncertainty")
    states, one_hot, parameters, _, _, _, _ = _records_to_tensors(
        selected, action_dim, device
    )
    center_tensor = parameters.mean(dim=0)
    scales_tensor = parameters.std(dim=0, unbiased=True).clamp_min(min_scale)
    abs_gradients = []
    surrogate.eval()
    value_model.eval()
    for index in range(states.shape[0]):
        raw = parameters[index].detach().clone().requires_grad_(True)
        delta, reward, terminal_logits = surrogate(
            states[index:index + 1], one_hot[index:index + 1], raw.unsqueeze(0)
        )
        predicted_next = (states[index:index + 1] + delta).clamp(0.0, 1.0)
        continuation = 1.0 - torch.sigmoid(terminal_logits).mean()
        target = reward.mean() + gamma * continuation * value_model(predicted_next).mean()
        grad, = torch.autograd.grad(target, raw)
        abs_gradients.append(grad.detach().abs())
    gradient = torch.stack(abs_gradients).mean(dim=0)
    center = tuple(float(v) for v in center_tensor.detach().cpu().tolist())
    base_scales = tuple(float(v) for v in scales_tensor.detach().cpu().tolist())
    grad_values = tuple(float(v) for v in gradient.detach().cpu().tolist())
    axes = sensitivity_axes(grad_values, base_scales, min_axis_fraction)
    return center, axes, alpha, grad_values


def worst_case_scenario(model, critic_value: Callable, state: Sequence[float],
                        action: int, center: Sequence[float], axes: Sequence[float],
                        alpha: float, template: ScenarioConfig, gamma: float = 0.99,
                        pgd_steps: int = 8, learning_rate: float = 0.05,
                        device: str = "cpu"):
    """PGD on the learned one-step Bellman surrogate; return physical parameters."""
    torch = _torch()
    from .torch_ops import projected_adversarial_search

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.nn.functional.one_hot(
        torch.tensor([action], dtype=torch.long, device=device),
        num_classes=model.action_dim,
    ).float()
    center_tensor = torch.tensor(center, dtype=torch.float32, device=device)
    axes_tensor = torch.tensor(axes, dtype=torch.float32, device=device)

    def objective(raw):
        delta, reward, terminal_logits = model(state_tensor, action_tensor, raw.unsqueeze(0))
        predicted_next = (state_tensor + delta).clamp(0.0, 1.0)
        next_value = critic_value(predicted_next)
        continuation = 1.0 - torch.sigmoid(terminal_logits).mean()
        return reward.mean() + gamma * continuation * next_value.mean()

    raw_worst, value = projected_adversarial_search(
        center_tensor, axes_tensor, alpha, objective, steps=pgd_steps,
        learning_rate=learning_rate,
    )
    scenario = raw_to_scenario(raw_worst.detach().cpu().tolist(), template)
    return scenario, tuple(float(v) for v in raw_worst.detach().cpu().tolist()), float(value.item())
