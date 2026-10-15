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
from .decision_features import AUX_FEATURES_PER_NODE
from .observation import SCHEMA_VERSION
from .uncertainty import sensitivity_axes, sigmoid, softplus


PARAMETER_NAMES = ("failure_multiplier", "congestion_enter", "congestion_leave",
                   "common_shock_probability", "load_burst_probability",
                   "load_burst_cpu", "disconnect_enter", "disconnect_leave")


def _inverse_softplus(value: float) -> float:
    value = max(value, 1e-8)
    return value if value > 30 else math.log(math.expm1(value))


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(probability / (1.0 - probability))


def scenario_to_raw(scenario: ScenarioConfig) -> tuple[float, ...]:
    """Encode the adversarially controlled simulator parameters."""
    return (
        _inverse_softplus(scenario.failure_multiplier),
        _logit(scenario.congestion_enter),
        _logit(scenario.congestion_leave),
        _logit(scenario.common_shock_probability),
        _logit(scenario.load_burst_probability),
        _logit(scenario.load_burst_cpu),
        _logit(scenario.disconnect_enter),
        _logit(scenario.disconnect_leave),
    )


def raw_to_scenario(raw: Sequence[float], template: ScenarioConfig,
                    name: str = "pgd_adversarial") -> ScenarioConfig:
    if len(raw) != len(PARAMETER_NAMES):
        raise ValueError("raw surrogate parameter vector must have exactly eight entries")
    values = [float(x) for x in raw]
    changes = dict(
        name=name,
        failure_multiplier=max(softplus(values[0]), 1e-12),
        congestion_enter=sigmoid(values[1]),
        congestion_leave=sigmoid(values[2]),
        common_shock_probability=sigmoid(values[3]),
        load_burst_probability=sigmoid(values[4]),
        load_burst_cpu=sigmoid(values[5]),
    )
    changes.update(disconnect_enter=sigmoid(values[6]),
                   disconnect_leave=sigmoid(values[7]))
    return replace(template, **changes)


def perturb_scenario(template: ScenarioConfig, rng: random.Random,
                     raw_noise_std: float = 0.25) -> ScenarioConfig:
    if raw_noise_std < 0:
        raise ValueError("raw_noise_std must be non-negative")
    raw = scenario_to_raw(template)
    radius = 2.0 * raw_noise_std
    perturbed = [min(value + radius,
                      max(value - radius, value + rng.gauss(0.0, raw_noise_std)))
                 for value in raw]
    return raw_to_scenario(perturbed, template, name=template.name + "_sample")


def training_raw_bounds(template: ScenarioConfig, noise_std: float = 0.25
                        ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the per-scenario raw-parameter region within two noise stddevs."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    center = scenario_to_raw(template)
    radius = 2.0 * noise_std
    return (tuple(value - radius for value in center),
            tuple(value + radius for value in center))


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'learning' extra to use the surrogate") from exc
    return torch


def build_transition_surrogate(state_dim: int, action_dim: int):
    torch = _torch()
    nn = torch.nn
    parameter_dim = len(PARAMETER_NAMES)
    input_dim = state_dim + action_dim + parameter_dim

    class TransitionSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.parameter_dim = parameter_dim
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


def _infer_state_layout(state_dim: int, action_dim: int) -> tuple[int, int]:
    """Infer node and task counts from the current MigrationEnvironment layout."""
    n_nodes = action_dim - 1
    if n_nodes < 2:
        raise ValueError("action_dim must include at least two nodes and the wait action")
    current_fixed = 11 + 15 * n_nodes + 5 * n_nodes * n_nodes
    current_stride = 21 + 2 * n_nodes
    if (state_dim - current_fixed >= current_stride
            and (state_dim - current_fixed) % current_stride == 0):
        return n_nodes, (state_dim - current_fixed) // current_stride
    if SCHEMA_VERSION == "decision-context-v3-recovery-aux8":
        base_dim = state_dim - AUX_FEATURES_PER_NODE * n_nodes
        if (base_dim - current_fixed >= current_stride
                and (base_dim - current_fixed) % current_stride == 0):
            return n_nodes, (base_dim - current_fixed) // current_stride
    # Node features (8 each), directed links (4 each), task state (16 + 2N),
    # global features (3), burst estimates (N), and migration progress (3/task).
    fixed = 4 * n_nodes * n_nodes + 9 * n_nodes + 3
    task_stride = 19 + 2 * n_nodes
    remaining = state_dim - fixed
    # New observations append one focus-task bit per task plus completion and
    # migration-cost estimates. Keep recognizing legacy observations as well.
    decision_feature_size = 2 * n_nodes + 2
    extended_stride = task_stride + 1
    extended_remaining = remaining - decision_feature_size
    if (extended_remaining >= extended_stride
            and extended_remaining % extended_stride == 0):
        return n_nodes, extended_remaining // extended_stride
    if remaining >= task_stride and remaining % task_stride == 0:
        return n_nodes, remaining // task_stride
    raise ValueError("state dimension does not match MigrationEnvironment layout")


def _has_migration_decision_features(state_dim: int, n_nodes: int,
                                     n_tasks: int) -> bool:
    """Return whether a state dimension includes the appended decision features."""
    fixed = 4 * n_nodes * n_nodes + 9 * n_nodes + 3
    legacy_task_stride = 19 + 2 * n_nodes
    extended_size = n_tasks + 2 * n_nodes + 2
    return state_dim == fixed + n_tasks * legacy_task_stride + extended_size


def project_predicted_next_state(predicted_state, action_dim: int):
    """Project a surrogate state onto the simulator's feature constraints.

    Legacy and time-cost features are clipped to [0, 1], while relative time
    savings and SLA slack are clipped to [-1, 1]. Straight-through gradients
    preserve the PGD path. Binary and one-hot groups are projected to legal
    values in the forward pass with differentiable backward approximations.
    """
    torch = _torch()
    n_nodes, n_tasks = _infer_state_layout(predicted_state.shape[-1], action_dim)
    base_dim = (11 + 15 * n_nodes + 5 * n_nodes * n_nodes
                + n_tasks * (21 + 2 * n_nodes))
    if (SCHEMA_VERSION == "decision-context-v3-recovery-aux8"
            and predicted_state.shape[-1] == base_dim + AUX_FEATURES_PER_NODE * n_nodes):
        base_state = project_predicted_next_state(predicted_state[..., :base_dim],
                                                  action_dim)
        raw_aux = predicted_state[..., base_dim:].reshape(
            *predicted_state.shape[:-1], n_nodes, AUX_FEATURES_PER_NODE)
        nonnegative = raw_aux.clamp_min(0.0)
        feasible_aux = torch.cat((nonnegative[..., :3], raw_aux[..., 3:4],
                                  nonnegative[..., 4:7],
                                  raw_aux[..., 7:8].clamp(0.0, 1.0)), dim=-1)
        feasible_aux = raw_aux + (feasible_aux - raw_aux).detach()
        target_mask = base_state[..., base_dim - action_dim:base_dim - 1]
        feasible_aux = feasible_aux * target_mask.unsqueeze(-1)
        return torch.cat((base_state, feasible_aux.flatten(start_dim=-2)), dim=-1)
    if (predicted_state.shape[-1] == 11 + 15 * n_nodes
            + 5 * n_nodes * n_nodes + n_tasks * (21 + 2 * n_nodes)):
        # Current schema has unbounded measured quantities. Only enforce
        # non-negativity and categorical fields; never clip large bursts.
        projected = predicted_state + (predicted_state.clamp_min(0)
                                       - predicted_state).detach()
        projected = projected.clone()
        groups = [(1, n_tasks + 1)]
        node_start = 10 + n_tasks
        task_start = node_start + 14 * n_nodes
        for task in range(n_tasks):
            start = task_start + task * (20 + 2 * n_nodes)
            groups.extend(((start + 10, 6), (start + 16, n_nodes + 1),
                           (start + 17 + n_nodes, n_nodes + 1)))
        for start, width in groups:
            logits = predicted_state[..., start:start + width]
            soft = torch.softmax(logits, dim=-1)
            hard = torch.nn.functional.one_hot(
                soft.argmax(dim=-1), num_classes=width).to(soft.dtype)
            projected[..., start:start + width] = hard - soft.detach() + soft
        binary = [node_start + node * 14 + 1 for node in range(n_nodes)]
        binary.extend(task_start + task * (20 + 2 * n_nodes)
                      + 19 + 2 * n_nodes for task in range(n_tasks))
        link_start = task_start + n_tasks * (20 + 2 * n_nodes)
        binary.extend(link_start + link * 5 + 2 for link in range(n_nodes * n_nodes))
        binary.extend(range(predicted_state.shape[-1] - action_dim,
                            predicted_state.shape[-1]))
        for index in binary:
            value = predicted_state[..., index]
            hard = (value >= 0.5).to(value.dtype)
            projected[..., index] = value + (hard - value).detach()
        return projected
    fixed = 4 * n_nodes * n_nodes + 9 * n_nodes + 3
    legacy_task_stride = 19 + 2 * n_nodes
    has_migration_features = _has_migration_decision_features(
        predicted_state.shape[-1], n_nodes, n_tasks
    )
    feature_start = fixed + n_tasks * legacy_task_stride
    clip_start = feature_start if has_migration_features else predicted_state.shape[-1]
    clipped = predicted_state.clone()
    clipped[..., :clip_start] = predicted_state[..., :clip_start].clamp(0.0, 1.0)
    if has_migration_features:
        signed_start = feature_start + n_tasks + 1 + n_nodes
        clipped[..., feature_start: signed_start] = predicted_state[
            ..., feature_start: signed_start
        ].clamp(0.0, 1.0)
        clipped[..., signed_start:] = predicted_state[..., signed_start:].clamp(-1.0, 1.0)
    projected = predicted_state + (clipped - predicted_state).detach()
    projected = projected.clone()

    binary_indices = [node * 8 for node in range(n_nodes)]
    link_offset = 8 * n_nodes
    binary_indices.extend(
        link_offset + link * 4 + field
        for link in range(n_nodes * n_nodes) for field in (2, 3)
    )
    task_offset = link_offset + 4 * n_nodes * n_nodes
    task_stride = 16 + 2 * n_nodes
    categorical_groups = []
    for task in range(n_tasks):
        offset = task_offset + task * task_stride
        binary_indices.append(offset + 6)  # SLA flag
        categorical_groups.extend((
            (offset + 8, 6),                    # task status
            (offset + 14, n_nodes + 1),         # task node, including None
            (offset + 15 + n_nodes, n_nodes + 1),  # migration target, including None
        ))
    global_offset = task_offset + n_tasks * task_stride
    binary_indices.append(global_offset + 2)  # common-shock flag

    focus_group = None
    if has_migration_features:
        focus_group = (feature_start, n_tasks)

    for index in binary_indices:
        value = predicted_state[..., index]
        hard = (value >= 0.5).to(dtype=value.dtype)
        projected[..., index] = value + (hard - value).detach()

    for start, width in categorical_groups:
        logits = predicted_state[..., start:start + width]
        probabilities = torch.softmax(logits, dim=-1)
        hard = torch.nn.functional.one_hot(
            probabilities.argmax(dim=-1), num_classes=width
        ).to(dtype=probabilities.dtype)
        projected[..., start:start + width] = (
            hard - probabilities.detach() + probabilities
        )
    if focus_group is not None:
        start, width = focus_group
        logits = predicted_state[..., start:start + width]
        probabilities = torch.softmax(logits, dim=-1)
        hard = torch.nn.functional.one_hot(
            probabilities.argmax(dim=-1), num_classes=width
        ).to(dtype=probabilities.dtype)
        present = (logits.max(dim=-1, keepdim=True).values >= 0.5).to(
            dtype=probabilities.dtype
        )
        projected[..., start:start + width] = present * (
            hard - probabilities.detach() + probabilities
        )
    return projected


def _records_to_tensors(records: Sequence[Mapping], action_dim: int, device: str):
    torch = _torch()
    if not records:
        raise ValueError("surrogate training needs transition records")
    parameter_dimensions = {len(record.get("parameters", ())) for record in records}
    if parameter_dimensions != {len(PARAMETER_NAMES)}:
        raise ValueError("surrogate records must all contain exactly eight parameters")
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
                                   device: str = "cpu",
                                   center_override: Sequence[float] | None = None
                                   ) -> tuple[tuple[float, ...],
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
        predicted_next = project_predicted_next_state(
            states[index:index + 1] + delta, action_dim
        )
        continuation = 1.0 - torch.sigmoid(terminal_logits).mean()
        target = reward.mean() + gamma * continuation * value_model(predicted_next).mean()
        grad, = torch.autograd.grad(target, raw)
        abs_gradients.append(grad.detach().abs())
    gradient = torch.stack(abs_gradients).mean(dim=0)
    center = tuple(float(v) for v in center_tensor.detach().cpu().tolist())
    if center_override is not None:
        if len(center_override) != parameters.shape[-1]:
            raise ValueError("center_override dimension does not match training parameters")
        center = tuple(float(value) for value in center_override)
    base_scales = tuple(float(v) for v in scales_tensor.detach().cpu().tolist())
    grad_values = tuple(float(v) for v in gradient.detach().cpu().tolist())
    axes = sensitivity_axes(grad_values, base_scales, min_axis_fraction)
    return center, axes, alpha, grad_values


def worst_case_scenario(model, critic_value: Callable, state: Sequence[float],
                        action: int, center: Sequence[float], axes: Sequence[float],
                        alpha: float, template: ScenarioConfig, gamma: float = 0.99,
                        pgd_steps: int = 8, learning_rate: float = 0.05,
                        device: str = "cpu",
                        lower_bounds: Sequence[float] | None = None,
                        upper_bounds: Sequence[float] | None = None):
    """PGD on the learned one-step Bellman surrogate; return physical parameters."""
    torch = _torch()
    from .torch_ops import projected_adversarial_search

    if (len(center) != len(PARAMETER_NAMES) or len(axes) != len(PARAMETER_NAMES)
            or (lower_bounds is not None and len(lower_bounds) != len(PARAMETER_NAMES))
            or (upper_bounds is not None and len(upper_bounds) != len(PARAMETER_NAMES))):
        raise ValueError("PGD center, axes and bounds must each have eight entries")

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.nn.functional.one_hot(
        torch.tensor([action], dtype=torch.long, device=device),
        num_classes=model.action_dim,
    ).float()
    center_tensor = torch.tensor(center, dtype=torch.float32, device=device)
    axes_tensor = torch.tensor(axes, dtype=torch.float32, device=device)

    def objective(raw):
        delta, reward, terminal_logits = model(state_tensor, action_tensor, raw.unsqueeze(0))
        predicted_next = project_predicted_next_state(state_tensor + delta,
                                                       model.action_dim)
        next_value = critic_value(predicted_next)
        continuation = 1.0 - torch.sigmoid(terminal_logits).mean()
        return reward.mean() + gamma * continuation * next_value.mean()

    raw_worst, value = projected_adversarial_search(
        center_tensor, axes_tensor, alpha, objective, steps=pgd_steps,
        learning_rate=learning_rate,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds,
    )
    scenario = raw_to_scenario(raw_worst.detach().cpu().tolist(), template)
    return scenario, tuple(float(v) for v in raw_worst.detach().cpu().tolist()), float(value.item())
