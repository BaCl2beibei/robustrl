"""Two-stage migration policy and optional high-confidence Gate supervision."""

from __future__ import annotations

import math

from .decision_features import AUX_FEATURES_PER_NODE
from .robust_ppo import FiniteScenarioRobustPPO, _torch


def build_hierarchical_actor_critic(state_dim: int, action_dim: int):
    torch = _torch()
    nn = torch.nn
    n_targets = action_dim - 1

    class HierarchicalActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(),
                                      nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
            self.target = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(),
                                        nn.Linear(128, 128), nn.Tanh(),
                                        nn.Linear(128, n_targets))
            self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(),
                                        nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

        def gate_logits(self, states):
            return self.gate(states).squeeze(-1)

        def forward(self, states, mask=None):
            gate = self.gate_logits(states)
            target_logits = self.target(states)
            if mask is not None:
                legal = mask[..., :-1]
                has_target = legal.any(dim=-1)
                target_logits = target_logits.masked_fill(~legal, -1.0e9)
                gate = torch.where(has_target, gate, torch.full_like(gate, -1.0e9))
            target_log_probs = torch.log_softmax(target_logits, dim=-1)
            move_logits = torch.nn.functional.logsigmoid(gate).unsqueeze(-1) + target_log_probs
            wait_logits = torch.nn.functional.logsigmoid(-gate).unsqueeze(-1)
            return torch.cat((move_logits, wait_logits), dim=-1), self.critic(states).squeeze(-1)

    return HierarchicalActorCritic()


def gate_labels(states, masks, n_targets: int):
    """High-margin time labels from the same public auxiliary features as the Actor."""
    torch = _torch()
    aux = states[..., -n_targets * AUX_FEATURES_PER_NODE:].reshape(
        -1, n_targets, AUX_FEATURES_PER_NODE)
    legal = masks[..., :-1]
    move = torch.sinh(aux[..., 2])
    wait = torch.sinh(aux[..., 6])
    recoverable = aux[..., 7] > 0.5
    valid = legal & recoverable
    best_move = move.masked_fill(~valid, math.inf).min(dim=-1).values
    wait_time = wait.max(dim=-1).values
    margin = torch.maximum(torch.full_like(wait_time, 2.0 / 600.0),
                           0.15 * wait_time)
    positive = valid.any(dim=-1) & (wait_time - best_move >= margin)
    negative = valid.any(dim=-1) & (best_move - wait_time >= margin)
    impossible = legal.any(dim=-1) & (~recoverable & legal).any(dim=-1)
    positive = positive | impossible
    labelled = positive | negative
    return positive.to(states.dtype), labelled


class HierarchicalPPO(FiniteScenarioRobustPPO):
    def __init__(self, *args, auxiliary: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary = auxiliary
        self.auxiliary_beta = 0.0

    def _auxiliary_loss(self, states, masks):
        if not self.auxiliary or self.auxiliary_beta <= 0.0:
            return states.new_zeros(())
        torch = self.torch
        labels, selected = gate_labels(states, masks, self.env_config.n_nodes)
        if not selected.any():
            return states.new_zeros(())
        logits = self.model.gate_logits(states[selected])
        return self.auxiliary_beta * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[selected])
