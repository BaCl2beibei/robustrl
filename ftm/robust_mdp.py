"""Finite-state robust value iteration for algorithm correctness experiments.

The inner minimization is exact over the finite candidate distributions supplied
for each state-action pair. It is not an exact solver for a continuous ellipsoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


State = Hashable
Action = Hashable
TransitionRow = tuple[tuple[State, float], ...]


@dataclass
class ValueIterationResult:
    values: dict[State, float]
    policy: dict[State, Action]
    iterations: int
    residual: float


class FiniteRobustMDP:
    def __init__(self, states: Sequence[State], actions: Mapping[State, Sequence[Action]],
                 rewards: Mapping[tuple[State, Action], float],
                 nominal: Mapping[tuple[State, Action], TransitionRow],
                 uncertainty: Mapping[tuple[State, Action], Sequence[TransitionRow]] | None = None
                 ) -> None:
        self.states = tuple(states)
        self.actions = {s: tuple(actions[s]) for s in self.states}
        self.rewards = dict(rewards)
        self.nominal = dict(nominal)
        self.uncertainty = {key: tuple(rows) for key, rows in (uncertainty or {}).items()}
        self._validate()

    def _validate(self) -> None:
        state_set = set(self.states)
        for state in self.states:
            if not self.actions[state]:
                raise ValueError(f"state {state!r} has no legal actions")
            for action in self.actions[state]:
                key = (state, action)
                if key not in self.rewards or key not in self.nominal:
                    raise ValueError(f"missing reward or nominal transition for {key!r}")
                rows = (self.nominal[key],) + self.uncertainty.get(key, ())
                for row in rows:
                    if any(next_state not in state_set for next_state, _ in row):
                        raise ValueError(f"transition from {key!r} references unknown state")
                    if any(probability < 0.0 for _, probability in row):
                        raise ValueError("transition probabilities must be non-negative")
                    if abs(sum(probability for _, probability in row) - 1.0) > 1e-8:
                        raise ValueError("each transition row must sum to one")

    def candidate_rows(self, state: State, action: Action,
                       robust: bool) -> tuple[TransitionRow, ...]:
        key = (state, action)
        nominal = self.nominal[key]
        if not robust:
            return (nominal,)
        return (nominal,) + self.uncertainty.get(key, ())

    def value_iteration(self, gamma: float = 0.95, robust: bool = True,
                        tolerance: float = 1e-9, max_iterations: int = 100_000
                        ) -> ValueIterationResult:
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        values = {state: 0.0 for state in self.states}
        policy: dict[State, Action] = {}
        residual = float("inf")
        for iteration in range(1, max_iterations + 1):
            updated: dict[State, float] = {}
            for state in self.states:
                action_values = []
                for action in self.actions[state]:
                    returns = []
                    for row in self.candidate_rows(state, action, robust):
                        continuation = sum(prob * values[next_state]
                                           for next_state, prob in row)
                        returns.append(self.rewards[(state, action)] + gamma * continuation)
                    action_values.append((min(returns), action))
                updated[state], policy[state] = max(action_values, key=lambda pair: pair[0])
            residual = max(abs(updated[state] - values[state]) for state in self.states)
            values = updated
            if residual < tolerance:
                return ValueIterationResult(values, policy, iteration, residual)
        return ValueIterationResult(values, policy, max_iterations, residual)


def two_state_example() -> FiniteRobustMDP:
    """Small example where nominal and robust policies can differ."""
    states = ("decision", "good", "bad")
    actions = {"decision": ("stay", "migrate"), "good": ("absorb",), "bad": ("absorb",)}
    rewards = {
        ("decision", "stay"): 2.0,
        ("decision", "migrate"): -5.0,
        ("good", "absorb"): 5.0,
        ("bad", "absorb"): -4.0,
    }
    nominal = {
        ("decision", "stay"): (("good", 0.9), ("bad", 0.1)),
        ("decision", "migrate"): (("good", 0.95), ("bad", 0.05)),
        ("good", "absorb"): (("good", 1.0),),
        ("bad", "absorb"): (("bad", 1.0),),
    }
    uncertainty = {
        ("decision", "stay"): ((("good", 0.6), ("bad", 0.4)),),
    }
    return FiniteRobustMDP(states, actions, rewards, nominal, uncertainty)
