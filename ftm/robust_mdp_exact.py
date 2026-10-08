"""Exact robust value iteration for finite MDPs with TV-ball uncertainty.

This solver is globally optimal for the supplied *finite* MDP and rectangular
total-variation ambiguity set. It is not an exact solver for the continuous
parameter ellipsoid used by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


State = Hashable
Action = Hashable
Row = tuple[tuple[State, float], ...]


def worst_case_tv_row(nominal: Row, values: Mapping[State, float],
                      radius: float) -> tuple[Row, float]:
    """Minimize q·V subject to TV(q,p) <= radius, exactly.

    TV distance is half the L1 distance; ``radius`` is therefore the total
    probability mass that can be moved from high-value to low-value outcomes.
    """
    if not 0 <= radius <= 1:
        raise ValueError("TV radius must be in [0, 1]")
    if not nominal or len({state for state, _ in nominal}) != len(nominal):
        raise ValueError("nominal row must be non-empty with unique states")
    states = [state for state, _ in nominal]
    q = [float(probability) for _, probability in nominal]
    if any(probability < 0 for probability in q) or abs(sum(q) - 1.0) > 1e-9:
        raise ValueError("nominal row must be a probability vector")
    if any(state not in values for state in states):
        raise ValueError("values is missing a successor state")

    donors = sorted(range(len(states)), key=lambda i: (-values[states[i]], i))
    receivers = sorted(range(len(states)), key=lambda i: (values[states[i]], i))
    donor_pos = receiver_pos = 0
    moved = 0.0
    while moved < radius - 1e-14 and donor_pos < len(donors) \
            and receiver_pos < len(receivers):
        donor, receiver = donors[donor_pos], receivers[receiver_pos]
        if values[states[donor]] <= values[states[receiver]] + 1e-14:
            break
        amount = min(radius - moved, q[donor], 1.0 - q[receiver])
        if amount <= 1e-14:
            if q[donor] <= 1e-14:
                donor_pos += 1
            if 1.0 - q[receiver] <= 1e-14:
                receiver_pos += 1
            continue
        q[donor] -= amount
        q[receiver] += amount
        moved += amount
        if q[donor] <= 1e-14:
            donor_pos += 1
        if 1.0 - q[receiver] <= 1e-14:
            receiver_pos += 1
    row = tuple((state, probability) for state, probability in zip(states, q))
    return row, sum(probability * values[state] for state, probability in row)


@dataclass(frozen=True)
class ExactRobustResult:
    values: dict[State, float]
    policy: dict[State, Action]
    iterations: int
    residual: float


class TVRobustMDP:
    """Discounted finite robust MDP with state-action rectangular TV balls."""

    def __init__(self, states: Sequence[State],
                 actions: Mapping[State, Sequence[Action]],
                 rewards: Mapping[tuple[State, Action], float],
                 nominal: Mapping[tuple[State, Action], Row],
                 tv_radii: Mapping[tuple[State, Action], float]) -> None:
        self.states = tuple(states)
        self.actions = {state: tuple(actions[state]) for state in self.states}
        self.rewards, self.nominal, self.tv_radii = dict(rewards), dict(nominal), dict(tv_radii)
        for state in self.states:
            if not self.actions[state]:
                raise ValueError(f"state {state!r} has no actions")
            for action in self.actions[state]:
                key = (state, action)
                if key not in self.rewards or key not in self.nominal or key not in self.tv_radii:
                    raise ValueError(f"missing MDP data for {key!r}")
                row = self.nominal[key]
                if any(next_state not in self.states for next_state, _ in row):
                    raise ValueError(f"unknown successor state in {key!r}")
                if not row or any(p < 0 for _, p in row) \
                        or abs(sum(p for _, p in row) - 1.0) > 1e-9:
                    raise ValueError(f"invalid transition row for {key!r}")
                if not 0 <= self.tv_radii[key] <= 1:
                    raise ValueError(f"invalid TV radius for {key!r}")

    def value_iteration(self, gamma: float = 0.95, tolerance: float = 1e-10,
                        max_iterations: int = 100_000) -> ExactRobustResult:
        if not 0 <= gamma < 1 or tolerance <= 0 or max_iterations < 1:
            raise ValueError("invalid value-iteration parameters")
        values = {state: 0.0 for state in self.states}
        residual = float("inf")
        for iteration in range(1, max_iterations + 1):
            updated, policy = {}, {}
            for state in self.states:
                candidates = []
                for action in self.actions[state]:
                    key = (state, action)
                    _, continuation = worst_case_tv_row(
                        self.nominal[key], values, self.tv_radii[key]
                    )
                    candidates.append((self.rewards[key] + gamma * continuation, action))
                updated[state], policy[state] = max(candidates, key=lambda pair: pair[0])
            residual = max(abs(updated[s] - values[s]) for s in self.states)
            values = updated
            if residual <= tolerance:
                return ExactRobustResult(values, policy, iteration, residual)
        return ExactRobustResult(values, policy, max_iterations, residual)


def two_state_tv_example(radius: float = 0.3) -> TVRobustMDP:
    states = ("decision", "good", "bad")
    actions = {"decision": ("stay", "migrate"), "good": ("absorb",),
               "bad": ("absorb",)}
    rewards = {("decision", "stay"): 2.0, ("decision", "migrate"): -5.0,
               ("good", "absorb"): 5.0, ("bad", "absorb"): -4.0}
    nominal = {("decision", "stay"): (("good", 0.9), ("bad", 0.1)),
               ("decision", "migrate"): (("good", 0.95), ("bad", 0.05)),
               ("good", "absorb"): (("good", 1.0),),
               ("bad", "absorb"): (("bad", 1.0),)}
    radii = {key: radius if key == ("decision", "stay") else 0.0 for key in nominal}
    return TVRobustMDP(states, actions, rewards, nominal, radii)
