"""Tiny tabular CFR head used by the anytime handoff schedule."""

from __future__ import annotations

from typing import Dict, Sequence


def _uniform_policy(legal_actions: Sequence[int]) -> Dict[int, float]:
    prob = 1.0 / max(len(legal_actions), 1)
    return {a: prob for a in legal_actions}


class TabularCFRHead:
    """Maintains cumulative regrets and strategy for marked subgames."""

    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.cumulative_regrets = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]
        self.cumulative_policy = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

    def policy(self, state_idx: int, legal_actions: Sequence[int]) -> Dict[int, float]:
        regrets = self.cumulative_regrets[state_idx]
        positive = {a: max(regrets[a], 0.0) for a in legal_actions}
        normaliser = sum(positive.values())
        if normaliser <= 1e-8:
            return _uniform_policy(legal_actions)
        return {a: positive[a] / normaliser for a in legal_actions}

    def observe(self, state_idx: int, legal_actions: Sequence[int], regrets: Dict[int, float]) -> None:
        table = self.cumulative_regrets[state_idx]
        for a in legal_actions:
            table[a] += regrets.get(a, 0.0)

    def accumulate_policy(self, state_idx: int, legal_actions: Sequence[int], probs: Dict[int, float]) -> None:
        table = self.cumulative_policy[state_idx]
        for a in legal_actions:
            table[a] += probs.get(a, 0.0)

    def average_policy(self, state_idx: int, legal_actions: Sequence[int]) -> Dict[int, float]:
        table = self.cumulative_policy[state_idx]
        normaliser = sum(table[a] for a in legal_actions)
        if normaliser <= 1e-8:
            return _uniform_policy(legal_actions)
        return {a: table[a] / normaliser for a in legal_actions}

    def reset(self) -> None:
        for row in self.cumulative_regrets:
            for i in range(len(row)):
                row[i] = 0.0
        for row in self.cumulative_policy:
            for i in range(len(row)):
                row[i] = 0.0
