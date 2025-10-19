"""Approximate best response utilities used for MARM-K meta-objective."""

from __future__ import annotations

import random
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pyspiel


Policy = Callable[[pyspiel.State, int], Dict[int, float]]


class ApproxBestResponse:
    """Rollout-based approximate best-response estimator."""

    def __init__(self, game: pyspiel.Game, *, budget: int = 32, seed: int | None = None) -> None:
        self.game = game
        self.budget = budget
        self.random = random.Random(seed)

    def _play_episode(
        self,
        state: pyspiel.State,
        player_policy: Policy,
        opponent_policy: Policy,
    ) -> Tuple[float, float]:
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = self.random.choices(actions, weights=probs, k=1)[0]
                state = state.child(action)
                continue
            player = state.current_player()
            policy = player_policy if player == 0 else opponent_policy
            info_policy = policy(state, player)
            actions, probs = zip(*info_policy.items())
            action = self.random.choices(actions, weights=probs, k=1)[0]
            state = state.child(action)
        returns = state.returns()
        return float(returns[0]), float(returns[1])

    def evaluate_experts(
        self,
        base_policy: Policy,
        experts: Sequence[Policy],
    ) -> np.ndarray:
        if not experts:
            return np.zeros(0)
        utilities = np.zeros(len(experts))
        for _ in range(self.budget):
            rng_state = self.random.getstate()
            base_return = self._play_episode(
                self.game.new_initial_state().clone(), base_policy, base_policy
            )[0]
            post_state = self.random.getstate()
            for idx, expert in enumerate(experts):
                self.random.setstate(rng_state)
                expert_return = self._play_episode(
                    self.game.new_initial_state().clone(), expert, base_policy
                )[0]
                utilities[idx] += expert_return - base_return
            self.random.setstate(post_state)
        utilities /= max(1, self.budget)
        return utilities
