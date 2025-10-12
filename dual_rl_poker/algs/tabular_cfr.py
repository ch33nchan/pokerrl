"""Tabular CFR anchor using OpenSpiel's exact solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pyspiel

from eval import OpenSpielExactEvaluator, PolicyAdapter, PolicyMetadata
from algs.base import TrainingState


@dataclass
class TabularCFRConfig:
    game_name: str = "kuhn_poker"
    iterations: int = 0
    eval_every: int = 1


class TabularCFRAgent:
    """Exact CFR baseline built on top of OpenSpiel's CFR solver."""

    def __init__(self, config: TabularCFRConfig):
        self.config = config
        self.game = pyspiel.load_game(config.game_name)
        self.solver = pyspiel.CFRSolver(self.game)
        self.iteration = 0
        self.evaluator = OpenSpielExactEvaluator(config.game_name)
        self._policy_cache: Optional[PolicyAdapter] = None

    def train_iteration(self) -> TrainingState:
        """Run a single CFR iteration and return a TrainingState."""

        start_iter = self.iteration
        self.iteration += 1
        self.solver.evaluate_and_update_policy()
        self._policy_cache = None

        return TrainingState(
            iteration=self.iteration,
            loss=0.0,
            wall_time=0.0,
            gradient_norm=0.0,
            learning_rate=0.0,
            buffer_size=0,
            extra_metrics={
                "algorithm": "tabular_cfr",
                "iterations_completed": self.iteration - start_iter,
            },
        )

    def run(self) -> Dict[str, float]:
        """Run CFR for the configured number of iterations, returning final metrics."""

        for _ in range(self.config.iterations):
            self.iteration += 1
            self.solver.evaluate_and_update_policy()
            self._policy_cache = None

        return self.evaluate()

    def evaluate(self) -> Dict[str, float]:
        policy = self.policy_adapter
        result = self.evaluator.evaluate(
            policy,
            metadata=PolicyMetadata(method="tabular_cfr", iteration=self.iteration),
        )
        return {
            "nash_conv": result.nash_conv,
            "exploitability": result.exploitability,
            "player_0_value": result.player_0_value,
            "player_1_value": result.player_1_value,
            "mean_value": result.mean_value,
        }

    def checkpoint_policy(self) -> Dict[str, np.ndarray]:
        """Return the tabular average policy mapping info states to probabilities."""

        openspiel_policy = self.solver.average_policy()
        policy_table: Dict[str, np.ndarray] = {}

        for state in self._enumerate_states():
            if state.is_terminal() or state.is_chance_node():
                continue
            player = state.current_player()
            info_state = state.information_state_string(player)
            legal_actions = state.legal_actions()
            probs_map = openspiel_policy.action_probabilities(state)
            full = np.zeros(self.game.num_distinct_actions(), dtype=np.float64)
            for action in legal_actions:
                full[action] = probs_map.get(action, 0.0)
            policy_table[info_state] = full

        return policy_table

    @property
    def policy_adapter(self) -> PolicyAdapter:
        if self._policy_cache is None:
            policy_table = self.checkpoint_policy()
            self._policy_cache = self.evaluator.build_tabular_policy(
                policy_table,
                metadata=PolicyMetadata(method="tabular_cfr", iteration=self.iteration),
            )
        return self._policy_cache

    def _enumerate_states(self):
        stack = [self.game.new_initial_state()]
        visited = set()
        while stack:
            state = stack.pop()
            state_str = state.history_str()
            if state_str in visited:
                continue
            visited.add(state_str)
            yield state
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    stack.append(state.child(action))
            else:
                for action in state.legal_actions():
                    stack.append(state.child(action))
