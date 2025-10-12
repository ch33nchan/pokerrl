"""Exact OpenSpiel evaluator for NashConv and exploitability computation."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pyspiel

from .policy_adapter import PolicyAdapter, PolicyMetadata, uniform_policy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationResult:
    """Container for exact evaluation outputs."""

    game_name: str
    nash_conv: float
    exploitability: float
    player_0_value: float
    player_1_value: float
    mean_value: float
    openspiel_version: str
    python_version: str
    info_state_count: int
    metadata: PolicyMetadata | None = None


class OpenSpielExactEvaluator:
    """Exact evaluator using OpenSpiel's NashConv/exploitability utilities."""

    def __init__(self, game_name: str) -> None:
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError(
                f"OpenSpielExactEvaluator only supports 2-player games, got {self.game.num_players()}"
            )
        game_type = self.game.get_type()
        if game_type.utility != pyspiel.GameType.Utility.ZERO_SUM:
            logger.warning("Game %s is not tagged as zero-sum; exploitability may be undefined.", game_name)

        logger.info(
            "Initialized OpenSpielExactEvaluator for %s (OpenSpiel %s)",
            game_name,
            pyspiel.__version__,
        )

    # ------------------------------------------------------------------
    # Policy adaptation helpers
    # ------------------------------------------------------------------
    def build_policy(self, policy_fn: Callable[[int, str, Sequence[int]], Sequence[float]], *, metadata: PolicyMetadata | None = None) -> PolicyAdapter:
        """Wrap a callable into an OpenSpiel ``Policy`` instance."""

        return PolicyAdapter(self.game, policy_fn, metadata=metadata)

    def build_tabular_policy(
        self,
        policy_table: Mapping[str, Sequence[float]],
        *,
        metadata: PolicyMetadata | None = None,
    ) -> PolicyAdapter:
        def _lookup(_: int, info_state: str, legal_actions: Sequence[int]) -> Sequence[float]:
            if info_state not in policy_table:
                return np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
            full = np.asarray(policy_table[info_state], dtype=np.float64)
            if full.size != self.game.num_distinct_actions():
                raise ValueError(
                    "Policy table for info_state %s has wrong length (expected %d, got %d)" % (
                        info_state,
                        self.game.num_distinct_actions(),
                        full.size,
                    )
                )
            return full.take(legal_actions)

        return PolicyAdapter(self.game, _lookup, metadata=metadata)

    # ------------------------------------------------------------------
    # Exact metrics
    # ------------------------------------------------------------------
    def evaluate(self, policy: pyspiel.Policy, *, metadata: PolicyMetadata | None = None) -> EvaluationResult:
        """Compute exact NashConv and exploitability for ``policy``."""

        tabular_mapping = self._policy_to_tabular_mapping(policy)
        tabular_policy = pyspiel.TabularPolicy(tabular_mapping)

        nash_conv = float(pyspiel.nash_conv(self.game, tabular_mapping))
        exploitability = float(pyspiel.exploitability(self.game, tabular_mapping))

        values = pyspiel.expected_returns(
            self.game.new_initial_state(),
            [tabular_policy, tabular_policy],
            -1,
            True,
        )
        player_0_value = float(values[0])
        player_1_value = float(values[1]) if len(values) > 1 else -player_0_value

        uniform = uniform_policy(self.game)
        uniform_mapping = self._policy_to_tabular_mapping(uniform)
        uniform_policy_tabular = pyspiel.TabularPolicy(uniform_mapping)
        mean_value = float(
            pyspiel.expected_returns(
                self.game.new_initial_state(),
                [tabular_policy, uniform_policy_tabular],
                -1,
                True,
            )[0]
        )

        return EvaluationResult(
            game_name=self.game_name,
            nash_conv=nash_conv,
            exploitability=exploitability,
            player_0_value=player_0_value,
            player_1_value=player_1_value,
            mean_value=mean_value,
            openspiel_version=str(pyspiel.__version__),
            python_version=self._python_version_string(),
            info_state_count=self._count_information_states(),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Head-to-head evaluation
    # ------------------------------------------------------------------
    def head_to_head_evs(
        self,
        policy_a: pyspiel.Policy,
        policy_b: pyspiel.Policy,
        *,
        num_matches: int,
        seed: int | None = None,
    ) -> Dict[str, float]:
        """Simulate head-to-head scores between two policies."""

        rng = np.random.default_rng(seed)
        returns: list[Tuple[float, float]] = []
        for match in range(num_matches):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    action = rng.choice(actions, p=np.asarray(probs, dtype=np.float64))
                    state.apply_action(action)
                    continue

                player = state.current_player()
                policy = policy_a if player == 0 else policy_b
                probs_map = policy.action_probabilities(state, player)
                actions, probs = zip(*probs_map.items())
                action = rng.choice(actions, p=np.asarray(probs, dtype=np.float64))
                state.apply_action(action)

            payoff = state.returns()
            returns.append((float(payoff[0]), float(payoff[1]) if len(payoff) > 1 else -float(payoff[0])))

        mean_0 = float(np.mean([r[0] for r in returns]))
        mean_1 = float(np.mean([r[1] for r in returns]))
        return {
            "player_0_mean": mean_0,
            "player_1_mean": mean_1,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _count_information_states(self) -> int:
        counter = 0
        to_visit = [self.game.new_initial_state()]
        seen: set[str] = set()
        while to_visit:
            state = to_visit.pop()
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    to_visit.append(state.child(action))
                continue
            player = state.current_player()
            info_state = state.information_state_string(player)
            if info_state not in seen:
                seen.add(info_state)
                counter += 1
            legal_actions = state.legal_actions()
            for action in legal_actions:
                to_visit.append(state.child(action))
        return counter

    def _policy_to_tabular_mapping(self, policy: pyspiel.Policy) -> Dict[str, Tuple[Tuple[int, float], ...]]:
        mapping: Dict[str, Tuple[Tuple[int, float], ...]] = {}
        stack = [self.game.new_initial_state()]
        visited: set[str] = set()

        while stack:
            state = stack.pop()
            state_key = state.history_str()
            if state_key in visited:
                continue
            visited.add(state_key)

            if state.is_terminal():
                continue

            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    stack.append(state.child(action))
                continue

            player = state.current_player()
            info_state = state.information_state_string(player)
            probs_map = policy.action_probabilities(state, player)

            if not probs_map:
                legal_actions = state.legal_actions()
                if legal_actions:
                    prob = 1.0 / len(legal_actions)
                    mapping[info_state] = tuple((action, prob) for action in legal_actions)
                else:
                    mapping[info_state] = tuple()
            else:
                normalized: list[Tuple[int, float]] = []
                total = sum(max(0.0, float(prob)) for prob in probs_map.values())
                if total <= 0.0:
                    legal_actions = state.legal_actions()
                    prob = 1.0 / len(legal_actions) if legal_actions else 0.0
                    normalized = [(action, prob) for action in legal_actions]
                else:
                    normalized = [
                        (action, float(max(0.0, prob) / total))
                        for action, prob in probs_map.items()
                    ]
                mapping[info_state] = tuple(normalized)

            for action in state.legal_actions():
                stack.append(state.child(action))

        return mapping

    @staticmethod
    def _python_version_string() -> str:
        import platform

        return platform.python_version()


def create_evaluator(game_name: str) -> OpenSpielExactEvaluator:
    return OpenSpielExactEvaluator(game_name)