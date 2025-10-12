"""Policy adapters bridging algorithm policies with OpenSpiel evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pyspiel

PolicyFunction = Callable[[int, str, Sequence[int]], Sequence[float]]


@dataclass(frozen=True)
class PolicyMetadata:
    """Metadata describing how a policy was produced."""

    method: str
    iteration: int | None = None
    checkpoint: str | None = None
    mixture_weight: float | None = None


class PolicyAdapter(pyspiel.Policy):
    """Wraps a callable policy into an OpenSpiel ``Policy`` interface."""

    def __init__(
        self,
        game: pyspiel.Game,
        policy_fn: PolicyFunction,
        *,
        normalize: bool = True,
        metadata: PolicyMetadata | None = None,
    ) -> None:
        super().__init__()
        self._policy_fn = policy_fn
        self._normalize = normalize
        self.metadata = metadata
        self._num_actions = game.num_distinct_actions()
        self._game = game

    def action_probabilities(
        self,
        state: pyspiel.State,
        player_id: int | None = None,
    ) -> Mapping[int, float]:
        if state.is_terminal():
            return {}
        if state.is_chance_node():
            return {action: prob for action, prob in state.chance_outcomes()}

        pid = player_id if player_id is not None else state.current_player()
        if state.is_simultaneous_node():
            legal_actions = state.legal_actions(pid)
        else:
            legal_actions = state.legal_actions()

        info_state = state.information_state_string(pid)
        raw_probs = np.asarray(
            self._policy_fn(pid, info_state, legal_actions),
            dtype=np.float64,
        )

        if raw_probs.shape[0] != len(legal_actions):
            raise ValueError(
                "Policy function returned a vector of incorrect size for "
                f"info_state={info_state}. Expected {len(legal_actions)} entries, "
                f"got {raw_probs.shape[0]}.",
            )

        probs = self._normalize_probs(raw_probs)
        return {action: prob for action, prob in zip(legal_actions, probs)}

    def get_state_policy(self, state: pyspiel.State) -> Mapping[int, float]:
        """Return action probabilities for the provided state."""

        return self.action_probabilities(state)

    def get_state_policy_as_parallel_vectors(
        self,
        state: pyspiel.State,
    ) -> tuple[np.ndarray, np.ndarray]:
        probs_map = self.get_state_policy(state)
        if not probs_map:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        actions, probs = zip(*probs_map.items())
        return np.asarray(actions, dtype=np.int32), np.asarray(probs, dtype=np.float64)

    def _normalize_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._normalize:
            return raw_probs
        clipped = np.clip(raw_probs, 0.0, None)
        total = clipped.sum()
        if total <= 0.0:
            clipped = np.ones_like(clipped, dtype=np.float64)
            total = clipped.sum()
        return clipped / total


def uniform_policy(game: pyspiel.Game) -> PolicyAdapter:
    """Return a uniform random policy for the provided OpenSpiel game."""

    def _uniform_policy(_: int, __: str, legal_actions: Sequence[int]) -> Sequence[float]:
        if not legal_actions:
            return []
        return np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)

    return PolicyAdapter(game, _uniform_policy, normalize=False)


def tabular_policy_from_dict(
    game: pyspiel.Game,
    policy_table: Mapping[str, Sequence[float]],
) -> PolicyAdapter:
    """Create an adapter from an information-state keyed probability table."""

    def _lookup_policy(_: int, info_state: str, legal_actions: Sequence[int]) -> Sequence[float]:
        full_probs = policy_table.get(info_state)
        if full_probs is None:
            return np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
        full_probs = np.asarray(full_probs, dtype=np.float64)
        return full_probs.take(legal_actions)

    return PolicyAdapter(game, _lookup_policy)
