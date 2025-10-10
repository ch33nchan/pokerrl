from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class InfoSet:
    """Canonical representation of a game information set.

    The `key` must uniquely identify the information available to the current
    player. `metadata` is a dictionary that can be used by exploitability code
    to reconstruct the underlying game state when required.
    """

    key: str
    metadata: Dict[str, object]


class BaseGameEnv(abc.ABC):
    """Abstract base class for two-player zero-sum poker-like environments."""

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the first information-set vector."""

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        """Apply an action for the current player and return (obs, reward, done, info)."""

    @abc.abstractmethod
    def legal_actions(self) -> List[int]:
        """Return the list of legal action indices for the current player."""

    @abc.abstractmethod
    def get_info_set_vector(self) -> np.ndarray:
        """Return the canonical fixed-size vector representing the info set."""

    @abc.abstractmethod
    def get_action_sequence(self) -> Sequence[int]:
        """Return the ordered action sequence (integer codes) observed so far."""

    @abc.abstractmethod
    def get_obs_dim(self) -> int:
        """Return the dimensionality of the canonical info-set vector."""

    @abc.abstractmethod
    def num_actions(self) -> int:
        """Return the total number of discrete actions supported by the game."""

    @abc.abstractmethod
    def enumerate_info_sets(self) -> Iterable[InfoSet]:
        """Enumerate every reachable information set in the game."""

    @abc.abstractmethod
    def info_set_to_vector(self, infoset: InfoSet) -> np.ndarray:
        """Convert an enumerated `InfoSet` into the canonical vector representation."""

    @abc.abstractmethod
    def transition_from_infoset(self, infoset: InfoSet, action: int) -> InfoSet | None:
        """Return the successor infoset for `action`, or None if the game terminates."""

    @abc.abstractmethod
    def payoff_at_terminal(self, infoset: InfoSet, player: int) -> float:
        """Return the payoff to `player` at a terminal infoset."""

    @abc.abstractmethod
    def clone(self) -> "BaseGameEnv":
        """Return a deep copy of the environment for tree traversal routines."""

    def shape_assertions(self) -> None:
        obs = self.get_info_set_vector()
        assert obs.ndim == 1, "Info set vector must be 1-D"
        assert len(self.legal_actions()) > 0, "At least one action must be legal after reset"
        assert self.get_obs_dim() == obs.shape[0], "Reported obs dim must match vector length"
        assert isinstance(self.get_action_sequence(), Sequence), "Action sequence must be a sequence"

    def to_numpy(self, sequence: Sequence[int], num_actions: int) -> np.ndarray:
        arr = np.zeros((len(sequence), num_actions), dtype=np.float32)
        for idx, action in enumerate(sequence):
            arr[idx, action] = 1.0
        return arr
