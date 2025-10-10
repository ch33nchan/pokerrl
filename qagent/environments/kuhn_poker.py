"""Canonical Kuhn Poker environment implementing the unified BaseGameEnv API."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from qagent.environments.base import BaseGameEnv, InfoSet


ACTION_PASS = 0
ACTION_BET = 1
ACTION_SYMBOLS = {ACTION_PASS: "p", ACTION_BET: "b"}
HISTORY_TERMINALS = {"pp", "bp", "bb", "pbp", "pbb"}


@dataclass(frozen=True)
class KuhnState:
    cards: Tuple[int, int]
    history: str
    terminal: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "cards": self.cards,
            "history": self.history,
            "terminal": self.terminal,
        }


class KuhnEnv(BaseGameEnv):
    num_players: int = 2
    num_cards: int = 3
    num_actions_val: int = 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._state: KuhnState | None = None
        self._current_player: int = 0
        self._pot: int = 2  # ante
        self.reset(seed)

    # ---------------------------------------------------------------------
    # BaseGameEnv interface
    # ---------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cards = list(range(self.num_cards))
        self._rng.shuffle(cards)
        self._state = KuhnState(cards=(cards[0], cards[1]), history="", terminal=False)
        self._current_player = 0
        self._pot = 2
        return self.get_info_set_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        self._assert_state()
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} for history {self._state.history}")

        history = self._state.history + ACTION_SYMBOLS[action]
        terminal = history in HISTORY_TERMINALS
        prev_player = self._current_player

        self._state = KuhnState(cards=self._state.cards, history=history, terminal=terminal)
        if terminal:
            reward = self._compute_payoff(prev_player)
            info = {"terminal_history": history, "cards": self._state.cards}
            return self.get_info_set_vector(), reward, True, info

        self._current_player = 1 - self._current_player
        reward = 0.0
        info = {"history": history}
        return self.get_info_set_vector(), reward, False, info

    def legal_actions(self, state: Dict[str, object] | None = None) -> List[int]:
        if state is None:
            self._assert_state()
            state = self._state.to_dict()
        return self.get_legal_actions_from_state(state)

    def get_info_set_vector(self) -> np.ndarray:
        self._assert_state()
        return self._info_vector_from_state(self._state.to_dict(), self._current_player)

    def get_action_sequence(self) -> Sequence[int]:
        self._assert_state()
        return [0 if c == "p" else 1 for c in self._state.history]

    def get_obs_dim(self) -> int:
        return self.num_cards + 4

    def num_actions(self) -> int:
        return self.num_actions_val

    # ------------------------------------------------------------------
    # Enumeration utilities
    # ------------------------------------------------------------------
    def enumerate_info_sets(self) -> Iterable[InfoSet]:
        infosets: Dict[str, InfoSet] = {}
        for cards in itertools.permutations(range(self.num_cards), 2):
            initial = {
                "cards": cards,
                "history": "",
                "terminal": False,
            }
            self._enumerate_from_state(initial, infosets)
        return infosets.values()

    def info_set_to_vector(self, infoset: InfoSet) -> np.ndarray:
        state_dict = {
            "cards": infoset.metadata["cards"],
            "history": infoset.metadata["history"],
            "terminal": infoset.metadata["terminal"],
        }
        return self._info_vector_from_state(state_dict, infoset.metadata["player"])

    def transition_from_infoset(self, infoset: InfoSet, action: int) -> InfoSet | None:
        if action not in (ACTION_PASS, ACTION_BET):
            raise ValueError(f"Unsupported action {action}")
        cards: Tuple[int, int] = infoset.metadata["cards"]
        history: str = infoset.metadata["history"]
        player: int = infoset.metadata["player"]

        next_history = history + ACTION_SYMBOLS[action]
        terminal = next_history in HISTORY_TERMINALS
        if terminal:
            metadata = {
                "cards": cards,
                "history": next_history,
                "player": player,
                "terminal": True,
            }
            return InfoSet(key=self._infoset_key(metadata), metadata=metadata)

        next_metadata = {
            "cards": cards,
            "history": next_history,
            "player": 1 - player,
            "terminal": False,
        }
        return InfoSet(key=self._infoset_key(next_metadata), metadata=next_metadata)

    def payoff_at_terminal(self, infoset: InfoSet, player: int) -> float:
        return self.get_payoff(
            {
                "cards": infoset.metadata["cards"],
                "history": infoset.metadata["history"],
                "terminal": True,
            },
            player,
        )

    def clone(self) -> "KuhnEnv":
        clone = KuhnEnv()
        clone._rng = np.random.default_rng()
        clone._state = self._state
        clone._current_player = self._current_player
        clone._pot = self._pot
        return clone

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assert_state(self) -> None:
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset().")

    def _compute_payoff(self, acting_player: int) -> float:
        assert self._state is not None
        cards = self._state.cards
        history = self._state.history
        if history == "pp":
            return 1.0 if cards[acting_player] > cards[1 - acting_player] else -1.0
        if history == "bp":
            return 1.0 if acting_player == 0 else -1.0
        if history == "pbp":
            return -1.0 if acting_player == 0 else 1.0
        if history in {"bb", "pbb"}:
            return 2.0 if cards[acting_player] > cards[1 - acting_player] else -2.0
        raise ValueError(f"Invalid terminal history {history}")

    def _enumerate_from_state(self, state: Dict[str, object], infosets: Dict[str, InfoSet]) -> None:
        history: str = state["history"]
        cards: Tuple[int, int] = state["cards"]
        terminal = history in HISTORY_TERMINALS
        current_player = len(history) % 2

        metadata = {
            "cards": cards,
            "history": history,
            "player": current_player,
            "terminal": terminal,
        }
        key = self._infoset_key(metadata)
        if key not in infosets:
            infosets[key] = InfoSet(key=key, metadata=metadata)

        if terminal:
            return

        for action in (ACTION_PASS, ACTION_BET):
            next_history = history + ACTION_SYMBOLS[action]
            next_state = {
                "cards": cards,
                "history": next_history,
                "terminal": next_history in HISTORY_TERMINALS,
            }
            self._enumerate_from_state(next_state, infosets)

    def _infoset_key(self, metadata: Dict[str, object]) -> str:
        history = metadata["history"]
        cards = metadata["cards"]
        player = metadata["player"]
        return f"P{player}|C{cards[player]}|H{history}"

    # ------------------------------------------------------------------
    # Legacy tree-based API for CFR agents
    # ------------------------------------------------------------------
    def get_initial_state(self) -> Dict[str, object]:
        return {"cards": None, "history": "", "terminal": False}

    def get_current_player(self, state: Dict[str, object]) -> int:
        return len(state["history"]) % self.num_players

    def get_legal_actions_from_state(self, state: Dict[str, object]) -> List[int]:
        if self.is_terminal(state):
            return []
        history = state["history"]
        if history in {"", "p", "b"}:
            return [ACTION_PASS, ACTION_BET]
        if history == "pb":
            return [ACTION_PASS, ACTION_BET]
        raise ValueError(f"Invalid history encountered: {history}")

    # Backwards compatibility shim
    def get_legal_actions(self, state: Dict[str, object]) -> List[int]:  # type: ignore[override]
        return self.get_legal_actions_from_state(state)

    def get_next_state(self, state: Dict[str, object], action: int) -> Dict[str, object]:
        if action not in (ACTION_PASS, ACTION_BET):
            raise ValueError(f"Unknown action {action}")
        action_char = ACTION_SYMBOLS[action]
        history = state["history"] + action_char
        terminal = history in HISTORY_TERMINALS
        return {
            "cards": state.get("cards"),
            "history": history,
            "terminal": terminal,
        }

    def is_terminal(self, state: Dict[str, object]) -> bool:
        return state["history"] in HISTORY_TERMINALS

    def get_payoff(self, state: Dict[str, object], player: int) -> float:
        history = state["history"]
        cards = state["cards"]
        if cards is None:
            raise ValueError("Payoff requested before cards are dealt")
        if history == "pp":
            return 1.0 if cards[player] > cards[1 - player] else -1.0
        if history == "bp":
            return 1.0 if player == 0 else -1.0
        if history == "pbp":
            return -1.0 if player == 0 else 1.0
        if history in {"bb", "pbb"}:
            return 2.0 if cards[player] > cards[1 - player] else -2.0
        raise ValueError(f"Payoff requested for non-terminal history: {history}")

    def get_state_string(self, state: Dict[str, object]) -> str:
        if state["cards"] is None:
            return f"Card:?-Hist:{state['history']}"
        player = self.get_current_player(state)
        card = state["cards"][player]
        return f"Card:{card}-Hist:{state['history']}"

    def get_info_set_size(self) -> int:
        return self.get_obs_dim()

    def get_info_set(self, state: Dict[str, object]) -> np.ndarray:
        player = self.get_current_player(state)
        return self._info_vector_from_state(state, player)

    def sample_chance_outcome(self, state: Dict[str, object]) -> Dict[str, object]:
        cards = list(range(self.num_cards))
        self._rng.shuffle(cards)
        return {
            "cards": (cards[0], cards[1]),
            "history": state["history"],
            "terminal": False,
        }

    def get_all_chance_outcomes(self) -> List[Dict[str, object]]:
        outcomes = []
        for cards in itertools.permutations(range(self.num_cards), 2):
            outcomes.append({"cards": cards, "history": "", "terminal": False})
        return outcomes

    @staticmethod
    def _info_vector_from_state(state: Dict[str, object], player: int) -> np.ndarray:
        vector = np.zeros(KuhnEnv.num_cards + 4, dtype=np.float32)
        cards = state["cards"]
        if cards is None:
            return vector
        vector[cards[player]] = 1.0
        history_map = {"": 0, "p": 1, "b": 2, "pb": 3}
        history = state["history"]
        prefix = history[:2]
        key = prefix if prefix in history_map else history
        if key in history_map:
            vector[KuhnEnv.num_cards + history_map[key]] = 1.0
        return vector


# Backwards compatibility for legacy imports
KuhnPoker = KuhnEnv
