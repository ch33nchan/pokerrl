"""Canonical Leduc Hold'em environment implementing the BaseGameEnv API."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from qagent.environments.base import BaseGameEnv, InfoSet


ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_BET = 2


@dataclass(frozen=True)
class LeducState:
    deck: Tuple[int, ...]
    private_cards: Tuple[int, int]
    board_card: int | None
    history: Tuple[Tuple[int, int], ...]
    round: int
    current_player: int
    bets: Tuple[int, int]
    pot_contributions: Tuple[int, int]
    folded_player: int | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "deck": self.deck,
            "private_cards": self.private_cards,
            "board_card": self.board_card,
            "history": self.history,
            "round": self.round,
            "current_player": self.current_player,
            "bets": self.bets,
            "pot_contributions": self.pot_contributions,
            "folded_player": self.folded_player,
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "LeducState":
        if data.get("private_cards") is None:
            raise ValueError("private_cards must be specified to construct LeducState")

        deck = tuple(int(card) for card in data.get("deck", ()))
        private_cards = tuple(int(card) for card in data["private_cards"])
        history_raw = data.get("history", ())
        history = tuple((int(player), int(action)) for player, action in history_raw)
        bets_raw = data.get("bets", (1, 1))
        pot_raw = data.get("pot_contributions", (1, 1))

        return LeducState(
            deck=deck,
            private_cards=private_cards,
            board_card=data.get("board_card"),
            history=history,
            round=int(data.get("round", 0)),
            current_player=int(data.get("current_player", 0)),
            bets=tuple(int(b) for b in bets_raw),
            pot_contributions=tuple(int(p) for p in pot_raw),
            folded_player=data.get("folded_player"),
        )


class LeducEnv(BaseGameEnv):
    num_players: int = 2
    num_ranks: int = 3
    num_suits: int = 2
    deck_size: int = num_ranks * num_suits
    num_actions_val: int = 3
    bet_sizes: Tuple[int, int] = (2, 4)
    max_bet_per_round: int = 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._state: LeducState | None = None
        self.reset(seed)

    # ------------------------------------------------------------------
    # BaseGameEnv interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        deck = np.arange(self.deck_size)
        self._rng.shuffle(deck)
        private_cards = (int(deck[0]), int(deck[1]))
        remaining = tuple(int(card) for card in deck[2:])
        bets = (1, 1)
        state = LeducState(
            deck=remaining,
            private_cards=private_cards,
            board_card=None,
            history=(),
            round=0,
            current_player=0,
            bets=bets,
            pot_contributions=bets,
            folded_player=None,
        )
        self._state = state
        return self.get_info_set_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        self._assert_state()
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action} for history {self._state.history}")

        state = self._state
        current_player = state.current_player
        opponent = 1 - current_player
        history = state.history + ((current_player, action),)
        bets = list(state.bets)
        pot = list(state.pot_contributions)
        folded = state.folded_player
        board_card = state.board_card
        round_ = state.round
        deck = state.deck

        if action == ACTION_FOLD:
            folded = current_player
        elif action == ACTION_CALL:
            if bets[opponent] > bets[current_player]:
                diff = bets[opponent] - bets[current_player]
                bets[current_player] += diff
                pot[current_player] += diff
        elif action == ACTION_BET:
            bet_amt = self.bet_sizes[round_]
            if bets[opponent] > bets[current_player]:
                total = bets[opponent] + bet_amt
                diff = total - bets[current_player]
            else:
                diff = bet_amt
            bets[current_player] += diff
            pot[current_player] += diff

        bets_tuple = (bets[0], bets[1])
        pot_tuple = (pot[0], pot[1])

        if folded is None and self._betting_round_over(history, pot_tuple, round_):
            if round_ == 0 and board_card is None:
                if not deck:
                    raise RuntimeError("No cards left to deal board card.")
                board_card = deck[0]
                deck = tuple(c for c in deck if c != board_card)
                round_ = 1
                history = ()
                bets_tuple = (bets_tuple[0], bets_tuple[1])
                bets_tuple = tuple(pot_tuple)
            else:
                current_player = 0
        else:
            current_player = opponent

        terminal = folded is not None or (round_ == 1 and self._betting_round_over(history, pot_tuple, round_))

        self._state = LeducState(
            deck=deck,
            private_cards=state.private_cards,
            board_card=board_card,
            history=history,
            round=round_,
            current_player=current_player,
            bets=bets_tuple,
            pot_contributions=pot_tuple,
            folded_player=folded,
        )

        if terminal:
            payoff = self._payoff(self._state, current_player)
            info = {
                "terminal": True,
                "history": self._state.history,
                "board_card": self._state.board_card,
            }
            return self.get_info_set_vector(), payoff, True, info

        info = {"history": self._state.history}
        return self.get_info_set_vector(), 0.0, False, info

    def legal_actions(self) -> List[int]:
        self._assert_state()
        state = self._state
        if self._is_terminal_state(state):
            return []
        actions: List[int] = []
        bets = state.bets
        opponent = 1 - state.current_player
        if bets[opponent] > bets[state.current_player]:
            actions.append(ACTION_FOLD)
        actions.append(ACTION_CALL)
        if self._raises_this_round(state.history) < self.max_bet_per_round:
            actions.append(ACTION_BET)
        return actions

    def get_info_set_vector(self) -> np.ndarray:
        self._assert_state()
        state = self._state
        player = state.current_player
        vec = np.zeros(self.get_obs_dim(), dtype=np.float32)
        card_rank = state.private_cards[player] % self.num_ranks
        vec[card_rank] = 1.0
        board_offset = self.num_ranks
        if state.board_card is None:
            vec[board_offset + self.num_ranks] = 1.0
        else:
            vec[board_offset + (state.board_card % self.num_ranks)] = 1.0
        history_offset = board_offset + self.num_ranks + 1
        vec[history_offset] = (state.pot_contributions[player] - 1) / 10.0
        vec[history_offset + 1] = (state.pot_contributions[1 - player] - 1) / 10.0
        vec[history_offset + 2] = state.round
        return vec

    def get_action_sequence(self) -> Sequence[int]:
        self._assert_state()
        return [action for _, action in self._state.history]

    def get_obs_dim(self) -> int:
        return self.num_ranks + (self.num_ranks + 1) + 3

    def num_actions(self) -> int:
        return self.num_actions_val

    # ------------------------------------------------------------------
    # Enumeration utilities
    # ------------------------------------------------------------------
    def enumerate_info_sets(self) -> Iterable[InfoSet]:
        infosets: Dict[str, InfoSet] = {}
        for card_pair in itertools.permutations(range(self.deck_size), 2):
            remaining = tuple(c for c in range(self.deck_size) if c not in card_pair)
            initial = LeducState(
                deck=remaining,
                private_cards=(card_pair[0], card_pair[1]),
                board_card=None,
                history=(),
                round=0,
                current_player=0,
                bets=(1, 1),
                pot_contributions=(1, 1),
                folded_player=None,
            )
            self._enumerate_from_state(initial, infosets)
        return infosets.values()

    def info_set_to_vector(self, infoset: InfoSet) -> np.ndarray:
        metadata = infoset.metadata
        player = metadata["player"]
        private_cards: Tuple[int, int] = metadata["private_cards"]
        board_card = metadata["board_card"]
        pot: Tuple[int, int] = metadata["pot"]
        round_ = metadata["round"]
        vec = np.zeros(self.get_obs_dim(), dtype=np.float32)
        vec[private_cards[player] % self.num_ranks] = 1.0
        board_offset = self.num_ranks
        if board_card is None:
            vec[board_offset + self.num_ranks] = 1.0
        else:
            vec[board_offset + (board_card % self.num_ranks)] = 1.0
        history_offset = board_offset + self.num_ranks + 1
        vec[history_offset] = (pot[player] - 1) / 10.0
        vec[history_offset + 1] = (pot[1 - player] - 1) / 10.0
        vec[history_offset + 2] = round_
        return vec

    def transition_from_infoset(self, infoset: InfoSet, action: int) -> InfoSet | None:
        metadata = infoset.metadata
        private_cards = metadata["private_cards"]
        pot = metadata["pot"]
        history = metadata["history"]
        player = metadata["player"]
        board_card = metadata["board_card"]
        round_ = metadata["round"]
        deck = metadata["deck"]

        opponent = 1 - player
        bets = list(metadata["bets"])
        pot_list = list(pot)
        history_new = history + ((player, action),)
        folded = metadata["folded"]

        if action == ACTION_FOLD:
            folded = player
        elif action == ACTION_CALL and bets[opponent] > bets[player]:
            diff = bets[opponent] - bets[player]
            bets[player] += diff
            pot_list[player] += diff
        elif action == ACTION_BET:
            bet_amt = self.bet_sizes[round_]
            if bets[opponent] > bets[player]:
                total = bets[opponent] + bet_amt
                diff = total - bets[player]
            else:
                diff = bet_amt
            bets[player] += diff
            pot_list[player] += diff

        bets_tuple = (bets[0], bets[1])
        pot_tuple = (pot_list[0], pot_list[1])
        current_player = opponent
        next_round = round_
        next_board = board_card
        next_history = history_new
        next_deck = deck

        if folded is None and self._betting_round_over(next_history, pot_tuple, round_):
            if round_ == 0 and board_card is None:
                for card in deck:
                    next_board = card
                    next_deck = tuple(c for c in deck if c != card)
                    break
                next_round = 1
                next_history = ()
                bets_tuple = pot_tuple
            else:
                current_player = 0

        terminal = folded is not None or (next_round == 1 and self._betting_round_over(next_history, pot_tuple, next_round))

        next_metadata = {
            "private_cards": private_cards,
            "board_card": next_board,
            "pot": pot_tuple,
            "round": next_round,
            "player": current_player,
            "history": next_history,
            "deck": next_deck,
            "folded": folded,
            "bets": bets_tuple,
        }

        key = self._infoset_key(next_metadata)
        if terminal:
            next_metadata["terminal"] = True
        return InfoSet(key=key, metadata=next_metadata)

    def payoff_at_terminal(self, infoset: InfoSet, player: int) -> float:
        metadata = infoset.metadata
        if metadata.get("folded") is not None:
            return metadata["pot"][1 - player] if metadata["folded"] != player else -metadata["pot"][player]
        board = metadata["board_card"]
        if board is None:
            raise ValueError("Terminal infoset without board card")
        pot = metadata["pot"]
        private = metadata["private_cards"]
        if private[player] % self.num_ranks == board % self.num_ranks:
            return pot[1 - player]
        if private[1 - player] % self.num_ranks == board % self.num_ranks:
            return -pot[player]
        if private[player] % self.num_ranks > private[1 - player] % self.num_ranks:
            return pot[1 - player]
        if private[player] % self.num_ranks < private[1 - player] % self.num_ranks:
            return -pot[player]
        return 0.0

    def clone(self) -> "LeducEnv":
        clone = LeducEnv()
        clone._rng = np.random.default_rng()
        clone._state = self._state
        return clone

    # ------------------------------------------------------------------
    # Legacy CFR-style helper methods (for agents relying on dict states)
    # ------------------------------------------------------------------
    def get_initial_state(self) -> Dict[str, object]:
        return {
            "deck": tuple(range(self.deck_size)),
            "private_cards": None,
            "board_card": None,
            "history": (),
            "round": 0,
            "current_player": 0,
            "bets": (1, 1),
            "pot_contributions": (1, 1),
            "folded_player": None,
        }

    def get_state_string(self, state: Dict[str, object]) -> str:
        player = state.get("current_player", 0)
        history = state.get("history", ())
        history_str = "".join(f"{p}{a}" for p, a in history)
        private_cards = state.get("private_cards")
        card_repr = "??"
        if private_cards is not None:
            card_repr = f"{private_cards[player]}|{private_cards[1 - player]}"
        board = state.get("board_card")
        board_repr = "N" if board is None else str(board)
        return f"P{player}-C{card_repr}-B{board_repr}-H{history_str}-R{state.get('round', 0)}"

    def get_current_player(self, state: Dict[str, object]) -> int:
        return int(state.get("current_player", 0))

    def get_legal_actions(self, state: Dict[str, object]) -> List[int]:  # type: ignore[override]
        leduc_state = LeducState.from_dict(state)
        self_state = self._state
        self._state = leduc_state
        actions = self.legal_actions()
        self._state = self_state
        return actions

    def get_next_state(self, state: Dict[str, object], action: int) -> Dict[str, object]:
        if state.get("private_cards") is None or any(card is None for card in state.get("private_cards", ())):
            raise ValueError("Cannot apply actions before private cards are dealt")
        leduc_state = LeducState.from_dict(state)
        self_state = self._state
        self._state = leduc_state
        _, _, _, _ = self.step(action)
        next_state = self._state.to_dict()
        self._state = self_state
        return next_state

    def is_terminal(self, state: Dict[str, object]) -> bool:
        private_cards = state.get("private_cards")
        if private_cards is None or any(card is None for card in private_cards):
            return False
        leduc_state = LeducState.from_dict(state)
        return self._is_terminal_state(leduc_state)

    def get_payoff(self, state: Dict[str, object], player: int) -> float:
        leduc_state = LeducState.from_dict(state)
        return self._payoff(leduc_state, player)

    def sample_chance_outcome(self, state: Dict[str, object]) -> Tuple[float, Dict[str, object]]:
        private_cards = state.get("private_cards")

        if private_cards is None or any(card is None for card in private_cards):
            deck = list(state.get("deck", tuple(range(self.deck_size))))
            self._rng.shuffle(deck)
            dealt_private = (int(deck[0]), int(deck[1]))
            remaining = tuple(int(card) for card in deck[2:])
            next_state = {
                "deck": remaining,
                "private_cards": dealt_private,
                "board_card": state.get("board_card"),
                "history": state.get("history", ()),
                "round": state.get("round", 0),
                "current_player": state.get("current_player", 0),
                "bets": state.get("bets", (1, 1)),
                "pot_contributions": state.get("pot_contributions", (1, 1)),
                "folded_player": state.get("folded_player"),
            }
            return 1.0, next_state

        # No stochastic outcome required; return state unchanged
        return 1.0, state

    def is_chance_node(self, state: Dict[str, object]) -> bool:
        private_cards = state.get("private_cards")
        if private_cards is None:
            return True
        if any(card is None for card in private_cards):
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assert_state(self) -> None:
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset().")

    def _raises_this_round(self, history: Sequence[Tuple[int, int]]) -> int:
        return sum(1 for _, action in history if action == ACTION_BET)

    def _betting_round_over(self, history: Sequence[Tuple[int, int]], pot: Tuple[int, int], round_: int) -> bool:
        if not history:
            return False
        if pot[0] != pot[1]:
            return False
        last_player, last_action = history[-1]
        if last_action == ACTION_CALL:
            return True
        if len(history) >= 2:
            prev_player, prev_action = history[-2]
            if last_action == ACTION_CALL and prev_action == ACTION_BET:
                return True
            if last_action == ACTION_CALL and prev_action == ACTION_CALL and last_player != prev_player:
                return True
        if round_ == 0 and all(action == ACTION_CALL for _, action in history):
            return True
        return False

    def _is_terminal_state(self, state: LeducState) -> bool:
        if state.folded_player is not None:
            return True
        if state.round == 1 and self._betting_round_over(state.history, state.pot_contributions, state.round):
            return True
        return False

    def _payoff(self, state: LeducState, player: int) -> float:
        if state.folded_player is not None:
            return state.pot_contributions[1 - player] if state.folded_player != player else -state.pot_contributions[player]
        board = state.board_card
        if board is None:
            raise ValueError("Showdown without board card")
        private = state.private_cards
        if private[player] % self.num_ranks == board % self.num_ranks:
            return state.pot_contributions[1 - player]
        if private[1 - player] % self.num_ranks == board % self.num_ranks:
            return -state.pot_contributions[player]
        if private[player] % self.num_ranks > private[1 - player] % self.num_ranks:
            return state.pot_contributions[1 - player]
        if private[player] % self.num_ranks < private[1 - player] % self.num_ranks:
            return -state.pot_contributions[player]
        return 0.0

    def _enumerate_from_state(self, state: LeducState, infosets: Dict[str, InfoSet]) -> None:
        metadata = {
            "private_cards": state.private_cards,
            "board_card": state.board_card,
            "history": state.history,
            "round": state.round,
            "player": state.current_player,
            "deck": state.deck,
            "pot": state.pot_contributions,
            "folded": state.folded_player,
            "bets": state.bets,
        }
        key = self._infoset_key(metadata)
        if key not in infosets:
            infosets[key] = InfoSet(key=key, metadata=metadata)

        if self._is_terminal_state(state):
            return

        for action in (ACTION_FOLD, ACTION_CALL, ACTION_BET):
            if action not in self.legal_actions_state(state):
                continue
            next_infoset = self.transition_from_infoset(infosets[key], action)
            if next_infoset is None:
                continue
            if next_infoset.key not in infosets:
                infosets[next_infoset.key] = next_infoset
            next_state = self._state_from_metadata(next_infoset.metadata)
            self._enumerate_from_state(next_state, infosets)

    def legal_actions_state(self, state: LeducState) -> List[int]:
        self_state = self._state
        self._state = state
        actions = self.legal_actions()
        self._state = self_state
        return actions

    def _infoset_key(self, metadata: Dict[str, object]) -> str:
        board = metadata["board_card"]
        board_str = "N" if board is None else str(board % self.num_ranks)
        return f"P{metadata['player']}|C{metadata['private_cards'][metadata['player']] % self.num_ranks}|B{board_str}|R{metadata['round']}|H{len(metadata['history'])}"

    def _state_from_metadata(self, metadata: Dict[str, object]) -> LeducState:
        return LeducState(
            deck=metadata["deck"],
            private_cards=metadata["private_cards"],
            board_card=metadata["board_card"],
            history=metadata["history"],
            round=metadata["round"],
            current_player=metadata["player"],
            bets=metadata["bets"],
            pot_contributions=metadata["pot"],
            folded_player=metadata["folded"],
        )


# Backwards compatibility
LeducHoldem = LeducEnv

def get_num_actions(self):
    return self.num_actions


def get_state_shape(self):
    # Shape of the state representation
    # [player_id, round, pot, player_card, board_card, betting_history]
    return 6
