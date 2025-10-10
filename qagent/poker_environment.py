"""Heads-up poker environment with enriched observations and opponent variety."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qagent.opponents.base import OpponentPolicy
from qagent.opponents.bots import (
    RandomBot, CallBot, TightAggressiveBot, LoosePassiveBot, 
    MixedStrategyBot, AdaptiveBot
)


HISTORY_LENGTH = 6
MAX_RANK = 14

@dataclass
class EnvironmentConfig:
    initial_stack: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    min_raise: int = 10
    max_stage_raises: int = 3
    opponent_policy: OpponentPolicy = None
    seed: Optional[int] = None


class PokerEnvironment:
    """Heads-up limit Hold'em environment with structured observations."""

    ACTION_FOLD = 0
    ACTION_CALL = 1
    ACTION_RAISE = 2
    ACTION_CHECK = 3

    STAGES = ("preflop", "flop", "turn", "river", "showdown")

    def __init__(self, config: Optional[EnvironmentConfig] = None, **legacy_kwargs):
        if config is None:
            config = EnvironmentConfig()
        for key, value in legacy_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.config = config

        if self.config.opponent_policy is None:
            self.config.opponent_policy = RandomBot()

        self.rng = np.random.default_rng(config.seed)
        self.opponent_policy = self.config.opponent_policy

        self.state_dim = 0
        self.action_dim = 4

        self.deck: List[Tuple[int, int]] = []
        self.community_cards: List[Tuple[int, int]] = []
        self.agent_hole: List[Tuple[int, int]] = []
        self.opponent_hole: List[Tuple[int, int]] = []

        self.agent_stack: int = config.initial_stack
        self.opponent_stack: int = config.initial_stack
        self.agent_bet: int = 0
        self.opponent_bet: int = 0
        self.current_bet: int = 0
        self.pot: int = 0

        self.stage: int = 0
        self.stage_actions: int = 0
        self.stage_raises: int = 0
        self.stage_raise_history: List[int] = [0 for _ in range(4)]
        self.total_raises: int = 0
        self.raise_available: bool = True
        self.current_player: str = "agent"
        self.last_action: int = -1
        self.action_history: List[int] = []
        self.actor_history: List[int] = []
        self.opponent_action_counts: Dict[int, int] = {
            self.ACTION_FOLD: 0,
            self.ACTION_CALL: 0,
            self.ACTION_RAISE: 0,
            self.ACTION_CHECK: 0,
        }

        self.hand_over: bool = False
        self.winner: Optional[str] = None
        self.terminal_reward: float = 0.0

    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self._initialise_deck()
        self.community_cards = []
        self.agent_hole = [self.deck.pop(), self.deck.pop()]
        self.opponent_hole = [self.deck.pop(), self.deck.pop()]

        cfg = self.config
        self.agent_stack = cfg.initial_stack - cfg.small_blind
        self.opponent_stack = cfg.initial_stack - cfg.big_blind
        self.agent_bet = cfg.small_blind
        self.opponent_bet = cfg.big_blind
        self.current_bet = cfg.big_blind
        self.pot = cfg.small_blind + cfg.big_blind

        self.stage = 0
        self.stage_actions = 0
        self.stage_raises = 0
        self.stage_raise_history = [0 for _ in range(4)]
        self.total_raises = 0
        self.raise_available = True
        self.current_player = "agent"
        self.last_action = -1
        self.action_history = []
        self.actor_history = []
        self.opponent_action_counts = {
            self.ACTION_FOLD: 0,
            self.ACTION_CALL: 0,
            self.ACTION_RAISE: 0,
            self.ACTION_CHECK: 0,
        }

        self.hand_over = False
        self.winner = None
        self.terminal_reward = 0.0

        obs = self._build_observation()
        mask = self._action_mask("agent")
        info = {
            "action_mask": mask,
            "stage": self.stage,
            "pot": self.pot,
            "hand_strength": self.estimate_hand_strength("agent"),
        }
        return obs, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        if self.hand_over:
            raise RuntimeError("Hand already finished. Call reset() before stepping again.")
        if self.current_player != "agent":
            raise RuntimeError("It is not the agent's turn to act.")

        valid_actions = self.available_actions("agent")
        if action_idx not in valid_actions:
            raise ValueError(f"Illegal action {action_idx}. Valid actions: {valid_actions}")

        self._apply_action("agent", action_idx)
        if self.hand_over:
            self._finalise_hand()
            return self._terminal_observation(), self.terminal_reward, True, self._terminal_info()

        if self._stage_ready_to_advance():
            self._advance_stage()
            if self.hand_over:
                self._finalise_hand()
                return self._terminal_observation(), self.terminal_reward, True, self._terminal_info()

        self._opponent_turns()
        if self.hand_over:
            self._finalise_hand()
            return self._terminal_observation(), self.terminal_reward, True, self._terminal_info()

        obs = self._build_observation()
        mask = self._action_mask("agent")
        info = {
            "action_mask": mask,
            "stage": self.stage,
            "pot": self.pot,
            "hand_strength": self.estimate_hand_strength("agent"),
            "opponent_action_distribution": self._opponent_action_distribution(),
        }
        return obs, 0.0, False, info

    def available_actions(self, player: str) -> List[int]:
        mask = self._action_mask(player)
        return [idx for idx, flag in enumerate(mask) if flag == 1]

    def _build_opponent(self, name: str) -> OpponentPolicy:
        name = name.lower()
        if name in {"call", "callbot"}:
            return CallBot()
        if name in {"random", "randombot"}:
            return RandomBot()
        if name in {"loosepassive", "loose", "lp"}:
            return LoosePassiveBot()
        if name in {"tightaggressive", "ta", "tag"}:
            return TightAggressiveBot()
        if name in {"adaptive", "adaptivebot"}:
            return AdaptiveBot()
        raise ValueError(f"Unknown opponent type: {name}")

    def _initialise_deck(self) -> None:
        self.deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
        self.rng.shuffle(self.deck)

    def _draw_cards(self, count: int) -> List[Tuple[int, int]]:
        return [self.deck.pop() for _ in range(count)]

    def _to_call(self, player: str) -> int:
        if player == "agent":
            return max(0, self.current_bet - self.agent_bet)
        return max(0, self.current_bet - self.opponent_bet)

    def _stack(self, player: str) -> int:
        return self.agent_stack if player == "agent" else self.opponent_stack

    def _commit(self, player: str, amount: int) -> None:
        if amount <= 0:
            return
        if player == "agent":
            commit = min(amount, self.agent_stack)
            self.agent_stack -= commit
            self.agent_bet += commit
        else:
            commit = min(amount, self.opponent_stack)
            self.opponent_stack -= commit
            self.opponent_bet += commit
        self.pot += commit

    def _apply_action(self, player: str, action_idx: int) -> None:
        to_call = self._to_call(player)
        stack = self._stack(player)
        cfg = self.config

        if action_idx == self.ACTION_FOLD:
            self.hand_over = True
            self.winner = "opponent" if player == "agent" else "agent"
            self.last_action = action_idx
            return

        if action_idx == self.ACTION_CALL:
            self._commit(player, min(to_call, stack))
        elif action_idx == self.ACTION_CHECK:
            if to_call != 0:
                raise RuntimeError("Cannot check when there is an outstanding bet to call.")
        elif action_idx == self.ACTION_RAISE:
            call_amount = min(to_call, stack)
            self._commit(player, call_amount)
            remaining_stack = self._stack(player)
            if remaining_stack > 0 and remaining_stack >= cfg.min_raise:
                self._commit(player, cfg.min_raise)
                if player == "agent":
                    self.current_bet = self.agent_bet
                else:
                    self.current_bet = self.opponent_bet
                self.stage_raises += 1
                stage_index = min(self.stage, len(self.stage_raise_history) - 1)
                self.stage_raise_history[stage_index] += 1
                self.total_raises += 1
                if self.stage_raises >= cfg.max_stage_raises:
                    self.raise_available = False
            else:
                if player == "agent":
                    self.current_bet = max(self.current_bet, self.agent_bet)
                else:
                    self.current_bet = max(self.current_bet, self.opponent_bet)
                self.raise_available = False
        else:
            raise ValueError(f"Unknown action index {action_idx}")

        self.stage_actions += 1
        self.last_action = action_idx
        self.current_player = "opponent" if player == "agent" else "agent"

        self.action_history.append(action_idx)
        self.actor_history.append(0 if player == "agent" else 1)
        if len(self.action_history) > HISTORY_LENGTH:
            self.action_history.pop(0)
            self.actor_history.pop(0)

        if player == "opponent":
            self.opponent_action_counts[action_idx] += 1

        if self.agent_stack == 0 or self.opponent_stack == 0:
            self._reveal_remaining_cards()
            self._handle_showdown()

    def _opponent_turns(self) -> None:
        while not self.hand_over and self.current_player == "opponent":
            valid = self.available_actions("opponent")
            action_idx = self.opponent_policy.decide(valid, self)
            self._apply_action("opponent", action_idx)
            if self.hand_over:
                break
            if self._stage_ready_to_advance():
                self._advance_stage()
                break

    def _stage_ready_to_advance(self) -> bool:
        if self.hand_over:
            return False
        to_call_agent = self._to_call("agent")
        to_call_opp = self._to_call("opponent")
        return (
            to_call_agent == 0
            and to_call_opp == 0
            and self.stage_actions >= 2
            and self.current_player == "agent"
        )

    def _advance_stage(self) -> None:
        if self.hand_over:
            return
        self.stage += 1
        self.stage_actions = 0
        self.stage_raises = 0
        self.raise_available = True
        self.current_bet = 0
        self.agent_bet = 0
        self.opponent_bet = 0
        self.current_player = "agent"
        self.last_action = -1

        if self.stage == 1:
            self.community_cards.extend(self._draw_cards(3))
        elif self.stage == 2:
            self.community_cards.extend(self._draw_cards(1))
        elif self.stage == 3:
            self.community_cards.extend(self._draw_cards(1))
        elif self.stage >= 4:
            self._handle_showdown()

    def _reveal_remaining_cards(self) -> None:
        missing = 5 - len(self.community_cards)
        if missing > 0:
            self.community_cards.extend(self._draw_cards(missing))

    def _handle_showdown(self) -> None:
        self._reveal_remaining_cards()
        agent_strength = self._evaluate_hand(self.agent_hole + self.community_cards)
        opp_strength = self._evaluate_hand(self.opponent_hole + self.community_cards)
        if agent_strength > opp_strength:
            self.winner = "agent"
        elif agent_strength < opp_strength:
            self.winner = "opponent"
        else:
            self.winner = "tie"
        self.hand_over = True

    def _finalise_hand(self) -> None:
        if self.pot > 0:
            if self.winner == "agent":
                self.agent_stack += self.pot
            elif self.winner == "opponent":
                self.opponent_stack += self.pot
            else:
                split = self.pot // 2
                self.agent_stack += split
                self.opponent_stack += self.pot - split
            self.pot = 0
        self.terminal_reward = float(self.agent_stack - self.config.initial_stack)

    def _evaluate_hand(self, cards: Sequence[Tuple[int, int]]) -> Tuple:
        """Evaluates a 7-card hand and returns its rank."""
        if len(cards) < 5:
            return (0, tuple(sorted([c[0] for c in cards], reverse=True)))

        all_hands = self.rng.permutation(list(combinations(cards, 5)))

        best_rank = (0, (0,))

        for hand in all_hands:
            ranks = sorted([c[0] for c in hand], reverse=True)
            suits = {c[1] for c in hand}

            is_flush = len(suits) == 1
            rank_counts = {r: ranks.count(r) for r in ranks}
            sorted_counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

            is_straight = len(set(ranks)) == 5 and (ranks[0] - ranks[4] == 4)
            # Ace-low straight
            if ranks == [14, 5, 4, 3, 2]:
                is_straight = True
                ranks = [5, 4, 3, 2, 1]  # Treat Ace as 1 for ranking

            hand_rank = 0
            kickers = tuple(r for r, c in sorted_counts)

            # Straight flush
            if is_straight and is_flush:
                hand_rank = 8
            # Four of a kind
            elif sorted_counts[0][1] == 4:
                hand_rank = 7
                kickers = (sorted_counts[0][0], sorted_counts[1][0])
            # Full house
            elif sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
                hand_rank = 6
                kickers = (sorted_counts[0][0], sorted_counts[1][0])
            # Flush
            elif is_flush:
                hand_rank = 5
            # Straight
            elif is_straight:
                hand_rank = 4
            # Three of a kind
            elif sorted_counts[0][1] == 3:
                hand_rank = 3
                kickers = (sorted_counts[0][0],) + tuple(r for r, c in sorted_counts[1:])
            # Two pair
            elif sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2:
                hand_rank = 2
                kickers = (sorted_counts[0][0], sorted_counts[1][0], sorted_counts[2][0])
            # One pair
            elif sorted_counts[0][1] == 2:
                hand_rank = 1
                kickers = (sorted_counts[0][0],) + tuple(r for r, c in sorted_counts[1:])
            # High card
            else:
                hand_rank = 0

            current_rank = (hand_rank, kickers)
            if current_rank > best_rank:
                best_rank = current_rank

        return best_rank

    def _hand_strength(self, cards: Sequence[Tuple[int, int]]) -> float:
        """Returns a numerical score for hand strength for learning, not for showdown."""
        if not cards:
            return 0.0
        
        from itertools import combinations
        
        if len(cards) < 5:
            # Simplified pre-flop/flop/turn estimation
            ranks = sorted([c[0] for c in cards], reverse=True)
            score = sum(ranks) / (len(ranks) * MAX_RANK)
            if len(cards) > 1 and ranks[0] == ranks[1]:
                score += 0.2 # Bonus for a pair
            return score

        # For 5 or more cards, use a more robust evaluation
        best_rank = self._evaluate_hand(cards)
        
        # Normalize the rank into a float score
        score = best_rank[0]
        # Normalize kickers
        for i, kicker in enumerate(best_rank[1]):
            score += kicker / (MAX_RANK ** (i + 2))
        
        return score / 9.0 # Normalize by max hand rank

    def _action_mask(self, player: str) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.int32)
        to_call = self._to_call(player)
        stack = self._stack(player)
        opponent_stack = self._stack("agent" if player == "opponent" else "opponent")

        if to_call > 0:
            if stack > 0:
                mask[self.ACTION_CALL] = 1
            mask[self.ACTION_FOLD] = 1
            can_raise = (
                self.raise_available
                and stack > to_call
                and (stack - to_call) >= self.config.min_raise
                and opponent_stack > 0
            )
            if can_raise:
                mask[self.ACTION_RAISE] = 1
        else:
            mask[self.ACTION_CHECK] = 1
            can_raise = (
                self.raise_available
                and stack >= self.config.min_raise
                and opponent_stack > 0
            )
            if can_raise:
                mask[self.ACTION_RAISE] = 1
        return mask

    def _build_observation(self) -> np.ndarray:
        cfg = self.config
        max_stack = cfg.initial_stack * 2
        max_pot = cfg.initial_stack * 4

        to_call = self._to_call("agent")
        effective_stack = min(self.agent_stack, self.opponent_stack)
        pot_plus_call = self.pot + to_call

        features: List[float] = [
            self.agent_stack / max_stack,
            self.opponent_stack / max_stack,
            self.pot / max_pot,
            to_call / max_stack,
            (to_call / pot_plus_call) if pot_plus_call > 0 else 0.0,
            self.agent_stack / (self.pot + 1.0),
            effective_stack / max_stack,
            (self.agent_stack - self.opponent_stack) / max_stack,
            self.total_raises / float(cfg.max_stage_raises * len(self.STAGES)),
            self.stage_actions / 10.0,
            self.stage_raises / float(max(cfg.max_stage_raises, 1)),
            float(self.current_player == "agent"),
            1.0 if self.raise_available else 0.0,
        ]

        stage_one_hot = np.zeros(4, dtype=np.float32)
        stage_idx = min(self.stage, 3)
        if self.stage < 4:
            stage_one_hot[stage_idx] = 1.0
        features.extend(stage_one_hot.tolist())

        hole_ranks = sorted((rank for rank, _ in self.agent_hole), reverse=True)
        high, low = hole_ranks if hole_ranks else (0, 0)
        suited = 1.0 if len(self.agent_hole) == 2 and self.agent_hole[0][1] == self.agent_hole[1][1] else 0.0
        pair = 1.0 if len(self.agent_hole) == 2 and self.agent_hole[0][0] == self.agent_hole[1][0] else 0.0
        gap = abs(high - low) if hole_ranks else 0
        hand_strength_estimate = self.estimate_hand_strength("agent")

        features.extend([
            high / MAX_RANK,
            low / MAX_RANK,
            suited,
            pair,
            gap / 12.0,
            hand_strength_estimate,
        ])

        community_ranks = [rank for rank, _ in self.community_cards]
        community_features = np.zeros(5, dtype=np.float32)
        if community_ranks:
            community_features[: len(community_ranks)] = [rank / MAX_RANK for rank in community_ranks]
        features.extend(community_features.tolist())

        community_mask = np.zeros(5, dtype=np.float32)
        community_mask[: len(community_ranks)] = 1.0
        features.extend(community_mask.tolist())

        features.extend(self._encode_action_history())
        features.extend(self._opponent_action_distribution())

        obs = np.array(features, dtype=np.float32)
        if self.state_dim == 0:
            self.state_dim = obs.shape[0]
        return obs

    def _terminal_observation(self) -> np.ndarray:
        dim = self.state_dim if self.state_dim > 0 else 1
        return np.zeros(dim, dtype=np.float32)

    def _terminal_info(self) -> Dict[str, np.ndarray]:
        return {
            "action_mask": np.zeros(self.action_dim, dtype=np.int32),
            "stage": self.stage,
            "pot": self.pot,
            "hand_strength": self.estimate_hand_strength("agent"),
            "opponent_action_distribution": self._opponent_action_distribution(),
        }

    def _encode_action_history(self) -> List[float]:
        encoded: List[float] = []
        for idx in range(HISTORY_LENGTH):
            if idx < len(self.action_history):
                action = self.action_history[idx]
                actor = self.actor_history[idx]
                one_hot = np.zeros(self.action_dim, dtype=np.float32)
                if 0 <= action < self.action_dim:
                    one_hot[action] = 1.0
                encoded.extend(one_hot.tolist())
                encoded.append(float(actor))
            else:
                encoded.extend([0.0] * self.action_dim)
                encoded.append(0.0)
        return encoded

    def _opponent_action_distribution(self) -> List[float]:
        total = sum(self.opponent_action_counts.values())
        if total == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [self.opponent_action_counts[action] / total for action in range(self.action_dim)]

    def estimate_hand_strength(self, player: str) -> float:
        hole = self.agent_hole if player == "agent" else self.opponent_hole
        return self._estimate_hand_strength(hole, self.community_cards)

    def _estimate_hand_strength(self, hole: Sequence[Tuple[int, int]], community: Sequence[Tuple[int, int]]) -> float:
        if len(hole) < 2:
            return 0.0
        preflop_strength = self._preflop_strength(hole)
        if not community:
            return preflop_strength
        
        return self._hand_strength(list(hole) + list(community))

    def _preflop_strength(self, hole: Sequence[Tuple[int, int]]) -> float:
        ranks = sorted((rank for rank, _ in hole), reverse=True)
        high, low = ranks
        suited = 1 if hole[0][1] == hole[1][1] else 0
        pair = 1 if high == low else 0
        gap = abs(high - low)
        base = (high + low) / (2 * MAX_RANK)
        if pair:
            base += 0.25
        if suited:
            base += 0.05
        base -= min(gap, 4) * 0.03
        return float(np.clip(base, 0.0, 0.99))

    def get_state_dimension(self) -> int:
        if self.state_dim == 0:
            self.reset()
        return self.state_dim

    def get_action_dimension(self) -> int:
        return self.action_dim

    def set_opponent(self, opponent: OpponentPolicy):
        """Allows dynamically changing the opponent."""
        self.opponent_policy = opponent


__all__ = [
    "PokerEnvironment",
    "EnvironmentConfig",
    "CallBot",
    "RandomBot",
    "LoosePassiveBot",
    "TightAggressiveBot",
]
