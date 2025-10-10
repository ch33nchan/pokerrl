
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import numpy as np

from qagent.opponents.base import OpponentPolicy

if TYPE_CHECKING:
    from qagent.poker_environment import PokerEnvironment

class RandomBot(OpponentPolicy):
    """Selects a random legal action."""

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        idx = env.rng.integers(0, len(valid_actions))
        return valid_actions[int(idx)]

class CallBot(OpponentPolicy):
    """Always checks when possible, otherwise calls."""

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        if env.ACTION_CHECK in valid_actions:
            return env.ACTION_CHECK
        if env.ACTION_CALL in valid_actions:
            return env.ACTION_CALL
        return valid_actions[0]

class TightAggressiveBot(OpponentPolicy):
    """
    Plays a tight-aggressive (TAG) style.
    - Folds most hands pre-flop.
    - Raises with the top range of hands.
    - 3-bets only with premium holdings.
    """
    def __init__(self, raise_threshold=0.85, fold_threshold=0.25, premium_threshold=0.95):
        self.raise_threshold = raise_threshold  # Top 15% of hands
        self.fold_threshold = fold_threshold    # Fold bottom 25%
        self.premium_threshold = premium_threshold # Top 5% for 3-betting

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        strength = env.estimate_hand_strength("opponent")
        to_call = env._to_call("opponent")

        # Pre-flop logic
        if env.stage == 0:
            if env.stage_raises > 0 and strength < self.premium_threshold:
                if env.ACTION_FOLD in valid_actions:
                    return env.ACTION_FOLD # Fold to 3-bets without premium
            if strength > self.raise_threshold and env.ACTION_RAISE in valid_actions:
                return env.ACTION_RAISE
            if strength > self.fold_threshold and env.ACTION_CALL in valid_actions:
                return env.ACTION_CALL
            if strength > self.fold_threshold and env.ACTION_CHECK in valid_actions:
                return env.ACTION_CHECK
        
        # Post-flop logic
        else:
            if strength > self.raise_threshold and env.ACTION_RAISE in valid_actions:
                return env.ACTION_RAISE
            if strength > self.fold_threshold and env.ACTION_CALL in valid_actions:
                return env.ACTION_CALL
            if env.ACTION_CHECK in valid_actions:
                return env.ACTION_CHECK

        if env.ACTION_FOLD in valid_actions:
            return env.ACTION_FOLD
        
        return valid_actions[0] # Fallback

class LoosePassiveBot(OpponentPolicy):
    """
    Plays a loose-passive style (calling station).
    - Calls very frequently.
    - Rarely raises or folds.
    - Chases draws regardless of pot odds.
    """
    def __init__(self, call_threshold=0.1, raise_prob=0.05):
        self.call_threshold = call_threshold # Call with almost anything
        self.raise_prob = raise_prob # Very low probability of raising

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        strength = env.estimate_hand_strength("opponent")

        if env.rng.random() < self.raise_prob and env.ACTION_RAISE in valid_actions:
            return env.ACTION_RAISE

        if strength > self.call_threshold:
            if env.ACTION_CALL in valid_actions:
                return env.ACTION_CALL
            if env.ACTION_CHECK in valid_actions:
                return env.ACTION_CHECK
        
        if env.ACTION_CHECK in valid_actions:
            return env.ACTION_CHECK
        if env.ACTION_FOLD in valid_actions:
            return env.ACTION_FOLD
            
        return valid_actions[0] # Fallback

class MixedStrategyBot(OpponentPolicy):
    """
    Randomly switches between a set of strategies at the start of each hand.
    """
    def __init__(self, strategies: list[OpponentPolicy], probabilities: list[float]):
        self.strategies = strategies
        self.probabilities = probabilities
        self.current_strategy: OpponentPolicy = self.strategies[0]
        self.on_hand_end() # Initialize strategy

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        return self.current_strategy.decide(valid_actions, env)

    def on_hand_end(self):
        """Select a new strategy for the next hand."""
        self.current_strategy = np.random.choice(self.strategies, p=self.probabilities)
        self.current_strategy.on_hand_end()

class AdaptiveBot(OpponentPolicy):
    """
    Adjusts its strategy based on the agent's win rate.
    - If the agent is winning, it plays tighter.
    - If the agent is losing, it plays more aggressively to apply pressure.
    """
    def __init__(self, initial_aggression=0.5, adjustment_rate=0.05):
        self.aggression = initial_aggression
        self.adjustment_rate = adjustment_rate
        self.agent_wins = 0
        self.total_hands = 0

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        strength = env.estimate_hand_strength("opponent")
        
        raise_threshold = 0.9 - self.aggression * 0.4 # Varies from 0.9 to 0.5
        call_threshold = 0.5 - self.aggression * 0.3  # Varies from 0.5 to 0.2

        if strength > raise_threshold and env.ACTION_RAISE in valid_actions:
            return env.ACTION_RAISE
        if strength > call_threshold and env.ACTION_CALL in valid_actions:
            return env.ACTION_CALL
        if env.ACTION_CHECK in valid_actions:
            return env.ACTION_CHECK
        if env.ACTION_FOLD in valid_actions:
            return env.ACTION_FOLD
        
        return valid_actions[0]

    def on_hand_end(self, winner: str):
        self.total_hands += 1
        if winner == "agent":
            self.agent_wins += 1
        
        win_rate = self.agent_wins / self.total_hands if self.total_hands > 0 else 0.5
        
        # If agent is winning a lot, become more passive/tight
        if win_rate > 0.6:
            self.aggression = max(0.1, self.aggression - self.adjustment_rate)
        # If agent is losing, become more aggressive
        elif win_rate < 0.4:
            self.aggression = min(1.0, self.aggression + self.adjustment_rate)
