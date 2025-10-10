
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from qagent.poker_environment import PokerEnvironment

class OpponentPolicy:
    """
    Base interface for all opponent strategies.
    
    An opponent policy must implement the `decide` method, which takes the
    current environment state and a list of valid actions, and returns the
    chosen action.
    """

    def decide(self, valid_actions: Sequence[int], env: "PokerEnvironment") -> int:
        """
        Decides on an action to take based on the game state.

        Args:
            valid_actions: A sequence of integers representing the legal actions.
            env: The current instance of the PokerEnvironment, providing state access.

        Returns:
            The integer representing the chosen action.
        """
        raise NotImplementedError("Each opponent policy must implement the 'decide' method.")

    def on_hand_end(self, winner: str | None = None):
        """
        Optional method to be called at the end of a hand.
        Useful for stateful/adaptive bots to reset or update their internal state.

        Args:
            winner: A string indicating the winner of the hand ('agent' or 'opponent').
        """
        pass
