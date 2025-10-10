"""Kuhn Poker game wrapper and utilities."""

import numpy as np
from typing import Any, List, Optional
try:
    import pyspiel
except ImportError:
    pyspiel = None

from .base import GameWrapper


class KuhnPokerWrapper(GameWrapper):
    """Wrapper for Kuhn Poker game using OpenSpiel."""

    def __init__(self):
        """Initialize Kuhn Poker wrapper."""
        super().__init__("kuhn_poker")
        self.num_cards = 3
        self.num_actions = 2  # Check/Call, Bet/Fold
        self.card_rank_to_string = {0: "J", 1: "Q", 2: "K"}

    def get_initial_state(self) -> Any:
        """Get the initial state of Kuhn Poker."""
        return self.game.new_initial_state()

    def encode_state(self, state: Any) -> np.ndarray:
        """Encode Kuhn Poker state into a vector.

        Encoding includes:
        - Private card (one-hot)
        - Public cards seen (one-hot)
        - Player to act (one-hot)
        - Betting round (one-hot)
        - Legal actions mask
        - Pot size normalized

        Args:
            state: OpenSpiel game state

        Returns:
            Encoded state vector
        """
        # Use OpenSpiel's information state tensor
        tensor = state.information_state_tensor()
        return np.array(tensor, dtype=np.float32)

    def decode_state(self, encoding: np.ndarray) -> dict:
        """Decode state vector back to human-readable representation.

        Args:
            encoding: State encoding vector

        Returns:
            Dictionary with decoded state information
        """
        # This is a simplified decoding for debugging/analysis
        info = {
            "player_to_act": int(encoding[0]),
            "private_card": self._decode_private_card(encoding[1:4]),
            "public_info": encoding[4:7].tolist(),
            "betting_round": int(encoding[7]),
            "pot_size": encoding[8]
        }
        return info

    def _decode_private_card(self, card_encoding: np.ndarray) -> str:
        """Decode private card from one-hot encoding."""
        card_idx = np.argmax(card_encoding)
        return self.card_rank_to_string.get(card_idx, "Unknown")

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        if action == 0:
            return "Check/Call"
        elif action == 1:
            return "Bet/Fold"
        else:
            return f"Action_{action}"

    def get_state_description(self, state: Any) -> str:
        """Get human-readable description of current state.

        Args:
            state: OpenSpiel game state

        Returns:
            Description string
        """
        if state.is_terminal():
            return f"Terminal. Returns: {state.returns()}"

        current_player = state.current_player()
        if current_player == pyspiel.PlayerId.TERMINAL:
            return "Game over"

        # Get private card for current player
        try:
            private_card = state.information_state_string(current_player).split()[0]
        except:
            private_card = "?"

        legal_actions = state.legal_actions()
        action_names = [self.get_action_name(action) for action in legal_actions]

        return (f"Player {current_player} (Card: {private_card}) - "
                f"Legal actions: {', '.join(action_names)}")

    def simulate_hand(self, seed: Optional[int] = None) -> List[dict]:
        """Simulate a single hand for debugging.

        Args:
            seed: Random seed for reproducibility

        Returns:
            List of state transitions
        """
        if seed is not None:
            np.random.seed(seed)

        state = self.get_initial_state()
        history = []

        while not state.is_terminal():
            # Encode current state
            encoding = self.encode_state(state)
            description = self.get_state_description(state)

            history.append({
                "encoding": encoding.copy(),
                "description": description,
                "player": state.current_player(),
                "legal_actions": state.legal_actions()
            })

            # Random action
            action = np.random.choice(state.legal_actions())
            state.apply_action(action)

        # Add terminal state
        history.append({
            "encoding": self.encode_state(state),
            "description": self.get_state_description(state),
            "returns": state.returns(),
            "terminal": True
        })

        return history