"""Leduc Hold'em Poker game wrapper and utilities."""

import numpy as np
from typing import Any, List, Optional
try:
    import pyspiel
except ImportError:
    pyspiel = None

from .base import GameWrapper


class LeducPokerWrapper(GameWrapper):
    """Wrapper for Leduc Hold'em Poker game using OpenSpiel."""

    def __init__(self):
        """Initialize Leduc Hold'em wrapper."""
        super().__init__("leduc_holdem")
        self.num_cards = 6  # Pairs: JJ, QQ, KK
        self.num_actions = 2  # Check/Call, Bet/Fold
        self.card_rank_to_string = {0: "J", 1: "Q", 2: "K"}

    def get_initial_state(self) -> Any:
        """Get the initial state of Leduc Hold'em."""
        return self.game.new_initial_state()

    def encode_state(self, state: Any) -> np.ndarray:
        """Encode Leduc Hold'em state into a vector.

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
        info = {
            "player_to_act": int(encoding[0]),
            "private_card": self._decode_private_card(encoding[1:4]),
            "public_card": self._decode_public_card(encoding[4:7]) if np.any(encoding[4:7]) else None,
            "betting_round": int(encoding[7]),
            "pot_size": encoding[8],
            "player_chips": encoding[9:11].tolist()
        }
        return info

    def _decode_private_card(self, card_encoding: np.ndarray) -> str:
        """Decode private card from one-hot encoding."""
        card_idx = np.argmax(card_encoding)
        return self.card_rank_to_string.get(card_idx, "Unknown")

    def _decode_public_card(self, card_encoding: np.ndarray) -> str:
        """Decode public card from one-hot encoding."""
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

        # Get information state string for current player
        try:
            info_str = state.information_state_string(current_player)
            parts = info_str.split()
            private_card = parts[0] if parts else "?"
            public_card = parts[1] if len(parts) > 1 else "None"
        except:
            private_card = "?"
            public_card = "None"

        legal_actions = state.legal_actions()
        action_names = [self.get_action_name(action) for action in legal_actions]

        # Determine betting round
        betting_round = "pre-flop" if public_card == "None" else "post-flop"

        return (f"Player {current_player} (Private: {private_card}, "
                f"Public: {public_card}, Round: {betting_round}) - "
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