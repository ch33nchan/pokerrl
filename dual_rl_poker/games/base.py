"""Base classes for game wrappers and information state encoders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
try:
    import pyspiel
except ImportError:
    pyspiel = None


class GameWrapper(ABC):
    """Abstract base class for game wrappers."""

    def __init__(self, game_name: str):
        """Initialize the game wrapper.

        Args:
            game_name: Name of the OpenSpiel game
        """
        if pyspiel is None:
            raise ImportError("OpenSpiel (pyspiel) is required. Please install it first.")

        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        self.num_players = self.game.num_players()
        self.encoder = InfoStateEncoder(self.game)

    @abstractmethod
    def get_initial_state(self) -> Any:
        """Get the initial state of the game."""
        pass

    @abstractmethod
    def encode_state(self, state: Any) -> np.ndarray:
        """Encode the game state into a vector."""
        pass

    @abstractmethod
    def decode_state(self, encoding: np.ndarray) -> Any:
        """Decode a vector back to game state representation."""
        pass


class InfoStateEncoder:
    """Information state encoder for OpenSpiel games."""

    def __init__(self, game):
        """Initialize the encoder.

        Args:
            game: OpenSpiel game object
        """
        self.game = game
        self.num_players = game.num_players()
        self.encoding_size = self._compute_encoding_size()

    def _compute_encoding_size(self) -> int:
        """Compute the size of the information state encoding."""
        # Get the size from OpenSpiel's information state tensor
        state = self.game.new_initial_state()
        tensor = state.information_state_tensor()
        return len(tensor)

    def encode(self, state) -> np.ndarray:
        """Encode a game state into a vector.

        Args:
            state: OpenSpiel game state

        Returns:
            Encoded state as numpy array
        """
        # Use OpenSpiel's information state tensor
        tensor = state.information_state_tensor()
        return np.array(tensor, dtype=np.float32)

    def encode_batch(self, states: List) -> np.ndarray:
        """Encode multiple states.

        Args:
            states: List of OpenSpiel game states

        Returns:
            Batch of encoded states as numpy array
        """
        tensors = [self.encode(state) for state in states]
        return np.stack(tensors)

    def get_legal_actions_mask(self, state) -> np.ndarray:
        """Get a mask for legal actions.

        Args:
            state: OpenSpiel game state

        Returns:
            Boolean mask of legal actions
        """
        legal_actions = state.legal_actions()
        num_actions = self.game.num_distinct_actions()
        mask = np.zeros(num_actions, dtype=bool)
        mask[legal_actions] = True
        return mask

    def get_action_from_index(self, action_index: int) -> Any:
        """Convert action index to action for the game.

        Args:
            action_index: Integer action index

        Returns:
            Action object for the game
        """
        return action_index  # OpenSpiel uses integer actions