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
        self.num_actions = self.game.num_distinct_actions()
        self.encoder = InfoStateEncoder(self.game)

    def encode_info_state_key(self, info_state_key: str, player: int) -> np.ndarray:
        """Encode an information state string using cached encodings.

        Args:
            info_state_key: Information state string from OpenSpiel
            player: Player index associated with the information state

        Returns:
            Encoded information state vector
        """

        return self.encoder.encode_info_state_key(info_state_key, player)

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

    def get_encoding_size(self) -> int:
        """Return the information state encoding dimension."""

        return self.encoder.encoding_size

    def get_initial_state(self):
        """Get a new initial game state."""

        return self.game.new_initial_state()

    def is_terminal(self, state) -> bool:
        return state.is_terminal()

    def get_current_player(self, state) -> int:
        return state.current_player()

    def get_legal_actions(self, state) -> List[int]:
        return state.legal_actions()

    def make_action(self, state, player: int, action: int):
        del player
        return state.child(action)

    def sample_chance_action(self, state):
        outcomes = state.chance_outcomes()
        if not outcomes:
            return state
        actions, probs = zip(*outcomes)
        action = np.random.choice(actions, p=np.asarray(probs, dtype=np.float64))
        return state.child(action)

    def get_info_state_key(self, state, player: int) -> str:
        return state.information_state_string(player)

    def get_rewards(self, state) -> List[float]:
        return state.returns()


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
        self.info_state_cache: Dict[tuple[str, int], np.ndarray] = {}
        self._build_info_state_cache()

    def _compute_encoding_size(self) -> int:
        """Compute the size of the information state encoding."""
        # Get the size from OpenSpiel's information state tensor
        state = self.game.new_initial_state()
        # Advance through chance nodes to reach a valid player state
        while state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            if not outcomes:
                break
            action = outcomes[0][0]
            state = state.child(action)

        player = state.current_player()
        if player < 0:
            player = 0

        tensor = state.information_state_tensor(player)
        return len(tensor)

    def encode(self, state) -> np.ndarray:
        """Encode a game state into a vector.

        Args:
            state: OpenSpiel game state

        Returns:
            Encoded state as numpy array
        """
        # Use OpenSpiel's information state tensor
        player = state.current_player()
        if player < 0:
            # Default to first player representation for chance/terminal nodes
            player = 0
        tensor = state.information_state_tensor(player)
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

    def encode_info_state_key(self, info_state_key: str, player: int) -> np.ndarray:
        """Encode an information state string using the cached mapping.

        Args:
            info_state_key: Information state string
            player: Player index

        Returns:
            Encoded state vector
        """

        key = (info_state_key, player)
        if key in self.info_state_cache:
            return self.info_state_cache[key]

        # Fallback to zero vector if unseen (should be rare once cache is built)
        return np.zeros(self.encoding_size, dtype=np.float32)

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

    def _build_info_state_cache(self) -> None:
        """Pre-compute encodings for all reachable information states."""

        visited_histories: set[str] = set()

        def traverse(state) -> None:
            history = state.history_str()
            if history in visited_histories:
                return
            visited_histories.add(history)

            if state.is_terminal():
                return

            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    traverse(state.child(action))
                return

            # Record information state encodings for all players
            for pid in range(self.num_players):
                info_key = state.information_state_string(pid)
                tensor = state.information_state_tensor(pid)
                self.info_state_cache[(info_key, pid)] = np.array(tensor, dtype=np.float32)

            for action in state.legal_actions():
                traverse(state.child(action))

        traverse(self.game.new_initial_state())