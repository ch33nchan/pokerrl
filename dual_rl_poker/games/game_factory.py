"""Game factory for creating game wrappers."""

from typing import Dict, Any
from .base import GameWrapper
from .kuhn_poker import KuhnPokerWrapper
from .leduc_poker import LeducPokerWrapper


class GameFactory:
    """Factory class for creating game wrappers."""

    def __init__(self):
        """Initialize game factory."""
        self._game_registry = {
            "kuhn_poker": KuhnPokerWrapper,
            "leduc_poker": LeducPokerWrapper,
        }

    def create_game(self, game_name: str, config: Dict[str, Any] = None) -> GameWrapper:
        """Create a game wrapper instance.

        Args:
            game_name: Name of the game to create
            config: Optional configuration dictionary

        Returns:
            Game wrapper instance
        """
        if game_name not in self._game_registry:
            raise ValueError(
                f"Unknown game: {game_name}. Available games: {list(self._game_registry.keys())}"
            )

        game_class = self._game_registry[game_name]
        if config is None:
            config = {}

        return game_class(**config)

    def get_available_games(self) -> list:
        """Get list of available games.

        Returns:
            List of available game names
        """
        return list(self._game_registry.keys())

    def register_game(self, game_name: str, game_class: type):
        """Register a new game class.

        Args:
            game_name: Name to register the game under
            game_class: Game wrapper class
        """
        if not issubclass(game_class, GameWrapper):
            raise ValueError(f"Game class must inherit from GameWrapper")

        self._game_registry[game_name] = game_class


# Global factory instance
game_factory = GameFactory()


def create_game(game_name: str, config: Dict[str, Any] = None) -> GameWrapper:
    """Convenience function to create a game.

    Args:
        game_name: Name of the game to create
        config: Optional configuration dictionary

    Returns:
        Game wrapper instance
    """
    return game_factory.create_game(game_name, config)
