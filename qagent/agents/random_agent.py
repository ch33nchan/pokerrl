import numpy as np

class RandomAgent:
    """A simple agent that chooses a random legal action."""

    def __init__(self, num_actions):
        """
        Initialize the RandomAgent.
        Args:
            num_actions (int): The number of possible actions in the game.
        """
        self.num_actions = num_actions

    def get_action(self, info_set, legal_actions_mask):
        """
        Returns a random legal action.
        Args:
            info_set: The current information set (unused by this agent).
            legal_actions_mask (np.ndarray): A boolean array indicating legal actions.
        Returns:
            int: A randomly chosen legal action.
        """
        legal_actions = np.where(legal_actions_mask)[0]
        return np.random.choice(legal_actions)

    def get_strategy(self, info_set, legal_actions_mask):
        """
        Returns a uniform strategy over legal actions.
        Args:
            info_set: The current information set (unused by this agent).
            legal_actions_mask (np.ndarray): A boolean array indicating legal actions.
        Returns:
            np.ndarray: A strategy vector with uniform probability over legal actions.
        """
        legal_actions = np.where(legal_actions_mask)[0]
        strategy = np.zeros(self.num_actions)
        if len(legal_actions) > 0:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy
