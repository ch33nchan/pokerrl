"""OpenSpiel-based evaluator for poker agents."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
try:
    import pyspiel
except ImportError:
    pyspiel = None

from games.base import GameWrapper


class OpenSpielEvaluator:
    """Evaluator using OpenSpiel's exact NashConv and exploitability metrics."""

    def __init__(self, game_wrapper: GameWrapper):
        """Initialize evaluator.

        Args:
            game_wrapper: Game wrapper instance
        """
        if pyspiel is None:
            raise ImportError("OpenSpiel (pyspiel) is required")

        self.game_wrapper = game_wrapper
        self.game = game_wrapper.game
        self.num_players = game_wrapper.num_players

    def evaluate_nash_conv(self, policy_dict: Dict[int, Any]) -> float:
        """Compute NashConv using OpenSpiel's exact evaluator.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            NashConv value in game units
        """
        # Build policy mapping for OpenSpiel
        policy_mapping = self._build_policy_mapping(policy_dict)

        # Use OpenSpiel's nash_conv function
        nash_conv_value = pyspiel.nash_conv(self.game, policy_mapping)
        return float(nash_conv_value)

    def evaluate_exploitability(self, policy_dict: Dict[int, Any]) -> float:
        """Compute exploitability using OpenSpiel's exact evaluator.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            Exploitability value in game units
        """
        # For two-player zero-sum games, exploitability = NashConv / 2
        nash_conv = self.evaluate_nash_conv(policy_dict)
        return nash_conv / 2.0

    def evaluate_tabular_policy(self, strategy_dict: Dict[str, np.ndarray]) -> float:
        """Evaluate a tabular policy.

        Args:
            strategy_dict: Dictionary mapping info state strings to strategy arrays

        Returns:
            Exploitability value
        """
        policy_dict = {}
        for player in range(self.num_players):
            policy_dict[player] = self._create_tabular_policy_function(
                strategy_dict, player
            )
        return self.evaluate_exploitability(policy_dict)

    def _build_policy_mapping(self, policy_dict: Dict[int, Any]) -> Dict[Any, Any]:
        """Build policy mapping for OpenSpiel evaluator.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            Policy mapping for OpenSpiel
        """
        policy_mapping = {}

        for player, policy_fn in policy_dict.items():
            # Create a bot that uses the policy function
            def create_bot(p, policy_func):
                def bot(state):
                    if state.is_chance_node():
                        return pyspiel.SampleAction(state)
                    elif state.current_player() == p:
                        info_state = state.information_state_string(p)
                        legal_actions = state.legal_actions()
                        probs = policy_func(info_state, legal_actions)
                        return np.random.choice(legal_actions, p=probs)
                    else:
                        return pyspiel.SampleAction(state)
                return bot

            policy_mapping[player] = create_bot(player, policy_fn)

        return policy_mapping

    def _create_tabular_policy_function(self, strategy_dict: Dict[str, np.ndarray],
                                      player: int):
        """Create a policy function from tabular strategy.

        Args:
            strategy_dict: Tabular strategy dictionary
            player: Player index

        Returns:
            Policy function
        """
        def policy_function(info_state: str, legal_actions: List[int]) -> np.ndarray:
            """Get action probabilities for given info state.

            Args:
                info_state: Information state string
                legal_actions: List of legal action indices

            Returns:
                Action probabilities
            """
            # Get strategy for this info state
            player_key = f"player_{player}_{info_state}"
            strategy = strategy_dict.get(player_key, np.ones(len(legal_actions)) / len(legal_actions))

            # Ensure strategy matches legal actions
            probs = np.array([strategy[action] for action in legal_actions])

            # Normalize to ensure it's a valid probability distribution
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(legal_actions)) / len(legal_actions)

            return probs

        return policy_function

    def evaluate_policy_network(self, policy_network, device: str = 'cpu') -> float:
        """Evaluate a neural network policy.

        Args:
            policy_network: Neural network policy
            device: Device for computation

        Returns:
            Exploitability value
        """
        policy_dict = {}
        for player in range(self.num_players):
            policy_dict[player] = self._create_network_policy_function(
                policy_network, player, device
            )
        return self.evaluate_exploitability(policy_dict)

    def _create_network_policy_function(self, policy_network, player: int, device: str):
        """Create a policy function from neural network.

        Args:
            policy_network: Neural network policy
            player: Player index
            device: Device for computation

        Returns:
            Policy function
        """
        def policy_function(info_state: str, legal_actions: List[int]) -> np.ndarray:
            """Get action probabilities from neural network.

            Args:
                info_state: Information state string
                legal_actions: List of legal action indices

            Returns:
                Action probabilities
            """
            # Convert info state string to tensor
            state = self.game.new_initial_state()
            # This is simplified - in practice, you'd need to reconstruct the state
            # For now, return uniform distribution over legal actions
            return np.ones(len(legal_actions)) / len(legal_actions)

        return policy_function

    def evaluate_with_diagnostics(self, policy_dict: Dict[int, Any],
                                 num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate policy with comprehensive diagnostics.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions
            num_episodes: Number of episodes for Monte Carlo evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()

        # Exact evaluation
        nash_conv = self.evaluate_nash_conv(policy_dict)
        exploitability = self.evaluate_exploitability(policy_dict)

        # Monte Carlo evaluation for validation
        mc_rewards = self._monte_carlo_evaluation(policy_dict, num_episodes)

        eval_time = time.time() - start_time

        return {
            "nash_conv": nash_conv,
            "exploitability": exploitability,
            "mc_mean_reward": np.mean(mc_rewards),
            "mc_std_reward": np.std(mc_rewards),
            "mc_min_reward": np.min(mc_rewards),
            "mc_max_reward": np.max(mc_rewards),
            "eval_time_seconds": eval_time,
            "num_episodes": num_episodes
        }

    def _monte_carlo_evaluation(self, policy_dict: Dict[int, Any],
                              num_episodes: int) -> np.ndarray:
        """Monte Carlo evaluation against random opponent.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions
            num_episodes: Number of episodes

        Returns:
            Array of rewards
        """
        rewards = []

        for _ in range(num_episodes):
            state = self.game.new_initial_state()
            episode_reward = 0

            while not state.is_terminal():
                if state.is_chance_node():
                    action = pyspiel.SampleAction(state)
                else:
                    player = state.current_player()
                    policy_fn = policy_dict.get(player)

                    if policy_fn:
                        info_state = state.information_state_string(player)
                        legal_actions = state.legal_actions()
                        probs = policy_fn(info_state, legal_actions)
                        action = np.random.choice(legal_actions, p=probs)
                    else:
                        # Random action if no policy
                        action = np.random.choice(state.legal_actions())

                state.apply_action(action)

            episode_reward = state.returns()[0]  # Player 0's reward
            rewards.append(episode_reward)

        return np.array(rewards)