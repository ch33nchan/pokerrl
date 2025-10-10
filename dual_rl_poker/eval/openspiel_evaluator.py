"""OpenSpiel exact evaluator for two-player zero-sum games.

Provides exact NashConv and exploitability computation using OpenSpiel's
exact evaluators, replacing any Monte Carlo approximations.
"""

import numpy as np
import pyspiel
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class OpenSpielExactEvaluator:
    """Exact evaluator using OpenSpiel's NashConv and exploitability functions.

    This evaluator provides exact metrics for two-player zero-sum games,
    eliminating any Monte Carlo estimation bias.
    """

    def __init__(self, game_name: str = "kuhn_poker"):
        """Initialize evaluator with specified game.

        Args:
            game_name: Name of the OpenSpiel game to evaluate
        """
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        self.num_players = self.game.num_players()

        if self.num_players != 2:
            raise ValueError(f"OpenSpielExactEvaluator only supports 2-player games, got {self.num_players}")

        # Validate game is zero-sum
        game_type = self.game.get_type()
        if not game_type.utility == pyspiel.GameType.Utility.ZERO_SUM:
            logger.warning(f"Game {game_name} may not be zero-sum")

        logger.info(f"Initialized OpenSpielExactEvaluator for {game_name}")
        logger.info(f"OpenSpiel version: {pyspiel.__version__}")

    def evaluate_nash_conv(self, policy_dict: Dict[int, Callable]) -> float:
        """Compute exact NashConv using OpenSpiel's evaluator.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions.
                         Policy functions should accept (info_state, legal_actions)
                         and return action probabilities.

        Returns:
            NashConv value in game units (exact, no sampling error)
        """
        # Build policy mapping for OpenSpiel
        policy_mapping = self._build_policy_mapping(policy_dict)

        # Use OpenSpiel's exact NashConv function
        try:
            nash_conv_value = pyspiel.nash_conv(self.game, policy_mapping)
            return float(nash_conv_value)
        except Exception as e:
            logger.error(f"Error computing NashConv: {e}")
            raise

    def evaluate_exploitability(self, policy_dict: Dict[int, Callable]) -> float:
        """Compute exact exploitability using OpenSpiel's evaluator.

        For two-player zero-sum games, exploitability = NashConv / 2.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            Exploitability value in game units (exact, no sampling error)
        """
        nash_conv = self.evaluate_nash_conv(policy_dict)
        exploitability = nash_conv / 2.0
        return float(exploitability)

    def evaluate_with_diagnostics(self, policy_dict: Dict[int, Callable],
                                 num_mc_episodes: int = 1000) -> Dict[str, Any]:
        """Evaluate policy with exact metrics and Monte Carlo validation.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions
            num_mc_episodes: Number of Monte Carlo episodes for validation

        Returns:
            Dictionary with exact and Monte Carlo metrics for validation
        """
        # Exact evaluation (primary metrics)
        nash_conv_exact = self.evaluate_nash_conv(policy_dict)
        exploitability_exact = self.evaluate_exploitability(policy_dict)

        # Monte Carlo evaluation (for validation only)
        mc_rewards = self._monte_carlo_evaluation(policy_dict, num_mc_episodes)

        # Compute additional diagnostics
        policy_entropy = self._compute_policy_entropy(policy_dict)

        return {
            "nash_conv_exact": nash_conv_exact,
            "exploitability_exact": exploitability_exact,
            "mc_mean_reward": float(np.mean(mc_rewards)),
            "mc_std_reward": float(np.std(mc_rewards)),
            "mc_min_reward": float(np.min(mc_rewards)),
            "mc_max_reward": float(np.max(mc_rewards)),
            "policy_entropy": policy_entropy,
            "mc_validation_episodes": num_mc_episodes,
            "game_name": self.game_name,
            "openspiel_version": pyspiel.__version__
        }

    def _build_policy_mapping(self, policy_dict: Dict[int, Callable]) -> Dict[Any, Any]:
        """Build policy mapping for OpenSpiel evaluator.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            Policy mapping compatible with OpenSpiel evaluator
        """
        policy_mapping = {}

        for player, policy_fn in policy_dict.items():
            def create_bot(p, policy_func):
                def bot(state):
                    if state.is_chance_node():
                        return pyspiel.SampleAction(state)
                    elif state.current_player() == p:
                        info_state = state.information_state_string(p)
                        legal_actions = state.legal_actions()
                        probs = policy_func(info_state, legal_actions)

                        # Ensure probabilities are valid
                        probs = np.array(probs)
                        if len(probs) != len(legal_actions):
                            # Handle mismatched dimensions
                            adjusted_probs = np.zeros(len(legal_actions))
                            for i, action in enumerate(legal_actions):
                                if i < len(probs):
                                    adjusted_probs[i] = probs[i]
                            probs = adjusted_probs

                        # Normalize probabilities
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                        else:
                            probs = np.ones(len(legal_actions)) / len(legal_actions)

                        # Sample action according to policy
                        return np.random.choice(legal_actions, p=probs)
                    else:
                        return pyspiel.SampleAction(state)
                return bot

            policy_mapping[player] = create_bot(player, policy_fn)

        return policy_mapping

    def _monte_carlo_evaluation(self, policy_dict: Dict[int, Callable],
                              num_episodes: int) -> np.ndarray:
        """Monte Carlo evaluation for validation purposes only.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions
            num_episodes: Number of episodes to simulate

        Returns:
            Array of rewards for player 0
        """
        rewards = []

        for episode in range(num_episodes):
            state = self.game.new_initial_state()

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
                        # Uniform random if no policy
                        action = np.random.choice(state.legal_actions())

                state.apply_action(action)

            # Get rewards for player 0
            returns = state.returns()
            rewards.append(returns[0])

        return np.array(rewards)

    def _compute_policy_entropy(self, policy_dict: Dict[int, Callable]) -> Dict[int, float]:
        """Compute average policy entropy for diagnostics.

        Args:
            policy_dict: Dictionary mapping player indices to policy functions

        Returns:
            Dictionary mapping players to average entropy
        """
        entropies = {}

        for player, policy_fn in policy_dict.items():
            # Sample some information states to estimate entropy
            sample_states = self._sample_information_states(player, 100)

            total_entropy = 0.0
            valid_states = 0

            for info_state, legal_actions in sample_states:
                probs = policy_fn(info_state, legal_actions)
                probs = np.array(probs)

                # Remove zero probabilities for log calculation
                valid_probs = probs[probs > 0]
                if len(valid_probs) > 0:
                    entropy = -np.sum(valid_probs * np.log(valid_probs))
                    total_entropy += entropy
                    valid_states += 1

            if valid_states > 0:
                entropies[player] = total_entropy / valid_states
            else:
                entropies[player] = 0.0

        return entropies

    def _sample_information_states(self, player: int, num_samples: int) -> List[Tuple[str, List[int]]]:
        """Sample information states for a player.

        Args:
            player: Player index
            num_samples: Number of states to sample

        Returns:
            List of (info_state_string, legal_actions) tuples
        """
        states = []
        visited_states = set()

        for _ in range(num_samples):
            state = self.game.new_initial_state()
            info_state = None

            # Traverse game tree to find player states
            while not state.is_terminal():
                if state.is_chance_node():
                    action = pyspiel.SampleAction(state)
                    state = state.child(action)
                elif state.current_player() == player:
                    info_state = state.information_state_string(player)
                    legal_actions = state.legal_actions()

                    # Avoid duplicate states
                    state_key = (info_state, tuple(legal_actions))
                    if state_key not in visited_states:
                        visited_states.add(state_key)
                        states.append((info_state, list(legal_actions)))

                    # Take random action to continue
                    action = np.random.choice(state.legal_actions())
                    state = state.child(action)
                else:
                    # Other player's turn
                    action = np.random.choice(state.legal_actions())
                    state = state.child(action)

            if len(states) >= num_samples:
                break

        return states[:num_samples]

    def get_game_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the game.

        Returns:
            Dictionary with game metadata
        """
        game_type = self.game.get_type()

        return {
            "name": self.game_name,
            "num_players": self.num_players,
            "utility_type": str(game_type.utility),
            "chance_mode": str(game_type.chance_mode),
            "information": str(game_type.information),
            "dynamics": str(game_type.dynamics),
            "max_game_length": game_type.max_game_length,
            "num_distinct_actions": self.game.num_distinct_actions(),
            "min_utility": game_type.min_utility(),
            "max_utility": game_type.max_utility(),
            "openspiel_version": pyspiel.__version__
        }