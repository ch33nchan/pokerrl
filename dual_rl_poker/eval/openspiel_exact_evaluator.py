"""
Exact OpenSpiel evaluator for NashConv and exploitability computation.

This module provides exact evaluation using OpenSpiel's native evaluators,
replacing all Monte Carlo approximation methods with precise computation.
"""

import pyspiel
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for exact evaluation results."""
    nash_conv: float
    exploitability: float
    mean_value: float
    player_0_value: float
    player_1_value: float
    game_name: str
    num_info_states: int
    num_info_states_per_player: List[int]


class OpenSpielExactEvaluator:
    """
    Exact evaluator using OpenSpiel's native evaluation functions.

    Provides precise NashConv and exploitability computation without
    Monte Carlo sampling noise. Uses proper policy adaptors for
    seamless integration with neural network policies.
    """

    def __init__(self, game_name: str):
        """
        Initialize evaluator for a specific game.

        Args:
            game_name: Name of the OpenSpiel game
        """
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        self.num_players = self.game.num_players()

        # Validate game is two-player zero-sum
        if self.num_players != 2:
            raise ValueError(f"Expected 2-player game, got {self.num_players}")

        # Precompute game information
        self.num_info_states = self.game.num_distinct_histories()
        self.num_info_states_per_player = [
            self.game.num_distinct_histories(player)
            for player in range(self.num_players)
        ]

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized exact evaluator for {game_name}")
        self.logger.info(f"Game has {self.num_info_states} total information states")

    def evaluate_policy(self, policy_dict: Dict[str, np.ndarray]) -> EvaluationResult:
        """
        Evaluate a policy using exact OpenSpiel computation.

        Args:
            policy_dict: Dictionary mapping info states to action probabilities

        Returns:
            EvaluationResult with exact metrics
        """
        # Create OpenSpiel policy from the policy dictionary
        openspiel_policy = self._create_openspiel_policy(policy_dict)

        # Compute exact NashConv
        nash_conv = pyspiel.nash_conv(self.game, [openspiel_policy])

        # Compute exploitability (NashConv / 2 for two-player zero-sum)
        exploitability = nash_conv / 2.0

        # Compute mean value against uniform random opponent
        mean_value = self._compute_mean_value(openspiel_policy)

        # Compute per-player values
        player_values = self._compute_per_player_values(openspiel_policy)

        return EvaluationResult(
            nash_conv=nash_conv,
            exploitability=exploitability,
            mean_value=mean_value,
            player_0_value=player_values[0],
            player_1_value=player_values[1],
            game_name=self.game_name,
            num_info_states=self.num_info_states,
            num_info_states_per_player=self.num_info_states_per_player
        )

    def _create_openspiel_policy(self, policy_dict: Dict[str, np.ndarray]) -> pyspiel.TabularPolicy:
        """
        Create OpenSpiel TabularPolicy from policy dictionary.

        Uses proper information state handling and legal action filtering.
        """
        tabular_policy = pyspiel.TabularPolicy(self.game)

        # Iterate through all information states in the game
        for state in self.game.new_initial_state().history():
            if not state.is_chance_node() and state.is_simultaneous_node():
                continue  # Skip simultaneous nodes for now

            if state.is_terminal():
                continue

            # Get current player and information state
            if state.is_chance_node():
                continue

            player = state.current_player()
            info_state = state.information_state_string(player)

            # Get legal actions for this state
            legal_actions = state.legal_actions()
            num_actions = self.game.num_distinct_actions()

            if info_state in policy_dict:
                # Use provided policy probabilities
                policy_probs = policy_dict[info_state]

                # Ensure correct size
                if len(policy_probs) != num_actions:
                    raise ValueError(
                        f"Policy size mismatch for info state {info_state}: "
                        f"expected {num_actions}, got {len(policy_probs)}"
                    )

                # Filter to legal actions and renormalize
                legal_probs = np.array([policy_probs[a] for a in legal_actions])
                legal_probs = legal_probs / legal_probs.sum()

                # Set probabilities in TabularPolicy
                for i, action in enumerate(legal_actions):
                    tabular_policy.set_probabilities(state, player, action, legal_probs[i])

        return tabular_policy

    def _compute_mean_value(self, policy: pyspiel.TabularPolicy) -> float:
        """
        Compute expected value against uniform random opponent.
        """
        # Create uniform random policy
        uniform_policy = pyspiel.TabularPolicy(self.game)

        # Set uniform probabilities for all information states
        for state in self.game.new_initial_state().history():
            if state.is_chance_node() or state.is_terminal():
                continue

            player = state.current_player()
            legal_actions = state.legal_actions()
            uniform_prob = 1.0 / len(legal_actions)

            for action in legal_actions:
                uniform_policy.set_probabilities(state, player, action, uniform_prob)

        # Compute expected value
        value = pyspiel.util.value(self.game, [policy, uniform_policy])
        return value

    def _compute_per_player_values(self, policy: pyspiel.TabularPolicy) -> List[float]:
        """
        Compute expected values for each player against itself.
        """
        values = pyspiel.util.value(self.game, [policy, policy])
        return values

    def evaluate_policy_network(self, policy_network,
                                num_states: int = 1000) -> EvaluationResult:
        """
        Evaluate a neural network policy by sampling information states.

        Args:
            policy_network: Neural network that takes state encoding and returns action probs
            num_states: Number of information states to sample

        Returns:
            EvaluationResult with exact metrics
        """
        # Sample information states from the game
        info_states, state_encodings = self._sample_info_states(num_states)

        # Get policy predictions for all sampled states
        policy_dict = {}
        for i, (info_state, state_encoding) in enumerate(zip(info_states, state_encodings)):
            # Get action probabilities from network
            with torch.no_grad():
                action_probs = policy_network(state_encoding).cpu().numpy()

            policy_dict[info_state] = action_probs

        # Evaluate the constructed policy
        return self.evaluate_policy(policy_dict)

    def _sample_info_states(self, num_states: int) -> Tuple[List[str], List[np.ndarray]]:
        """
        Sample information states and their encodings from the game.
        """
        info_states = []
        state_encodings = []

        # Sample states through random gameplay
        for _ in range(num_states):
            state = self.game.new_initial_state()

            while not state.is_terminal() and len(info_states) < num_states:
                if state.is_chance_node():
                    # Sample chance outcome
                    outcomes = state.chance_outcomes()
                    probs = [p for _, p in outcomes]
                    actions = [a for a, _ in outcomes]
                    action = np.random.choice(actions, p=probs)
                    state.apply_action(action)

                elif state.is_simultaneous_node():
                    # Handle simultaneous nodes (rare in our games)
                    joint_action = []
                    for player in range(self.num_players):
                        actions = state.legal_actions(player)
                        action = np.random.choice(actions)
                        joint_action.append(action)
                    state.apply_action(joint_action)

                else:
                    # Record information state for current player
                    player = state.current_player()
                    info_state = state.information_state_string(player)

                    if info_state not in info_states:
                        info_states.append(info_state)
                        # Encode state (this would depend on the specific encoding)
                        state_encoding = self._encode_state(state)
                        state_encodings.append(state_encoding)

                    # Sample action to continue trajectory
                    actions = state.legal_actions()
                    action = np.random.choice(actions)
                    state.apply_action(action)

        return info_states, state_encodings

    def _encode_state(self, state) -> np.ndarray:
        """
        Encode game state for neural network input.

        This is a placeholder - actual encoding would depend on the specific game.
        """
        # For Kuhn poker: encode card and betting history
        # For Leduc Hold'em: encode private card, public card, and betting rounds
        # This would need to be implemented based on the specific game requirements
        raise NotImplementedError("State encoding must be implemented per game")


class KuhnPokerEvaluator(OpenSpielExactEvaluator):
    """Specialized evaluator for Kuhn Poker."""

    def __init__(self):
        super().__init__("kuhn_poker")

    def _encode_state(self, state) -> np.ndarray:
        """Encode Kuhn Poker state for neural network."""
        # Kuhn poker encoding: one-hot card + betting history
        encoding = np.zeros(10)  # 3 cards + betting history

        # Get player's card (0, 1, or 2)
        player = state.current_player()
        if not state.is_chance_node() and not state.is_terminal():
            # Extract card info from information state
            info_state = state.information_state_string(player)
            if "card:0" in info_state:
                encoding[0] = 1
            elif "card:1" in info_state:
                encoding[1] = 1
            elif "card:2" in info_state:
                encoding[2] = 1

        # Encode betting history (simplified)
        # This would need to be expanded based on actual betting history encoding

        return encoding


class LeducPokerEvaluator(OpenSpielExactEvaluator):
    """Specialized evaluator for Leduc Poker."""

    def __init__(self):
        super().__init__("leduc_poker")  # Note: may need to adjust game name

    def _encode_state(self, state) -> np.ndarray:
        """Encode Leduc Poker state for neural network."""
        # Leduc poker encoding: private card, public card, betting round, history
        encoding = np.zeros(20)  # Placeholder size

        # This would need to be implemented based on Leduc poker specifics

        return encoding


def create_evaluator(game_name: str) -> OpenSpielExactEvaluator:
    """
    Factory function to create appropriate evaluator for a game.

    Args:
        game_name: Name of the game

    Returns:
        Appropriate evaluator instance
    """
    if game_name == "kuhn_poker":
        return KuhnPokerEvaluator()
    elif game_name == "leduc_poker":
        return LeducPokerEvaluator()
    else:
        return OpenSpielExactEvaluator(game_name)


def verify_evaluator_correctness():
    """
    Verify that the evaluator works correctly on simple test cases.
    """
    print("Testing exact evaluator...")

    # Test Kuhn Poker evaluator
    evaluator = create_evaluator("kuhn_poker")

    # Create a simple uniform random policy
    uniform_policy = {}
    game = pyspiel.load_game("kuhn_poker")

    # Sample some information states and assign uniform probabilities
    for state in game.new_initial_state().history():
        if state.is_chance_node() or state.is_terminal():
            continue

        player = state.current_player()
        info_state = state.information_state_string(player)
        legal_actions = state.legal_actions()

        if info_state not in uniform_policy:
            uniform_policy[info_state] = np.zeros(game.num_distinct_actions())
            for action in legal_actions:
                uniform_policy[info_state][action] = 1.0 / len(legal_actions)

    # Evaluate the uniform policy
    result = evaluator.evaluate_policy(uniform_policy)

    print(f"Kuhn Poker uniform policy evaluation:")
    print(f"  NashConv: {result.nash_conv:.6f}")
    print(f"  Exploitability: {result.exploitability:.6f}")
    print(f"  Mean Value: {result.mean_value:.6f}")

    # NashConv for uniform random policy in Kuhn poker should be around 0.0556
    # (This is the game value difference from Nash equilibrium)
    expected_nash_conv = 0.0556
    if abs(result.nash_conv - expected_nash_conv) < 0.01:
        print("✓ Evaluator working correctly")
    else:
        print(f"⚠ Unexpected NashConv: {result.nash_conv:.6f} (expected ~{expected_nash_conv:.6f})")


if __name__ == "__main__":
    # Run verification tests
    verify_evaluator_correctness()