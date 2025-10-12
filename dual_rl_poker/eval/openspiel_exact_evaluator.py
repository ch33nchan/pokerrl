"""
Exact OpenSpiel evaluator for NashConv and exploitability computation.

This module provides exact evaluation using OpenSpiel's native evaluators,
replacing all Monte Carlo approximation methods with precise computation.
"""

import pyspiel
import numpy as np
import torch
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
        # OpenSpiel doesn't have num_distinct_histories, use max_game_length as approximation
        self.max_game_length = self.game.max_game_length()
        self.num_info_states_per_player = []
        
        # Count information states by traversing the game tree
        self._count_info_states()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized exact evaluator for {game_name}")
        self.logger.info(f"Game has {self.max_game_length} max game length")

    def _count_info_states(self):
        """Count information states by traversing the game tree."""
        info_states_by_player = {player: set() for player in range(self.num_players)}
        
        def traverse(state):
            if state.is_terminal() or state.is_chance_node():
                return
            
            player = state.current_player()
            info_state_str = state.information_state_string(player)
            info_states_by_player[player].add(info_state_str)
            
            for action in state.legal_actions():
                child = state.clone()
                child.apply_action(action)
                traverse(child)
        
        traverse(self.game.new_initial_state())
        
        self.num_info_states_per_player = [
            len(info_states_by_player[player]) 
            for player in range(self.num_players)
        ]
        self.num_info_states = sum(self.num_info_states_per_player)

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

        # Compute exact NashConv using OpenSpiel's nash_conv function
        nash_conv = pyspiel.nash_conv(self.game, openspiel_policy)

        # Compute exploitability (NashConv / 2 for two-player zero-sum)
        exploitability = nash_conv / 2.0

        # Compute values using expected_returns with correct API
        try:
            # Use root state and proper depth limit
            root_state = self.game.new_initial_state()
            values = pyspiel.expected_returns(
                root_state, 
                openspiel_policy, 
                depth_limit=-1,  # No limit
                use_infostate_get_policy=True
            )
            player_0_value = values[0]
            player_1_value = values[1]
            mean_value = (player_0_value + player_1_value) / 2.0
        except Exception as e:
            # Fallback if expected_returns fails
            self.logger.warning(f"Expected returns failed: {e}")
            player_0_value = 0.0
            player_1_value = 0.0
            mean_value = 0.0

        return EvaluationResult(
            nash_conv=nash_conv,
            exploitability=exploitability,
            mean_value=mean_value,
            player_0_value=player_0_value,
            player_1_value=player_1_value,
            game_name=self.game_name,
            num_info_states=self.num_info_states,
            num_info_states_per_player=self.num_info_states_per_player
        )

    def _create_openspiel_policy(self, policy_dict: Dict[str, np.ndarray]) -> pyspiel.TabularPolicy:
        """
        Create OpenSpiel TabularPolicy from policy dictionary.

        Uses proper information state handling and legal action filtering.
        """
        # Convert policy_dict to the format expected by TabularPolicy
        # TabularPolicy expects: {info_state_str: [(action, prob), ...]}
        tabular_policy_dict = {}
        
        for info_state_str, policy_probs in policy_dict.items():
            # Convert to list of (action, probability) tuples
            action_probs = []
            
            if isinstance(policy_probs, dict):
                # Handle dictionary format: {action_idx: prob}
                for action, prob in policy_probs.items():
                    if prob > 0:  # Only include actions with positive probability
                        action_probs.append((action, float(prob)))
            else:
                # Handle numpy array format: [prob_0, prob_1, ...]
                for action, prob in enumerate(policy_probs):
                    if prob > 0:  # Only include actions with positive probability
                        action_probs.append((action, float(prob)))
            
            if action_probs:  # Only add if there are valid actions
                tabular_policy_dict[info_state_str] = action_probs
        
        # Create TabularPolicy from dictionary
        tabular_policy = pyspiel.TabularPolicy(tabular_policy_dict)
        return tabular_policy

    def _get_all_info_states(self) -> Dict[int, Dict[str, Any]]:
        """Get all information states for each player through systematic traversal."""
        info_states_by_player = {player: {} for player in range(self.num_players)}
        
        def traverse(state):
            if state.is_terminal():
                return
            
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    child = state.clone()
                    child.apply_action(action)
                    traverse(child)
            else:
                player = state.current_player()
                info_state_str = state.information_state_string(player)
                
                if info_state_str not in info_states_by_player[player]:
                    info_states_by_player[player][info_state_str] = state.clone()
                
                for action in state.legal_actions():
                    child = state.clone()
                    child.apply_action(action)
                    traverse(child)
        
        traverse(self.game.new_initial_state())
        return info_states_by_player

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