"""
Evaluates the exploitability of a pre-trained tabular CFR policy for Kuhn Poker.

This script loads a policy from a JSON file, wraps it in an interface compatible
with the exploitability calculation function, and then computes and prints the
exploitability of that policy. This value represents the theoretical performance
limit (optimality gap) for the game.
"""

import json
import numpy as np
from qagent.environments.kuhn_poker import KuhnPoker
from qagent.evaluation.exploitability_kuhn import calculate_exploitability

class TabularAgentWrapper:
    """
    A wrapper for a pre-trained tabular policy to make it compatible with the
    exploitability calculation function.
    """
    def __init__(self, policy_path: str, game: KuhnPoker):
        """
        Initializes the wrapper by loading the policy from a JSON file.

        Args:
            policy_path: Path to the JSON file containing the policy.
            game: An instance of the game environment.
        """
        self.game = game
        try:
            with open(policy_path, 'r') as f:
                self.policy = json.load(f)
        except FileNotFoundError:
            print(f"Error: Policy file not found at {policy_path}")
            self.policy = {}

    def get_action_probabilities(self, state: dict) -> np.ndarray:
        """
        Returns the learned strategy for a given state.

        Args:
            state: The current game state dictionary.

        Returns:
            A numpy array representing the action probabilities.
        """
        if not self.policy:
            # Fallback to uniform random if policy wasn't loaded
            return np.ones(self.game.get_num_actions()) / self.game.get_num_actions()

        info_set_str = self.game.get_state_string(state)
        
        if info_set_str in self.policy:
            return np.array(self.policy[info_set_str])
        else:
            # If an information set is not in the policy (it shouldn't happen for a
            # fully trained tabular agent, but as a safeguard), return a uniform strategy.
            return np.ones(self.game.get_num_actions()) / self.game.get_num_actions()

if __name__ == '__main__':
    POLICY_FILE = "kuhn_tabular_cfr_policy.json"
    
    print(f"Loading tabular CFR policy from {POLICY_FILE}...")
    
    # 1. Initialize the game environment
    kuhn_env = KuhnPoker()
    
    # 2. Create the wrapper for the tabular policy
    tabular_agent_wrapper = TabularAgentWrapper(policy_path=POLICY_FILE, game=kuhn_env)
    
    # 3. Calculate exploitability
    print("Calculating exploitability...")
    exploitability = calculate_exploitability(kuhn_env, tabular_agent_wrapper)
    
    print("\n--- Tabular CFR Exploitability ---")
    # For Kuhn Poker, exploitability is often measured in mbb/hand, but here we use the raw value.
    # A value very close to 0 indicates a near-perfect Nash Equilibrium strategy.
    print(f"Calculated Exploitability: {exploitability}")
    print(f"This value represents the theoretical 'best possible' score (optimality gap).")

