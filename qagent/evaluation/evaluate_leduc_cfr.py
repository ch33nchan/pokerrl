"""
Evaluates the exploitability of a trained CFR policy for Leduc Hold'em.

Exploitability is a measure of how far a policy is from a Nash Equilibrium.
It is calculated by finding the best response strategy against the policy and
determining the value of the game for the best response player. A lower
exploitability score indicates a stronger policy.
"""
import pickle
import numpy as np
from collections import defaultdict
import sys

from qagent.environments.leduc_holdem import LeducHoldem


def load_policy(policy_file):
    """Load a pickled policy."""
    with open(policy_file, 'rb') as f:
        return pickle.load(f)

def monte_carlo_best_response(game, policy, br_player_id, num_samples=10000):
    """
    Calculates an estimate of the best response utility using Monte Carlo sampling.
    This avoids traversing the entire game tree, saving memory.
    """
    total_utility = 0.0

    for _ in range(num_samples):
        # Start a new game and play until a terminal state or it's BR player's turn
        state = game.get_initial_state()
        
        # Handle initial chance node (dealing private cards)
        # We need to traverse through the chance node to get to a player node
        while game.is_chance_node(state):
            # In evaluation, to reduce variance, we can iterate through all outcomes,
            # but for MC sampling, we just sample one path.
            outcomes = game.sample_chance_outcome(state)
            # A simple way to handle this is to just pick one outcome
            if not outcomes: break
            _, state = outcomes[0]


        while not game.is_terminal(state):
            current_player = game.get_current_player(state)

            if game.is_chance_node(state):
                 outcomes = game.sample_chance_outcome(state)
                 if not outcomes: break
                 _, state = outcomes[0]
                 continue

            if current_player == br_player_id:
                # If it's the BR player's turn, find the best action and follow it
                legal_actions = game.get_legal_actions(state)
                if not legal_actions:
                    break
                
                best_action = -1
                max_utility = -np.inf
                
                # To find the best action, we need to estimate the utility of each action.
                # We can do this by playing out from each action.
                for action in legal_actions:
                    next_state = game.get_next_state(state, action)
                    # Simulate from the next state to get an idea of the utility
                    # A single simulation is a very noisy estimate.
                    # For a better estimate, we could run multiple simulations per action.
                    utility = simulate_from_state(game, policy, next_state, br_player_id)
                    if utility > max_utility:
                        max_utility = utility
                        best_action = action
                
                if best_action == -1:
                    # This can happen if no legal actions lead to a positive outcome
                    # in the simulation. Fallback to a random action.
                    best_action = np.random.choice(legal_actions)

                state = game.get_next_state(state, best_action)

            else:
                # If it's the policy player's turn, sample an action from their strategy
                info_set = game.get_info_set(state)
                strategy = policy.get(info_set)
                legal_actions = game.get_legal_actions(state)

                if strategy is None:
                    # Default to uniform random if info_set not in policy
                    action = np.random.choice(legal_actions)
                else:
                    # The strategy is a numpy array where indices correspond to actions.
                    action_probs = strategy[legal_actions]
                    prob_sum = np.sum(action_probs)
                    if prob_sum > 0:
                        normalized_probs = action_probs / prob_sum
                        action = np.random.choice(legal_actions, p=normalized_probs)
                    else:
                        # Fallback for sparsely sampled info sets
                        action = np.random.choice(legal_actions)

                state = game.get_next_state(state, action)

        # Once the game is over, add the payoff to the total utility
        total_utility += game.get_payoff(state, br_player_id)

    return total_utility / num_samples

def simulate_from_state(game, policy, state, br_player_id):
    """
    Simulates a game from a given state to a terminal node and returns the payoff.
    """
    sim_state = state.copy() # Work on a copy to not alter the original state object
    while not game.is_terminal(sim_state):
        if game.is_chance_node(sim_state):
            outcomes = game.sample_chance_outcome(sim_state)
            if not outcomes: break
            _, sim_state = outcomes[0]
            continue

        current_player = game.get_current_player(sim_state)
        legal_actions = game.get_legal_actions(sim_state)
        if not legal_actions:
            break

        if current_player == br_player_id:
            # BR player plays randomly during simulation for speed.
            # A better approach would be to do a deeper lookahead, but this is a start.
            action = np.random.choice(legal_actions)
        else:
            # Policy player follows their strategy
            info_set = game.get_info_set(sim_state)
            strategy = policy.get(info_set)
            if strategy is None:
                action = np.random.choice(legal_actions)
            else:
                # The strategy is a numpy array where indices correspond to actions.
                action_probs = strategy[legal_actions]
                prob_sum = np.sum(action_probs)
                if prob_sum > 0:
                    normalized_probs = action_probs / prob_sum
                    action = np.random.choice(legal_actions, p=normalized_probs)
                else:
                    # Fallback for sparsely sampled info sets
                    action = np.random.choice(legal_actions)
        
        sim_state = game.get_next_state(sim_state, action)
        
    return game.get_payoff(sim_state, br_player_id)


def calculate_exploitability(game, policy, num_samples=10000):
    """
    Calculates the overall exploitability of the policy.
    It's the sum of the best-response values against the policy for each player.
    """
    initial_state = game.get_initial_state()
    
    # Exploitability = (Value(BR vs P1) + Value(P0 vs BR)) / 2
    # Value(BR vs P1) is what player 0 gets playing a best response.
    # print("Calculating best response for Player 0...")
    br0_utility = monte_carlo_best_response(game, policy, 0, num_samples)
    
    # Value(P0 vs BR) is what player 0 gets. The best response for player 1
    # will try to minimize player 0's payoff.
    # print("Calculating best response for Player 1...")
    br1_utility = monte_carlo_best_response(game, policy, 1, num_samples)

    # br0_utility is the value player 0 gets by playing a best response to the policy.
    # br1_utility is the value player 1 gets by playing a best response to the policy.
    # In a zero-sum game, the value for P1 is the negative of the value for P0.
    # So, the value P0 *loses* to a BR P1 is -br1_utility.
    exploitability = (br0_utility - br1_utility) / 2
    
    return exploitability

def main():
    """
    Load the policy and calculate its exploitability.
    """
    policy_file = 'leduc_cfr_policy.pkl'
    print(f"Loading policy from {policy_file}...")
    policy = load_policy(policy_file)
    game = LeducHoldem()
    print("Calculating exploitability...")
    exploitability = calculate_exploitability(game, policy, num_samples=20000)
    print(f"Estimated Exploitability of the CFR policy: {exploitability}")

if __name__ == "__main__":
    # This allows the script to be run standalone to evaluate a saved policy
    main()
