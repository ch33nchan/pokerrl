"""
A tabular implementation of Counterfactual Regret Minimization (CFR) for Leduc Hold'em.
This version uses an iterative, chance-sampled approach (MCCFR) to avoid recursion limits.
"""
import numpy as np
from collections import defaultdict
import pickle
import time
from qagent.environments.leduc_holdem import LeducHoldem
from qagent.evaluation.evaluate_leduc_cfr import calculate_exploitability as calculate_leduc_exploitability

class CFRTrainer:
    """
    A trainer for a tabular MCCFR agent for Leduc Hold'em.
    """
    def __init__(self, game: LeducHoldem):
        self.game = game
        self.num_actions = game.num_actions
        self.regret_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.exploitability_history = []

    def get_strategy(self, info_set: str, legal_actions: list) -> np.ndarray:
        """
        Get the current strategy for an information set, using regret matching.
        """
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        
        legal_mask = np.zeros(self.num_actions)
        if not legal_actions:
            return legal_mask
        legal_mask[legal_actions] = 1
        positive_regrets *= legal_mask

        sum_positive_regrets = np.sum(positive_regrets)
        
        if sum_positive_regrets > 0:
            strategy = positive_regrets / sum_positive_regrets
        else:
            num_legal_actions = len(legal_actions)
            strategy = np.zeros(self.num_actions)
            if num_legal_actions > 0:
                strategy[legal_actions] = 1.0 / num_legal_actions
        
        return strategy

    def train(self, iterations: int, eval_interval: int = 10000, eval_samples: int = 1000):
        """
        Train the CFR agent for a given number of iterations using iterative MCCFR.
        """
        start_time = time.time()
        for i in range(iterations):
            # --- Forward Pass: Sample a trajectory ---
            # We sample one trajectory per iteration (Monte Carlo approach)
            
            # History stores (state, player, action) tuples for the backward pass
            history = []
            
            state = self.game.get_initial_state()
            
            # Traverse the tree until a terminal state is reached
            while not self.game.is_terminal(state):
                player = self.game.get_current_player(state)

                if self.game.is_chance_node(state):
                    # Sample a chance outcome
                    _, state = self.game.sample_chance_outcome(state)[0]
                    continue

                info_set = self.game.get_info_set(state)
                legal_actions = self.game.get_legal_actions(state)
                strategy = self.get_strategy(info_set, legal_actions)
                
                # Update average strategy sum for the current player
                # In MCCFR, reach probabilities are implicitly 1 for the sampled path
                self.strategy_sum[info_set] += strategy

                # Sample an action according to the current strategy
                action = np.random.choice(np.arange(self.num_actions), p=strategy)
                
                # Record the state and action for the backward pass
                history.append({'info_set': info_set, 'player': player, 'action': action, 'strategy': strategy})
                
                state = self.game.get_next_state(state, action)

            # --- Backward Pass: Update regrets ---
            # Get the terminal payoff from player 0's perspective
            payoff = self.game.get_payoff(state, 0)
            
            # Iterate backwards through the recorded history
            for record in reversed(history):
                info_set = record['info_set']
                player = record['player']
                strategy = record['strategy']
                
                # Calculate counterfactual values (utility of each action)
                # For the path taken, the value is the final payoff.
                # For other paths, the value is 0 as they were not explored in this sample.
                action_taken = record['action']
                action_utils = np.zeros(self.num_actions)
                
                if player == 0:
                    action_utils[action_taken] = payoff
                else: # Player 1
                    action_utils[action_taken] = -payoff

                # Node utility is the expected utility given the strategy
                node_util = np.sum(strategy * action_utils)
                
                # Calculate regrets
                regrets = action_utils - node_util
                self.regret_sum[info_set] += regrets

            # --- Periodic Evaluation ---
            if (i + 1) % eval_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"\n--- Iteration {i + 1}/{iterations} ---")
                print(f"Time elapsed: {elapsed_time:.2f}s")
                print("Calculating exploitability...")
                
                avg_policy = self.get_average_policy()
                exploitability = calculate_leduc_exploitability(self.game, avg_policy, num_samples=eval_samples)
                self.exploitability_history.append((i + 1, exploitability))
                
                print(f"Estimated Exploitability: {exploitability:.4f}")
                
                # Save the policy periodically
                with open(f"leduc_cfr_policy_iter_{i+1}.pkl", "wb") as f:
                    pickle.dump(avg_policy, f)
                print(f"Saved average policy to leduc_cfr_policy_iter_{i+1}.pkl")


    def get_average_policy(self):
        """
        Calculate the average policy from the strategy sums.
        """
        avg_policy = {}
        for info_set, summed_strategy in self.strategy_sum.items():
            total_sum = np.sum(summed_strategy)
            if total_sum > 0:
                avg_policy[info_set] = summed_strategy / total_sum
            else:
                # This case should ideally not be hit in info sets that are reachable
                legal_actions = self.game.get_legal_actions_from_info_set(info_set)
                avg_policy[info_set] = np.ones(self.num_actions) / len(legal_actions) if legal_actions else np.zeros(self.num_actions)
        return avg_policy


def main():
    """
    Run the CFR trainer for Leduc Hold'em.
    """
    game = LeducHoldem()
    trainer = CFRTrainer(game)
    
    iterations = 100000 # More iterations for a better policy
    print(f"Running MCCFR for {iterations} iterations...")
    trainer.train(iterations, eval_interval=20000, eval_samples=5000)
    
    print("\nTraining complete.")
    
    # Get the final average policy
    average_policy = trainer.get_average_policy()
    
    # Save the final policy
    final_policy_filename = "leduc_cfr_policy.pkl"
    with open(final_policy_filename, "wb") as f:
        pickle.dump(average_policy, f)
    print(f"Saved final average policy to {final_policy_filename}")

    print("\nExploitability History:")
    for iteration, exploitability in trainer.exploitability_history:
        print(f"  Iteration {iteration}: {exploitability:.4f}")

    # Final, more accurate exploitability calculation
    print("\nCalculating final exploitability with more samples...")
    final_exploitability = calculate_leduc_exploitability(game, average_policy, num_samples=20000)
    print(f"\nFinal Estimated Exploitability: {final_exploitability:.4f}")


if __name__ == "__main__":
    main()
