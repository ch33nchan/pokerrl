import numpy as np
from collections import defaultdict
from typing import List, Dict
import json

# Constants for Kuhn Poker
NUM_CARDS = 3
PASS = 0
BET = 1

class KuhnCFRTrainer:
    """
    A trainer for a Counterfactual Regret Minimization (CFR) agent for Kuhn Poker.
    
    This implementation uses vanilla CFR, not any of the more advanced variants.
    """
    def __init__(self):
        self.nodes: Dict[str, 'KuhnNode'] = {}
        self.expected_game_value = 0

    def get_or_create_node(self, info_set: str):
        """
        Retrieves a node for a given information set, creating it if it doesn't exist.
        """
        if info_set not in self.nodes:
            self.nodes[info_set] = KuhnNode(info_set)
        return self.nodes[info_set]

    def cfr(self, history: str, card1: int, card2: int, p1: float, p2: float) -> float:
        """
        The recursive Counterfactual Regret Minimization algorithm.

        Args:
            history (str): The history of actions taken so far.
            card1 (int): The card of player 1.
            card2 (int): The card of player 2.
            p1 (float): The probability of reaching this history for player 1.
            p2 (float): The probability of reaching this history for player 2.

        Returns:
            float: The expected value of the game state.
        """
        player = len(history) % 2
        
        # --- Return payoff for terminal states ---
        if len(history) > 1:
            if history == "pp": # Pass, Pass
                return 1 if card1 > card2 else -1
            if history == "bb": # Bet, Bet (Call)
                return 2 if card1 > card2 else -2
            if history == "pb": # Pass, Bet
                # Player 1's turn to Fold (Pass) or Call (Bet)
                # This is a new decision point, so we continue
                pass
            if history == "bp": # Bet, Pass (Fold)
                return 1
            if history == "pbp": # Pass, Bet, Pass (Fold)
                return -1
            if history == "pbb": # Pass, Bet, Call
                return 2 if card1 > card2 else -2

        # --- Get node for the current information set ---
        info_set = f"{card1 if player == 0 else card2}{history}"
        node = self.get_or_create_node(info_set)
        
        strategy = node.get_strategy(p1 if player == 0 else p2)
        
        util = np.zeros(2) # Utilities for Pass and Bet
        node_util = 0

        for action in [PASS, BET]:
            action_prob = strategy[action]
            new_history = history + ('p' if action == PASS else 'b')
            
            if player == 0:
                util[action] = -self.cfr(new_history, card1, card2, p1 * action_prob, p2)
            else:
                util[action] = -self.cfr(new_history, card1, card2, p1, p2 * action_prob)
            
            node_util += action_prob * util[action]

        # --- Update regrets ---
        if player == 0:
            reach_prob = p2
        else:
            reach_prob = p1
            
        for action in [PASS, BET]:
            regret = util[action] - node_util
            node.regret_sum[action] += reach_prob * regret
            
        return node_util

    def train(self, iterations: int):
        """
        Trains the CFR agent for a specified number of iterations.
        """
        print(f"Running CFR training for {iterations} iterations...")
        cards = list(range(NUM_CARDS))
        util = 0
        
        for i in range(iterations):
            np.random.shuffle(cards)
            util += self.cfr("", cards[0], cards[1], 1, 1)
            if i % 1000 == 0 and i > 0:
                print(f"Iteration {i}, Avg Game Value: {util / i:.4f}")

        self.expected_game_value = util / iterations
        print(f"Training complete. Average game value: {self.expected_game_value:.4f}")

        # Save the final strategy
        self.save_strategy(self.get_final_strategy())

    def save_strategy(self, strategy: Dict[str, np.ndarray], file_path: str = "kuhn_cfr_policy.json"):
        """Saves the strategy to a JSON file."""
        print(f"Saving strategy to {file_path}...")
        # Convert numpy arrays to lists for JSON serialization
        serializable_strategy = {info_set: policy.tolist() for info_set, policy in strategy.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_strategy, f, indent=4)
        print("Strategy saved.")

    def get_final_strategy(self) -> Dict[str, np.ndarray]:
        """
        Computes the average strategy from the accumulated regrets.
        """
        strategy = {}
        for info_set, node in self.nodes.items():
            strategy[info_set] = node.get_average_strategy()
        return strategy

class KuhnNode:
    """
    Represents a node in the CFR calculation for Kuhn Poker.
    It stores the cumulative regrets and strategy for a given information set.
    """
    def __init__(self, info_set: str):
        self.info_set = info_set
        self.regret_sum = np.zeros(2) # For Pass and Bet
        self.strategy_sum = np.zeros(2)
        self.num_actions = 2

    def get_strategy(self, reach_probability: float) -> np.ndarray:
        """
        Calculates the current strategy for this node based on regret matching.
        """
        # Regret matching
        strategy = np.maximum(0, self.regret_sum)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            # Default to uniform random strategy if no regrets are positive
            strategy = np.full(self.num_actions, 1.0 / self.num_actions)
            
        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        Calculates the average strategy over all iterations.
        This is the strategy that should be used in practice.
        """
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.full(self.num_actions, 1.0 / self.num_actions)

if __name__ == "__main__":
    trainer = KuhnCFRTrainer()
    trainer.train(50000)
    
    final_strategy = trainer.get_final_strategy()
    
    print("\n--- Final Average Strategy (Nash Equilibrium Approximation) ---")
    # Sort by card, then history length, then history
    for info_set, strategy in sorted(final_strategy.items(), key=lambda x: (x[0][0], len(x[0][1:]), x[0][1:])):
        card = info_set[0]
        history = info_set[1:]
        # Player is 1-indexed
        player = (len(history)) % 2 + 1 if len(history) < 2 or history != 'pb' else 1
        if history == 'pb':
            player = 1
        else:
            player = (len(history)) % 2 + 1
        
        print(f"Player {player}, Card: {card}, History: '{history}' -> Bet Prob: {strategy[BET]:.3f}")
