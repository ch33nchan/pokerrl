import numpy as np
from collections import defaultdict
import random
import json

from qagent.environments.kuhn_poker import KuhnPoker

class CFRKuhnAgent:
    """
    A Tabular Counterfactual Regret Minimization (CFR) agent for Kuhn Poker.
    """
    def __init__(self, environment: KuhnPoker):
        self.env = environment
        self.num_actions = self.env.get_num_actions()

        # Regret and strategy are stored in dictionaries with info_set as key
        self.regret_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.policy = defaultdict(lambda: np.ones(self.num_actions) / self.num_actions)

    def get_strategy(self, info_set):
        """
        Get the current strategy for a given information set.
        The strategy is based on regret matching.
        """
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)

        if sum_positive_regrets > 0:
            strategy = positive_regrets / sum_positive_regrets
        else:
            # Default to a uniform random strategy if no positive regrets
            strategy = np.ones(self.num_actions) / self.num_actions
        
        return strategy

    def train(self, iterations):
        """
        Train the CFR agent for a given number of iterations.
        This implements the vanilla CFR algorithm.
        """
        utils = np.zeros(self.env.num_players)
        # Get all possible starting hands (chance outcomes)
        chance_outcomes = self.env.get_all_chance_outcomes()
        
        for _ in range(iterations):
            # In each iteration, we traverse the game tree for each player,
            # starting from each possible initial hand.
            for player in range(self.env.num_players):
                # The overall utility for the player in this iteration
                iteration_utility = 0
                for initial_state in chance_outcomes:
                    # Reach probabilities are 1 at the start.
                    player_reach_prob = 1.0
                    opponent_reach_prob = 1.0
                    # The utility from a single traversal is weighted by the chance probability (1/num_deals)
                    # but since we sum over all deals, we can just average at the end.
                    iteration_utility += self._walk_tree(initial_state, player, player_reach_prob, opponent_reach_prob)
                
                # Average utility over all deals for this iteration
                utils[player] += iteration_utility / len(chance_outcomes)
        
        print(f"Completed {iterations} training iterations.")

    def _walk_tree(self, state, training_player, player_reach_prob, opponent_reach_prob):
        """
        Recursively traverse the game tree using CFR.
        """
        if self.env.is_terminal(state):
            return self.env.get_payoff(state, training_player)

        if state['player'] == training_player:
            info_set = self.env.get_state_string(state)
            strategy = self.get_strategy(info_set)
            self.policy[info_set] = strategy

            # Update average strategy sum
            self.strategy_sum[info_set] += player_reach_prob * strategy

            # Counterfactual values for each action
            cf_utils = np.zeros(self.num_actions)
            util = 0

            legal_actions = self.env.get_legal_actions(state)
            for action in legal_actions:
                next_state = self.env.get_next_state(state, action)
                cf_utils[action] = self._walk_tree(next_state, training_player, player_reach_prob * strategy[action], opponent_reach_prob)
                util += strategy[action] * cf_utils[action]

            # Update regrets
            for action in legal_actions:
                # Regret is the difference between the utility of an action and the average utility of the node
                regret = cf_utils[action] - util
                self.regret_sum[info_set][action] += opponent_reach_prob * regret
            
            return util
        else: # Opponent's turn
            info_set = self.env.get_state_string(state)
            strategy = self.get_strategy(info_set)
            
            util = 0
            legal_actions = self.env.get_legal_actions(state)
            for action in legal_actions:
                next_state = self.env.get_next_state(state, action)
                util += strategy[action] * self._walk_tree(next_state, training_player, player_reach_prob, opponent_reach_prob * strategy[action])
            return util

    def get_average_strategy(self):
        """
        Calculate the average strategy over all iterations.
        """
        avg_strategy = {}
        for info_set, s_sum in self.strategy_sum.items():
            norm_sum = np.sum(s_sum)
            if norm_sum > 0:
                avg_strategy[info_set] = s_sum / norm_sum
            else:
                avg_strategy[info_set] = np.ones(self.num_actions) / self.num_actions
        return avg_strategy

if __name__ == '__main__':
    # Example of how to train the tabular CFR agent
    kuhn_env = KuhnPoker()
    cfr_agent = CFRKuhnAgent(kuhn_env)
    
    print("Training Tabular CFR for Kuhn Poker...")
    cfr_agent.train(100000)
    
    average_strategy = cfr_agent.get_average_strategy()
    
    print("\nTraining complete.")
    print(f"Learned policy for {len(average_strategy)} information sets.")

    # Convert numpy arrays to lists for JSON serialization
    serializable_strategy = {k: v.tolist() for k, v in average_strategy.items()}
    
    # Save the policy to a file
    policy_filename = "kuhn_tabular_cfr_policy.json"
    with open(policy_filename, 'w') as f:
        json.dump(serializable_strategy, f)
    print(f"\nSaved average strategy to {policy_filename}")
    
    # Print a few learned strategies
    print("\nExample strategies (Player, Card, History -> [Pass, Bet]):")
    for info_set in sorted(average_strategy.keys())[:5]:
        strategy_str = np.array2string(average_strategy[info_set], formatter={'float_kind':lambda x: "%.3f" % x})
        print(f"{info_set} -> {strategy_str}")

    # A known optimal strategy for Kuhn Poker involves betting with a King,
    # betting with a Jack 1/3 of the time, and checking/calling with a Queen.
    # Let's check the strategy for Player 0 holding a Jack with no actions taken yet.
    info_set_p0_jack = "Card:0-Hist:" # Player 0, Card 0 (Jack), History ()
    if info_set_p0_jack in average_strategy:
        print("\nStrategy for Player 0 with a Jack (should be ~[0.667, 0.333]):")
        print(average_strategy[info_set_p0_jack])

    info_set_p0_king = "Card:2-Hist:" # Player 0, Card 2 (King), History ()
    if info_set_p0_king in average_strategy:
        print("\nStrategy for Player 0 with a King (should be ~[0.0, 1.0]):")
        print(average_strategy[info_set_p0_king])
        
    # To properly evaluate exploitability, we would need to integrate this with the
    # evaluation scripts, similar to the Deep CFR agents.
    # This example just demonstrates the training process.
