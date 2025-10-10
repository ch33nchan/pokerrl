import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from typing import Dict, Callable

from qagent.environments.kuhn_poker import KuhnPoker, INFO_SET_SIZE, NUM_ACTIONS, NUM_CARDS
from qagent.environments.kuhn_poker import KuhnPoker, INFO_SET_SIZE, NUM_ACTIONS, NUM_CARDS
from qagent.agents.deep_cfr import RegretNet, DeepCFRTrainer
from qagent.evaluation.exploitability import ExploitabilityCalculator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_nn_policy_map(model_path: str) -> Dict[str, np.ndarray]:
    """
    Loads a trained RegretNet model and generates a full policy map for all
    information sets in Kuhn Poker.

    Args:
        model_path (str): Path to the saved .pt model file.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping info_set strings to strategy probabilities.
    """
    print(f"Loading model from {model_path} and generating policy map...")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}.")
        print("Please run 'python qagent/games/deep_cfr.py' first to train and save a model.")
        sys.exit(1)

    # Load the network
    model = RegretNet(input_size=INFO_SET_SIZE, output_size=NUM_ACTIONS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    policy_map = {}
    game = KuhnPoker()
    
    # There are 12 non-terminal info sets in Kuhn Poker
    # Cards: 0 (J), 1 (Q), 2 (K)
    # Histories: "", "p", "b", "pb"
    cards = range(NUM_CARDS)
    histories = ["", "p", "b", "pb"]

    for card in cards:
        for history in histories:
            # Player is determined by history length
            player = len(history) % 2
            
            # Create a dummy state to generate the info set vector
            dummy_state = {
                "cards": [card, 0] if player == 0 else [0, card],
                "history": history
            }
            info_set_str = f"{card}{history}"
            info_set_vec = game.get_info_set_vector(dummy_state)

            # Get regrets from the network
            with torch.no_grad():
                regrets = model(torch.FloatTensor(info_set_vec).to(DEVICE))
            
            # Convert regrets to strategy via regret matching
            positive_regrets = torch.clamp(regrets, min=0)
            regret_sum = torch.sum(positive_regrets)
            if regret_sum > 0:
                strategy = positive_regrets / regret_sum
            else:
                strategy = torch.ones(NUM_ACTIONS) / NUM_ACTIONS
            
            policy_map[info_set_str] = strategy.cpu().numpy()

    print("Policy map generated successfully.")
    return policy_map

if __name__ == '__main__':
    # Define the path to the saved model
    MODEL_SAVE_PATH = "models/deep_cfr_avg_strategy_net_p0.pt"

    # 1. Generate the policy map from the trained neural network
    nn_policy_map = create_nn_policy_map(MODEL_SAVE_PATH)

    # 2. Use the ExploitabilityCalculator to measure the policy's performance
    print("\n--- Calculating Exploitability of Deep CFR Policy ---")
    # Reduce iterations for a quicker, approximate result.
    exploitability_calc = ExploitabilityCalculator(br_iterations=20000, sim_iterations=100000)
    exploitability = exploitability_calc.measure_exploitability(nn_policy_map)

    print("\n--- Results ---")
    print(f"Total Exploitability of the Deep CFR-trained policy: {exploitability:.6f}")
    print("(A value very close to 0 indicates a near-perfect Nash Equilibrium strategy)")
