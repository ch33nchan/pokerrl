import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict

from qagent.agent import ActorCriticAgent
from qagent.poker_environment import PokerEnvironment, EnvironmentConfig
from qagent.configs.default_config import AGENT_CONFIG, ENV_CONFIG, OPPONENTS

def analyze_behavior(agent: ActorCriticAgent, opponent_name: str, episodes: int, device: str) -> dict:
    """
    Analyzes the agent's action distribution against a specific opponent.

    Args:
        agent (ActorCriticAgent): The agent to analyze.
        opponent_name (str): The name of the opponent policy.
        episodes (int): The number of episodes to run.
        device (str): The device to run on.

    Returns:
        dict: A dictionary with counts of each action type.
    """
    print(f"\n--- Analyzing behavior against {opponent_name} for {episodes} episodes ---")
    
    eval_env_config = EnvironmentConfig(**ENV_CONFIG)
    eval_env = PokerEnvironment(config=eval_env_config)
    opponent_policy = OPPONENTS[opponent_name]()
    eval_env.set_opponent(opponent_policy)
    
    agent.actor.to(device)
    agent.critic.to(device)
    agent.actor.eval()
    agent.critic.eval()

    action_counts = defaultdict(int)
    total_actions = 0

    for _ in tqdm(range(episodes), desc=f"Analyzing vs {opponent_name}"):
        obs, info = eval_env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                action, _ = agent.select_action(obs, info['action_mask'])
                
                # Categorize and count the action
                if action == 0:
                    action_counts['Fold'] += 1
                elif action == 1:
                    action_counts['Check/Call'] += 1
                else: # Actions 2, 3, 4, 5 are bet/raise
                    action_counts['Bet/Raise'] += 1
                total_actions += 1
                
                obs, _, done, info = eval_env.step(action)
    
    action_counts['Total'] = total_actions
    return action_counts

def run_behavioral_analysis(model_path: str, episodes_per_opponent: int, opponents_to_eval: list):
    """
    Main function to run the full behavioral analysis.

    Args:
        model_path (str): Path to the trained model directory.
        episodes_per_opponent (int): Number of episodes per opponent.
        opponents_to_eval (list): List of opponent names.
    """
    device = 'cpu'
    model_name = os.path.basename(model_path) or os.path.basename(os.path.dirname(model_path))
    
    # --- 1. Load Agent ---
    temp_env = PokerEnvironment(config=EnvironmentConfig(**ENV_CONFIG))
    state_dim = temp_env.get_state_dimension()
    action_dim = temp_env.get_action_dimension()
    
    agent = ActorCriticAgent(state_dim, action_dim, AGENT_CONFIG['lr'], AGENT_CONFIG['gamma'], device)
    try:
        agent.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model files not found in {model_path}.")
        return

    # --- 2. Run Analysis ---
    all_results = []
    for opponent_name in opponents_to_eval:
        if opponent_name not in OPPONENTS:
            print(f"Warning: Opponent '{opponent_name}' not found. Skipping.")
            continue
        
        counts = analyze_behavior(agent, opponent_name, episodes_per_opponent, device)
        total = counts['Total']
        
        if total > 0:
            frequencies = {
                "Opponent": opponent_name,
                "Fold (%)": (counts['Fold'] / total) * 100,
                "Check/Call (%)": (counts['Check/Call'] / total) * 100,
                "Bet/Raise (%)": (counts['Bet/Raise'] / total) * 100,
            }
            all_results.append(frequencies)

    # --- 3. Report Results ---
    print(f"\n\n--- Behavioral Analysis Summary for '{model_name}' ---")
    if not all_results:
        print("No results to display.")
        return
        
    report_df = pd.DataFrame(all_results)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the behavior of a trained Poker RL agent.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/final_model",
        help="Path to the directory containing the saved model weights."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes to run against each opponent for analysis."
    )
    parser.add_argument(
        "--opponents",
        nargs='+',
        default=['RandomBot', 'CallBot', 'TightAggressiveBot', 'LoosePassiveBot'],
        help="List of opponent names to analyze against."
    )
    args = parser.parse_args()
    
    run_behavioral_analysis(
        model_path=args.model_path,
        episodes_per_opponent=args.episodes,
        opponents_to_eval=args.opponents
    )
