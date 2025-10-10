import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

from qagent.agent import ActorCriticAgent
from qagent.poker_environment import PokerEnvironment, EnvironmentConfig, EnvironmentConfig
from qagent.configs.default_config import (
    AGENT_CONFIG,
    ENV_CONFIG,
    PROJECT_CONFIG,
    OPPONENTS
)
from qagent.analysis.plot import plot_reward_distribution, plot_win_rate_pie
from qagent.evaluation.statistical_tests import bootstrap_ci

def evaluate_agent(agent: ActorCriticAgent, opponent_name: str, episodes: int, device: str) -> dict:
    """
    Evaluates the agent against a specific opponent.

    Args:
        agent (ActorCriticAgent): The trained agent to evaluate.
        opponent_name (str): The name of the opponent policy to use.
        episodes (int): The number of episodes to run the evaluation for.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation results (rewards, win/loss/draw stats).
    """
    print(f"\n--- Evaluating against {opponent_name} for {episodes} episodes ---")
    
    # Setup environment with the specified opponent
    eval_env_config = EnvironmentConfig(**ENV_CONFIG)
    eval_env = PokerEnvironment(config=eval_env_config)
    opponent_policy = OPPONENTS[opponent_name]()
    eval_env.set_opponent(opponent_policy)
    
    agent.actor.to(device)
    agent.critic.to(device)
    agent.actor.eval()
    agent.critic.eval()

    all_rewards = []
    wins = 0
    losses = 0
    draws = 0

    for _ in tqdm(range(episodes), desc=f"Evaluating vs {opponent_name}"):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        
        with torch.no_grad():
            while not done:
                action, _ = agent.select_action(obs, info['action_mask'])
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
        
        all_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
            
    return {
        "rewards": np.array(all_rewards),
        "win_loss_draw": {"wins": wins, "losses": losses, "draws": draws}
    }

def run_evaluation(model_path: str, episodes_per_opponent: int, opponents_to_eval: list):
    """
    Main function to run the full evaluation suite.

    Args:
        model_path (str): Path to the directory containing the trained model files (actor.pt, critic.pt).
        episodes_per_opponent (int): Number of episodes to evaluate against each opponent.
        opponents_to_eval (list): A list of opponent names to evaluate against.
    """
    device = 'cpu'  # Evaluation is typically less intensive, CPU is fine
    
    # --- 0. Determine model name for output files ---
    model_name = os.path.basename(model_path)
    if not model_name: # Handles cases like "path/to/model/"
        model_name = os.path.basename(os.path.dirname(model_path))

    # --- 1. Load Agent ---
        # Initialize a dummy environment to get dimensions
    temp_env_config = EnvironmentConfig(**ENV_CONFIG)
    temp_env = PokerEnvironment(config=temp_env_config)
    state_dim = temp_env.get_state_dimension()
    action_dim = temp_env.get_action_dimension()
    
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=AGENT_CONFIG['lr'],
        gamma=AGENT_CONFIG['gamma'],
        device=device
    )
    try:
        agent.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model files not found in {model_path}. Please check the path.")
        return

    # --- 2. Run Evaluation ---
    results = {}
    for opponent_name in opponents_to_eval:
        if opponent_name not in OPPONENTS:
            print(f"Warning: Opponent '{opponent_name}' not found. Skipping.")
            continue
        results[opponent_name] = evaluate_agent(agent, opponent_name, episodes_per_opponent, device)

    # --- 3. Analyze and Report Results ---
    print("\n\n--- Evaluation Summary ---")
    report_df = pd.DataFrame()
    
    for opponent_name, res in results.items():
        rewards = res['rewards']
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ci_lower, ci_upper = bootstrap_ci(rewards)
        win_rate = res['win_loss_draw']['wins'] / episodes_per_opponent * 100
        
        print(f"\nResults against {opponent_name}:")
        print(f"  - Mean Reward: {mean_reward:.2f} (Std: {std_reward:.2f})")
        print(f"  - 95% CI for Mean Reward: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  - Win Rate: {win_rate:.1f}%")
        
        report_df = pd.concat([report_df, pd.DataFrame([{
            "opponent": opponent_name,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "win_rate": win_rate
        }])], ignore_index=True)

    # --- 4. Save Results and Plots ---
    eval_results_dir = os.path.join(PROJECT_CONFIG['results_dir'], 'evaluation')
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # Save raw rewards for statistical analysis
    for opponent_name, res in results.items():
        rewards_filename = f'{model_name}_rewards_{opponent_name}.npy'
        np.save(os.path.join(eval_results_dir, rewards_filename), res['rewards'])

    # Save summary CSV
    summary_filename = f'{model_name}_evaluation_summary.csv'
    report_df.to_csv(os.path.join(eval_results_dir, summary_filename), index=False)
    print(f"\nSaved evaluation summary to {os.path.join(eval_results_dir, summary_filename)}")
    print(f"Saved raw reward arrays to {eval_results_dir}")

    # Generate and save plots
    for opponent_name, res in results.items():
        plot_reward_distribution(
            pd.DataFrame(res['rewards'], columns=['reward']),
            save_path=os.path.join(eval_results_dir, f'{model_name}_reward_dist_{opponent_name}.png')
        )
        plot_win_rate_pie(
            res['win_loss_draw'],
            save_path=os.path.join(eval_results_dir, f'{model_name}_win_rate_{opponent_name}.png')
        )
    print(f"Saved analysis plots to {eval_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Poker RL agent.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the directory containing the saved model weights (actor.pt, critic.pt)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="Number of episodes to run against each opponent."
    )
    parser.add_argument(
        "--opponents",
        nargs='+',
        default=['RandomBot', 'CallBot', 'TightAggressiveBot', 'LoosePassiveBot'],
        help="List of opponent names to evaluate against."
    )
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        episodes_per_opponent=args.episodes,
        opponents_to_eval=args.opponents
    )