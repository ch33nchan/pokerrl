import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse

def load_evaluation_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all evaluation summary CSVs from a directory into a single DataFrame.
    
    Args:
        results_dir (str): The directory containing the evaluation CSV files.
        
    Returns:
        pd.DataFrame: A consolidated DataFrame with a new 'model_name' column.
    """
    csv_files = glob.glob(os.path.join(results_dir, '*_evaluation_summary.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No evaluation summary files found in {results_dir}")
        
    all_data = []
    for f in csv_files:
        model_name = os.path.basename(f).replace('_evaluation_summary.csv', '')
        df = pd.read_csv(f)
        df['model_name'] = model_name
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def plot_mean_reward_comparison(data: pd.DataFrame, save_path: str):
    """
    Generates a bar plot comparing the mean reward of each model against each opponent.
    
    Args:
        data (pd.DataFrame): The consolidated evaluation data.
        save_path (str): Path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.barplot(data=data, x='opponent', y='mean_reward', hue='model_name', ax=ax)
    
    ax.set_title('Comparison of Mean Reward vs. Opponents', fontsize=18, pad=20)
    ax.set_xlabel('Opponent Type', fontsize=12)
    ax.set_ylabel('Mean Reward per Hand', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(title='Training Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved mean reward comparison plot to {save_path}")

def plot_win_rate_comparison(data: pd.DataFrame, save_path: str):
    """
    Generates a bar plot comparing the win rate of each model against each opponent.
    
    Args:
        data (pd.DataFrame): The consolidated evaluation data.
        save_path (str): Path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.barplot(data=data, x='opponent', y='win_rate', hue='model_name', ax=ax)
    
    ax.set_title('Comparison of Win Rate vs. Opponents', fontsize=18, pad=20)
    ax.set_xlabel('Opponent Type', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.legend(title='Training Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved win rate comparison plot to {save_path}")

def main(results_dir: str, output_dir: str):
    """
    Main function to load data and generate all comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        full_data = load_evaluation_data(results_dir)
        
        # Generate and save plots
        plot_mean_reward_comparison(full_data, os.path.join(output_dir, 'mean_reward_comparison.png'))
        plot_win_rate_comparison(full_data, os.path.join(output_dir, 'win_rate_comparison.png'))
        
        print("\nComparative analysis complete.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare performance of different trained models.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/evaluation",
        help="Directory where the evaluation summary CSVs are stored."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Directory to save the comparison plots."
    )
    args = parser.parse_args()
    
    main(results_dir=args.results_dir, output_dir=args.output_dir)
