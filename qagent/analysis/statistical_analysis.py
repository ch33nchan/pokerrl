import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import os
import argparse
import glob

def cohen_d(x, y):
    """
    Calculates Cohen's d for independent samples.
    
    Args:
        x (np.array): Sample 1.
        y (np.array): Sample 2.
        
    Returns:
        float: The calculated Cohen's d.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def perform_statistical_analysis(results_dir: str, model1_name: str, model2_name: str):
    """
    Performs statistical tests comparing the reward distributions of two models.
    
    Args:
        results_dir (str): Directory containing the raw reward .npy files.
        model1_name (str): Name of the first model (e.g., 'final_model').
        model2_name (str): Name of the second model (e.g., 'final_model_no_curriculum').
    """
    print(f"--- Statistical Comparison: '{model1_name}' vs. '{model2_name}' ---")
    
    opponents = ['RandomBot', 'CallBot', 'TightAggressiveBot', 'LoosePassiveBot']
    analysis_results = []

    for opponent in opponents:
        try:
            # Load raw reward arrays
            rewards1_path = os.path.join(results_dir, f'{model1_name}_rewards_{opponent}.npy')
            rewards2_path = os.path.join(results_dir, f'{model2_name}_rewards_{opponent}.npy')
            
            rewards1 = np.load(rewards1_path)
            rewards2 = np.load(rewards2_path)
            
            # Perform independent t-test
            t_stat, p_value = ttest_ind(rewards1, rewards2, equal_var=False) # Welch's t-test
            
            # Calculate effect size (Cohen's d)
            effect_size = cohen_d(rewards1, rewards2)
            
            analysis_results.append({
                "Opponent": opponent,
                "T-statistic": t_stat,
                "P-value": p_value,
                "Cohen's d": effect_size,
                "Result": "Statistically Significant" if p_value < 0.05 else "Not Significant"
            })
            
        except FileNotFoundError:
            print(f"Warning: Reward files for opponent '{opponent}' not found for one or both models. Skipping.")
            continue
            
    if not analysis_results:
        print("No analysis was performed. Please check file paths and names.")
        return

    # Display results in a clean table
    results_df = pd.DataFrame(analysis_results)
    print(results_df.to_string(index=False))
    
    print("\n--- Interpretation ---")
    print("P-value < 0.05: The difference in mean rewards is statistically significant.")
    print("Cohen's d: Measures the size of the difference (small: ~0.2, medium: ~0.5, large: ~0.8).")
    print(f"A positive Cohen's d means '{model1_name}' performed better on average.")
    print(f"A negative Cohen's d means '{model2_name}' performed better on average.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical analysis on model evaluation results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/evaluation",
        help="Directory where the raw reward .npy files are stored."
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="final_model",
        help="Name of the first model for comparison."
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="final_model_no_curriculum",
        help="Name of the second model for comparison."
    )
    args = parser.parse_args()
    
    perform_statistical_analysis(
        results_dir=args.results_dir,
        model1_name=args.model1,
        model2_name=args.model2
    )
