#!/usr/bin/env python3
"""
Generate LSTM performance distribution plot.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_lstm_distribution_plot():
    """Create LSTM performance distribution plot."""
    try:
        with open('hyperparameter_search_results_lstm.json', 'r') as f:
            lstm_data = json.load(f)
        
        valid_scores = []
        for config in lstm_data:
            score = config.get('mean_exploitability')
            if score and score != float('inf') and not np.isnan(score):
                valid_scores.append(score)
        
        if not valid_scores:
            print("No valid LSTM results found")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(valid_scores, bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
        plt.axvline(np.mean(valid_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(valid_scores):.3f}')
        plt.axvline(np.min(valid_scores), color='blue', linestyle='--', linewidth=2, 
                   label=f'Best: {np.min(valid_scores):.3f}')
        
        plt.xlabel('Exploitability', fontsize=12)
        plt.ylabel('Number of Configurations', fontsize=12)
        plt.title(f'LSTM Performance Distribution (n={len(valid_scores)} successful configs)', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/lstm_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Generated: plots/lstm_performance_distribution.png")
        
    except FileNotFoundError:
        print("LSTM results file not found")

if __name__ == "__main__":
    create_lstm_distribution_plot()