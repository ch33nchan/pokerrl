#!/usr/bin/env python3
"""
Generate final performance comparison plot.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def create_final_comparison_plot():
    """Create final performance comparison plot."""
    try:
        with open('ablation_study_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Ablation results not found")
        return
    
    # Extract final values (20000 iterations)
    baseline_final = []
    lstm_final = []
    
    for run in data.get('baseline', []):
        if run:
            final_result = [r for r in run if r[0] == 20000]
            if final_result:
                baseline_final.append(final_result[0][1])
    
    for run in data.get('lstm', []):
        if run:
            final_result = [r for r in run if r[0] == 20000]
            if final_result:
                lstm_final.append(final_result[0][1])
    
    if not baseline_final or not lstm_final:
        print("Insufficient final performance data")
        return
    
    methods = ['Baseline', 'LSTM']
    means = [np.mean(baseline_final), np.mean(lstm_final)]
    stds = [np.std(baseline_final), np.std(lstm_final)]
    
    plt.figure(figsize=(8, 6))
    colors = ['#d62728', '#2ca02c']
    bars = plt.bar(methods, means, yerr=stds, capsize=8, alpha=0.8, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    plt.ylabel('Final Exploitability', fontsize=12)
    plt.title('Final Performance Comparison (20,000 iterations)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add statistical significance
    if len(baseline_final) > 1 and len(lstm_final) > 1:
        t_stat, p_value = ttest_ind(baseline_final, lstm_final)
        significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        plt.text(0.5, max(means) + max(stds) + 0.1, f'p={p_value:.3f} {significance}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                transform=plt.gca().transData)
    
    # Add improvement percentage
    improvement = (means[0] - means[1]) / means[0] * 100
    plt.text(0.02, 0.98, f'LSTM improvement: {improvement:.1f}%', 
           transform=plt.gca().transAxes, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
           fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: plots/final_performance_comparison.png")

if __name__ == "__main__":
    create_final_comparison_plot()