#!/usr/bin/env python3
"""
Generate ablation study convergence comparison plot.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_convergence_plot():
    """Create ablation study convergence plot."""
    try:
        with open('ablation_study_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Ablation results not found")
        return
    
    iterations = [5000, 10000, 15000, 20000]
    
    # Process baseline data
    baseline_curves = []
    for run in data.get('baseline', []):
        if run:
            curve = []
            for iter_target in iterations:
                iter_result = [r for r in run if r[0] == iter_target]
                if iter_result:
                    curve.append(iter_result[0][1])
            if len(curve) == len(iterations):
                baseline_curves.append(curve)
    
    # Process LSTM data
    lstm_curves = []
    for run in data.get('lstm', []):
        if run:
            curve = []
            for iter_target in iterations:
                iter_result = [r for r in run if r[0] == iter_target]
                if iter_result:
                    curve.append(iter_result[0][1])
            if len(curve) == len(iterations):
                lstm_curves.append(curve)
    
    if not baseline_curves or not lstm_curves:
        print("Insufficient data for convergence plot")
        return
    
    baseline_curves = np.array(baseline_curves)
    lstm_curves = np.array(lstm_curves)
    
    plt.figure(figsize=(12, 8))
    
    # Calculate means and stds
    baseline_mean = np.mean(baseline_curves, axis=0)
    baseline_std = np.std(baseline_curves, axis=0)
    lstm_mean = np.mean(lstm_curves, axis=0)
    lstm_std = np.std(lstm_curves, axis=0)
    
    # Plot with error bars
    plt.errorbar(iterations, baseline_mean, yerr=baseline_std, 
               label='Baseline', marker='o', linewidth=3, capsize=8, markersize=8, color='#d62728')
    plt.errorbar(iterations, lstm_mean, yerr=lstm_std, 
               label='LSTM', marker='s', linewidth=3, capsize=8, markersize=8, color='#2ca02c')
    
    # Plot individual runs as thin lines
    for curve in baseline_curves:
        plt.plot(iterations, curve, color='#d62728', alpha=0.3, linewidth=1)
    for curve in lstm_curves:
        plt.plot(iterations, curve, color='#2ca02c', alpha=0.3, linewidth=1)
    
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Exploitability', fontsize=14)
    plt.title('Convergence Comparison: Baseline vs LSTM', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add final performance annotation
    baseline_final = baseline_mean[-1]
    lstm_final = lstm_mean[-1]
    improvement = (baseline_final - lstm_final) / baseline_final * 100
    
    plt.text(0.02, 0.98, f'LSTM improves {improvement:.1f}% over Baseline', 
           transform=plt.gca().transAxes, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
           fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/ablation_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: plots/ablation_convergence.png")

if __name__ == "__main__":
    create_convergence_plot()