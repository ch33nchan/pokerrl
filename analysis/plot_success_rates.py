#!/usr/bin/env python3
"""
Generate hyperparameter search success rate comparison plot.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_hyperparameter_results():
    """Load and process hyperparameter search results."""
    results = {}
    
    for arch in ['baseline', 'lstm', 'transformer']:
        try:
            with open(f'hyperparameter_search_results_{arch}.json', 'r') as f:
                data = json.load(f)
            
            valid_results = []
            for config in data:
                if (config.get('mean_exploitability') and 
                    config['mean_exploitability'] != float('inf') and
                    not np.isnan(config['mean_exploitability'])):
                    valid_results.append(config['mean_exploitability'])
            
            results[arch] = {
                'valid_count': len(valid_results),
                'total_count': len(data),
                'success_rate': len(valid_results) / len(data) if data else 0
            }
            
        except FileNotFoundError:
            results[arch] = {'valid_count': 0, 'total_count': 0, 'success_rate': 0}
    
    return results

def create_success_rate_plot():
    """Create success rate comparison plot."""
    results = load_hyperparameter_results()
    
    architectures = ['Baseline', 'LSTM', 'Transformer']
    success_rates = [results['baseline']['success_rate'] * 100,
                    results['lstm']['success_rate'] * 100,
                    results['transformer']['success_rate'] * 100]
    total_configs = [results['baseline']['total_count'],
                    results['lstm']['total_count'],
                    results['transformer']['total_count']]
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#2ca02c', '#ff7f0e']
    bars = plt.bar(architectures, success_rates, color=colors, alpha=0.8, edgecolor='black')
    
    plt.ylabel('Training Success Rate (%)', fontsize=12)
    plt.title('Training Success Rates by Architecture', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add configuration count labels
    for bar, total in zip(bars, total_configs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{total} configs', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/hyperparameter_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: plots/hyperparameter_success_rates.png")

if __name__ == "__main__":
    create_success_rate_plot()