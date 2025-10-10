"""Plotting utilities for Dual RL Poker experiments."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_learning_curves(experiment_data: Dict[str, Any],
                         save_path: Optional[str] = None,
                         metrics: List[str] = ['loss', 'exploitability']) -> plt.Figure:
    """Plot learning curves for training metrics.

    Args:
        experiment_data: Dictionary containing experiment results
        save_path: Path to save the plot
        metrics: List of metrics to plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, metric in enumerate(metrics[:4]):  # Limit to 4 subplots
        ax = axes[i]

        # Plot each algorithm
        for j, (alg_name, alg_data) in enumerate(experiment_data.items()):
            if isinstance(alg_data, dict) and 'training_history' in alg_data:
                history = alg_data['training_history']
                iterations = [state['iteration'] for state in history]
                values = [state.get(metric, 0) for state in history]

                if values:  # Only plot if we have data
                    ax.plot(iterations, values,
                           label=alg_name.replace('_', ' ').title(),
                           color=colors[j], linewidth=2, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(metrics), 4):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_exploitability_curves(results_dir: str,
                             save_path: Optional[str] = None) -> plt.Figure:
    """Plot exploitability curves across algorithms and games.

    Args:
        results_dir: Directory containing experiment results
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    results_path = Path(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Load all result files
    all_results = {}
    for result_file in results_path.glob("*_results.json"):
        with open(result_file, 'r') as f:
            result = json.load(f)
            key = f"{result['game']}_{result['method']}"
            all_results[key] = result

    # Group by game
    games = set(result['game'] for result in all_results.values())

    for game_idx, game in enumerate(sorted(games)):
        ax = axes[game_idx]

        # Plot each algorithm for this game
        game_results = {k: v for k, v in all_results.items() if v['game'] == game}

        for alg_name, result in game_results.items():
            if 'evaluation_history' in result:
                eval_history = result['evaluation_history']
                iterations = [eval_item['iteration'] for eval_item in eval_history]
                exploitability = [eval_item['exploitability'] for eval_item in eval_history]

                # Calculate moving average for smoother plot
                if len(exploitability) > 10:
                    window = min(10, len(exploitability) // 4)
                    smooth_exploit = pd.Series(exploitability).rolling(window, center=True).mean()
                    ax.plot(iterations, smooth_exploit,
                           label=result['method'].replace('_', ' ').title(),
                           linewidth=2, alpha=0.8)
                else:
                    ax.plot(iterations, exploitability,
                           label=result['method'].replace('_', ' ').title(),
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Exploitability (mbb/h)')
        ax.set_title(f'Exploitability Curves - {game.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_comparison_charts(results_data: Dict[str, Any],
                          save_path: Optional[str] = None) -> plt.Figure:
    """Create comparison charts for final performance metrics.

    Args:
        results_data: Dictionary containing aggregated results
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract final performance metrics
    algorithms = []
    games = []
    final_exploitability = []
    training_time = []
    nash_conv = []

    for exp_id, result in results_data.items():
        if isinstance(result, dict):
            algorithms.append(result.get('method', 'Unknown'))
            games.append(result.get('game', 'Unknown'))

            # Get final exploitability
            if 'evaluation_history' in result and result['evaluation_history']:
                final_exploitability.append(result['evaluation_history'][-1]['exploitability'])
            else:
                final_exploitability.append(0)

            # Get training time
            training_time.append(result.get('total_time', 0))

            # Get final NashConv
            if 'evaluation_history' in result and result['evaluation_history']:
                nash_conv.append(result['evaluation_history'][-1].get('nash_conv', 0))
            else:
                nash_conv.append(0)

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Algorithm': algorithms,
        'Game': games,
        'Final Exploitability': final_exploitability,
        'Training Time (s)': training_time,
        'NashConv': nash_conv
    })

    # Plot 1: Final Exploitability by Algorithm and Game
    ax1 = axes[0, 0]
    pivot_exploit = df.pivot(index='Algorithm', columns='Game', values='Final Exploitability')
    pivot_exploit.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_ylabel('Final Exploitability (mbb/h)')
    ax1.set_title('Final Exploitability Comparison')
    ax1.legend(title='Game')
    ax1.set_yscale('log')

    # Plot 2: Training Time by Algorithm
    ax2 = axes[0, 1]
    time_by_alg = df.groupby('Algorithm')['Training Time (s)'].mean()
    time_by_alg.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Average Training Time by Algorithm')

    # Plot 3: NashConv Comparison
    ax3 = axes[1, 0]
    pivot_nash = df.pivot(index='Algorithm', columns='Game', values='NashConv')
    pivot_nash.plot(kind='bar', ax=ax3, rot=45)
    ax3.set_ylabel('NashConv')
    ax3.set_title('Final NashConv Comparison')
    ax3.legend(title='Game')

    # Plot 4: Performance vs Training Time scatter
    ax4 = axes[1, 1]
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        ax4.scatter(alg_data['Training Time (s)'], alg_data['Final Exploitability'],
                   label=alg, alpha=0.7, s=50)

    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Final Exploitability (mbb/h)')
    ax4.set_title('Performance vs Training Time')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_strategy_analysis(strategy_data: Dict[str, np.ndarray],
                          info_state: str,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot strategy analysis for a specific information state.

    Args:
        strategy_data: Dictionary mapping algorithms to strategy arrays
        info_state: Information state identifier
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algorithms = list(strategy_data.keys())
    num_actions = len(list(strategy_data.values())[0])

    # Plot 1: Strategy comparison as bar chart
    x = np.arange(num_actions)
    width = 0.8 / len(algorithms)

    for i, (alg, strategy) in enumerate(strategy_data.items()):
        ax1.bar(x + i * width, strategy, width,
               label=alg.replace('_', ' ').title(), alpha=0.8)

    ax1.set_xlabel('Action')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Strategy Comparison - {info_state}')
    ax1.set_xticks(x + width * len(algorithms) / 2)
    ax1.set_xticklabels([f'Action {i}' for i in range(num_actions)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Strategy heatmap
    strategy_matrix = np.array(list(strategy_data.values()))
    im = ax2.imshow(strategy_matrix, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Algorithm')
    ax2.set_title('Strategy Heatmap')
    ax2.set_yticks(range(len(algorithms)))
    ax2.set_yticklabels([alg.replace('_', ' ').title() for alg in algorithms])
    ax2.set_xticks(range(num_actions))
    ax2.set_xticklabels([f'{i}' for i in range(num_actions)])

    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_loss_components(training_data: Dict[str, List[Dict[str, float]]],
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot individual loss components over training.

    Args:
        training_data: Dictionary mapping algorithms to training history
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    loss_components = ['regret_loss', 'strategy_loss', 'value_loss']
    titles = ['Regret Loss', 'Strategy Loss', 'Value Loss']

    for idx, (component, title) in enumerate(zip(loss_components, titles)):
        ax = axes[idx]

        for alg_name, history in training_data.items():
            iterations = [state['iteration'] for state in history if component in state]
            values = [state[component] for state in history if component in state]

            if values:
                # Smooth the curve with moving average
                if len(values) > 10:
                    window = min(10, len(values) // 4)
                    smooth_values = pd.Series(values).rolling(window, center=True).mean()
                    ax.plot(iterations, smooth_values,
                           label=alg_name.replace('_', ' ').title(),
                           linewidth=2, alpha=0.8)
                else:
                    ax.plot(iterations, values,
                           label=alg_name.replace('_', ' ').title(),
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_convergence_analysis(convergence_data: Dict[str, Dict[str, List[float]]],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Plot convergence analysis comparing algorithms.

    Args:
        convergence_data: Dictionary containing convergence metrics
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Metric 1: Convergence rate (slope of exploitability decrease)
    ax1 = axes[0, 0]
    for alg, metrics in convergence_data.items():
        if 'convergence_rate' in metrics:
            ax1.bar(alg.replace('_', ' ').title(), metrics['convergence_rate'],
                   alpha=0.8)
    ax1.set_ylabel('Convergence Rate')
    ax1.set_title('Convergence Rate Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Metric 2: Time to convergence
    ax2 = axes[0, 1]
    for alg, metrics in convergence_data.items():
        if 'time_to_convergence' in metrics:
            ax2.bar(alg.replace('_', ' ').title(), metrics['time_to_convergence'],
                   alpha=0.8)
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Time to Convergence')
    ax2.tick_params(axis='x', rotation=45)

    # Metric 3: Final performance stability
    ax3 = axes[1, 0]
    for alg, metrics in convergence_data.items():
        if 'stability_score' in metrics:
            ax3.bar(alg.replace('_', ' ').title(), metrics['stability_score'],
                   alpha=0.8)
    ax3.set_ylabel('Stability Score')
    ax3.set_title('Performance Stability')
    ax3.tick_params(axis='x', rotation=45)

    # Metric 4: Efficiency (performance per training time)
    ax4 = axes[1, 1]
    for alg, metrics in convergence_data.items():
        if 'efficiency' in metrics:
            ax4.bar(alg.replace('_', ' ').title(), metrics['efficiency'],
                   alpha=0.8)
    ax4.set_ylabel('Efficiency (Performance/Time)')
    ax4.set_title('Training Efficiency')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig