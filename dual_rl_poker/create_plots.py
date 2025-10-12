#!/usr/bin/env python3
"""Create plots from experimental results."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_results():
    """Load all experimental results."""
    results_dir = Path("results")
    all_results = {}

    for result_file in results_dir.glob("*_results.json"):
        with open(result_file, 'r') as f:
            result = json.load(f)
            key = f"{result['game']}_{result['method']}"
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(result)

    return all_results

def plot_exploitability_curves(all_results, save_path="plots/exploitability_curves.png"):
    """Plot exploitability curves during training."""
    # Create plots directory
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    games = ['kuhn_poker', 'leduc_poker']
    algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Professional color scheme

    for game_idx, game in enumerate(games):
        ax = axes[game_idx]

        for alg_idx, algorithm in enumerate(algorithms):
            key = f"{game}_{algorithm}"
            if key in all_results and all_results[key]:
                # Get average learning curve
                all_curves = []
                for result in all_results[key]:
                    if 'evaluation_history' in result:
                        eval_history = result['evaluation_history']
                        iterations = [eval_item['iteration'] for eval_item in eval_history]
                        exploitability = [eval_item['exploitability'] for eval_item in eval_history]
                        all_curves.append((iterations, exploitability))

                if all_curves:
                    # Calculate mean and confidence intervals
                    max_iter = max(max(curve[0]) for curve in all_curves)
                    mean_curve = np.zeros(max_iter)
                    std_curve = np.zeros(max_iter)
                    count_curve = np.zeros(max_iter)

                    for iterations, exploitability in all_curves:
                        for i, (it, exp) in enumerate(zip(iterations, exploitability)):
                            mean_curve[it-1] += exp
                            std_curve[it-1] += exp * exp
                            count_curve[it-1] += 1

                    # Calculate mean and std
                    mask = count_curve > 0
                    mean_curve[mask] /= count_curve[mask]
                    std_curve[mask] = np.sqrt(std_curve[mask] / count_curve[mask] - mean_curve[mask] ** 2)

                    # Plot mean curve with confidence interval
                    x_axis = np.arange(1, max_iter + 1)
                    ax.plot(x_axis, mean_curve, label=algorithm.replace('_', ' ').title(),
                           color=colors[alg_idx], linewidth=2.5)
                    ax.fill_between(x_axis,
                                   mean_curve - std_curve,
                                   mean_curve + std_curve,
                                   color=colors[alg_idx], alpha=0.2)

        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Exploitability (game units)')
        ax.set_title(f'{game.replace("_", " ").title()}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved exploitability curves to {save_path}")
    return fig

def plot_comparison_bar_chart(all_results, save_path="plots/performance_comparison.png"):
    """Create comparison bar chart."""
    Path(save_path).parent.mkdir(exist_ok=True)

    # Prepare data
    data = []
    algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    games = ['kuhn_poker', 'leduc_poker']

    for algorithm in algorithms:
        for game in games:
            key = f"{game}_{algorithm}"
            if key in all_results and all_results[key]:
                exploitabilities = []
                for result in all_results[key]:
                    if 'evaluation_history' in result and result['evaluation_history']:
                        final_exploit = result['evaluation_history'][-1]['exploitability']
                        exploitabilities.append(final_exploit)

                if exploitabilities:
                    data.append({
                        'Algorithm': algorithm.replace('_', ' ').title(),
                        'Game': game.replace('_', ' ').title(),
                        'Exploitability': np.mean(exploitabilities),
                        'Std': np.std(exploitabilities),
                        'Count': len(exploitabilities)
                    })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot grouped bar chart
    games_list = df['Game'].unique()
    algorithms_list = df['Algorithm'].unique()
    x = np.arange(len(games_list))
    width = 0.25

    for i, algorithm in enumerate(algorithms_list):
        algorithm_data = df[df['Algorithm'] == algorithm]
        values = []
        errors = []
        for game in games_list:
            game_data = algorithm_data[algorithm_data['Game'] == game]
            if not game_data.empty:
                values.append(game_data.iloc[0]['Exploitability'])
                errors.append(game_data.iloc[0]['Std'])
            else:
                values.append(0)
                errors.append(0)

        ax.bar(x + i * width, values, width, label=algorithm,
               yerr=errors, capsize=5, alpha=0.8)

    ax.set_xlabel('Game')
    ax.set_ylabel('Final Exploitability (game units)')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(games_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to {save_path}")
    return fig

def plot_training_efficiency(all_results, save_path="plots/training_efficiency.png"):
    """Plot training efficiency (performance vs time)."""
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    for alg_idx, algorithm in enumerate(algorithms):
        all_data = []
        for key, results in all_results.items():
            if algorithm in key and results:
                for result in results:
                    if 'evaluation_history' in result and result['evaluation_history']:
                        final_exploit = result['evaluation_history'][-1]['exploitability']
                        training_time = result.get('total_time', 0)
                        all_data.append((training_time, final_exploit))

        if all_data:
            times, exploits = zip(*all_data)
            ax.scatter(times, exploits, label=algorithm.replace('_', ' ').title(),
                      color=colors[alg_idx], marker=markers[alg_idx],
                      s=50, alpha=0.7)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Final Exploitability (game units)')
    ax.set_title('Training Efficiency: Performance vs Time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency plot to {save_path}")
    return fig

def plot_loss_components(all_results, save_path="plots/loss_components.png"):
    """Plot loss components during training."""
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for alg_idx, algorithm in enumerate(algorithms):
        ax = axes[alg_idx]

        # Collect all loss curves for this algorithm
        all_regret_losses = []
        all_strategy_losses = []
        all_value_losses = []

        for key, results in all_results.items():
            if algorithm in key and results:
                for result in results:
                    if 'training_history' in result:
                        history = result['training_history']
                        regret_losses = [h.get('regret_loss', 0) for h in history]
                        strategy_losses = [h.get('strategy_loss', 0) for h in history]
                        value_losses = [h.get('value_loss', 0) for h in history]

                        all_regret_losses.append(regret_losses)
                        all_strategy_losses.append(strategy_losses)
                        all_value_losses.append(value_losses)

        # Plot average loss curves
        if all_regret_losses:
            max_len = max(len(losses) for losses in all_regret_losses)

            # Calculate mean curves
            mean_regret = np.zeros(max_len)
            mean_strategy = np.zeros(max_len)
            mean_value = np.zeros(max_len)

            for regret_losses, strategy_losses, value_losses in zip(
                all_regret_losses, all_strategy_losses, all_value_losses):

                for i in range(len(regret_losses)):
                    mean_regret[i] += regret_losses[i]
                    mean_strategy[i] += strategy_losses[i]
                    mean_value[i] += value_losses[i]

            n_runs = len(all_regret_losses)
            mean_regret /= n_runs
            mean_strategy /= n_runs
            mean_value /= n_runs

            x_axis = range(1, max_len + 1)
            ax.plot(x_axis, mean_regret, label='Regret Loss', linewidth=2)
            ax.plot(x_axis, mean_strategy, label='Strategy Loss', linewidth=2)
            if algorithm == 'armac':
                ax.plot(x_axis, mean_value, label='Value Loss', linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{algorithm.replace("_", " ").title()} Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss components plot to {save_path}")
    return fig

def create_summary_table(all_results, save_path="tables/performance_table.tex"):
    """Create LaTeX table for results."""
    Path(save_path).parent.mkdir(exist_ok=True)

    # Prepare data
    algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    games = ['kuhn_poker', 'leduc_poker']

    table_data = []
    for algorithm in algorithms:
        row = [algorithm.replace('_', ' ').title()]
        for game in games:
            key = f"{game}_{algorithm}"
            if key in all_results and all_results[key]:
                exploitabilities = []
                for result in all_results[key]:
                    if 'evaluation_history' in result and result['evaluation_history']:
                        final_exploit = result['evaluation_history'][-1]['exploitability']
                        exploitabilities.append(final_exploit)

                if exploitabilities:
                    mean_exp = np.mean(exploitabilities)
                    std_exp = np.std(exploitabilities)
                    row.append(f"{mean_exp:.3f} ± {std_exp:.3f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        table_data.append(row)

    # Create LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Algorithm performance comparison across games. Values show mean exploitability (game units) with standard deviation.}\n"
    latex_table += "\\label{tab:performance_comparison}\n"
    latex_table += "\\begin{tabular}{lcc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Algorithm & Kuhn Poker & Leduc Hold'em \\\\\n"
    latex_table += "\\midrule\n"

    for row in table_data:
        latex_table += " & ".join(row) + " \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    with open(save_path, 'w') as f:
        f.write(latex_table)

    print(f"Saved LaTeX table to {save_path}")
    return latex_table

def main():
    """Generate all plots and tables."""
    print("Creating plots from experimental results...")

    # Load results
    all_results = load_results()
    print(f"Loaded results for {len(all_results)} algorithm-game combinations")

    # Create plots
    plot_exploitability_curves(all_results)
    plot_comparison_bar_chart(all_results)
    plot_training_efficiency(all_results)
    plot_loss_components(all_results)

    # Create table
    create_summary_table(all_results)

    print("All plots and tables generated successfully!")

if __name__ == "__main__":
    main()