"""
Figure and Table Generation for Deep CFR Architecture Study

This module generates publication-ready figures and tables from experiment results,
including:
- Exploitability curves with bootstrap confidence intervals
- Architecture comparison plots
- Statistical significance tables
- Head-to-head evaluation matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path

from qagent.analysis.comprehensive_statistics import (
    analyze_experiment_results,
    create_summary_table,
    generate_statistical_report,
    bootstrap_confidence_interval
)


class FigureGenerator:
    """Generate publication-ready figures for the Deep CFR study."""

    def __init__(self, results_dir: str = "results", output_dir: str = "paper"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 8)

    def load_experiment_data(self) -> List[Dict[str, Any]]:
        """Load experiment data from JSON files."""
        all_results = []

        # Try to load comprehensive results
        all_results_file = self.results_dir / "all_experiments.json"
        if all_results_file.exists():
            with open(all_results_file, 'r') as f:
                all_results = json.load(f)

        return all_results

    def plot_exploitability_curves(
        self,
        experiment_data: List[Dict[str, Any]],
        game: str = "kuhn_poker"
    ) -> plt.Figure:
        """
        Plot exploitability curves with bootstrap confidence intervals.

        Args:
            experiment_data: List of experiment results
            game: Game to plot

        Returns:
            matplotlib Figure
        """
        # Filter data for the specified game
        game_data = [r for r in experiment_data if r.get('game') == game and r.get('status') == 'completed']

        if not game_data:
            raise ValueError(f"No successful experiments found for {game}")

        # Group by architecture
        arch_data = {}
        for result in game_data:
            arch = result.get('architecture', 'unknown')
            if arch not in arch_data:
                arch_data[arch] = []
            arch_data[arch].append(result)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Mean trajectories with confidence intervals
        for i, (arch, results) in enumerate(arch_data.items()):
            # Collect all trajectories
            all_iterations = []
            all_exploitabilities = []

            for result in results:
                if 'iterations' in result and 'exploitability' in result:
                    iterations = result['iterations']
                    exploitabilities = result['exploitability']

                    # Pad or truncate to common length
                    max_len = max(len(all_iterations), len(iterations)) if all_iterations else len(iterations)
                    if len(iterations) < max_len:
                        iterations = iterations + [iterations[-1]] * (max_len - len(iterations))
                        exploitabilities = exploitabilities + [exploitabilities[-1]] * (max_len - len(exploitabilities))

                    all_iterations.append(iterations)
                    all_exploitabilities.append(exploitabilities)

            if all_iterations and all_exploitabilities:
                # Convert to numpy array
                all_iterations = np.array(all_iterations)
                all_exploitabilities = np.array(all_exploitabilities)

                # Calculate mean and confidence intervals
                mean_exploitability = np.mean(all_exploitabilities, axis=0)
                lower_ci = np.percentile(all_exploitabilities, 2.5, axis=0)
                upper_ci = np.percentile(all_exploitabilities, 97.5, axis=0)

                # Plot
                ax1.plot(all_iterations[0], mean_exploitability,
                        label=arch, color=self.colors[i], linewidth=2)
                ax1.fill_between(all_iterations[0], lower_ci, upper_ci,
                                alpha=0.2, color=self.colors[i])

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Exploitability (mBB/100)')
        ax1.set_title(f'{game.replace("_", " ").title()} - Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final performance comparison (scatter/violin)
        final_exploitabilities = []
        arch_labels = []

        for arch, results in arch_data.items():
            arch_final = [r.get('final_exploitability', float('inf'))
                         for r in results if r.get('final_exploitability', float('inf')) != float('inf')]
            if arch_final:
                final_exploitabilities.extend(arch_final)
                arch_labels.extend([arch] * len(arch_final))

        if final_exploitabilities:
            df_plot = pd.DataFrame({
                'Architecture': arch_labels,
                'Exploitability': final_exploitabilities
            })

            sns.violinplot(data=df_plot, x='Architecture', y='Exploitability', ax=ax2)
            ax2.set_title(f'{game.replace("_", " ").title()} - Final Performance')
            ax2.set_ylabel('Exploitability (mBB/100)')

        plt.tight_layout()
        return fig

    def plot_architecture_comparison(
        self,
        experiment_data: List[Dict[str, Any]]
    ) -> plt.Figure:
        """
        Create architecture comparison plots.

        Args:
            experiment_data: List of experiment results

        Returns:
            matplotlib Figure
        """
        # Filter successful experiments
        successful_data = [r for r in experiment_data if r.get('status') == 'completed']

        if not successful_data:
            raise ValueError("No successful experiments found")

        # Analyze results
        analysis = analyze_experiment_results(successful_data)

        if "overall_comparison" not in analysis:
            raise ValueError("No analysis results available")

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Bar plot with confidence intervals
        groups = analysis["overall_comparison"]["groups"]
        bootstrap_cis = analysis["overall_comparison"]["bootstrap_cis"]

        means = []
        errors = []
        labels = []

        for group in groups:
            if group in bootstrap_cis:
                ci = bootstrap_cis[group]
                mean_val = ci["mean"]
                error = (ci["upper_95"] - ci["lower_95"]) / 2

                means.append(mean_val)
                errors.append(error)
                labels.append(group.replace("_", " ").title())

        bars = ax1.bar(labels, means, yerr=errors, capsize=5, alpha=0.7)
        ax1.set_ylabel('Exploitability (mBB/100)')
        ax1.set_title('Architecture Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # Add significance annotations
        comparisons = analysis["overall_comparison"]["pairwise_comparisons"]
        y_offset = max(means) * 0.1
        y_pos = max(means) + y_offset

        for comp in comparisons:
            if comp["significant"]:
                group1_idx = groups.index(comp["group1"])
                group2_idx = groups.index(comp["group2"])

                x1, x2 = group1_idx, group2_idx
                y = max(means[x1], means[x2]) + y_offset

                ax1.plot([x1, x2], [y, y], 'k-', linewidth=1)
                ax1.text((x1 + x2) / 2, y + y_offset/4, '**', ha='center', fontsize=12)

        # Plot 2: Effect size heatmap
        if comparisons:
            effect_matrix = np.zeros((len(groups), len(groups)))
            for comp in comparisons:
                i, j = groups.index(comp["group1"]), groups.index(comp["group2"])
                effect = comp["effect_size"]
                effect_matrix[i, j] = effect
                effect_matrix[j, i] = -effect  # Symmetric

            im = ax2.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax2.set_xticks(range(len(groups)))
            ax2.set_yticks(range(len(groups)))
            ax2.set_xticklabels([g.replace("_", " ").title() for g in groups], rotation=45)
            ax2.set_yticklabels([g.replace("_", " ").title() for g in groups])
            ax2.set_title("Effect Sizes (Hedges' g)")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("Effect Size")

        # Plot 3: Wall-clock time comparison
        wall_times = {}
        for group in groups:
            group_results = [r for r in successful_data if r.get('architecture') == group]
            times = [r.get('wall_clock_s', 0) for r in group_results if r.get('wall_clock_s', 0) > 0]
            if times:
                wall_times[group] = times

        if wall_times:
            ax3.boxplot([wall_times[g] for g in groups], labels=[g.replace("_", " ").title() for g in groups])
            ax3.set_ylabel('Wall-clock Time (seconds)')
            ax3.set_title('Computational Efficiency')
            ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Sample sizes
        n_per_group = analysis["overall_comparison"]["n_per_group"]
        ax4.bar(labels, [n_per_group[g] for g in groups], alpha=0.7)
        ax4.set_ylabel('Number of Experiments')
        ax4.set_title('Sample Sizes')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def create_statistics_table(
        self,
        experiment_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a comprehensive statistics table.

        Args:
            experiment_data: List of experiment results

        Returns:
            pandas DataFrame with statistics
        """
        # Filter successful experiments
        successful_data = [r for r in experiment_data if r.get('status') == 'completed']

        if not successful_data:
            raise ValueError("No successful experiments found")

        # Analyze results
        analysis = analyze_experiment_results(successful_data)

        # Create summary table
        summary_table = create_summary_table(analysis)

        return summary_table

    def generate_all_figures(
        self,
        experiment_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Generate all figures and save them.

        Args:
            experiment_data: List of experiment results (optional, will load if not provided)

        Returns:
            Dictionary mapping figure names to file paths
        """
        if experiment_data is None:
            experiment_data = self.load_experiment_data()

        if not experiment_data:
            raise ValueError("No experiment data available")

        figure_files = {}

        # Get unique games
        games = list(set(r.get('game') for r in experiment_data if r.get('game')))

        # Generate game-specific exploitability curves
        for game in games:
            try:
                fig = self.plot_exploitability_curves(experiment_data, game)
                filename = f"exploitability_curves_{game}.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(exist_ok=True)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                figure_files[f"exploitability_curves_{game}"] = str(filepath)
                plt.close(fig)
                print(f"Generated: {filename}")
            except Exception as e:
                print(f"Error generating exploitability curves for {game}: {e}")

        # Generate architecture comparison
        try:
            fig = self.plot_architecture_comparison(experiment_data)
            filename = "architecture_comparison.png"
            filepath = self.output_dir / "plots" / filename
            filepath.parent.mkdir(exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            figure_files["architecture_comparison"] = str(filepath)
            plt.close(fig)
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating architecture comparison: {e}")

        # Generate statistics table
        try:
            table = self.create_statistics_table(experiment_data)
            filename = "results_table.tex"
            filepath = self.output_dir / "tables" / filename
            filepath.parent.mkdir(exist_ok=True)

            # Convert to LaTeX table
            latex_table = table.to_latex(
                index=False,
                escape=False,
                caption="Architecture Performance Summary",
                label="tab:architecture_summary"
            )

            with open(filepath, 'w') as f:
                f.write(latex_table)

            # Also save as CSV
            csv_filename = "results_table.csv"
            csv_filepath = self.output_dir / "tables" / csv_filename
            table.to_csv(csv_filepath, index=False)

            figure_files["results_table"] = str(filepath)
            print(f"Generated: {filename} and {csv_filename}")
        except Exception as e:
            print(f"Error generating statistics table: {e}")

        # Generate statistical report
        try:
            analysis = analyze_experiment_results(
                [r for r in experiment_data if r.get('status') == 'completed']
            )
            report = generate_statistical_report(analysis)

            filename = "statistical_report.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                f.write(report)

            figure_files["statistical_report"] = str(filepath)
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating statistical report: {e}")

        return figure_files


if __name__ == "__main__":
    # Example usage
    generator = FigureGenerator()

    # This would normally load real experiment data
    print("Figure generator ready. Run with actual experiment data to generate figures.")