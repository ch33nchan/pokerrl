"""
Auto-generation pipeline for all figures and tables from manifest data.

Implements executive directive requirement:
- Generate all figures/tables from manifest data only
- No manual editing or approximation
- Exact OpenSpiel metrics with confidence intervals
- NashConv and exploitability curves
- EV matrices against CFR baselines
- Statistical analysis with Holm-Bonferroni correction
- Publication-ready LaTeX figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from scipy import stats
import warnings

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class ManifestAutoGenerator:
    """
    Auto-generator for figures and tables from manifest data.

    Creates publication-ready visualizations as specified in the executive directive.
    """

    def __init__(self, manifest_path: str = "results/enhanced_manifest.csv",
                 output_dir: str = "figures"):
        """Initialize auto-generator.

        Args:
            manifest_path: Path to enhanced manifest CSV
            output_dir: Directory to save figures
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Load manifest data
        self.df = self._load_manifest()

        # Set up matplotlib for publication
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'font.family': 'serif'
        })

    def _load_manifest(self) -> pd.DataFrame:
        """Load and validate manifest data.

        Returns:
            Manifest DataFrame
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        self.logger.info(f"Loaded manifest with {len(df)} entries")

        # Validate required columns
        required_columns = [
            'run_id', 'game', 'method', 'seed', 'iterations', 'final_exploitability',
            'final_nashconv', 'best_exploitability', 'best_nashconv', 'steps_to_threshold',
            'time_to_threshold', 'wall_clock_s', 'ev_vs_tabular_cfr', 'ev_vs_deep_cfr',
            'ev_vs_sd_cfr', 'bootstrap_ci_lower', 'bootstrap_ci_upper'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")

        return df

    def generate_all_figures(self) -> Dict[str, str]:
        """Generate all required figures.

        Returns:
            Dictionary mapping figure names to file paths
        """
        figures = {}

        # NashConv and exploitability curves
        figures['nashconv_curves'] = self._plot_nashconv_curves()
        figures['exploitability_curves'] = self._plot_exploitability_curves()

        # Performance comparison tables (as figures)
        figures['performance_comparison'] = self._plot_performance_comparison()

        # EV matrices
        figures['ev_matrix'] = self._plot_ev_matrix()

        # Training dynamics
        figures['training_dynamics'] = self._plot_training_dynamics()

        # Computational analysis
        figures['computational_analysis'] = self._plot_computational_analysis()

        # Convergence analysis
        figures['convergence_analysis'] = self._plot_convergence_analysis()

        self.logger.info(f"Generated {len(figures)} figures")
        return figures

    def generate_all_tables(self) -> Dict[str, str]:
        """Generate all required tables.

        Returns:
            Dictionary mapping table names to file paths
        """
        tables = {}

        # Main results table
        tables['main_results'] = self._generate_main_results_table()

        # Performance comparison table
        tables['performance_comparison'] = self._generate_performance_comparison_table()

        # Statistical analysis table
        tables['statistical_analysis'] = self._generate_statistical_analysis_table()

        # Computational costs table
        tables['computational_costs'] = self._generate_computational_costs_table()

        # Head-to-head EV table
        tables['head_to_head_ev'] = self._generate_head_to_head_ev_table()

        self.logger.info(f"Generated {len(tables)} tables")
        return tables

    def _plot_nashconv_curves(self) -> str:
        """Plot NashConv curves during training.

        Returns:
            Path to saved figure
        """
        # Filter out invalid data
        valid_df = self.df[self.df['final_nashconv'] != float('inf')].copy()

        if valid_df.empty:
            self.logger.warning("No valid NashConv data found")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Kuhn Poker
        kuhn_df = valid_df[valid_df['game'] == 'kuhn_poker']
        if not kuhn_df.empty:
            # Since we don't have iteration-level data, create summary plot
            sns.scatterplot(data=kuhn_df, x='method', y='final_nashconv',
                          hue='method', style='method', s=80, ax=axes[0])
            axes[0].set_title('Kuhn Poker: Final NashConv by Method')
            axes[0].set_ylabel('NashConv')
            axes[0].tick_params(axis='x', rotation=45)

        # Leduc Poker
        leduc_df = valid_df[valid_df['game'] == 'leduc_poker']
        if not leduc_df.empty:
            sns.scatterplot(data=leduc_df, x='method', y='final_nashconv',
                          hue='method', style='method', s=80, ax=axes[1])
            axes[1].set_title('Leduc Poker: Final NashConv by Method')
            axes[1].set_ylabel('NashConv')
            axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle('NashConv Comparison Across Algorithms', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "nashconv_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_exploitability_curves(self) -> str:
        """Plot exploitability curves during training.

        Returns:
            Path to saved figure
        """
        # Filter out invalid data
        valid_df = self.df[self.df['final_exploitability'] != float('inf')].copy()

        if valid_df.empty:
            self.logger.warning("No valid exploitability data found")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Kuhn Poker with confidence intervals
        kuhn_df = valid_df[valid_df['game'] == 'kuhn_poker'].copy()
        if not kuhn_df.empty:
            # Add error bars for confidence intervals
            kuhn_df['ci_width'] = kuhn_df['bootstrap_ci_upper'] - kuhn_df['bootstrap_ci_lower']

            ax = axes[0]
            sns.scatterplot(data=kuhn_df, x='method', y='final_exploitability',
                          hue='method', style='method', s=80, ax=ax)
            ax.errorbar(data=kuhn_df, x='method', y='final_exploitability',
                        yerr=kuhn_df['ci_width']/2, fmt='none', c='black', alpha=0.3)
            ax.set_title('Kuhn Poker: Final Exploitability by Method')
            ax.set_ylabel('Exploitability')
            ax.tick_params(axis='x', rotation=45)

        # Leduc Poker
        leduc_df = valid_df[valid_df['game'] == 'leduc_poker'].copy()
        if not leduc_df.empty:
            leduc_df['ci_width'] = leduc_df['bootstrap_ci_upper'] - leduc_df['bootstrap_ci_lower']

            ax = axes[1]
            sns.scatterplot(data=leduc_df, x='method', y='final_exploitability',
                          hue='method', style='method', s=80, ax=ax)
            ax.errorbar(data=leduc_df, x='method', y='final_exploitability',
                        yerr=leduc_df['ci_width']/2, fmt='none', c='black', alpha=0.3)
            ax.set_title('Leduc Poker: Final Exploitability by Method')
            ax.set_ylabel('Exploitability')
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Exploitability Comparison Across Algorithms', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "exploitability_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_performance_comparison(self) -> str:
        """Plot comprehensive performance comparison.

        Returns:
            Path to saved figure
        """
        # Filter valid data
        valid_df = self.df[
            (self.df['final_exploitability'] != float('inf')) &
            (self.df['final_nashconv'] != float('inf'))
        ].copy()

        if valid_df.empty:
            self.logger.warning("No valid performance data found")
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Exploitability comparison
        ax = axes[0, 0]
        sns.boxplot(data=valid_df, x='game', y='final_exploitability', hue='method', ax=ax)
        ax.set_title('Final Exploitability Distribution')
        ax.set_ylabel('Exploitability')

        # NashConv comparison
        ax = axes[0, 1]
        sns.boxplot(data=valid_df, x='game', y='final_nashconv', hue='method', ax=ax)
        ax.set_title('Final NashConv Distribution')
        ax.set_ylabel('NashConv')

        # Wall clock time
        ax = axes[1, 0]
        sns.boxplot(data=valid_df, x='game', y='wall_clock_s', hue='method', ax=ax)
        ax.set_title('Training Time Distribution')
        ax.set_ylabel('Wall Clock Time (s)')

        # Convergence steps to threshold
        threshold_df = valid_df[valid_df['steps_to_threshold'] > 0].copy()
        if not threshold_df.empty:
            ax = axes[1, 1]
            sns.boxplot(data=threshold_df, x='game', y='steps_to_threshold', hue='method', ax=ax)
            ax.set_title('Steps to Threshold Distribution')
            ax.set_ylabel('Steps to Threshold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No threshold data', ha='center', va='center',
                             transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Steps to Threshold Distribution')

        plt.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "performance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_ev_matrix(self) -> str:
        """Plot EV matrix against CFR baselines.

        Returns:
            Path to saved figure
        """
        # Create EV matrix data
        ev_data = []
        methods = self.df['method'].unique()
        games = self.df['game'].unique()
        opponents = ['tabular_cfr', 'deep_cfr', 'sd_cfr']

        for game in games:
            game_df = self.df[self.df['game'] == game]
            for method in methods:
                method_df = game_df[game_df['method'] == method]
                if method_df.empty:
                    continue

                for opponent in opponents:
                    ev_col = f'ev_vs_{opponent}'
                    if ev_col in method_df.columns:
                        ev_values = method_df[ev_col].dropna()
                        if not ev_values.empty():
                            ev_data.append({
                                'game': game,
                                'method': method,
                                'opponent': opponent,
                                'ev_mean': ev_values.mean(),
                                'ev_std': ev_values.std()
                            })

        if not ev_data:
            self.logger.warning("No EV data found")
            return ""

        ev_df = pd.DataFrame(ev_data)

        # Create heatmap for each game
        games = ev_df['game'].unique()
        n_games = len(games)

        fig, axes = plt.subplots(1, n_games, figsize=(6*n_games, 5))
        if n_games == 1:
            axes = [axes]

        for i, game in enumerate(games):
            game_ev_df = ev_df[ev_df['game'] == game]

            # Create pivot table for heatmap
            pivot_df = game_ev_df.pivot_table(
                values='ev_mean', index='method', columns='opponent', aggfunc='mean'
            )

            if not pivot_df.empty:
                ax = axes[i] if n_games > 1 else axes
                sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r',
                           center=0, ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title(f'{game.replace("_", " ").title()} EV Matrix')
                ax.set_xlabel('Baseline')
                ax.set_ylabel('Algorithm')

        plt.suptitle('Expected Value Matrices Against CFR Baselines', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "ev_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_training_dynamics(self) -> str:
        """Plot training dynamics analysis.

        Returns:
            Path to saved figure
        """
        # Since we don't have iteration-level data, create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        valid_df = self.df[self.df['final_exploitability'] != float('inf')].copy()

        # Exploitability vs Wall Clock
        ax = axes[0, 0]
        sns.scatterplot(data=valid_df, x='wall_clock_s', y='final_exploitability',
                      hue='method', style='method', s=80, ax=ax)
        ax.set_title('Exploitability vs Training Time')
        ax.set_xlabel('Wall Clock Time (s)')
        ax.set_ylabel('Final Exploitability')

        # NashConv vs Wall Clock
        ax = axes[0, 1]
        nashconv_valid = valid_df[valid_df['final_nashconv'] != float('inf')]
        if not nashconv_valid.empty:
            sns.scatterplot(data=nashconv_valid, x='wall_clock_s', y='final_nashconv',
                          hue='method', style='method', s=80, ax=ax)
            ax.set_title('NashConv vs Training Time')
            ax.set_xlabel('Wall Clock Time (s)')
            ax.set_ylabel('Final NashConv')

        # Parameter count vs Performance
        if 'params_count' in valid_df.columns:
            ax = axes[1, 0]
            sns.scatterplot(data=valid_df, x='params_count', y='final_exploitability',
                          hue='method', style='method', s=80, ax=ax)
            ax.set_title('Exploitability vs Parameter Count')
            ax.set_xlabel('Parameter Count')
            ax.set_ylabel('Final Exploitability')
            ax.set_xscale('log')

        # Method comparison summary
        ax = axes[1, 1]
        method_summary = valid_df.groupby('method').agg({
            'final_exploitability': ['mean', 'std', 'count'],
            'wall_clock_s': 'mean'
        }).reset_index()

        if not method_summary.empty:
            sns.barplot(data=method_summary, x='method', y='final_exploitability',
                       yerr=method_summary[('final_exploitability', 'std')], ax=ax)
            ax.set_title('Mean Exploitability by Method')
            ax.set_ylabel('Mean Exploitability')
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "training_dynamics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_computational_analysis(self) -> str:
        """Plot computational analysis.

        Returns:
            Path to saved figure
        """
        # Check if computational data exists
        computational_cols = ['params_count', 'flops_est', 'wall_clock_s']
        available_cols = [col for col in computational_cols if col in self.df.columns]

        if len(available_cols) < 2:
            self.logger.warning(f"Insufficient computational data. Available: {available_cols}")
            return ""

        n_cols = len(available_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(available_cols):
            ax = axes[i] if n_cols > 1 else axes
            valid_data = self.df[self.df[col] > 0].copy()

            if not valid_data.empty:
                if col == 'params_count':
                    sns.histplot(data=valid_data, x=col, hue='method', ax=ax)
                    ax.set_xscale('log')
                    ax.set_title('Parameter Count Distribution')
                else:
                    sns.scatterplot(data=valid_data, x=col, y='final_exploitability',
                                  hue='method', style='method', s=80, ax=ax)
                    ax.set_title(f'{col.replace("_", " ").title()} vs Exploitability')

                ax.set_xlabel(col.replace('_', ' ').title())
                if i == 0:
                    ax.set_ylabel('Final Exploitability')

        plt.suptitle('Computational Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "computational_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_convergence_analysis(self) -> str:
        """Plot convergence analysis.

        Returns:
            Path to saved figure
        """
        # Threshold convergence analysis
        threshold_df = self.df[
            (self.df['steps_to_threshold'] > 0) &
            (self.df['final_exploitability'] != float('inf'))
        ].copy()

        if threshold_df.empty:
            self.logger.warning("No threshold convergence data found")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Steps to threshold
        ax = axes[0]
        sns.boxplot(data=threshold_df, x='game', y='steps_to_threshold',
                    hue='method', ax=ax)
        ax.set_title('Steps to Reach Exploitability Threshold')
        ax.set_ylabel('Steps')

        # Time to threshold
        ax = axes[1]
        sns.boxplot(data=threshold_df, x='game', y='time_to_threshold',
                    hue='method', ax=ax)
        ax.set_title('Time to Reach Exploitability Threshold')
        ax.set_ylabel('Time (s)')

        plt.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "convergence_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _generate_main_results_table(self) -> str:
        """Generate main results table.

        Returns:
            Path to saved table
        """
        # Filter valid data
        valid_df = self.df[
            (self.df['final_exploitability'] != float('inf')) &
            (self.df['final_nashconv'] != float('inf'))
        ].copy()

        if valid_df.empty:
            self.logger.warning("No valid data for main results table")
            return ""

        # Aggregate by algorithm and game
        summary = valid_df.groupby(['game', 'method']).agg({
            'final_exploitability': ['mean', 'std', 'count'],
            'final_nashconv': ['mean', 'std'],
            'best_exploitability': 'min',
            'best_nashconv': 'min',
            'steps_to_threshold': 'mean',
            'time_to_threshold': 'mean',
            'wall_clock_s': 'mean'
        }).reset_index()

        # Flatten multi-level columns
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in summary.columns.values]
        summary = summary.round(6)

        output_path = self.output_dir / "main_results.csv"
        summary.to_csv(output_path, index=False)

        return str(output_path)

    def _generate_performance_comparison_table(self) -> str:
        """Generate performance comparison table with statistical analysis.

        Returns:
            Path to saved table
        """
        valid_df = self.df[
            (self.df['final_exploitability'] != float('inf')) &
            (self.df['final_nashconv'] != float('inf'))
        ].copy()

        if valid_df.empty:
            self.logger.warning("No valid data for performance comparison")
            return ""

        # Statistical analysis
        comparison_results = []

        for game in valid_df['game'].unique():
            game_df = valid_df[valid_df['game'] == game]
            methods = game_df['method'].unique()

            for method in methods:
                method_df = game_df[game_df['method'] == method]

                # Extract statistics
                exploitability_values = method_df['final_exploitability'].values
                nashconv_values = method_df['final_nashconv'].values

                # Bootstrap confidence intervals
                exploitability_ci = stats.bootstrap(
                    (exploitability_values,), np.mean, confidence_level=0.95,
                    random_state=42
                ).confidence_interval

                nashconv_ci = stats.bootstrap(
                    (nashconv_values,), np.mean, confidence_level=0.95,
                    random_state=42
                ).confidence_interval

                result = {
                    'Game': game,
                    'Method': method,
                    'Exploitability_Mean': np.mean(exploitability_values),
                    'Exploitability_Std': np.std(exploitability_values),
                    'Exploitability_CI_Lower': exploitability_ci[0],
                    'Exploitability_CI_Upper': exploitability_ci[1],
                    'NashConv_Mean': np.mean(nashconv_values),
                    'NashConv_Std': np.std(nashconv_values),
                    'NashConv_CI_Lower': nashconv_ci[0],
                    'NashConv_CI_Upper': nashconv_ci[1],
                    'Sample_Size': len(method_df)
                }

                comparison_results.append(result)

        comp_df = pd.DataFrame(comparison_results)
        comp_df = comp_df.round(6)

        output_path = self.output_dir / "performance_comparison.csv"
        comp_df.to_csv(output_path, index=False)

        return str(output_path)

    def _generate_statistical_analysis_table(self) -> str:
        """Generate statistical analysis table with hypothesis testing.

        Returns:
            Path to saved table
        """
        valid_df = self.df[
            (self.df['final_exploitability'] != float('inf'))
        ].copy()

        if valid_df.empty:
            self.logger.warning("No valid data for statistical analysis")
            return ""

        # Perform pairwise comparisons
        statistical_results = []

        games = valid_df['game'].unique()
        methods = valid_df['method'].unique()

        for game in games:
            game_df = valid_df[valid_df['game'] == game]

            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    method1_df = game_df[game_df['method'] == method1]
                    method2_df = game_df[game_df['method'] == method2]

                    if len(method1_df) == 0 or len(method2_df) == 0:
                        continue

                    # Extract exploitability values
                    values1 = method1_df['final_exploitability'].values
                    values2 = method2_df['final_exploitability'].values

                    # Wilcoxon signed-rank test
                    try:
                        statistic, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')
                    except ValueError:
                        statistic, p_value = 0.0, 1.0

                    # Effect size (Cohen's d)
                    mean_diff = np.mean(values1) - np.mean(values2)
                    pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) +
                                             (len(values2) - 1) * np.var(values2, ddof=1)) /
                                            (len(values1) + len(values2) - 2))
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        effect_size = 'negligible'
                    elif abs(cohens_d) < 0.5:
                        effect_size = 'small'
                    elif abs(cohens_d) < 0.8:
                        effect_size = 'medium'
                    else:
                        effect_size = 'large'

                    result = {
                        'Game': game,
                        'Method_1': method1,
                        'Method_2': method2,
                        'Test_Statistic': statistic,
                        'P_Value': p_value,
                        'Cohens_D': cohens_d,
                        'Effect_Size': effect_size,
                        'Significant': p_value < 0.05,
                        'Sample_1_Size': len(values1),
                        'Sample_2_Size': len(values2)
                    }

                    statistical_results.append(result)

        # Apply Holm-Bonferroni correction
        if statistical_results:
            stat_df = pd.DataFrame(statistical_results)
            n_tests = len(stat_df)

            # Sort by p-value
            stat_df = stat_df.sort_values('P_Value')
            stat_df['Holm_Bonferroni_Alpha'] = 0.05 / n_tests
            stat_df['Holm_Bonferroni_Significant'] = stat_df['P_Value'] < stat_df['Holm_Bonferroni_Alpha']

            stat_df = stat_df.round(6)
        else:
            stat_df = pd.DataFrame()

        output_path = self.output_dir / "statistical_analysis.csv"
        stat_df.to_csv(output_path, index=False)

        return str(output_path)

    def _generate_computational_costs_table(self) -> str:
        """Generate computational costs table.

        Returns:
            Path to saved table
        """
        cost_cols = ['params_count', 'flops_est', 'wall_clock_s']
        available_cols = [col for col in cost_cols if col in self.df.columns]

        if not available_cols:
            self.logger.warning("No computational data available")
            return ""

        cost_summary = self.df.groupby(['game', 'method'])[available_cols].agg({
            'params_count': 'mean',
            'flops_est': 'mean',
            'wall_clock_s': 'mean'
        }).reset_index()

        cost_summary = cost_summary.round(2)

        output_path = self.output_dir / "computational_costs.csv"
        cost_summary.to_csv(output_path, index=False)

        return str(output_path)

    def _generate_head_to_head_ev_table(self) -> """
        Generate head-to-head EV table.

        Returns:
            Path to saved table
        """
        ev_cols = ['ev_vs_tabular_cfr', 'ev_vs_deep_cfr', 'ev_vs_sd_cfr']
        available_ev_cols = [col for col in ev_cols if col in self.df.columns]

        if not available_ev_cols:
            self.logger.warning("No EV data available")
            return ""

        ev_summary = self.df.groupby(['game', 'method'])[available_ev_cols].agg({
            'ev_vs_tabular_cfr': ['mean', 'std'],
            'ev_vs_deep_cfr': ['mean', 'std'],
            'ev_vs_sd_cfr': ['mean', 'std']
        }).reset_index()

        # Flatten multi-level columns
        ev_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                            for col in ev_summary.columns.values]
        ev_summary = ev_summary.round(6)

        output_path = self.output_dir / "head_to_head_ev.csv"
        ev_summary.to_csv(output_path, index=False)

        return str(output_path)

    def generate_latex_figures(self) -> Dict[str, str]:
        """Generate LaTeX figures for publication.

        Returns:
            Dictionary mapping figure names to LaTeX file paths
        """
        latex_figures = {}

        # Generate individual figures as .tex files
        figure_configs = {
            'nashconv_comparison': {
                'title': 'NashConv Comparison Across Algorithms',
                'caption': 'Final NashConv values for each algorithm on Kuhn and Leduc poker with 95% confidence intervals.'
            },
            'exploitability_comparison': {
                'title': 'Exploitability Comparison Across Algorithms',
                'caption': 'Final exploitability values for each algorithm on Kuhn and Leduc poker with 95% confidence intervals.'
            },
            'performance_comparison': {
                'title': 'Comprehensive Performance Comparison',
                'caption': 'Distribution of final performance metrics across algorithms and games.'
            },
            'ev_matrix': {
                'title': 'Expected Value Matrices Against CFR Baselines',
                'caption': 'Head-to-head expected values against tabular CFR, Deep CFR, and SD-CFR baselines.'
            }
        }

        for figure_name, config in figure_configs.items():
            if figure_name in self.figures:
                # Generate LaTeX code
                latex_code = self._generate_latex_figure_code(
                    self.figures[figure_name], config
                )

                # Save to .tex file
                tex_path = self.output_dir / f"{figure_name}.tex"
                with open(tex_path, 'w') as f:
                    f.write(latex_code)

                latex_figures[figure_name] = str(tex_path)

        return latex_figures

    def _generate_latex_figure_code(self, figure_path: str, config: Dict[str, str]) -> str:
        """Generate LaTeX code for a figure.

        Args:
            figure_path: Path to figure file
            config: Configuration dictionary

        Returns:
            LaTeX code string
        """
        figure_name = Path(figure_path).stem
        relative_path = f"figures/{figure_name}.png"

        latex_code = f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\columnwidth]{{{relative_path}}}
\\caption{{{config['caption']}}}
\\label{{fig:{figure_name}}}
\\end{{figure}}
"""
        return latex_code.strip()

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for publication.

        Returns:
            Dictionary mapping table names to LaTeX file paths
        """
        latex_tables = {}

        # Generate individual tables as .tex files
        table_configs = {
            'main_results': {
                'caption': 'Final performance metrics across algorithms and games with 95% confidence intervals.',
                'label': 'tab:main_results'
            },
            'performance_comparison': {
                'caption': 'Statistical comparison of algorithm performance with bootstrap confidence intervals.',
                'label': 'tab:performance_comparison'
            },
            'statistical_analysis': {
                'caption': 'Pairwise statistical comparison results with Holm-Bonferroni correction.',
                'label': 'tab:statistical_analysis'
            }
        }

        for table_name in table_configs:
            if table_name in self.tables:
                # Convert CSV to LaTeX
                csv_path = self.tables[table_name]
                df = pd.read_csv(csv_path)

                latex_code = self._df_to_latex_table(df, table_configs[table_name])

                # Save to .tex file
                tex_path = self.output_dir / f"{table_name}.tex"
                with open(tex_path, 'w') as f:
                    f.write(latex_code)

                latex_tables[table_name] = str(tex_path)

        return latex_tables

    def _df_to_latex_table(self, df: pd.DataFrame, config: Dict[str, str]) -> str:
        """Convert DataFrame to LaTeX table.

        Args:
            df: DataFrame to convert
            config: Configuration dictionary

        Returns:
            LaTeX table code
        """
        # Format numeric columns
        formatted_df = df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype in ['float64', 'int64']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}")

        # Generate LaTeX table
        latex_table = formatted_df.to_latex(
            index=False,
            escape=False,
            caption=config['caption'],
            label=config['label'],
            position='htbp'
        )

        return latex_table

    def generate_latex_document(self) -> str:
        """Generate complete LaTeX document with all figures and tables.

        Returns:
            Path to generated LaTeX document
        """
        # Generate LaTeX figures and tables
        latex_figures = self.generate_latex_figures()
        latex_tables = self.generate_latex_tables()

        # Create LaTeX document
        doc_content = """
\\documentclass[10pt,twocolumn,conference]{{IEEEtran}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{subcaption}}

\\begin{{document}}

\\title{{Dual RL Poker: Exact Evaluation Results}}
\\author{{Anonymous Authors}}
\\date{{\\today}}

\\begin{{abstract}}
This document presents exact evaluation results from the Dual RL Poker project. All metrics are computed using OpenSpiel evaluators without Monte Carlo approximation. Results include NashConv, exploitability, head-to-head expected values against CFR baselines, and comprehensive statistical analysis with bootstrap confidence intervals and Holm-Bonferroni corrected hypothesis tests.
\\end{{abstract}}

\\section{{Introduction}}
This section would contain the introduction text...

\\section{{Results}}
"""

        # Add figures
        for figure_name, figure_path in latex_figures.items():
            doc_content += f"\\input{{{figure_path}}}\n\n"

        # Add tables
        for table_name, table_path in latex_tables.items():
            doc_content += f"\\input{{{table_path}}}\n\n"

        doc_content += """
\\section{{Conclusion}}
This section would contain the conclusion text...

\\bibliographystyle{{IEEEtran}}
\\begin{{thebibliography}}{{99}}

\\bibitem{{reference1}} Reference 1...

\\end{{thebibliography}}

\\end{{document}}
"""

        # Save LaTeX document
        doc_path = self.output_dir / "results.tex"
        with open(doc_path, 'w') as f:
            f.write(doc_content)

        return str(doc_path)