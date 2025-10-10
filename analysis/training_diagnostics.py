"""
Gradient and loss stability tracking for Deep CFR architectures.
Generates diagnostic plots to supplement the main results.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TrainingDiagnostics:
    """Analyzes training diagnostics for Deep CFR experiments."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'diagnostic_plots'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experimental results
        self.results_df = self._load_results()
        
    def _load_results(self) -> pd.DataFrame:
        """Load experimental results from CSV file."""
        results_file = self.results_dir / 'comprehensive_results.csv'
        if results_file.exists():
            return pd.read_csv(results_file)
        else:
            # Load from individual JSON files if CSV doesn't exist
            result_files = list(self.results_dir.glob('*_result.json'))
            results = []
            
            for file in result_files:
                with open(file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            
            return pd.DataFrame(results)
    
    def generate_gradient_stability_plots(self):
        """Generate gradient norm analysis plots."""
        # Filter successful experiments with gradient data
        successful_df = self.results_df[
            (self.results_df['success'] == True) & 
            (self.results_df['gradient_norms'].notna())
        ]
        
        if len(successful_df) == 0:
            print("No successful experiments with gradient data found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gradient Stability Analysis Across Architectures', fontsize=16)
        
        # Plot 1: Gradient norm distributions by architecture
        gradient_data = []
        for _, row in successful_df.iterrows():
            if isinstance(row['gradient_norms'], str):
                try:
                    grad_norms = json.loads(row['gradient_norms'])
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON
            else:
                grad_norms = row['gradient_norms']
            
            # Skip empty gradient data
            if not grad_norms:
                continue
            
            for player, norms in grad_norms.items():
                if norms:  # Only if we have gradient data
                    for norm in norms[-100:]:  # Last 100 updates
                        gradient_data.append({
                            'architecture': row['architecture'],
                            'game': row['game'],
                            'gradient_norm': norm,
                            'player': player
                        })
        
        if gradient_data:
            grad_df = pd.DataFrame(gradient_data)
            
            # Box plot of gradient norms by architecture
            sns.boxplot(data=grad_df, x='architecture', y='gradient_norm', ax=axes[0, 0])
            axes[0, 0].set_title('Gradient Norm Distributions')
            axes[0, 0].set_ylabel('Gradient Norm')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Violin plot for more detailed distribution
            sns.violinplot(data=grad_df, x='architecture', y='gradient_norm', ax=axes[0, 1])
            axes[0, 1].set_title('Gradient Norm Density Distributions')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            # No gradient data available
            axes[0, 0].text(0.5, 0.5, 'No gradient data available\n(gradient tracking not implemented)', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Gradient Norm Distributions')
            axes[0, 1].text(0.5, 0.5, 'No gradient data available\n(gradient tracking not implemented)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Gradient Norm Density Distributions')
        
        # Plot 2: Gradient explosion events for Transformer
        transformer_df = successful_df[successful_df['architecture'] == 'transformer']
        explosion_data = []
        
        for _, row in transformer_df.iterrows():
            # Check for gradient explosion events
            if 'gradient_explosion_events' in row and row['gradient_explosion_events']:
                if isinstance(row['gradient_explosion_events'], str):
                    events = json.loads(row['gradient_explosion_events'])
                else:
                    events = row['gradient_explosion_events']
                
                explosion_data.append({
                    'experiment_id': row['experiment_id'],
                    'num_explosions': len(events),
                    'game': row['game']
                })
        
        if explosion_data:
            explosion_df = pd.DataFrame(explosion_data)
            explosion_df.plot(x='experiment_id', y='num_explosions', kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Gradient Explosion Events (Transformer)')
            axes[1, 0].set_ylabel('Number of Explosions')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Transformer gradient explosion data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Gradient Explosion Events (Transformer)')
        
        # Plot 3: Training loss variance
        loss_variance_data = []
        for _, row in successful_df.iterrows():
            if isinstance(row['loss_trajectory'], str):
                try:
                    losses = json.loads(row['loss_trajectory'])
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON
            else:
                losses = row['loss_trajectory']
            
            # Skip empty loss data
            if not losses:
                continue
            
            for player, loss_history in losses.items():
                if loss_history and len(loss_history) > 10:
                    variance = np.var(loss_history[-100:])  # Variance of last 100 losses
                    loss_variance_data.append({
                        'architecture': row['architecture'],
                        'game': row['game'],
                        'loss_variance': variance,
                        'player': player
                    })
        
        if loss_variance_data:
            loss_var_df = pd.DataFrame(loss_variance_data)
            sns.boxplot(data=loss_var_df, x='architecture', y='loss_variance', ax=axes[1, 1])
            axes[1, 1].set_title('Training Loss Variance')
            axes[1, 1].set_ylabel('Loss Variance')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No loss trajectory data available\n(loss tracking not implemented)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Loss Variance')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gradient_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gradient stability plots saved to {self.output_dir}/gradient_stability_analysis.png")
    
    def generate_training_curves_with_variance(self):
        """Generate training curves with confidence intervals."""
        # Group experiments by architecture and game
        grouped = self.results_df.groupby(['architecture', 'game'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Progress with Variance Bands', fontsize=16)
        axes = axes.flatten()
        
        for idx, (game, game_data) in enumerate(self.results_df.groupby('game')):
            if idx >= 2:
                break
                
            ax = axes[idx]
            architectures = list(game_data['architecture'].unique())
            palette = sns.color_palette("tab10", n_colors=max(len(architectures), 1))
            arch_colors = {arch: palette[i % len(palette)] for i, arch in enumerate(architectures)}
            
            for arch, arch_data in game_data.groupby('architecture'):
                # Extract exploitability trajectories
                exploitability_series = []
                
                for _, row in arch_data.iterrows():
                    if row['success'] and 'exploitability_trajectory' in row:
                        if isinstance(row['exploitability_trajectory'], str):
                            trajectory = json.loads(row['exploitability_trajectory'])
                        else:
                            trajectory = row['exploitability_trajectory']
                        
                        if trajectory:
                            # Convert to series indexed by iteration
                            series = pd.Series(
                                [point[1] for point in trajectory],
                                index=[point[0] for point in trajectory]
                            )
                            exploitability_series.append(series)
                
                if exploitability_series:
                    # Align series and compute statistics
                    df_aligned = pd.DataFrame(exploitability_series).T
                    mean_trajectory = df_aligned.mean(axis=1)
                    std_trajectory = df_aligned.std(axis=1)
                    
                    # Plot mean with confidence band
                    iterations = mean_trajectory.index
                    ax.plot(iterations, mean_trajectory, 
                           color=arch_colors[arch], label=f'{arch} (n={len(exploitability_series)})',
                           linewidth=2)
                    
                    # Add 95% confidence band
                    ax.fill_between(iterations, 
                                   mean_trajectory - 1.96 * std_trajectory / np.sqrt(len(exploitability_series)),
                                   mean_trajectory + 1.96 * std_trajectory / np.sqrt(len(exploitability_series)),
                                   color=arch_colors[arch], alpha=0.2)
            
            ax.set_title(f'{game.title()} - Exploitability Over Time')
            ax.set_xlabel('Training Iteration')
            ax.set_ylabel('Exploitability')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Success rate comparison
        success_rates = self.results_df.groupby(['architecture', 'game'])['success'].agg(['count', 'sum', 'mean'])
        success_rates['success_rate'] = success_rates['mean'] * 100
        
        ax = axes[2]
        success_pivot = success_rates.reset_index().pivot(index='architecture', columns='game', values='success_rate')
        success_pivot.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
        ax.set_title('Success Rates by Architecture and Game')
        ax.set_ylabel('Success Rate (%)')
        ax.legend(title='Game')
        ax.tick_params(axis='x', rotation=45)
        
        # Performance distribution for successful runs
        ax = axes[3]
        successful_results = self.results_df[self.results_df['success'] == True]
        if len(successful_results) > 0:
            # Check if we have successful results for each architecture
            architectures_with_data = successful_results['architecture'].value_counts()
            if len(architectures_with_data) > 0:
                try:
                    # Check if we have data to plot
                    if len(successful_results) > 0 and 'final_exploitability' in successful_results.columns:
                        # Remove any NaN or infinite values
                        clean_data = successful_results.dropna(subset=['final_exploitability'])
                        clean_data = clean_data[clean_data['final_exploitability'].notna()]
                        clean_data = clean_data[np.isfinite(clean_data['final_exploitability'])]
                        
                        if len(clean_data) > 0:
                            sns.boxplot(data=clean_data, x='architecture', y='final_exploitability', ax=ax)
                            ax.set_title('Final Performance Distribution (Successful Runs)')
                            ax.set_ylabel('Final Exploitability')
                            ax.tick_params(axis='x', rotation=45)
                            # Only set log scale if we have positive values
                            if clean_data['final_exploitability'].min() > 0:
                                ax.set_yscale('log')
                        else:
                            ax.text(0.5, 0.5, 'No valid performance data to plot', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title('Final Performance Distribution (Successful Runs)')
                    else:
                        ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Final Performance Distribution (Successful Runs)')
                except Exception as e:
                    print(f"Error creating performance plot: {e}")
                    ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Final Performance Distribution (Successful Runs)')
            else:
                ax.text(0.5, 0.5, 'No successful runs to plot', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Final Performance Distribution (Successful Runs)')
        else:
            ax.text(0.5, 0.5, 'No successful runs to plot', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final Performance Distribution (Successful Runs)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_with_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves with variance saved to {self.output_dir}/training_curves_with_variance.png")
    
    def generate_transformer_failure_analysis(self):
        """Detailed analysis of Transformer failure modes."""
        transformer_data = self.results_df[self.results_df['architecture'] == 'transformer']
        
        if len(transformer_data) == 0:
            print("No Transformer experiments found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Transformer Architecture Failure Analysis', fontsize=16)
        
        # Success rate by hyperparameters
        if 'hyperparams' in transformer_data.columns:
            # Extract hyperparameters if stored as JSON
            hyperparam_data = []
            for _, row in transformer_data.iterrows():
                if isinstance(row['hyperparams'], str):
                    try:
                        hyperparams = json.loads(row['hyperparams'])
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Warning: Could not parse hyperparams JSON: {e}")
                        # Try to extract key info from string representation
                        hyperparam_str = str(row['hyperparams'])
                        hyperparams = {
                            'learning_rate': 'unknown',
                            'hidden_size': 'unknown',
                            'batch_size': 'unknown',
                            'memory_size': 'unknown'
                        }
                        # Basic parsing for common patterns
                        import re
                        lr_match = re.search(r'lr([\d.]+)', hyperparam_str)
                        if lr_match:
                            hyperparams['learning_rate'] = float(lr_match.group(1))
                        h_match = re.search(r'h(\d+)', hyperparam_str)
                        if h_match:
                            hyperparams['hidden_size'] = int(h_match.group(1))
                else:
                    hyperparams = row['hyperparams']
                
                hyperparam_data.append({
                    'learning_rate': hyperparams.get('learning_rate', 'unknown'),
                    'hidden_size': hyperparams.get('hidden_size', 'unknown'),
                    'success': row['success']
                })
            
            if hyperparam_data:
                hp_df = pd.DataFrame(hyperparam_data)
                
                # Success by learning rate
                lr_success = hp_df.groupby('learning_rate')['success'].agg(['count', 'sum', 'mean'])
                lr_success['success_rate'] = lr_success['mean'] * 100
                lr_success['success_rate'].plot(kind='bar', ax=axes[0, 0])
                axes[0, 0].set_title('Transformer Success Rate by Learning Rate')
                axes[0, 0].set_ylabel('Success Rate (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Success by hidden size
                hs_success = hp_df.groupby('hidden_size')['success'].agg(['count', 'sum', 'mean'])
                hs_success['success_rate'] = hs_success['mean'] * 100
                hs_success['success_rate'].plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Transformer Success Rate by Hidden Size')
                axes[0, 1].set_ylabel('Success Rate (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time analysis
        if 'training_time' in transformer_data.columns:
            successful = transformer_data[transformer_data['success'] == True]['training_time']
            failed = transformer_data[transformer_data['success'] == False]['training_time']
            
            axes[0, 2].hist([successful, failed], bins=20, alpha=0.7, 
                           label=['Successful', 'Failed'], color=['green', 'red'])
            axes[0, 2].set_title('Training Time Distribution')
            axes[0, 2].set_xlabel('Training Time (seconds)')
            axes[0, 2].legend()
        
        # Error message analysis
        error_messages = transformer_data[transformer_data['success'] == False]['error_message'].value_counts()
        if len(error_messages) > 0:
            error_messages.head(5).plot(kind='barh', ax=axes[1, 0])
            axes[1, 0].set_title('Most Common Error Messages')
            axes[1, 0].set_xlabel('Frequency')
        
        # Gradient clipping effectiveness (if data available)
        grad_clip_data = []
        for _, row in transformer_data.iterrows():
            if 'gradient_explosion_events' in row and row['gradient_explosion_events']:
                if isinstance(row['gradient_explosion_events'], str):
                    events = json.loads(row['gradient_explosion_events'])
                else:
                    events = row['gradient_explosion_events']
                
                grad_clip_data.append({
                    'experiment_id': row['experiment_id'],
                    'explosions': len(events),
                    'success': row['success']
                })
        
        if grad_clip_data:
            grad_df = pd.DataFrame(grad_clip_data)
            success_by_explosions = grad_df.groupby('explosions')['success'].agg(['count', 'sum', 'mean'])
            success_by_explosions['success_rate'] = success_by_explosions['mean'] * 100
            
            axes[1, 1].scatter(success_by_explosions.index, success_by_explosions['success_rate'])
            axes[1, 1].set_title('Success Rate vs Gradient Explosions')
            axes[1, 1].set_xlabel('Number of Gradient Explosions')
            axes[1, 1].set_ylabel('Success Rate (%)')
        
        # Overall failure rate
        total_experiments = len(transformer_data)
        successful_experiments = len(transformer_data[transformer_data['success'] == True])
        failure_rate = (total_experiments - successful_experiments) / total_experiments * 100
        
        axes[1, 2].pie([successful_experiments, total_experiments - successful_experiments],
                      labels=['Successful', 'Failed'],
                      colors=['green', 'red'],
                      autopct='%1.1f%%')
        axes[1, 2].set_title(f'Transformer Overall Success Rate\n({failure_rate:.1f}% failure rate)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transformer_failure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Transformer failure analysis saved to {self.output_dir}/transformer_failure_analysis.png")
    
    def generate_comprehensive_report(self):
        """Generate all diagnostic plots and summary report."""
        print("Generating comprehensive diagnostic report...")
        
        # Generate all plots
        self.generate_gradient_stability_plots()
        self.generate_training_curves_with_variance()
        self.generate_individual_architecture_plots()
        self.generate_transformer_failure_analysis()
        
        # Generate summary statistics
        summary_stats = {
            'total_experiments': len(self.results_df),
            'overall_success_rate': self.results_df['success'].mean() * 100,
            'architecture_breakdown': self.results_df.groupby('architecture')['success'].agg([
                'count', 'sum', lambda x: x.mean() * 100
            ]).round(2).to_dict(),
            'game_breakdown': self.results_df.groupby('game')['success'].agg([
                'count', 'sum', lambda x: x.mean() * 100
            ]).round(2).to_dict()
        }
        
        # Save summary report
        with open(self.output_dir / 'diagnostic_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Diagnostic report complete. Results saved to {self.output_dir}")
        return summary_stats

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training diagnostic plots')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experimental results')
    
    args = parser.parse_args()
    
    diagnostics = TrainingDiagnostics(args.results_dir)
    summary = diagnostics.generate_comprehensive_report()
    
    print("\nDiagnostic Summary:")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.2f}%")
    
    return summary

if __name__ == "__main__":
    main()