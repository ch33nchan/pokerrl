"""Standardized experiment matrix for reproducible scientific evaluation.

Defines fixed experimental protocols with exact hyperparameters, evaluation metrics,
and statistical procedures to ensure fair comparison across algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class ExperimentConfig:
    """Configuration for a standardized experiment."""
    algorithm: str
    game: str
    network_hidden_dims: List[int]
    batch_size: int
    learning_rate: float
    num_iterations: int
    eval_frequency: int
    num_eval_episodes: int
    seeds: List[int]
    description: str


class StandardizedMatrix:
    """Standardized experiment matrix for reproducible research."""

    def __init__(self):
        """Initialize standardized experiment matrix."""
        self.experiments = self._define_experiments()
        self.base_configs = self._define_base_configs()

    def _define_base_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define base configurations for different algorithms."""
        return {
            'deep_cfr': {
                'regret_lr': 1e-3,
                'strategy_lr': 1e-3,
                'regret_buffer_size': 10000,
                'strategy_buffer_size': 10000,
                'batch_size': 64,
                'gradient_clip': 5.0,
                'weight_decay': 0.0
            },
            'sd_cfr': {
                'regret_lr': 1e-3,
                'regret_buffer_size': 10000,
                'batch_size': 64,
                'gradient_clip': 5.0,
                'weight_decay': 0.0,
                'regret_decay': 0.99,
                'initial_epsilon': 0.5,
                'final_epsilon': 0.01,
                'epsilon_decay_steps': 1000
            },
            'armac': {
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'regret_lr': 1e-3,
                'buffer_size': 10000,
                'batch_size': 64,
                'gamma': 0.99,
                'tau': 0.005,
                'regret_weight': 0.1,
                'gradient_clip': 5.0,
                'weight_decay': 0.0,
                'initial_noise_scale': 0.5,
                'final_noise_scale': 0.01,
                'noise_decay_steps': 1000
            }
        }

    def _define_experiments(self) -> List[ExperimentConfig]:
        """Define standardized experiment configurations."""
        experiments = []

        # Kuhn Poker experiments
        kuhn_network_dims = [[64, 64], [128, 128], [256, 128]]

        for i, hidden_dims in enumerate(kuhn_network_dims):
            experiments.append(ExperimentConfig(
                algorithm='deep_cfr',
                game='kuhn_poker',
                network_hidden_dims=hidden_dims,
                batch_size=64,
                learning_rate=1e-3,
                num_iterations=5000,
                eval_frequency=500,
                num_eval_episodes=1000,
                seeds=[42, 123, 456, 789, 999],
                description=f'Kuhn Deep CFR network_size_{i+1}'
            ))

            experiments.append(ExperimentConfig(
                algorithm='sd_cfr',
                game='kuhn_poker',
                network_hidden_dims=hidden_dims,
                batch_size=64,
                learning_rate=1e-3,
                num_iterations=5000,
                eval_frequency=500,
                num_eval_episodes=1000,
                seeds=[42, 123, 456, 789, 999],
                description=f'Kuhn SD-CFR network_size_{i+1}'
            ))

            experiments.append(ExperimentConfig(
                algorithm='armac',
                game='kuhn_poker',
                network_hidden_dims=hidden_dims,
                batch_size=64,
                learning_rate=1e-4,  # Actor LR
                num_iterations=5000,
                eval_frequency=500,
                num_eval_episodes=1000,
                seeds=[42, 123, 456, 789, 999],
                description=f'Kuhn ARMAC network_size_{i+1}'
            ))

        # Leduc Hold'em experiments
        leduc_network_dims = [[128, 128], [256, 128], [512, 256]]

        for i, hidden_dims in enumerate(leduc_network_dims):
            experiments.append(ExperimentConfig(
                algorithm='deep_cfr',
                game='leduc_poker',
                network_hidden_dims=hidden_dims,
                batch_size=128,
                learning_rate=1e-3,
                num_iterations=10000,
                eval_frequency=1000,
                num_eval_episodes=500,
                seeds=[42, 123, 456, 789, 999],
                description=f'Leduc Deep CFR network_size_{i+1}'
            ))

            experiments.append(ExperimentConfig(
                algorithm='sd_cfr',
                game='leduc_poker',
                network_hidden_dims=hidden_dims,
                batch_size=128,
                learning_rate=1e-3,
                num_iterations=10000,
                eval_frequency=1000,
                num_eval_episodes=500,
                seeds=[42, 123, 456, 789, 999],
                description=f'Leduc SD-CFR network_size_{i+1}'
            ))

            experiments.append(ExperimentConfig(
                algorithm='armac',
                game='leduc_poker',
                network_hidden_dims=hidden_dims,
                batch_size=128,
                learning_rate=1e-4,  # Actor LR
                num_iterations=10000,
                eval_frequency=1000,
                num_eval_episodes=500,
                seeds=[42, 123, 456, 789, 999],
                description=f'Leduc ARMAC network_size_{i+1}'
            ))

        return experiments

    def get_experiment_config(self, experiment: ExperimentConfig) -> Dict[str, Any]:
        """Generate full configuration for an experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            Complete configuration dictionary
        """
        # Start with base configuration
        config = {
            'algorithm': experiment.algorithm,
            'game': experiment.game,
            'training': {
                'num_iterations': experiment.num_iterations,
                'batch_size': experiment.batch_size,
                'eval_frequency': experiment.eval_frequency,
                'num_eval_episodes': experiment.num_eval_episodes
            },
            'network': {
                'hidden_dims': experiment.network_hidden_dims
            },
            'evaluation': {
                'method': 'openspiel_exact',  # Use exact evaluation
                'num_bootstrap_samples': 100,
                'confidence_level': 0.95
            },
            'logging': {
                'log_level': 'INFO',
                'log_frequency': 100,
                'save_checkpoints': True,
                'checkpoint_frequency': 1000
            },
            'reproducibility': {
                'seeds': experiment.seeds,
                'deterministic_cudnn': True,
                'track_diagnostics': True
            }
        }

        # Add algorithm-specific parameters
        if experiment.algorithm in self.base_configs:
            algorithm_config = self.base_configs[experiment.algorithm].copy()

            # Override learning rate if specified
            if experiment.learning_rate != algorithm_config.get('regret_lr', 1e-3):
                if experiment.algorithm == 'armac':
                    algorithm_config['actor_lr'] = experiment.learning_rate
                else:
                    algorithm_config['regret_lr'] = experiment.learning_rate

            config.update(algorithm_config)

        return config

    def get_experiments_by_filter(self,
                                 algorithm: str = None,
                                 game: str = None,
                                 description_contains: str = None) -> List[ExperimentConfig]:
        """Get experiments matching specified filters.

        Args:
            algorithm: Filter by algorithm name
            game: Filter by game name
            description_contains: Filter by description substring

        Returns:
            List of matching experiments
        """
        filtered = self.experiments

        if algorithm:
            filtered = [e for e in filtered if e.algorithm == algorithm]
        if game:
            filtered = [e for e in filtered if e.game == game]
        if description_contains:
            filtered = [e for e in filtered if description_contains in e.description]

        return filtered

    def export_matrix(self, output_path: str):
        """Export experiment matrix to JSON.

        Args:
            output_path: Path to save the matrix
        """
        matrix_data = []
        for exp in self.experiments:
            exp_dict = {
                'algorithm': exp.algorithm,
                'game': exp.game,
                'network_hidden_dims': exp.network_hidden_dims,
                'batch_size': exp.batch_size,
                'learning_rate': exp.learning_rate,
                'num_iterations': exp.num_iterations,
                'eval_frequency': exp.eval_frequency,
                'num_eval_episodes': exp.num_eval_episodes,
                'seeds': exp.seeds,
                'description': exp.description
            }
            matrix_data.append(exp_dict)

        with open(output_path, 'w') as f:
            json.dump(matrix_data, f, indent=2)

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all experiments.

        Returns:
            DataFrame with experiment summaries
        """
        data = []
        for exp in self.experiments:
            row = {
                'Algorithm': exp.algorithm.upper(),
                'Game': exp.game.replace('_', '-').title(),
                'Network Size': 'x'.join(map(str, exp.network_hidden_dims)),
                'Iterations': f"{exp.num_iterations:,}",
                'Batch Size': exp.batch_size,
                'LR': exp.learning_rate,
                'Seeds': len(exp.seeds),
                'Description': exp.description
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_statistical_requirements(self) -> Dict[str, Any]:
        """Get statistical analysis requirements for experiments.

        Returns:
            Statistical analysis configuration
        """
        return {
            'significance_tests': {
                'method': 'wilcoxon_signed_rank',  # Non-parametric paired test
                'alpha': 0.05,
                'correction': 'holm_bonferroni',  # Multiple comparison correction
                'effect_size': 'cliffs_delta'
            },
            'confidence_intervals': {
                'method': 'bootstrap',
                'n_bootstrap': 10000,
                'confidence_level': 0.95,
                'seed': 42
            },
            'performance_metrics': [
                'exploitability',
                'nash_conv',
                'wall_clock_time',
                'convergence_iteration'
            ],
            'reporting_requirements': {
                'mean_std': True,
                'confidence_intervals': True,
                'effect_sizes': True,
                'statistical_tests': True,
                'computational_cost': True
            }
        }

    def validate_experiment_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate experiment configuration for scientific rigor.

        Args:
            config: Configuration to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check evaluation method
        eval_method = config.get('evaluation', {}).get('method', 'unknown')
        if eval_method != 'openspiel_exact':
            warnings.append(f"Non-exact evaluation method: {eval_method}")

        # Check number of seeds
        seeds = config.get('reproducibility', {}).get('seeds', [])
        if len(seeds) < 3:
            warnings.append(f"Insufficient seeds for statistical significance: {len(seeds)}")

        # Check evaluation frequency
        eval_freq = config.get('training', {}).get('eval_frequency', 0)
        total_iter = config.get('training', {}).get('num_iterations', 0)
        if total_iter // eval_freq < 10:
            warnings.append(f"Insufficient evaluation points: {total_iter // eval_freq}")

        # Check network architecture
        hidden_dims = config.get('network', {}).get('hidden_dims', [])
        if len(hidden_dims) < 2:
            warnings.append(f"Shallow network architecture: {hidden_dims}")

        # Check diagnostic tracking
        diagnostics = config.get('reproducibility', {}).get('track_diagnostics', False)
        if not diagnostics:
            warnings.append("Diagnostic tracking not enabled")

        return warnings


# Global instance for easy access
STANDARDIZED_MATRIX = StandardizedMatrix()