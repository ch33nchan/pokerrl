"""
Canonical experiment matrix with exact protocols following executive directive.

Fixed protocols:
- Kuhn: 20 seeds (0-19), 500 iterations, eval_every=25, External Sampling
- Leduc: 10 seeds (0-9), 200 iterations, eval_every=20, External Sampling
- Identical 2×64 MLP trunks with logged parameter counts and FLOPs
- OpenSpiel exact evaluation only
- No Monte Carlo or loss-only metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib


@dataclass
class CanonicalExperimentConfig:
    """Configuration for canonical experiment."""
    # Core experiment identification
    run_id: str
    game: str  # 'kuhn_poker' or 'leduc_poker'
    algorithm: str  # 'deep_cfr', 'sd_cfr_canonical', 'armac_canonical'
    traversal: str = 'external_sampling'  # Fixed to external sampling

    # Training parameters (exact as specified)
    seeds: List[int] = None  # Will be set per game
    iterations: int = 0  # Will be set per game
    eval_every: int = 0  # Will be set per game
    batch_size: int = 64

    # Network architecture (identical 2×64 MLP trunks)
    hidden_dims: List[int] = None  # Will be set per game
    network_type: str = 'mlp'

    # Optimization parameters (exact)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 5.0

    # Evaluation parameters
    eval_method: str = 'openspiel_exact'  # Exact evaluation only
    bootstrap_samples: int = 10000
    confidence_level: float = 0.95

    # Computational tracking
    log_flops: bool = True
    log_diagnostics: bool = True
    log_timing: bool = True

    # Description
    description: str = ""

    def __post_init__(self):
        """Post-initialization validation."""
        if self.seeds is None:
            raise ValueError("Seeds must be specified")
        if self.iterations == 0:
            raise ValueError("Iterations must be specified")
        if self.eval_every == 0:
            raise ValueError("Eval frequency must be specified")
        if self.hidden_dims is None:
            raise ValueError("Hidden dimensions must be specified")


class CanonicalExperimentMatrix:
    """Canonical experiment matrix with fixed protocols per executive directive."""

    def __init__(self):
        """Initialize canonical experiment matrix."""
        self.experiments = self._define_canonical_experiments()
        self.base_configs = self._define_base_algorithm_configs()

    def _define_base_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define base configurations for each algorithm."""
        return {
            'deep_cfr': {
                'regret_lr': 1e-3,
                'strategy_lr': 1e-3,
                'buffer_size': 10000,
                'replay_window': 10
            },
            'sd_cfr_canonical': {
                'regret_lr': 1e-3,
                'buffer_size': 10000
                # No regret decay or adaptive exploration (canonical)
            },
            'armac_canonical': {
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'regret_lr': 1e-3,
                'buffer_size': 10000,
                'gamma': 0.99,
                'tau': 0.005,
                'regret_weight': 0.1,  # λ mixing weight
                'entropy_coeff': 0.01
            }
        }

    def _define_canonical_experiments(self) -> List[CanonicalExperimentConfig]:
        """Define canonical experiments per executive directive."""
        experiments = []

        # Kuhn Poker experiments (exact specification)
        kuhn_seeds = list(range(20))  # seeds 0-19
        kuhn_iterations = 500
        kuhn_eval_every = 25
        kuhn_hidden_dims = [64, 64]  # Identical 2×64 MLP trunks

        algorithms = ['deep_cfr', 'sd_cfr_canonical', 'armac_canonical']

        for algorithm in algorithms:
            experiment = CanonicalExperimentConfig(
                run_id=f"{algorithm}_kuhn_poker_external_sampling",
                game='kuhn_poker',
                algorithm=algorithm,
                seeds=kuhn_seeds,
                iterations=kuhn_iterations,
                eval_every=kuhn_eval_every,
                hidden_dims=kuhn_hidden_dims,
                description=f"Kuhn Poker {algorithm.replace('_', ' ').title()} with External Sampling (20 seeds, 500 iterations)"
            )
            experiments.append(experiment)

        # Leduc Hold'em experiments (limited-budget extension)
        leduc_seeds = list(range(10))  # seeds 0-9
        leduc_iterations = 200
        leduc_eval_every = 20
        leduc_hidden_dims = [64, 64]  # Identical capacity to Kuhn for fair comparison

        for algorithm in algorithms:
            experiment = CanonicalExperimentConfig(
                run_id=f"{algorithm}_leduc_poker_external_sampling",
                game='leduc_poker',
                algorithm=algorithm,
                seeds=leduc_seeds,
                iterations=leduc_iterations,
                eval_every=leduc_eval_every,
                hidden_dims=leduc_hidden_dims,
                description=f"Leduc Poker {algorithm.replace('_', ' ').title()} with External Sampling (10 seeds, 200 iterations) - Limited Budget Extension"
            )
            experiments.append(experiment)

        return experiments

    def get_experiment_config(self, experiment: CanonicalExperimentConfig) -> Dict[str, Any]:
        """Generate full configuration for an experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            Complete configuration dictionary
        """
        # Start with base configuration
        config = {
            'experiment': asdict(experiment),
            'training': {
                'num_iterations': experiment.iterations,
                'batch_size': experiment.batch_size,
                'eval_frequency': experiment.eval_every,
                'traversal': experiment.traversal
            },
            'network': {
                'hidden_dims': experiment.hidden_dims,
                'network_type': experiment.network_type
            },
            'evaluation': {
                'method': experiment.eval_method,
                'bootstrap_samples': experiment.bootstrap_samples,
                'confidence_level': experiment.confidence_level
            },
            'logging': {
                'log_level': 'INFO',
                'log_frequency': max(1, experiment.eval_every // 5),
                'log_flops': experiment.log_flops,
                'log_diagnostics': experiment.log_diagnostics,
                'log_timing': experiment.log_timing
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
            config.update(algorithm_config)

        # Generate configuration hash for reproducibility
        config_str = json.dumps(config, sort_keys=True)
        config['config_hash'] = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return config

    def get_experiments_by_filter(self,
                                 game: str = None,
                                 algorithm: str = None,
                                 traversal: str = None) -> List[CanonicalExperimentConfig]:
        """Get experiments matching specified filters.

        Args:
            game: Filter by game name
            algorithm: Filter by algorithm name
            traversal: Filter by traversal method

        Returns:
            List of matching experiments
        """
        filtered = self.experiments

        if game:
            filtered = [e for e in filtered if e.game == game]
        if algorithm:
            filtered = [e for e in filtered if e.algorithm == algorithm]
        if traversal:
            filtered = [e for e in filtered if e.traversal == traversal]

        return filtered

    def export_matrix(self, output_path: str):
        """Export experiment matrix to JSON.

        Args:
            output_path: Path to save the matrix
        """
        matrix_data = []
        for exp in self.experiments:
            exp_dict = asdict(exp)
            matrix_data.append(exp_dict)

        with open(output_path, 'w') as f:
            json.dump(matrix_data, f, indent=2)

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all canonical experiments.

        Returns:
            DataFrame with experiment summaries
        """
        data = []
        for exp in self.experiments:
            row = {
                'Algorithm': exp.algorithm.replace('_', ' ').title(),
                'Game': exp.game.replace('_', '-').title(),
                'Traversal': exp.traversal.replace('_', ' ').title(),
                'Network': '×'.join(map(str, exp.hidden_dims)),
                'Iterations': f"{exp.iterations:,}",
                'Eval Every': exp.eval_every,
                'Seeds': len(exp.seeds),
                'Seed Range': f"{exp.seeds[0]}-{exp.seeds[-1]}",
                'Run ID': exp.run_id,
                'Description': exp.description
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_computational_requirements(self) -> Dict[str, Any]:
        """Get computational requirements for all experiments.

        Returns:
            Computational analysis
        """
        total_runs = sum(len(exp.seeds) for exp in self.experiments)
        total_iterations = sum(exp.iterations * len(exp.seeds) for exp in self.experiments)

        # Estimate FLOPs per forward pass (2×64 MLP)
        # Input size depends on game, use approximate estimates
        flops_per_forward = {
            'kuhn_poker': 2 * 64 * 64,  # Two layers of 64×64 operations
            'leduc_poker': 2 * 64 * 64 * 2  # Larger input, same network
        }

        total_flops = 0
        for exp in self.experiments:
            game_flops = flops_per_forward.get(exp.game, 2 * 64 * 64)
            total_flops += exp.iterations * len(exp.seeds) * game_flops * 3  # 3 networks for ARMAC

        return {
            'total_experiments': len(self.experiments),
            'total_runs': total_runs,
            'total_iterations': total_iterations,
            'estimated_flops': total_flops,
            'algorithms': list(set(exp.algorithm for exp in self.experiments)),
            'games': list(set(exp.game for exp in self.experiments)),
            'traversal_methods': list(set(exp.traversal for exp in self.experiments))
        }

    def validate_experiment_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate experiment configuration for scientific rigor.

        Args:
            config: Configuration to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check traversal method
        traversal = config.get('experiment', {}).get('traversal', 'unknown')
        if traversal != 'external_sampling':
            warnings.append(f"Non-canonical traversal method: {traversal}")

        # Check evaluation method
        eval_method = config.get('evaluation', {}).get('method', 'unknown')
        if eval_method != 'openspiel_exact':
            warnings.append(f"Non-exact evaluation method: {eval_method}")

        # Check number of seeds
        seeds = config.get('experiment', {}).get('seeds', [])
        if len(seeds) < 5:
            warnings.append(f"Insufficient seeds for statistical significance: {len(seeds)}")

        # Check evaluation frequency
        eval_freq = config.get('training', {}).get('eval_frequency', 0)
        total_iter = config.get('experiment', {}).get('iterations', 0)
        if total_iter // eval_freq < 10:
            warnings.append(f"Insufficient evaluation points: {total_iter // eval_freq}")

        # Check network architecture
        hidden_dims = config.get('network', {}).get('hidden_dims', [])
        if hidden_dims != [64, 64]:
            warnings.append(f"Non-standard network architecture: {hidden_dims}")

        # Check diagnostic tracking
        diagnostics = config.get('logging', {}).get('log_diagnostics', False)
        if not diagnostics:
            warnings.append("Diagnostic tracking not enabled")

        # Check FLOPs logging
        log_flops = config.get('logging', {}).get('log_flops', False)
        if not log_flops:
            warnings.append("FLOPs logging not enabled")

        return warnings


# Global instance for easy access
CANONICAL_MATRIX = CanonicalExperimentMatrix()


def main():
    """Main function for testing the canonical matrix."""
    matrix = CANONICAL_MATRIX

    print("Canonical Experiment Matrix Summary:")
    print("=" * 50)

    # Display summary table
    summary = matrix.generate_summary_table()
    print(summary.to_string(index=False))

    print("\nComputational Requirements:")
    print("=" * 50)
    reqs = matrix.get_computational_requirements()
    for key, value in reqs.items():
        print(f"{key}: {value}")

    print("\nValidation Results:")
    print("=" * 50)
    for exp in matrix.experiments[:3]:  # Validate first 3 as example
        config = matrix.get_experiment_config(exp)
        warnings = matrix.validate_experiment_config(config)
        if warnings:
            print(f"{exp.run_id}: {len(warnings)} warnings")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print(f"{exp.run_id}: ✓ Valid")


if __name__ == "__main__":
    main()