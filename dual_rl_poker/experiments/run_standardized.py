"""Standardized experiment runner for reproducible scientific evaluation.

Executes experiments according to the standardized matrix with exact protocols,
statistical analysis, and result logging to ensure reproducibility.
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.standardized_matrix import STANDARDIZED_MATRIX, ExperimentConfig
from utils.manifest_manager import ManifestManager
from utils.model_analysis import analyze_model_capacity
from utils.diagnostics import TrainingDiagnostics
from utils.logging import setup_logging


class StandardizedRunner:
    """Runner for standardized experiments with scientific rigor."""

    def __init__(self, matrix_config_path: str = None):
        """Initialize standardized runner.

        Args:
            matrix_config_path: Optional path to custom matrix configuration
        """
        self.matrix = STANDARDIZED_MATRIX
        self.manifest_manager = ManifestManager()
        self.logger = setup_logging("INFO")

        if matrix_config_path and Path(matrix_config_path).exists():
            self.logger.info(f"Loading custom matrix from {matrix_config_path}")
            # TODO: Load custom matrix if needed

        self.logger.info("Initialized standardized experiment runner")

    def run_experiment(self, experiment: ExperimentConfig, seed: int) -> Dict[str, Any]:
        """Run a single experiment with standardized protocol.

        Args:
            experiment: Experiment configuration
            seed: Random seed for reproducibility

        Returns:
            Experiment results
        """
        self.logger.info(f"Running {experiment.description} with seed {seed}")

        # Set random seeds for reproducibility
        self._set_seeds(seed)

        # Get full configuration
        config = self.matrix.get_experiment_config(experiment)
        config['reproducibility']['seed'] = seed

        # Validate configuration
        warnings = self.matrix.validate_experiment_config(config)
        if warnings:
            self.logger.warning(f"Configuration warnings: {warnings}")

        # Initialize components
        game_wrapper = self._create_game_wrapper(experiment.game)
        algorithm = self._create_algorithm(experiment.algorithm, game_wrapper, config)
        diagnostics = TrainingDiagnostics()

        # Model capacity analysis
        input_shapes = {experiment.algorithm: (1, game_wrapper.get_encoding_size())}
        models = {experiment.algorithm: algorithm.regret_network}
        capacity_analysis = analyze_model_capacity(models, input_shapes)
        model_info = capacity_analysis[experiment.algorithm]

        # Training loop
        start_time = time.time()
        best_exploitability = float('inf')
        final_metrics = {}

        for iteration in range(config['training']['num_iterations']):
            # Training iteration
            training_state = algorithm.train_iteration()

            # Diagnostics logging
            if iteration % 100 == 0:
                if hasattr(algorithm, 'regret_network'):
                    diagnostics.log_gradient_norms(algorithm.regret_network, iteration)

                if hasattr(training_state, 'extra_metrics') and 'advantage_mean' in training_state.extra_metrics:
                    # Log advantage statistics if available
                    pass  # Implementation depends on specific algorithm

            # Evaluation
            if iteration % config['training']['eval_frequency'] == 0:
                eval_metrics = algorithm.evaluate()
                exploitability = eval_metrics.get('exploitability', float('inf'))

                if exploitability < best_exploitability:
                    best_exploitability = exploitability

                diagnostics.log_checkpoint_metrics(
                    iteration=iteration,
                    nash_conv=eval_metrics.get('nash_conv', 0.0),
                    exploitability=exploitability,
                    additional_metrics={
                        'wall_clock_time': time.time() - start_time,
                        'training_loss': training_state.loss
                    }
                )

                self.logger.info(f"Iter {iteration}: exploitability={exploitability:.6f}")

        # Final evaluation
        final_eval = algorithm.evaluate()
        wall_clock_time = time.time() - start_time

        # Compute training FLOPs
        training_flops = model_info.get('total_training_flops', 0)

        # Prepare results
        results = {
            'algorithm': experiment.algorithm,
            'game': experiment.game,
            'config': config,
            'seed': seed,
            'iteration': config['training']['num_iterations'],
            'nash_conv': final_eval.get('nash_conv', 0.0),
            'exploitability': final_eval.get('exploitability', 0.0),
            'wall_clock_time': wall_clock_time,
            'final_reward': 0.0,  # Not applicable for two-player zero-sum
            'parameters': model_info.get('total_parameters', 0),
            'flops_per_forward': model_info.get('flops_per_forward', 0),
            'training_flops': training_flops,
            'model_size_mb': model_info.get('parameter_size_mb', 0.0),
            'best_exploitability': best_exploitability,
            'diagnostics_path': diagnostics.output_dir,
            'notes': experiment.description
        }

        # Log to manifest
        run_id = self.manifest_manager.log_experiment(**results)
        results['run_id'] = run_id

        # Flush diagnostics
        diagnostics.flush_to_disk()

        self.logger.info(f"Completed {experiment.description} (seed {seed}): "
                        f"exploitability={results['exploitability']:.6f}, "
                        f"time={wall_clock_time:.1f}s")

        return results

    def run_experiment_suite(self,
                           algorithm: str = None,
                           game: str = None,
                           description_contains: str = None) -> List[Dict[str, Any]]:
        """Run a suite of experiments matching filters.

        Args:
            algorithm: Filter by algorithm name
            game: Filter by game name
            description_contains: Filter by description substring

        Returns:
            List of all experiment results
        """
        experiments = self.matrix.get_experiments_by_filter(
            algorithm=algorithm, game=game, description_contains=description_contains
        )

        if not experiments:
            self.logger.warning("No experiments match the specified filters")
            return []

        self.logger.info(f"Running {len(experiments)} experiments")

        all_results = []

        for experiment in experiments:
            experiment_results = []

            for seed in experiment.seeds:
                try:
                    result = self.run_experiment(experiment, seed)
                    experiment_results.append(result)
                except Exception as e:
                    self.logger.error(f"Experiment failed: {experiment.description} "
                                    f"seed {seed}: {e}")
                    continue

            if experiment_results:
                # Compute statistics across seeds
                self._log_experiment_statistics(experiment, experiment_results)
                all_results.extend(experiment_results)

        return all_results

    def _log_experiment_statistics(self, experiment: ExperimentConfig, results: List[Dict[str, Any]]):
        """Log statistical summary for an experiment.

        Args:
            experiment: Experiment configuration
            results: Results from all seeds
        """
        exploitabilities = [r['exploitability'] for r in results]
        times = [r['wall_clock_time'] for r in results]

        import numpy as np

        stats = {
            'algorithm': experiment.algorithm,
            'game': experiment.game,
            'description': experiment.description,
            'num_seeds': len(results),
            'exploitability_mean': np.mean(exploitabilities),
            'exploitability_std': np.std(exploitabilities),
            'exploitability_min': np.min(exploitabilities),
            'exploitability_max': np.max(exploitabilities),
            'exploitability_median': np.median(exploitabilities),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'parameters': results[0]['parameters'],
            'model_size_mb': results[0]['model_size_mb']
        }

        self.logger.info(f"Experiment statistics: {experiment.description}")
        self.logger.info(f"  Exploitability: {stats['exploitability_mean']:.6f} ± {stats['exploitability_std']:.6f}")
        self.logger.info(f"  Range: [{stats['exploitability_min']:.6f}, {stats['exploitability_max']:.6f}]")
        self.logger.info(f"  Time: {stats['time_mean']:.1f} ± {stats['time_std']:.1f}s")
        self.logger.info(f"  Model: {stats['parameters']:,} params, {stats['model_size_mb']:.2f}MB")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed
        """
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_game_wrapper(self, game_name: str):
        """Create game wrapper instance.

        Args:
            game_name: Name of the game

        Returns:
            Game wrapper instance
        """
        if game_name == 'kuhn_poker':
            from environments.kuhn_poker import KuhnPokerGame
            return KuhnPokerGame()
        elif game_name == 'leduc_poker':
            from environments.leduc_holdem import LeducHoldemGame
            return LeducHoldemGame()
        else:
            raise ValueError(f"Unsupported game: {game_name}")

    def _create_algorithm(self, algorithm_name: str, game_wrapper, config: Dict[str, Any]):
        """Create algorithm instance.

        Args:
            algorithm_name: Name of the algorithm
            game_wrapper: Game wrapper instance
            config: Algorithm configuration

        Returns:
            Algorithm instance
        """
        if algorithm_name == 'deep_cfr':
            from algs.deep_cfr import DeepCFRAlgorithm
            return DeepCFRAlgorithm(game_wrapper, config)
        elif algorithm_name == 'sd_cfr':
            from algs.sd_cfr import SDCFRAlgorithm
            return SDCFRAlgorithm(game_wrapper, config)
        elif algorithm_name == 'armac':
            from algs.armac import ARMACAlgorithm
            return ARMACAlgorithm(game_wrapper, config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def main():
    """Main entry point for standardized experiment runner."""
    parser = argparse.ArgumentParser(description="Run standardized experiments")
    parser.add_argument('--algorithm', type=str, choices=['deep_cfr', 'sd_cfr', 'armac'],
                       help='Filter by algorithm')
    parser.add_argument('--game', type=str, choices=['kuhn_poker', 'leduc_poker'],
                       help='Filter by game')
    parser.add_argument('--description', type=str,
                       help='Filter by description substring')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments and exit')
    parser.add_argument('--summary', action='store_true',
                       help='Generate experiment summary table')

    args = parser.parse_args()

    runner = StandardizedRunner()

    if args.list:
        experiments = runner.matrix.get_experiments_by_filter(
            algorithm=args.algorithm, game=args.game, description_contains=args.description
        )
        summary = runner.matrix.generate_summary_table()
        if args.algorithm:
            summary = summary[summary['Algorithm'] == args.algorithm.upper()]
        if args.game:
            summary = summary[summary['Game'] == args.game.replace('_', '-').title()]
        print(summary.to_string(index=False))
        return

    if args.summary:
        summary = runner.matrix.generate_summary_table()
        print(summary.to_string(index=False))
        return

    # Run experiments
    results = runner.run_experiment_suite(
        algorithm=args.algorithm,
        game=args.game,
        description_contains=args.description
    )

    if results:
        # Generate final summary
        runner.manifest_manager.generate_summary_report(
            output_path="results/experiment_summary.txt"
        )
        print(f"Completed {len(results)} experiments. Results logged to manifest.")


if __name__ == "__main__":
    main()