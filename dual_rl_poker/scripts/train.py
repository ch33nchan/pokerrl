#!/usr/bin/env python3
"""
Training script for individual algorithm experiments.

Usage:
    python scripts/train.py --game kuhn_poker --method deep_cfr --seeds 0-4
    python scripts/train.py --game leduc_poker --method sd_cfr --seeds 0-2 --config configs/custom.yaml
"""

import argparse
import logging
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games import KuhnPokerWrapper, LeducPokerWrapper
from algs import DeepCFRAlgorithm
from eval import OpenSpielEvaluator
from utils.config import load_config
from utils.logging import setup_logging


def parse_seed_range(seeds_str: str) -> List[int]:
    """Parse seed range like '0-19' into list [0, 1, ..., 19]."""
    if '-' in seeds_str:
        start, end = map(int, seeds_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(seeds_str)]


def get_game_wrapper(game_name: str):
    """Get game wrapper for specified game."""
    if game_name == "kuhn_poker":
        return KuhnPokerWrapper()
    elif game_name == "leduc_poker":
        return LeducPokerWrapper()
    else:
        raise ValueError(f"Unknown game: {game_name}")


def get_algorithm(method_name: str, game_wrapper, config: Dict[str, Any]):
    """Get algorithm instance for specified method."""
    if method_name == "deep_cfr":
        return DeepCFRAlgorithm(game_wrapper, config)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_experiment(config: Dict[str, Any], game_name: str, method_name: str,
                         seed: int, output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment.

    Args:
        config: Configuration dictionary
        game_name: Name of the game
        method_name: Name of the method
        seed: Random seed
        output_dir: Output directory for results

    Returns:
        Experiment results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {method_name} on {game_name} with seed {seed}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize game and algorithm
    game_wrapper = get_game_wrapper(game_name)
    algorithm = get_algorithm(method_name, game_wrapper, config)

    # Training parameters
    num_iterations = config['training']['iterations']
    eval_every = config['training']['eval_every']
    save_every = config['training']['save_every']

    # Initialize evaluator
    evaluator = OpenSpielEvaluator(game_wrapper)

    # Training loop
    training_states = []
    evaluation_results = []
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        # Training step
        training_state = algorithm.train_iteration()
        training_states.append(training_state.to_dict())

        # Evaluation
        if iteration % eval_every == 0 or iteration == num_iterations:
            logger.info(f"Evaluating at iteration {iteration}/{num_iterations}")

            eval_metrics = algorithm.evaluate()
            eval_metrics.update({
                'iteration': iteration,
                'wall_time': time.time() - start_time
            })
            evaluation_results.append(eval_metrics)

            logger.info(f"Iteration {iteration}: "
                       f"Exploitability = {eval_metrics['exploitability']:.4f}, "
                       f"NashConv = {eval_metrics['nash_conv']:.4f}")

        # Save checkpoint
        if iteration % save_every == 0 or iteration == num_iterations:
            checkpoint_dir = output_dir / "checkpoints" / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model state
            if hasattr(algorithm, 'regret_network'):
                torch.save(algorithm.regret_network.state_dict(),
                          checkpoint_dir / f"regret_network_iter_{iteration}.pt")
            if hasattr(algorithm, 'strategy_network'):
                torch.save(algorithm.strategy_network.state_dict(),
                          checkpoint_dir / f"strategy_network_iter_{iteration}.pt")

            # Save strategies and regrets
            checkpoint_data = {
                'iteration': iteration,
                'seed': seed,
                'config': config,
                'training_states': training_states[-eval_every:],
                'evaluation_results': evaluation_results[-1],
                'average_strategy': algorithm.get_average_strategy(),
                'regrets': algorithm.get_regrets()
            }

            with open(checkpoint_dir / f"checkpoint_iter_{iteration}.json", 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

    total_time = time.time() - start_time

    # Final evaluation with more episodes
    logger.info("Running final comprehensive evaluation...")
    final_eval = evaluator.evaluate_with_diagnostics(
        {player: algorithm.get_policy(player) for player in range(2)},
        num_episodes=1000
    )

    # Prepare final results
    results = {
        'experiment_id': f"{game_name}_{method_name}_seed_{seed}",
        'game': game_name,
        'method': method_name,
        'seed': seed,
        'config': config,
        'num_iterations': num_iterations,
        'total_time': total_time,
        'final_evaluation': final_eval,
        'training_history': training_states,
        'evaluation_history': evaluation_results,
        'final_strategy': algorithm.get_average_strategy(),
        'final_regrets': algorithm.get_regrets(),
        'network_info': {
            'regret_network_params': sum(p.numel() for p in algorithm.regret_network.parameters()),
            'strategy_network_params': sum(p.numel() for p in algorithm.strategy_network.parameters())
        } if hasattr(algorithm, 'regret_network') else {}
    }

    # Save final results
    results_file = output_dir / f"{results['experiment_id']}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Experiment completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to {results_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a single algorithm experiment")
    parser.add_argument("--game", type=str, required=True,
                        choices=["kuhn_poker", "leduc_poker"],
                        help="Game to train on")
    parser.add_argument("--method", type=str, required=True,
                        choices=["deep_cfr", "sd_cfr", "armac"],
                        help="Method to train")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Seed range (e.g., '0-4' or '42')")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config)

    # Update config with command line arguments
    config['game']['name'] = args.game
    config['algorithm']['method'] = args.method

    # Parse seeds
    seeds = parse_seed_range(args.seeds)
    logger.info(f"Running experiment for seeds: {seeds}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.game / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments for each seed
    all_results = []
    for seed in seeds:
        try:
            result = run_single_experiment(config, args.game, args.method, seed, output_dir)
            all_results.append(result)
            logger.info(f"✓ Completed seed {seed}")
        except Exception as e:
            logger.error(f"✗ Failed seed {seed}: {e}")
            continue

    # Save aggregate results
    aggregate_results = {
        'experiment_summary': {
            'game': args.game,
            'method': args.method,
            'seeds': seeds,
            'successful_runs': len(all_results),
            'failed_runs': len(seeds) - len(all_results)
        },
        'config': config,
        'results': all_results
    }

    aggregate_file = output_dir / "aggregate_results.json"
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)

    logger.info(f"Aggregate results saved to {aggregate_file}")
    logger.info(f"Completed {len(all_results)}/{len(seeds)} experiments successfully")


if __name__ == "__main__":
    main()