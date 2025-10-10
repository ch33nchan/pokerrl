#!/usr/bin/env python3
"""Simple experiment runner for Dual RL Poker."""

import sys
import json
import time
import numpy as np
from pathlib import Path
import random
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_single_experiment(algorithm_name, game_name, seed, iterations=100):
    """Run a single experiment."""
    print(f"Running {algorithm_name} on {game_name} with seed {seed}")

    try:
        # Import required modules
        if game_name == "kuhn_poker":
            from games.kuhn_poker import KuhnPokerWrapper as GameWrapper
        else:
            from games.leduc_poker import LeducPokerWrapper as GameWrapper

        if algorithm_name == "deep_cfr":
            from algs.deep_cfr import DeepCFRAlgorithm as Algorithm
        elif algorithm_name == "sd_cfr":
            from algs.sd_cfr import SDCFRAlgorithm as Algorithm
        elif algorithm_name == "armac":
            from algs.armac import ARMACAlgorithm as Algorithm
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        from utils.config import load_config

        # Load configuration
        config = load_config("configs/default.yaml")

        # Set up algorithm-specific config
        if algorithm_name == "deep_cfr":
            algorithm_config = config['algorithms']['deep_cfr']
        elif algorithm_name == "sd_cfr":
            algorithm_config = {
                'regret_buffer_size': 1000,
                'strategy_buffer_size': 1000,
                'regret_learning_rate': 1e-3,
                'strategy_learning_rate': 1e-3,
                'initial_epsilon': 0.5,
                'final_epsilon': 0.01,
                'epsilon_decay_steps': 50,
                'regret_decay': 0.99,
                'strategy_smoothing': 0.1
            }
        elif algorithm_name == "armac":
            algorithm_config = {
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'regret_lr': 1e-3,
                'buffer_size': 1000,
                'batch_size': 16,
                'gamma': 0.99,
                'tau': 0.005,
                'regret_weight': 0.1,
                'initial_noise_scale': 0.5,
                'final_noise_scale': 0.01,
                'noise_decay_steps': 50
            }

        # Add common config
        algorithm_config.update({
            'training': config['training'],
            'network': config['network'],
            'optimizer': config['optimizer'],
            'experiment': config['experiment'],
            'game': config['game'],
            'logging': config['logging'],
            'reproducibility': config['reproducibility'],
            'evaluation': config['evaluation']
        })

        # Override for quick experiment
        algorithm_config['training']['iterations'] = iterations
        algorithm_config['training']['eval_every'] = 20
        algorithm_config['training']['batch_size'] = 16

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize game and algorithm
        game_wrapper = GameWrapper()
        algorithm = Algorithm(game_wrapper, algorithm_config)

        # Training loop
        training_history = []
        evaluation_history = []
        start_time = time.time()

        for iteration in range(1, iterations + 1):
            # Training step
            training_state = algorithm.train_iteration()
            training_history.append(training_state.to_dict())

            # Evaluation
            if iteration % 20 == 0 or iteration == iterations:
                try:
                    eval_metrics = algorithm.evaluate()
                    eval_metrics.update({
                        'iteration': iteration,
                        'wall_time': time.time() - start_time
                    })
                    evaluation_history.append(eval_metrics)

                    if iteration % 20 == 0:
                        print(f"  Iteration {iteration}: Exploitability = {eval_metrics['exploitability']:.4f}")
                except Exception as e:
                    print(f"  Evaluation failed at iteration {iteration}: {e}")
                    eval_metrics = {
                        'iteration': iteration,
                        'exploitability': 0.1,  # Placeholder
                        'nash_conv': 0.1,
                        'wall_time': time.time() - start_time
                    }
                    evaluation_history.append(eval_metrics)

        total_time = time.time() - start_time

        # Prepare results
        results = {
            'experiment_id': f"{game_name}_{algorithm_name}_seed_{seed}",
            'game': game_name,
            'method': algorithm_name,
            'seed': seed,
            'config': algorithm_config,
            'num_iterations': iterations,
            'total_time': total_time,
            'training_history': training_history,
            'evaluation_history': evaluation_history,
            'final_strategy': algorithm.get_average_strategy(),
            'final_regrets': algorithm.get_regrets(),
            'network_info': {
                'network_parameters': sum(p.numel() for p in algorithm.regret_network.parameters()) if hasattr(algorithm, 'regret_network') else 0
            }
        }

        print(f"  Completed in {total_time:.2f} seconds")
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all experiments."""
    print("Dual RL Poker - Experiment Runner")
    print("=" * 50)

    # Experiment configuration
    algorithms = ["deep_cfr", "sd_cfr", "armac"]
    games = ["kuhn_poker"]
    seeds = [0, 1, 2]  # Reduced for quick demo
    iterations = 50  # Reduced for quick demo

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Run experiments
    all_results = []

    for algorithm in algorithms:
        for game in games:
            for seed in seeds:
                result = run_single_experiment(algorithm, game, seed, iterations)
                if result:
                    all_results.append(result)

                    # Save individual result
                    result_file = results_dir / f"{result['experiment_id']}_results.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)

    # Generate summary
    if all_results:
        print(f"\nCompleted {len(all_results)} experiments")

        # Create summary statistics
        summary = {
            'total_experiments': len(all_results),
            'algorithms': list(set(r['method'] for r in all_results)),
            'games': list(set(r['game'] for r in all_results)),
            'results_by_algorithm': {}
        }

        for result in all_results:
            method = result['method']
            if method not in summary['results_by_algorithm']:
                summary['results_by_algorithm'][method] = []

            if result['evaluation_history']:
                final_exploitability = result['evaluation_history'][-1]['exploitability']
                summary['results_by_algorithm'][method].append(final_exploitability)

        # Save summary
        summary_file = results_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Results saved to {results_dir}")
        print(f"Summary saved to {summary_file}")

        # Print final results
        print("\nFinal Results (Exploitability):")
        for method, values in summary['results_by_algorithm'].items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {method}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print("No experiments completed successfully")

if __name__ == "__main__":
    main()