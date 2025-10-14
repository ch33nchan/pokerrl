#!/usr/bin/env python3
"""
Quick ARMAC scheduler experiment runner.

This script runs a simple experiment to test the ARMAC algorithm
with scheduler components on Kuhn poker.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.game_factory import GameFactory
from algs.armac import ARMACAlgorithm
from eval.openspiel_evaluator import OpenSpielExactEvaluator


def run_experiment(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single ARMAC experiment.

    Args:
        config_name: Name of the configuration
        config: Configuration dictionary

    Returns:
        Experiment results
    """
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {config_name}")
    print(f"{'=' * 60}")

    # Set random seed
    seed = config.get("seed", 42)
    import torch
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create game
    game_factory = GameFactory()
    game_wrapper = game_factory.create_game("kuhn_poker")

    # Create algorithm
    algorithm = ARMACAlgorithm(game_wrapper, config["algorithm"])

    # Create evaluator
    evaluator = OpenSpielExactEvaluator("kuhn_poker")

    # Training parameters
    total_iterations = config.get("total_iterations", 200)
    eval_every = config.get("eval_every", 25)

    # Training loop
    start_time = time.time()
    results = {
        "config_name": config_name,
        "iterations": [],
        "exploitabilities": [],
        "nash_convs": [],
        "losses": [],
        "scheduler_stats": [],
    }

    best_exploitability = float("inf")

    try:
        for iteration in range(total_iterations):
            # Training step
            training_state = algorithm.train_iteration()

            # Evaluation
            if iteration % eval_every == 0:
                # Get policy for evaluation
                policy_adapter = algorithm.get_policy_adapter()

                # Evaluate
                eval_result = evaluator.evaluate(policy_adapter)
                exploitability = eval_result.exploitability
                nash_conv = eval_result.nash_conv

                # Get scheduler stats
                scheduler_stats = {}
                if hasattr(algorithm, "armac_dual_rl"):
                    scheduler_stats = algorithm.armac_dual_rl.get_scheduler_stats()

                # Record results
                results["iterations"].append(iteration)
                results["exploitabilities"].append(exploitability)
                results["nash_convs"].append(nash_conv)
                results["losses"].append(training_state.loss)
                results["scheduler_stats"].append(scheduler_stats)

                # Track best performance
                if exploitability < best_exploitability:
                    best_exploitability = exploitability

                print(
                    f"Iter {iteration:3d}: Exploitability={exploitability:.6f}, "
                    f"NashConv={nash_conv:.6f}, Loss={training_state.loss:.6f}"
                )

                if scheduler_stats:
                    lambda_mean = scheduler_stats.get("lambda_mean", "N/A")
                    print(f"         Lambda: {lambda_mean}")

        # Final evaluation
        final_policy_adapter = algorithm.get_policy_adapter()
        final_eval_result = evaluator.evaluate(final_policy_adapter)
        final_exploitability = final_eval_result.exploitability
        final_nash_conv = final_eval_result.nash_conv

        training_time = time.time() - start_time

        # Final results
        results.update(
            {
                "final_exploitability": final_exploitability,
                "final_nash_conv": final_nash_conv,
                "best_exploitability": best_exploitability,
                "training_time": training_time,
                "success": True,
                "error": None,
            }
        )

        print(f"\nExperiment completed successfully!")
        print(f"Final exploitability: {final_exploitability:.6f}")
        print(f"Best exploitability: {best_exploitability:.6f}")
        print(f"Training time: {training_time:.2f}s")

    except Exception as e:
        error_msg = str(e)
        print(f"\nExperiment failed: {error_msg}")
        results.update(
            {
                "final_exploitability": float("inf"),
                "final_nash_conv": float("inf"),
                "best_exploitability": float("inf"),
                "training_time": time.time() - start_time,
                "success": False,
                "error": error_msg,
            }
        )

    return results


def main():
    """Main experiment runner."""
    print("ARMAC Scheduler Quick Test")
    print("=" * 60)

    # Create output directory
    output_dir = Path("results/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiment configurations
    experiments = {
        "fixed_lambda": {
            "seed": 42,
            "total_iterations": 200,
            "eval_every": 25,
            "algorithm": {
                "type": "armac",
                "use_scheduler": False,
                "mixture_weight": 0.1,
                "lambda_mode": "fixed",
                "hidden_dims": [64, 32],
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "regret_lr": 1e-3,
                "training": {
                    "batch_size": 32,
                    "buffer_size": 5000,
                    "gamma": 0.99,
                },
            },
        },
        "continuous_scheduler": {
            "seed": 42,
            "total_iterations": 200,
            "eval_every": 25,
            "algorithm": {
                "type": "armac",
                "use_scheduler": True,
                "scheduler": {
                    "hidden": [32, 16],
                    # No k_bins -> continuous mode
                },
                "policy_mixer": {
                    "discrete": False,
                },
                "hidden_dims": [64, 32],
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "regret_lr": 1e-3,
                "scheduler_lr": 1e-4,
                "training": {
                    "batch_size": 32,
                    "buffer_size": 5000,
                    "gamma": 0.99,
                },
            },
        },
        "discrete_scheduler": {
            "seed": 42,
            "total_iterations": 200,
            "eval_every": 25,
            "algorithm": {
                "type": "armac",
                "use_scheduler": True,
                "scheduler": {
                    "hidden": [32, 16],
                    "k_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "temperature": 1.0,
                    "use_gumbel": True,
                },
                "policy_mixer": {
                    "discrete": True,
                    "lambda_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
                },
                "meta_regret": {
                    "K": 5,
                    "decay": 0.99,
                },
                "hidden_dims": [64, 32],
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "regret_lr": 1e-3,
                "scheduler_lr": 1e-4,
                "training": {
                    "batch_size": 32,
                    "buffer_size": 5000,
                    "gamma": 0.99,
                },
            },
        },
    }

    # Run experiments
    all_results = {}

    for exp_name, exp_config in experiments.items():
        result = run_experiment(exp_name, exp_config)
        all_results[exp_name] = result

        # Save individual result
        result_file = output_dir / f"{exp_name}_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    # Generate summary report
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    successful_results = {k: v for k, v in all_results.items() if v["success"]}

    if successful_results:
        # Sort by final exploitability
        sorted_results = sorted(
            successful_results.items(), key=lambda x: x[1]["final_exploitability"]
        )

        print("\nResults (ranked by exploitability):")
        print("-" * 50)
        print(f"{'Rank':<5} {'Experiment':<20} {'Exploitability':<15} {'Time (s)':<10}")
        print("-" * 50)

        for i, (exp_name, result) in enumerate(sorted_results):
            print(
                f"{i + 1:<5} {exp_name:<20} {result['final_exploitability']:<15.6f} "
                f"{result['training_time']:<10.2f}"
            )

        # Best performer details
        best_exp_name, best_result = sorted_results[0]
        print(f"\nBest performer: {best_exp_name}")
        print(f"Final exploitability: {best_result['final_exploitability']:.6f}")
        print(f"Best exploitability: {best_result['best_exploitability']:.6f}")

        # Scheduler comparison
        scheduler_results = {
            k: v for k, v in successful_results.items() if "scheduler" in k
        }
        if len(scheduler_results) >= 2:
            print(f"\nScheduler comparison:")
            for exp_name, result in scheduler_results.items():
                if result["scheduler_stats"]:
                    final_stats = (
                        result["scheduler_stats"][-1]
                        if result["scheduler_stats"]
                        else {}
                    )
                    lambda_mean = final_stats.get("lambda_mean", "N/A")
                    print(f"  {exp_name}: lambda_mean={lambda_mean}")
    else:
        print("No experiments completed successfully!")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
