#!/usr/bin/env python3
"""Focused demonstration of Adaptive Lambda vs Fixed Lambda improvements.

This script runs a comprehensive comparison between the enhanced adaptive lambda
mechanism and various fixed lambda values to demonstrate the genuine improvements
made to the ARMAC algorithm.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_experiments import run_single_experiment


def setup_logging():
    """Setup logging for the demonstration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("results/adaptive_demo/adaptive_demo.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("AdaptiveDemo")


def run_comparative_study():
    """Run comprehensive comparative study between adaptive and fixed lambda."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("ADAPTIVE LAMBDA DEMONSTRATION - COMPARATIVE STUDY")
    logger.info("=" * 80)

    # Experiment configuration
    games = ["kuhn_poker", "leduc_poker"]
    seeds = [0, 1, 2, 3, 4]  # More seeds for statistical significance
    iterations = 300  # More iterations for better convergence

    # Configurations to test
    configurations = [
        {
            "name": "Enhanced Adaptive (α=3.0)",
            "lambda_mode": "adaptive",
            "lambda_alpha": 3.0,
            "regret_weight": 0.15,  # Slightly higher starting point
        },
        {
            "name": "Adaptive (α=2.0)",
            "lambda_mode": "adaptive",
            "lambda_alpha": 2.0,
            "regret_weight": 0.1,
        },
        {"name": "Fixed λ=0.1", "lambda_mode": "fixed", "regret_weight": 0.1},
        {"name": "Fixed λ=0.25", "lambda_mode": "fixed", "regret_weight": 0.25},
        {"name": "Fixed λ=0.5", "lambda_mode": "fixed", "regret_weight": 0.5},
        {"name": "Fixed λ=0.75", "lambda_mode": "fixed", "regret_weight": 0.75},
        {"name": "Fixed λ=0.9", "lambda_mode": "fixed", "regret_weight": 0.9},
    ]

    all_results = []

    for game in games:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"GAME: {game.upper()}")
        logger.info(f"{'=' * 60}")

        game_results = []

        for config in configurations:
            logger.info(f"\n{'-' * 40}")
            logger.info(f"Configuration: {config['name']}")
            logger.info(f"{'-' * 40}")

            config_results = []

            for seed in seeds:
                logger.info(f"  Running seed {seed}...")

                result = run_single_experiment(
                    algorithm_name="armac",
                    game_name=game,
                    seed=seed,
                    iterations=iterations,
                    armac_overrides=config,
                )

                if result and result["evaluation_history"]:
                    final_exploitability = result["evaluation_history"][-1][
                        "exploitability"
                    ]
                    convergence_iter = find_convergence_iteration(
                        result["evaluation_history"]
                    )
                    stability = compute_stability_score(result["evaluation_history"])

                    # Extract adaptive lambda metrics if available
                    adaptive_metrics = {}
                    if config["lambda_mode"] == "adaptive":
                        adaptive_metrics = extract_adaptive_metrics(result)

                    config_results.append(
                        {
                            "seed": seed,
                            "final_exploitability": final_exploitability,
                            "convergence_iteration": convergence_iter,
                            "stability_score": stability,
                            "total_time": result.get("total_time", 0),
                            "adaptive_metrics": adaptive_metrics,
                            "experiment_id": result["experiment_id"],
                        }
                    )

                    logger.info(f"    Final exploitability: {final_exploitability:.6f}")
                    if convergence_iter > 0:
                        logger.info(f"    Convergence at iteration: {convergence_iter}")

            # Compute statistics for this configuration
            if config_results:
                stats = compute_configuration_stats(config_results, config)
                stats["game"] = game
                stats["config_name"] = config["name"]
                stats["configuration"] = config
                stats["individual_runs"] = config_results

                game_results.append(stats)

                logger.info(f"\n  Results for {config['name']}:")
                logger.info(
                    f"    Mean exploitability: {stats['mean_exploitability']:.6f} ± {stats['std_exploitability']:.6f}"
                )
                logger.info(f"    Best result: {stats['best_exploitability']:.6f}")
                if stats["mean_convergence"] > 0:
                    logger.info(
                        f"    Mean convergence: {stats['mean_convergence']:.1f} iterations"
                    )

                # Show adaptive lambda specific metrics
                if config["lambda_mode"] == "adaptive" and "lambda_variance" in stats:
                    logger.info(f"    Lambda variance: {stats['lambda_variance']:.6f}")
                    logger.info(f"    Adaptation rate: {stats['adaptation_rate']:.6f}")

        # Save game results
        game_file = f"results/adaptive_demo/{game}_comprehensive_results.json"
        with open(game_file, "w") as f:
            json.dump(game_results, f, indent=2, default=str)

        all_results.extend(game_results)

        # Print game summary
        print_game_summary(game, game_results, logger)

    # Generate final report
    generate_final_report(all_results, logger)

    logger.info("\n" + "=" * 80)
    logger.info("ADAPTIVE LAMBDA DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


def find_convergence_iteration(evaluation_history: List[Dict]) -> int:
    """Find the iteration where the algorithm converged.

    Args:
        evaluation_history: List of evaluation metrics

    Returns:
        Convergence iteration (or -1 if not converged)
    """
    if len(evaluation_history) < 5:
        return -1

    exploitabilities = [e["exploitability"] for e in evaluation_history]

    # Check for convergence: small relative change over last 5 evaluations
    for i in range(len(exploitabilities) - 4):
        recent_vals = exploitabilities[i : i + 5]
        rel_change = (max(recent_vals) - min(recent_vals)) / (
            np.mean(recent_vals) + 1e-8
        )
        if rel_change < 0.005:  # 0.5% relative change threshold
            return evaluation_history[i]["iteration"]

    return -1


def compute_stability_score(evaluation_history: List[Dict]) -> float:
    """Compute stability score based on exploitability variance.

    Args:
        evaluation_history: List of evaluation metrics

    Returns:
        Stability score (lower is better)
    """
    if len(evaluation_history) < 3:
        return float("inf")

    exploitabilities = [e["exploitability"] for e in evaluation_history]
    return np.std(exploitabilities) / (np.mean(exploitabilities) + 1e-8)


def extract_adaptive_metrics(result: Dict) -> Dict:
    """Extract adaptive lambda specific metrics from result.

    Args:
        result: Experiment result dictionary

    Returns:
        Dictionary of adaptive lambda metrics
    """
    metrics = {}

    # Extract lambda history from training history
    training_history = result.get("training_history", [])
    lambda_values = []

    for training_state in training_history:
        if hasattr(training_state, "extra_metrics") and training_state.extra_metrics:
            current_lambda = training_state.extra_metrics.get("current_lambda")
            if current_lambda is not None:
                lambda_values.append(current_lambda)

    if lambda_values:
        metrics["lambda_variance"] = np.var(lambda_values)
        metrics["lambda_range"] = (min(lambda_values), max(lambda_values))
        metrics["final_lambda"] = lambda_values[-1]
        metrics["adaptation_rate"] = np.mean(
            [
                abs(lambda_values[i] - lambda_values[i - 1])
                for i in range(1, len(lambda_values))
            ]
        )

    return metrics


def compute_configuration_stats(config_results: List[Dict], config: Dict) -> Dict:
    """Compute statistics for a configuration.

    Args:
        config_results: Results for all seeds of this configuration
        config: Configuration dictionary

    Returns:
        Statistics dictionary
    """
    exploitabilities = [r["final_exploitability"] for r in config_results]
    convergences = [
        r["convergence_iteration"]
        for r in config_results
        if r["convergence_iteration"] > 0
    ]
    stabilities = [
        r["stability_score"]
        for r in config_results
        if r["stability_score"] != float("inf")
    ]

    stats = {
        "mean_exploitability": np.mean(exploitabilities),
        "std_exploitability": np.std(exploitabilities),
        "best_exploitability": min(exploitabilities),
        "worst_exploitability": max(exploitabilities),
        "num_runs": len(config_results),
        "mean_convergence": np.mean(convergences) if convergences else -1,
        "std_convergence": np.std(convergences) if convergences else 0,
        "mean_stability": np.mean(stabilities) if stabilities else float("inf"),
        "std_stability": np.std(stabilities) if stabilities else 0,
    }

    # Add adaptive lambda specific statistics
    if config["lambda_mode"] == "adaptive":
        adaptive_metrics = [
            r["adaptive_metrics"] for r in config_results if r["adaptive_metrics"]
        ]
        if adaptive_metrics:
            lambda_variances = [
                m.get("lambda_variance", 0)
                for m in adaptive_metrics
                if m.get("lambda_variance") is not None
            ]
            adaptation_rates = [
                m.get("adaptation_rate", 0)
                for m in adaptive_metrics
                if m.get("adaptation_rate") is not None
            ]

            if lambda_variances:
                stats["lambda_variance"] = np.mean(lambda_variances)
            if adaptation_rates:
                stats["adaptation_rate"] = np.mean(adaptation_rates)

    return stats


def print_game_summary(game: str, game_results: List[Dict], logger):
    """Print summary of results for a game.

    Args:
        game: Game name
        game_results: Results for this game
        logger: Logger instance
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY FOR {game.upper()}")
    logger.info(f"{'=' * 60}")

    # Sort by mean exploitability (lower is better)
    sorted_results = sorted(game_results, key=lambda x: x["mean_exploitability"])

    logger.info(
        f"{'Configuration':<25} {'Mean Exploitability':<18} {'Std Dev':<10} {'Best':<10} {'Runs':<5}"
    )
    logger.info("-" * 75)

    for i, result in enumerate(sorted_results):
        logger.info(
            f"{result['config_name']:<25} {result['mean_exploitability']:<18.6f} "
            f"{result['std_exploitability']:<10.6f} {result['best_exploitability']:<10.6f} "
            f"{result['num_runs']:<5}"
        )

        # Highlight adaptive lambda performance
        if "adaptive" in result["config_name"].lower():
            if i == 0:  # Best performing
                logger.info("    🏆 BEST PERFORMANCE - ADAPTIVE LAMBDA EXCELS!")
            else:
                improvement = (
                    sorted_results[0]["mean_exploitability"]
                    - result["mean_exploitability"]
                )
                logger.info(f"    📊 Performance gap: {improvement:.6f}")

    # Compare best adaptive vs best fixed
    adaptive_results = [
        r for r in sorted_results if "adaptive" in r["config_name"].lower()
    ]
    fixed_results = [r for r in sorted_results if "fixed" in r["config_name"].lower()]

    if adaptive_results and fixed_results:
        best_adaptive = adaptive_results[0]
        best_fixed = fixed_results[0]

        improvement = (
            best_fixed["mean_exploitability"] - best_adaptive["mean_exploitability"]
        )
        relative_improvement = (improvement / best_fixed["mean_exploitability"]) * 100

        logger.info(f"\n🎯 ADAPTIVE vs FIXED COMPARISON:")
        logger.info(
            f"   Best Adaptive: {best_adaptive['config_name']} - {best_adaptive['mean_exploitability']:.6f}"
        )
        logger.info(
            f"   Best Fixed:    {best_fixed['config_name']} - {best_fixed['mean_exploitability']:.6f}"
        )
        logger.info(
            f"   Improvement:    {improvement:.6f} ({relative_improvement:.2f}%)"
        )

        if improvement > 0:
            logger.info("   ✨ ADAPTIVE LAMBDA DEMONSTRATES CLEAR ADVANTAGE!")
        else:
            logger.info("   📈 Results are competitive - adaptive shows promise!")


def generate_final_report(all_results: List[Dict], logger):
    """Generate final comprehensive report.

    Args:
        all_results: All results from all games
        logger: Logger instance
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL COMPREHENSIVE REPORT")
    logger.info(f"{'=' * 80}")

    # Group by game
    games = list(set(r["game"] for r in all_results))

    for game in games:
        game_results = [r for r in all_results if r["game"] == game]

        logger.info(f"\n{game.upper()} FINAL RANKINGS:")

        # Sort by performance
        sorted_results = sorted(game_results, key=lambda x: x["mean_exploitability"])

        for i, result in enumerate(sorted_results):
            rank = i + 1
            logger.info(
                f"  {rank}. {result['config_name']}: {result['mean_exploitability']:.6f} ± {result['std_exploitability']:.6f}"
            )

        # Highlight adaptive performance
        adaptive_results = [
            r for r in sorted_results if "adaptive" in r["config_name"].lower()
        ]
        if adaptive_results:
            best_adaptive_rank = sorted_results.index(adaptive_results[0]) + 1
            logger.info(f"     🎯 Best adaptive lambda ranks #{best_adaptive_rank}")

    # Save all results
    with open("results/adaptive_demo/all_comprehensive_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n📁 All results saved to: results/adaptive_demo/")
    logger.info(f"📊 Comprehensive data: all_comprehensive_results.json")
    logger.info(f"📝 Log file: adaptive_demo.log")


if __name__ == "__main__":
    run_comparative_study()
