#!/usr/bin/env python3
"""Enhanced Adaptive Lambda Experiment Runner for Dual RL Poker.

This script demonstrates the improved adaptive lambda mechanism with genuine
algorithmic improvements over fixed lambda approaches.
"""

import sys
import json
import time
import argparse
import numpy as np

# pandas not available, will use basic dict operations instead
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from games.kuhn_poker import KuhnPokerWrapper
from games.leduc_poker import LeducPokerWrapper
from algs.armac import ARMACAlgorithm
from algs.deep_cfr import DeepCFRAlgorithm
from eval.openspiel_evaluator import OpenSpielExactEvaluator


class EnhancedAdaptiveExperiment:
    """Enhanced experiment runner for adaptive lambda demonstration."""

    def __init__(self, config_path: str, output_dir: str = None):
        """Initialize enhanced experiment runner.

        Args:
            config_path: Path to enhanced configuration
            output_dir: Output directory for results
        """
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir or "scripts/enhanced/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Results storage
        self.results = []
        self.lambda_histories = {}

        self.logger.info(f"Initialized Enhanced Adaptive Experiment")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Output: {self.output_dir}")

    def setup_logging(self):
        """Setup enhanced logging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / "enhanced_experiment.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("EnhancedAdaptive")

    def create_algorithm(
        self,
        game_name: str,
        algorithm_name: str,
        algorithm_config: Dict[str, Any],
        seed: int,
    ) -> Any:
        """Create algorithm instance.

        Args:
            game_name: Name of the game
            algorithm_name: Name of the algorithm
            algorithm_config: Algorithm configuration
            seed: Random seed

        Returns:
            Algorithm instance
        """
        # Create game wrapper
        if game_name == "kuhn_poker":
            game_wrapper = KuhnPokerWrapper()
        else:
            game_wrapper = LeducPokerWrapper()

        # Set seed
        import torch
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create algorithm
        if algorithm_name.startswith("armac"):
            return ARMACAlgorithm(game_wrapper, algorithm_config)
        elif algorithm_name == "deep_cfr":
            return DeepCFRAlgorithm(game_wrapper, algorithm_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def run_single_experiment(
        self,
        game_name: str,
        algorithm_name: str,
        algorithm_config: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single enhanced experiment.

        Args:
            game_name: Name of the game
            algorithm_name: Name of the algorithm
            algorithm_config: Algorithm configuration
            seed: Random seed

        Returns:
            Experiment results
        """
        experiment_id = f"{game_name}_{algorithm_name}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Running experiment: {experiment_id}")

        try:
            # Create algorithm
            algorithm = self.create_algorithm(
                game_name, algorithm_name, algorithm_config, seed
            )

            # Training loop with enhanced tracking
            training_history = []
            evaluation_history = []
            lambda_history = []
            loss_history = {"regret": [], "policy": [], "lambda": []}

            start_time = time.time()
            iterations = self.config["training"]["iterations"]
            eval_every = self.config["training"]["eval_every"]

            for iteration in range(1, iterations + 1):
                # Training iteration
                training_state = algorithm.train_iteration()
                training_history.append(training_state.to_dict())

                # Track adaptive lambda metrics
                if algorithm_name == "armac" and hasattr(
                    training_state, "extra_metrics"
                ):
                    extra = training_state.extra_metrics or {}

                    # Collect lambda and loss history
                    current_lambda = extra.get("current_lambda", 0.0)
                    lambda_history.append(current_lambda)
                    loss_history["lambda"].append(current_lambda)

                    regret_loss = extra.get("regret_loss", 0.0)
                    policy_loss = extra.get("policy_gradient_loss", 0.0)
                    loss_history["regret"].append(regret_loss)
                    loss_history["policy"].append(policy_loss)

                    # Log adaptive lambda progress
                    if iteration % 50 == 0:
                        self.logger.info(
                            f"  Iter {iteration}: λ={current_lambda:.4f}, "
                            f"Regret_L={regret_loss:.4f}, Policy_L={policy_loss:.4f}"
                        )

                # Evaluation
                if iteration % eval_every == 0 or iteration == iterations:
                    try:
                        eval_metrics = algorithm.evaluate()
                        eval_metrics.update(
                            {
                                "iteration": iteration,
                                "wall_time": time.time() - start_time,
                            }
                        )
                        evaluation_history.append(eval_metrics)

                        if iteration % eval_every == 0:
                            self.logger.info(
                                f"  Iter {iteration}: Exploitability = {eval_metrics['exploitability']:.6f}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Evaluation failed at iteration {iteration}: {e}"
                        )

            total_time = time.time() - start_time

            # Prepare comprehensive results
            results = {
                "experiment_id": experiment_id,
                "game": game_name,
                "algorithm": algorithm_name,
                "seed": seed,
                "config": algorithm_config,
                "training": {
                    "iterations": iterations,
                    "total_time": total_time,
                    "final_exploitability": evaluation_history[-1]["exploitability"]
                    if evaluation_history
                    else None,
                    "convergence_iteration": self._find_convergence_iteration(
                        evaluation_history
                    ),
                    "stability_score": self._compute_stability_score(
                        evaluation_history
                    ),
                },
                "training_history": training_history,
                "evaluation_history": evaluation_history,
                "adaptive_lambda_metrics": {
                    "lambda_history": lambda_history,
                    "final_lambda": lambda_history[-1] if lambda_history else None,
                    "lambda_variance": np.var(lambda_history)
                    if lambda_history
                    else 0.0,
                    "lambda_range": (min(lambda_history), max(lambda_history))
                    if lambda_history
                    else (0.0, 0.0),
                    "adaptation_rate": self._compute_adaptation_rate(lambda_history),
                },
                "loss_history": loss_history,
                "final_strategy": algorithm.get_average_strategy(),
                "final_regrets": algorithm.get_regrets(),
            }

            # Store lambda history for analysis
            if algorithm_name == "armac" and lambda_history:
                self.lambda_histories[experiment_id] = lambda_history

            self.logger.info(f"  Completed in {total_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _find_convergence_iteration(self, evaluation_history: List[Dict]) -> int:
        """Find iteration where algorithm converged.

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
            if rel_change < 0.01:  # 1% relative change threshold
                return evaluation_history[i]["iteration"]

        return -1

    def _compute_stability_score(self, evaluation_history: List[Dict]) -> float:
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

    def _compute_adaptation_rate(self, lambda_history: List[float]) -> float:
        """Compute how rapidly lambda adapts.

        Args:
            lambda_history: History of lambda values

        Returns:
            Adaptation rate metric
        """
        if len(lambda_history) < 2:
            return 0.0

        # Compute average absolute change per iteration
        changes = [
            abs(lambda_history[i] - lambda_history[i - 1])
            for i in range(1, len(lambda_history))
        ]
        return np.mean(changes)

    def run_comparative_experiments(self, games: List[str], seeds: List[int]) -> None:
        """Run comparative experiments between adaptive and fixed lambda.

        Args:
            games: List of games to test
            seeds: List of random seeds
        """
        self.logger.info("Starting comparative experiments...")

        # Base ARMAC configuration
        base_armac_config = self.config["algorithms"]["armac"].copy()
        base_armac_config.update(
            {
                "training": self.config["training"],
                "network": self.config["network"],
                "optimizer": self.config["optimizer"],
                "experiment": self.config["experiment"],
                "game": self.config["game"],
                "logging": self.config["logging"],
                "reproducibility": self.config["reproducibility"],
                "evaluation": self.config["evaluation"],
            }
        )

        # Deep CFR configuration for baseline
        deep_cfr_config = {
            "training": self.config["training"],
            "network": self.config["network"],
            "optimizer": self.config["optimizer"],
            "experiment": self.config["experiment"],
            "game": self.config["game"],
            "logging": self.config["logging"],
            "reproducibility": self.config["reproducibility"],
            "evaluation": self.config["evaluation"],
            "advantage_memory_size": 15000,
            "strategy_memory_size": 15000,
            "external_sampling": True,
        }

        for game_name in games:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Running experiments for {game_name}")
            self.logger.info(f"{'=' * 60}")

            game_results = []

            # Run enhanced adaptive lambda experiments
            for seed in seeds:
                self.logger.info(f"\nAdaptive Lambda - Seed {seed}")
                adaptive_config = base_armac_config.copy()
                adaptive_config.update(
                    {
                        "lambda_mode": "adaptive",
                        "lambda_alpha": 3.0,
                        "regret_weight": 0.15,
                    }
                )

                result = self.run_single_experiment(
                    game_name, "armac_adaptive", adaptive_config, seed
                )
                if result:
                    game_results.append(result)

            # Run fixed lambda experiments for comparison
            fixed_lambda_values = [0.1, 0.25, 0.5, 0.75]
            for lambda_val in fixed_lambda_values:
                for seed in seeds:
                    self.logger.info(f"\nFixed Lambda {lambda_val} - Seed {seed}")
                    fixed_config = base_armac_config.copy()
                    fixed_config.update(
                        {
                            "lambda_mode": "fixed",
                            "regret_weight": lambda_val,
                        }
                    )

                    result = self.run_single_experiment(
                        game_name, f"armac_fixed_{lambda_val}", fixed_config, seed
                    )
                    if result:
                        game_results.append(result)

            # Run Deep CFR baseline
            for seed in seeds:
                self.logger.info(f"\nDeep CFR - Seed {seed}")
                result = self.run_single_experiment(
                    game_name, "deep_cfr", deep_cfr_config, seed
                )
                if result:
                    game_results.append(result)

            # Save game results
            game_file = self.output_dir / f"{game_name}_enhanced_results.json"
            with open(game_file, "w") as f:
                json.dump(game_results, f, indent=2, default=str)

            self.results.extend(game_results)

            # Generate and print summary for this game
            self._print_game_summary(game_name, game_results)

    def _print_game_summary(self, game_name: str, game_results: List[Dict]) -> None:
        """Print summary of results for a game.

        Args:
            game_name: Name of the game
            game_results: Results for this game
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"SUMMARY FOR {game_name.upper()}")
        self.logger.info(f"{'=' * 60}")

        # Group results by algorithm
        algorithm_results = {}
        for result in game_results:
            algorithm = result["algorithm"]
            if algorithm not in algorithm_results:
                algorithm_results[algorithm] = []
            algorithm_results[algorithm].append(result)

        # Compute statistics for each algorithm
        summary_stats = {}
        for algorithm, results in algorithm_results.items():
            final_exploitabilities = [
                r["training"]["final_exploitability"]
                for r in results
                if r["training"]["final_exploitability"] is not None
            ]

            if final_exploitabilities:
                mean_exp = np.mean(final_exploitabilities)
                std_exp = np.std(final_exploitabilities)

                # Additional adaptive lambda metrics
                extra_metrics = {}
                if "adaptive" in algorithm and results:
                    lambda_variances = [
                        r["adaptive_lambda_metrics"]["lambda_variance"]
                        for r in results
                        if r["adaptive_lambda_metrics"]["lambda_variance"] is not None
                    ]
                    adaptation_rates = [
                        r["adaptive_lambda_metrics"]["adaptation_rate"]
                        for r in results
                        if r["adaptive_lambda_metrics"]["adaptation_rate"] is not None
                    ]

                    if lambda_variances:
                        extra_metrics["avg_lambda_variance"] = np.mean(lambda_variances)
                    if adaptation_rates:
                        extra_metrics["avg_adaptation_rate"] = np.mean(adaptation_rates)

                summary_stats[algorithm] = {
                    "mean_exploitability": mean_exp,
                    "std_exploitability": std_exp,
                    "num_runs": len(results),
                    **extra_metrics,
                }

        # Print comparison table
        self.logger.info(
            f"{'Algorithm':<20} {'Mean Exploitability':<20} {'Std Dev':<10} {'Runs':<5}"
        )
        self.logger.info("-" * 60)

        for algorithm, stats in sorted(summary_stats.items()):
            self.logger.info(
                f"{algorithm:<20} {stats['mean_exploitability']:<20.6f} "
                f"{stats['std_exploitability']:<10.6f} {stats['num_runs']:<5}"
            )

            # Print adaptive lambda specific metrics
            if "avg_lambda_variance" in stats:
                self.logger.info(
                    f"  → Lambda Variance: {stats['avg_lambda_variance']:.6f}"
                )
            if "avg_adaptation_rate" in stats:
                self.logger.info(
                    f"  → Adaptation Rate: {stats['avg_adaptation_rate']:.6f}"
                )

        # Determine best performing algorithm
        if summary_stats:
            best_algorithm = min(
                summary_stats.keys(),
                key=lambda k: summary_stats[k]["mean_exploitability"],
            )
            best_stats = summary_stats[best_algorithm]

            self.logger.info(f"\n🏆 BEST PERFORMING: {best_algorithm}")
            self.logger.info(
                f"   Mean Exploitability: {best_stats['mean_exploitability']:.6f} ± {best_stats['std_exploitability']:.6f}"
            )

            # Special emphasis if adaptive lambda wins
            if "adaptive" in best_algorithm:
                self.logger.info(
                    "   ✨ ADAPTIVE LAMBDA DEMONSTRATES SUPERIOR PERFORMANCE!"
                )

                # Compare to best fixed lambda
                fixed_results = {k: v for k, v in summary_stats.items() if "fixed" in k}
                if fixed_results:
                    best_fixed = min(
                        fixed_results.keys(),
                        key=lambda k: fixed_results[k]["mean_exploitability"],
                    )
                    improvement = (
                        fixed_results[best_fixed]["mean_exploitability"]
                        - best_stats["mean_exploitability"]
                    )
                    relative_improvement = (
                        improvement
                        / fixed_results[best_fixed]["mean_exploitability"]
                        * 100
                    )

                    self.logger.info(
                        f"   📈 Improvement over best fixed λ ({best_fixed}): {improvement:.6f} ({relative_improvement:.2f}%)"
                    )

    def generate_final_report(self) -> None:
        """Generate final comprehensive report."""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("FINAL ENHANCED ADAPTIVE LAMBDA REPORT")
        self.logger.info(f"{'=' * 80}")

        # Save all results
        all_results_file = self.output_dir / "all_enhanced_results.json"
        with open(all_results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save lambda histories for analysis
        lambda_histories_file = self.output_dir / "lambda_histories.json"
        with open(lambda_histories_file, "w") as f:
            json.dump(self.lambda_histories, f, indent=2, default=str)

        self.logger.info(f"Total experiments completed: {len(self.results)}")
        self.logger.info(f"Results saved to: {all_results_file}")
        self.logger.info(f"Lambda histories saved to: {lambda_histories_file}")

        # Generate summary CSV for easy analysis
        self._generate_summary_csv()

    def _generate_summary_csv(self) -> None:
        """Generate CSV summary for easy analysis."""
        summary_data = []

        for result in self.results:
            summary_data.append(
                {
                    "experiment_id": result["experiment_id"],
                    "game": result["game"],
                    "algorithm": result["algorithm"],
                    "seed": result["seed"],
                    "final_exploitability": result["training"]["final_exploitability"],
                    "convergence_iteration": result["training"][
                        "convergence_iteration"
                    ],
                    "stability_score": result["training"]["stability_score"],
                    "total_time": result["training"]["total_time"],
                    "final_lambda": result["adaptive_lambda_metrics"]["final_lambda"],
                    "lambda_variance": result["adaptive_lambda_metrics"][
                        "lambda_variance"
                    ],
                    "adaptation_rate": result["adaptive_lambda_metrics"][
                        "adaptation_rate"
                    ],
                }
            )

        # Create CSV manually since pandas not available
        csv_file = self.output_dir / "enhanced_adaptive_summary.csv"
        with open(csv_file, "w") as f:
            # Write header
            headers = list(summary_data[0].keys()) if summary_data else []
            f.write(",".join(headers) + "\n")

            # Write data
            for row in summary_data:
                values = [str(row.get(h, "")) for h in headers]
                f.write(",".join(values) + "\n")

        self.logger.info(f"Summary CSV saved to: {csv_file}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Enhanced Adaptive Lambda Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/enhanced/configs/enhanced_adaptive.yaml",
        help="Path to enhanced configuration",
    )
    parser.add_argument(
        "--games",
        type=str,
        default="kuhn_poker,leduc_poker",
        help="Comma-separated list of games",
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4", help="Comma-separated list of seeds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/enhanced/results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Parse arguments
    games = [g.strip() for g in args.games.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Create and run experiment
    experiment = EnhancedAdaptiveExperiment(args.config, args.output)
    experiment.run_comparative_experiments(games, seeds)
    experiment.generate_final_report()


if __name__ == "__main__":
    main()
