"""Comprehensive experiment runner for Dual RL Poker project.

This script runs experiments across multiple algorithms (ARMAC, NFSP, PSRO, CFR variants)
with different configurations and logs all results systematically.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.metrics_logger import ExperimentLogger
from utils.logging import get_experiment_logger
from games.game_wrapper import GameWrapper
from eval.openspiel_evaluator import OpenSpielExactEvaluator
from algs.armac import ARMACAlgorithm
from algs.nfsp_runner import NFSPAlgorithm
from algs.psro_runner import PSROAlgorithm
from algs.deep_cfr import DeepCFRAlgorithm
from algs.sd_cfr import SDCFRAlgorithm


class ComprehensiveExperimentRunner:
    """Runs comprehensive experiments across multiple algorithms."""

    def __init__(self, config_path: str, output_dir: str = "results"):
        """Initialize experiment runner.

        Args:
            config_path: Path to configuration file
            output_dir: Output directory for results
        """
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.experiment_logger = get_experiment_logger("comprehensive_experiments")
        self.logger = self.experiment_logger.get_logger()

        # Initialize experiment metrics logger
        self.metrics_logger = ExperimentLogger(
            experiment_name=self.config["name"],
            output_dir=str(self.output_dir),
            log_format="both",
            tensorboard_dir="logs/tensorboard",
        )

        # Results storage
        self.results = []

        self.logger.info("Initialized comprehensive experiment runner")
        self.logger.info(f"Configuration: {self.config['name']}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_algorithms(self, game_name: str) -> Dict[str, Any]:
        """Setup all algorithms for a given game.

        Args:
            game_name: Name of the game

        Returns:
            Dictionary of algorithm instances
        """
        # Create game wrapper
        game_wrapper = GameWrapper(game_name)

        algorithms = {}

        # Setup ARMAC
        try:
            armac_config = self.config["algorithms"].get("armac", {})
            armac_config.update(
                {
                    "game": game_name,
                    "training": {
                        "iterations": self.config["iterations"],
                        "eval_every": self.config["eval_every"],
                        "batch_size": self.config["batch_size"],
                        "gradient_clip": self.config["gradient_clip"],
                    },
                    "network": {
                        "hidden_dims": self.config["hidden_dims"],
                        "dropout": self.config["dropout"],
                    },
                }
            )
            algorithms["armac"] = ARMACAlgorithm(game_wrapper, armac_config)
            self.logger.info("Setup ARMAC algorithm")
        except Exception as e:
            self.logger.error(f"Failed to setup ARMAC: {e}")

        # Setup NFSP
        try:
            nfsp_config = self.config["algorithms"].get("nfsp", {})
            nfsp_config.update(
                {
                    "game": game_name,
                    "training": {
                        "iterations": self.config["iterations"],
                        "eval_every": self.config["eval_every"],
                        "batch_size": nfsp_config.get("batch_size", 256),
                    },
                    "network": {
                        "hidden_dims": nfsp_config.get("hidden_dims", [128, 128]),
                        "dropout": nfsp_config.get("dropout", 0.1),
                    },
                }
            )
            algorithms["nfsp"] = NFSPAlgorithm(game_wrapper, nfsp_config)
            self.logger.info("Setup NFSP algorithm")
        except Exception as e:
            self.logger.error(f"Failed to setup NFSP: {e}")

        # Setup PSRO
        try:
            psro_config = self.config["algorithms"].get("psro", {})
            psro_config.update(
                {
                    "game": game_name,
                    "training": {
                        "iterations": self.config["iterations"],
                        "eval_every": self.config["eval_every"],
                        "batch_size": psro_config.get("batch_size", 256),
                    },
                    "network": {
                        "hidden_dims": psro_config.get("hidden_dims", [128, 128]),
                        "dropout": psro_config.get("dropout", 0.1),
                    },
                }
            )
            algorithms["psro"] = PSROAlgorithm(game_wrapper, psro_config)
            self.logger.info("Setup PSRO algorithm")
        except Exception as e:
            self.logger.error(f"Failed to setup PSRO: {e}")

        # Setup Deep CFR
        try:
            deep_cfr_config = self.config["algorithms"].get("deep_cfr", {})
            deep_cfr_config.update(
                {
                    "game": game_name,
                    "training": {
                        "iterations": self.config["iterations"],
                        "eval_every": self.config["eval_every"],
                        "batch_size": self.config["batch_size"],
                    },
                    "network": {
                        "hidden_dims": self.config["hidden_dims"],
                        "dropout": self.config["dropout"],
                    },
                }
            )
            algorithms["deep_cfr"] = DeepCFRAlgorithm(game_wrapper, deep_cfr_config)
            self.logger.info("Setup Deep CFR algorithm")
        except Exception as e:
            self.logger.error(f"Failed to setup Deep CFR: {e}")

        # Setup SD-CFR
        try:
            sd_cfr_config = self.config["algorithms"].get("sd_cfr", {})
            sd_cfr_config.update(
                {
                    "game": game_name,
                    "training": {
                        "iterations": self.config["iterations"],
                        "eval_every": self.config["eval_every"],
                        "batch_size": self.config["batch_size"],
                    },
                    "network": {
                        "hidden_dims": self.config["hidden_dims"],
                        "dropout": self.config["dropout"],
                    },
                }
            )
            algorithms["sd_cfr"] = SDCFRAlgorithm(game_wrapper, sd_cfr_config)
            self.logger.info("Setup SD-CFR algorithm")
        except Exception as e:
            self.logger.error(f"Failed to setup SD-CFR: {e}")

        return algorithms

    def run_algorithm_experiment(
        self, algorithm_name: str, algorithm, game_name: str, seed: int
    ) -> Dict[str, Any]:
        """Run experiment for a single algorithm.

        Args:
            algorithm_name: Name of the algorithm
            algorithm: Algorithm instance
            game_name: Name of the game
            seed: Random seed

        Returns:
            Experiment results
        """
        self.logger.info(f"Running {algorithm_name} on {game_name} with seed {seed}")

        # Set random seed
        np.random.seed(seed)
        if hasattr(algorithm, "set_seed"):
            algorithm.set_seed(seed)

        # Get algorithm-specific metrics logger
        alg_logger = self.metrics_logger.get_algorithm_logger(algorithm_name)

        start_time = time.time()
        training_results = []

        try:
            # Training loop
            for iteration in range(self.config["iterations"]):
                # Training step
                training_state = algorithm.train_iteration()

                # Log training metrics
                alg_logger.log_training_step(
                    iteration=iteration,
                    losses={
                        "total": training_state.loss,
                        "gradient_norm": training_state.gradient_norm,
                    },
                    metrics=training_state.extra_metrics,
                    runtime=training_state.wall_time,
                )

                # Evaluation
                if iteration % self.config["eval_every"] == 0:
                    eval_results = algorithm.evaluate()

                    # Log evaluation metrics
                    alg_logger.log_evaluation(
                        iteration=iteration,
                        exploitability=eval_results.get("exploitability", 0.0),
                        nash_conv=eval_results.get("nash_conv"),
                        additional_metrics={
                            k: v
                            for k, v in eval_results.items()
                            if k not in ["exploitability", "nash_conv"]
                        },
                    )

                    training_results.append(
                        {
                            "iteration": iteration,
                            "exploitability": eval_results.get("exploitability", 0.0),
                            "nash_conv": eval_results.get("nash_conv"),
                            "wall_time": time.time() - start_time,
                            **eval_results,
                        }
                    )

                    self.logger.info(
                        f"{algorithm_name} {game_name} seed {seed} "
                        f"iter {iteration}: exploitability = {eval_results.get('exploitability', 0.0):.6f}"
                    )

                # Save checkpoint
                if iteration % self.config["save_every"] == 0 and iteration > 0:
                    checkpoint_path = (
                        self.output_dir
                        / "checkpoints"
                        / f"{algorithm_name}_{game_name}_seed{seed}_iter{iteration}.pt"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    algorithm.save_checkpoint(str(checkpoint_path))

        except Exception as e:
            self.logger.error(f"Error in {algorithm_name} training: {e}")
            return {"error": str(e)}

        total_time = time.time() - start_time

        # Get final evaluation
        try:
            final_eval = algorithm.evaluate()
        except Exception as e:
            self.logger.error(f"Error in final evaluation for {algorithm_name}: {e}")
            final_eval = {"exploitability": float("inf")}

        return {
            "algorithm": algorithm_name,
            "game": game_name,
            "seed": seed,
            "final_exploitability": final_eval.get("exploitability", float("inf")),
            "final_nash_conv": final_eval.get("nash_conv"),
            "total_training_time": total_time,
            "training_results": training_results,
            "config": self.config,
        }

    def run_ablation_studies(
        self, game_name: str, base_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run ARMAC ablation studies.

        Args:
            game_name: Name of the game
            base_config: Base ARMAC configuration

        Returns:
            Ablation study results
        """
        ablation_results = []

        # Ablation configurations
        ablation_configs = {
            "no_regret": {
                "disable_regret": True,
                "regret_weight": 0.0,
            },
            "no_critic": {
                "disable_critic": True,
                "critic_lr": 0.0,
            },
            "no_actor": {
                "disable_actor": True,
                "actor_lr": 0.0,
            },
            "fixed_lambda": {
                "lambda_mode": "fixed",
                "regret_weight": 0.1,
            },
        }

        for ablation_name, ablation_overrides in ablation_configs.items():
            self.logger.info(f"Running ARMAC ablation: {ablation_name}")

            # Create ablation config
            ablation_config = base_config.copy()
            ablation_config.update(ablation_overrides)

            # Create game wrapper and algorithm
            try:
                game_wrapper = GameWrapper(game_name)
                algorithm = ARMACAlgorithm(game_wrapper, ablation_config)

                # Run experiment with reduced iterations for ablation
                ablation_iterations = min(self.config["iterations"], 200)
                original_iterations = self.config["iterations"]
                self.config["iterations"] = ablation_iterations

                result = self.run_algorithm_experiment(
                    f"armac_{ablation_name}", algorithm, game_name, 0
                )

                result["ablation_type"] = ablation_name
                ablation_results.append(result)

                # Restore original iterations
                self.config["iterations"] = original_iterations

            except Exception as e:
                self.logger.error(f"Error in ablation {ablation_name}: {e}")

        return ablation_results

    def run_scalability_experiments(self) -> List[Dict[str, Any]]:
        """Run scalability experiments on larger games.

        Returns:
            Scalability experiment results
        """
        scalability_results = []

        # Test on No-Limit Leduc (if available)
        try:
            self.logger.info("Running No-Limit Leduc experiments")

            # Create no-limit leduc config
            nl_leduc_config = self.config.copy()
            nl_leduc_config["game"] = {
                "name": "leduc_poker",
                "betting_abstraction": "no_limit",
            }

            for algorithm_name in ["deep_cfr", "nfsp", "psro"]:  # Skip tabular CFR
                try:
                    game_wrapper = GameWrapper(
                        "leduc_poker", {"betting_abstraction": "no_limit"}
                    )

                    # Get algorithm config
                    alg_config = nl_leduc_config["algorithms"].get(algorithm_name, {})
                    alg_config.update(
                        {
                            "game": "leduc_poker",
                            "training": {
                                "iterations": min(
                                    self.config["iterations"], 200
                                ),  # Reduced for scalability
                                "eval_every": self.config["eval_every"],
                            },
                        }
                    )

                    # Create algorithm
                    if algorithm_name == "deep_cfr":
                        algorithm = DeepCFRAlgorithm(game_wrapper, alg_config)
                    elif algorithm_name == "nfsp":
                        algorithm = NFSPAlgorithm(game_wrapper, alg_config)
                    elif algorithm_name == "psro":
                        algorithm = PSROAlgorithm(game_wrapper, alg_config)
                    else:
                        continue

                    result = self.run_algorithm_experiment(
                        f"{algorithm_name}_nl_leduc", algorithm, "nl_leduc", 0
                    )
                    result["game_variant"] = "no_limit_leduc"
                    scalability_results.append(result)

                except Exception as e:
                    self.logger.error(
                        f"Error in scalability experiment for {algorithm_name}: {e}"
                    )

        except Exception as e:
            self.logger.error(f"Error in scalability experiments: {e}")

        return scalability_results

    def run_full_experiment(self):
        """Run the full comprehensive experiment."""
        self.logger.info("Starting comprehensive experiment")

        games = ["kuhn_poker", "leduc_poker"]
        seeds = [0, 1, 2, 3, 4]  # 5 seeds per algorithm-game combination

        all_results = []

        # Main experiments
        for game_name in games:
            self.logger.info(f"Running experiments on {game_name}")

            # Setup algorithms
            algorithms = self.setup_algorithms(game_name)

            for algorithm_name, algorithm in algorithms.items():
                for seed in seeds:
                    result = self.run_algorithm_experiment(
                        algorithm_name, algorithm, game_name, seed
                    )
                    all_results.append(result)

                    # Save intermediate results
                    self.save_results(all_results, "intermediate")

        # Ablation studies
        self.logger.info("Running ARMAC ablation studies")
        try:
            game_wrapper = GameWrapper("kuhn_poker")
            base_armac_config = self.config["algorithms"].get("armac", {})
            base_armac_config.update(
                {
                    "game": "kuhn_poker",
                    "training": {
                        "iterations": min(self.config["iterations"], 200),
                        "eval_every": self.config["eval_every"],
                    },
                }
            )

            ablation_results = self.run_ablation_studies(
                "kuhn_poker", base_armac_config
            )
            all_results.extend(ablation_results)

        except Exception as e:
            self.logger.error(f"Error in ablation studies: {e}")

        # Scalability experiments
        self.logger.info("Running scalability experiments")
        try:
            scalability_results = self.run_scalability_experiments()
            all_results.extend(scalability_results)

        except Exception as e:
            self.logger.error(f"Error in scalability experiments: {e}")

        # Save final results
        self.save_results(all_results, "final")

        # Generate summary report
        self.generate_summary_report(all_results)

        # Close metrics logger
        self.metrics_logger.close_all()

        self.logger.info("Comprehensive experiment completed")

    def save_results(self, results: List[Dict[str, Any]], suffix: str = "final"):
        """Save experiment results to CSV.

        Args:
            results: List of experiment results
            suffix: Suffix for output filename
        """
        if not results:
            self.logger.warning("No results to save")
            return

        # Flatten results for CSV
        flattened_results = []
        for result in results:
            if "error" in result:
                flattened_results.append(
                    {
                        "algorithm": result.get("algorithm", "unknown"),
                        "game": result.get("game", "unknown"),
                        "seed": result.get("seed", -1),
                        "error": result["error"],
                    }
                )
            else:
                base_row = {
                    "algorithm": result["algorithm"],
                    "game": result["game"],
                    "seed": result["seed"],
                    "final_exploitability": result["final_exploitability"],
                    "final_nash_conv": result.get("final_nash_conv"),
                    "total_training_time": result["total_training_time"],
                }

                # Add ablation info if present
                if "ablation_type" in result:
                    base_row["ablation_type"] = result["ablation_type"]

                # Add scalability info if present
                if "game_variant" in result:
                    base_row["game_variant"] = result["game_variant"]

                flattened_results.append(base_row)

        # Save to CSV
        df = pd.DataFrame(flattened_results)
        output_path = self.output_dir / f"comprehensive_results_{suffix}.csv"
        df.to_csv(output_path, index=False)

        self.logger.info(f"Saved {len(flattened_results)} results to {output_path}")

    def generate_summary_report(self, results: List[Dict[str, Any]]):
        """Generate a summary report of the experiments.

        Args:
            results: List of experiment results
        """
        self.logger.info("Generating summary report")

        # Convert to DataFrame for analysis
        flattened_results = []
        for result in results:
            if "error" not in result:
                base_row = {
                    "algorithm": result["algorithm"],
                    "game": result["game"],
                    "final_exploitability": result["final_exploitability"],
                    "total_training_time": result["total_training_time"],
                }
                if "ablation_type" in result:
                    base_row["algorithm"] = (
                        f"{result['algorithm']}_{result['ablation_type']}"
                    )
                flattened_results.append(base_row)

        if not flattened_results:
            self.logger.warning("No valid results for summary report")
            return

        df = pd.DataFrame(flattened_results)

        # Generate summary statistics
        summary_path = self.output_dir / "experiment_summary.txt"

        with open(summary_path, "w") as f:
            f.write("Comprehensive Experiment Summary\n")
            f.write("=" * 40 + "\n\n")

            # Overall statistics
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Algorithms tested: {df['algorithm'].nunique()}\n")
            f.write(f"Games tested: {df['game'].nunique()}\n\n")

            # Performance by algorithm and game
            f.write("Performance Summary (mean ± std):\n")
            f.write("-" * 30 + "\n")

            for game in df["game"].unique():
                game_df = df[df["game"] == game]
                f.write(f"\n{game}:\n")

                for algorithm in game_df["algorithm"].unique():
                    alg_df = game_df[game_df["algorithm"] == algorithm]
                    mean_exp = alg_df["final_exploitability"].mean()
                    std_exp = alg_df["final_exploitability"].std()
                    mean_time = alg_df["total_training_time"].mean()

                    f.write(
                        f"  {algorithm}: {mean_exp:.6f} ± {std_exp:.6f} mbb/h ({mean_time:.1f}s)\n"
                    )

            # Best performing algorithms
            f.write("\nBest Performing Algorithms:\n")
            f.write("-" * 25 + "\n")

            for game in df["game"].unique():
                game_df = df[df["game"] == game]
                best_alg = game_df.loc[game_df["final_exploitability"].idxmin()]
                f.write(
                    f"{game}: {best_alg['algorithm']} ({best_alg['final_exploitability']:.6f})\n"
                )

        self.logger.info(f"Summary report saved to {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Dual RL Poker experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comprehensive",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiments
    runner = ComprehensiveExperimentRunner(args.config, args.output_dir)
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
