#!/usr/bin/env python3
"""
Comprehensive scheduler experiment runner for ARMAC framework.

This script runs experiments comparing different scheduler configurations:
- Fixed lambda baseline
- Adaptive lambda (legacy)
- Continuous scheduler
- Discrete scheduler with meta-regret
- Various ablation studies

Results are saved with comprehensive logging and visualization.
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import get_experiment_logger
from games.game_factory import GameFactory
from algs.armac import ARMACAlgorithm
from eval.openspiel_evaluator import OpenSpielExactEvaluator
from analysis.auto_generator import ManifestAutoGenerator


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    description: str
    algorithm_config: Dict[str, Any]
    game_config: Dict[str, Any]
    training_config: Dict[str, Any]
    eval_config: Dict[str, Any]
    seed: int
    total_iterations: int
    eval_every: int
    output_dir: str


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    experiment_name: str
    config: ExperimentConfig
    final_exploitability: float
    final_nash_conv: float
    training_time: float
    convergence_iteration: int
    metrics_history: List[Dict[str, float]]
    scheduler_stats: Optional[Dict[str, Any]] = None
    final_loss: float = 0.0
    success: bool = True
    error_message: str = ""


class SchedulerExperimentRunner:
    """Runner for scheduler experiments with comprehensive evaluation."""

    def __init__(
        self, config_path: str, output_base_dir: str = "results/scheduler_experiments"
    ):
        """Initialize experiment runner.

        Args:
            config_path: Path to experiment configuration file
            output_base_dir: Base directory for results
        """
        self.config_path = Path(config_path)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.experiment_logger = get_experiment_logger("scheduler_experiments")
        self.logger = self.experiment_logger.get_logger()

        # Results storage
        self.results: List[ExperimentResult] = []
        self.experiment_configs: List[ExperimentConfig] = []

        # Setup experiment configurations
        self._setup_experiment_configs()

    def _setup_experiment_configs(self):
        """Setup experiment configurations from yaml file."""
        base_config = self.config.copy()

        # Remove experiment-specific configs from base
        experiments_config = base_config.pop("experiments", {})

        # Base experiment (default config)
        base_exp_config = ExperimentConfig(
            name="base_scheduler",
            description="Base scheduler configuration",
            algorithm_config=base_config["algorithm"],
            game_config=base_config["game"],
            training_config=base_config.get("training", {}),
            eval_config=base_config.get("evaluation", {}),
            seed=base_config["experiment"]["seed"],
            total_iterations=base_config["experiment"]["total_iterations"],
            eval_every=base_config["experiment"]["eval_every"],
            output_dir=str(self.output_base_dir / "base_scheduler"),
        )
        self.experiment_configs.append(base_exp_config)

        # Additional experiment configurations
        for exp_name, exp_overrides in experiments_config.items():
            # Create deep copy of base config
            exp_config = ExperimentConfig(
                name=f"{base_exp_config.name}_{exp_name}",
                description=f"Base scheduler with {exp_name} modifications",
                algorithm_config=self._deep_merge(
                    base_config["algorithm"], exp_overrides.get("algorithm", {})
                ),
                game_config=self._deep_merge(
                    base_config["game"], exp_overrides.get("game", {})
                ),
                training_config=self._deep_merge(
                    base_config.get("training", {}), exp_overrides.get("training", {})
                ),
                eval_config=self._deep_merge(
                    base_config.get("evaluation", {}),
                    exp_overrides.get("evaluation", {}),
                ),
                seed=base_config["experiment"]["seed"],
                total_iterations=base_config["experiment"]["total_iterations"],
                eval_every=base_config["experiment"]["eval_every"],
                output_dir=str(self.output_base_dir / f"base_scheduler_{exp_name}"),
            )
            self.experiment_configs.append(exp_config)

        self.logger.info(f"Configured {len(self.experiment_configs)} experiments")

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def run_single_experiment(self, exp_config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment.

        Args:
            exp_config: Experiment configuration

        Returns:
            Experiment results
        """
        self.logger.info(f"Starting experiment: {exp_config.name}")
        self.logger.info(f"Description: {exp_config.description}")

        # Set random seed
        np.random.seed(exp_config.seed)
        torch.manual_seed(exp_config.seed)

        # Create output directory
        output_dir = Path(exp_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize game
        game_factory = GameFactory()
        game_wrapper = game_factory.create_game(exp_config.game_config["name"])

        # Initialize algorithm
        algorithm = ARMACAlgorithm(game_wrapper, exp_config.algorithm_config)

        # Initialize evaluator
        evaluator = OpenSpielExactEvaluator(exp_config.game_config["name"])

        # Training metrics storage
        metrics_history = []
        start_time = time.time()
        convergence_iteration = exp_config.total_iterations
        best_exploitability = float("inf")

        try:
            # Training loop
            for iteration in range(exp_config.total_iterations):
                # Training step
                training_state = algorithm.train_iteration()

                # Evaluation
                if iteration % exp_config.eval_every == 0:
                    # Get policy adapter for evaluation
                    policy_adapter = algorithm.get_policy_adapter()

                    # Evaluate exploitability and NashConv
                    exploitability = evaluator.compute_exploitability(policy_adapter)
                    nash_conv = evaluator.compute_nash_conv(policy_adapter)

                    # Get scheduler statistics if available
                    scheduler_stats = {}
                    if hasattr(algorithm, "armac_dual_rl"):
                        scheduler_stats = algorithm.armac_dual_rl.get_scheduler_stats()

                    # Record metrics
                    metrics = {
                        "iteration": iteration,
                        "exploitability": exploitability,
                        "nash_conv": nash_conv,
                        "training_loss": training_state.loss,
                        "wall_time": time.time() - start_time,
                        **scheduler_stats,
                    }
                    metrics_history.append(metrics)

                    # Check for convergence
                    if exploitability < best_exploitability:
                        best_exploitability = exploitability
                        convergence_iteration = iteration

                    # Log progress
                    self.logger.info(
                        f"Iter {iteration}: Exploitability={exploitability:.6f}, "
                        f"NashConv={nash_conv:.6f}, Loss={training_state.loss:.6f}"
                    )

                    # Save checkpoint
                    if iteration % (exp_config.eval_every * 4) == 0:
                        self._save_checkpoint(algorithm, metrics, output_dir, iteration)

            # Final evaluation
            final_policy_adapter = algorithm.get_policy_adapter()
            final_exploitability = evaluator.compute_exploitability(
                final_policy_adapter
            )
            final_nash_conv = evaluator.compute_nash_conv(final_policy_adapter)

            # Get final scheduler statistics
            final_scheduler_stats = {}
            if hasattr(algorithm, "armac_dual_rl"):
                final_scheduler_stats = algorithm.armac_dual_rl.get_scheduler_stats()

            # Save final results
            training_time = time.time() - start_time
            final_loss = training_state.loss

            # Save results
            results_data = {
                "experiment_config": asdict(exp_config),
                "final_metrics": {
                    "exploitability": final_exploitability,
                    "nash_conv": final_nash_conv,
                    "training_loss": final_loss,
                    "training_time": training_time,
                    "convergence_iteration": convergence_iteration,
                },
                "metrics_history": metrics_history,
                "scheduler_stats": final_scheduler_stats,
            }

            with open(output_dir / "results.json", "w") as f:
                json.dump(results_data, f, indent=2)

            # Save final model
            self._save_checkpoint(
                algorithm,
                metrics_history[-1] if metrics_history else {},
                output_dir,
                "final",
            )

            self.logger.info(f"Experiment {exp_config.name} completed successfully")
            self.logger.info(f"Final exploitability: {final_exploitability:.6f}")
            self.logger.info(f"Training time: {training_time:.2f}s")

            return ExperimentResult(
                experiment_name=exp_config.name,
                config=exp_config,
                final_exploitability=final_exploitability,
                final_nash_conv=final_nash_conv,
                training_time=training_time,
                convergence_iteration=convergence_iteration,
                metrics_history=metrics_history,
                scheduler_stats=final_scheduler_stats,
                final_loss=final_loss,
                success=True,
            )

        except Exception as e:
            error_msg = f"Experiment {exp_config.name} failed: {str(e)}"
            self.logger.error(error_msg)
            return ExperimentResult(
                experiment_name=exp_config.name,
                config=exp_config,
                final_exploitability=float("inf"),
                final_nash_conv=float("inf"),
                training_time=time.time() - start_time,
                convergence_iteration=exp_config.total_iterations,
                metrics_history=metrics_history,
                success=False,
                error_message=error_msg,
            )

    def _save_checkpoint(
        self,
        algorithm: ARMACAlgorithm,
        metrics: Dict[str, Any],
        output_dir: Path,
        iteration: int,
    ):
        """Save algorithm checkpoint and metrics."""
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model state
        if hasattr(algorithm, "actor") and hasattr(algorithm, "critic"):
            checkpoint = {
                "iteration": iteration,
                "actor_state_dict": algorithm.actor.state_dict(),
                "critic_state_dict": algorithm.critic.state_dict(),
                "regret_network_state_dict": algorithm.regret_network.state_dict(),
                "metrics": metrics,
            }

            # Save scheduler components if available
            if algorithm.use_scheduler and algorithm.scheduler_components:
                if algorithm.scheduler_components["scheduler"] is not None:
                    checkpoint["scheduler_state_dict"] = algorithm.scheduler_components[
                        "scheduler"
                    ].state_dict()

            torch.save(checkpoint, checkpoint_dir / f"checkpoint_{iteration}.pt")

    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all configured experiments."""
        self.logger.info(f"Starting {len(self.experiment_configs)} experiments")

        for i, exp_config in enumerate(self.experiment_configs):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(
                f"Experiment {i + 1}/{len(self.experiment_configs)}: {exp_config.name}"
            )
            self.logger.info(f"{'=' * 60}")

            result = self.run_single_experiment(exp_config)
            self.results.append(result)

            # Save intermediate results
            self._save_summary_results()

        self.logger.info("All experiments completed")
        return self.results

    def _save_summary_results(self):
        """Save summary of all results."""
        summary_data = {
            "experiment_config_path": str(self.config_path),
            "total_experiments": len(self.results),
            "successful_experiments": sum(1 for r in self.results if r.success),
            "results": [],
        }

        for result in self.results:
            result_data = {
                "experiment_name": result.experiment_name,
                "success": result.success,
                "final_exploitability": result.final_exploitability,
                "final_nash_conv": result.final_nash_conv,
                "training_time": result.training_time,
                "convergence_iteration": result.convergence_iteration,
                "final_loss": result.final_loss,
                "error_message": result.error_message,
            }
            if result.scheduler_stats:
                result_data["scheduler_stats"] = result.scheduler_stats
            summary_data["results"].append(result_data)

        with open(self.output_base_dir / "summary_results.json", "w") as f:
            json.dump(summary_data, f, indent=2)

    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        if not self.results:
            self.logger.warning("No results to generate report from")
            return

        # Sort results by final exploitability
        successful_results = [r for r in self.results if r.success]
        successful_results.sort(key=lambda x: x.final_exploitability)

        report = []
        report.append("# ARMAC Scheduler Experiments - Comparison Report\n")
        report.append(f"Total experiments: {len(self.results)}")
        report.append(f"Successful experiments: {len(successful_results)}")
        report.append(
            f"Failed experiments: {len(self.results) - len(successful_results)}\n"
        )

        # Results table
        report.append("## Results Summary\n")
        report.append(
            "| Rank | Experiment | Exploitability | NashConv | Training Time | Convergence | Final Loss |"
        )
        report.append(
            "|------|------------|----------------|----------|---------------|-------------|------------|"
        )

        for i, result in enumerate(successful_results):
            report.append(
                f"| {i + 1} | {result.experiment_name} | "
                f"{result.final_exploitability:.6f} | "
                f"{result.final_nash_conv:.6f} | "
                f"{result.training_time:.2f}s | "
                f"{result.convergence_iteration} | "
                f"{result.final_loss:.6f} |"
            )

        # Failed experiments
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            report.append("\n## Failed Experiments\n")
            for result in failed_results:
                report.append(f"- **{result.experiment_name}**: {result.error_message}")

        # Detailed analysis
        report.append("\n## Detailed Analysis\n")

        # Best performer
        if successful_results:
            best = successful_results[0]
            report.append(f"**Best performing experiment**: {best.experiment_name}")
            report.append(f"- Final exploitability: {best.final_exploitability:.6f}")
            report.append(f"- Training time: {best.training_time:.2f}s")
            report.append(f"- Convergence iteration: {best.convergence_iteration}")

            # Scheduler analysis
            scheduler_results = [r for r in successful_results if r.scheduler_stats]
            if scheduler_results:
                report.append("\n### Scheduler Performance Analysis\n")
                for result in scheduler_results:
                    stats = result.scheduler_stats
                    report.append(f"**{result.experiment_name}**:")
                    report.append(
                        f"- Lambda mean: {stats.get('lambda_mean', 'N/A'):.4f}"
                    )
                    report.append(f"- Lambda std: {stats.get('lambda_std', 'N/A'):.4f}")
                    report.append(
                        f"- Scheduler type: {stats.get('scheduler_type', 'N/A')}"
                    )
                    report.append("")

        # Save report
        report_text = "\n".join(report)
        with open(self.output_base_dir / "comparison_report.md", "w") as f:
            f.write(report_text)

        self.logger.info(
            f"Comparison report saved to {self.output_base_dir / 'comparison_report.md'}"
        )

    def generate_visualizations(self):
        """Generate visualization plots for experiment results."""
        try:
            # Create manifest for auto-generator
            manifest_data = []
            for result in self.results:
                if result.success and result.metrics_history:
                    for metrics in result.metrics_history:
                        manifest_data.append(
                            {
                                "experiment_name": result.experiment_name,
                                "iteration": metrics["iteration"],
                                "exploitability": metrics["exploitability"],
                                "nash_conv": metrics["nash_conv"],
                                "training_loss": metrics["training_loss"],
                                "wall_time": metrics["wall_time"],
                                **{
                                    k: v
                                    for k, v in metrics.items()
                                    if k
                                    not in [
                                        "iteration",
                                        "exploitability",
                                        "nash_conv",
                                        "training_loss",
                                        "wall_time",
                                    ]
                                },
                            }
                        )

            if manifest_data:
                # Save manifest
                import pandas as pd

                df = pd.DataFrame(manifest_data)
                manifest_path = self.output_base_dir / "experiment_manifest.csv"
                df.to_csv(manifest_path, index=False)

                # Generate plots
                generator = ManifestAutoGenerator(
                    str(manifest_path), str(self.output_base_dir / "plots")
                )
                generator.generate_all_figures()

                self.logger.info(
                    f"Visualizations saved to {self.output_base_dir / 'plots'}"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ARMAC scheduler experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scheduler_experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create experiment runner
    runner = SchedulerExperimentRunner(args.config, args.output)

    # Filter experiments if specified
    if args.experiments:
        filtered_configs = []
        for exp_name in args.experiments:
            for config in runner.experiment_configs:
                if exp_name in config.name:
                    filtered_configs.append(config)
                    break
        if not filtered_configs:
            print(f"No experiments found matching: {args.experiments}")
            return
        runner.experiment_configs = filtered_configs

    # Run experiments
    results = runner.run_all_experiments()

    # Generate report and visualizations
    runner.generate_comparison_report()
    runner.generate_visualizations()

    print(f"\nExperiments completed. Results saved to: {runner.output_base_dir}")


if __name__ == "__main__":
    main()
