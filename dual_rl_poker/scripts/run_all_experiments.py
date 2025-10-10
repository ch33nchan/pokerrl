#!/usr/bin/env python3
"""
Script to run all experiments for the Dual RL Poker project.

This script runs experiments for all three algorithm families (Deep CFR, SD-CFR, ARMAC)
on both Kuhn and Leduc poker, following the standardized evaluation protocol.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logging import setup_logging
from utils.experiment_runner import ExperimentRunner
from utils.config import load_config


def setup_logging_config(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/experiments_{int(time.time())}.log")
        ]
    )


def parse_seed_range(seeds_str: str) -> List[int]:
    """Parse seed range like '0-19' into list [0, 1, ..., 19]."""
    if '-' in seeds_str:
        start, end = map(int, seeds_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(seeds_str)]


def run_experiment_suite(config_path: str, games: List[str], methods: List[str],
                        seed_ranges: Dict[str, List[int]], force_rerun: bool = False):
    """Run a complete suite of experiments.

    Args:
        config_path: Path to configuration file
        games: List of games to run
        methods: List of methods to test
        seed_ranges: Dictionary mapping (game, method) to seed lists
        force_rerun: Whether to rerun existing experiments
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment suite")

    # Load base configuration
    base_config = load_config(config_path)

    # Initialize experiment runner
    runner = ExperimentRunner(base_config)

    # Track overall results
    all_results = []
    start_time = time.time()

    # Run experiments for each game and method combination
    for game in games:
        logger.info(f"Running experiments for game: {game}")

        for method in methods:
            # Get seeds for this combination
            seed_key = f"{game}_{method}"
            seeds = seed_ranges.get(seed_key, [0])  # Default to seed 0

            logger.info(f"Running {method} on {game} with seeds {seeds}")

            # Update config for this experiment
            config = base_config.copy()
            config['game']['name'] = game
            config['algorithm']['method'] = method
            config['experiment']['seeds'] = seeds

            # Adjust iterations for Leduc (smaller budget)
            if game == "leduc_poker":
                config['training']['iterations'] = 200
                logger.info("Using smaller budget for Leduc Hold'em (200 iterations)")

            try:
                # Run the experiment
                results = runner.run_experiment(config, force_rerun=force_rerun)
                all_results.extend(results)

                logger.info(f"Completed {method} on {game}: {len(results)} runs")

            except Exception as e:
                logger.error(f"Failed to run {method} on {game}: {e}")
                continue

    total_time = time.time() - start_time
    logger.info(f"Experiment suite completed in {total_time:.2f} seconds")

    # Save aggregate results
    aggregate_results = {
        'config_path': config_path,
        'total_experiments': len(all_results),
        'total_time': total_time,
        'games': games,
        'methods': methods,
        'results': [result.to_dict() for result in all_results]
    }

    results_dir = Path(base_config['logging']['save_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"experiment_suite_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)

    logger.info(f"Aggregate results saved to {results_file}")
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all Dual RL Poker experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--games", nargs='+', default=["kuhn_poker", "leduc_poker"],
                        help="Games to run experiments on")
    parser.add_argument("--methods", nargs='+', default=["deep_cfr", "sd_cfr", "armac"],
                        help="Methods to test")
    parser.add_argument("--kuhn-seeds", type=str, default="0-19",
                        help="Seed range for Kuhn poker (e.g., '0-19')")
    parser.add_argument("--leduc-seeds", type=str, default="0-9",
                        help="Seed range for Leduc poker (e.g., '0-9')")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Rerun experiments even if results exist")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging_config(args.log_level)
    logger = logging.getLogger(__name__)

    # Parse seed ranges
    seed_ranges = {
        "kuhn_poker_deep_cfr": parse_seed_range(args.kuhn_seeds),
        "kuhn_poker_sd_cfr": parse_seed_range(args.kuhn_seeds),
        "kuhn_poker_armac": parse_seed_range(args.kuhn_seeds),
        "leduc_poker_deep_cfr": parse_seed_range(args.leduc_seeds),
        "leduc_poker_sd_cfr": parse_seed_range(args.leduc_seeds),
        "leduc_poker_armac": parse_seed_range(args.leduc_seeds),
    }

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    try:
        # Run the experiment suite
        results = run_experiment_suite(
            config_path=args.config,
            games=args.games,
            methods=args.methods,
            seed_ranges=seed_ranges,
            force_rerun=args.force_rerun
        )

        logger.info(f"Successfully completed {len(results)} experiments")
        logger.info("Results saved to results/ directory")

    except Exception as e:
        logger.error(f"Experiment suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()