#!/usr/bin/env python3
"""
Run working experiments with Tabular CFR and Deep CFR across all architectures.
This will run 20 seeds per condition for statistical significance.
"""

import os
import sys
sys.path.append('/Users/cheencheen/Desktop/lossfunk/q-agent')

import json
import time
import numpy as np
import pyspiel
from experiments.run_comprehensive_cpu_experiments import (
    TabularCFR, DeepCFRCanonical, ExactEvaluator, ExperimentConfig,
    calculate_flops, get_git_hash, SEEDS, ITERATIONS, EVAL_EVERY
)
from dataclasses import asdict

def run_focused_experiments():
    """Run experiments with working algorithms."""
    print("Starting Focused Deep CFR Architecture Study")
    print(f"OpenSpiel version: {pyspiel.__version__}")
    print(f"Running 20 seeds per condition for statistical significance")
    print()

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("manifests", exist_ok=True)

    # Define experiments (working ones only)
    experiments = []

    # Tabular CFR (baseline only)
    for seed in SEEDS:
        experiments.append({
            "run_id": f"kuhn_poker_tabular_cfr_{seed:03d}",
            "game": "kuhn_poker",
            "baseline_type": "tabular_cfr",
            "architecture": "baseline",
            "seed": seed
        })

    # Deep CFR with all architectures
    architectures = ["baseline", "wide", "deep", "fast"]
    for architecture in architectures:
        for seed in SEEDS:
            experiments.append({
                "run_id": f"kuhn_poker_deep_cfr_{architecture}_{seed:03d}",
                "game": "kuhn_poker",
                "baseline_type": "deep_cfr",
                "architecture": architecture,
                "seed": seed
            })

    print(f"Total experiments: {len(experiments)}")
    print(f"Tabular CFR experiments: {len([e for e in experiments if e['baseline_type'] == 'tabular_cfr'])}")
    print(f"Deep CFR experiments: {len([e for e in experiments if e['baseline_type'] == 'deep_cfr'])}")
    print()

    # Run experiments
    all_configs = []
    failed_configs = []

    for i, exp_config in enumerate(experiments):
        print(f"Progress: {i+1}/{len(experiments)} - {exp_config['run_id']}")

        try:
            config = run_single_experiment(exp_config)

            if config.final_exploitability < float('inf'):
                all_configs.append(config)
            else:
                failed_configs.append(config)

        except Exception as e:
            print(f"Exception in experiment {exp_config['run_id']}: {e}")
            failed_configs.append(create_failed_config(exp_config))

    # Save results
    successful_data = [asdict(config) for config in all_configs]
    failed_data = [asdict(config) for config in failed_configs]

    with open("manifests/focused_experiments.json", "w") as f:
        json.dump({
            "successful": successful_data,
            "failed": failed_data,
            "metadata": {
                "total_experiments": len(experiments),
                "successful_experiments": len(all_configs),
                "failed_experiments": len(failed_configs),
                "timestamp": time.time()
            }
        }, f, indent=2)

    # Generate summary statistics
    print(f"\n=== Summary ===")
    print(f"Successful experiments: {len(all_configs)}/{len(experiments)}")
    print(f"Failed experiments: {len(failed_configs)}")

    if all_configs:
        print(f"\n=== Results Summary ===")

        # Group by baseline type and architecture
        summary = {}
        for config in all_configs:
            key = f"{config.baseline_type}_{config.architecture}"
            if key not in summary:
                summary[key] = []
            summary[key].append(config.final_exploitability)

        for key, values in summary.items():
            if values and all(v < float('inf') for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"{key}:")
                print(f"  Mean: {mean_val:.6f} ± {std_val:.6f}")
                print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
                print(f"  Samples: {len(values)}")

        # Performance comparison
        print(f"\n=== Performance Comparison ===")
        deep_results = {}
        for architecture in architectures:
            key = f"deep_cfr_{architecture}"
            if key in summary:
                deep_results[architecture] = np.mean(summary[key])

        if deep_results:
            best_arch = min(deep_results.keys(), key=lambda k: deep_results[k])
            worst_arch = max(deep_results.keys(), key=lambda k: deep_results[k])
            improvement = ((deep_results[worst_arch] - deep_results[best_arch]) / deep_results[worst_arch]) * 100

            print(f"Best architecture: {best_arch} ({deep_results[best_arch]:.6f})")
            print(f"Worst architecture: {worst_arch} ({deep_results[worst_arch]:.6f})")
            print(f"Improvement: {improvement:.1f}%")

        # Tabular vs Deep comparison
        if "tabular_cfr_baseline" in summary and "deep_cfr_deep" in summary:
            tabular_mean = np.mean(summary["tabular_cfr_baseline"])
            deep_mean = np.mean(summary["deep_cfr_deep"])
            gap = ((deep_mean - tabular_mean) / tabular_mean) * 100
            print(f"\nTabular CFR: {tabular_mean:.6f}")
            print(f"Deep CFR (deep): {deep_mean:.6f}")
            print(f"Performance gap: {gap:.1f}% (negative means Deep CFR is better)")

    print(f"\nResults saved to: manifests/focused_experiments.json")
    return all_configs, failed_configs

def run_single_experiment(config):
    """Run a single experiment."""
    game_name = config["game"]
    baseline_type = config["baseline_type"]
    architecture = config.get("architecture", "baseline")
    seed = config["seed"]
    run_id = config["run_id"]

    start_time = time.time()

    try:
        # Load game
        game = pyspiel.load_game(game_name)

        # Initialize agent
        if baseline_type == "tabular_cfr":
            agent = TabularCFR(game, seed)
        elif baseline_type == "deep_cfr":
            agent = DeepCFRCanonical(game, architecture, seed)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Train
        results = agent.train(ITERATIONS, EVAL_EVERY)
        end_time = time.time()

        # Calculate metrics
        final_exploitability = results["exploitability"][-1] if results["exploitability"] else float('inf')
        final_nashconv = results["nashconv"][-1] if results["nashconv"] else float('inf')

        # Calculate capacity metrics
        state_size = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()

        if baseline_type == "tabular_cfr":
            params_count = 0
            flops_est = 0
        else:
            # Estimate parameters for neural networks
            if architecture == "baseline":
                params_count = (state_size + 1) * 64 + (64 + 1) * 64 + (64 + 1) * num_actions
            elif architecture == "wide":
                params_count = (state_size + 1) * 128 + (128 + 1) * 128 + (128 + 1) * num_actions
            elif architecture == "deep":
                params_count = (state_size + 1) * 64 + (64 + 1) * 64 + (64 + 1) * 64 + (64 + 1) * num_actions
            elif architecture == "fast":
                params_count = (state_size + 1) * 32 + (32 + 1) * 32 + (32 + 1) * num_actions
            else:
                params_count = 0

            flops_est = calculate_flops(architecture, state_size, num_actions)

        return ExperimentConfig(
            run_id=run_id,
            game=game_name,
            baseline_type=baseline_type,
            architecture=architecture,
            seed=seed,
            params_count=params_count,
            flops_est=flops_est,
            optimizer_cfg={"type": "adam", "lr": 0.001},
            replay_cfg={"buffer_size": 10000, "batch_size": 64},
            update_cadence=10,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            final_exploitability=final_exploitability,
            final_nashconv=final_nashconv,
            steps_to_threshold=None,
            time_to_threshold=None,
            wall_clock_s=end_time - start_time,
            openspiel_version=pyspiel.__version__,
            python_version="3.9",
            git_hash=get_git_hash()
        )

    except Exception as e:
        print(f"Failed experiment {run_id}: {e}")
        return create_failed_config(config)

def create_failed_config(config):
    """Create a failed experiment config."""
    return ExperimentConfig(
        run_id=config["run_id"],
        game=config["game"],
        baseline_type=config["baseline_type"],
        architecture=config.get("architecture", "baseline"),
        seed=config["seed"],
        params_count=0,
        flops_est=0,
        optimizer_cfg={},
        replay_cfg={},
        update_cadence=10,
        iterations=ITERATIONS,
        eval_every=EVAL_EVERY,
        final_exploitability=float('inf'),
        final_nashconv=float('inf'),
        steps_to_threshold=None,
        time_to_threshold=None,
        wall_clock_s=0.0,
        openspiel_version=pyspiel.__version__,
        python_version="3.9",
        git_hash=get_git_hash()
    )

if __name__ == "__main__":
    successful, failed = run_focused_experiments()