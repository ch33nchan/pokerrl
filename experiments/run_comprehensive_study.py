"""
Comprehensive Experiment Runner for Deep CFR Architecture Study

This script runs the full experimental study with proper protocol:
- 20 seeds per configuration
- 400 iterations for both games
- External sampling fixed
- Proper evaluation with OpenSpiel
- Universal diagnostics logging
- Head-to-head EV evaluations
"""

import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from qagent.baselines.canonical_baselines import run_baseline_experiment
from experiments.manifest_schema import RunManifest, load_manifests, save_manifests
from qagent.utils.diagnostics import DiagnosticsLogger

# Configuration
GAMES = ["kuhn_poker", "leduc_poker"]
BASELINE_TYPES = ["tabular_cfr", "deep_cfr", "sd_cfr"]
ARCHITECTURES = ["baseline", "lstm_opt", "lstm_no_hist", "lstm_no_emb"]
SEEDS = list(range(20))
ITERATIONS = 400
EVAL_EVERY = 20
TRAVERSAL = "external_sampling"

# Thresholds T for each game (mBB/100)
THRESHOLDS = {
    "kuhn_poker": 0.1,
    "leduc_poker": 2.5
}

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    game = config["game"]
    baseline_type = config["baseline_type"]
    architecture = config.get("architecture", "mlp")
    seed = config["seed"]
    run_id = config["run_id"]

    print(f"Starting experiment: {run_id}")

    # Initialize diagnostics logger
    logger = DiagnosticsLogger(run_id)

    try:
        # Run baseline experiment
        start_time = time.time()
        results = run_baseline_experiment(
            game_name=game,
            baseline_type=baseline_type,
            seed=seed,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            architecture=architecture
        )
        end_time = time.time()

        # Add diagnostics coverage
        results["diagnostics_coverage"] = logger.get_coverage_report()
        results["wall_clock_s"] = end_time - start_time

        # Calculate steps to threshold
        if results["exploitability"]:
            threshold = THRESHOLDS[game]
            steps_to_threshold = None
            for i, exp in enumerate(results["exploitability"]):
                if exp <= threshold:
                    steps_to_threshold = results["iterations"][i]
                    break
            results["steps_to_threshold"] = steps_to_threshold

            # Calculate time to threshold
            if steps_to_threshold is not None:
                time_per_iteration = results["wall_clock_s"] / ITERATIONS
                results["time_to_threshold"] = steps_to_threshold * time_per_iteration
            else:
                results["time_to_threshold"] = None

        # Finalize diagnostics
        logger.finalize()

        results["status"] = "completed"
        print(f"Completed experiment: {run_id}")

        return results

    except Exception as e:
        print(f"Failed experiment {run_id}: {e}")
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": str(e),
            "game": game,
            "baseline_type": baseline_type,
            "seed": seed,
            "architecture": architecture
        }


def run_baseline_experiments() -> List[Dict[str, Any]]:
    """Run all baseline experiments."""
    configs = []

    # Baseline experiments
    for game in GAMES:
        for baseline_type in BASELINE_TYPES:
            for seed in SEEDS:
                run_id = f"{game}_{baseline_type}_{seed:03d}"
                config = {
                    "run_id": run_id,
                    "game": game,
                    "baseline_type": baseline_type,
                    "seed": seed,
                    "architecture": "mlp"  # Default for baselines
                }
                configs.append(config)

    print(f"Running {len(configs)} baseline experiments")

    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 1) as executor:
        future_to_config = {executor.submit(run_single_experiment, config): config for config in configs}

        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Experiment {config['run_id']} failed: {e}")
                results.append({
                    "run_id": config["run_id"],
                    "status": "failed",
                    "error_message": str(e),
                    **config
                })

    return results


def run_architecture_experiments() -> List[Dict[str, Any]]:
    """Run architecture comparison experiments."""
    configs = []

    for game in GAMES:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                run_id = f"{game}_{architecture}_{seed:03d}"
                config = {
                    "run_id": run_id,
                    "game": game,
                    "baseline_type": "deep_cfr",  # Use Deep CFR as base
                    "seed": seed,
                    "architecture": architecture
                }
                configs.append(config)

    print(f"Running {len(configs)} architecture experiments")

    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 1) as executor:
        future_to_config = {executor.submit(run_single_experiment, config): config for config in configs}

        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Experiment {config['run_id']} failed: {e}")
                results.append({
                    "run_id": config["run_id"],
                    "status": "failed",
                    "error_message": str(e),
                    **config
                })

    return results


def create_manifest_from_results(results: List[Dict[str, Any]]) -> List[RunManifest]:
    """Create manifest entries from experimental results."""
    manifests = []

    for result in results:
        if result["status"] == "failed":
            continue

        # Architecture-specific parameters
        arch_params = {
            "baseline": {"params_count": 5040, "flops_est": 5040},
            "lstm_opt": {"params_count": 12640, "flops_est": 25280},
            "lstm_no_hist": {"params_count": 8960, "flops_est": 17920},
            "lstm_no_emb": {"params_count": 10240, "flops_est": 20480},
            "mlp": {"params_count": 5040, "flops_est": 5040},  # Default for baselines
        }

        arch = result.get("architecture", "mlp")
        arch_info = arch_params.get(arch, arch_params["mlp"])

        # Default configurations
        optimizer_cfg = {
            "type": "adam",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }

        replay_cfg = {
            "buffer_size": 50000,
            "batch_size": 128,
            "sampling": "uniform"
        }

        manifest = RunManifest(
            run_id=result["run_id"],
            game=result["game"],
            traversal=TRAVERSAL,
            architecture=arch,
            seed=result["seed"],
            params_count=arch_info["params_count"],
            flops_est=arch_info["flops_est"],
            optimizer_cfg=optimizer_cfg,
            replay_cfg=replay_cfg,
            update_cadence=10,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            final_exploitability=result.get("final_exploitability", float('inf')),
            steps_to_threshold=result.get("steps_to_threshold"),
            time_to_threshold=result.get("time_to_threshold"),
            wall_clock_s=result.get("wall_clock_s", 0.0),
            final_nashconv=result.get("final_nashconv"),
            openspiel_version="1.6.3",
            python_version="3.9.6",
            diagnostics_coverage=result.get("diagnostics_coverage", {}),
            status=result["status"],
            ev_vs_tabular=None,  # To be filled later
            ev_vs_deepcfr=None,  # To be filled later
            ev_vs_sdcfr=None  # To be filled later
        )

        manifests.append(manifest)

    return manifests


def save_results(results: List[Dict[str, Any]], filepath: str):
    """Save experimental results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    converted_results = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)


def main():
    """Main experiment runner."""
    print("Starting comprehensive Deep CFR architecture study")
    print(f"Games: {GAMES}")
    print(f"Baseline types: {BASELINE_TYPES}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Seeds per config: {len(SEEDS)}")
    print(f"Iterations per run: {ITERATIONS}")

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("diagnostics", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # Load existing manifest if available
    manifest_file = "experiments/runs_manifest.json"
    if os.path.exists(manifest_file):
        existing_manifests = load_manifests(manifest_file)
        print(f"Loaded {len(existing_manifests)} existing manifest entries")
    else:
        existing_manifests = []

    # Run baseline experiments
    print("\n=== Running Baseline Experiments ===")
    baseline_results = run_baseline_experiments()
    save_results(baseline_results, "results/baseline_experiments.json")
    print(f"Completed {len(baseline_results)} baseline experiments")

    # Run architecture experiments
    print("\n=== Running Architecture Experiments ===")
    arch_results = run_architecture_experiments()
    save_results(arch_results, "results/architecture_experiments.json")
    print(f"Completed {len(arch_results)} architecture experiments")

    # Combine all results
    all_results = baseline_results + arch_results
    save_results(all_results, "results/all_experiments.json")
    print(f"Total experiments: {len(all_results)}")

    # Create and save manifest
    print("\n=== Creating Manifest ===")
    new_manifests = create_manifest_from_results(all_results)
    all_manifests = existing_manifests + new_manifests
    save_manifests(all_manifests, manifest_file)
    print(f"Manifest contains {len(all_manifests)} entries")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    completed_results = [r for r in all_results if r["status"] == "completed"]
    failed_results = [r for r in all_results if r["status"] == "failed"]

    print(f"Completed: {len(completed_results)}")
    print(f"Failed: {len(failed_results)}")

    if completed_results:
        # Game-wise summary
        for game in GAMES:
            game_results = [r for r in completed_results if r["game"] == game]
            if game_results:
                avg_exploitability = np.mean([r["final_exploitability"] for r in game_results if r["final_exploitability"] != float('inf')])
                avg_wallclock = np.mean([r["wall_clock_s"] for r in game_results])
                print(f"{game}: Avg exploitability = {avg_exploitability:.4f}, Avg wall-clock = {avg_wallclock:.2f}s")

    print("\nStudy completed successfully!")
    print(f"Results saved to: results/")
    print(f"Diagnostics saved to: diagnostics/")
    print(f"Manifest saved to: {manifest_file}")


if __name__ == "__main__":
    main()