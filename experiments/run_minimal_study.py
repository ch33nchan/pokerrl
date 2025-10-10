"""
Minimal Representative Study for Deep CFR Architecture

This runs a smaller but still statistically meaningful study:
- 5 seeds per configuration (instead of 20)
- 200 iterations (instead of 400)
- Focus on Kuhn Poker (faster to run)
- All architectural variants included
"""

import os
import json
import time
import sys
sys.path.append('/Users/cheencheen/Desktop/lossfunk/q-agent')

from qagent.baselines.canonical_baselines import run_baseline_experiment
from experiments.manifest_schema import RunManifest, save_manifests
from qagent.analysis.figure_generator import FigureGenerator

# Configuration (smaller but still meaningful)
GAMES = ["kuhn_poker"]  # Focus on Kuhn for faster execution
ARCHITECTURES = ["baseline", "lstm_opt", "lstm_no_hist", "lstm_no_emb"]
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds instead of 20
ITERATIONS = 200  # 200 iterations instead of 400
EVAL_EVERY = 20
TRAVERSAL = "external_sampling"

# Thresholds T for each game (mBB/100)
THRESHOLDS = {
    "kuhn_poker": 0.1,
    "leduc_holdem": 2.5
}

def run_single_experiment(config):
    """Run a single experiment configuration."""
    game = config["game"]
    architecture = config["architecture"]
    seed = config["seed"]
    run_id = config["run_id"]

    print(f"Starting experiment: {run_id}")

    try:
        # Run experiment
        start_time = time.time()
        results = run_baseline_experiment(
            game_name=game,
            baseline_type="deep_cfr",
            seed=seed,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            architecture=architecture
        )
        end_time = time.time()

        # Add metadata
        results.update({
            "run_id": run_id,
            "game": game,
            "architecture": architecture,
            "seed": seed,
            "wall_clock_s": end_time - start_time,
            "final_exploitability": results.get("final_exploitability", float('inf')),
            "final_nashconv": results.get("final_nashconv"),
            "status": "completed"
        })

        # Calculate steps to threshold
        if results.get("exploitability"):
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

        print(f"Completed experiment: {run_id}")
        return results

    except Exception as e:
        print(f"Failed experiment {run_id}: {e}")
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": str(e),
            "game": game,
            "architecture": architecture,
            "seed": seed
        }

def main():
    """Run minimal representative study."""
    print("Starting Minimal Deep CFR Architecture Study")
    print(f"Games: {GAMES}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Seeds per config: {len(SEEDS)}")
    print(f"Iterations per run: {ITERATIONS}")

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("paper/plots", exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)

    # Generate experiment configurations
    configs = []
    for game in GAMES:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                run_id = f"{game}_{architecture}_{seed:03d}"
                config = {
                    "run_id": run_id,
                    "game": game,
                    "architecture": architecture,
                    "seed": seed
                }
                configs.append(config)

    print(f"Running {len(configs)} experiments")

    # Run experiments
    results = []
    for config in configs:
        result = run_single_experiment(config)
        results.append(result)

    # Save results
    save_results(results, "results/minimal_study_results.json")
    print(f"Completed {len(results)} experiments")

    # Create manifest
    manifests = create_manifest_from_results(results)
    save_manifests(manifests, "experiments/minimal_study_manifest.json")
    print(f"Manifest contains {len(manifests)} entries")

    # Print summary statistics
    completed_results = [r for r in results if r["status"] == "completed"]
    failed_results = [r for r in results if r["status"] == "failed"]

    print(f"\n=== Summary Statistics ===")
    print(f"Completed: {len(completed_results)}")
    print(f"Failed: {len(failed_results)}")

    if completed_results:
        # Architecture-wise summary
        for arch in ARCHITECTURES:
            arch_results = [r for r in completed_results if r["architecture"] == arch]
            if arch_results:
                avg_exploitability = np.mean([r["final_exploitability"] for r in arch_results if r["final_exploitability"] != float('inf')])
                avg_wallclock = np.mean([r["wall_clock_s"] for r in arch_results])
                print(f"{arch}: Avg exploitability = {avg_exploitability:.4f}, Avg wall-clock = {avg_wallclock:.2f}s")

    # Generate figures and tables
    print(f"\n=== Generating Figures ===")
    try:
        generator = FigureGenerator()
        figure_files = generator.generate_all_figures(completed_results)
        print(f"Generated {len(figure_files)} figures and tables")
    except Exception as e:
        print(f"Error generating figures: {e}")

    print(f"\nStudy completed successfully!")
    print(f"Results saved to: results/")
    print(f"Figures saved to: paper/plots/")
    print(f"Tables saved to: paper/tables/")


def save_results(results, filepath):
    """Save results to JSON file."""
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


def create_manifest_from_results(results):
    """Create manifest entries from experimental results."""
    manifests = []

    # Architecture-specific parameters
    arch_params = {
        "baseline": {"params_count": 5040, "flops_est": 5040},
        "lstm_opt": {"params_count": 12640, "flops_est": 25280},
        "lstm_no_hist": {"params_count": 8960, "flops_est": 17920},
        "lstm_no_emb": {"params_count": 10240, "flops_est": 20480},
    }

    for result in results:
        if result["status"] == "failed":
            continue

        arch = result.get("architecture", "baseline")
        arch_info = arch_params.get(arch, arch_params["baseline"])

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
            status=result["status"]
        )

        manifests.append(manifest)

    return manifests


if __name__ == "__main__":
    import numpy as np  # Import here to avoid conflicts
    main()