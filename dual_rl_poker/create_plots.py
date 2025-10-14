#!/usr/bin/env python3
"""Create plots from experimental results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Optional dependencies (guarded)
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import seaborn as sns

    sns.set_palette("husl")
except Exception:
    sns = None

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-paper")


def load_results():
    """Load all experimental results, robust to multiple JSON schemas.

    Supports:
    - Top-level keys: 'game' and 'method'
    - 'metadata' block containing 'game' and 'algorithm' or 'method'
    - Fallback parsing from 'experiment_id' or 'run_id' (e.g., 'kuhn_poker_deep_cfr_seed_00')
    Skips files that don't provide a recognizable (game, method) pair.
    """
    results_dir = Path("results")
    all_results = {}

    for result_file in results_dir.rglob("*_results.json"):
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
        except Exception:
            # Skip unreadable or invalid JSON files
            continue

        game = None
        method = None

        if isinstance(result, dict):
            # Schema 1: top-level fields
            if "game" in result and "method" in result:
                game = result.get("game")
                method = result.get("method")

            # Schema 2: metadata block
            if (game is None or method is None) and isinstance(
                result.get("metadata"), dict
            ):
                md = result["metadata"]
                game = game or md.get("game")
                method = method or md.get("method") or md.get("algorithm")

            # Schema 3: top-level algorithm field
            if method is None and "algorithm" in result:
                method = result.get("algorithm")

            # Schema 4: parse from experiment_id or run_id
            if game is None or method is None:
                exp_id = result.get("experiment_id") or result.get("run_id")
                # Also check metadata for run_id
                if exp_id is None and isinstance(result.get("metadata"), dict):
                    exp_id = result["metadata"].get("run_id")

                if isinstance(exp_id, str):
                    # Try formats like "kuhn_poker_deep_cfr_seed_00" or "kuhn_armac_seed_00"
                    if "_seed_" in exp_id:
                        prefix = exp_id.split("_seed_")[0]
                        parts = prefix.split("_")
                        if len(parts) >= 2:
                            # Heuristic: last token is method; the rest form game name
                            method = method or parts[-1]
                            game = game or "_".join(parts[:-1])

        # Validate extracted identifiers
        if not game or not method:
            # Skip incompatible files
            continue

        key = f"{game}_{method}"
        if key not in all_results:
            all_results[key] = []
        all_results[key].append(result)

    return all_results


def plot_exploitability_curves(
    all_results, save_path="results/plots/exploitability_curves.png"
):
    """Plot exploitability curves during training."""
    # Create plots directory
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    games = ["kuhn_poker", "leduc_poker"]
    algorithms = ["deep_cfr", "sd_cfr", "armac"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]  # Professional color scheme

    for game_idx, game in enumerate(games):
        ax = axes[game_idx]

        for alg_idx, algorithm in enumerate(algorithms):
            key = f"{game}_{algorithm}"
            if key in all_results and all_results[key]:
                # Get average learning curve
                all_curves = []
                for result in all_results[key]:
                    if "evaluation_history" in result:
                        eval_history = result["evaluation_history"]
                        iterations = [
                            eval_item["iteration"] for eval_item in eval_history
                        ]
                        exploitability = [
                            eval_item["exploitability"] for eval_item in eval_history
                        ]
                        all_curves.append((iterations, exploitability))

                if all_curves:
                    # Calculate mean and confidence intervals
                    max_iter = max(max(curve[0]) for curve in all_curves)
                    mean_curve = np.zeros(max_iter)
                    std_curve = np.zeros(max_iter)
                    count_curve = np.zeros(max_iter)

                    for iterations, exploitability in all_curves:
                        for i, (it, exp) in enumerate(zip(iterations, exploitability)):
                            mean_curve[it - 1] += exp
                            std_curve[it - 1] += exp * exp
                            count_curve[it - 1] += 1

                    # Calculate mean and std
                    mask = count_curve > 0
                    mean_curve[mask] /= count_curve[mask]
                    # Clamp variance to >= 0 to avoid numerical issues
                    var = std_curve[mask] / count_curve[mask] - mean_curve[mask] ** 2
                    var = np.maximum(var, 0.0)
                    std_curve[mask] = np.sqrt(var)

                    # Plot mean curve with confidence interval
                    x_axis = np.arange(1, max_iter + 1)
                    ax.plot(
                        x_axis,
                        mean_curve,
                        label=algorithm.replace("_", " ").title(),
                        color=colors[alg_idx],
                        linewidth=2.5,
                    )
                    ax.fill_between(
                        x_axis,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=colors[alg_idx],
                        alpha=0.2,
                    )

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Exploitability (game units)")
        ax.set_title(f"{game.replace('_', ' ').title()}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved exploitability curves to {save_path}")
    return fig


def plot_comparison_bar_chart(
    all_results, save_path="results/plots/performance_comparison.png"
):
    """Create comparison bar chart of final exploitability (mean ± std) from actual results."""
    Path(save_path).parent.mkdir(exist_ok=True)

    # Data-driven aggregation from actual experiment results
    # Collect games and algorithms that have results
    known_games = ["kuhn_poker", "leduc_poker"]
    known_algs = ["deep_cfr", "sd_cfr", "armac"]
    pretty_names = {"deep_cfr": "Deep CFR", "sd_cfr": "SD-CFR", "armac": "ARMAC"}

    games = [
        g
        for g in known_games
        if any(
            f"{g}_{a}" in all_results and all_results[f"{g}_{a}"] for a in known_algs
        )
    ]
    if not games:
        print("No data-driven results found; skipping comparison bar chart.")
        return None

    algs_present = []
    for alg in known_algs:
        if any(
            f"{g}_{alg}" in all_results and all_results[f"{g}_{alg}"] for g in games
        ):
            algs_present.append(alg)

    if not algs_present:
        print("No algorithms with results; skipping comparison bar chart.")
        return None

    # Aggregate final exploitability per game/algorithm
    means = {g: [] for g in games}
    stds = {g: [] for g in games}
    for g in games:
        for alg in algs_present:
            key = f"{g}_{alg}"
            vals = []
            for res in all_results.get(key, []):
                eval_hist = res.get("evaluation_history", [])
                if eval_hist:
                    val = eval_hist[-1].get("exploitability", None)
                    if val is not None:
                        vals.append(float(val))
            if vals:
                means[g].append(float(np.mean(vals)))
                stds[g].append(float(np.std(vals)))
            else:
                means[g].append(np.nan)
                stds[g].append(0.0)

    # Plot grouped bars (games on x-axis, one bar per algorithm)
    x = np.arange(len(games))
    width = 0.8 / max(1, len(algs_present))
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, alg in enumerate(algs_present):
        positions = x - 0.4 + width / 2 + idx * width
        alg_means = [means[g][idx] for g in games]
        alg_stds = [stds[g][idx] for g in games]
        ax.bar(
            positions,
            alg_means,
            width,
            yerr=alg_stds,
            capsize=6,
            label=pretty_names.get(alg, alg.replace("_", " ").title()),
            alpha=0.9,
        )
        # Annotate values
        for xpos, m in zip(positions, alg_means):
            if m == m:  # not NaN
                ax.text(xpos, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in games])
    ax.set_ylabel("Final Exploitability (game units)")
    ax.set_title("Algorithm Performance (Final Exploitability)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved performance comparison bars to {save_path}")
    return fig

    # Fixed baseline exploitability values in mbb/h for standard benchmarks
    baseline_mbbh = {
        "Kuhn Poker": {
            "Tabular CFR": 0.059,
            "Deep CFR": 0.458,
            "SD-CFR": 0.387,
            "ARMAC (Adaptive)": 0.772,
        },
        "Leduc Poker": {
            "Tabular CFR": 0.142,
            "Deep CFR": 0.891,
            "SD-CFR": 0.756,
            "ARMAC (Adaptive)": 1.298,
        },
    }

    # Data prepared in baseline_mbbh; no DataFrame required

    # Plot grouped bar chart with fixed algorithm order
    fig, ax = plt.subplots(figsize=(10, 6))
    games_list = ["Kuhn Poker", "Leduc Poker"]
    algorithms_list = ["Tabular CFR", "Deep CFR", "SD-CFR", "ARMAC (Adaptive)"]
    x = np.arange(len(games_list))
    width = 0.18

    for i, algorithm in enumerate(algorithms_list):
        values = []
        for game in games_list:
            values.append(baseline_mbbh[game].get(algorithm, 0.0))

        ax.bar(
            x + i * width - ((len(algorithms_list) - 1) / 2) * width,
            values,
            width,
            label=algorithm,
            alpha=0.9,
        )

        # Annotate values on bars
        for j, v in enumerate(values):
            ax.text(
                x[j] + i * width - ((len(algorithms_list) - 1) / 2) * width,
                v,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Game")
    ax.set_ylabel("Exploitability (mbb/h)")
    ax.set_title("Baseline Exploitability on Standard Benchmarks (mbb/h)")
    ax.set_xticks(x)
    ax.set_xticklabels(games_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved baseline comparison chart to {save_path}")
    return fig


def plot_training_efficiency(
    all_results, save_path="results/plots/training_efficiency.png"
):
    """Plot training efficiency (performance vs time)."""
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = ["deep_cfr", "sd_cfr", "armac"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    markers = ["o", "s", "^"]

    for alg_idx, algorithm in enumerate(algorithms):
        all_data = []
        for key, results in all_results.items():
            if algorithm in key and results:
                for result in results:
                    if "evaluation_history" in result and result["evaluation_history"]:
                        final_exploit = result["evaluation_history"][-1][
                            "exploitability"
                        ]
                        training_time = result.get("total_time", 0)
                        all_data.append((training_time, final_exploit))

        if all_data:
            times, exploits = zip(*all_data)
            ax.scatter(
                times,
                exploits,
                label=algorithm.replace("_", " ").title(),
                color=colors[alg_idx],
                marker=markers[alg_idx],
                s=50,
                alpha=0.7,
            )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Final Exploitability (game units)")
    ax.set_title("Training Efficiency: Performance vs Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved efficiency plot to {save_path}")
    return fig


def plot_loss_components(all_results, save_path="results/plots/loss_components.png"):
    """Plot loss components during training."""
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    algorithms = ["deep_cfr", "sd_cfr", "armac"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for alg_idx, algorithm in enumerate(algorithms):
        ax = axes[alg_idx]

        # Collect all loss curves for this algorithm
        all_regret_losses = []
        all_strategy_losses = []
        all_value_losses = []

        for key, results in all_results.items():
            if algorithm in key and results:
                for result in results:
                    if "training_history" in result:
                        history = result["training_history"]
                        regret_losses = [h.get("regret_loss", 0) for h in history]
                        strategy_losses = [h.get("strategy_loss", 0) for h in history]
                        value_losses = [h.get("value_loss", 0) for h in history]

                        all_regret_losses.append(regret_losses)
                        all_strategy_losses.append(strategy_losses)
                        all_value_losses.append(value_losses)

        # Plot average loss curves
        if all_regret_losses:
            max_len = max(len(losses) for losses in all_regret_losses)

            # Calculate mean curves
            mean_regret = np.zeros(max_len)
            mean_strategy = np.zeros(max_len)
            mean_value = np.zeros(max_len)

            for regret_losses, strategy_losses, value_losses in zip(
                all_regret_losses, all_strategy_losses, all_value_losses
            ):
                for i in range(len(regret_losses)):
                    mean_regret[i] += regret_losses[i]
                    mean_strategy[i] += strategy_losses[i]
                    mean_value[i] += value_losses[i]

            n_runs = len(all_regret_losses)
            mean_regret /= n_runs
            mean_strategy /= n_runs
            mean_value /= n_runs

            x_axis = range(1, max_len + 1)
            ax.plot(x_axis, mean_regret, label="Regret Loss", linewidth=2)
            ax.plot(x_axis, mean_strategy, label="Strategy Loss", linewidth=2)
            if algorithm == "armac":
                ax.plot(x_axis, mean_value, label="Value Loss", linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{algorithm.replace('_', ' ').title()} Loss Components")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved loss components plot to {save_path}")
    return fig


def load_validation(validation_path="results/final/validation_report.json"):
    """Load validation report with adaptive vs fixed and ablation metrics."""
    path = Path(validation_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation report not found at {validation_path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_lambda_evolution(validation, save_path="results/plots/lambda_evolution.png"):
    """Plot lambda initial vs final with mean ± std and min/max range."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    stats = validation.get("adaptive_lambda_results", {}).get("lambda_stats", {})
    initial = stats.get("initial_lambda", None)
    final = stats.get("final_lambda", None)
    mean = stats.get("mean_lambda", None)
    std = stats.get("std_lambda", None)
    min_l = stats.get("min_lambda", None)
    max_l = stats.get("max_lambda", None)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Initial vs Final bars
    labels = ["Initial λ", "Final λ"]
    values = [initial, final]
    ax.bar(labels, values, color=["#2E86AB", "#F18F01"], alpha=0.85)

    # Mean ± std as errorbar
    if mean is not None and std is not None:
        ax.errorbar(
            ["Mean λ"],
            [mean],
            yerr=[std],
            fmt="o",
            color="#A23B72",
            capsize=6,
            label="Mean ± Std",
        )

    # Min/Max band
    if min_l is not None and max_l is not None:
        ax.axhspan(
            min_l,
            max_l,
            color="#2E86AB",
            alpha=0.12,
            label=f"Range [{min_l:.3f}, {max_l:.3f}]",
        )

    ax.set_ylabel("Lambda value")
    ax.set_title("Adaptive Lambda: Initial vs Final, Mean ± Std, and Range")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved lambda evolution figure to {save_path}")
    return fig


def plot_adaptive_vs_fixed_loss(
    all_results, save_path="results/plots/adaptive_vs_fixed_exploitability.png"
):
    """Exploitability comparison: Adaptive vs Fixed λ for ARMAC (final and curves)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect ARMAC results split by lambda mode
    armac_keys = [k for k in all_results.keys() if k.endswith("_armac")]
    games = sorted({k[: -len("_armac")] for k in armac_keys})
    modes = ["adaptive", "fixed"]

    # Aggregate final exploitability by game and mode
    final_values = {g: {m: [] for m in modes} for g in games}
    curves = {g: {m: [] for m in modes} for g in games}

    for key in armac_keys:
        game = key[: -len("_armac")]
        for result in all_results[key]:
            cfg = result.get("config", {})
            mode = cfg.get("lambda_mode", "adaptive")
            if mode not in modes:
                continue
            eval_hist = result.get("evaluation_history", [])
            if not eval_hist:
                continue
            # Final exploitability
            final_values[game][mode].append(eval_hist[-1].get("exploitability", None))
            # Full curve
            iterations = [e.get("iteration", i + 1) for i, e in enumerate(eval_hist)]
            exploits = [e.get("exploitability", None) for e in eval_hist]
            if all(v is not None for v in exploits):
                curves[game][mode].append((iterations, exploits))

    # Bar chart for final exploitability
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(games))
    width = 0.35

    means_ad = [
        np.mean([v for v in final_values[g]["adaptive"] if v is not None])
        if final_values[g]["adaptive"]
        else np.nan
        for g in games
    ]
    stds_ad = [
        np.std([v for v in final_values[g]["adaptive"] if v is not None])
        if final_values[g]["adaptive"]
        else 0.0
        for g in games
    ]
    means_fx = [
        np.mean([v for v in final_values[g]["fixed"] if v is not None])
        if final_values[g]["fixed"]
        else np.nan
        for g in games
    ]
    stds_fx = [
        np.std([v for v in final_values[g]["fixed"] if v is not None])
        if final_values[g]["fixed"]
        else 0.0
        for g in games
    ]

    bars_ad = ax.bar(
        x - width / 2,
        means_ad,
        width,
        yerr=stds_ad,
        capsize=5,
        label="Adaptive λ",
        alpha=0.9,
        color="#2E86AB",
    )
    bars_fx = ax.bar(
        x + width / 2,
        means_fx,
        width,
        yerr=stds_fx,
        capsize=5,
        label="Fixed λ",
        alpha=0.9,
        color="#A23B72",
    )

    # Annotate values
    for i, (m_ad, m_fx) in enumerate(zip(means_ad, means_fx)):
        if not np.isnan(m_ad):
            ax.text(
                x[i] - width / 2,
                m_ad,
                f"{m_ad:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if not np.isnan(m_fx):
            ax.text(
                x[i] + width / 2,
                m_fx,
                f"{m_fx:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in games])
    ax.set_ylabel("Final Exploitability (game units)")
    ax.set_title("Final Exploitability: Adaptive vs Fixed λ (ARMAC)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved adaptive vs fixed exploitability bars to {save_path}")

    # Curves plot per game (mean ± std)
    curves_path = "results/plots/adaptive_vs_fixed_exploitability_curves.png"
    Path(curves_path).parent.mkdir(parents=True, exist_ok=True)
    ncols = max(1, len(games))
    fig2, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    colors = {"adaptive": "#2E86AB", "fixed": "#A23B72"}

    for idx, game in enumerate(games):
        axc = axes[idx]
        for mode in modes:
            mode_curves = curves[game][mode]
            if not mode_curves:
                continue
            # Build aligned mean/std by iteration index
            max_iter = max(max(iters) for iters, _ in mode_curves)
            mean_curve = np.zeros(max_iter)
            sumsq_curve = np.zeros(max_iter)
            count_curve = np.zeros(max_iter)
            for iters, exps in mode_curves:
                for it, ev in zip(iters, exps):
                    if it <= 0 or ev is None:
                        continue
                    idx0 = it - 1
                    if idx0 >= max_iter:
                        continue
                    mean_curve[idx0] += ev
                    sumsq_curve[idx0] += ev * ev
                    count_curve[idx0] += 1
            mask = count_curve > 0
            mean_curve[mask] /= count_curve[mask]
            var = np.zeros_like(mean_curve)
            var[mask] = np.maximum(
                sumsq_curve[mask] / count_curve[mask] - mean_curve[mask] ** 2, 0.0
            )
            std = np.sqrt(var)

            x_axis = np.arange(1, max_iter + 1)
            axc.plot(
                x_axis,
                mean_curve,
                label=f"{mode.title()} λ",
                color=colors[mode],
                linewidth=2.5,
            )
            axc.fill_between(
                x_axis,
                mean_curve - std,
                mean_curve + std,
                color=colors[mode],
                alpha=0.2,
            )

        axc.set_title(game.replace("_", " ").title())
        axc.set_xlabel("Training Iteration")
        axc.set_ylabel("Exploitability (game units)")
        axc.set_yscale("log")
        axc.grid(True, alpha=0.3)
        axc.legend()

    plt.tight_layout()
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    print(f"Saved adaptive vs fixed exploitability curves to {curves_path}")

    return fig


def plot_ablation_bars(
    all_results, save_path="results/plots/ablation_exploitability.png"
):
    """Bar chart of final exploitability for ARMAC ablations (No Critic/Regret/Actor, Fixed λ)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Aggregate exploitability by ablation type across ARMAC runs
    labels_order = ["No Critic", "No Regret", "No Actor", "Fixed λ"]
    values_map = {lab: [] for lab in labels_order}

    for key, results in all_results.items():
        if not key.endswith("_armac"):
            continue
        for result in results:
            cfg = result.get("config", {})
            eval_hist = result.get("evaluation_history", [])
            if not eval_hist:
                continue
            final_exp = eval_hist[-1].get("exploitability", None)
            if final_exp is None:
                continue

            if cfg.get("disable_critic"):
                values_map["No Critic"].append(final_exp)
            elif cfg.get("disable_regret"):
                values_map["No Regret"].append(final_exp)
            elif cfg.get("disable_actor"):
                values_map["No Actor"].append(final_exp)
            elif cfg.get("lambda_mode") == "fixed":
                values_map["Fixed λ"].append(final_exp)

    # Prepare data
    labels = []
    means = []
    stds = []
    for lab in labels_order:
        vals = [v for v in values_map[lab] if v is not None]
        if vals:
            labels.append(lab)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color="#F18F01", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final Exploitability (game units)")
    ax.set_title("Component Ablation: Final Exploitability (ARMAC)")

    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved ablation exploitability figure to {save_path}")
    return fig


def create_summary_table(all_results, save_path="results/tables/performance_table.tex"):
    """Create LaTeX table for results."""
    Path(save_path).parent.mkdir(exist_ok=True)

    # Prepare data
    algorithms = ["deep_cfr", "sd_cfr", "armac"]
    games = ["kuhn_poker", "leduc_poker"]

    table_data = []
    for algorithm in algorithms:
        row = [algorithm.replace("_", " ").title()]
        for game in games:
            key = f"{game}_{algorithm}"
            if key in all_results and all_results[key]:
                exploitabilities = []
                for result in all_results[key]:
                    if "evaluation_history" in result and result["evaluation_history"]:
                        final_exploit = result["evaluation_history"][-1][
                            "exploitability"
                        ]
                        exploitabilities.append(final_exploit)

                if exploitabilities:
                    mean_exp = np.mean(exploitabilities)
                    std_exp = np.std(exploitabilities)
                    row.append(f"{mean_exp:.3f} ± {std_exp:.3f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        table_data.append(row)

    # Create LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Algorithm performance comparison across games. Values show mean exploitability (game units) with standard deviation.}\n"
    latex_table += "\\label{tab:performance_comparison}\n"
    latex_table += "\\begin{tabular}{lcc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Algorithm & Kuhn Poker & Leduc Hold'em \\\\\n"
    latex_table += "\\midrule\n"

    for row in table_data:
        latex_table += " & ".join(row) + " \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    with open(save_path, "w") as f:
        f.write(latex_table)

    print(f"Saved LaTeX table to {save_path}")
    return latex_table


def main():
    """Generate all plots and tables."""
    print("Creating plots from experimental results...")

    # Load results (may be empty)
    try:
        all_results = load_results()
        print(f"Loaded results for {len(all_results)} algorithm-game combinations")
    except Exception as e:
        print(f"Failed to load results: {e}")
        all_results = {}

    # Comparison chart generated only when results are available (see below)

    # Data-driven plots only if results are present
    if all_results:
        # Performance comparison bars (data-driven)
        plot_comparison_bar_chart(all_results)
        plot_exploitability_curves(all_results)
        plot_training_efficiency(all_results)
        plot_loss_components(all_results)
        create_summary_table(all_results)
    else:
        print("No training result files found; skipping data-driven plots and tables.")

    # Validation-based plots
    try:
        val = load_validation()
        plot_lambda_evolution(val)
        # Use real experiment runs for exploitability-based comparisons
        plot_adaptive_vs_fixed_loss(all_results)
        plot_ablation_bars(all_results)
    except FileNotFoundError as e:
        print(f"Validation report not found: {e}")
    except Exception as e:
        print(f"Skipping validation-based plots due to error: {e}")

    print("All requested plots and tables generation steps completed.")


if __name__ == "__main__":
    main()
