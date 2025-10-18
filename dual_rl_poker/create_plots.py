#!/usr/bin/env python3
"""Generate publication-ready plots for poker experiments."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)
plt.style.use("seaborn-v0_8-darkgrid")


RunHistory = List[Dict[str, float]]


def load_runs(results_dir: Path) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    runs: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for path in results_dir.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        history = data.get("training_history")
        if not history:
            continue
        game = data.get("game")
        policy = data.get("policy_type", "unknown")
        if not game:
            continue
        runs[game][policy].append(
            {
                "seed": data.get("seed", -1),
                "iterations": data.get("iterations", len(history)),
                "history": history,
                "path": str(path),
            }
        )
    return runs


def _extract_matrix(runs: Sequence[Dict[str, object]], metric: str) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if not runs:
        return None, None
    max_iter = max(int(run.get("iterations", 0)) for run in runs)
    if max_iter <= 0:
        max_iter = max(len(run["history"]) for run in runs)
    matrix = np.full((len(runs), max_iter), np.nan, dtype=float)
    for row, run in enumerate(runs):
        for step in run["history"]:
            iteration = int(step.get("iteration", 0))
            if iteration <= 0 or iteration > max_iter:
                continue
            if metric in step:
                matrix[row, iteration - 1] = float(step[metric])
    valid_cols = ~np.isnan(matrix).all(axis=0)
    if not valid_cols.any():
        return None, None
    matrix = matrix[:, valid_cols]
    iterations = np.arange(1, matrix.shape[1] + 1)
    return iterations, matrix


def _mean_and_ci(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = ~np.isnan(matrix)
    counts = valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
    std = np.zeros_like(mean)
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        col_valid = ~np.isnan(col_values)
        n = int(col_valid.sum())
        if n > 1:
            deviations = col_values[col_valid] - col_values[col_valid].mean()
            variance = (deviations ** 2).sum() / (n - 1)
            std[col] = math.sqrt(max(variance, 0.0))
        else:
            std[col] = 0.0
    counts_safe = np.maximum(counts, 1)
    ci = 1.96 * std / np.sqrt(counts_safe)
    lower = mean - ci
    upper = mean + ci
    return mean, lower, upper, counts


def _metric_title(metric: str) -> str:
    if metric.lower() == "nash_conv":
        return "Nash Conv"
    return metric.replace("_", " ").title()


def plot_metric(
    output_dir: Path,
    game: str,
    metric: str,
    grouped_runs: Dict[str, List[Dict[str, object]]],
    *,
    dpi: int,
    formats: Sequence[str],
    include_individual: bool,
    log_scale_metrics: Iterable[str],
) -> None:
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = False
    for idx, (policy, runs) in enumerate(sorted(grouped_runs.items())):
        iterations, matrix = _extract_matrix(runs, metric)
        if iterations is None or matrix is None:
            continue
        mean, lower, upper, counts = _mean_and_ci(matrix)
        color = palette(idx % palette.N)
        if include_individual:
            for row_idx, curve in enumerate(matrix):
                ax.plot(
                    iterations,
                    curve,
                    color=color,
                    alpha=0.25,
                    linewidth=1.0,
                )
        ax.plot(
            iterations,
            mean,
            color=color,
            linewidth=2.4,
            label=f"{policy} mean (n={int(counts.max())})",
        )
        ax.fill_between(
            iterations,
            lower,
            upper,
            color=color,
            alpha=0.18,
            label=f"{policy} 95% CI",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Iteration")
    ax.set_ylabel(_metric_title(metric))
    ax.set_title(f"{game.replace('_', ' ').title()} — {_metric_title(metric)}")
    if metric in log_scale_metrics:
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(output_dir / f"{game}_{metric}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploitability/NashConv plots with publication quality styling."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/submission_suite"),
        help="Root directory containing per-run JSON logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to write figures to.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="exploitability,nash_conv,gate_entropy",
        help="Comma separated list of metrics to visualise.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution for saved figures.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma separated list of output formats (e.g. png,pdf).",
    )
    parser.add_argument(
        "--include-individual-curves",
        action="store_true",
        help="Overlay individual seed curves alongside the mean ± CI band.",
    )
    parser.add_argument(
        "--log-scale",
        type=str,
        default="exploitability",
        help="Comma separated metrics to render on a log scale.",
    )
    return parser.parse_args()


def main() -> None:
    opts = parse_args()
    metrics = [metric.strip() for metric in opts.metrics.split(",") if metric.strip()]
    formats = [fmt.strip() for fmt in opts.formats.split(",") if fmt.strip()]
    log_scale_metrics = {metric.strip() for metric in opts.log_scale.split(",") if metric.strip()}

    runs = load_runs(opts.results_dir)
    if not runs:
        print(f"No JSON runs with training history found under {opts.results_dir}")
        return

    for game, policies in runs.items():
        for metric in metrics:
            plot_metric(
                opts.output_dir,
                game,
                metric,
                policies,
                dpi=opts.dpi,
                formats=formats,
                include_individual=opts.include_individual_curves,
                log_scale_metrics=log_scale_metrics,
            )

    print(f"Wrote publication-quality plots to {opts.output_dir}")


if __name__ == "__main__":
    main()
