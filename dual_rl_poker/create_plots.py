#!/usr/bin/env python3
"""Quick plotting script for ARMAC++ poker experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")


def load_runs(results_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for path in results_dir.rglob("*.json"):
        try:
            with path.open() as fh:
                data = json.load(fh)
            runs.append(
                {
                    "path": str(path),
                    "game": data["game"],
                    "seed": data["seed"],
                    "policy": data["policy_type"],
                    "iterations": data["iterations"],
                    "exploit_curve": [item["exploitability"] for item in data["training_history"]],
                }
            )
        except Exception:
            continue
    return runs


def plot_exploitability(runs: List[Dict[str, object]], output_dir: Path) -> None:
    grouped = defaultdict(list)
    for run in runs:
        key = (run["game"], run["policy"])
        grouped[key].append(run)

    output_dir.mkdir(parents=True, exist_ok=True)

    for (game, policy), entries in grouped.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        curves = []
        for run in sorted(entries, key=lambda r: r["seed"]):
            curve = run["exploit_curve"]
            curves.append(curve)
            ax.plot(
                curve,
                linewidth=1.5,
                alpha=0.5,
                label=f"seed {run['seed']}"
            )

        lengths = {len(curve) for curve in curves}
        if len(curves) > 1 and len(lengths) == 1:
            arr = np.array(curves)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(mean, color="black", linewidth=2.4, label="mean")
            ax.fill_between(
                np.arange(len(mean)),
                mean - std,
                mean + std,
                color="black",
                alpha=0.12,
                label="mean ± std"
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"{policy} on {game}")
        ax.legend(loc="upper right", frameon=True)

        output_path = output_dir / f"{game}_{policy}_exploitability.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploitability plots for ARMAC++ experiments.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/submission_suite"),
        help="Root directory containing experiment JSON logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to write plots into.",
    )
    opts = parser.parse_args()

    runs = load_runs(opts.results_dir)
    if not runs:
        print(f"No JSON runs found under {opts.results_dir}")
        return
    plot_exploitability(runs, opts.output_dir)
    print(f"Wrote plots to {opts.output_dir}")


if __name__ == "__main__":
    main()
