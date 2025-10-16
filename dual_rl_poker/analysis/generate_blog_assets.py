#!/usr/bin/env python3
"""Generate blog-ready tables and plots from submission results.

The script expects a summary JSON (default: results/submission_summary.json)
containing the structure emitted by ``generate_results.py``.  It produces:

* results/plots/exploitability_bar.png
* results/plots/exploitability_curves.png
* results/blog_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_runs(summary_path: Path) -> List[Dict]:
    data = json.loads(summary_path.read_text())
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {summary_path}")
    return runs


def group_runs(runs: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for run in runs:
        key = (run["game"], run.get("policy_type", "unknown"))
        grouped.setdefault(key, []).append(run)
    return grouped


def ensure_dirs() -> None:
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)


def make_bar_plot(grouped: Dict[Tuple[str, str], List[Dict]], output: Path) -> None:
    games = ["kuhn_poker", "leduc_poker"]
    policies = ["neural_tabular", "cfr"]
    pretty_game = {"kuhn_poker": "Kuhn Poker", "leduc_poker": "Leduc Poker"}
    pretty_policy = {"neural_tabular": "Neural ARMAC", "cfr": "CFR"}

    width = 0.35
    x = np.arange(len(games))

    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, policy in enumerate(policies):
        means = []
        stds = []
        for game in games:
            runs = grouped.get((game, policy), [])
            values = [run["final_exploitability"] for run in runs]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (idx - 0.5) * width
        ax.bar(x + offset, means, width, label=pretty_policy[policy], yerr=stds, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_game[g] for g in games])
    ax.set_ylabel("Final Exploitability")
    ax.set_title("Exploitability after Rust-backed Training")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_curve_plot(grouped: Dict[Tuple[str, str], List[Dict]], output: Path) -> None:
    games = ["kuhn_poker", "leduc_poker"]
    policies = ["neural_tabular", "cfr"]
    pretty_game = {"kuhn_poker": "Kuhn Poker", "leduc_poker": "Leduc Poker"}
    pretty_policy = {"neural_tabular": "Neural ARMAC", "cfr": "CFR"}
    colors = {"neural_tabular": "#2E86AB", "cfr": "#F18F01"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, game in zip(axes, games):
        for policy in policies:
            runs = grouped.get((game, policy), [])
            if not runs:
                continue
            curves = [np.array(run["exploitability_curve"], dtype=float) for run in runs]
            max_len = max(len(curve) for curve in curves)
            padded = []
            for curve in curves:
                if len(curve) < max_len:
                    pad = np.full(max_len, np.nan)
                    pad[: len(curve)] = curve
                    padded.append(pad)
                else:
                    padded.append(curve)
            stacked = np.vstack(padded)
            mean_curve = np.nanmean(stacked, axis=0)
            std_curve = np.nanstd(stacked, axis=0)
            iterations = np.arange(1, len(mean_curve) + 1)
            ax.plot(iterations, mean_curve, label=pretty_policy[policy], color=colors[policy], linewidth=2)
            ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, color=colors[policy], alpha=0.2)
        ax.set_title(pretty_game[game])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(grouped: Dict[Tuple[str, str], List[Dict]], output: Path) -> None:
    games = ["kuhn_poker", "leduc_poker"]
    policies = ["neural_tabular", "cfr"]
    pretty_game = {"kuhn_poker": "Kuhn Poker", "leduc_poker": "Leduc Poker"}
    pretty_policy = {"neural_tabular": "Neural ARMAC", "cfr": "CFR"}

    lines = ["| Game | Policy | Final Exploitability (mean ± std) |", "| --- | --- | --- |"]
    for game in games:
        for policy in policies:
            runs = grouped.get((game, policy), [])
            if not runs:
                continue
            values = [run["final_exploitability"] for run in runs]
            mean = np.mean(values)
            std = np.std(values)
            lines.append(
                f"| {pretty_game[game]} | {pretty_policy[policy]} | {mean:.4f} ± {std:.4f} |"
            )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blog plots/tables from submission summary.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/submission_summary.json"),
        help="Path to the summary JSON.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"), help="Directory for plots.")
    args = parser.parse_args()

    runs = load_runs(args.summary)
    grouped = group_runs(runs)
    ensure_dirs()

    make_bar_plot(grouped, Path("results/plots/exploitability_bar.png"))
    make_curve_plot(grouped, Path("results/plots/exploitability_curves.png"))
    write_table(grouped, Path("results/blog_table.md"))


if __name__ == "__main__":
    main()
