#!/usr/bin/env python3
"""Produce LaTeX-ready tables from experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _format_ci(mean: float, ci: float) -> str:
    return f"{mean:.4f} \\pm {ci:.4f}"


def build_rows(summary: Dict[str, object]) -> Iterable[Tuple[str, str, Dict[str, float]]]:
    games = summary.get("games", {})
    for game, policies in games.items():
        for policy, stats in policies.items():
            yield game, policy, stats


def make_table(summary: Dict[str, object]) -> str:
    header = (
        "\\begin{tabular}{l l r r r}\\toprule\n"
        "Game & Policy & Exploitability & NashConv & AUC \\\\ \\midrule\n"
    )
    lines = [header]
    for game, policy, stats in build_rows(summary):
        exp = _format_ci(stats.get("mean_exploitability", 0.0), stats.get("ci95_exploitability", 0.0))
        nash = _format_ci(stats.get("mean_nash_conv", 0.0), stats.get("ci95_nash_conv", 0.0))
        auc = _format_ci(stats.get("mean_exploit_auc", 0.0), stats.get("ci95_exploit_auc", 0.0))
        lines.append(f"{game.replace('_', ' ')} & {policy.replace('_', ' ')} & {exp} & {nash} & {auc} \\\\ \n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert experiment summary JSON into a LaTeX table.")
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to experiment_summary.json produced by generate_results.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table to (defaults to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    opts = parse_args()
    with opts.summary.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    table = make_table(summary)
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(table, encoding="utf-8")
        print(f"Wrote LaTeX table to {opts.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
