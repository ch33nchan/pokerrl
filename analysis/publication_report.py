#!/usr/bin/env python3
"""Produce LaTeX-ready tables from experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _format_ci(mean: float, ci: float) -> str:
    return f"{mean:.4f} \\pm {ci:.4f}"


def _has_metrics(stats: Dict[str, object]) -> bool:
    return any(key in stats for key in ("mean_exploitability", "mean_nash_conv", "mean_exploit_auc"))


def build_rows(summary: Dict[str, object]) -> Iterable[Tuple[str, str, Dict[str, float]]]:
    """Yield (game, policy, stats) triples from historical or new summaries."""

    games = summary.get("games", {})
    for game, payload in games.items():
        if not isinstance(payload, dict):
            continue

        rows: List[Tuple[str, str, Dict[str, float]]] = []

        def add_entries(container: Dict[str, object]) -> None:
            for policy, stats in container.items():
                if not isinstance(stats, dict):
                    continue
                if not _has_metrics(stats):
                    continue
                rows.append((game, policy, stats))

        if "policies" in payload and isinstance(payload["policies"], dict):
            add_entries(payload["policies"])
        elif "policy_types" in payload and isinstance(payload["policy_types"], dict):
            add_entries(payload["policy_types"])
        elif "policy types" in payload and isinstance(payload["policy types"], dict):
            add_entries(payload["policy types"])
        else:
            add_entries(payload)

        if not rows and _has_metrics(payload):
            rows.append((game, "aggregate", payload))

        for row in rows:
            yield row


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
