#!/usr/bin/env python3
"""Produce LaTeX-ready tables from experiment summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def _policy_display_name(name: str) -> str:
    """Return a human-friendly representation for a policy identifier."""

    if not name:
        return "Unknown"
    tokens = name.replace("-", " ").replace("_", " ").split()
    return " ".join(token.capitalize() for token in tokens)


def _game_display_name(name: str) -> str:
    if not name:
        return "Game"
    pretty = name.replace("_", " ")
    if pretty.lower().endswith("poker"):
        pretty = pretty[:-5] + "Poker"
    return pretty.title()


def _format_value(stats: Dict[str, object]) -> str:
    """Format mean ± spread for exploitability values."""

    mean = float(stats.get("mean_exploitability", 0.0) or 0.0)
    std = float(stats.get("stdev_exploitability", 0.0) or 0.0)
    if std == 0.0:
        ci = float(stats.get("ci95_exploitability", 0.0) or 0.0)
        if ci != 0.0:
            std = ci
    return f"{mean:.3f} \\pm {std:.3f}" if std or mean else "--"


def _has_metrics(stats: Dict[str, object]) -> bool:
    return any(key in stats for key in ("mean_exploitability", "mean_nash_conv", "mean_exploit_auc"))


def _extract_policy_stats(summary: Dict[str, object]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    games_payload = summary.get("games", {})
    games: List[str] = []
    policies: Dict[str, Dict[str, Dict[str, float]]] = {}

    for game, payload in games_payload.items():
        if not isinstance(payload, dict):
            continue
        games.append(game)

        containers: List[Dict[str, object]] = []
        for key in ("policies", "policy_types", "policy types"):
            container = payload.get(key)
            if isinstance(container, dict):
                containers.append(container)
        if not containers:
            containers.append(payload)

        added = False
        for container in containers:
            for policy, stats in container.items():
                if not isinstance(stats, dict) or not _has_metrics(stats):
                    continue
                policies.setdefault(policy, {})[game] = stats
                added = True

        if not added and _has_metrics(payload):
            policies.setdefault("aggregate", {})[game] = payload

    games = sorted(set(games))
    return games, policies


def _aggregate_stats(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    runs = summary.get("runs", [])
    per_game: Dict[str, List[float]] = {}
    for run in runs:
        game = run.get("game")
        value = run.get("final_exploitability")
        if game is None or value is None:
            continue
        per_game.setdefault(str(game), []).append(float(value))

    aggregates: Dict[str, Dict[str, float]] = {}
    for game, values in per_game.items():
        if not values:
            continue
        mean_val = statistics.fmean(values)
        if len(values) > 1:
            stdev_val = statistics.stdev(values)
        else:
            stdev_val = 0.0
        aggregates[game] = {
            "mean_exploitability": mean_val,
            "stdev_exploitability": stdev_val,
            "ci95_exploitability": 1.96 * stdev_val / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        }
    return aggregates


def make_table(summary: Dict[str, object]) -> str:
    games, policies = _extract_policy_stats(summary)
    aggregate = _aggregate_stats(summary)
    if aggregate:
        policies.setdefault("aggregate", {}).update(aggregate)
    if not games or not policies:
        return (
            "\\begin{tabular}{l}\n"
            "\\toprule\n"
            "No data \\ \n"
            "\\midrule\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )

    column_spec = "l" + "r" * len(games)
    lines = [f"\\begin{{tabular}}{{{column_spec}}}\n", "\\toprule\n"]
    header_cells = ["Algorithm"] + [_game_display_name(game) for game in games]
    header_line = " & ".join(header_cells)
    lines.append(f"{header_line} \\ \n")
    lines.append("\\midrule\n")

    ordered_policies = sorted(policies.keys())
    if "aggregate" in policies:
        ordered_policies = ["aggregate"] + [p for p in ordered_policies if p != "aggregate"]

    for policy in ordered_policies:
        display_policy = _policy_display_name(policy)
        cells = [display_policy]
        for game in games:
            stats = policies[policy].get(game)
            cells.append(_format_value(stats) if stats else "--")
        row_line = " & ".join(cells)
        lines.append(f"{row_line} \\ \n")

    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
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
