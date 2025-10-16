#!/usr/bin/env python3
"""Aggregate experiment artifacts produced by ``run_real_training.py``.

The utility scans a results directory (``results/`` by default) for JSON runs
and produces an aggregate manifest containing per-game statistics plus the full
exploitability/NashConv curves. Use ``--results-dir`` to target alternative
folders (e.g. ``results/submission_runs``) and ``--output`` to control the
manifest location.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_SUMMARY_FILE = DEFAULT_RESULTS_DIR / "experiment_summary.json"


def load_run(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    history = data.get("training_history", [])
    if not history:
        raise ValueError(f"{path} does not contain any training history.")

    final = history[-1]
    if "iterations" not in data:
        raise ValueError("Missing iterations field; skipping legacy result")
    return {
        "path": str(path),
        "game": data["game"],
        "seed": int(data["seed"]),
        "iterations": int(data["iterations"]),
        "final_exploitability": float(final["exploitability"]),
        "final_nash_conv": float(final["nash_conv"]),
        "history": history,
        "policy_type": data.get("policy_type", "unknown"),
    }


def discover_runs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("*_seed*.json"))


def aggregate(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(run["game"], []).append(run)

    summary: Dict[str, Dict[str, float]] = {}
    for game, game_runs in grouped.items():
        exploitabilities = [float(run["final_exploitability"]) for run in game_runs]
        nash_convs = [float(run["final_nash_conv"]) for run in game_runs]
        summary[game] = {
            "num_runs": len(game_runs),
            "iterations": int(game_runs[0]["iterations"]),
            "mean_exploitability": statistics.fmean(exploitabilities),
            "stdev_exploitability": statistics.pstdev(exploitabilities) if len(exploitabilities) > 1 else 0.0,
            "mean_nash_conv": statistics.fmean(nash_convs),
            "stdev_nash_conv": statistics.pstdev(nash_convs) if len(nash_convs) > 1 else 0.0,
            "policy_types": {
                policy_type: sum(1 for run in game_runs if run.get("policy_type", "unknown") == policy_type)
                for policy_type in {run.get("policy_type", "unknown") for run in game_runs}
            },
        }
    return summary


def build_payload(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary = aggregate(runs)
    payload = {
        "num_runs": len(runs),
        "games": summary,
        "runs": [
            {
                "path": run["path"],
                "game": run["game"],
                "seed": run["seed"],
                "iterations": run["iterations"],
                "final_exploitability": run["final_exploitability"],
                "final_nash_conv": run["final_nash_conv"],
                "exploitability_curve": [float(step["exploitability"]) for step in run["history"]],
                "nash_conv_curve": [float(step["nash_conv"]) for step in run["history"]],
                "scheduler_loss_curve": [float(step.get("scheduler_loss", 0.0)) for step in run["history"]],
                "policy_type": run.get("policy_type", "unknown"),
            }
            for run in runs
        ],
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate JSON runs produced by run_real_training.py."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to scan for *_seed*.json files (default: results/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to the output summary JSON "
            "(default: <results-dir>/experiment_summary.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    opts = parse_args()
    results_dir: Path = opts.results_dir
    output_path: Path = (
        opts.output
        if opts.output is not None
        else (results_dir / DEFAULT_SUMMARY_FILE.name)
    )

    paths = discover_runs(results_dir)
    if not paths:
        print(
            f"No real experiment runs found under '{results_dir}'. Nothing to summarise."
        )
        return

    runs = []
    for path in paths:
        try:
            runs.append(load_run(path))
        except ValueError:
            continue
    payload = build_payload(runs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote summary for {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
