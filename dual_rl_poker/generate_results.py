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
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_SUMMARY_FILE = DEFAULT_RESULTS_DIR / "experiment_summary.json"


def _trapezoid(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum((values[i] + values[i - 1]) * 0.5 for i in range(1, len(values)))


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean_val = statistics.fmean(values)
    if len(values) > 1:
        stdev_val = statistics.stdev(values)
        ci = 1.96 * stdev_val / math.sqrt(len(values))
    else:
        stdev_val = 0.0
        ci = 0.0
    return mean_val, stdev_val, ci


def load_run(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    history = data.get("training_history", [])
    if not history:
        raise ValueError(f"{path} does not contain any training history.")

    final = history[-1]
    exploit_curve = [float(step["exploitability"]) for step in history]
    nash_curve = [float(step["nash_conv"]) for step in history]
    gate_entropy_curve = [float(step.get("gate_entropy", 0.0)) for step in history if "gate_entropy" in step]
    gate_prob_keys = [key for key in final.keys() if key.startswith("gate_prob_")]
    gate_prob_curves = {
        key: [float(step.get(key, 0.0)) for step in history]
        for key in gate_prob_keys
    }
    iterations_curve = [int(step.get("iteration", idx + 1)) for idx, step in enumerate(history)]

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
        "exploit_curve": exploit_curve,
        "nash_curve": nash_curve,
        "gate_entropy_curve": gate_entropy_curve,
        "gate_prob_curves": gate_prob_curves,
        "iterations_curve": iterations_curve,
        "exploit_auc": _trapezoid(exploit_curve),
        "nash_auc": _trapezoid(nash_curve),
        "final_gate_entropy": gate_entropy_curve[-1] if gate_entropy_curve else None,
        "final_gate_probs": {key: float(final.get(key, 0.0)) for key in gate_prob_keys},
        "policy_type": data.get("policy_type", "unknown"),
    }


def discover_runs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("*_seed*.json"))


def aggregate(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for run in runs:
        key = (run["game"], run.get("policy_type", "unknown"))
        grouped.setdefault(key, []).append(run)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (game, policy), game_runs in grouped.items():
        exploitabilities = [float(run["final_exploitability"]) for run in game_runs]
        nash_convs = [float(run["final_nash_conv"]) for run in game_runs]
        aucs = [float(run["exploit_auc"]) for run in game_runs]
        gate_entropies = [float(run["final_gate_entropy"]) for run in game_runs if run.get("final_gate_entropy") is not None]
        mean_exp, std_exp, ci_exp = _mean_std_ci(exploitabilities)
        mean_nash, std_nash, ci_nash = _mean_std_ci(nash_convs)
        mean_auc, std_auc, ci_auc = _mean_std_ci(aucs)
        mean_gate, std_gate, ci_gate = _mean_std_ci(gate_entropies) if gate_entropies else (0.0, 0.0, 0.0)

        policy_entry = {
            "num_runs": len(game_runs),
            "iterations": int(game_runs[0]["iterations"]),
            "mean_exploitability": mean_exp,
            "stdev_exploitability": std_exp,
            "ci95_exploitability": ci_exp,
            "mean_nash_conv": mean_nash,
            "stdev_nash_conv": std_nash,
            "ci95_nash_conv": ci_nash,
            "mean_exploit_auc": mean_auc,
            "stdev_exploit_auc": std_auc,
            "ci95_exploit_auc": ci_auc,
            "mean_gate_entropy": mean_gate,
            "stdev_gate_entropy": std_gate,
            "ci95_gate_entropy": ci_gate,
        }
        summary.setdefault(game, {})[policy] = policy_entry
    return summary


def build_payload(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary = aggregate(runs)
    policy_counts: Dict[str, int] = {}
    for run in runs:
        policy = run.get("policy_type", "unknown")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
    payload = {
        "num_runs": len(runs),
        "games": summary,
        "policy_counts": policy_counts,
        "runs": [
            {
                "path": run["path"],
                "game": run["game"],
                "seed": run["seed"],
                "iterations": run["iterations"],
                "final_exploitability": run["final_exploitability"],
                "final_nash_conv": run["final_nash_conv"],
                "iterations_curve": run["iterations_curve"],
                "exploitability_curve": run["exploit_curve"],
                "nash_conv_curve": run["nash_curve"],
                "gate_entropy_curve": run["gate_entropy_curve"],
                "gate_prob_curves": run["gate_prob_curves"],
                "scheduler_loss_curve": [float(step.get("scheduler_loss", 0.0)) for step in run["history"]],
                "exploit_auc": run["exploit_auc"],
                "nash_auc": run["nash_auc"],
                "final_gate_entropy": run.get("final_gate_entropy"),
                "final_gate_probs": run.get("final_gate_probs", {}),
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
