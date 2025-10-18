#!/usr/bin/env python3
"""Convenience runner for the canonical ARMAC++ poker experiment suite.

This script orchestrates the baseline runs required for reproducible reports:

  • Neural ARMAC on Kuhn & Leduc poker (seeds 0–4)
  • CFR anchors on Kuhn & Leduc poker
  • Aggregation of results & plot generation

The underlying training binary already exposes rich per-iteration progress bars.
This wrapper adds run-level progress tracking and ensures outputs land in a
structured hierarchy tagged with experiment names and timestamps.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "run_real_training.py"
GENERATE_RESULTS = PROJECT_ROOT / "generate_results.py"
CREATE_PLOTS = PROJECT_ROOT / "create_plots.py"


def build_command(base: Sequence[str], extra: Iterable[str]) -> List[str]:
    cmd = list(base)
    cmd.extend(extra)
    return cmd


def run_command(cmd: Sequence[str]) -> None:
    process = subprocess.Popen(cmd)
    try:
        exit_code = process.wait()
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        process.terminate()
        process.wait()
        raise
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}: {' '.join(cmd)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the canonical ARMAC++ poker experiment suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "poker_suite",
        help="Root directory in which all run artefacts will be stored.",
    )
    parser.add_argument(
        "--backend",
        choices=["pyspiel", "rust"],
        default="rust",
        help="Environment backend for neural ARMAC runs (default: rust).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Iterations for neural ARMAC runs (default: 500).",
    )
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=128,
        help="Episodes per iteration for neural ARMAC runs (default: 128).",
    )
    parser.add_argument(
        "--cfr-iterations",
        type=int,
        default=1000,
        help="Iterations for CFR baselines (default: 1000).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed list for neural ARMAC runs (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment bucket; defaults to poker_suite_<timestamp> if not provided.",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="If set, skip aggregation/plotting steps.",
    )

    opts = parser.parse_args(argv)

    timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = opts.experiment_name or f"poker_suite_{timestamp_suffix}"

    neural_jobs: List[Tuple[str, int]] = [(game, seed) for game in ("kuhn_poker", "leduc_poker") for seed in opts.seeds]
    cfr_jobs: List[Tuple[str, int]] = [(game, 0) for game in ("kuhn_poker", "leduc_poker")]

    run_label = f"iter{opts.iterations}_epi{opts.episodes_per_iteration}"

    # Execute neural ARMAC runs
    neural_desc = "Neural ARMAC runs"
    neural_progress = (
        tqdm(neural_jobs, total=len(neural_jobs), desc=neural_desc, unit="run", dynamic_ncols=True)
        if tqdm is not None
        else neural_jobs
    )
    for game, seed in neural_progress:
        cmd = [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--game",
            game,
            "--iterations",
            str(opts.iterations),
            "--episodes-per-iteration",
            str(opts.episodes_per_iteration),
            "--seed",
            str(seed),
            "--backend",
            opts.backend,
            "--output-dir",
            str(opts.output_dir),
            "--experiment-name",
            experiment_name,
            "--run-label",
            run_label,
            "--tag",
            "neural",
        ]
        run_command(cmd)

    # Execute CFR baselines
    cfr_desc = "CFR anchor runs"
    cfr_progress = (
        tqdm(cfr_jobs, total=len(cfr_jobs), desc=cfr_desc, unit="run", dynamic_ncols=True)
        if tqdm is not None
        else cfr_jobs
    )
    for game, seed in cfr_progress:
        cmd = [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--game",
            game,
            "--algorithm",
            "cfr",
            "--iterations",
            str(opts.cfr_iterations),
            "--seed",
            str(seed),
            "--output-dir",
            str(opts.output_dir),
            "--experiment-name",
            experiment_name,
            "--run-label",
            f"cfr_iter{opts.cfr_iterations}",
            "--tag",
            "cfr",
        ]
        run_command(cmd)

    # Aggregate
    if not opts.skip_aggregation:
        summary_path = opts.output_dir / experiment_name / "summary"
        summary_path.mkdir(parents=True, exist_ok=True)
        aggregate_cmd = [
            sys.executable,
            str(GENERATE_RESULTS),
            "--results-dir",
            str(opts.output_dir / experiment_name),
            "--output",
            str(summary_path / "experiment_summary.json"),
        ]
        run_command(aggregate_cmd)

        if GENERATE_RESULTS.exists():
            run_command([sys.executable, str(GENERATE_RESULTS)])
        else:
            print("[WARN] generate_results.py not found for global summary; skipping")

        if CREATE_PLOTS.exists():
            run_command([sys.executable, str(CREATE_PLOTS)])
        else:
            print("[WARN] create_plots.py not found; skipping plot refresh")

    print(f"Experiment suite complete. Artefacts stored under {opts.output_dir / experiment_name}")


if __name__ == "__main__":
    main()
