#!/usr/bin/env python3
"""Aggregate experiment artifacts produced by ``run_real_training.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.results_manager import (
    DEFAULT_SUMMARY_NAME,
    refresh_indices,
)

DEFAULT_RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate JSON runs produced by run_real_training.py.",
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
    summary_path: Path = (
        opts.output
        if opts.output is not None
        else (results_dir / DEFAULT_SUMMARY_NAME)
    )

    result = refresh_indices(results_dir, summary_output=summary_path)
    if not result:
        print(
            f"No real experiment runs found under '{results_dir}'. Nothing to summarise."
        )
        return

    print(f"Wrote summary for {result['payload']['num_runs']} runs to {summary_path}")
    combined_dir = result.get("combined_dir")
    if combined_dir is not None:
        print(f"Combined artefacts available under {combined_dir}")
    algorithms_root = result.get("algorithms_root")
    if algorithms_root is not None:
        print(f"Per-algorithm digests refreshed under {algorithms_root}")


if __name__ == "__main__":
    main()
