#!/usr/bin/env python3
"""Backfill per-iteration CSV files from JSON training logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def iter_json(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.suffix.lower() == ".json" and path.name.endswith(".json"):
            yield path


def backfill(json_path: Path, overwrite: bool) -> Path | None:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    history = payload.get("training_history")
    if not history:
        return None
    csv_path = json_path.with_name(json_path.stem + "_history.csv")
    if csv_path.exists() and not overwrite:
        return None
    fieldnames = sorted({key for row in history for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate *_history.csv files next to JSON runs if missing."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("results"),
        help="Directory tree to scan for JSON artefacts (default: results/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of leaving them untouched.",
    )
    return parser.parse_args()


def main() -> None:
    opts = parse_args()
    json_files = list(opts.results_dir.rglob("*_seed*.json"))
    if not json_files:
        print(f"No JSON runs found under {opts.results_dir}")
        return
    generated = 0
    skipped = 0
    for path in json_files:
        csv_path = backfill(path, opts.overwrite)
        if csv_path is None:
            skipped += 1
        else:
            generated += 1
            print(f"Wrote {csv_path.relative_to(opts.results_dir)}")
    print(
        f"Backfill complete: {generated} CSV files created, {skipped} skipped"
    )


if __name__ == "__main__":
    main()
