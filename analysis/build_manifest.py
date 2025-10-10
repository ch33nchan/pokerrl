#!/usr/bin/env python3
"""Construct a single source-of-truth manifest for Deep CFR experiments.

The manifest consolidates metadata from:
- `analysis_output/comprehensive_results.csv` when available
- `ablation_study_results.json` (and nested variants)
- training diagnostics under `logs/`

It emits rows with the fields requested by the reporting pipeline so that
subsequent tables, figures, and manuscript references can draw from a single
canonical source.

Example usage:
    python -m analysis.build_manifest \
        --output run_manifest.json \
        --exploitability-threshold 2.0

The script is CPU-only and reads existing artifacts; it does not launch
experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_OUTPUT = PROJECT_ROOT / "analysis_output"
DEFAULT_RESULTS_CSV = ANALYSIS_OUTPUT / "comprehensive_results.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "analysis_output" / "run_manifest.json"
ABALATION_JSON_NAME = "ablation_study_results.json"

# Ordered list of manifest fields, matching the user specification.
MANIFEST_FIELDS = [
    "run_id",
    "experiment",
    "architecture",
    "seed",
    "traversal_scheme",
    "params_count",
    "optimizer",
    "lr",
    "batch_size",
    "replay_ratio",
    "steps",
    "wall_clock_s",
    "success_flag",
    "final_exploitability",
    "notes",
    "commit_hash",
]


@dataclass
class ManifestEntry:
    run_id: str
    experiment: str
    architecture: str
    seed: Optional[int]
    traversal_scheme: Optional[str]
    params_count: Optional[int]
    optimizer: Optional[str]
    lr: Optional[float]
    batch_size: Optional[int]
    replay_ratio: Optional[float]
    steps: Optional[int]
    wall_clock_s: Optional[float]
    success_flag: bool
    final_exploitability: Optional[float]
    notes: Optional[str]
    commit_hash: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Keep column order predictable
        return {field: payload.get(field) for field in MANIFEST_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified run manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="Where to write the manifest (JSON or CSV)")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument(
        "--exploitability-threshold",
        type=float,
        default=2.0,
        help="Success criterion: final exploitability must be <= threshold",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Optional aggregated CSV for additional metadata",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root directory for discovering ablation JSON files",
    )
    args = parser.parse_args()

    commit_hash = _read_git_commit()
    manifest_entries = list(
        build_manifest(
            search_root=args.search_root,
            results_csv=args.results_csv,
            exploitability_threshold=args.exploitability_threshold,
            commit_hash=commit_hash,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        args.output.write_text(json.dumps([entry.to_dict() for entry in manifest_entries], indent=2))
    else:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for entry in manifest_entries:
                writer.writerow(entry.to_dict())

    print(f"Wrote {len(manifest_entries)} manifest rows to {args.output}")


def build_manifest(
    search_root: Path,
    results_csv: Path,
    exploitability_threshold: float,
    commit_hash: Optional[str],
) -> Iterator[ManifestEntry]:
    """Yield manifest entries derived from available artifacts."""

    supplemental_rows: Dict[str, Dict[str, Any]] = {}
    if results_csv.exists():
        supplemental_rows = _index_results_csv(results_csv)

    for json_path in _discover_ablation_json(search_root):
        experiment = _experiment_name(json_path, search_root)
        payload = _load_json(json_path)
        if not isinstance(payload, dict):
            continue

        for agent_key, runs in payload.items():
            if not isinstance(runs, list):
                continue
            for run_idx, trajectory in enumerate(runs, start=1):
                if not trajectory:
                    continue
                iterations, exploitabilities = zip(*trajectory)
                final_exploitability = float(exploitabilities[-1])
                success_flag = final_exploitability <= exploitability_threshold
                run_id = f"{experiment}-{agent_key}-run{run_idx}"
                supplemental = supplemental_rows.get(run_id, {})

                entry = ManifestEntry(
                    run_id=run_id,
                    experiment=experiment,
                    architecture=str(supplemental.get("architecture", agent_key)),
                    seed=_maybe_int(supplemental.get("seed")),
                    traversal_scheme=supplemental.get("traversal_scheme"),
                    params_count=_extract_params_count(supplemental),
                    optimizer=supplemental.get("optimizer"),
                    lr=_maybe_float(supplemental.get("lr")),
                    batch_size=_maybe_int(supplemental.get("batch_size")),
                    replay_ratio=_maybe_float(supplemental.get("replay_ratio")),
                    steps=int(iterations[-1]) if iterations else None,
                    wall_clock_s=_maybe_float(supplemental.get("cumulative_wall_clock_sec_final")),
                    success_flag=success_flag,
                    final_exploitability=final_exploitability,
                    notes=supplemental.get("notes"),
                    commit_hash=commit_hash,
                )
                yield entry


def _discover_ablation_json(search_root: Path) -> Iterable[Path]:
    for path in search_root.rglob(ABALATION_JSON_NAME):
        yield path


def _experiment_name(json_path: Path, project_root: Path) -> str:
    try:
        rel = json_path.relative_to(project_root)
    except ValueError:
        return json_path.stem
    if len(rel.parts) == 1:
        return json_path.stem
    parent = rel.parent
    if parent == Path(""):
        return json_path.stem
    return parent.name


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _index_results_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    index: Dict[str, Dict[str, Any]] = {}
    if df.empty:
        return index

    for _, row in df.iterrows():
        experiment = row.get("experiment") or "unknown"
        agent_key = row.get("agent_key") or row.get("architecture") or "unknown"
        run_index = row.get("run_index") or row.get("run")
        if pd.isna(run_index):
            continue
        run_id = f"{experiment}-{agent_key}-run{int(run_index)}"
        index[run_id] = row.to_dict()
    return index


def _extract_params_count(row: Dict[str, Any]) -> Optional[int]:
    param_counts = row.get("param_counts")
    if isinstance(param_counts, str):
        try:
            parsed = json.loads(param_counts)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            # Sum regret + strategy counts when both available
            values = [value for value in parsed.values() if isinstance(value, (int, float))]
            if values:
                return int(sum(values))
    return None


def _maybe_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    main()
