"""Generate summary tables for Deep CFR experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            records = json.loads(path.read_text())
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)

    # Ensure numeric columns are properly typed
    df = df.copy()
    if "final_exploitability" in df.columns:
        df["final_exploitability"] = pd.to_numeric(df["final_exploitability"], errors="coerce")
    return df


def compute_architecture_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("architecture")["final_exploitability"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(
        columns={
            "count": "num_runs",
            "mean": "mean_final_exploitability",
            "std": "std_final_exploitability",
            "min": "min_final_exploitability",
            "max": "max_final_exploitability",
        }
    )
    return summary.sort_values("mean_final_exploitability")


def compute_experiment_architecture_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["experiment", "architecture"])["final_exploitability"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(
        columns={
            "count": "num_runs",
            "mean": "mean_final_exploitability",
            "std": "std_final_exploitability",
            "min": "min_final_exploitability",
            "max": "max_final_exploitability",
        }
    )
    return summary.sort_values(["experiment", "mean_final_exploitability"])


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Add delta vs baseline for each experiment where baseline exists."""
    baseline_means = (
        summary[summary["architecture"] == "baseline"]
        .set_index("experiment")["mean_final_exploitability"]
        .to_dict()
    )

    def compute_delta(row):
        baseline_mean = baseline_means.get(row["experiment"])
        if baseline_mean is None:
            return np.nan
        return row["mean_final_exploitability"] - baseline_mean

    summary = summary.copy()
    summary["delta_vs_baseline"] = summary.apply(compute_delta, axis=1)
    return summary


def export_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # Also export Markdown for quick reference
    markdown_path = path.with_suffix(".md")
    try:
        markdown_path.write_text(df.to_markdown(index=False))
    except ImportError:
        markdown_path.write_text("Optional dependency 'tabulate' not installed. Install with `pip install tabulate` to enable markdown export.\n\n")
        markdown_path.write_text(df.to_string(index=False))


def export_summary_json(architecture_summary: pd.DataFrame, experiment_summary: pd.DataFrame, path: Path) -> None:
    summary_payload: Dict[str, List[Dict[str, object]]] = {
        "architecture_summary": architecture_summary.to_dict(orient="records"),
        "experiment_architecture_summary": experiment_summary.to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate tables for Deep CFR diagnostics.")
    parser.add_argument("--input", type=Path, required=True, help="Path to comprehensive_results.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output/tables"),
        help="Directory to emit summary tables",
    )
    args = parser.parse_args()

    df = load_results(args.input)

    architecture_summary = compute_architecture_summary(df)
    experiment_summary = compute_experiment_architecture_summary(df)
    experiment_summary = add_baseline_deltas(experiment_summary)

    export_table(architecture_summary, args.output_dir / "architecture_summary.csv")
    export_table(experiment_summary, args.output_dir / "experiment_architecture_summary.csv")
    export_summary_json(
        architecture_summary,
        experiment_summary,
        args.output_dir / "summary_tables.json",
    )

    print(
        "Saved architecture and experiment-level summaries to",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
