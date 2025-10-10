"""Aggregate exploitability results for the Leduc ablation study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

RunResults = List[List[Tuple[int, float]]]
AblationResults = Dict[str, RunResults]


def _load_results(path: Path) -> AblationResults:
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Ablation results must be a dictionary keyed by agent name.")
    return data  # type: ignore[return-value]


def _runs_to_dataframe(agent: str, runs: RunResults) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for run_idx, run in enumerate(runs, start=1):
        if not isinstance(run, Iterable):
            raise ValueError(f"Run data for agent '{agent}' must be iterable.")
        for iteration, exploitability in run:
            records.append(
                {
                    "agent": agent,
                    "run": run_idx,
                    "iteration": int(iteration),
                    "exploitability": float(exploitability),
                }
            )
    if not records:
        return pd.DataFrame(columns=["agent", "run", "iteration", "exploitability"])
    return pd.DataFrame.from_records(records)


def _aggregate_agent(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["agent", "iteration"])  # type: ignore[arg-type]
    stats = grouped["exploitability"].agg(["mean", "std", "count"]).reset_index()
    stats.rename(columns={"count": "num_runs"}, inplace=True)
    stats.loc[:, "std"] = stats["std"].fillna(0.0)
    stats["ci95"] = stats.apply(
        lambda row: 1.96 * row["std"] / math.sqrt(row["num_runs"]) if row["num_runs"] > 0 else np.nan,
        axis=1,
    )
    stats["ci95_lower"] = stats["mean"] - stats["ci95"]
    stats["ci95_upper"] = stats["mean"] + stats["ci95"]
    return stats


def aggregate_results(results: AblationResults) -> pd.DataFrame:
    frames = [_runs_to_dataframe(agent, runs) for agent, runs in results.items()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise ValueError("No exploitability data found to aggregate.")
    return _aggregate_agent(combined)


def _write_outputs(summary: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_summary.csv"
    summary.to_csv(csv_path, index=False)

    formatted = summary.assign(
        formatted=lambda df: df.apply(
            lambda row: f"{row['mean']:.4f} ± {row['ci95']:.4f}", axis=1
        )
    )
    table = formatted.pivot(index="iteration", columns="agent", values="formatted").sort_index()

    latex_path = output_dir / "ablation_summary.tex"
    with latex_path.open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(na_rep="--", escape=False))

    return csv_path, latex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exploitability results from the Leduc ablation study.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="ablation_study_results.json",
        help="Path to the JSON file produced by the ablation runner.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs",
        help="Directory to store the aggregated CSV and LaTeX outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    results = _load_results(results_path)
    summary = aggregate_results(results)
    csv_path, latex_path = _write_outputs(summary, output_dir)

    print("Aggregated summary saved to:")
    print(f"  CSV : {csv_path}")
    print(f"  LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
