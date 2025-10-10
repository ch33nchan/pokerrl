"""Aggregate experimental outputs into a unified diagnostics CSV.

Now also parses per-run TrainingLogger CSVs written during training to attach
diagnostics (gradient norms, losses, KL, param counts, wall-clock) to each run.
Logs are expected at:
  logs/leduc_ablation/run_{i}/{agent_key}/*metrics.csv

If logs are missing, aggregation falls back gracefully.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import pandas as pd
from pathlib import Path

LOG_ROOT = Path("logs/leduc_ablation")


@dataclass
class RunRecord:
    experiment: str
    agent_key: str
    run_index: int
    iterations: List[int]
    exploitabilities: List[float]
    gradient_norms: Optional[Dict[str, List[float]]] = None
    loss_trajectory: Optional[Dict[str, List[float]]] = None
    gradient_explosion_events: Optional[Dict[str, Iterable[int]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_row(self) -> Dict[str, Any]:
        final_exploitability = self.exploitabilities[-1] if self.exploitabilities else None
        payload: Dict[str, Any] = {
            "experiment": self.experiment,
            "architecture": self.metadata.get("architecture", self.agent_key) if self.metadata else self.agent_key,
            "agent_key": self.agent_key,
            "run_index": self.run_index,
            "game": self.metadata.get("game", "leduc") if self.metadata else "leduc",
            "success": True if final_exploitability is not None else False,
            "final_exploitability": final_exploitability,
            "exploitability_trajectory": json.dumps(list(zip(self.iterations, self.exploitabilities))),
            "gradient_norms": json.dumps(self.gradient_norms) if self.gradient_norms else None,
            "loss_trajectory": json.dumps(self.loss_trajectory) if self.loss_trajectory else None,
            "gradient_explosion_events": json.dumps(self.gradient_explosion_events) if self.gradient_explosion_events else None,
            "hyperparams": json.dumps(self.metadata.get("hyperparams", {})) if self.metadata and self.metadata.get("hyperparams") else None,
            "training_time": self.metadata.get("training_time") if self.metadata else None,
            "error_message": self.metadata.get("error_message") if self.metadata else None,
        }
        return payload


def discover_ablation_results(search_root: Path) -> List[Path]:
    patterns = ["ablation_study_results.json", "**/ablation_study_results.json"]
    discovered: List[Path] = []
    for pattern in patterns:
        discovered.extend(search_root.glob(pattern))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_results: List[Path] = []
    for path in discovered:
        if path not in seen:
            unique_results.append(path)
            seen.add(path)
    return unique_results


def parse_ablation_file(json_path: Path, project_root: Path) -> List[RunRecord]:
    try:
        payload = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return []

    experiment_name = derive_experiment_name(json_path, project_root)
    records: List[RunRecord] = []

    for agent_key, runs in payload.items():
        if not isinstance(runs, list):
            continue
        for idx, trajectory in enumerate(runs, start=1):
            if not trajectory:
                continue
            iterations = [point[0] for point in trajectory]
            exploitabilities = [point[1] for point in trajectory]
            record = RunRecord(
                experiment=experiment_name,
                agent_key=agent_key,
                run_index=idx,
                iterations=iterations,
                exploitabilities=exploitabilities,
                metadata={"architecture": agent_key, "game": "leduc"},
            )
            records.append(record)
    return records


def derive_experiment_name(json_path: Path, project_root: Path) -> str:
    try:
        relative = json_path.relative_to(project_root)
    except ValueError:
        return json_path.stem

    parts = relative.parts
    if len(parts) == 1:
        return parts[0].replace(".json", "")
    # Ignore generic filenames and use parent directory labels when available
    parent = json_path.parent
    if parent == project_root:
        return json_path.stem
    return parent.name


def aggregate_runs(logs_dir: Path, project_root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # Gather trajectories from ablation JSON files
    for json_file in discover_ablation_results(project_root):
        for record in parse_ablation_file(json_file, project_root):
            # Attempt to attach diagnostics from CSV logs
            diag = _load_run_diagnostics(
                LOG_ROOT / f"run_{record.run_index}" / record.agent_key
            )
            row = record.to_row()
            if diag:
                row.update(diag)
            records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _load_run_diagnostics(run_dir: Path) -> Dict[str, Any]:
    """Load metrics CSV in a run directory and compress sequences as JSON strings.

    Returns a dict with keys: gradient_norms, loss_trajectory, kl_history,
    param_counts, iteration_wall_clock_sec (mean), cumulative_wall_clock_sec (final).
    Missing data are omitted.
    """
    if not run_dir.exists():
        return {}
    # find the single CSV file produced by TrainingLogger
    csv_files = list(run_dir.glob("*.csv"))
    if not csv_files:
        return {}
    # pick largest CSV (most rows)
    csv_path = max(csv_files, key=lambda p: p.stat().st_size)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    out: Dict[str, Any] = {}

    # Gradient norms
    grad = {}
    if "regret_grad_norm" in df.columns:
        grad["regret"] = df["regret_grad_norm"].dropna().tolist()
    if "strategy_grad_norm" in df.columns:
        grad["strategy"] = df["strategy_grad_norm"].dropna().tolist()
    if grad:
        out["gradient_norms"] = json.dumps(grad)

    # Loss trajectories
    losses = {}
    if "regret_loss" in df.columns:
        losses["regret"] = df["regret_loss"].dropna().tolist()
    if "strategy_loss" in df.columns:
        losses["strategy"] = df["strategy_loss"].dropna().tolist()
    if losses:
        out["loss_trajectory"] = json.dumps(losses)

    # KL history (if produced)
    if "strategy_kl" in df.columns:
        out["kl_history"] = json.dumps(df["strategy_kl"].dropna().tolist())

    # Parameter counts
    param_counts = {}
    for k in ("regret_param_count", "strategy_param_count"):
        if k in df.columns and pd.notna(df[k]).any():
            param_counts[k] = float(df[k].dropna().iloc[-1])
    if param_counts:
        out["param_counts"] = json.dumps(param_counts)

    # Timing summaries
    if "iteration_wall_clock_sec" in df.columns:
        out["iteration_wall_clock_sec_mean"] = (
            float(df["iteration_wall_clock_sec"].dropna().mean())
        )
    if "cumulative_wall_clock_sec" in df.columns:
        out["cumulative_wall_clock_sec_final"] = (
            float(df["cumulative_wall_clock_sec"].dropna().iloc[-1])
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Deep CFR experiment diagnostics.")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Directory containing per-run logs")
    parser.add_argument("--results-file", type=Path, required=True, help="Path to the aggregated CSV to write")
    args = parser.parse_args()

    project_root = Path.cwd()
    results_df = aggregate_runs(args.logs_dir, project_root)

    if results_df.empty:
        print("Warning: No experiment data discovered. Ensure ablation JSON files are present.")

    results_file = args.results_file
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"Aggregated {len(results_df)} run records to {results_file}")


if __name__ == "__main__":
    main()
