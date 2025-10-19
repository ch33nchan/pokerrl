from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_SUMMARY_NAME = "experiment_summary.json"


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


def discover_runs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("*_seed*.json"))


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
        "backend": data.get("backend", "unknown"),
        "device": data.get("device", "unknown"),
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
        gate_entropies = [
            float(run["final_gate_entropy"])
            for run in game_runs
            if run.get("final_gate_entropy") is not None
        ]
        mean_exp, std_exp, ci_exp = _mean_std_ci(exploitabilities)
        mean_nash, std_nash, ci_nash = _mean_std_ci(nash_convs)
        mean_auc, std_auc, ci_auc = _mean_std_ci(aucs)
        mean_gate, std_gate, ci_gate = (
            _mean_std_ci(gate_entropies) if gate_entropies else (0.0, 0.0, 0.0)
        )

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
                "scheduler_loss_curve": [
                    float(step.get("scheduler_loss", 0.0))
                    for step in run["history"]
                ],
                "exploit_auc": run["exploit_auc"],
                "nash_auc": run["nash_auc"],
                "final_gate_entropy": run.get("final_gate_entropy"),
                "final_gate_probs": run.get("final_gate_probs", {}),
                "policy_type": run.get("policy_type", "unknown"),
                "backend": run.get("backend", "unknown"),
                "device": run.get("device", "unknown"),
            }
            for run in runs
        ],
    }
    return payload


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_tex_table(path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        path.write_text("% No data available\n", encoding="utf-8")
        return
    align = "l" + "r" * (len(columns) - 1)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated summary table\n")
        fh.write(f"\\begin{{tabular}}{{{align}}}\n")
        fh.write("\\hline\\hline\n")
        header = " & ".join(_tex_escape(col) for col in columns)
        fh.write(f"{header}\\\\\n")
        fh.write("\\hline\n")
        for row in rows:
            formatted = []
            for value in row:
                if isinstance(value, float):
                    formatted.append(f"{value:.6g}")
                else:
                    formatted.append(_tex_escape(str(value)))
            fh.write(" & ".join(formatted) + "\\\\\n")
        fh.write("\\hline\\hline\n")
        fh.write("\\end{tabular}\n")


def refresh_indices(
    results_dir: Path,
    *,
    summary_output: Optional[Path] = None,
    ensure_combined: bool = True,
    ensure_by_algorithm: bool = True,
) -> Dict[str, object]:
    paths = discover_runs(results_dir)
    if not paths:
        return {}

    runs = []
    for path in paths:
        try:
            runs.append(load_run(path))
        except ValueError:
            continue
    if not runs:
        return {}

    payload = build_payload(runs)

    summary_path = summary_output or results_dir / DEFAULT_SUMMARY_NAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    combined_dir = None
    if ensure_combined:
        combined_dir = results_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        combined_json = combined_dir / "runs_summary.json"
        with combined_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        per_run_rows: List[Dict[str, object]] = []
        history_rows: List[Dict[str, object]] = []
        for run in payload["runs"]:
            record = {
                "run_id": Path(run["path"]).stem,
                "game": run["game"],
                "policy": run.get("policy_type", "unknown"),
                "backend": run.get("backend", "unknown"),
                "device": run.get("device", "unknown"),
                "seed": run["seed"],
                "iterations": run["iterations"],
                "final_exploitability": run["final_exploitability"],
                "final_nash_conv": run["final_nash_conv"],
                "exploit_auc": run["exploit_auc"],
                "nash_auc": run["nash_auc"],
            }
            per_run_rows.append(record)
            scheduler_curve = run.get("scheduler_loss_curve", [])
            for iteration, exploit, nash, sched in zip(
                run["iterations_curve"],
                run["exploitability_curve"],
                run["nash_conv_curve"],
                scheduler_curve,
            ):
                history_rows.append(
                    {
                        "run_id": record["run_id"],
                        "game": run["game"],
                        "policy": record["policy"],
                        "backend": run.get("backend", "unknown"),
                        "device": run.get("device", "unknown"),
                        "iteration": iteration,
                        "exploitability": exploit,
                        "nash_conv": nash,
                        "scheduler_loss": sched,
                    }
                )

        per_run_fieldnames = [
            "run_id",
            "game",
            "policy",
            "backend",
            "device",
            "seed",
            "iterations",
            "final_exploitability",
            "final_nash_conv",
            "exploit_auc",
            "nash_auc",
        ]
        _write_csv(combined_dir / "runs_summary.csv", per_run_fieldnames, per_run_rows)

        history_fieldnames = [
            "run_id",
            "game",
            "policy",
            "backend",
            "device",
            "iteration",
            "exploitability",
            "nash_conv",
            "scheduler_loss",
        ]
        _write_csv(combined_dir / "all_iterations.csv", history_fieldnames, history_rows)

        summary_rows = []
        summary_csv_rows = []
        for game, policies in payload["games"].items():
            for policy, stats in policies.items():
                row = (
                    game,
                    policy,
                    int(stats.get("num_runs", 0)),
                    float(stats.get("mean_exploitability", 0.0)),
                    float(stats.get("ci95_exploitability", 0.0)),
                    float(stats.get("mean_nash_conv", 0.0)),
                    float(stats.get("ci95_nash_conv", 0.0)),
                    float(stats.get("mean_exploit_auc", 0.0)),
                )
                summary_rows.append(row)
                summary_csv_rows.append(
                    {
                        "game": row[0],
                        "policy": row[1],
                        "num_runs": row[2],
                        "mean_exploitability": row[3],
                        "ci95_exploitability": row[4],
                        "mean_nash_conv": row[5],
                        "ci95_nash_conv": row[6],
                        "mean_exploit_auc": row[7],
                    }
                )

        summary_columns = [
            "Game",
            "Policy",
            "Runs",
            "Mean exploitability",
            "CI95 exploitability",
            "Mean NashConv",
            "CI95 NashConv",
            "Mean exploit AUC",
        ]
        _write_csv(
            combined_dir / "summary.csv",
            [
                "game",
                "policy",
                "num_runs",
                "mean_exploitability",
                "ci95_exploitability",
                "mean_nash_conv",
                "ci95_nash_conv",
                "mean_exploit_auc",
            ],
            summary_csv_rows,
        )
        _write_tex_table(combined_dir / "summary.tex", summary_columns, summary_rows)

    algorithms_root = None
    if ensure_by_algorithm:
        algorithms_root = results_dir / "by_algorithm"
        algorithms_root.mkdir(parents=True, exist_ok=True)
        summary_columns = [
            "Game",
            "Policy",
            "Runs",
            "Mean exploitability",
            "CI95 exploitability",
            "Mean NashConv",
            "CI95 NashConv",
            "Mean exploit AUC",
        ]
        per_run_fieldnames = [
            "run_id",
            "game",
            "policy",
            "backend",
            "device",
            "seed",
            "iterations",
            "final_exploitability",
            "final_nash_conv",
            "exploit_auc",
            "nash_auc",
        ]
        per_algorithm_rows: Dict[str, List[Dict[str, object]]] = {}
        for run in payload["runs"]:
            policy = run.get("policy_type", "unknown")
            per_algorithm_rows.setdefault(policy, []).append(
                {
                    "run_id": Path(run["path"]).stem,
                    "game": run["game"],
                    "policy": policy,
                    "backend": run.get("backend", "unknown"),
                    "device": run.get("device", "unknown"),
                    "seed": run["seed"],
                    "iterations": run["iterations"],
                    "final_exploitability": run["final_exploitability"],
                    "final_nash_conv": run["final_nash_conv"],
                    "exploit_auc": run["exploit_auc"],
                    "nash_auc": run["nash_auc"],
                }
            )

        for policy, rows in per_algorithm_rows.items():
            algo_dir = algorithms_root / str(policy)
            algo_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(algo_dir / "runs.csv", per_run_fieldnames, rows)
            with (algo_dir / "runs.json").open("w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)

            policy_summary_rows = []
            policy_summary_json: Dict[str, Dict[str, float]] = {}
            for game, policies in payload["games"].items():
                stats = policies.get(policy)
                if not stats:
                    continue
                policy_summary_json[game] = stats
                policy_summary_rows.append(
                    (
                        game,
                        policy,
                        int(stats.get("num_runs", 0)),
                        float(stats.get("mean_exploitability", 0.0)),
                        float(stats.get("ci95_exploitability", 0.0)),
                        float(stats.get("mean_nash_conv", 0.0)),
                        float(stats.get("ci95_nash_conv", 0.0)),
                        float(stats.get("mean_exploit_auc", 0.0)),
                    )
                )

            _write_csv(
                algo_dir / "summary.csv",
                [
                    "game",
                    "policy",
                    "num_runs",
                    "mean_exploitability",
                    "ci95_exploitability",
                    "mean_nash_conv",
                    "ci95_nash_conv",
                    "mean_exploit_auc",
                ],
                [
                    {
                        "game": row[0],
                        "policy": row[1],
                        "num_runs": row[2],
                        "mean_exploitability": row[3],
                        "ci95_exploitability": row[4],
                        "mean_nash_conv": row[5],
                        "ci95_nash_conv": row[6],
                        "mean_exploit_auc": row[7],
                    }
                    for row in policy_summary_rows
                ],
            )
            _write_tex_table(algo_dir / "summary.tex", summary_columns, policy_summary_rows)
            with (algo_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(policy_summary_json, fh, indent=2)

    return {
        "payload": payload,
        "summary_path": summary_path,
        "combined_dir": combined_dir,
        "algorithms_root": algorithms_root,
    }


__all__ = [
    "DEFAULT_SUMMARY_NAME",
    "discover_runs",
    "load_run",
    "aggregate",
    "build_payload",
    "refresh_indices",
]
