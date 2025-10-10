"""OpenSpiel evaluation utilities for exploitability and NashConv."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # optional dependency
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pyspiel  # type: ignore
        from open_spiel.python.algorithms import exploitability as os_exploit  # type: ignore
        from open_spiel.python.algorithms import best_response as os_best_response  # type: ignore
except Exception:  # pragma: no cover
    pyspiel = None
    os_exploit = None
    os_best_response = None


@dataclasses.dataclass
class EvaluationResult:
    seed: int
    game: str
    traversal_scheme: Optional[str]
    exploitability: float
    nash_conv: float
    wall_clock_s: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def ensure_openspiel() -> None:
    if pyspiel is None:
        raise RuntimeError(
            "OpenSpiel is not available. Install with `pip install open_spiel`."
        )


def load_tabular_policy(game_name: str, policy_path: Path) -> "pyspiel.TabularPolicy":  # type: ignore[name-defined]
    ensure_openspiel()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with policy_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    game = pyspiel.load_game(game_name)
    policy = pyspiel.TabularPolicy(game)
    for infoset, probs in data.items():
        index = policy.state_lookup[infoset]
        policy.policy_table[index] = probs
    return policy


def exploitability_from_policy(game_name: str, policy) -> float:
    ensure_openspiel()
    return float(os_exploit.exploitability(pyspiel.load_game(game_name), policy))


def nash_conv_from_policy(game_name: str, policy) -> float:
    ensure_openspiel()
    return float(os_exploit.nash_conv(pyspiel.load_game(game_name), policy))


def evaluate_policy(game_name: str, policy, *, seed: int, traversal_scheme: Optional[str] = None,
                    wall_clock_s: Optional[float] = None, notes: Optional[str] = None) -> EvaluationResult:
    exploit = exploitability_from_policy(game_name, policy)
    nash_conv = nash_conv_from_policy(game_name, policy)
    return EvaluationResult(
        seed=seed,
        game=game_name,
        traversal_scheme=traversal_scheme,
        exploitability=exploit,
        nash_conv=nash_conv,
        wall_clock_s=wall_clock_s,
        notes=notes,
    )


def aggregate_results(results: Iterable[EvaluationResult]) -> Dict[str, Any]:
    rows = list(results)
    exploits = np.array([row.exploitability for row in rows], dtype=float)
    nash_convs = np.array([row.nash_conv for row in rows], dtype=float)
    summary = {
        "count": len(rows),
        "exploitability_mean": float(np.mean(exploits)) if len(exploits) else None,
        "exploitability_ci95": _confidence_interval(exploits) if len(exploits) else None,
        "nash_conv_mean": float(np.mean(nash_convs)) if len(nash_convs) else None,
        "nash_conv_ci95": _confidence_interval(nash_convs) if len(nash_convs) else None,
    }
    return summary


def _confidence_interval(data: np.ndarray, alpha: float = 0.95) -> Optional[Dict[str, float]]:
    if data.size == 0:
        return None
    mean = float(np.mean(data))
    stderr = float(np.std(data, ddof=1) / np.sqrt(data.size)) if data.size > 1 else 0.0
    margin = 1.96 * stderr if data.size > 1 else 0.0
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin}


__all__ = [
    "EvaluationResult",
    "aggregate_results",
    "ensure_openspiel",
    "evaluate_policy",
    "exploitability_from_policy",
    "nash_conv_from_policy",
    "load_tabular_policy",
]
