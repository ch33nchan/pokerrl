"""
OpenSpiel reference baselines and standardized evaluation helpers.

This module provides:
- tabular CFR baseline curves (Kuhn and Leduc) via OpenSpiel
- Deep CFR and SD-CFR reference runs where available in OpenSpiel
- standardized exploitability/NashConv reporting and CSV export

Usage examples:

  # Tabular CFR on Leduc with exploitability curve every 1k iters
  python -m qagent.evaluation.openspiel_baselines --game leduc_poker --algo cfr --iterations 20000 --eval-interval 1000 --out cfr_leduc_curve.csv

  # Deep CFR and SD-CFR reference (if available)
  python -m qagent.evaluation.openspiel_baselines --game leduc_poker --algo deep_cfr --iterations 20000 --eval-interval 1000 --out deepcfr_leduc_curve.csv

Notes:
- Requires `open_spiel` to be installed (pip install open_spiel==1.4 or compatible).
- If OpenSpiel is unavailable, the script exits with a clear message.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
from typing import List, Tuple

try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

pyspiel = None
os_exploit = None
os_cfr = None
os_deep_cfr = None
os_sd_cfr = None
OPEN_SPIEL_OK = False
_IMPORT_ERR: Exception | None = None

def _load_openspiel_once() -> None:
    global pyspiel, os_exploit, os_cfr, os_deep_cfr, os_sd_cfr, OPEN_SPIEL_OK, _IMPORT_ERR
    if OPEN_SPIEL_OK or _IMPORT_ERR is not None:
        return
    try:
        # Suppress noisy prints during third-party imports
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import pyspiel as _pyspiel  # type: ignore
            from open_spiel.python.algorithms import exploitability as _os_exploit
            from open_spiel.python.algorithms import cfr as _os_cfr
            try:
                from open_spiel.python.algorithms import deep_cfr as _os_deep_cfr  # type: ignore
            except Exception:
                _os_deep_cfr = None
            try:
                from open_spiel.python.algorithms import single_deep_cfr as _os_sd_cfr  # type: ignore
            except Exception:
                _os_sd_cfr = None
        pyspiel = _pyspiel
        os_exploit = _os_exploit
        os_cfr = _os_cfr
        os_deep_cfr = _os_deep_cfr
        os_sd_cfr = _os_sd_cfr
        OPEN_SPIEL_OK = True
    except Exception as e:  # pragma: no cover
        _IMPORT_ERR = e

# Eagerly attempt to load once at module import, but quietly.
_load_openspiel_once()


@dataclass
class CurvePoint:
    iteration: int
    nash_conv: float


def _ensure_openspiel() -> None:
    if not OPEN_SPIEL_OK:
        print("OpenSpiel not available: install with `pip install open_spiel`.")
        print(f"Import error: {_IMPORT_ERR}")
        sys.exit(2)


def run_tabular_cfr(game_name: str, iterations: int, eval_interval: int) -> List[CurvePoint]:
    _load_openspiel_once()
    game = pyspiel.load_game(game_name)
    solver = os_cfr.CFRSolver(game)
    curve: List[CurvePoint] = []
    iterator = range(1, iterations + 1)
    pbar = tqdm(iterator, total=iterations, desc=f"CFR({game_name})", leave=True) if tqdm else iterator
    try:
        for it in pbar:
            solver.evaluate_and_update_policy()
            if it % eval_interval == 0 or it == iterations:
                pol = solver.average_policy()
                nc = os_exploit.nash_conv(game, pol)
                curve.append(CurvePoint(it, float(nc)))
                if tqdm:
                    pbar.set_postfix({"nash_conv": f"{float(nc):.4g}"})
    except KeyboardInterrupt:
        print("Interrupted: returning partial CFR curve.")
    return curve


def run_deep_cfr(game_name: str, iterations: int, eval_interval: int) -> List[CurvePoint]:
    _load_openspiel_once()
    if os_deep_cfr is None:
        print("Deep CFR not available in this OpenSpiel build.")
        return []
    game = pyspiel.load_game(game_name)
    solver = os_deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(64, 64),
        advantage_network_layers=(64, 64),
        num_iterations=iterations,
        num_traversals=400,
        learning_rate=1e-3,
        batch_size_advantage=512,
        batch_size_strategy=512,
        memory_capacity=2_000_000,
        reinitialize_advantage_networks=False,
    )
    curve: List[CurvePoint] = []
    iterator = range(1, iterations + 1)
    pbar = tqdm(iterator, total=iterations, desc=f"DeepCFR({game_name})", leave=True) if tqdm else iterator
    try:
        for it in pbar:
            solver.iteration(it)
            if it % eval_interval == 0 or it == iterations:
                pol = solver.average_policy()
                nc = os_exploit.nash_conv(game, pol)
                curve.append(CurvePoint(it, float(nc)))
                if tqdm:
                    pbar.set_postfix({"nash_conv": f"{float(nc):.4g}"})
    except KeyboardInterrupt:
        print("Interrupted: returning partial Deep CFR curve.")
    return curve


def run_sd_cfr(game_name: str, iterations: int, eval_interval: int) -> List[CurvePoint]:
    _load_openspiel_once()
    if os_sd_cfr is None:
        print("SD-CFR not available in this OpenSpiel build.")
        return []
    game = pyspiel.load_game(game_name)
    solver = os_sd_cfr.SdCfrSolver(
        game,
        policy_network_layers=(64, 64),
        advantage_network_layers=(64, 64),
        num_iterations=iterations,
        num_traversals=400,
        learning_rate=1e-3,
        batch_size_advantage=512,
        batch_size_strategy=512,
        memory_capacity=2_000_000,
    )
    curve: List[CurvePoint] = []
    iterator = range(1, iterations + 1)
    pbar = tqdm(iterator, total=iterations, desc=f"SD-CFR({game_name})", leave=True) if tqdm else iterator
    try:
        for it in pbar:
            solver.iteration(it)
            if it % eval_interval == 0 or it == iterations:
                pol = solver.average_policy()
                nc = os_exploit.nash_conv(game, pol)
                curve.append(CurvePoint(it, float(nc)))
                if tqdm:
                    pbar.set_postfix({"nash_conv": f"{float(nc):.4g}"})
    except KeyboardInterrupt:
        print("Interrupted: returning partial SD-CFR curve.")
    return curve


def write_curve(curve: List[CurvePoint], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "nash_conv"])  # OpenSpiel standard metric
        for p in curve:
            w.writerow([p.iteration, p.nash_conv])
    print(f"Saved curve: {out}")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="OpenSpiel reference baselines")
    parser.add_argument("--game", default="leduc_poker", choices=["leduc_poker", "kuhn_poker"], help="OpenSpiel game name")
    parser.add_argument("--algo", default="cfr", choices=["cfr", "deep_cfr", "sd_cfr"], help="Baseline algorithm")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("analysis_output/openspiel_baseline.csv"))
    args = parser.parse_args()

    _ensure_openspiel()

    if args.algo == "cfr":
        curve = run_tabular_cfr(args.game, args.iterations, args.eval_interval)
    elif args.algo == "deep_cfr":
        curve = run_deep_cfr(args.game, args.iterations, args.eval_interval)
    else:
        curve = run_sd_cfr(args.game, args.iterations, args.eval_interval)
    # If requested algo is unavailable, do not emit an empty CSV; exit clearly.
    if args.algo in ("deep_cfr", "sd_cfr") and args.iterations > 0 and len(curve) == 0:
        print(f"{args.algo.replace('_', ' ').title()} not available in this OpenSpiel build. No curve written.")
        sys.exit(2)

    write_curve(curve, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
