#!/usr/bin/env python3
"""
All-in-one experiment runner for Dual RL Poker:
- Runs ARMAC adaptive-λ sweeps (different lambda_alpha values)
- Runs ARMAC fixed-λ sweeps (different λ values)
- Runs baseline algorithms (Deep CFR, SD-CFR) for comparison
- Generates plots and tables from actual experiment outputs

This script:
- Uses the project's run_experiments.py and create_plots.py
- Produces no placeholders or simulated results
- Summarizes final exploitability across seeds for each configuration
- Helps assess whether adaptive λ supports the hypothesis

Usage:
  python scripts/run_all_experiments.py \
    --seeds 0,1,2 \
    --iterations 120 \
    --games kuhn_poker,leduc_poker \
    --adaptive-alphas 0.5,1.0,2.0 \
    --fixed-lambdas 0.0,0.1,0.25,0.5

Flags:
  --skip-armac-adaptive     Skip ARMAC adaptive λ runs
  --skip-armac-fixed        Skip ARMAC fixed λ runs
  --skip-baselines          Skip Deep CFR / SD-CFR runs
  --plots-only              Only regenerate plots from existing results
  --dry-run                 Print commands without executing
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Any


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    print(f"\n[RUN] {' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def ensure_project_layout(project_root: Path):
    required = ["run_experiments.py", "create_plots.py"]
    missing = [p for p in required if not (project_root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in project root: {', '.join(missing)}"
        )


def collect_results(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*_results.json"))


def summarize_exploitability(results_paths: List[Path]) -> Dict[str, Any]:
    """
    Build summary by:
    - key: composite label: game/method/(adaptive alpha | fixed λ)/seed-aggregated group
    - metrics: mean, std, n
    """
    groups: Dict[str, List[float]] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}

    for fp in results_paths:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        game = data.get("game") or (data.get("metadata") or {}).get("game")
        method = data.get("method") or data.get("algorithm")
        if not game or not method:
            continue

        cfg = data.get("config", {})
        eval_hist = data.get("evaluation_history") or []
        if not eval_hist:
            # no evaluation recorded -> skip to avoid placeholders
            continue
        final_exp = eval_hist[-1].get("exploitability")
        if final_exp is None:
            continue

        # Build grouping key
        key_parts = [game, method]
        if method == "armac":
            lm = cfg.get("lambda_mode", "adaptive")
            key_parts.append(f"lambda_mode={lm}")
            if lm == "adaptive":
                la = cfg.get("lambda_alpha", None)
                if la is not None:
                    key_parts.append(f"lambda_alpha={la}")
            elif lm == "fixed":
                rw = cfg.get("regret_weight", None)
                if rw is not None:
                    key_parts.append(f"lambda={rw}")
            # Include actor mix CE weight in grouping if present (applies to both modes)
            ce = cfg.get("mix_ce_weight", None)
            if ce is not None:
                key_parts.append(f"mix_ce_weight={ce}")
        group_key = "/".join(key_parts)

        groups.setdefault(group_key, []).append(float(final_exp))
        if group_key not in meta_map:
            meta_map[group_key] = {"game": game, "method": method, "config": cfg}

    summary = {}
    for k, vals in groups.items():
        summary[k] = {
            "n": len(vals),
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "meta": meta_map.get(k, {}),
        }
    return summary


def compare_adaptive_vs_fixed(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each game, compare the best adaptive (min mean exploitability)
    vs the best fixed across sweeps. Lower is better.
    """
    per_game = {}
    # Organize by game
    for key, stats in summary.items():
        meta = stats["meta"]
        game = meta.get("game")
        method = meta.get("method")
        if method != "armac":
            continue
        if game not in per_game:
            per_game[game] = {"adaptive": [], "fixed": []}

        cfg = meta.get("config", {})
        lm = cfg.get("lambda_mode", "adaptive")
        record = {
            "key": key,
            "mean": stats["mean"],
            "std": stats["std"],
            "n": stats["n"],
            "lambda_alpha": cfg.get("lambda_alpha"),
            "regret_weight": cfg.get("regret_weight"),
            "mix_ce_weight": cfg.get("mix_ce_weight"),
        }
        if lm == "adaptive":
            per_game[game]["adaptive"].append(record)
        else:
            per_game[game]["fixed"].append(record)

    # Find best entries per mode
    results = {}
    for game, buckets in per_game.items():
        best_ad = (
            min(buckets["adaptive"], key=lambda r: r["mean"])
            if buckets["adaptive"]
            else None
        )
        best_fx = (
            min(buckets["fixed"], key=lambda r: r["mean"]) if buckets["fixed"] else None
        )
        results[game] = {"best_adaptive": best_ad, "best_fixed": best_fx}

    return results


def print_hypothesis_assessment(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced hypothesis assessment:
    - Loads raw final exploitabilities for the selected best adaptive vs best fixed configs
    - Computes Welch t-statistic (t, df) for mean difference (adaptive - fixed)
    - Computes bootstrap CI for mean difference and P(adaptive < fixed)
    - Prints a verdict and saves a JSON report with raw values and statistics
    """
    import random

    print("\n=== Hypothesis Assessment: Does adaptive λ outperform fixed λ? ===")

    # Locate results directory to read raw runs
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    result_files = sorted(results_dir.glob("*_results.json"))

    def collect_values(
        game: str,
        mode: str,
        lambda_alpha: Any = None,
        regret_weight: Any = None,
        mix_ce_weight: Any = None,
    ) -> List[float]:
        vals: List[float] = []
        for fp in result_files:
            try:
                data = json.loads(fp.read_text())
            except Exception:
                continue

            method = data.get("method") or data.get("algorithm")
            game_name = data.get("game") or (data.get("metadata") or {}).get("game")
            if method != "armac" or game_name != game:
                continue

            cfg = data.get("config", {}) or {}
            if cfg.get("lambda_mode", "adaptive") != mode:
                continue

            if mode == "adaptive":
                la = cfg.get("lambda_alpha", None)
                # If a specific alpha is chosen, filter by exact match; otherwise include all adaptive
                if lambda_alpha is not None and la != lambda_alpha:
                    continue
            else:  # fixed
                rw = cfg.get("regret_weight", None)
                if regret_weight is not None and rw != regret_weight:
                    continue

            # Optional filter by mix_ce_weight (actor CE-to-mixture term)
            ce = cfg.get("mix_ce_weight", None)
            if mix_ce_weight is not None and ce != mix_ce_weight:
                continue

            eval_hist = data.get("evaluation_history") or []
            if not eval_hist:
                continue
            final_exp = eval_hist[-1].get("exploitability")
            if final_exp is None:
                continue
            vals.append(float(final_exp))
        return vals

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        n = len(vals)
        if not n:
            return 0.0, 0.0
        mu = sum(vals) / n
        if n < 2:
            return mu, 0.0
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        return mu, var**0.5

    def welch_t_df(a: List[float], b: List[float]) -> Tuple[float, float]:
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0
        m1, m2 = sum(a) / n1, sum(b) / n2
        v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
        v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
        denom = (v1 / n1) + (v2 / n2)
        if denom <= 0:
            return 0.0, 1.0
        t_stat = (m1 - m2) / (denom**0.5)
        df_num = denom**2
        df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
        df = df_num / df_den if df_den > 0 else 1.0
        return t_stat, df

    def bootstrap_diff_ci(
        a: List[float], b: List[float], n_bootstrap: int = 10000, alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            return 0.0, 0.0, 0.0
        diffs: List[float] = []
        for _ in range(n_bootstrap):
            s1 = [a[random.randrange(n1)] for _ in range(n1)]
            s2 = [b[random.randrange(n2)] for _ in range(n2)]
            diffs.append(sum(s1) / n1 - sum(s2) / n2)
        diffs.sort()
        lower = diffs[int((alpha / 2) * n_bootstrap)]
        upper = diffs[int((1 - alpha / 2) * n_bootstrap) - 1]
        prob_ad_better = sum(1 for d in diffs if d < 0) / n_bootstrap
        return lower, upper, prob_ad_better

    report: Dict[str, Any] = {"games": {}, "overall_supported": False}
    all_supported = True

    for game, rec in results.items():
        best_ad = rec.get("best_adaptive")
        best_fx = rec.get("best_fixed")

        print(f"\nGame: {game}")
        if not best_ad:
            print("  No adaptive results found.")
            all_supported = False
            continue
        if not best_fx:
            print("  No fixed-λ results found.")
            all_supported = False
            continue

        # Collect raw values for the selected configs
        ad_la = best_ad.get("lambda_alpha")
        fx_lam = best_fx.get("regret_weight")
        ad_ce = best_ad.get("mix_ce_weight")
        fx_ce = best_fx.get("mix_ce_weight")

        ad_vals = collect_values(
            game,
            "adaptive",
            lambda_alpha=ad_la,
            regret_weight=None,
            mix_ce_weight=ad_ce,
        )
        fx_vals = collect_values(
            game, "fixed", lambda_alpha=None, regret_weight=fx_lam, mix_ce_weight=fx_ce
        )

        ad_mean, ad_std = mean_std(ad_vals)
        fx_mean, fx_std = mean_std(fx_vals)
        t_stat, df = welch_t_df(ad_vals, fx_vals)
        ci_lo, ci_hi, p_ad_better = bootstrap_diff_ci(
            ad_vals, fx_vals, n_bootstrap=5000, alpha=0.05
        )  # 5k for speed

        # Print summary
        print(
            f"  Best Adaptive: mean={ad_mean:.4f} ± {ad_std:.4f} (alpha={ad_la}, mix_ce={ad_ce}, n={len(ad_vals)})"
        )
        print(
            f"  Best Fixed:    mean={fx_mean:.4f} ± {fx_std:.4f} (λ={fx_lam}, mix_ce={fx_ce}, n={len(fx_vals)})"
        )
        print(
            f"  Welch t-test (no p-value, df≈{df:.1f}): t={t_stat:.3f}, mean_diff={ad_mean - fx_mean:+.5f}"
        )
        print(
            f"  Bootstrap 95% CI of (adaptive - fixed): [{ci_lo:+.5f}, {ci_hi:+.5f}], P(adaptive<fixed)={p_ad_better:.3f}"
        )

        # Decide support: requires lower mean AND bootstrap CI excludes 0 in the favorable direction
        adaptive_better = ad_mean < fx_mean
        significant_boot = (ci_hi < 0) or (ci_lo > 0)
        supported = adaptive_better and significant_boot

        if supported:
            print(
                "  -> Verdict: SUPPORTED (adaptive lower and bootstrap CI excludes 0)."
            )
        else:
            print("  -> Verdict: NOT SUPPORTED with current seeds/iterations.")
            all_supported = False

        # Populate report
        report["games"][game] = {
            "adaptive": {
                "lambda_alpha": ad_la,
                "values": ad_vals,
                "mean": ad_mean,
                "std": ad_std,
                "n": len(ad_vals),
            },
            "fixed": {
                "lambda": fx_lam,
                "values": fx_vals,
                "mean": fx_mean,
                "std": fx_std,
                "n": len(fx_vals),
            },
            "welch_t": {"t_statistic": t_stat, "df": df},
            "bootstrap": {
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "prob_adaptive_better": p_ad_better,
            },
            "supported": supported,
        }

    report["overall_supported"] = all_supported
    print(
        "\nOverall hypothesis status:",
        "SUPPORTED" if all_supported else "NOT YET SUPPORTED",
    )

    # Persist JSON report next to aggregate summary
    try:
        final_dir = results_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        report_path = final_dir / "hypothesis_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nHypothesis report saved to: {report_path}")
    except Exception as e:
        print(f"[WARN] Could not save hypothesis report: {e}")

    return report


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    ensure_project_layout(project_root)

    parser = argparse.ArgumentParser(
        description="Run all required experiments (adaptive/fixed λ sweeps + baselines) and plot."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated integer seeds (e.g., 0,1,2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=120,
        help="Training iterations per run (default: 120)",
    )
    parser.add_argument(
        "--games",
        type=str,
        default="kuhn_poker,leduc_poker",
        help="Comma-separated games (kuhn_poker,leduc_poker)",
    )
    parser.add_argument(
        "--adaptive-alphas",
        type=str,
        default="0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated lambda_alpha values for adaptive ARMAC (e.g., 0.25,0.5,1.0,2.0,4.0)",
    )
    parser.add_argument(
        "--fixed-lambdas",
        type=str,
        default="0.0,0.05,0.1,0.25,0.5,0.75",
        help="Comma-separated λ values for fixed ARMAC (e.g., 0.0,0.05,0.1,0.25,0.5,0.75)",
    )
    parser.add_argument(
        "--support-preset",
        type=str,
        choices=["fast", "standard", "strong"],
        default=None,
        help="Preset for statistical support: fast (3 seeds, 120 iters), standard (5 seeds, 300 iters), strong (10 seeds, 1000 iters). Overrides seeds/iterations.",
    )
    parser.add_argument(
        "--mix-ce-weights",
        type=str,
        default=None,
        help="Comma-separated cross-entropy-to-mixture weights for ARMAC actor loss (e.g., 0.1,0.2,0.5).",
    )
    parser.add_argument(
        "--skip-armac-adaptive", action="store_true", help="Skip ARMAC adaptive λ runs"
    )
    parser.add_argument(
        "--skip-armac-fixed", action="store_true", help="Skip ARMAC fixed λ runs"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip Deep CFR / SD-CFR runs"
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only regenerate plots from existing results (no new runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )

    args = parser.parse_args()
    seeds = parse_csv_ints(args.seeds)
    games = parse_csv_strs(args.games)
    adaptive_alphas = parse_csv_floats(args.adaptive_alphas)
    fixed_lambdas = parse_csv_floats(args.fixed_lambdas)
    mix_ce_weights = (
        parse_csv_floats(args.mix_ce_weights) if args.mix_ce_weights else []
    )
    # Apply support presets to strengthen evidence if requested
    if args.support_preset:
        preset = args.support_preset
        if preset == "fast":
            # keep user-provided seeds/iters; ensure at least defaults
            args.iterations = max(args.iterations, 120)
            if not seeds:
                seeds = [0, 1, 2]
        elif preset == "standard":
            args.iterations = max(args.iterations, 300)
            seeds = list(range(5))
            # ensure grids are reasonably broad
            adaptive_alphas = sorted(set(adaptive_alphas + [0.25, 0.5, 1.0, 2.0]))
            fixed_lambdas = sorted(set(fixed_lambdas + [0.0, 0.1, 0.25, 0.5]))
        elif preset == "strong":
            args.iterations = max(args.iterations, 1000)
            seeds = list(range(10))
            # broaden grids further
            adaptive_alphas = sorted(set(adaptive_alphas + [0.25, 0.5, 1.0, 2.0, 4.0]))
            fixed_lambdas = sorted(
                set(fixed_lambdas + [0.0, 0.05, 0.1, 0.25, 0.5, 0.75])
            )

    print("Dual RL Poker - All-in-One Experiment Runner")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Games: {games}")
    print(f"Seeds: {seeds}")
    print(f"Iterations: {args.iterations}")
    print(f"Adaptive alphas: {adaptive_alphas}")
    print(f"Fixed lambdas: {fixed_lambdas}")
    print(
        f"Mix CE weights: {mix_ce_weights if mix_ce_weights else '[algorithm default]'}"
    )
    print(f"Support preset: {args.support_preset or 'none'}")
    print(f"Skip adaptive: {args.skip_armac_adaptive}")
    print(f"Skip fixed: {args.skip_armac_fixed}")
    print(f"Skip baselines: {args.skip_baselines}")
    print(f"Plots only: {args.plots_only}")
    print(f"Dry run: {args.dry_run}")

    if not args.plots_only:
        # 1) ARMAC adaptive-λ sweeps
        if not args.skip_armac_adaptive:
            ce_grid = mix_ce_weights if mix_ce_weights else [None]
            for ce_w in ce_grid:
                for alpha in adaptive_alphas:
                    cmd = [
                        sys.executable,
                        str(project_root / "run_experiments.py"),
                        "--algorithms",
                        "armac",
                        "--games",
                        ",".join(games),
                        "--seeds",
                        ",".join(str(s) for s in seeds),
                        "--iterations",
                        str(args.iterations),
                        "--armac-lambda-mode",
                        "adaptive",
                        "--armac-regret-weight",
                        "0.1",  # initial weight for mixing
                        "--armac-lambda-alpha",
                        str(alpha),
                    ]
                    if ce_w is not None:
                        cmd += ["--armac-mix-ce-weight", str(ce_w)]
                    code = run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)
                    if code != 0:
                        print(
                            f"[WARN] Adaptive λ run failed (alpha={alpha}, mix_ce_weight={ce_w}), continuing."
                        )

        # 2) ARMAC fixed-λ sweeps
        if not args.skip_armac_fixed:
            ce_grid = mix_ce_weights if mix_ce_weights else [None]
            for ce_w in ce_grid:
                for lam in fixed_lambdas:
                    cmd = [
                        sys.executable,
                        str(project_root / "run_experiments.py"),
                        "--algorithms",
                        "armac",
                        "--games",
                        ",".join(games),
                        "--seeds",
                        ",".join(str(s) for s in seeds),
                        "--iterations",
                        str(args.iterations),
                        "--armac-lambda-mode",
                        "fixed",
                        "--armac-regret-weight",
                        str(lam),
                    ]
                    if ce_w is not None:
                        cmd += ["--armac-mix-ce-weight", str(ce_w)]
                    code = run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)
                    if code != 0:
                        print(
                            f"[WARN] Fixed λ run failed (λ={lam}, mix_ce_weight={ce_w}), continuing."
                        )

        # 3) Baselines (Deep CFR, SD-CFR)
        if not args.skip_baselines:
            cmd = [
                sys.executable,
                str(project_root / "run_experiments.py"),
                "--algorithms",
                "deep_cfr,sd_cfr",
                "--games",
                ",".join(games),
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--iterations",
                str(args.iterations),
            ]
            code = run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)
            if code != 0:
                print(f"[WARN] Baseline run failed, continuing.")

    # 4) Generate plots/tables from actual results
    plot_cmd = [sys.executable, str(project_root / "create_plots.py")]
    code = run_cmd(plot_cmd, cwd=project_root, dry_run=args.dry_run)
    if code != 0:
        print("[WARN] Plot generation failed.")

    # 5) Summarize and assess hypothesis using actual results
    try:
        all_results = collect_results(results_dir)
        summary = summarize_exploitability(all_results)
        # Save an aggregate summary for records
        final_dir = results_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
        # Print quick baseline comparison table (by key)
        print("\n=== Final Exploitability Summary (mean ± std, n) ===")
        for key in sorted(summary.keys()):
            s = summary[key]
            print(f"{key}: {s['mean']:.4f} ± {s['std']:.4f} (n={s['n']})")

        ad_vs_fx = compare_adaptive_vs_fixed(summary)
        print_hypothesis_assessment(ad_vs_fx)
        print(f"\nAggregate summary saved to: {final_dir / 'aggregate_summary.json'}")
        print(f"Plots saved under: {results_dir / 'plots'}")
    except Exception as e:
        print(f"[WARN] Could not summarize results: {e}")

    print("\nAll requested runs and plotting steps completed.")
    print(
        "Note: This script uses only actual evaluations; no placeholders or simulated metrics are written."
    )


if __name__ == "__main__":
    main()
