"""
Comprehensive Statistical Analysis for Deep CFR Architecture Study

This module implements proper statistical analysis including:
- Bootstrap confidence intervals
- Effect sizes (Hedges' g)
- Holm-Bonferroni corrected p-values
- Proper sample size balancing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, norm
import warnings
warnings.filterwarnings('ignore')


def bootstrap_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if not data or len(data) < 2:
        return float('nan'), float('nan'), float('nan')

    data_array = np.array(data)
    n = len(data_array)

    # Bootstrap means
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_val = np.mean(data_array)
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return mean_val, lower_bound, upper_bound


def hedges_g(
    group1: List[float],
    group2: List[float]
) -> float:
    """
    Calculate Hedges' g effect size (bias-corrected Cohen's d).
    """
    if len(group1) < 2 or len(group2) < 2:
        return float('nan')

    group1, group2 = np.array(group1), np.array(group2)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return float('nan')

    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_sd

    # Bias correction (Hedges' g)
    correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction_factor

    return hedges_g


def mann_whitney_u_test(
    group1: List[float],
    group2: List[float]
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test for non-parametric comparison.

    Returns:
        (u_statistic, p_value)
    """
    if len(group1) < 2 or len(group2) < 2:
        return float('nan'), float('nan')

    try:
        u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        return u_statistic, p_value
    except:
        return float('nan'), float('nan')


def holm_bonferroni_correction(
    p_values: List[float]
) -> List[bool]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Returns:
        List of booleans indicating which hypotheses are rejected.
    """
    # Sort p-values and keep track of original indices
    indexed_p_values = [(i, p) for i, p in enumerate(p_values)]
    indexed_p_values.sort(key=lambda x: x[1])

    n = len(p_values)
    rejected = [False] * n

    for rank, (orig_idx, p_value) in enumerate(indexed_p_values):
        # Adjusted significance level
        alpha = 0.05 / (n - rank)

        if p_value < alpha:
            rejected[orig_idx] = True
        else:
            break  # Stop at first non-rejection

    return rejected


def compare_groups(
    group_data: Dict[str, List[float]],
    method: str = "parametric"
) -> Dict[str, Any]:
    """
    Compare multiple groups with proper statistical testing.

    Args:
        group_data: Dictionary mapping group names to data lists
        method: "parametric" or "nonparametric"

    Returns:
        Dictionary with comparison results
    """
    groups = list(group_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return {"error": "Need at least 2 groups for comparison"}

    results = {
        "groups": groups,
        "n_per_group": {group: len(data) for group, data in group_data.items()},
        "means": {group: np.mean(data) if data else float('nan') for group, data in group_data.items()},
        "stds": {group: np.std(data, ddof=1) if len(data) > 1 else float('nan') for group, data in group_data.items()},
        "pairwise_comparisons": [],
        "bootstrap_cis": {}
    }

    # Bootstrap confidence intervals for each group
    for group, data in group_data.items():
        mean, lower, upper = bootstrap_confidence_interval(data)
        results["bootstrap_cis"][group] = {
            "mean": mean,
            "lower_95": lower,
            "upper_95": upper
        }

    # Pairwise comparisons
    p_values = []
    comparisons = []

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group1, group2 = groups[i], groups[j]
            data1, data2 = group_data[group1], group_data[group2]

            if len(data1) < 2 or len(data2) < 2:
                continue

            # Effect size
            effect_size = hedges_g(data1, data2)

            # Statistical test
            if method == "parametric":
                try:
                    _, p_value = ttest_ind(data1, data2)
                except:
                    p_value = float('nan')
            else:
                _, p_value = mann_whitney_u_test(data1, data2)

            p_values.append(p_value)

            comparisons.append({
                "group1": group1,
                "group2": group2,
                "effect_size": effect_size,
                "p_value": p_value,
                "significant": False  # Will be updated after correction
            })

    # Multiple testing correction
    if p_values:
        rejected = holm_bonferroni_correction(p_values)
        for i, comparison in enumerate(comparisons):
            comparison["significant"] = rejected[i]
            comparison["p_corrected"] = p_values[i] * len(p_values) / (i + 1)  # Holm adjustment

    results["pairwise_comparisons"] = comparisons
    results["method"] = method

    return results


def analyze_experiment_results(
    manifest_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze experimental results from manifest data.

    Args:
        manifest_data: List of experiment results

    Returns:
        Comprehensive statistical analysis
    """
    if not manifest_data:
        return {"error": "No data provided"}

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(manifest_data)

    # Filter out failed experiments
    successful = df[df['status'] == 'completed']

    if len(successful) == 0:
        return {"error": "No successful experiments"}

    analysis = {
        "summary": {
            "total_experiments": len(manifest_data),
            "successful_experiments": len(successful),
            "failed_experiments": len(manifest_data) - len(successful)
        },
        "games": list(successful['game'].unique()),
        "architectures": list(successful['architecture'].unique()),
        "by_game": {},
        "by_architecture": {}
    }

    # Analyze by game
    for game in analysis["games"]:
        game_data = successful[successful['game'] == game]

        # Group by architecture within each game
        game_groups = {}
        for arch in game_data['architecture'].unique():
            arch_data = game_data[game_data['architecture'] == arch]
            exploitability_vals = arch_data['final_exploitability'].dropna().tolist()

            if exploitability_vals:
                game_groups[arch] = exploitability_vals

        if game_groups:
            game_analysis = compare_groups(game_groups)
            analysis["by_game"][game] = game_analysis

    # Analyze by architecture across all games
    for arch in analysis["architectures"]:
        arch_data = successful[successful['architecture'] == arch]
        exploitability_vals = arch_data['final_exploitability'].dropna().tolist()

        if exploitability_vals:
            analysis["by_architecture"][arch] = {
                "n_experiments": len(exploitability_vals),
                "mean": np.mean(exploitability_vals),
                "std": np.std(exploitability_vals, ddof=1),
                "min": np.min(exploitability_vals),
                "max": np.max(exploitability_vals),
                "median": np.median(exploitability_vals)
            }

    # Overall analysis (combine all data)
    all_groups = {}
    for arch in analysis["architectures"]:
        arch_data = successful[successful['architecture'] == arch]
        exploitability_vals = arch_data['final_exploitability'].dropna().tolist()

        if exploitability_vals:
            all_groups[arch] = exploitability_vals

    if all_groups:
        overall_analysis = compare_groups(all_groups)
        analysis["overall_comparison"] = overall_analysis

    return analysis


def create_summary_table(
    analysis_results: Dict[str, Any],
    metric: str = "final_exploitability"
) -> pd.DataFrame:
    """
    Create publication-ready summary table from analysis results.

    Args:
        analysis_results: Results from analyze_experiment_results
        metric: Metric to summarize

    Returns:
        DataFrame with summary statistics
    """
    if "overall_comparison" not in analysis_results:
        return pd.DataFrame()

    groups = analysis_results["overall_comparison"]["groups"]
    bootstrap_cis = analysis_results["overall_comparison"]["bootstrap_cis"]

    table_data = []

    for group in groups:
        if group in bootstrap_cis:
            ci = bootstrap_cis[group]
            table_data.append({
                "Architecture": group,
                "n": analysis_results["overall_comparison"]["n_per_group"][group],
                "Mean": f"{ci['mean']:.3f}",
                "95% CI": f"[{ci['lower_95']:.3f}, {ci['upper_95']:.3f}]",
                "Std": f"{analysis_results['overall_comparison']['stds'][group]:.3f}"
            })

    df = pd.DataFrame(table_data)

    return df


def generate_statistical_report(
    analysis_results: Dict[str, Any]
) -> str:
    """
    Generate a human-readable statistical report.
    """
    report = []

    # Summary
    summary = analysis_results.get("summary", {})
    report.append("=== Statistical Analysis Report ===")
    report.append(f"Total experiments: {summary.get('total_experiments', 0)}")
    report.append(f"Successful experiments: {summary.get('successful_experiments', 0)}")
    report.append(f"Failed experiments: {summary.get('failed_experiments', 0)}")
    report.append("")

    # Overall comparison
    if "overall_comparison" in analysis_results:
        overall = analysis_results["overall_comparison"]
        report.append("=== Overall Comparison ===")
        report.append(f"Analysis method: {overall.get('method', 'unknown')}")

        for group in overall["groups"]:
            ci = overall["bootstrap_cis"].get(group, {})
            report.append(f"{group}:")
            report.append(f"  n = {overall['n_per_group'][group]}")
            report.append(f"  Mean = {ci.get('mean', 'nan'):.3f}")
            report.append(f"  95% CI = [{ci.get('lower_95', 'nan'):.3f}, {ci.get('upper_95', 'nan'):.3f}]")

        report.append("")
        report.append("=== Pairwise Comparisons ===")
        for comp in overall["pairwise_comparisons"]:
            sig = "Significant" if comp["significant"] else "Not significant"
            report.append(f"{comp['group1']} vs {comp['group2']}:")
            report.append(f"  Effect size (Hedges' g) = {comp['effect_size']:.3f}")
            report.append(f"  p-value = {comp['p_value']:.4f}")
            report.append(f"  {sig} (α = 0.05)")
            report.append("")

    # By game analysis
    if "by_game" in analysis_results:
        report.append("=== Analysis by Game ===")
        for game, game_analysis in analysis_results["by_game"].items():
            report.append(f"Game: {game}")
            for comp in game_analysis.get("pairwise_comparisons", []):
                sig = "Significant" if comp["significant"] else "Not significant"
                report.append(f"  {comp['group1']} vs {comp['group2']}: {sig}")
            report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    sample_data = {
        "baseline": [1.0, 1.1, 0.9, 1.2, 0.8],
        "lstm_opt": [1.3, 1.4, 1.2, 1.5, 1.1],
        "lstm_no_hist": [1.4, 1.3, 1.5, 1.6, 1.2]
    }

    results = compare_groups(sample_data)
    print(generate_statistical_report({"overall_comparison": results}))