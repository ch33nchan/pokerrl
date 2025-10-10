"""Statistical analysis utilities for Dual RL Poker experiments."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path


def compute_confidence_intervals(data: List[float],
                               confidence_level: float = 0.95,
                               method: str = 'bootstrap') -> Tuple[float, float, float]:
    """Compute confidence intervals for a list of values.

    Args:
        data: List of numerical values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Method for computing CI ('bootstrap' or 't')

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0

    data = np.array(data)
    mean = np.mean(data)

    if method == 'bootstrap':
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        n = len(data)

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    elif method == 't':
        # Student's t-distribution confidence intervals
        n = len(data)
        std_err = stats.sem(data)
        alpha = 1 - confidence_level
        t_val = stats.t.ppf(1 - alpha / 2, n - 1)
        margin_error = t_val * std_err
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error

    else:
        raise ValueError(f"Unknown method: {method}")

    return mean, lower_bound, upper_bound


def perform_statistical_tests(results_data: Dict[str, List[float]],
                            control_method: str = 'deep_cfr',
                            alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Perform statistical tests comparing methods to control.

    Args:
        results_data: Dictionary mapping methods to lists of results
        control_method: Name of the control method
        alpha: Significance level

    Returns:
        Dictionary containing test results
    """
    if control_method not in results_data:
        raise ValueError(f"Control method {control_method} not found in data")

    control_data = results_data[control_method]
    test_results = {}

    for method, method_data in results_data.items():
        if method == control_method:
            continue

        # Perform t-test
        t_stat, t_pvalue = stats.ttest_ind(method_data, control_data)

        # Perform Wilcoxon rank-sum test (non-parametric)
        wilcoxon_stat, wilcoxon_pvalue = stats.ranksums(method_data, control_data)

        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(((len(method_data) - 1) * np.var(method_data, ddof=1) +
                             (len(control_data) - 1) * np.var(control_data, ddof=1)) /
                            (len(method_data) + len(control_data) - 2))
        cohens_d = (np.mean(method_data) - np.mean(control_data)) / pooled_std if pooled_std > 0 else 0

        test_results[method] = {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_pvalue': wilcoxon_pvalue,
            'cohens_d': cohens_d,
            'significant_t': t_pvalue < alpha,
            'significant_wilcoxon': wilcoxon_pvalue < alpha,
            'effect_size_interpretation': _interpret_cohens_d(cohens_d)
        }

    # Apply Holm-Bonferroni correction
    p_values = [result['t_pvalue'] for result in test_results.values()]
    _, corrected_p_values, _, _ = stats.multitest.multipletests(p_values, method='holm')

    for i, method in enumerate(test_results.keys()):
        test_results[method]['holm_corrected_pvalue'] = corrected_p_values[i]
        test_results[method]['significant_holm'] = corrected_p_values[i] < alpha

    return test_results


def _interpret_cohens_d(cohens_d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        cohens_d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_convergence_metrics(evaluation_history: List[Dict[str, float]],
                               convergence_threshold: float = 0.01) -> Dict[str, float]:
    """Compute convergence metrics from evaluation history.

    Args:
        evaluation_history: List of evaluation dictionaries
        convergence_threshold: Threshold for determining convergence

    Returns:
        Dictionary of convergence metrics
    """
    if not evaluation_history:
        return {}

    exploitability = [eval_item['exploitability'] for eval_item in evaluation_history]
    iterations = [eval_item['iteration'] for eval_item in evaluation_history]

    # Compute convergence rate (slope of log exploitability decrease)
    if len(exploitability) > 10:
        log_exploit = np.log(np.array(exploitability) + 1e-8)
        # Use linear regression on the last 50% of the data
        start_idx = len(log_exploit) // 2
        x = np.array(iterations[start_idx:])
        y = log_exploit[start_idx:]

        if len(x) > 1:
            slope, _, _, _, _ = stats.linregress(x, y)
            convergence_rate = -slope  # Negative slope is good
        else:
            convergence_rate = 0.0
    else:
        convergence_rate = 0.0

    # Compute time to convergence
    converged = False
    time_to_convergence = len(iterations)  # Default to full training

    for i in range(10, len(exploitability)):  # Start checking after 10 iterations
        recent_vals = exploitability[max(0, i-10):i]
        if np.std(recent_vals) / (np.mean(recent_vals) + 1e-8) < convergence_threshold:
            time_to_convergence = iterations[i]
            converged = True
            break

    # Compute stability score (inverse of variance in later iterations)
    if len(exploitability) > 20:
        tail_vals = exploitability[-20:]
        stability_score = 1.0 / (np.var(tail_vals) + 1e-8)
    else:
        stability_score = 0.0

    # Compute efficiency (final performance / training time)
    if exploitability and iterations:
        final_performance = exploitability[-1]
        training_time = iterations[-1]
        efficiency = 1.0 / (final_performance * training_time + 1e-8)
    else:
        efficiency = 0.0

    return {
        'convergence_rate': convergence_rate,
        'time_to_convergence': time_to_convergence,
        'converged': converged,
        'stability_score': stability_score,
        'efficiency': efficiency,
        'final_exploitability': exploitability[-1] if exploitability else 0.0
    }


def analyze_performance_variance(results_data: Dict[str, Dict[str, List[float]]],
                               metrics: List[str] = ['exploitability', 'nash_conv']) -> Dict[str, Dict[str, Any]]:
    """Analyze performance variance across seeds and algorithms.

    Args:
        results_data: Nested dictionary of results
        metrics: List of metrics to analyze

    Returns:
        Dictionary containing variance analysis
    """
    analysis = {}

    for method, method_results in results_data.items():
        method_analysis = {}

        for metric in metrics:
            if metric in method_results:
                values = method_results[metric]
                values_array = np.array(values)

                method_analysis[metric] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'var': np.var(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'median': np.median(values_array),
                    'q25': np.percentile(values_array, 25),
                    'q75': np.percentile(values_array, 75),
                    'cv': np.std(values_array) / (np.mean(values_array) + 1e-8),  # Coefficient of variation
                    'num_samples': len(values_array)
                }

                # Compute confidence intervals
                mean, lower, upper = compute_confidence_intervals(values)
                method_analysis[metric]['ci_mean'] = mean
                method_analysis[metric]['ci_lower'] = lower
                method_analysis[metric]['ci_upper'] = upper

        analysis[method] = method_analysis

    return analysis


def compute_rankings(results_data: Dict[str, List[float]],
                    lower_is_better: bool = True) -> Dict[str, Dict[str, Any]]:
    """Compute rankings and statistical significance of differences.

    Args:
        results_data: Dictionary mapping methods to lists of results
        lower_is_better: Whether lower values are better

    Returns:
        Dictionary containing rankings and significance tests
    """
    # Compute mean performance for each method
    mean_performance = {}
    for method, values in results_data.items():
        mean_performance[method] = np.mean(values)

    # Sort methods by performance
    sorted_methods = sorted(mean_performance.items(),
                          key=lambda x: x[1] if lower_is_better else -x[1])

    rankings = {}
    for rank, (method, mean_perf) in enumerate(sorted_methods, 1):
        rankings[method] = {
            'rank': rank,
            'mean_performance': mean_perf,
            'num_samples': len(results_data[method])
        }

    # Perform pairwise comparisons
    pairwise_tests = {}
    methods = list(results_data.keys())

    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            data1 = results_data[method1]
            data2 = results_data[method2]

            # Statistical test
            t_stat, p_value = stats.ttest_ind(data1, data2)

            # Effect size
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                                 (len(data2) - 1) * np.var(data2, ddof=1)) /
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

            pairwise_tests[f"{method1}_vs_{method2}"] = {
                'method1': method1,
                'method2': method2,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }

    return {
        'rankings': rankings,
        'pairwise_tests': pairwise_tests,
        'best_method': sorted_methods[0][0] if sorted_methods else None
    }


def bootstrap_performance_comparison(method1_data: List[float],
                                  method2_data: List[float],
                                  n_bootstrap: int = 10000,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
    """Perform bootstrap comparison between two methods.

    Args:
        method1_data: Performance data for method 1
        method2_data: Performance data for method 2
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary containing bootstrap comparison results
    """
    n1, n2 = len(method1_data), len(method2_data)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Bootstrap samples
        sample1 = np.random.choice(method1_data, size=n1, replace=True)
        sample2 = np.random.choice(method2_data, size=n2, replace=True)

        # Compute difference in means
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Probability that method1 is better than method2
    prob_method1_better = np.mean(bootstrap_diffs < 0) if np.mean(method1_data) < np.mean(method2_data) else np.mean(bootstrap_diffs > 0)

    return {
        'bootstrap_mean_difference': np.mean(bootstrap_diffs),
        'bootstrap_std_difference': np.std(bootstrap_diffs),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'prob_method1_better': prob_method1_better,
        'significant_difference': (ci_lower > 0) or (ci_upper < 0)
    }


def generate_summary_statistics(results_dir: str) -> Dict[str, Any]:
    """Generate comprehensive summary statistics from experiment results.

    Args:
        results_dir: Directory containing experiment result files

    Returns:
        Dictionary containing summary statistics
    """
    results_path = Path(results_dir)
    summary = {
        'total_experiments': 0,
        'algorithms': set(),
        'games': set(),
        'results_by_algorithm': defaultdict(list),
        'results_by_game': defaultdict(list),
        'performance_summary': {}
    }

    # Load all result files
    for result_file in results_path.glob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)

            summary['total_experiments'] += 1
            summary['algorithms'].add(result['method'])
            summary['games'].add(result['game'])

            # Extract final performance metrics
            if 'evaluation_history' in result and result['evaluation_history']:
                final_eval = result['evaluation_history'][-1]
                summary['results_by_algorithm'][result['method']].append(final_eval['exploitability'])
                summary['results_by_game'][result['game']].append(final_eval['exploitability'])

        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    # Convert sets to lists for JSON serialization
    summary['algorithms'] = list(summary['algorithms'])
    summary['games'] = list(summary['games'])

    # Compute performance summary for each algorithm
    for algorithm in summary['algorithms']:
        if algorithm in summary['results_by_algorithm']:
            values = summary['results_by_algorithm'][algorithm]
            mean, lower, upper = compute_confidence_intervals(values)

            summary['performance_summary'][algorithm] = {
                'mean_exploitability': mean,
                'ci_lower': lower,
                'ci_upper': upper,
                'std_dev': np.std(values),
                'num_runs': len(values)
            }

    return summary