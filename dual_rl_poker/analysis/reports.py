"""Report generation utilities for Dual RL Poker experiments."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .statistics import (
    compute_confidence_intervals,
    perform_statistical_tests,
    compute_convergence_metrics,
    analyze_performance_variance,
    generate_summary_statistics
)
from .plotting import (
    plot_learning_curves,
    plot_exploitability_curves,
    plot_comparison_charts,
    plot_loss_components,
    plot_convergence_analysis
)


def generate_experiment_report(results_dir: str,
                            output_dir: str,
                            report_name: str = "experiment_report") -> Dict[str, Any]:
    """Generate a comprehensive experiment report.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save report files
        report_name: Base name for report files

    Returns:
        Dictionary containing report data
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_dir)

    # Load detailed experiment data
    detailed_results = {}
    convergence_data = {}
    training_data = {}

    for result_file in results_path.glob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)

            exp_id = result.get('experiment_id', result_file.stem)
            detailed_results[exp_id] = result

            # Extract convergence metrics
            if 'evaluation_history' in result:
                convergence_metrics = compute_convergence_metrics(result['evaluation_history'])
                convergence_data[exp_id] = convergence_metrics

            # Extract training data for plotting
            if 'training_history' in result:
                training_data[exp_id] = result['training_history']

        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    # Generate plots
    plot_files = {}

    # Learning curves
    if training_data:
        learning_curves_file = output_path / f"{report_name}_learning_curves.png"
        plot_learning_curves(training_data, save_path=str(learning_curves_file))
        plot_files['learning_curves'] = str(learning_curves_file)

    # Exploitability curves
    exploitability_file = output_path / f"{report_name}_exploitability_curves.png"
    plot_exploitability_curves(results_dir, save_path=str(exploitability_file))
    plot_files['exploitability_curves'] = str(exploitability_file)

    # Comparison charts
    comparison_file = output_path / f"{report_name}_comparison_charts.png"
    plot_comparison_charts(detailed_results, save_path=str(comparison_file))
    plot_files['comparison_charts'] = str(comparison_file)

    # Loss components
    if training_data:
        loss_components_file = output_path / f"{report_name}_loss_components.png"
        plot_loss_components(training_data, save_path=str(loss_components_file))
        plot_files['loss_components'] = str(loss_components_file)

    # Convergence analysis
    if convergence_data:
        convergence_file = output_path / f"{report_name}_convergence_analysis.png"
        plot_convergence_analysis(convergence_data, save_path=str(convergence_file))
        plot_files['convergence_analysis'] = str(convergence_file)

    # Generate statistical analysis
    statistical_analysis = {}
    for game in summary_stats['games']:
        # Collect data for this game
        game_data = {}
        for exp_id, result in detailed_results.items():
            if result.get('game') == game and 'evaluation_history' in result:
                method = result.get('method', exp_id)
                final_exploitability = result['evaluation_history'][-1]['exploitability']
                if method not in game_data:
                    game_data[method] = []
                game_data[method].append(final_exploitability)

        if len(game_data) > 1:
            statistical_analysis[game] = perform_statistical_tests(game_data)

    # Create comprehensive report
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'results_directory': results_dir,
            'total_experiments': summary_stats['total_experiments'],
            'algorithms': summary_stats['algorithms'],
            'games': summary_stats['games']
        },
        'summary_statistics': summary_stats,
        'detailed_results': detailed_results,
        'convergence_analysis': convergence_data,
        'statistical_analysis': statistical_analysis,
        'plot_files': plot_files,
        'performance_summary': _create_performance_summary(detailed_results),
        'recommendations': _generate_recommendations(summary_stats, statistical_analysis, convergence_data)
    }

    # Save report
    report_file = output_path / f"{report_name}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate markdown report
    _generate_markdown_report(report, output_path / f"{report_name}.md")

    # Generate summary tables
    create_summary_tables(detailed_results, output_path)

    return report


def _create_performance_summary(detailed_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a performance summary from detailed results.

    Args:
        detailed_results: Dictionary of detailed experiment results

    Returns:
        Performance summary dictionary
    """
    summary = {
        'by_algorithm': {},
        'by_game': {},
        'overall_best': {},
        'training_efficiency': {}
    }

    # Aggregate by algorithm
    alg_performance = {}
    for exp_id, result in detailed_results.items():
        alg = result.get('method', 'unknown')
        if alg not in alg_performance:
            alg_performance[alg] = []

        if 'evaluation_history' in result and result['evaluation_history']:
            final_exploitability = result['evaluation_history'][-1]['exploitability']
            training_time = result.get('total_time', 0)
            alg_performance[alg].append({
                'exploitability': final_exploitability,
                'training_time': training_time
            })

    for alg, runs in alg_performance.items():
        exploitabilities = [run['exploitability'] for run in runs]
        training_times = [run['training_time'] for run in runs]

        mean_exploit, ci_lower, ci_upper = compute_confidence_intervals(exploitabilities)

        summary['by_algorithm'][alg] = {
            'mean_exploitability': mean_exploit,
            'confidence_interval': [ci_lower, ci_upper],
            'num_runs': len(runs),
            'mean_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times)
        }

    # Find best performing methods
    if summary['by_algorithm']:
        best_exploit = min(summary['by_algorithm'].items(),
                          key=lambda x: x[1]['mean_exploitability'])
        summary['overall_best']['exploitability'] = {
            'algorithm': best_exploit[0],
            'performance': best_exploit[1]['mean_exploitability']
        }

        fastest = min(summary['by_algorithm'].items(),
                     key=lambda x: x[1]['mean_training_time'])
        summary['overall_best']['training_time'] = {
            'algorithm': fastest[0],
            'time': fastest[1]['mean_training_time']
        }

    return summary


def _generate_recommendations(summary_stats: Dict[str, Any],
                            statistical_analysis: Dict[str, Any],
                            convergence_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on experimental results.

    Args:
        summary_stats: Summary statistics
        statistical_analysis: Statistical test results
        convergence_data: Convergence metrics

    Returns:
        List of recommendations
    """
    recommendations = []

    # Performance recommendations
    if summary_stats['performance_summary']:
        best_method = min(summary_stats['performance_summary'].items(),
                         key=lambda x: x[1]['mean_exploitability'])

        recommendations.append(
            f"Best overall performance: {best_method[0]} with mean exploitability of "
            f"{best_method[1]['mean_exploitability']:.4f} mbb/h"
        )

    # Statistical significance recommendations
    significant_differences = []
    for game, tests in statistical_analysis.items():
        for method, test_results in tests.items():
            if test_results.get('significant_holm', False):
                effect_size = test_results.get('cohens_d', 0)
                if abs(effect_size) > 0.5:  # Medium or large effect size
                    significant_differences.append(
                        f"{method} shows significant improvement in {game} "
                        f"(Cohen's d: {effect_size:.2f})"
                    )

    if significant_differences:
        recommendations.extend(significant_differences)

    # Convergence recommendations
    if convergence_data:
        convergence_rates = [(exp_id, data.get('convergence_rate', 0))
                           for exp_id, data in convergence_data.items()]
        if convergence_rates:
            fastest_convergence = max(convergence_rates, key=lambda x: x[1])
            recommendations.append(
                f"Fastest convergence: {fastest_convergence[0]} "
                f"(rate: {fastest_convergence[1]:.6f})"
            )

        # Stability analysis
        stable_methods = [(exp_id, data.get('stability_score', 0))
                         for exp_id, data in convergence_data.items()
                         if data.get('stability_score', 0) > 0]
        if stable_methods:
            most_stable = max(stable_methods, key=lambda x: x[1])
            recommendations.append(
                f"Most stable training: {most_stable[0]} "
                f"(stability score: {most_stable[1]:.2f})"
            )

    # Practical recommendations
    if len(summary_stats['algorithms']) > 1:
        recommendations.append(
            "Consider computational resources when choosing between algorithms. "
            "Some methods may achieve better performance but require significantly more training time."
        )

    recommendations.append(
        "Run additional seeds for methods with high variance to improve statistical confidence."
    )

    return recommendations


def create_summary_tables(detailed_results: Dict[str, Any],
                         output_dir: str):
    """Create summary tables in CSV format for easy analysis.

    Args:
        detailed_results: Dictionary of detailed experiment results
        output_dir: Output directory for CSV files
    """
    output_path = Path(output_dir)

    # Create performance table
    performance_data = []
    for exp_id, result in detailed_results.items():
        if 'evaluation_history' in result and result['evaluation_history']:
            final_eval = result['evaluation_history'][-1]
            performance_data.append({
                'experiment_id': exp_id,
                'game': result.get('game', 'unknown'),
                'method': result.get('method', 'unknown'),
                'seed': result.get('seed', 'unknown'),
                'final_exploitability': final_eval.get('exploitability', 0),
                'final_nash_conv': final_eval.get('nash_conv', 0),
                'training_time': result.get('total_time', 0),
                'num_iterations': result.get('num_iterations', 0)
            })

    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(output_path / 'performance_summary.csv', index=False)

        # Create summary by algorithm
        alg_summary = performance_df.groupby(['method']).agg({
            'final_exploitability': ['mean', 'std', 'min', 'max'],
            'final_nash_conv': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'experiment_id': 'count'
        }).round(6)

        alg_summary.columns = ['_'.join(col).strip() for col in alg_summary.columns]
        alg_summary.to_csv(output_path / 'algorithm_summary.csv')

        # Create summary by game
        game_summary = performance_df.groupby(['game', 'method']).agg({
            'final_exploitability': ['mean', 'std', 'count'],
            'training_time': 'mean'
        }).round(6)

        game_summary.columns = ['_'.join(col).strip() for col in game_summary.columns]
        game_summary.to_csv(output_path / 'game_algorithm_summary.csv')


def _generate_markdown_report(report: Dict[str, Any], output_file: Path):
    """Generate a markdown report from the report data.

    Args:
        report: Report dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("# Dual RL Poker Experiment Report\n\n")
        f.write(f"**Generated:** {report['metadata']['generated_at']}\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write(f"- **Total Experiments:** {report['metadata']['total_experiments']}\n")
        f.write(f"- **Algorithms:** {', '.join(report['metadata']['algorithms'])}\n")
        f.write(f"- **Games:** {', '.join(report['metadata']['games'])}\n\n")

        # Performance Summary
        f.write("## Performance Summary\n\n")
        if 'overall_best' in report['performance_summary']:
            best = report['performance_summary']['overall_best']
            f.write(f"- **Best Performance:** {best['exploitability']['algorithm']} "
                   f"({best['exploitability']['performance']:.4f} mbb/h)\n")
            f.write(f"- **Fastest Training:** {best['training_time']['algorithm']} "
                   f"({best['training_time']['time']:.2f}s)\n\n")

        # Algorithm Performance Table
        f.write("### Algorithm Performance\n\n")
        f.write("| Algorithm | Mean Exploitability | 95% CI | Runs | Mean Training Time |\n")
        f.write("|-----------|-------------------|--------|------|-------------------|\n")

        for alg, perf in report['performance_summary']['by_algorithm'].items():
            ci_lower, ci_upper = perf['confidence_interval']
            f.write(f"| {alg} | {perf['mean_exploitability']:.4f} | "
                   f"[{ci_lower:.4f}, {ci_upper:.4f}] | {perf['num_runs']} | "
                   f"{perf['mean_training_time']:.2f}s |\n")

        f.write("\n")

        # Statistical Analysis
        if report['statistical_analysis']:
            f.write("## Statistical Analysis\n\n")
            for game, tests in report['statistical_analysis'].items():
                f.write(f"### {game.replace('_', ' ').title()}\n\n")
                for method, results in tests.items():
                    if results.get('significant_holm', False):
                        f.write(f"- **{method}**: Significant improvement "
                               f"(p={results['holm_corrected_pvalue']:.4f}, "
                               f"Cohen's d={results['cohens_d']:.2f})\n")
                f.write("\n")

        # Recommendations
        if report['recommendations']:
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

        # Plots
        if report['plot_files']:
            f.write("## Visualizations\n\n")
            for plot_name, plot_path in report['plot_files'].items():
                plot_filename = Path(plot_path).name
                f.write(f"- **{plot_name.replace('_', ' ').title()}:** "
                       f"![{plot_name}]({plot_filename})\n")


def create_latex_tables(detailed_results: Dict[str, Any],
                       output_dir: str):
    """Create LaTeX tables for inclusion in papers.

    Args:
        detailed_results: Dictionary of detailed experiment results
        output_dir: Output directory for LaTeX files
    """
    output_path = Path(output_dir)

    # Performance table
    performance_data = []
    for exp_id, result in detailed_results.items():
        if 'evaluation_history' in result and result['evaluation_history']:
            final_eval = result['evaluation_history'][-1]
            performance_data.append({
                'method': result.get('method', 'unknown'),
                'game': result.get('game', 'unknown'),
                'exploitability': final_eval.get('exploitability', 0),
                'nash_conv': final_eval.get('nash_conv', 0),
                'time': result.get('total_time', 0)
            })

    if performance_data:
        df = pd.DataFrame(performance_data)

        # Create main results table
        summary = df.groupby(['method', 'game']).agg({
            'exploitability': ['mean', 'std'],
            'nash_conv': 'mean',
            'time': 'mean'
        }).round(4)

        summary.columns = ['Exploitability', 'Std', 'NashConv', 'Time']
        summary = summary.reset_index()

        latex_table = summary.to_latex(index=False, escape=False,
                                      caption="Algorithm performance comparison",
                                      label="tab:performance_comparison")

        with open(output_path / 'performance_table.tex', 'w') as f:
            f.write(latex_table)

        # Create algorithm comparison table
        alg_summary = df.groupby('method').agg({
            'exploitability': ['mean', 'std'],
            'time': 'mean'
        }).round(4)

        alg_summary.columns = ['Exploitability', 'Std', 'Time']
        alg_summary = alg_summary.reset_index()

        alg_table = alg_summary.to_latex(index=False, escape=False,
                                        caption="Algorithm summary across all games",
                                        label="tab:algorithm_summary")

        with open(output_path / 'algorithm_table.tex', 'w') as f:
            f.write(alg_table)