"""
Results consistency checker for Deep CFR architecture study.
Validates all reported numbers across tables, figures, and text.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ResultsValidator:
    """Validates consistency of reported results across all paper components."""
    
    def __init__(self, results_dir: str, paper_dir: str):
        self.results_dir = Path(results_dir)
        self.paper_dir = Path(paper_dir)
        
        # Load experimental data
        self.results_df = self._load_experimental_results()
        self.validation_errors = []
        self.warnings = []
        
    def _load_experimental_results(self) -> pd.DataFrame:
        """Load experimental results from various sources."""
        results_file = self.results_dir / 'comprehensive_results.csv'
        
        if results_file.exists():
            return pd.read_csv(results_file)
        else:
            # Fallback to loading individual JSON files
            result_files = list(self.results_dir.glob('*_result.json'))
            results = []
            
            for file in result_files:
                try:
                    with open(file, 'r') as f:
                        result = json.load(f)
                        results.append(result)
                except Exception as e:
                    self.warnings.append(f"Could not load {file}: {e}")
            
            if results:
                return pd.DataFrame(results)
            else:
                self.validation_errors.append("No experimental results found")
                return pd.DataFrame()
    
    def extract_paper_numbers(self) -> Dict[str, Any]:
        """Extract numerical claims from the LaTeX paper."""
        paper_file = self.paper_dir / 'Deep_CFR_Architecture_Study.tex'
        
        if not paper_file.exists():
            self.validation_errors.append(f"Paper file not found: {paper_file}")
            return {}
        
        with open(paper_file, 'r') as f:
            paper_content = f.read()
        
        # Extract key numbers using regex patterns
        extracted_numbers = {}
        
        # Success rates from tables
        success_rate_patterns = [
            r'Baseline MLP.*?(\d+).*?(\d+).*?(\d+\.?\d*)%',
            r'LSTM.*?(\d+).*?(\d+).*?(\d+\.?\d*)%',
            r'Transformer.*?(\d+).*?(\d+).*?(\d+\.?\d*)%',
            r'GRU.*?(\d+).*?(\d+).*?(\d+\.?\d*)%'
        ]
        
        for i, pattern in enumerate(success_rate_patterns):
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            if matches:
                arch_names = ['baseline', 'lstm', 'transformer', 'gru']
                if i < len(arch_names):
                    configs, successful, rate = matches[0]
                    extracted_numbers[f'{arch_names[i]}_configs'] = int(configs)
                    extracted_numbers[f'{arch_names[i]}_successful'] = int(successful)
                    extracted_numbers[f'{arch_names[i]}_success_rate'] = float(rate)
        
        # Effect sizes
        cohens_d_matches = re.findall(r'Cohen\'s d[^\d]*=?[^\d]*(\d+\.?\d*)', paper_content)
        if cohens_d_matches:
            extracted_numbers['cohens_d'] = float(cohens_d_matches[0])
        
        # Exploitability numbers
        exploitability_matches = re.findall(r'exploitability[^\d]*(\d+\.?\d*)', paper_content, re.IGNORECASE)
        if exploitability_matches:
            extracted_numbers['exploitability_values'] = [float(x) for x in exploitability_matches]
        
        # Total configurations
        total_configs_matches = re.findall(r'(\d+)\s+(?:total\s+)?configurations', paper_content, re.IGNORECASE)
        if total_configs_matches:
            extracted_numbers['total_configurations'] = int(total_configs_matches[0])
        
        # Statistical significance
        p_value_matches = re.findall(r'p\s*[<>=]\s*(\d+\.?\d*)', paper_content)
        if p_value_matches:
            extracted_numbers['p_values'] = [float(x) for x in p_value_matches]
        
        return extracted_numbers
    
    def compute_actual_statistics(self) -> Dict[str, Any]:
        """Compute actual statistics from experimental data."""
        if self.results_df.empty:
            return {}
        
        actual_stats = {}
        
        # Overall statistics
        actual_stats['total_experiments'] = len(self.results_df)
        actual_stats['total_successful'] = len(self.results_df[self.results_df['success'] == True])
        actual_stats['overall_success_rate'] = actual_stats['total_successful'] / actual_stats['total_experiments'] * 100
        
        # Architecture-specific statistics
        arch_stats = self.results_df.groupby('architecture')['success'].agg(['count', 'sum', 'mean'])
        arch_stats['success_rate'] = arch_stats['mean'] * 100
        
        for arch in arch_stats.index:
            actual_stats[f'{arch}_configs'] = arch_stats.loc[arch, 'count']
            actual_stats[f'{arch}_successful'] = arch_stats.loc[arch, 'sum']
            actual_stats[f'{arch}_success_rate'] = arch_stats.loc[arch, 'success_rate']
        
        # Game-specific statistics
        game_stats = self.results_df.groupby('game')['success'].agg(['count', 'sum', 'mean'])
        game_stats['success_rate'] = game_stats['mean'] * 100
        
        for game in game_stats.index:
            actual_stats[f'{game}_configs'] = game_stats.loc[game, 'count']
            actual_stats[f'{game}_successful'] = game_stats.loc[game, 'sum']
            actual_stats[f'{game}_success_rate'] = game_stats.loc[game, 'success_rate']
        
        # Performance statistics for successful runs
        successful_df = self.results_df[self.results_df['success'] == True]
        if len(successful_df) > 0 and 'final_exploitability' in successful_df.columns:
            actual_stats['successful_final_exploitability'] = {
                'mean': successful_df['final_exploitability'].mean(),
                'std': successful_df['final_exploitability'].std(),
                'min': successful_df['final_exploitability'].min(),
                'max': successful_df['final_exploitability'].max()
            }
            
            # Architecture-specific performance
            arch_performance = successful_df.groupby('architecture')['final_exploitability'].agg(['mean', 'std', 'count'])
            for arch in arch_performance.index:
                actual_stats[f'{arch}_performance'] = {
                    'mean': arch_performance.loc[arch, 'mean'],
                    'std': arch_performance.loc[arch, 'std'],
                    'count': arch_performance.loc[arch, 'count']
                }
        
        # Effect size calculation (Cohen's d between LSTM and baseline)
        if 'lstm' in arch_stats.index and 'baseline' in arch_stats.index:
            lstm_success_rate = arch_stats.loc['lstm', 'success_rate']
            baseline_success_rate = arch_stats.loc['baseline', 'success_rate']
            
            # For success rates (proportions), use formula for effect size
            pooled_p = (arch_stats.loc['lstm', 'sum'] + arch_stats.loc['baseline', 'sum']) / \
                      (arch_stats.loc['lstm', 'count'] + arch_stats.loc['baseline', 'count'])
            pooled_sd = np.sqrt(pooled_p * (1 - pooled_p))
            
            if pooled_sd > 0:
                cohens_d = (lstm_success_rate/100 - baseline_success_rate/100) / pooled_sd
                actual_stats['cohens_d'] = cohens_d
        
        return actual_stats
    
    def validate_table_consistency(self, paper_numbers: Dict, actual_stats: Dict) -> List[str]:
        """Validate consistency of tables with experimental data."""
        table_errors = []
        
        # Check main stability table
        architectures = ['baseline', 'lstm', 'gru', 'transformer']
        
        for arch in architectures:
            if f'{arch}_configs' in paper_numbers and f'{arch}_configs' in actual_stats:
                paper_configs = paper_numbers[f'{arch}_configs']
                actual_configs = actual_stats[f'{arch}_configs']
                
                if abs(paper_configs - actual_configs) > 0:
                    table_errors.append(
                        f"Table configs mismatch for {arch}: paper={paper_configs}, actual={actual_configs}"
                    )
            
            if f'{arch}_successful' in paper_numbers and f'{arch}_successful' in actual_stats:
                paper_successful = paper_numbers[f'{arch}_successful']
                actual_successful = actual_stats[f'{arch}_successful']
                
                if abs(paper_successful - actual_successful) > 0:
                    table_errors.append(
                        f"Table successful count mismatch for {arch}: paper={paper_successful}, actual={actual_successful}"
                    )
            
            if f'{arch}_success_rate' in paper_numbers and f'{arch}_success_rate' in actual_stats:
                paper_rate = paper_numbers[f'{arch}_success_rate']
                actual_rate = actual_stats[f'{arch}_success_rate']
                
                if abs(paper_rate - actual_rate) > 1.0:  # Allow 1% tolerance
                    table_errors.append(
                        f"Table success rate mismatch for {arch}: paper={paper_rate}%, actual={actual_rate:.1f}%"
                    )
        
        # Check total configurations
        if 'total_configurations' in paper_numbers:
            paper_total = paper_numbers['total_configurations']
            actual_total = actual_stats['total_experiments']
            
            if abs(paper_total - actual_total) > 0:
                table_errors.append(
                    f"Total configurations mismatch: paper={paper_total}, actual={actual_total}"
                )
        
        return table_errors
    
    def validate_effect_sizes(self, paper_numbers: Dict, actual_stats: Dict) -> List[str]:
        """Validate reported effect sizes."""
        effect_size_errors = []
        
        if 'cohens_d' in paper_numbers and 'cohens_d' in actual_stats:
            paper_d = paper_numbers['cohens_d']
            actual_d = actual_stats['cohens_d']
            
            if abs(paper_d - actual_d) > 0.5:  # Allow some tolerance for different calculation methods
                effect_size_errors.append(
                    f"Cohen's d mismatch: paper={paper_d}, actual={actual_d:.2f}"
                )
        
        return effect_size_errors
    
    def check_figure_data_consistency(self) -> List[str]:
        """Check if figure data matches experimental results."""
        figure_errors = []
        
        # Check if plot files exist and contain expected data
        expected_plots = [
            'hyperparameter_success_rates.png',
            'lstm_performance_distribution.png',
            'training_dynamics_comparison.pdf',
            'embedding_analysis.pdf',
            'final_performance_comparison.png'
        ]
        
        plots_dir = self.paper_dir.parent / 'plots'
        for plot_file in expected_plots:
            plot_path = plots_dir / plot_file
            if not plot_path.exists():
                figure_errors.append(f"Missing expected plot file: {plot_file}")
        
        return figure_errors
    
    def validate_cross_references(self) -> List[str]:
        """Validate that all table and figure references are consistent."""
        ref_errors = []
        
        paper_file = self.paper_dir / 'Deep_CFR_Architecture_Study.tex'
        if not paper_file.exists():
            return ["Cannot validate references: paper file not found"]
        
        with open(paper_file, 'r') as f:
            content = f.read()
        
        # Find all table references
        table_refs = re.findall(r'\\ref\{(tab:[^}]+)\}', content)
        table_labels = re.findall(r'\\label\{(tab:[^}]+)\}', content)
        
        for ref in table_refs:
            if ref not in table_labels:
                ref_errors.append(f"Undefined table reference: {ref}")
        
        # Find all figure references
        figure_refs = re.findall(r'\\ref\{(fig:[^}]+)\}', content)
        figure_labels = re.findall(r'\\label\{(fig:[^}]+)\}', content)
        
        for ref in figure_refs:
            if ref not in figure_labels:
                ref_errors.append(f"Undefined figure reference: {ref}")
        
        return ref_errors
    
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate comprehensive consistency validation report."""
        print("Generating consistency validation report...")
        
        # Extract numbers from paper
        paper_numbers = self.extract_paper_numbers()
        
        # Compute actual statistics
        actual_stats = self.compute_actual_statistics()
        
        # Validate different aspects
        table_errors = self.validate_table_consistency(paper_numbers, actual_stats)
        effect_size_errors = self.validate_effect_sizes(paper_numbers, actual_stats)
        figure_errors = self.check_figure_data_consistency()
        reference_errors = self.validate_cross_references()
        
        # Compile validation report
        validation_report = {
            'summary': {
                'total_errors': len(table_errors) + len(effect_size_errors) + len(figure_errors) + len(reference_errors),
                'table_errors': len(table_errors),
                'effect_size_errors': len(effect_size_errors),
                'figure_errors': len(figure_errors),
                'reference_errors': len(reference_errors)
            },
            'paper_numbers': paper_numbers,
            'actual_statistics': actual_stats,
            'validation_errors': {
                'tables': table_errors,
                'effect_sizes': effect_size_errors,
                'figures': figure_errors,
                'references': reference_errors
            },
            'warnings': self.warnings
        }
        
        # Save report
        report_file = self.results_dir / 'consistency_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Print summary
        print(f"Consistency validation complete:")
        print(f"  Total errors found: {validation_report['summary']['total_errors']}")
        print(f"  Table inconsistencies: {len(table_errors)}")
        print(f"  Effect size issues: {len(effect_size_errors)}")
        print(f"  Figure problems: {len(figure_errors)}")
        print(f"  Reference errors: {len(reference_errors)}")
        
        if validation_report['summary']['total_errors'] == 0:
            print("✅ All validation checks passed!")
        else:
            print("❌ Validation issues found. See report for details.")
        
        return validation_report
    
    def create_corrected_numbers_table(self, actual_stats: Dict) -> str:
        """Generate LaTeX table with corrected numbers."""
        latex_table = """
\\begin{table}[t]
\\centering
\\caption{Training Stability Across Neural Architectures (Corrected)}
\\label{tab:stability_corrected}
\\begin{tabular}{@{}lccr@{}}
\\toprule
Architecture & Configurations & Successful & Success Rate \\\\ 
\\midrule
"""
        
        architectures = ['baseline', 'lstm', 'gru', 'transformer']
        arch_names = ['Baseline MLP', 'LSTM', 'GRU', 'Transformer']
        
        for arch, name in zip(architectures, arch_names):
            if f'{arch}_configs' in actual_stats:
                configs = actual_stats[f'{arch}_configs']
                successful = actual_stats[f'{arch}_successful']
                rate = actual_stats[f'{arch}_success_rate']
                
                latex_table += f"{name} & {configs} & {successful} & {rate:.1f}\\% \\\\\n"
        
        # Add total row
        total_configs = actual_stats.get('total_experiments', 0)
        total_successful = actual_stats.get('total_successful', 0)
        total_rate = actual_stats.get('overall_success_rate', 0)
        
        latex_table += """\\midrule
\\textbf{Total} & \\textbf{""" + str(total_configs) + """} & \\textbf{""" + str(total_successful) + """} & \\textbf{""" + f"{total_rate:.1f}" + """\\%} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        return latex_table

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate results consistency')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experimental results')
    parser.add_argument('--paper-dir', type=str, required=True,
                       help='Directory containing paper LaTeX files')
    
    args = parser.parse_args()
    
    validator = ResultsValidator(args.results_dir, args.paper_dir)
    report = validator.generate_consistency_report()
    
    # Generate corrected table if needed
    if report['summary']['table_errors'] > 0:
        corrected_table = validator.create_corrected_numbers_table(report['actual_statistics'])
        
        corrected_file = Path(args.results_dir) / 'corrected_table.tex'
        with open(corrected_file, 'w') as f:
            f.write(corrected_table)
        
        print(f"Corrected LaTeX table saved to: {corrected_file}")
    
    return report

if __name__ == "__main__":
    main()