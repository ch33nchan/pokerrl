import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime

from qagent.agents.sum_agent import SUMAgent
from qagent.agents.baseline_agents import BaselineAgentFactory
from qagent.environments.poker_environment import PokerEnvironment, PokerGameLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    hands_per_match: int = 1_000_000
    num_trials: int = 3
    confidence_level: float = 0.95
    save_detailed_logs: bool = True
    generate_plots: bool = True
    results_dir: str = "benchmark_results"
    log_dir: str = "benchmark_logs"
    plot_dir: str = "benchmark_plots"
    parallel_execution: bool = True
    max_workers: int = 4

class StatisticalAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def calculate_mbb_statistics(self, winnings_list: List[int], hands_played: int, big_blind: int = 2) -> Dict[str, float]:
        if not winnings_list or hands_played == 0:
            return {
                'mean_mbb_per_100': 0.0,
                'std_mbb_per_100': 0.0,
                'confidence_interval_lower': 0.0,
                'confidence_interval_upper': 0.0,
                'sample_size': 0
            }
        
        mbb_scores = [(winnings / (hands_played / 100)) / big_blind * 1000 for winnings in winnings_list]
        
        mean_mbb = statistics.mean(mbb_scores)
        std_mbb = statistics.stdev(mbb_scores) if len(mbb_scores) > 1 else 0.0
        
        if len(mbb_scores) > 1:
            confidence_interval = stats.t.interval(
                self.confidence_level,
                len(mbb_scores) - 1,
                loc=mean_mbb,
                scale=stats.sem(mbb_scores)
            )
            ci_lower, ci_upper = confidence_interval
        else:
            ci_lower = ci_upper = mean_mbb
        
        return {
            'mean_mbb_per_100': mean_mbb,
            'std_mbb_per_100': std_mbb,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'sample_size': len(mbb_scores)
        }
    
    def perform_significance_test(self, sum_winnings: List[int], baseline_winnings: List[int], hands_played: int) -> Dict[str, Any]:
        if len(sum_winnings) < 2 or len(baseline_winnings) < 2:
            return {
                'test_type': 'insufficient_data',
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0
            }
        
        big_blind = 2
        sum_mbb = [(w / (hands_played / 100)) / big_blind * 1000 for w in sum_winnings]
        baseline_mbb = [(w / (hands_played / 100)) / big_blind * 1000 for w in baseline_winnings]
        
        try:
            t_stat, p_value = stats.ttest_ind(sum_mbb, baseline_mbb)
            
            pooled_std = np.sqrt(((len(sum_mbb) - 1) * np.var(sum_mbb, ddof=1) + 
                                 (len(baseline_mbb) - 1) * np.var(baseline_mbb, ddof=1)) / 
                                (len(sum_mbb) + len(baseline_mbb) - 2))
            
            effect_size = (np.mean(sum_mbb) - np.mean(baseline_mbb)) / pooled_std if pooled_std > 0 else 0.0
            
            return {
                'test_type': 'welch_t_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < (1 - self.confidence_level),
                'effect_size': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return {
                'test_type': 'failed',
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0,
                'error': str(e)
            }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

class BenchmarkMatch:
    def __init__(self, sum_agent: SUMAgent, baseline_agent, config: BenchmarkConfig):
        self.sum_agent = sum_agent
        self.baseline_agent = baseline_agent
        self.config = config
        
        self.poker_env = PokerEnvironment()
        
        if config.save_detailed_logs:
            log_file = os.path.join(
                config.log_dir,
                f"match_{sum_agent.name}_vs_{baseline_agent.name}_{int(time.time())}.json"
            )
            self.game_logger = PokerGameLogger(log_file)
        else:
            self.game_logger = None
        
        logger.info(f"BenchmarkMatch initialized: {sum_agent.name} vs {baseline_agent.name}")
    
    def run_match(self) -> Dict[str, Any]:
        logger.info(f"Starting benchmark match: {self.sum_agent.name} vs {self.baseline_agent.name}")
        
        match_start_time = time.time()
        
        trial_results = []
        
        # Progress bar for trials
        trial_desc = f"{self.sum_agent.name} vs {self.baseline_agent.name}"
        with tqdm(range(self.config.num_trials), desc=trial_desc, unit="trial") as pbar:
            for trial in pbar:
                trial_start = time.time()
                trial_result = self._run_single_trial(trial)
                trial_results.append(trial_result)
                
                # Update progress bar with ETA
                elapsed = time.time() - match_start_time
                avg_time_per_trial = elapsed / (trial + 1)
                remaining_trials = self.config.num_trials - (trial + 1)
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                pbar.set_postfix({
                    'ETA': eta_str,
                    'Avg/trial': f"{avg_time_per_trial:.1f}s"
                })
        
        match_end_time = time.time()
        
        aggregated_results = self._aggregate_trial_results(trial_results)
        
        match_summary = {
            'sum_agent_name': self.sum_agent.name,
            'baseline_agent_name': self.baseline_agent.name,
            'config': asdict(self.config),
            'total_duration': match_end_time - match_start_time,
            'trial_results': trial_results,
            'aggregated_results': aggregated_results,
            'timestamp': int(time.time())
        }
        
        return match_summary
    
    def _run_single_trial(self, trial_id: int) -> Dict[str, Any]:
        trial_start = time.time()
        
        self.sum_agent.set_training_mode(False)
        
        logger.info(f"Starting trial {trial_id} with {self.config.hands_per_match} hands")
        
        tournament_result = self.poker_env.run_tournament(
            self.sum_agent,
            self.baseline_agent,
            num_games=self.config.hands_per_match
        )
        
        logger.info(f"Trial {trial_id} raw tournament result: {tournament_result}")
        
        trial_end = time.time()
        
        if self.game_logger:
            self.game_logger.log_game_state(
                {'trial_id': trial_id, 'tournament_result': tournament_result},
                0,
                {'action': 'trial_complete'},
                trial_end
            )
        
        trial_summary = {
            'trial_id': trial_id,
            'duration': trial_end - trial_start,
            'tournament_result': tournament_result,
            'performance_metrics': self._extract_performance_metrics(tournament_result)
        }
        
        return trial_summary
    
    def _extract_performance_metrics(self, tournament_result: Dict) -> Dict[str, Any]:
        # Add detailed logging for debugging
        logger.info(f"Tournament result keys: {list(tournament_result.keys())}")
        logger.info(f"Tournament success: {tournament_result.get('success', False)}")
        logger.info(f"Tournament total_hands: {tournament_result.get('total_hands', 'NOT_FOUND')}")
        
        if not tournament_result.get('success', False):
            logger.warning("Tournament was not successful, returning zero metrics")
            return {
                'sum_winnings': 0,
                'baseline_winnings': 0,
                'sum_mbb_per_100': 0.0,
                'baseline_mbb_per_100': 0.0,
                'hands_per_second': 0.0,
                'total_hands': 0
            }
        
        player_results = tournament_result.get('player_results', {})
        logger.info(f"Player results: {player_results}")
        
        sum_result = player_results.get(self.sum_agent.name, {})
        baseline_result = None
        
        for player_name, result in player_results.items():
            if player_name != self.sum_agent.name:
                baseline_result = result
                break
        
        if baseline_result is None:
            baseline_result = {}
        
        sum_winnings = sum_result.get('winnings', 0)
        baseline_winnings = baseline_result.get('winnings', 0)
        
        total_hands = tournament_result.get('total_hands', self.config.hands_per_match)
        
        # Fix division by zero error
        if total_hands == 0:
            logger.error(f"Total hands is 0! Using fallback value: {self.config.hands_per_match}")
            total_hands = self.config.hands_per_match
        
        big_blind = 2
        hands_per_100 = total_hands / 100
        
        # Prevent division by zero
        if hands_per_100 == 0:
            logger.error("Hands per 100 is 0, cannot calculate mBB/100")
            sum_mbb = 0.0
            baseline_mbb = 0.0
        else:
            sum_mbb = (sum_winnings / hands_per_100) / big_blind * 1000
            baseline_mbb = (baseline_winnings / hands_per_100) / big_blind * 1000
        
        logger.info(f"Calculated metrics - Sum mBB/100: {sum_mbb}, Baseline mBB/100: {baseline_mbb}")
        
        return {
            'sum_winnings': sum_winnings,
            'baseline_winnings': baseline_winnings,
            'sum_mbb_per_100': sum_mbb,
            'baseline_mbb_per_100': baseline_mbb,
            'hands_per_second': tournament_result.get('hands_per_second', 0.0),
            'total_hands': total_hands
        }
    
    def _aggregate_trial_results(self, trial_results: List[Dict]) -> Dict[str, Any]:
        sum_winnings = [trial['performance_metrics']['sum_winnings'] for trial in trial_results]
        baseline_winnings = [trial['performance_metrics']['baseline_winnings'] for trial in trial_results]
        sum_mbb_scores = [trial['performance_metrics']['sum_mbb_per_100'] for trial in trial_results]
        baseline_mbb_scores = [trial['performance_metrics']['baseline_mbb_per_100'] for trial in trial_results]
        hands_per_second = [trial['performance_metrics']['hands_per_second'] for trial in trial_results]
        
        total_hands = trial_results[0]['performance_metrics']['total_hands'] if trial_results else 0
        
        analyzer = StatisticalAnalyzer(self.config.confidence_level)
        
        sum_statistics = analyzer.calculate_mbb_statistics(sum_winnings, total_hands)
        baseline_statistics = analyzer.calculate_mbb_statistics(baseline_winnings, total_hands)
        
        significance_test = analyzer.perform_significance_test(sum_winnings, baseline_winnings, total_hands)
        
        return {
            'sum_agent_statistics': sum_statistics,
            'baseline_agent_statistics': baseline_statistics,
            'significance_test': significance_test,
            'performance_summary': {
                'sum_mean_mbb': statistics.mean(sum_mbb_scores),
                'baseline_mean_mbb': statistics.mean(baseline_mbb_scores),
                'mbb_difference': statistics.mean(sum_mbb_scores) - statistics.mean(baseline_mbb_scores),
                'sum_win_rate': sum(1 for w in sum_winnings if w > 0) / len(sum_winnings),
                'average_hands_per_second': statistics.mean(hands_per_second)
            }
        }

class BenchmarkGauntlet:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        if config.generate_plots:
            os.makedirs(config.plot_dir, exist_ok=True)
        
        self.baseline_agents = BaselineAgentFactory.create_benchmark_suite()
        
        logger.info(f"BenchmarkGauntlet initialized with {len(self.baseline_agents)} baseline agents")
    
    def run_gauntlet(self, sum_agent: SUMAgent) -> Dict[str, Any]:
        logger.info(f"Starting benchmark gauntlet for {sum_agent.name}")
        
        gauntlet_start_time = time.time()
        
        match_results = []
        
        if self.config.parallel_execution:
            match_results = self._run_parallel_matches(sum_agent)
        else:
            match_results = self._run_sequential_matches(sum_agent)
        
        gauntlet_end_time = time.time()
        
        gauntlet_summary = self._create_gauntlet_summary(
            sum_agent, match_results, gauntlet_start_time, gauntlet_end_time
        )
        
        self._save_gauntlet_results(gauntlet_summary)
        
        if self.config.generate_plots:
            self._generate_visualization_plots(gauntlet_summary)
        
        return gauntlet_summary
    
    def _run_sequential_matches(self, sum_agent: SUMAgent) -> List[Dict]:
        match_results = []
        
        for i, baseline_agent in enumerate(self.baseline_agents):
            logger.info(f"Running match {i + 1}/{len(self.baseline_agents)}: vs {baseline_agent.name}")
            
            match = BenchmarkMatch(sum_agent, baseline_agent, self.config)
            result = match.run_match()
            match_results.append(result)
        
        return match_results
    
    def _run_parallel_matches(self, sum_agent: SUMAgent) -> List[Dict]:
        match_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for baseline_agent in self.baseline_agents:
                match = BenchmarkMatch(sum_agent, baseline_agent, self.config)
                future = executor.submit(match.run_match)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=3600)
                    match_results.append(result)
                except Exception as e:
                    logger.error(f"Match execution failed: {e}")
                    match_results.append(self._create_error_match_result(str(e)))
        
        return match_results
    
    def _create_error_match_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'sum_agent_name': 'Unknown',
            'baseline_agent_name': 'Unknown',
            'error': error_message,
            'aggregated_results': {
                'sum_agent_statistics': {'mean_mbb_per_100': 0.0},
                'performance_summary': {'mbb_difference': 0.0}
            }
        }
    
    def _create_gauntlet_summary(self, sum_agent: SUMAgent, match_results: List[Dict], start_time: float, end_time: float) -> Dict[str, Any]:
        total_duration = end_time - start_time
        
        successful_matches = [result for result in match_results if 'error' not in result]
        
        overall_performance = self._calculate_overall_performance(successful_matches)
        
        gauntlet_summary = {
            'sum_agent_name': sum_agent.name,
            'gauntlet_config': asdict(self.config),
            'total_duration': total_duration,
            'num_baseline_agents': len(self.baseline_agents),
            'successful_matches': len(successful_matches),
            'failed_matches': len(match_results) - len(successful_matches),
            'match_results': match_results,
            'overall_performance': overall_performance,
            'timestamp': int(time.time())
        }
        
        return gauntlet_summary
    
    def _calculate_overall_performance(self, match_results: List[Dict]) -> Dict[str, Any]:
        if not match_results:
            return {
                'average_mbb_per_100': 0.0,
                'win_rate_against_baselines': 0.0,
                'significant_wins': 0,
                'significant_losses': 0,
                'performance_ranking': []
            }
        
        mbb_scores = []
        significant_wins = 0
        significant_losses = 0
        performance_ranking = []
        
        for result in match_results:
            aggregated = result.get('aggregated_results', {})
            sum_stats = aggregated.get('sum_agent_statistics', {})
            performance = aggregated.get('performance_summary', {})
            significance = aggregated.get('significance_test', {})
            
            mbb_score = sum_stats.get('mean_mbb_per_100', 0.0)
            mbb_scores.append(mbb_score)
            
            if significance.get('significant', False):
                if mbb_score > 0:
                    significant_wins += 1
                else:
                    significant_losses += 1
            
            performance_ranking.append({
                'baseline_agent': result.get('baseline_agent_name', 'Unknown'),
                'mbb_per_100': mbb_score,
                'mbb_difference': performance.get('mbb_difference', 0.0),
                'significant': significance.get('significant', False),
                'p_value': significance.get('p_value', 1.0)
            })
        
        performance_ranking.sort(key=lambda x: x['mbb_per_100'], reverse=True)
        
        return {
            'average_mbb_per_100': statistics.mean(mbb_scores) if mbb_scores else 0.0,
            'win_rate_against_baselines': sum(1 for score in mbb_scores if score > 0) / len(mbb_scores) if mbb_scores else 0.0,
            'significant_wins': significant_wins,
            'significant_losses': significant_losses,
            'performance_ranking': performance_ranking,
            'best_matchup': performance_ranking[0] if performance_ranking else None,
            'worst_matchup': performance_ranking[-1] if performance_ranking else None
        }
    
    def _save_gauntlet_results(self, gauntlet_summary: Dict[str, Any]):
        timestamp = gauntlet_summary['timestamp']
        agent_name = gauntlet_summary['sum_agent_name']
        
        filename = f"gauntlet_results_{agent_name}_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(gauntlet_summary, f, indent=2, default=str)
        
        logger.info(f"Gauntlet results saved: {filepath}")
    
    def _generate_visualization_plots(self, gauntlet_summary: Dict[str, Any]):
        try:
            self._create_performance_bar_chart(gauntlet_summary)
            self._create_confidence_interval_plot(gauntlet_summary)
            self._create_significance_heatmap(gauntlet_summary)
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
    
    def _create_performance_bar_chart(self, gauntlet_summary: Dict[str, Any]):
        performance_ranking = gauntlet_summary['overall_performance']['performance_ranking']
        
        if not performance_ranking:
            return
        
        baseline_names = [item['baseline_agent'] for item in performance_ranking]
        mbb_scores = [item['mbb_per_100'] for item in performance_ranking]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(baseline_names, mbb_scores)
        
        for i, (bar, item) in enumerate(zip(bars, performance_ranking)):
            if item['significant']:
                bar.set_color('green' if item['mbb_per_100'] > 0 else 'red')
            else:
                bar.set_color('gray')
        
        plt.title(f"SUM Agent Performance vs Baseline Agents\n{gauntlet_summary['sum_agent_name']}")
        plt.xlabel("Baseline Agents")
        plt.ylabel("mBB/100 hands")
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, f"performance_bar_chart_{gauntlet_summary['timestamp']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance bar chart saved: {plot_path}")
    
    def _create_confidence_interval_plot(self, gauntlet_summary: Dict[str, Any]):
        match_results = gauntlet_summary['match_results']
        
        baseline_names = []
        mean_scores = []
        ci_lowers = []
        ci_uppers = []
        
        for result in match_results:
            if 'error' in result:
                continue
            
            baseline_names.append(result['baseline_agent_name'])
            
            sum_stats = result['aggregated_results']['sum_agent_statistics']
            mean_scores.append(sum_stats['mean_mbb_per_100'])
            ci_lowers.append(sum_stats['confidence_interval_lower'])
            ci_uppers.append(sum_stats['confidence_interval_upper'])
        
        if not baseline_names:
            return
        
        plt.figure(figsize=(12, 8))
        
        x_pos = np.arange(len(baseline_names))
        
        plt.errorbar(x_pos, mean_scores, 
                    yerr=[np.array(mean_scores) - np.array(ci_lowers), 
                          np.array(ci_uppers) - np.array(mean_scores)],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        plt.xticks(x_pos, baseline_names, rotation=45)
        plt.title(f"SUM Agent Performance with Confidence Intervals\n{gauntlet_summary['sum_agent_name']}")
        plt.xlabel("Baseline Agents")
        plt.ylabel("mBB/100 hands")
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, f"confidence_intervals_{gauntlet_summary['timestamp']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence interval plot saved: {plot_path}")
    
    def _create_significance_heatmap(self, gauntlet_summary: Dict[str, Any]):
        match_results = gauntlet_summary['match_results']
        
        baseline_names = []
        p_values = []
        effect_sizes = []
        
        for result in match_results:
            if 'error' in result:
                continue
            
            baseline_names.append(result['baseline_agent_name'])
            
            significance = result['aggregated_results']['significance_test']
            p_values.append(significance.get('p_value', 1.0))
            effect_sizes.append(significance.get('effect_size', 0.0))
        
        if not baseline_names:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        p_value_matrix = np.array(p_values).reshape(1, -1)
        effect_size_matrix = np.array(effect_sizes).reshape(1, -1)
        
        sns.heatmap(p_value_matrix, annot=True, fmt='.3f', 
                   xticklabels=baseline_names, yticklabels=['P-Value'],
                   cmap='RdYlBu_r', ax=ax1, cbar_kws={'label': 'P-Value'})
        ax1.set_title('Statistical Significance (P-Values)')
        
        sns.heatmap(effect_size_matrix, annot=True, fmt='.3f',
                   xticklabels=baseline_names, yticklabels=['Effect Size'],
                   cmap='RdBu_r', center=0, ax=ax2, cbar_kws={'label': 'Effect Size'})
        ax2.set_title('Effect Sizes')
        
        plt.suptitle(f"Statistical Analysis Heatmap\n{gauntlet_summary['sum_agent_name']}")
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, f"significance_heatmap_{gauntlet_summary['timestamp']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Significance heatmap saved: {plot_path}")

def run_comprehensive_benchmark(sum_agent: SUMAgent, config: BenchmarkConfig = None) -> Dict[str, Any]:
    if config is None:
        config = BenchmarkConfig()
    
    gauntlet = BenchmarkGauntlet(config)
    return gauntlet.run_gauntlet(sum_agent)

if __name__ == "__main__":
    config = BenchmarkConfig(
        hands_per_match=100_000,
        num_trials=3,
        generate_plots=True
    )
    
    sum_agent = SUMAgent(name="Test_SUM_Agent", device="cpu")
    
    sum_agent.self_play_training(num_episodes=100, save_frequency=50)
    
    results = run_comprehensive_benchmark(sum_agent, config)
    
    print("\n=== Benchmark Results Summary ===")
    overall_perf = results['overall_performance']
    print(f"Average mBB/100: {overall_perf['average_mbb_per_100']:.2f}")
    print(f"Win rate vs baselines: {overall_perf['win_rate_against_baselines']:.1%}")
    print(f"Significant wins: {overall_perf['significant_wins']}")
    print(f"Significant losses: {overall_perf['significant_losses']}")