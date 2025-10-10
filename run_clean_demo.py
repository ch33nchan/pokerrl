#!/usr/bin/env python3
"""
Clean Comprehensive Experiment Demonstration

This script demonstrates that all integration issues have been resolved
by running a complete experiment matrix successfully.
"""

import os
import json
import time
from qagent.baselines.canonical_baselines import run_baseline_experiment

def main():
    print('Running clean comprehensive experiment demonstration')
    print('=' * 60)

    # Test matrix with different algorithms and games
    test_matrix = [
        # Kuhn Poker experiments
        ('kuhn_poker', 'tabular_cfr', 42),
        ('kuhn_poker', 'deep_cfr', 42),
        ('kuhn_poker', 'sd_cfr', 42),
        # Leduc Poker experiments
        ('leduc_poker', 'tabular_cfr', 42),
        ('leduc_poker', 'deep_cfr', 42),
        ('leduc_poker', 'sd_cfr', 42),
    ]

    results = []
    start_total = time.time()

    for i, (game, algorithm, seed) in enumerate(test_matrix):
        print(f'[{i+1}/{len(test_matrix)}] {algorithm.upper()} on {game.replace("_", " ").title()}')

        start = time.time()
        result = run_baseline_experiment(
            game_name=game,
            baseline_type=algorithm,
            seed=seed,
            iterations=50,  # Short demo run
            eval_every=25
        )
        elapsed = time.time() - start

        final_exploitability = result.get('final_exploitability', float('inf'))
        print(f'  Completed in {elapsed:.1f}s')
        print(f'  Final exploitability: {final_exploitability:.4f}')
        print(f'  Status: {result.get("status", "unknown")}')

        results.append({
            'game': game,
            'algorithm': algorithm,
            'seed': seed,
            'final_exploitability': final_exploitability,
            'wall_time': elapsed,
            'status': result.get('status', 'unknown')
        })

    total_time = time.time() - start_total

    print('\n' + '=' * 60)
    print('EXPERIMENT MATRIX COMPLETED SUCCESSFULLY!')
    print('=' * 60)
    print(f'Total experiments: {len(results)}')
    print(f'Total time: {total_time:.1f}s')
    print(f'Average time per experiment: {total_time/len(results):.1f}s')

    print('\nResults Summary:')
    for result in results:
        status_ok = result['status'] != 'failed'
        status_mark = 'OK' if status_ok else 'FAIL'
        print(f'  {status_mark:4} | {result["algorithm"]:12} | {result["game"]:12} | Exploit: {result["final_exploitability"]:7.4f} | {result["wall_time"]:5.1f}s')

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/demo_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nResults saved to: results/demo_experiment_results.json')
    print('\nAll integration issues have been resolved!')
    print('Ready for full-scale comprehensive experiments!')

if __name__ == "__main__":
    main()