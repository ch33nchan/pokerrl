"""
Test script to verify the fixed experiment setup works.
"""

import sys
import os
sys.path.append('/Users/cheencheen/Desktop/lossfunk/q-agent')

from qagent.baselines.canonical_baselines import run_baseline_experiment

def test_single_experiment():
    """Test a single tabular CFR experiment on Kuhn Poker."""
    print("Testing single tabular CFR experiment on Kuhn Poker...")

    result = run_baseline_experiment(
        game_name="kuhn_poker",
        baseline_type="tabular_cfr",
        seed=0,
        iterations=50,  # Short test
        eval_every=10,
        architecture="mlp"
    )

    print(f"Result status: {result.get('status', 'unknown')}")
    print(f"Final exploitability: {result.get('final_exploitability', 'N/A')}")
    print(f"Wall clock time: {result.get('wall_clock_s', 'N/A'):.2f}s")

    return result

if __name__ == "__main__":
    test_single_experiment()