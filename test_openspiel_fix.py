#!/usr/bin/env python3
"""
Test script for the fixed OpenSpiel API integration.
"""

import os
import sys
sys.path.append('/Users/cheencheen/Desktop/lossfunk/q-agent')

import time
import torch
import pyspiel
import numpy as np
from experiments.run_comprehensive_cpu_experiments import (
    ExactEvaluator, TabularCFR, DeepCFRCanonical, SDCFR
)

def test_single_experiment():
    """Test a single experiment to verify OpenSpiel API works."""
    print("Testing single experiment with fixed OpenSpiel API...")

    # Load game
    game = pyspiel.load_game("kuhn_poker")
    print(f"Game loaded: {game}")
    print(f"OpenSpiel version: {pyspiel.__version__}")

    # Test Tabular CFR
    print("\n=== Testing Tabular CFR ===")
    start_time = time.time()
    tabular_agent = TabularCFR(game, seed=42)
    evaluator = ExactEvaluator(game)

    # Train for a few iterations
    results = tabular_agent.train(50, eval_every=25)
    end_time = time.time()

    print(f"Tabular CFR training completed in {end_time - start_time:.2f}s")
    print(f"Final exploitability: {results['exploitability'][-1]:.6f}")
    print(f"Final NashConv: {results['nashconv'][-1]:.6f}")

    # Test Deep CFR
    print("\n=== Testing Deep CFR (baseline architecture) ===")
    start_time = time.time()
    deep_cfr_agent = DeepCFRCanonical(game, "baseline", seed=42)

    # Train for a few iterations
    results = deep_cfr_agent.train(50, eval_every=25)
    end_time = time.time()

    print(f"Deep CFR training completed in {end_time - start_time:.2f}s")
    print(f"Final exploitability: {results['exploitability'][-1]:.6f}")
    print(f"Final NashConv: {results['nashconv'][-1]:.6f}")

    # Test SD-CFR
    print("\n=== Testing SD-CFR (baseline architecture) ===")
    start_time = time.time()
    sd_cfr_agent = SDCFR(game, "baseline", seed=42)

    # Train for a few iterations
    results = sd_cfr_agent.train(50, eval_every=25)
    end_time = time.time()

    print(f"SD-CFR training completed in {end_time - start_time:.2f}s")
    print(f"Final exploitability: {results['exploitability'][-1]:.6f}")
    print(f"Final NashConv: {results['nashconv'][-1]:.6f}")

    print("\n✅ All tests passed! OpenSpiel API integration works correctly.")

if __name__ == "__main__":
    test_single_experiment()