#!/usr/bin/env python3
"""
Quick test script to verify fixed imports and run a minimal experiment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from typing import Dict, Any

# Test all imports
def test_imports():
    """Test that all algorithm imports work correctly."""
    print("Testing imports...")

    try:
        from algs.deep_cfr import DeepCFRAlgorithm
        print("✓ Deep CFR import successful")
    except Exception as e:
        print(f"✗ Deep CFR import failed: {e}")
        return False

    try:
        from algs.sd_cfr import SDCFRAlgorithm
        print("✓ SD-CFR import successful")
    except Exception as e:
        print(f"✗ SD-CFR import failed: {e}")
        return False

    try:
        from algs.armac import ARMACAlgorithm
        print("✓ ARMAC import successful")
    except Exception as e:
        print(f"✗ ARMAC import failed: {e}")
        return False

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        print("✓ Kuhn Poker import successful")
    except Exception as e:
        print(f"✗ Kuhn Poker import failed: {e}")
        return False

    try:
        from nets.mlp import DeepCFRNetwork, ARMACNetwork
        print("✓ Network imports successful")
    except Exception as e:
        print(f"✗ Network import failed: {e}")
        return False

    try:
        from eval.evaluator import OpenSpielEvaluator
        print("✓ Evaluator import successful")
    except Exception as e:
        print(f"✗ Evaluator import failed: {e}")
        return False

    return True

def create_minimal_config() -> Dict[str, Any]:
    """Create minimal configuration for testing."""
    return {
        'device': 'cpu',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'iterations': 5,  # Very short test
        'network': {
            'hidden_dims': [32, 32]
        },
        'training': {
            'gradient_clip': 5.0,
            'weight_decay': 0.0
        }
    }

def test_deep_cfr():
    """Test Deep CFR algorithm with minimal setup."""
    print("\nTesting Deep CFR...")

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.deep_cfr import DeepCFRAlgorithm

        # Create game
        game = KuhnPokerWrapper()

        # Create algorithm
        config = create_minimal_config()
        algorithm = DeepCFRAlgorithm(game, config)

        print(f"  - Initialized with {sum(p.numel() for p in algorithm.regret_network.parameters())} regret parameters")
        print(f"  - Initialized with {sum(p.numel() for p in algorithm.strategy_network.parameters())} strategy parameters")

        # Run a few iterations
        for i in range(3):
            start_time = time.time()
            state = algorithm.train_iteration()
            iter_time = time.time() - start_time
            print(f"  - Iteration {i+1}: loss={state.loss:.4f}, time={iter_time:.2f}s")

        print("✓ Deep CFR test successful")
        return True

    except Exception as e:
        print(f"✗ Deep CFR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("DUAL RL POKER IMPORT TEST")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False

    print("\n✅ All imports successful!")

    # Test Deep CFR
    if not test_deep_cfr():
        print("\n❌ Deep CFR test failed!")
        return False

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("Ready to run actual experiments!")
    print("=" * 50)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)