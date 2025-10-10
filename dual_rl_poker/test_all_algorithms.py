#!/usr/bin/env python3
"""Test script for all implemented algorithms."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_deep_cfr():
    """Test Deep CFR algorithm."""
    print("Testing Deep CFR...")

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.deep_cfr import DeepCFRAlgorithm
        from utils.config import load_config

        # Load configuration
        config = load_config("configs/default.yaml")
        algorithm_config = config['algorithms']['deep_cfr']
        algorithm_config.update({
            'training': config['training'],
            'network': config['network'],
            'optimizer': config['optimizer'],
            'experiment': config['experiment'],
            'game': config['game'],
            'logging': config['logging'],
            'reproducibility': config['reproducibility'],
            'evaluation': config['evaluation']
        })

        # Quick test configuration
        algorithm_config['training']['iterations'] = 5
        algorithm_config['training']['eval_every'] = 5
        algorithm_config['training']['batch_size'] = 16

        # Initialize game and algorithm
        game_wrapper = KuhnPokerWrapper()
        algorithm = DeepCFRAlgorithm(game_wrapper, algorithm_config)

        print(f"  Game: {game_wrapper.game_name}")
        print(f"  Regret network params: {sum(p.numel() for p in algorithm.regret_network.parameters())}")
        print(f"  Strategy network params: {sum(p.numel() for p in algorithm.strategy_network.parameters())}")

        # Run training iterations
        for i in range(5):
            training_state = algorithm.train_iteration()
            print(f"  Iteration {i+1}: Loss = {training_state.loss:.4f}")

        # Test policy retrieval
        policy_fn = algorithm.get_policy(0)
        test_actions = [0, 1]
        action_probs = policy_fn("0", test_actions)
        print(f"  Policy test: {action_probs}")

        print("  ✅ Deep CFR test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Deep CFR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sd_cfr():
    """Test SD-CFR algorithm."""
    print("Testing SD-CFR...")

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.sd_cfr import SDCFRAlgorithm
        from utils.config import load_config

        # Load configuration
        config = load_config("configs/default.yaml")
        algorithm_config = {
            'regret_buffer_size': 1000,
            'strategy_buffer_size': 1000,
            'regret_learning_rate': 1e-3,
            'strategy_learning_rate': 1e-3,
            'initial_epsilon': 0.5,
            'final_epsilon': 0.01,
            'epsilon_decay_steps': 100,
            'training': config['training'],
            'network': config['network'],
            'optimizer': config['optimizer'],
            'experiment': config['experiment'],
            'game': config['game'],
            'logging': config['logging'],
            'reproducibility': config['reproducibility'],
            'evaluation': config['evaluation']
        }

        # Quick test configuration
        algorithm_config['training']['iterations'] = 5
        algorithm_config['training']['eval_every'] = 5
        algorithm_config['training']['batch_size'] = 16

        # Initialize game and algorithm
        game_wrapper = KuhnPokerWrapper()
        algorithm = SDCFRAlgorithm(game_wrapper, algorithm_config)

        print(f"  Game: {game_wrapper.game_name}")
        print(f"  Regret network params: {sum(p.numel() for p in algorithm.regret_network.parameters())}")
        print(f"  Strategy network params: {sum(p.numel() for p in algorithm.strategy_network.parameters())}")

        # Run training iterations
        for i in range(5):
            training_state = algorithm.train_iteration()
            print(f"  Iteration {i+1}: Loss = {training_state.loss:.4f}")

        # Test policy retrieval
        policy_fn = algorithm.get_policy(0)
        test_actions = [0, 1]
        action_probs = policy_fn("0", test_actions)
        print(f"  Policy test: {action_probs}")

        print("  ✅ SD-CFR test passed!")
        return True

    except Exception as e:
        print(f"  ❌ SD-CFR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_armac():
    """Test ARMAC algorithm."""
    print("Testing ARMAC...")

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.armac import ARMACAlgorithm
        from utils.config import load_config

        # Load configuration
        config = load_config("configs/default.yaml")
        algorithm_config = {
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'regret_lr': 1e-3,
            'buffer_size': 1000,
            'batch_size': 16,
            'gamma': 0.99,
            'tau': 0.005,
            'regret_weight': 0.1,
            'initial_noise_scale': 0.5,
            'final_noise_scale': 0.01,
            'noise_decay_steps': 100,
            'training': config['training'],
            'network': config['network'],
            'optimizer': config['optimizer'],
            'experiment': config['experiment'],
            'game': config['game'],
            'logging': config['logging'],
            'reproducibility': config['reproducibility'],
            'evaluation': config['evaluation']
        }

        # Quick test configuration
        algorithm_config['training']['iterations'] = 5
        algorithm_config['training']['eval_every'] = 5

        # Initialize game and algorithm
        game_wrapper = KuhnPokerWrapper()
        algorithm = ARMACAlgorithm(game_wrapper, algorithm_config)

        print(f"  Game: {game_wrapper.game_name}")
        print(f"  Actor network params: {sum(p.numel() for p in algorithm.actor.parameters())}")
        print(f"  Critic network params: {sum(p.numel() for p in algorithm.critic.parameters())}")
        print(f"  Regret network params: {sum(p.numel() for p in algorithm.regret_network.parameters())}")

        # Run training iterations
        for i in range(5):
            training_state = algorithm.train_iteration()
            print(f"  Iteration {i+1}: Loss = {training_state.loss:.4f}")

        # Test policy retrieval
        policy_fn = algorithm.get_policy(0)
        test_actions = [0, 1]
        action_probs = policy_fn("0", test_actions)
        print(f"  Policy test: {action_probs}")

        print("  ✅ ARMAC test passed!")
        return True

    except Exception as e:
        print(f"  ❌ ARMAC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functionality."""
    print("Testing evaluation...")

    try:
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.deep_cfr import DeepCFRAlgorithm
        from eval.evaluator import OpenSpielEvaluator
        from utils.config import load_config

        # Load configuration
        config = load_config("configs/default.yaml")
        algorithm_config = config['algorithms']['deep_cfr']
        algorithm_config.update({
            'training': config['training'],
            'network': config['network'],
            'optimizer': config['optimizer'],
            'experiment': config['experiment'],
            'game': config['game'],
            'logging': config['logging'],
            'reproducibility': config['reproducibility'],
            'evaluation': config['evaluation']
        })

        # Initialize components
        game_wrapper = KuhnPokerWrapper()
        algorithm = DeepCFRAlgorithm(game_wrapper, algorithm_config)
        evaluator = OpenSpielEvaluator(game_wrapper)

        # Train for a few iterations
        for _ in range(3):
            algorithm.train_iteration()

        # Get policies
        policies = {0: algorithm.get_policy(0), 1: algorithm.get_policy(1)}

        # Run evaluation
        eval_metrics = evaluator.evaluate_with_diagnostics(policies, num_episodes=10)

        print(f"  Exploitability: {eval_metrics['exploitability']:.4f}")
        print(f"  NashConv: {eval_metrics['nash_conv']:.4f}")

        print("  ✅ Evaluation test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Dual RL Poker - Algorithm Test Suite")
    print("=" * 50)

    # Import torch locally to avoid dependency issues
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available. Install with: pip install torch")
        return

    # Run all tests
    tests = [
        test_deep_cfr,
        test_sd_cfr,
        test_armac,
        test_evaluation
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The dual RL poker implementation is working.")
    else:
        print("💥 Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()