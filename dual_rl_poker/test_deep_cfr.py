#!/usr/bin/env python3
"""Simple test script for Deep CFR implementation."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_deep_cfr_kuhn():
    """Test Deep CFR on Kuhn Poker."""
    print("Testing Deep CFR on Kuhn Poker...")

    try:
        # Import required modules
        from games.kuhn_poker import KuhnPokerWrapper
        from algs.deep_cfr import DeepCFRAlgorithm
        from utils.config import get_algorithm_config, load_config

        # Load configuration
        base_config = load_config("configs/default.yaml")
        config = get_algorithm_config(base_config, "deep_cfr")

        # Adjust for quick test
        config['training']['iterations'] = 10
        config['training']['eval_every'] = 5
        config['training']['batch_size'] = 32
        config['advantage_memory_size'] = 100
        config['strategy_memory_size'] = 100

        print(f"Configuration loaded successfully")
        print(f"Training iterations: {config['training']['iterations']}")
        print(f"Batch size: {config['training']['batch_size']}")

        # Initialize game and algorithm
        game_wrapper = KuhnPokerWrapper()
        algorithm = DeepCFRAlgorithm(game_wrapper, config)

        print(f"Game: {game_wrapper.game_name}")
        print(f"Network parameters: {sum(p.numel() for p in algorithm.regret_network.parameters())} (regret)")
        print(f"Network parameters: {sum(p.numel() for p in algorithm.strategy_network.parameters())} (strategy)")

        # Run a few training iterations
        print("\nStarting training...")
        for i in range(config['training']['iterations']):
            training_state = algorithm.train_iteration()
            print(f"Iteration {i+1:2d}: Loss = {training_state.loss:.4f}, "
                  f"Buffer = {training_state.buffer_size:4d}, "
                  f"Grad Norm = {training_state.gradient_norm:.4f}")

            if (i + 1) % config['training']['eval_every'] == 0:
                eval_metrics = algorithm.evaluate()
                print(f"  Evaluation: Exploitability = {eval_metrics['exploitability']:.4f}")

        # Test policy retrieval
        print("\nTesting policy retrieval...")
        policy_fn = algorithm.get_policy(0)
        test_info_state = "0"  # Simple test info state
        test_actions = [0, 1]  # Check/Call, Bet/Fold

        try:
            action_probs = policy_fn(test_info_state, test_actions)
            print(f"Policy for info state '{test_info_state}': {action_probs}")
            print(f"Sum: {action_probs.sum():.4f}")
        except Exception as e:
            print(f"Policy retrieval failed: {e}")

        # Test strategy and regret retrieval
        print("\nTesting internal state retrieval...")
        avg_strategy = algorithm.get_average_strategy()
        regrets = algorithm.get_regrets()

        print(f"Number of info states in average strategy: {len(avg_strategy)}")
        print(f"Number of info states in regrets: {len(regrets)}")

        if len(avg_strategy) > 0:
            sample_key = list(avg_strategy.keys())[0]
            print(f"Sample strategy for '{sample_key}': {avg_strategy[sample_key]}")

        if len(regrets) > 0:
            sample_key = list(regrets.keys())[0]
            print(f"Sample regrets for '{sample_key}': {regrets[sample_key]}")

        print("\n✅ Deep CFR test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Deep CFR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_components():
    """Test individual components."""
    print("Testing individual components...")

    # Test game wrapper
    try:
        from games.kuhn_poker import KuhnPokerWrapper
        game = KuhnPokerWrapper()
        state = game.get_initial_state()
        encoding = game.encode_state(state)
        print(f"✅ Game wrapper: encoding shape {encoding.shape}")
    except Exception as e:
        print(f"❌ Game wrapper failed: {e}")
        return False

    # Test neural networks
    try:
        from nets.mlp import DeepCFRNetwork
        network = DeepCFRNetwork(encoding.shape[0], game.num_actions, [32, 32])
        with torch.no_grad():
            output = network(torch.tensor(encoding).unsqueeze(0))
        print(f"✅ Neural network: output keys {list(output.keys())}")
    except Exception as e:
        print(f"❌ Neural network failed: {e}")
        return False

    print("✅ All components working!")
    return True


def main():
    """Main test function."""
    print("Dual RL Poker - Deep CFR Test")
    print("=" * 50)

    # Import torch locally to avoid dependency issues
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available. Install with: pip install torch")
        return

    # Test components first
    if not test_components():
        return

    # Test Deep CFR algorithm
    if test_deep_cfr_kuhn():
        print("\n🎉 All tests passed! Deep CFR implementation is working.")
    else:
        print("\n💥 Deep CFR test failed. Please check the implementation.")


if __name__ == "__main__":
    main()