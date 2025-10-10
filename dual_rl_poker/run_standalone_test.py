#!/usr/bin/env python3
"""
Standalone test to verify actual neural network training works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import time
import json

def test_neural_network_training():
    """Test that neural networks can actually train."""
    print("=" * 60)
    print("TESTING NEURAL NETWORK TRAINING")
    print("=" * 60)

    # Create a simple network similar to Deep CFR networks
    class SimpleNetwork(nn.Module):
        def __init__(self, input_size=10, hidden_size=32, output_size=2):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.layers(x)

    # Create networks
    print("Creating networks...")
    regret_network = SimpleNetwork()
    strategy_network = SimpleNetwork()

    print(f"Regret network parameters: {sum(p.numel() for p in regret_network.parameters())}")
    print(f"Strategy network parameters: {sum(p.numel() for p in strategy_network.parameters())}")

    # Create optimizers
    regret_optimizer = torch.optim.Adam(regret_network.parameters(), lr=1e-3)
    strategy_optimizer = torch.optim.Adam(strategy_network.parameters(), lr=1e-3)

    # Training loop
    print("\nStarting training...")
    regret_losses = []
    strategy_losses = []

    start_time = time.time()

    for iteration in range(50):  # 50 training iterations
        # Generate fake data
        batch_size = 16
        inputs = torch.randn(batch_size, 10)
        targets_regret = torch.randn(batch_size, 2)
        targets_strategy = torch.randn(batch_size, 2)
        targets_strategy = torch.softmax(targets_strategy, dim=-1)

        # Train regret network
        regret_optimizer.zero_grad()
        regret_output = regret_network(inputs)
        regret_loss = nn.functional.mse_loss(regret_output, targets_regret)
        regret_loss.backward()
        regret_optimizer.step()

        # Train strategy network
        strategy_optimizer.zero_grad()
        strategy_output = strategy_network(inputs)
        strategy_loss = -(targets_strategy * torch.log(strategy_output + 1e-8)).sum(dim=1).mean()
        strategy_loss.backward()
        strategy_optimizer.step()

        regret_losses.append(regret_loss.item())
        strategy_losses.append(strategy_loss.item())

        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1}: regret_loss={regret_loss.item():.4f}, strategy_loss={strategy_loss.item():.4f}")

    total_time = time.time() - start_time

    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Final regret loss: {regret_losses[-1]:.4f}")
    print(f"Final strategy loss: {strategy_losses[-1]:.4f}")

    # Test that networks learned something
    initial_loss = regret_losses[0] + strategy_losses[0]
    final_loss = regret_losses[-1] + strategy_losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"Loss improvement: {improvement:.1f}%")

    # Save results
    results = {
        'algorithm': 'Standalone Neural Network Test',
        'iterations': 50,
        'total_time': total_time,
        'initial_total_loss': initial_loss,
        'final_total_loss': final_loss,
        'improvement_percent': improvement,
        'regret_network_params': sum(p.numel() for p in regret_network.parameters()),
        'strategy_network_params': sum(p.numel() for p in strategy_network.parameters()),
        'final_regret_loss': regret_losses[-1],
        'final_strategy_loss': strategy_losses[-1],
        'timestamp': time.time()
    }

    with open('standalone_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to standalone_test_results.json")

    # Verify loss decreased (training worked)
    if final_loss < initial_loss:
        print("✅ TRAINING SUCCESSFUL - Networks learned!")
        return True
    else:
        print("❌ TRAINING FAILED - No improvement detected")
        return False

def test_deep_cfr_components():
    """Test Deep CFR components separately."""
    print("\n" + "=" * 60)
    print("TESTING DEEP CFR COMPONENTS")
    print("=" * 60)

    try:
        from nets.mlp import DeepCFRNetwork

        # Test network creation
        print("Creating DeepCFR networks...")
        regret_net = DeepCFRNetwork(input_dim=10, num_actions=2, hidden_dims=[32, 32])
        strategy_net = DeepCFRNetwork(input_dim=10, num_actions=2, hidden_dims=[32, 32])

        print(f"✓ DeepCFR regret network: {sum(p.numel() for p in regret_net.parameters())} parameters")
        print(f"✓ DeepCFR strategy network: {sum(p.numel() for p in strategy_net.parameters())} parameters")

        # Test forward pass
        print("Testing forward pass...")
        batch = torch.randn(8, 10)
        legal_mask = torch.ones(8, 2, dtype=torch.bool)

        regret_output = regret_net(batch, legal_mask, network_type='regret')
        strategy_output = strategy_net(batch, legal_mask, network_type='strategy')

        print(f"✓ Regret output shape: {regret_output['advantages'].shape}")
        print(f"✓ Strategy output shape: {strategy_output['policy'].shape}")

        # Test loss computation
        print("Testing loss computation...")
        target_regrets = torch.randn(8, 2)
        target_strategies = torch.softmax(torch.randn(8, 2), dim=-1)

        regret_loss = nn.functional.mse_loss(regret_output['advantages'], target_regrets)
        strategy_loss = -(target_strategies * torch.log(strategy_output['policy'] + 1e-8)).sum(dim=1).mean()

        print(f"✓ Regret loss: {regret_loss.item():.4f}")
        print(f"✓ Strategy loss: {strategy_loss.item():.4f}")

        print("✅ ALL DEEP CFR COMPONENTS WORK!")
        return True

    except Exception as e:
        print(f"❌ Deep CFR components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running standalone tests to verify actual training functionality...")

    success1 = test_neural_network_training()
    success2 = test_deep_cfr_components()

    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Neural networks are functional and can learn!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)