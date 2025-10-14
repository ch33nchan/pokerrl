#!/usr/bin/env python3
"""
Simple test script for ARMAC scheduler components.

This script tests the basic functionality of the scheduler, policy mixer,
and meta-regret components before running full experiments.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algs.scheduler.scheduler import (
    Scheduler,
    compute_scheduler_input,
    create_scheduler,
)
from algs.scheduler.policy_mixer import PolicyMixer, mix_policies, create_policy_mixer
from algs.scheduler.meta_regret import MetaRegretManager, create_meta_regret_manager


def test_scheduler_continuous():
    """Test continuous scheduler functionality."""
    print("Testing continuous scheduler...")

    # Create continuous scheduler
    scheduler = Scheduler(
        input_dim=10,
        hidden=[32, 16],
        k_bins=None,  # Continuous mode
    )

    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 10)
    lambda_vals = scheduler(z)

    assert lambda_vals.shape == (batch_size, 1), (
        f"Expected shape {(batch_size, 1)}, got {lambda_vals.shape}"
    )
    assert torch.all(lambda_vals >= 0) and torch.all(lambda_vals <= 1), (
        "Lambda values should be in [0, 1]"
    )

    # Test get_lambda_values
    lambda_eff = scheduler.get_lambda_values(z)
    assert lambda_eff.shape == (batch_size,), (
        f"Expected shape {(batch_size,)}, got {lambda_eff.shape}"
    )

    print("✓ Continuous scheduler test passed")
    return scheduler


def test_scheduler_discrete():
    """Test discrete scheduler functionality."""
    print("Testing discrete scheduler...")

    # Create discrete scheduler
    k_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    scheduler = Scheduler(
        input_dim=10,
        hidden=[32, 16],
        k_bins=k_bins,
        temperature=1.0,
        use_gumbel=True,
    )

    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 10)

    # Test training mode (soft)
    scheduler.train()
    logits = scheduler(z)
    assert logits.shape == (batch_size, len(k_bins)), (
        f"Expected shape {(batch_size, len(k_bins))}, got {logits.shape}"
    )

    # Test evaluation mode (hard)
    scheduler.eval()
    hard_choices = scheduler(z, hard=True)
    assert hard_choices.shape == (batch_size, len(k_bins)), (
        f"Expected shape {(batch_size, len(k_bins))}, got {hard_choices.shape}"
    )

    # Test get_lambda_values
    lambda_vals = scheduler.get_lambda_values(z)
    assert lambda_vals.shape == (batch_size,), (
        f"Expected shape {(batch_size,)}, got {lambda_vals.shape}"
    )

    print("✓ Discrete scheduler test passed")
    return scheduler


def test_policy_mixer():
    """Test policy mixer functionality."""
    print("Testing policy mixer...")

    # Create test data
    batch_size = 4
    num_actions = 3
    actor_logits = torch.randn(batch_size, num_actions)
    regret_logits = torch.randn(batch_size, num_actions)

    # Test continuous mixing
    mixer_continuous = PolicyMixer(discrete=False)
    lambda_vals = torch.rand(batch_size)

    mixed_policy = mixer_continuous.mix(actor_logits, regret_logits, lambda_vals)
    assert mixed_policy.shape == (batch_size, num_actions), (
        f"Expected shape {(batch_size, num_actions)}, got {mixed_policy.shape}"
    )

    # Check if valid probabilities
    assert torch.allclose(
        mixed_policy.sum(dim=-1), torch.ones(batch_size), atol=1e-6
    ), "Mixed policy should sum to 1"
    assert torch.all(mixed_policy >= 0), "Mixed policy should be non-negative"

    # Test discrete mixing
    lambda_bins = torch.tensor([0.0, 0.5, 1.0])
    mixer_discrete = PolicyMixer(discrete=True, lambda_bins=lambda_bins)
    discrete_choices = torch.randint(0, 3, (batch_size,))

    mixed_policy_discrete = mixer_discrete.mix(
        actor_logits, regret_logits, discrete_choices
    )
    assert mixed_policy_discrete.shape == (batch_size, num_actions), (
        f"Expected shape {(batch_size, num_actions)}, got {mixed_policy_discrete.shape}"
    )

    # Test mixing stats
    stats = mixer_continuous.compute_mixing_stats(
        actor_logits, regret_logits, lambda_vals
    )
    assert "lambda_mean" in stats, "Stats should contain lambda_mean"
    assert "entropy_mix" in stats, "Stats should contain entropy_mix"

    print("✓ Policy mixer test passed")
    return mixer_continuous, mixer_discrete


def test_meta_regret():
    """Test meta-regret manager functionality."""
    print("Testing meta-regret manager...")

    # Create meta-regret manager
    K = 5
    meta_regret = MetaRegretManager(
        K=K,
        state_key_func=lambda x: f"state_{hash(tuple(x.numpy())) % 100}",
        decay=0.99,
    )

    # Test recording and action selection
    state_key = "test_state"

    # Record some utilities
    for k in range(K):
        utility = np.random.randn()
        stats = meta_regret.record(state_key, k, utility)
        assert "regret_increment" in stats, "Stats should contain regret_increment"

    # Test action probabilities
    probs = meta_regret.get_action_probs(state_key)
    assert probs.shape == (K,), f"Expected shape {(K,)}, got {probs.shape}"
    assert abs(probs.sum() - 1.0) < 1e-6, "Probabilities should sum to 1"

    # Test action sampling
    action, probs = meta_regret.sample_action(state_key)
    assert 0 <= action < K, f"Action should be in [0, {K - 1}]"

    # Test statistics
    stats = meta_regret.get_regret_stats(state_key)
    assert "regrets" in stats, "Stats should contain regrets"
    assert "utilities" in stats, "Stats should contain utilities"

    global_stats = meta_regret.get_global_stats()
    assert "total_updates" in global_stats, "Global stats should contain total_updates"

    print("✓ Meta-regret manager test passed")
    return meta_regret


def test_scheduler_input_computation():
    """Test scheduler input computation."""
    print("Testing scheduler input computation...")

    batch_size = 4
    state_dim = 8
    num_actions = 3

    # Create test data
    state_encoding = torch.randn(batch_size, state_dim)
    actor_logits = torch.randn(batch_size, num_actions)
    regret_logits = torch.randn(batch_size, num_actions)
    iteration = 100
    additional_features = torch.randn(batch_size, 2)

    # Compute scheduler input
    scheduler_input = compute_scheduler_input(
        state_encoding, actor_logits, regret_logits, iteration, additional_features
    )

    # Check dimensions (state + KL + actor_entropy + regret_entropy + iteration + additional)
    expected_dim = state_dim + 1 + 1 + 1 + 1 + 2  # 8 + 1 + 1 + 1 + 1 + 2 = 14
    assert scheduler_input.shape == (batch_size, expected_dim), (
        f"Expected shape {(batch_size, expected_dim)}, got {scheduler_input.shape}"
    )

    print("✓ Scheduler input computation test passed")


def test_integration():
    """Test integration of all components."""
    print("Testing component integration...")

    batch_size = 4
    state_dim = 8
    num_actions = 3

    # Create components (state_dim + KL + actor_entropy + regret_entropy + iteration = 8 + 1 + 1 + 1 + 1 = 12)
    scheduler = Scheduler(
        input_dim=state_dim + 4, hidden=[32, 16], k_bins=[0.0, 0.5, 1.0]
    )
    mixer = PolicyMixer(discrete=True, lambda_bins=[0.0, 0.5, 1.0])
    meta_regret = MetaRegretManager(
        K=3,
        state_key_func=lambda x: f"state_{hash(tuple(x.numpy())) % 100}",
    )

    # Create test data
    state_encoding = torch.randn(batch_size, state_dim)
    actor_logits = torch.randn(batch_size, num_actions)
    regret_logits = torch.randn(batch_size, num_actions)

    # Compute scheduler input
    scheduler_input = compute_scheduler_input(
        state_encoding, actor_logits, regret_logits, 50
    )

    # Get lambda values from scheduler
    lambda_vals = scheduler.get_lambda_values(scheduler_input)

    # For discrete mode, convert lambda values to discrete choices
    discrete_choices = torch.argmin(
        torch.abs(
            torch.tensor([0.0, 0.5, 1.0]).unsqueeze(0) - lambda_vals.unsqueeze(1)
        ),
        dim=1,
    )

    # Mix policies using discrete choices
    mixed_policy = mixer.mix(actor_logits, regret_logits, discrete_choices)

    # Simulate utilities and update meta-regret
    utilities = torch.randn(batch_size)
    for i in range(batch_size):
        state_key = meta_regret.state_key_func(state_encoding[i])
        meta_regret.record(state_key, discrete_choices[i].item(), utilities[i].item())

    # Check results
    assert mixed_policy.shape == (batch_size, num_actions)
    assert torch.allclose(mixed_policy.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    print("✓ Integration test passed")


def test_factory_functions():
    """Test factory functions."""
    print("Testing factory functions...")

    # Test scheduler factory
    config = {
        "scheduler": {
            "hidden": [32, 16],
            "k_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
            "temperature": 1.0,
        }
    }
    scheduler = create_scheduler(config, input_dim=10)
    assert scheduler.discrete == True
    assert len(scheduler.k_bins) == 5

    # Test policy mixer factory
    config = {
        "policy_mixer": {
            "discrete": True,
            "lambda_bins": [0.0, 0.5, 1.0],
        }
    }
    mixer = create_policy_mixer(config)
    assert mixer.discrete == True
    assert len(mixer.lambda_bins) == 3

    # Test meta-regret factory
    config = {
        "meta_regret": {
            "K": 5,
            "decay": 0.95,
        }
    }
    meta_regret = create_meta_regret_manager(config)
    assert meta_regret.K == 5
    assert meta_regret.decay == 0.95

    print("✓ Factory functions test passed")


def main():
    """Run all tests."""
    print("Running ARMAC scheduler component tests...\n")

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run individual component tests
        test_scheduler_continuous()
        test_scheduler_discrete()
        test_policy_mixer()
        test_meta_regret()
        test_scheduler_input_computation()
        test_integration()
        test_factory_functions()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("Scheduler components are working correctly.")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
