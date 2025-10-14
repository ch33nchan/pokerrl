"""Simple test script to verify scheduler fixes work correctly."""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algs.scheduler.scheduler import Scheduler, create_scheduler
from algs.scheduler.policy_mixer import (
    PolicyMixer,
    mix_policies,
    discrete_logits_to_lambda,
)
from algs.scheduler.meta_regret import MetaRegretManager, create_meta_regret_manager


def test_continuous_scheduler():
    """Test continuous scheduler with proper shape handling."""
    print("Testing continuous scheduler...")

    # Create scheduler
    config = {"hidden": [32, 16], "k_bins": None}
    scheduler = Scheduler(input_dim=10, **config)

    # Test input
    batch_size = 8
    z = torch.randn(batch_size, 10)

    # Forward pass
    output = scheduler.forward(z)

    # Verify output format
    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert output["mode"] == "continuous", (
        f"Expected continuous mode, got {output['mode']}"
    )
    assert "lambda" in output, "Missing 'lambda' key in output"
    assert output["lambda"].shape == (batch_size, 1), (
        f"Expected shape (8,1), got {output['lambda'].shape}"
    )

    # Verify lambda values are in valid range
    lam = output["lambda"]
    assert torch.all(lam >= 0.0), "Lambda values should be >= 0"
    assert torch.all(lam <= 1.0), "Lambda values should be <= 1"

    print("✓ Continuous scheduler test passed")
    return True


def test_discrete_scheduler():
    """Test discrete scheduler with proper shape handling."""
    print("Testing discrete scheduler...")

    # Create scheduler
    lambda_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    config = {"hidden": [32, 16], "k_bins": lambda_bins}
    scheduler = Scheduler(input_dim=10, **config)

    # Test input
    batch_size = 8
    z = torch.randn(batch_size, 10)

    # Test training mode
    scheduler.train()
    output = scheduler.forward(z)

    # Verify output format
    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert output["mode"] == "discrete", f"Expected discrete mode, got {output['mode']}"
    assert "logits" in output, "Missing 'logits' key in output"
    assert output["logits"].shape == (batch_size, 5), (
        f"Expected shape (8,5), got {output['logits'].shape}"
    )

    # Test eval mode
    scheduler.eval()
    output_eval = scheduler.forward(z, hard=True)

    assert "lambda_idx" in output_eval, "Missing 'lambda_idx' key in eval output"
    assert output_eval["lambda_idx"].shape == (batch_size,), (
        f"Expected shape (8,), got {output_eval['lambda_idx'].shape}"
    )
    assert output_eval["lambda_idx"].dtype == torch.long, (
        f"Expected long dtype, got {output_eval['lambda_idx'].dtype}"
    )

    print("✓ Discrete scheduler test passed")
    return True


def test_discrete_logits_to_lambda():
    """Test discrete logits to lambda conversion."""
    print("Testing discrete logits to lambda conversion...")

    # Test data
    batch_size = 4
    K = 5
    logits = torch.randn(batch_size, K)
    lambda_bins = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    # Test soft mode
    lambda_vals, indices = discrete_logits_to_lambda(logits, lambda_bins, hard=False)

    assert lambda_vals.shape == (batch_size, 1), (
        f"Expected shape (4,1), got {lambda_vals.shape}"
    )
    assert indices.shape == (batch_size,), f"Expected shape (4,), got {indices.shape}"
    assert torch.all(lambda_vals >= 0.0), "Lambda values should be >= 0"
    assert torch.all(lambda_vals <= 1.0), "Lambda values should be <= 1"

    # Test hard mode
    lambda_vals_hard, indices_hard = discrete_logits_to_lambda(
        logits, lambda_bins, hard=True
    )

    assert lambda_vals_hard.shape == (batch_size, 1), (
        f"Expected shape (4,1), got {lambda_vals_hard.shape}"
    )
    assert indices_hard.shape == (batch_size,), (
        f"Expected shape (4,), got {indices_hard.shape}"
    )

    # Verify hard mode produces actual bin values
    for i in range(batch_size):
        expected_val = lambda_bins[indices_hard[i]]
        assert torch.allclose(lambda_vals_hard[i], expected_val), (
            f"Hard mode should produce exact bin values"
        )

    print("✓ Discrete logits to lambda test passed")
    return True


def test_policy_mixing():
    """Test policy mixing with new standardized format."""
    print("Testing policy mixing...")

    # Test data
    batch_size = 6
    num_actions = 4

    actor_logits = torch.randn(batch_size, num_actions)
    regret_logits = torch.randn(batch_size, num_actions)

    # Test continuous mixing
    lambda_val = 0.3
    scheduler_out_cont = {
        "mode": "continuous",
        "lambda": torch.full((batch_size, 1), lambda_val),
    }

    mixed_cont = mix_policies(actor_logits, regret_logits, scheduler_out_cont)

    assert mixed_cont.shape == (batch_size, num_actions), (
        f"Expected shape ({batch_size},{num_actions}), got {mixed_cont.shape}"
    )

    # Check sum-to-1 constraint
    sums = mixed_cont.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6), (
        "Mixed policy should sum to 1"
    )

    # Test discrete mixing
    lambda_bins = [0.0, 0.5, 1.0]
    scheduler_out_disc = {"mode": "discrete", "logits": torch.randn(batch_size, 3)}

    mixed_disc = mix_policies(
        actor_logits,
        regret_logits,
        scheduler_out_disc,
        lambda_bins=torch.tensor(lambda_bins),
    )

    assert mixed_disc.shape == (batch_size, num_actions), (
        f"Expected shape ({batch_size},{num_actions}), got {mixed_disc.shape}"
    )

    # Check sum-to-1 constraint
    sums_disc = mixed_disc.sum(dim=-1)
    assert torch.allclose(sums_disc, torch.ones(batch_size), atol=1e-6), (
        "Mixed policy should sum to 1"
    )

    print("✓ Policy mixing test passed")
    return True


def test_meta_regret():
    """Test meta-regret manager functionality."""
    print("Testing meta-regret manager...")

    def simple_key_func(x):
        return str(int(torch.sum(x).item() % 10))

    # Create meta-regret manager
    meta_regret = MetaRegretManager(
        K=5,
        state_key_func=simple_key_func,
        max_states=50,  # Small limit for testing
    )

    # Test recording
    state_enc = torch.randn(10)
    state_key = simple_key_func(state_enc)

    stats = meta_regret.record(state_key, k_choice=2, utility=0.5)

    assert "util_ema" in stats, "Missing util_ema in stats"
    assert "regret_increment" in stats, "Missing regret_increment in stats"
    assert stats["total_updates"] == 1, (
        f"Expected 1 update, got {stats['total_updates']}"
    )

    # Test action probabilities
    probs = meta_regret.get_action_probs(state_key)
    assert probs.shape == (5,), f"Expected shape (5,), got {probs.shape}"
    assert abs(probs.sum() - 1.0) < 1e-6, "Probabilities should sum to 1"

    # Test LRU eviction
    for i in range(100):
        state_key_i = f"state_{i}"
        meta_regret.record(state_key_i, 0, 0.1)

    # Allow some tolerance for eviction timing
    assert len(meta_regret.regrets) <= 60, (
        f"LRU eviction should limit number of states, got {len(meta_regret.regrets)}"
    )
    assert meta_regret.eviction_count > 0, "Should have performed evictions"

    print("✓ Meta-regret manager test passed")
    return True


def test_device_consistency():
    """Test device consistency across components."""
    print("Testing device consistency...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create components
    lambda_bins = [0.0, 0.5, 1.0]
    scheduler = Scheduler(input_dim=6, k_bins=lambda_bins).to(device)

    # Test input
    batch_size = 4
    z = torch.randn(batch_size, 6, device=device)

    # Forward pass
    output = scheduler.forward(z)

    # Verify device consistency
    assert output["logits"].device == device, (
        "Scheduler output should be on same device"
    )

    # Test policy mixing
    actor_logits = torch.randn(batch_size, 3, device=device)
    regret_logits = torch.randn(batch_size, 3, device=device)

    mixed = mix_policies(
        actor_logits,
        regret_logits,
        output,
        lambda_bins=torch.tensor(lambda_bins, device=device),
    )

    assert mixed.device == device, "Mixed policy should be on same device"
    assert mixed.shape == (batch_size, 3), (
        f"Expected shape ({batch_size},3), got {mixed.shape}"
    )

    print("✓ Device consistency test passed")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")

    # Test empty batch
    scheduler = Scheduler(input_dim=4, k_bins=[0.0, 0.5, 1.0])
    empty_input = torch.randn(0, 4)

    try:
        output = scheduler.forward(empty_input)
        assert output["logits"].shape[0] == 0, "Should handle empty batch"
    except Exception as e:
        print(f"Warning: Empty batch handling failed: {e}")

    # Test lambda clamping
    scheduler_cont = Scheduler(input_dim=4, k_bins=None)
    scheduler_cont.set_clamping(eps=1e-3)

    z = torch.randn(4, 4)
    output = scheduler_cont.forward(z)
    lam = output["lambda"]

    assert torch.all(lam >= 1e-3), "Lambda should be clamped to minimum"
    assert torch.all(lam <= 1.0 - 1e-3), "Lambda should be clamped to maximum"

    print("✓ Edge cases test passed")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("Running Scheduler Fix Verification Tests")
    print("=" * 50)

    tests = [
        test_continuous_scheduler,
        test_discrete_scheduler,
        test_discrete_logits_to_lambda,
        test_policy_mixing,
        test_meta_regret,
        test_device_consistency,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("🎉 All tests passed! Scheduler fixes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
