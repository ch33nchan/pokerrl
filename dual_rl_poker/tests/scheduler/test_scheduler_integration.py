"""Comprehensive unit tests for scheduler components.

This module tests all scheduler functionality including:
- Shape and device consistency
- Discrete and continuous modes
- Policy mixing correctness
- Meta-regret functionality
- Utility computation
- Deterministic replay
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json

from algs.scheduler.scheduler import Scheduler, create_scheduler
from algs.scheduler.policy_mixer import (
    PolicyMixer,
    mix_policies,
    discrete_logits_to_lambda,
    create_policy_mixer,
)
from algs.scheduler.meta_regret import (
    MetaRegretManager,
    create_meta_regret_manager,
    compute_state_key_simple,
)
from algs.scheduler.utils.state_keying import StateKeyManager, create_state_key_manager
from algs.scheduler.utils.utility_computation import (
    UtilitySignalComputer,
    create_utility_computer,
)
from algs.scheduler.utils.replay.deterministic_replay import (
    DeterministicReplayWriter,
    DeterministicReplayReader,
    ReplayVerifier,
    ReplayStep,
    ReplayEpisode,
)
from algs.scheduler.training.scheduler_trainer import (
    SchedulerTrainer,
    create_scheduler_trainer,
)


class TestScheduler:
    """Test scheduler network functionality."""

    @pytest.fixture
    def continuous_scheduler(self):
        """Create a continuous scheduler for testing."""
        config = {"hidden": [32, 16], "k_bins": None}
        return Scheduler(input_dim=10, **config)

    @pytest.fixture
    def discrete_scheduler(self):
        """Create a discrete scheduler for testing."""
        lambda_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        config = {"hidden": [32, 16], "k_bins": lambda_bins}
        return Scheduler(input_dim=10, **config)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(8, 10)  # batch_size=8, input_dim=10

    def test_continuous_scheduler_shapes(self, continuous_scheduler, sample_input):
        """Test continuous scheduler output shapes."""
        output = continuous_scheduler(sample_input)

        assert isinstance(output, dict)
        assert output["mode"] == "continuous"
        assert "lambda" in output
        assert output["lambda"].shape == (8, 1)
        assert output["lambda"].device == sample_input.device

        # Check lambda values are in valid range
        assert torch.all(output["lambda"] >= 0.0)
        assert torch.all(output["lambda"] <= 1.0)

    def test_discrete_scheduler_shapes(self, discrete_scheduler, sample_input):
        """Test discrete scheduler output shapes."""
        output = discrete_scheduler(sample_input)

        assert isinstance(output, dict)
        assert output["mode"] == "discrete"
        assert "logits" in output
        assert output["logits"].shape == (8, 5)  # 5 bins
        assert output["logits"].device == sample_input.device

        # During training, should not have lambda_idx initially
        if discrete_scheduler.training:
            assert "lambda_idx" not in output

    def test_discrete_scheduler_hard_mode(self, discrete_scheduler, sample_input):
        """Test discrete scheduler in hard mode."""
        discrete_scheduler.eval()  # Set to eval mode
        output = discrete_scheduler(sample_input, hard=True)

        assert output["mode"] == "discrete"
        assert "lambda_idx" in output
        assert output["lambda_idx"].shape == (8,)
        assert output["lambda_idx"].dtype == torch.long

    def test_scheduler_device_consistency(self, discrete_scheduler):
        """Test scheduler maintains device consistency."""
        # Test on CPU
        cpu_input = torch.randn(4, 10)
        cpu_output = discrete_scheduler(cpu_input)
        assert cpu_output["logits"].device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            gpu_input = cpu_input.cuda()
            gpu_output = discrete_scheduler(gpu_input)
            assert gpu_output["logits"].device.type == "cuda"

    def test_scheduler_get_lambda_values(self, discrete_scheduler, sample_input):
        """Test get_lambda_values method."""
        lambda_vals = discrete_scheduler.get_lambda_values(sample_input)
        assert lambda_vals.shape == (8,)
        assert torch.all(lambda_vals >= 0.0)
        assert torch.all(lambda_vals <= 1.0)

    def test_scheduler_regularization(self, continuous_scheduler):
        """Test scheduler regularization."""
        config = {"beta_l2": 0.01, "beta_ent": 0.01}
        continuous_scheduler.set_regularization(config)

        reg_loss = continuous_scheduler.compute_regularization_loss()
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_scheduler_warmup(self, continuous_scheduler, sample_input):
        """Test scheduler warmup functionality."""
        continuous_scheduler.set_warmup(warmup_iters=100, init_lambda=0.7)

        # During warmup
        warmup_output = continuous_scheduler.forward_with_warmup(sample_input, 50)
        assert warmup_output["mode"] == "continuous"
        assert torch.allclose(warmup_output["lambda"], torch.tensor(0.7))

        # After warmup
        normal_output = continuous_scheduler.forward_with_warmup(sample_input, 150)
        assert normal_output["mode"] == "continuous"
        # Should not be exactly 0.7 anymore
        assert not torch.allclose(normal_output["lambda"], torch.tensor(0.7))


class TestPolicyMixer:
    """Test policy mixing functionality."""

    @pytest.fixture
    def sample_logits(self):
        """Create sample actor and regret logits."""
        batch_size, num_actions = 8, 4
        return {
            "actor": torch.randn(batch_size, num_actions),
            "regret": torch.randn(batch_size, num_actions),
        }

    @pytest.fixture
    def continuous_mixer(self):
        """Create a continuous policy mixer."""
        config = {"discrete": False}
        return PolicyMixer(**config)

    @pytest.fixture
    def discrete_mixer(self):
        """Create a discrete policy mixer."""
        lambda_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        config = {"discrete": True, "lambda_bins": lambda_bins}
        return PolicyMixer(**config)

    def test_discrete_logits_to_lambda(self):
        """Test discrete logits to lambda conversion."""
        logits = torch.randn(4, 5)
        lambda_bins = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        # Test soft mode
        lambda_vals, indices = discrete_logits_to_lambda(
            logits, lambda_bins, hard=False
        )
        assert lambda_vals.shape == (4, 1)
        assert indices.shape == (4,)
        assert torch.all(lambda_vals >= 0.0)
        assert torch.all(lambda_vals <= 1.0)

        # Test hard mode
        lambda_vals_hard, indices_hard = discrete_logits_to_lambda(
            logits, lambda_bins, hard=True
        )
        assert lambda_vals_hard.shape == (4, 1)
        assert indices_hard.shape == (4,)

    def test_continuous_mixing(self, sample_logits, continuous_mixer):
        """Test continuous policy mixing."""
        scheduler_out = {"mode": "continuous", "lambda": torch.full((8, 1), 0.3)}
        mixed_policy = continuous_mixer.mix(
            sample_logits["actor"], sample_logits["regret"], scheduler_out
        )

        assert mixed_policy.shape == (8, 4)
        # Check sum-to-1 constraint
        sums = mixed_policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-6)

    def test_discrete_mixing(self, sample_logits, discrete_mixer):
        """Test discrete policy mixing."""
        logits = torch.randn(8, 5)
        scheduler_out = {"mode": "discrete", "logits": logits}
        mixed_policy = discrete_mixer.mix(
            sample_logits["actor"], sample_logits["regret"], scheduler_out
        )

        assert mixed_policy.shape == (8, 4)
        # Check sum-to-1 constraint
        sums = mixed_policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-6)

    def test_mixing_edge_cases(self, continuous_mixer):
        """Test mixing edge cases."""
        actor_logits = torch.randn(4, 3)
        regret_logits = torch.randn(4, 3)

        # Test lambda = 0 (pure regret)
        scheduler_out = {"mode": "continuous", "lambda": torch.zeros((4, 1))}
        mixed_0 = continuous_mixer.mix(actor_logits, regret_logits, scheduler_out)

        # Test lambda = 1 (pure actor)
        scheduler_out = {"mode": "continuous", "lambda": torch.ones((4, 1))}
        mixed_1 = continuous_mixer.mix(actor_logits, regret_logits, scheduler_out)

        # Results should be different
        assert not torch.allclose(mixed_0, mixed_1)

    def test_mixing_stats(self, sample_logits, continuous_mixer):
        """Test mixing statistics computation."""
        scheduler_out = {"mode": "continuous", "lambda": torch.full((8, 1), 0.5)}
        stats = continuous_mixer.compute_mixing_stats(
            sample_logits["actor"], sample_logits["regret"], scheduler_out
        )

        required_keys = [
            "lambda_mean",
            "lambda_std",
            "lambda_min",
            "lambda_max",
            "kl_actor_regret",
            "kl_mix_actor",
            "kl_mix_regret",
            "entropy_actor",
            "entropy_regret",
            "entropy_mix",
        ]

        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float)


class TestMetaRegret:
    """Test meta-regret manager functionality."""

    @pytest.fixture
    def meta_regret(self):
        """Create a meta-regret manager."""

        def simple_key_func(x):
            return str(int(x.item() % 10))  # Simple discretization

        return MetaRegretManager(
            K=5,
            state_key_func=simple_key_func,
            decay=0.99,
            max_states=100,
        )

    def test_state_key_computation(self):
        """Test state key computation."""
        encoding = torch.randn(10)
        key = compute_state_key_simple(encoding)
        assert isinstance(key, str)

    def test_meta_regret_record(self, meta_regret):
        """Test recording utilities to meta-regret."""
        state_key = "test_state"
        k_choice = 2
        utility = 0.5

        stats = meta_regret.record(state_key, k_choice, utility)

        assert "util_ema" in stats
        assert "regret_increment" in stats
        assert "total_updates" in stats
        assert stats["total_updates"] == 1

    def test_meta_regret_action_probs(self, meta_regret):
        """Test action probability computation."""
        state_key = "test_state"

        # Initially uniform
        probs = meta_regret.get_action_probs(state_key)
        assert probs.shape == (5,)
        assert torch.allclose(torch.tensor(probs), torch.tensor(0.2), atol=1e-6)

        # After recording some utilities
        meta_regret.record(state_key, 0, 1.0)
        meta_regret.record(state_key, 1, -1.0)
        meta_regret.record(state_key, 2, 0.5)

        probs = meta_regret.get_action_probs(state_key)
        assert torch.allclose(torch.tensor(probs).sum(), torch.tensor(1.0), atol=1e-6)

    def test_meta_regret_lru_eviction(self, meta_regret):
        """Test LRU eviction functionality."""
        # Fill beyond max_states
        for i in range(150):
            state_key = f"state_{i}"
            meta_regret.record(state_key, 0, 0.1)

        # Should have been evicted
        assert len(meta_regret.regrets) <= meta_regret.max_states
        assert meta_regret.eviction_count > 0

    def test_meta_regret_stats(self, meta_regret):
        """Test meta-regret statistics."""
        # Add some data
        for i in range(10):
            state_key = f"state_{i % 3}"
            k_choice = i % 5
            utility = np.random.randn()
            meta_regret.record(state_key, k_choice, utility)

        stats = meta_regret.get_global_stats()
        assert "total_states" in stats
        assert "total_updates" in stats
        assert "regret_mean" in stats
        assert "util_mean" in stats

    def test_meta_regret_persistence(self, meta_regret):
        """Test meta-regret save/load functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Add some data
            meta_regret.record("state_1", 0, 1.0)
            meta_regret.record("state_2", 1, -0.5)

            # Save state
            meta_regret.save_state(filepath)

            # Create new manager and load state
            new_meta_regret = MetaRegretManager(
                K=5,
                state_key_func=meta_regret.state_key_func,
                max_states=100,
            )
            new_meta_regret.load_state(filepath)

            # Check data was restored
            assert new_meta_regret.total_updates == meta_regret.total_updates
            assert len(new_meta_regret.regrets) == len(meta_regret.regrets)

        finally:
            import os

            os.unlink(filepath)


class TestUtilityComputation:
    """Test utility signal computation."""

    @pytest.fixture
    def utility_computer(self):
        """Create a utility computer."""
        return UtilitySignalComputer(
            utility_type="immediate",
            gamma=0.99,
            baseline_window=10,
        )

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory."""
        trajectory = []
        for i in range(10):
            step = {
                "s": {"encoding": np.random.randn(5)},
                "reward": 0.1 if i < 5 else -0.1,
                "action": i % 3,
            }
            trajectory.append(step)
        return trajectory

    def test_immediate_utility_computation(self, utility_computer, sample_trajectory):
        """Test immediate utility computation."""
        utility = utility_computer.compute_scheduler_utility(
            sample_trajectory, decision_index=3
        )
        assert isinstance(utility, float)

    def test_advantage_based_utility(self):
        """Test advantage-based utility computation."""
        # Create mock critic
        critic = Mock()
        critic.return_value = torch.tensor([[0.5, 0.3, 0.2]])

        utility_computer = UtilitySignalComputer(utility_type="advantage_based")
        trajectory = [
            {"s": {"encoding": np.random.randn(5)}, "reward": 0.1},
            {"s": {"encoding": np.random.randn(5)}, "reward": -0.1},
        ]

        utility = utility_computer.compute_scheduler_utility(
            trajectory,
            decision_index=0,
            critic_network=critic,
            state_encoding_fn=lambda s: torch.tensor(
                s["encoding"], dtype=torch.float32
            ),
        )
        assert isinstance(utility, float)

    def test_episode_metrics(self, utility_computer, sample_trajectory):
        """Test episode metrics computation."""
        metrics = utility_computer.compute_episode_metrics(sample_trajectory)

        required_keys = [
            "total_reward",
            "episode_length",
            "avg_reward_per_step",
        ]
        for key in required_keys:
            assert key in metrics


class TestDeterministicReplay:
    """Test deterministic replay functionality."""

    @pytest.fixture
    def replay_writer(self):
        """Create a replay writer."""
        return DeterministicReplayWriter()

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode."""
        steps = []
        for i in range(5):
            step = ReplayStep(
                t=i,
                s={"pot": 10, "player_pos": i % 2},
                actor_logits=[0.1, 0.2, 0.7],
                regret_logits=[0.3, 0.4, 0.3],
                scheduler_logits=[0.2, 0.3, 0.2, 0.2, 0.1],
                lambda_val=0.5,
                k_choice=2,
                action_sampled=2,
                reward=0.1,
                done=(i == 4),
                legal_actions=[0, 1, 2],
            )
            steps.append(step)

        return ReplayEpisode(
            run_id="test_run",
            seed=12345,
            env="kuhn",
            config={"test": True},
            deck_order=[1, 2, 3],
            trajectory=steps,
            rng_state="dummy_state",
            timestamp=1234567890.0,
            total_reward=0.5,
            episode_length=5,
        )

    def test_replay_write_read(self, replay_writer, sample_episode):
        """Test writing and reading replay data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_writer.replay_dir = tmpdir
            replay_file = replay_writer.start_new_file("test_run")

            # Write episode
            replay_writer.write_episode(sample_episode)
            replay_writer.close()

            # Read episodes
            reader = DeterministicReplayReader(replay_file)
            episodes = reader.read_episodes()

            assert len(episodes) == 1
            assert episodes[0].run_id == sample_episode.run_id
            assert len(episodes[0].trajectory) == len(sample_episode.trajectory)

    def test_replay_verification(self, sample_episode):
        """Test replay verification."""
        # Create mock networks
        actor = Mock()
        actor.return_value = torch.tensor([0.1, 0.2, 0.7])
        regret = Mock()
        regret.return_value = torch.tensor([0.3, 0.4, 0.3])

        verifier = ReplayVerifier(tolerance=1e-6)

        # This should succeed (mock networks return consistent values)
        results = verifier.verify_episode(sample_episode, actor, regret)

        assert "run_id" in results
        assert "verified_steps" in results
        assert "failed_steps" in results
        assert "success" in results


class TestSchedulerTrainer:
    """Test scheduler trainer functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create a discrete scheduler."""
        lambda_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        return Scheduler(input_dim=5, k_bins=lambda_bins)

    @pytest.fixture
    def meta_regret(self):
        """Create a meta-regret manager."""

        def key_func(x):
            return "test_key"

        return MetaRegretManager(K=5, state_key_func=key_func)

    @pytest.fixture
    def scheduler_trainer(self, scheduler, meta_regret):
        """Create a scheduler trainer."""
        config = {
            "scheduler_lr": 1e-4,
            "gumbel_tau_start": 1.0,
            "gumbel_tau_end": 0.1,
            "gumbel_anneal_iters": 1000,
        }
        return SchedulerTrainer(scheduler, meta_regret, config)

    def test_temperature_annealing(self, scheduler_trainer):
        """Test temperature annealing."""
        # Initial temperature
        scheduler_trainer.current_iteration = 0
        tau1 = scheduler_trainer.update_temperature()
        assert tau1 == scheduler_trainer.gumbel_tau_start

        # Mid-way
        scheduler_trainer.current_iteration = 500
        tau2 = scheduler_trainer.update_temperature()
        assert tau1 > tau2 > scheduler_trainer.gumbel_tau_end

        # End
        scheduler_trainer.current_iteration = 2000
        tau3 = scheduler_trainer.update_temperature()
        assert tau3 == scheduler_trainer.gumbel_tau_end

    def test_scheduler_loss_computation(self, scheduler_trainer):
        """Test scheduler loss computation."""
        # Create sample logits
        logits = [torch.randn(5) for _ in range(3)]
        state_keys = ["key1", "key2", "key3"]

        loss = scheduler_trainer._compute_scheduler_loss(logits, state_keys)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0

    def test_training_stats(self, scheduler_trainer):
        """Test training statistics."""
        stats = scheduler_trainer.get_training_stats()

        required_keys = [
            "current_iteration",
            "training_stats",
            "current_temperature",
            "meta_regret_stats",
        ]

        for key in required_keys:
            assert key in stats

    def test_checkpoint_save_load(self, scheduler_trainer):
        """Test checkpoint save/load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/test_checkpoint.pt"

            # Save checkpoint
            scheduler_trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load
            new_trainer = SchedulerTrainer(
                scheduler_trainer.scheduler,
                scheduler_trainer.meta_regret,
                scheduler_trainer.config,
            )
            new_trainer.load_checkpoint(checkpoint_path)

            # Check that stats were loaded
            assert new_trainer.current_iteration == scheduler_trainer.current_iteration


class TestIntegration:
    """Integration tests for the complete scheduler system."""

    def test_end_to_end_discrete_training(self):
        """Test end-to-end discrete scheduler training."""
        # Create components
        lambda_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        scheduler = Scheduler(input_dim=4, k_bins=lambda_bins)
        scheduler.train()

        def key_func(x):
            return "test_state"

        meta_regret = MetaRegretManager(K=5, state_key_func=key_func)

        config = {
            "scheduler_lr": 1e-3,
            "utility_computation": {"utility_type": "immediate"},
            "state_keying": {"level": 0},
        }

        trainer = create_scheduler_trainer(scheduler, meta_regret, config)

        # Create sample batch
        batch_size = 4
        trajectories = []
        scheduler_outputs = []
        state_encodings = []

        for _ in range(batch_size):
            # Sample trajectory
            trajectory = []
            sched_outputs = []
            state_encs = []

            for t in range(5):
                # Sample state encoding
                state_enc = torch.randn(4)
                state_encs.append(state_enc)

                # Get scheduler output
                sched_out = scheduler(state_enc.unsqueeze(0))
                sched_outputs.append(sched_out)

                # Create step
                step = {
                    "s": {"pot": 10, "player_pos": 0, "embedding": state_enc},
                    "reward": np.random.randn(),
                    "action": np.random.randint(0, 3),
                }
                trajectory.append(step)

            trajectories.append(trajectory)
            scheduler_outputs.append(sched_outputs)
            state_encodings.append(state_encs)

        # Process batch
        stats = trainer.process_trajectory_batch(
            trajectories, scheduler_outputs, state_encodings, iteration=0
        )

        # Check that training happened
        assert "scheduler_loss" in stats
        assert "meta_regret_updates" in stats
        assert stats["meta_regret_updates"] > 0

    def test_device_consistency_across_components(self):
        """Test device consistency across all components."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create components
        lambda_bins = [0.0, 0.5, 1.0]
        scheduler = Scheduler(input_dim=4, k_bins=lambda_bins).to(device)
        mixer = PolicyMixer(discrete=True, lambda_bins=lambda_bins)

        # Test input
        batch_size = 8
        state_enc = torch.randn(batch_size, 4, device=device)
        actor_logits = torch.randn(batch_size, 3, device=device)
        regret_logits = torch.randn(batch_size, 3, device=device)

        # Forward pass through scheduler
        scheduler_out = scheduler(state_enc)
        assert scheduler_out["logits"].device == device

        # Mix policies
        mixed_policy = mixer.mix(actor_logits, regret_logits, scheduler_out)
        assert mixed_policy.device == device

        # Check invariants
        assert mixed_policy.shape == (batch_size, 3)
        sums = mixed_policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size, device=device), atol=1e-6)

    def test_invariant_checks(self):
        """Test critical invariants for the scheduler system."""
        # Test sum-to-1 invariant
        lambda_bins = [0.0, 0.5, 1.0]
        mixer = PolicyMixer(discrete=True, lambda_bins=lambda_bins)

        actor_logits = torch.randn(4, 3)
        regret_logits = torch.randn(4, 3)
        scheduler_out = {"mode": "discrete", "logits": torch.randn(4, 3)}

        mixed_policy = mixer.mix(actor_logits, regret_logits, scheduler_out)

        # Invariant A: pi_mix.shape == [B, A] and sums to 1
        assert mixed_policy.shape == (4, 3)
        sums = mixed_policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)

        # Test lambda range invariant for continuous mode
        continuous_mixer = PolicyMixer(discrete=False)
        scheduler_out_cont = {"mode": "continuous", "lambda": torch.full((4, 1), 0.7)}

        mixed_policy_cont = continuous_mixer.mix(
            actor_logits, regret_logits, scheduler_out_cont
        )
        assert mixed_policy_cont.shape == (4, 3)
        sums_cont = mixed_policy_cont.sum(dim=-1)
        assert torch.allclose(sums_cont, torch.ones(4), atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
