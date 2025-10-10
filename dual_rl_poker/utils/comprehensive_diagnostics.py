"""
Comprehensive diagnostics logging system with Parquet output.

Tracks all training dynamics as specified in executive directive:
- Per-update gradient norms
- Advantage mean/std/quantiles
- Policy KL to previous checkpoint (fixed cadence)
- Clipping events
- Phase wall-clock timing
- Event-aligned analysis for NashConv spikes
- All output in Parquet format keyed by run_id/config/seed
"""

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import json
from collections import defaultdict
import logging


@dataclass
class TrainingMetrics:
    """Container for training step metrics."""
    iteration: int
    timestamp: float
    run_id: str
    config_hash: str
    seed: int
    algorithm: str
    game: str

    # Gradient norms
    actor_gradient_norm: float = 0.0
    critic_gradient_norm: float = 0.0
    regret_gradient_norm: float = 0.0
    total_gradient_norm: float = 0.0
    clipping_events: int = 0

    # Advantage statistics
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_q25: float = 0.0
    advantage_q50: float = 0.0
    advantage_q75: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0

    # Loss values
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    regret_loss: float = 0.0
    total_loss: float = 0.0

    # Policy KL divergence (at fixed cadence)
    policy_kl_divergence: float = 0.0
    policy_entropy: float = 0.0

    # Timing information
    wall_clock_time: float = 0.0
    actor_time: float = 0.0
    critic_time: float = 0.0
    regret_time: float = 0.0
    evaluation_time: float = 0.0

    # Evaluation metrics (exact OpenSpiel)
    nash_conv: float = float('inf')
    exploitability: float = float('inf')
    mean_value: float = 0.0

    # Network parameters
    actor_params: int = 0
    critic_params: int = 0
    regret_params: int = 0
    total_params: int = 0

    # Training dynamics
    buffer_size: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0


@dataclass
class CheckpointMetrics:
    """Container for checkpoint-level metrics."""
    iteration: int
    timestamp: float
    run_id: str
    config_hash: str
    seed: int

    # Policy distribution statistics
    policy_entropy_mean: float = 0.0
    policy_entropy_std: float = 0.0
    policy_sparsity_mean: float = 0.0  # Fraction of near-zero probabilities

    # Regret statistics
    regret_mean: float = 0.0
    regret_std: float = 0.0
    positive_regret_fraction: float = 0.0

    # Value function statistics
    value_mean: float = 0.0
    value_std: float = 0.0
    value_range: float = 0.0

    # NashConv spike detection
    nash_conv_change: float = 0.0
    nash_conv_spike: bool = False
    exploitability_change: float = 0.0


class ComprehensiveDiagnostics:
    """
    Comprehensive diagnostics logging system for training dynamics.

    Tracks all required metrics with event-aligned analysis capabilities.
    """

    def __init__(self, output_dir: str = "diagnostics"):
        """Initialize diagnostics system.

        Args:
            output_dir: Directory to store diagnostic outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Data storage
        self.training_metrics: List[TrainingMetrics] = []
        self.checkpoint_metrics: List[CheckpointMetrics] = []
        self.event_correlations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Previous policy for KL computation
        self.previous_policies: Dict[str, Dict[str, np.ndarray]] = {}

        # NashConv tracking for spike detection
        self.previous_nash_conv: Dict[str, float] = {}
        self.nash_conv_spike_threshold = 0.1  # 10% change considered spike

        # Timing phases
        self.phase_timers: Dict[str, float] = {}

        # Parquet writers
        self.training_writer = None
        self.checkpoint_writer = None
        self.correlation_writer = None

    def start_phase_timer(self, phase_name: str):
        """Start timing a training phase.

        Args:
            phase_name: Name of the phase (e.g., 'actor_update')
        """
        self.phase_timers[phase_name] = time.time()

    def end_phase_timer(self, phase_name: str) -> float:
        """End timing a training phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Duration in seconds
        """
        if phase_name in self.phase_timers:
            duration = time.time() - self.phase_timers[phase_name]
            del self.phase_timers[phase_name]
            return duration
        return 0.0

    def log_gradient_norms(self, networks: Dict[str, torch.nn.Module],
                           iteration: int, run_id: str, config_hash: str, seed: int):
        """Log gradient norms for all networks.

        Args:
            networks: Dictionary of network name to network module
            iteration: Training iteration
            run_id: Unique run identifier
            config_hash: Configuration hash
            seed: Random seed
        """
        gradient_norms = {}
        clipping_events = 0

        for name, network in networks.items():
            total_norm = 0.0
            for param in network.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    # Check for clipping (simplified detection)
                    if torch.any(torch.abs(param.grad) > 5.0):  # Assuming clip at 5.0
                        clipping_events += 1

            total_norm = total_norm ** 0.5
            gradient_norms[name] = total_norm

        # Store in current metrics
        if hasattr(self, '_current_metrics'):
            self._current_metrics.actor_gradient_norm = gradient_norms.get('actor', 0.0)
            self._current_metrics.critic_gradient_norm = gradient_norms.get('critic', 0.0)
            self._current_metrics.regret_gradient_norm = gradient_norms.get('regret', 0.0)
            self._current_metrics.total_gradient_norm = sum(gradient_norms.values())
            self._current_metrics.clipping_events = clipping_events

    def log_advantage_statistics(self, advantages: torch.Tensor, iteration: int):
        """Log advantage statistics with quantiles.

        Args:
            advantages: Advantage tensor
            iteration: Training iteration
        """
        if hasattr(self, '_current_metrics'):
            with torch.no_grad():
                adv_np = advantages.cpu().numpy()

                self._current_metrics.advantage_mean = float(np.mean(adv_np))
                self._current_metrics.advantage_std = float(np.std(adv_np))
                self._current_metrics.advantage_q25 = float(np.percentile(adv_np, 25))
                self._current_metrics.advantage_q50 = float(np.percentile(adv_np, 50))
                self._current_metrics.advantage_q75 = float(np.percentile(adv_np, 75))
                self._current_metrics.advantage_min = float(np.min(adv_np))
                self._current_metrics.advantage_max = float(np.max(adv_np))

    def log_policy_kl_divergence(self, current_policy: Dict[str, np.ndarray],
                               iteration: int, cadence: int = 100):
        """Log policy KL divergence at fixed cadence.

        Args:
            current_policy: Current policy dictionary
            iteration: Training iteration
            cadence: How often to compute KL (in iterations)
        """
        if iteration % cadence != 0:
            return

        if not hasattr(self, '_current_metrics'):
            return

        run_id = self._current_metrics.run_id
        total_kl = 0.0
        total_entropy = 0.0
        num_states = 0

        for info_state, policy in current_policy.items():
            if info_state in self.previous_policies.get(run_id, {}):
                prev_policy = self.previous_policies[run_id][info_state]

                # Compute KL divergence
                kl = np.sum(policy * (np.log(policy + 1e-8) - np.log(prev_policy + 1e-8)))
                total_kl += kl

                # Compute entropy
                entropy = -np.sum(policy * np.log(policy + 1e-8))
                total_entropy += entropy

                num_states += 1

        if num_states > 0:
            self._current_metrics.policy_kl_divergence = total_kl / num_states
            self._current_metrics.policy_entropy = total_entropy / num_states

        # Store current policy for next comparison
        if run_id not in self.previous_policies:
            self.previous_policies[run_id] = {}
        self.previous_policies[run_id] = current_policy.copy()

    def log_nash_conv_spike(self, nash_conv: float, iteration: int):
        """Detect and log NashConv spikes.

        Args:
            nash_conv: Current NashConv value
            iteration: Training iteration
        """
        if not hasattr(self, '_current_metrics'):
            return

        run_id = self._current_metrics.run_id

        if run_id in self.previous_nash_conv:
            prev_nash_conv = self.previous_nash_conv[run_id]

            if prev_nash_conv > 0:
                change = abs(nash_conv - prev_nash_conv) / prev_nash_conv
                is_spike = change > self.nash_conv_spike_threshold

                self._current_metrics.nash_conv_change = change

                if is_spike:
                    self._current_metrics.nash_conv_spike = True
                    self._detect_event_correlations('nash_conv_spike', iteration, nash_conv, change)

        self.previous_nash_conv[run_id] = nash_conv

    def _detect_event_correlations(self, event_type: str, iteration: int,
                                 nash_conv: float, magnitude: float):
        """Detect correlations between training events and NashConv spikes.

        Args:
            event_type: Type of event (e.g., 'nash_conv_spike')
            iteration: Training iteration
            nash_conv: NashConv value
            magnitude: Event magnitude
        """
        # Look back at recent training metrics for correlations
        recent_metrics = [m for m in self.training_metrics
                        if abs(m.iteration - iteration) <= 10]

        if not recent_metrics:
            return

        # Compute correlations with various metrics
        correlations = {
            'event_type': event_type,
            'iteration': iteration,
            'nash_conv': nash_conv,
            'magnitude': magnitude,
            'correlations': {}
        }

        # Correlate with gradient norms
        grad_norms = [m.total_gradient_norm for m in recent_metrics if m.total_gradient_norm > 0]
        if len(grad_norms) > 1:
            correlations['correlations']['gradient_norm_trend'] = grad_norms[-1] - grad_norms[0]

        # Correlate with advantage statistics
        if recent_metrics:
            correlations['correlations']['advantage_mean_before'] = recent_metrics[0].advantage_mean
            correlations['correlations']['advantage_mean_after'] = recent_metrics[-1].advantage_mean

        # Correlate with loss values
        losses = [m.total_loss for m in recent_metrics if m.total_loss > 0]
        if len(losses) > 1:
            correlations['correlations']['loss_trend'] = losses[-1] - losses[0]

        self.event_correlations[event_type].append(correlations)

    def start_training_iteration(self, iteration: int, run_id: str, config_hash: str,
                               seed: int, algorithm: str, game: str):
        """Start logging a training iteration.

        Args:
            iteration: Training iteration
            run_id: Unique run identifier
            config_hash: Configuration hash
            seed: Random seed
            algorithm: Algorithm name
            game: Game name
        """
        self._current_metrics = TrainingMetrics(
            iteration=iteration,
            timestamp=time.time(),
            run_id=run_id,
            config_hash=config_hash,
            seed=seed,
            algorithm=algorithm,
            game=game
        )

    def end_training_iteration(self, evaluation_metrics: Dict[str, float],
                              network_param_counts: Dict[str, int],
                              training_config: Dict[str, Any]):
        """End logging a training iteration and store metrics.

        Args:
            evaluation_metrics: Evaluation metrics from OpenSpiel
            network_param_counts: Parameter counts for networks
            training_config: Training configuration
        """
        if not hasattr(self, '_current_metrics'):
            return

        # Update evaluation metrics
        self._current_metrics.nash_conv = evaluation_metrics.get('nash_conv', float('inf'))
        self._current_metrics.exploitability = evaluation_metrics.get('exploitability', float('inf'))
        self._current_metrics.mean_value = evaluation_metrics.get('mean_value', 0.0)

        # Update network parameters
        self._current_metrics.actor_params = network_param_counts.get('actor', 0)
        self._current_metrics.critic_params = network_param_counts.get('critic', 0)
        self._current_metrics.regret_params = network_param_counts.get('regret', 0)
        self._current_metrics.total_params = sum(network_param_counts.values())

        # Update training configuration
        self._current_metrics.learning_rate = training_config.get('learning_rate', 0.0)
        self._current_metrics.batch_size = training_config.get('batch_size', 0)
        self._current_metrics.buffer_size = training_config.get('buffer_size', 0)

        # Store metrics
        self.training_metrics.append(self._current_metrics)

        # Write to Parquet periodically
        if len(self.training_metrics) % 100 == 0:
            self._flush_training_metrics()

        del self._current_metrics

    def log_checkpoint(self, iteration: int, run_id: str, config_hash: str, seed: int,
                      policy_dict: Dict[str, np.ndarray], regret_dict: Dict[str, np.ndarray],
                      value_dict: Dict[str, np.ndarray]):
        """Log checkpoint-level metrics.

        Args:
            iteration: Training iteration
            run_id: Unique run identifier
            config_hash: Configuration hash
            seed: Random seed
            policy_dict: Current policy
            regret_dict: Current regrets
            value_dict: Current value estimates
        """
        # Compute policy statistics
        entropies = []
        sparsities = []
        for policy in policy_dict.values():
            entropy = -np.sum(policy * np.log(policy + 1e-8))
            sparsity = np.mean(policy < 1e-6)
            entropies.append(entropy)
            sparsities.append(sparsity)

        # Compute regret statistics
        regret_values = list(regret_dict.values())
        positive_regrets = [np.sum(np.maximum(r, 0)) for r in regret_values]
        total_regrets = [np.sum(np.abs(r)) for r in regret_values]

        # Compute value statistics
        value_values = list(value_dict.values())

        # Detect NashConv spike
        nash_conv = self.training_metrics[-1].nash_conv if self.training_metrics else float('inf')
        nash_conv_change = 0.0
        nash_conv_spike = False

        if run_id in self.previous_nash_conv:
            prev_nash_conv = self.previous_nash_conv[run_id]
            if prev_nash_conv > 0:
                nash_conv_change = abs(nash_conv - prev_nash_conv) / prev_nash_conv
                nash_conv_spike = nash_conv_change > self.nash_conv_spike_threshold

        checkpoint = CheckpointMetrics(
            iteration=iteration,
            timestamp=time.time(),
            run_id=run_id,
            config_hash=config_hash,
            seed=seed,
            policy_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
            policy_entropy_std=float(np.std(entropies)) if len(entropies) > 1 else 0.0,
            policy_sparsity_mean=float(np.mean(sparsities)) if sparsities else 0.0,
            regret_mean=float(np.mean(total_regrets)) if total_regrets else 0.0,
            regret_std=float(np.std(total_regrets)) if len(total_regrets) > 1 else 0.0,
            positive_regret_fraction=float(np.mean(positive_regrets)) / float(np.mean(total_regrets)) if total_regrets and np.mean(total_regrets) > 0 else 0.0,
            value_mean=float(np.mean(value_values)) if value_values else 0.0,
            value_std=float(np.std(value_values)) if len(value_values) > 1 else 0.0,
            value_range=float(np.max(value_values) - np.min(value_values)) if len(value_values) > 1 else 0.0,
            nash_conv_change=nash_conv_change,
            nash_conv_spike=nash_conv_spike
        )

        self.checkpoint_metrics.append(checkpoint)

        # Write to Parquet periodically
        if len(self.checkpoint_metrics) % 10 == 0:
            self._flush_checkpoint_metrics()

    def _flush_training_metrics(self):
        """Flush training metrics to Parquet file."""
        if not self.training_metrics:
            return

        # Convert to DataFrame
        df = pd.DataFrame([asdict(m) for m in self.training_metrics])

        # Write to Parquet
        output_path = self.output_dir / "training_metrics.parquet"
        df.to_parquet(output_path, engine='pyarrow')

        self.logger.info(f"Flushed {len(self.training_metrics)} training metrics to {output_path}")

    def _flush_checkpoint_metrics(self):
        """Flush checkpoint metrics to Parquet file."""
        if not self.checkpoint_metrics:
            return

        # Convert to DataFrame
        df = pd.DataFrame([asdict(m) for m in self.checkpoint_metrics])

        # Write to Parquet
        output_path = self.output_dir / "checkpoint_metrics.parquet"
        df.to_parquet(output_path, engine='pyarrow')

        self.logger.info(f"Flushed {len(self.checkpoint_metrics)} checkpoint metrics to {output_path}")

    def _flush_correlation_metrics(self):
        """Flush event correlation metrics to Parquet file."""
        if not self.event_correlations:
            return

        # Flatten correlation data
        correlation_data = []
        for event_type, correlations in self.event_correlations.items():
            for corr in correlations:
                flat_corr = {
                    'event_type': corr['event_type'],
                    'iteration': corr['iteration'],
                    'nash_conv': corr['nash_conv'],
                    'magnitude': corr['magnitude']
                }
                # Add correlation fields
                for corr_name, corr_value in corr['correlations'].items():
                    flat_corr[corr_name] = corr_value
                correlation_data.append(flat_corr)

        if correlation_data:
            df = pd.DataFrame(correlation_data)
            output_path = self.output_dir / "event_correlations.parquet"
            df.to_parquet(output_path, engine='pyarrow')
            self.logger.info(f"Flushed {len(correlation_data)} correlation events to {output_path}")

    def flush_to_disk(self):
        """Flush all metrics to disk."""
        self._flush_training_metrics()
        self._flush_checkpoint_metrics()
        self._flush_correlation_metrics()

        # Save metadata
        metadata = {
            'total_training_metrics': len(self.training_metrics),
            'total_checkpoint_metrics': len(self.checkpoint_metrics),
            'event_types': list(self.event_correlations.keys()),
            'output_directory': str(self.output_dir)
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"All diagnostics flushed to {self.output_dir}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the current run.

        Returns:
            Dictionary with summary statistics
        """
        if not self.training_metrics:
            return {}

        df = pd.DataFrame([asdict(m) for m in self.training_metrics])

        summary = {
            'run_info': {
                'run_id': df['run_id'].iloc[0] if not df.empty else 'unknown',
                'algorithm': df['algorithm'].iloc[0] if not df.empty else 'unknown',
                'game': df['game'].iloc[0] if not df.empty else 'unknown',
                'seed': df['seed'].iloc[0] if not df.empty else 'unknown',
                'total_iterations': len(df),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()) / 3600 if len(df) > 1 else 0
            },
            'training_dynamics': {
                'final_nash_conv': df['nash_conv'].iloc[-1] if not df.empty else float('inf'),
                'final_exploitability': df['exploitability'].iloc[-1] if not df.empty else float('inf'),
                'best_nash_conv': df['nash_conv'].min() if not df.empty else float('inf'),
                'best_exploitability': df['exploitability'].min() if not df.empty else float('inf'),
                'nash_conv_spikes': df['nash_conv_spike'].sum() if not df.empty else 0
            },
            'gradient_analysis': {
                'mean_gradient_norm': df['total_gradient_norm'].mean() if not df.empty else 0,
                'max_gradient_norm': df['total_gradient_norm'].max() if not df.empty else 0,
                'total_clipping_events': df['clipping_events'].sum() if not df.empty else 0
            },
            'computational_analysis': {
                'total_parameters': df['total_params'].iloc[0] if not df.empty else 0,
                'mean_wall_clock_time': df['wall_clock_time'].mean() if not df.empty else 0,
                'total_training_time': df['wall_clock_time'].sum() if not df.empty else 0
            }
        }

        return summary