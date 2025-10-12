"""Comprehensive diagnostics logging for training monitoring.

Tracks gradient norms, advantage statistics, policy KL divergence,
clipping events, and timing metrics with Parquet-compatible output.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import json
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class TrainingDiagnostics:
    """Comprehensive diagnostics logger for training monitoring."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        seed: Optional[int] = None,
        output_root: Union[str, Path] = "results/diagnostics",
    ):
        """Initialize diagnostics logger.

        Args:
            experiment_name: Optional descriptive experiment label used for directory naming.
            seed: Optional random seed for run-specific logging separation.
            output_root: Root directory to save diagnostic data.
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        if experiment_name is not None:
            safe_name = experiment_name.strip().lower().replace(" ", "_")
        else:
            safe_name = "run"

        if seed is not None:
            subdir = f"{safe_name}_seed_{seed}"
        else:
            subdir = safe_name

        self.output_dir = output_root / subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = self._generate_run_id()
        self.start_time = time.time()
        self.checkpoint_data = []

        # Diagnostic tracking
        self.gradient_norms = []
        self.advantage_stats = []
        self.policy_kls = []
        self.clipping_events = []
        self.timing_data = []
        self.iteration_logs = []
        self.evaluation_logs = []

        logger.info(f"Initialized diagnostics with run_id: {self.run_id}")

    def _generate_run_id(self) -> str:
        """Generate unique run ID based on timestamp and config hash."""
        timestamp = int(time.time())
        config_hash = hashlib.md5(f"{timestamp}".encode()).hexdigest()[:8]
        return f"run_{timestamp}_{config_hash}"

    def log_gradient_norms(self, model: torch.nn.Module, iteration: int):
        """Log gradient norms for all parameters.

        Args:
            model: Model to analyze gradients
            iteration: Current training iteration
        """
        total_norm = 0.0
        param_norms = {}
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[name] = param_norm
                total_norm += param_norm ** 2

                # Extract layer name
                layer_name = name.split('.')[0]
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm)

        total_norm = total_norm ** 0.5

        # Store gradient norm data
        gradient_data = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "total_gradient_norm": total_norm,
            "param_norms_mean": np.mean(list(param_norms.values())) if param_norms else 0.0,
            "param_norms_std": np.std(list(param_norms.values())) if param_norms else 0.0,
            "layer_norms": layer_norms
        }

        self.gradient_norms.append(gradient_data)

    def log_advantage_statistics(self, advantages: torch.Tensor, iteration: int):
        """Log advantage value statistics.

        Args:
            advantages: Advantage values tensor
            iteration: Current training iteration
        """
        if advantages.numel() == 0:
            return

        adv_np = advantages.detach().cpu().numpy()

        stats = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "advantage_mean": float(np.mean(adv_np)),
            "advantage_std": float(np.std(adv_np)),
            "advantage_min": float(np.min(adv_np)),
            "advantage_max": float(np.max(adv_np)),
            "advantage_25th": float(np.percentile(adv_np, 25)),
            "advantage_50th": float(np.percentile(adv_np, 50)),
            "advantage_75th": float(np.percentile(adv_np, 75)),
            "advantage_count": int(adv_np.size),
            "positive_ratio": float(np.mean(adv_np > 0))
        }

        self.advantage_stats.append(stats)

    def log_policy_kl(self, current_policy: torch.Tensor,
                      previous_policy: torch.Tensor,
                      legal_actions_mask: torch.Tensor,
                      iteration: int,
                      infoset_key: Optional[str] = None):
        """Log KL divergence between current and previous policies.

        Args:
            current_policy: Current policy tensor
            previous_policy: Previous policy tensor
            legal_actions_mask: Legal actions mask
            iteration: Current training iteration
            infoset_key: Optional information state identifier
        """
        if current_policy.shape != previous_policy.shape:
            logger.warning("Policy shape mismatch, skipping KL computation")
            return

        # Apply legal actions mask
        curr_masked = current_policy * legal_actions_mask
        prev_masked = previous_policy * legal_actions_mask

        # Normalize to create valid probability distributions
        curr_sum = curr_masked.sum(dim=-1, keepdim=True)
        prev_sum = prev_masked.sum(dim=-1, keepdim=True)

        curr_norm = curr_masked / (curr_sum + 1e-8)
        prev_norm = prev_masked / (prev_sum + 1e-8)

        # Compute KL divergence
        kl_per_sample = torch.sum(prev_norm * (torch.log(prev_norm + 1e-8) - torch.log(curr_norm + 1e-8)), dim=-1)

        # Use correction=0 to avoid warnings when there is only a single sample.
        kl_stats = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "kl_mean": float(torch.mean(kl_per_sample)),
            "kl_std": float(torch.std(kl_per_sample, correction=0)),
            "kl_max": float(torch.max(kl_per_sample)),
            "kl_min": float(torch.min(kl_per_sample)),
            "kl_sum": float(torch.sum(kl_per_sample)),
            "num_samples": int(kl_per_sample.numel()),
            "infoset_key": infoset_key
        }

        self.policy_kls.append(kl_stats)

    def log_clipping_event(self, grad_norm: float, threshold: float, iteration: int):
        """Log gradient clipping event.

        Args:
            grad_norm: Gradient norm before clipping
            threshold: Clipping threshold
            iteration: Current training iteration
        """
        if grad_norm > threshold:
            clipping_event = {
                "run_id": self.run_id,
                "iteration": iteration,
                "timestamp": time.time(),
                "grad_norm": grad_norm,
                "threshold": threshold,
                "clipping_applied": True
            }
            self.clipping_events.append(clipping_event)

    def log_timing(self, phase: str, duration: float, iteration: int):
        """Log timing information for different training phases.

        Args:
            phase: Training phase name (e.g., "forward", "backward", "data_collection")
            duration: Duration in seconds
            iteration: Current training iteration
        """
        timing_info = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "phase": phase,
            "duration_seconds": duration,
            "wall_clock_elapsed": time.time() - self.start_time
        }

        self.timing_data.append(timing_info)

    def log_iteration(self, training_state: Any):
        """Log a training iteration state.

        Args:
            training_state: Training state object or dictionary.
        """
        if hasattr(training_state, "to_dict"):
            payload = training_state.to_dict()
        elif isinstance(training_state, dict):
            payload = dict(training_state)
        else:
            raise TypeError("training_state must be dict-like or expose to_dict()")

        record = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            **payload,
        }
        self.iteration_logs.append(record)

        iteration_index = record.get("iteration", len(self.iteration_logs))

        grad_norm = record.get("gradient_norm")
        if grad_norm is not None:
            grad_entry = {
                "run_id": self.run_id,
                "iteration": iteration_index,
                "timestamp": record["timestamp"],
                "total_gradient_norm": float(grad_norm),
                "param_norms_mean": float(grad_norm),
                "param_norms_std": 0.0,
                "layer_norms": {"aggregate": [float(grad_norm)]},
            }
            self.gradient_norms.append(grad_entry)

        avg_regret_norm = record.get("avg_regret_norm")
        if avg_regret_norm is not None:
            advantage_entry = {
                "run_id": self.run_id,
                "iteration": iteration_index,
                "timestamp": record["timestamp"],
                "advantage_mean": float(avg_regret_norm),
                "advantage_std": 0.0,
                "advantage_min": float(avg_regret_norm),
                "advantage_max": float(avg_regret_norm),
                "advantage_25th": float(avg_regret_norm),
                "advantage_50th": float(avg_regret_norm),
                "advantage_75th": float(avg_regret_norm),
                "advantage_count": 1,
                "positive_ratio": 1.0 if avg_regret_norm >= 0 else 0.0,
            }
            self.advantage_stats.append(advantage_entry)

        wall_time = record.get("wall_time")
        if wall_time is not None:
            timing_entry = {
                "run_id": self.run_id,
                "iteration": iteration_index,
                "timestamp": time.time(),
                "phase": "iteration",
                "duration_seconds": float(wall_time),
                "wall_clock_elapsed": time.time() - self.start_time,
            }
            self.timing_data.append(timing_entry)

    def log_evaluation(self, iteration: int, metrics: Dict[str, Any], elapsed_time: float):
        """Log evaluation metrics for an iteration."""
        record = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "wall_clock_elapsed": elapsed_time,
        }
        if metrics:
            record.update(metrics)
        self.evaluation_logs.append(record)

    def log_checkpoint_metrics(self, iteration: int,
                              nash_conv: float,
                              exploitability: float,
                              additional_metrics: Optional[Dict[str, Any]] = None):
        """Log checkpoint evaluation metrics.

        Args:
            iteration: Current iteration
            nash_conv: NashConv value
            exploitability: Exploitability value
            additional_metrics: Additional metrics to log
        """
        checkpoint_info = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": time.time(),
            "nash_conv": nash_conv,
            "exploitability": exploitability,
            "wall_clock_elapsed": time.time() - self.start_time
        }

        if additional_metrics:
            checkpoint_info.update(additional_metrics)

        self.checkpoint_data.append(checkpoint_info)

    def flush_to_disk(self):
        """Flush all diagnostic data to disk in Parquet format."""
        self._flush_gradient_norms()
        self._flush_advantage_stats()
        self._flush_policy_kls()
        self._flush_clipping_events()
        self._flush_timing_data()
        self._flush_iteration_logs()
        self._flush_evaluation_logs()
        self._flush_checkpoint_data()

        logger.info(f"Flushed diagnostics data to {self.output_dir}")

    def _flush_gradient_norms(self):
        """Flush gradient norm data to Parquet."""
        if not self.gradient_norms:
            return

        # Flatten layer norms for tabular format
        flattened_data = []
        for data in self.gradient_norms:
            base_record = {k: v for k, v in data.items() if k != "layer_norms"}
            for layer_name, norms in data["layer_norms"].items():
                for i, norm in enumerate(norms):
                    record = base_record.copy()
                    record.update({
                        "layer_name": layer_name,
                        "param_index": i,
                        "param_norm": norm
                    })
                    flattened_data.append(record)

        df = pd.DataFrame(flattened_data)
        output_path = self.output_dir / "gradient_norms.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_advantage_stats(self):
        """Flush advantage statistics to Parquet."""
        if not self.advantage_stats:
            return

        df = pd.DataFrame(self.advantage_stats)
        output_path = self.output_dir / "advantage_stats.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_policy_kls(self):
        """Flush policy KL divergence data to Parquet."""
        if not self.policy_kls:
            return

        df = pd.DataFrame(self.policy_kls)
        output_path = self.output_dir / "policy_kls.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_clipping_events(self):
        """Flush clipping events to Parquet."""
        if not self.clipping_events:
            return

        df = pd.DataFrame(self.clipping_events)
        output_path = self.output_dir / "clipping_events.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_timing_data(self):
        """Flush timing data to Parquet."""
        if not self.timing_data:
            return

        df = pd.DataFrame(self.timing_data)
        output_path = self.output_dir / "timing_data.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_iteration_logs(self):
        """Flush iteration logs to Parquet."""
        if not self.iteration_logs:
            return

        df = pd.DataFrame(self.iteration_logs)
        output_path = self.output_dir / "iteration_logs.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_evaluation_logs(self):
        """Flush evaluation logs to Parquet."""
        if not self.evaluation_logs:
            return

        df = pd.DataFrame(self.evaluation_logs)
        output_path = self.output_dir / "evaluation_logs.parquet"
        df.to_parquet(output_path, index=False)

    def _flush_checkpoint_data(self):
        """Flush checkpoint data to Parquet."""
        if not self.checkpoint_data:
            return

        df = pd.DataFrame(self.checkpoint_data)
        output_path = self.output_dir / "checkpoint_metrics.parquet"
        df.to_parquet(output_path, index=False)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all diagnostic data.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "run_id": self.run_id,
            "total_duration": time.time() - self.start_time,
            "num_checkpoints": len(self.checkpoint_data),
            "num_clipping_events": len(self.clipping_events)
        }

        if self.gradient_norms:
            norms = [d["total_gradient_norm"] for d in self.gradient_norms]
            summary["gradient_norm_mean"] = np.mean(norms)
            summary["gradient_norm_std"] = np.std(norms)
            summary["gradient_norm_max"] = np.max(norms)

        if self.advantage_stats:
            means = [d["advantage_mean"] for d in self.advantage_stats]
            summary["advantage_mean_trend"] = means[-1] - means[0] if len(means) > 1 else 0

        if self.policy_kls:
            kl_means = [d["kl_mean"] for d in self.policy_kls]
            summary["policy_kl_mean"] = np.mean(kl_means)
            summary["policy_kl_max"] = np.max(kl_means)

        if self.checkpoint_data:
            nash_convs = [d["nash_conv"] for d in self.checkpoint_data]
            exploitabilities = [d["exploitability"] for d in self.checkpoint_data]
            summary["final_nash_conv"] = nash_convs[-1] if nash_convs else None
            summary["final_exploitability"] = exploitabilities[-1] if exploitabilities else None
            summary["nash_conv_improvement"] = nash_convs[0] - nash_convs[-1] if len(nash_convs) > 1 else 0

        if self.iteration_logs:
            summary["num_iterations"] = len(self.iteration_logs)
            summary["final_loss"] = self.iteration_logs[-1].get("loss")

        if self.evaluation_logs:
            summary["num_evaluations"] = len(self.evaluation_logs)
            summary["best_exploitability"] = min(
                (log.get("exploitability") for log in self.evaluation_logs if log.get("exploitability") is not None),
                default=None,
            )

        return summary


class DiagnosticAnalyzer:
    """Analyzer for correlating diagnostic spikes with performance changes."""

    def __init__(self, diagnostics_dir: str = "results/diagnostics"):
        """Initialize analyzer.

        Args:
            diagnostics_dir: Directory containing diagnostic data
        """
        self.diagnostics_dir = Path(diagnostics_dir)

    def analyze_gradient_spikes(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Analyze gradient norm spikes and their correlation with NashConv changes.

        Args:
            threshold: Threshold for defining gradient spikes (in std deviations)

        Returns:
            Analysis results
        """
        try:
            grad_df = pd.read_parquet(self.diagnostics_dir / "gradient_norms.parquet")
            checkpoint_df = pd.read_parquet(self.diagnostics_dir / "checkpoint_metrics.parquet")

            if grad_df.empty or checkpoint_df.empty:
                return {"error": "No diagnostic data available"}

            # Identify gradient spikes
            mean_norm = grad_df["total_gradient_norm"].mean()
            std_norm = grad_df["total_gradient_norm"].std()
            spike_threshold = mean_norm + threshold * std_norm

            spikes = grad_df[grad_df["total_gradient_norm"] > spike_threshold]

            # Find nearest checkpoints after spikes
            spike_correlations = []
            for _, spike in spikes.iterrows():
                next_checkpoint = checkpoint_df[checkpoint_df["iteration"] > spike["iteration"]]
                if not next_checkpoint.empty:
                    next_cp = next_checkpoint.iloc[0]
                    nash_conv_change = self._get_nash_conv_change(
                        checkpoint_df, spike["iteration"], next_cp["iteration"]
                    )
                    spike_correlations.append({
                        "spike_iteration": spike["iteration"],
                        "spike_norm": spike["total_gradient_norm"],
                        "next_checkpoint_iteration": next_cp["iteration"],
                        "nash_conv_change": nash_conv_change,
                        "spike_magnitude": (spike["total_gradient_norm"] - mean_norm) / std_norm
                    })

            return {
                "total_spikes": len(spikes),
                "spike_threshold": spike_threshold,
                "correlations": spike_correlations,
                "avg_correlation": np.mean([c["nash_conv_change"] for c in spike_correlations]) if spike_correlations else 0
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_nash_conv_change(self, checkpoint_df: pd.DataFrame, start_iter: int, end_iter: int) -> float:
        """Get NashConv change between two iterations.

        Args:
            checkpoint_df: Checkpoint data
            start_iter: Starting iteration
            end_iter: Ending iteration

        Returns:
            NashConv change
        """
        start_data = checkpoint_df[checkpoint_df["iteration"] == start_iter]
        end_data = checkpoint_df[checkpoint_df["iteration"] == end_iter]

        if start_data.empty or end_data.empty:
            return 0.0

        return start_data.iloc[0]["nash_conv"] - end_data.iloc[0]["nash_conv"]

    def generate_diagnostic_report(self) -> str:
        """Generate comprehensive diagnostic analysis report.

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("Diagnostic Analysis Report")
        lines.append("=" * 50)

        # Gradient spike analysis
        gradient_analysis = self.analyze_gradient_spikes()
        if "error" not in gradient_analysis:
            lines.append(f"\nGradient Spike Analysis:")
            lines.append(f"  Total spikes: {gradient_analysis['total_spikes']}")
            lines.append(f"  Spike threshold: {gradient_analysis['spike_threshold']:.4f}")
            lines.append(f"  Average correlation with NashConv: {gradient_analysis['avg_correlation']:.6f}")
        else:
            lines.append(f"\nGradient Analysis: {gradient_analysis['error']}")

        # Summary statistics
        try:
            grad_df = pd.read_parquet(self.diagnostics_dir / "gradient_norms.parquet")
            if not grad_df.empty:
                lines.append(f"\nGradient Norm Statistics:")
                lines.append(f"  Mean: {grad_df['total_gradient_norm'].mean():.6f}")
                lines.append(f"  Std: {grad_df['total_gradient_norm'].std():.6f}")
                lines.append(f"  Max: {grad_df['total_gradient_norm'].max():.6f}")
                lines.append(f"  Min: {grad_df['total_gradient_norm'].min():.6f}")
        except:
            lines.append("\nGradient statistics: No data available")

        return "\n".join(lines)