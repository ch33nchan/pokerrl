"""
Universal Diagnostics Logging for Deep CFR Experiments

This module provides comprehensive diagnostics logging for all training runs,
including gradient norms, policy KL divergence, advantage statistics, and
performance metrics.
"""

import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from collections import defaultdict
import logging

class DiagnosticsLogger:
    """Universal diagnostics logger for Deep CFR training runs."""

    def __init__(self, run_id: str, log_dir: str = "diagnostics"):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize data storage
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()

        # Setup file logging
        self.log_file = self.log_dir / f"{run_id}.jsonl"
        self.parquet_file = self.log_dir / f"{run_id}.parquet"

        # Coverage tracking
        self.coverage = {
            "gradient_norms": False,
            "advantage_stats": False,
            "policy_kl": False,
            "wall_clock": False,
            "loss_values": False,
            "learning_rates": False
        }

        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"diagnostics.{run_id}")

    def log_training_step(self, iteration: int, metrics: Dict[str, Any]):
        """Log metrics for a single training step."""

        timestamp = time.time() - self.start_time
        wall_clock = timestamp

        entry = {
            "run_id": self.run_id,
            "iteration": iteration,
            "timestamp": timestamp,
            "wall_clock_s": wall_clock,
            **metrics
        }

        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=np.float32) + '\n')

        # Store in memory
        for key, value in metrics.items():
            self.metrics[key].append(value)

        # Update coverage
        self._update_coverage(metrics)

        # Log important metrics
        if iteration % 50 == 0:
            self.logger.info(f"Iteration {iteration}: {metrics}")

    def log_gradient_norms(self, iteration: int, model: torch.nn.Module):
        """Log gradient norms for all model parameters."""
        total_norm = 0.0
        param_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[f"grad_norm_{name}"] = param_norm
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5
        param_norms["grad_norm_total"] = total_norm

        self.log_training_step(iteration, param_norms)
        self.coverage["gradient_norms"] = True

    def log_advantage_stats(self, iteration: int, advantages: torch.Tensor):
        """Log statistics for advantage values."""
        advantages_np = advantages.detach().cpu().numpy()

        stats = {
            "advantage_mean": float(np.mean(advantages_np)),
            "advantage_std": float(np.std(advantages_np)),
            "advantage_min": float(np.min(advantages_np)),
            "advantage_max": float(np.max(advantages_np)),
            "advantage_q25": float(np.percentile(advantages_np, 25)),
            "advantage_q75": float(np.percentile(advantages_np, 75)),
            "advantage_count": len(advantages_np)
        }

        self.log_training_step(iteration, stats)
        self.coverage["advantage_stats"] = True

    def log_policy_kl(self, iteration: int,
                      old_policy: Dict[str, torch.Tensor],
                      new_policy: Dict[str, torch.Tensor],
                      infoset_list: List[str]):
        """Log KL divergence between old and new policies."""
        kl_divs = []

        for infoset in infoset_list:
            if infoset in old_policy and infoset in new_policy:
                old_p = old_policy[infoset]
                new_p = new_policy[infoset]

                # KL divergence KL(old || new)
                kl = torch.sum(old_p * torch.log((old_p + 1e-8) / (new_p + 1e-8)))
                kl_divs.append(kl.item())

        if kl_divs:
            stats = {
                "policy_kl_mean": float(np.mean(kl_divs)),
                "policy_kl_std": float(np.std(kl_divs)),
                "policy_kl_max": float(np.max(kl_divs)),
                "policy_kl_count": len(kl_divs)
            }

            self.log_training_step(iteration, stats)
            self.coverage["policy_kl"] = True

    def log_loss_values(self, iteration: int,
                       regret_loss: float,
                       policy_loss: float,
                       additional_losses: Optional[Dict[str, float]] = None):
        """Log training loss values."""
        losses = {
            "regret_loss": regret_loss,
            "policy_loss": policy_loss,
            "total_loss": regret_loss + policy_loss
        }

        if additional_losses:
            losses.update(additional_losses)

        self.log_training_step(iteration, losses)
        self.coverage["loss_values"] = True

    def log_learning_rates(self, iteration: int,
                          regret_lr: float,
                          policy_lr: float):
        """Log current learning rates."""
        lrs = {
            "regret_lr": regret_lr,
            "policy_lr": policy_lr
        }

        self.log_training_step(iteration, lrs)
        self.coverage["learning_rates"] = True

    def log_evaluation_metrics(self, iteration: int,
                              exploitability: float,
                              nashconv: Optional[float] = None):
        """Log evaluation metrics."""
        metrics = {
            "exploitability": exploitability,
            "eval_iteration": iteration
        }

        if nashconv is not None:
            metrics["nashconv"] = nashconv

        self.log_training_step(iteration, metrics)
        self.coverage["wall_clock"] = True

    def log_memory_usage(self, iteration: int):
        """Log memory usage statistics."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            # System memory approximation
            import psutil
            memory_allocated = psutil.Process().memory_info().rss / 1024**3  # GB
            memory_reserved = memory_allocated

        metrics = {
            "memory_gb_allocated": memory_allocated,
            "memory_gb_reserved": memory_reserved
        }

        self.log_training_step(iteration, metrics)

    def detect_spikes(self, metric_name: str, threshold: float = 2.0) -> List[int]:
        """Detect spikes in a given metric using z-score threshold."""
        if metric_name not in self.metrics:
            return []

        values = np.array(self.metrics[metric_name])
        if len(values) < 10:
            return []

        # Compute rolling statistics
        window = min(20, len(values) // 4)
        rolling_mean = pd.Series(values).rolling(window=window).mean().fillna(method='bfill').values
        rolling_std = pd.Series(values).rolling(window=window).std().fillna(method='bfill').values

        # Detect spikes
        z_scores = np.abs((values - rolling_mean) / (rolling_std + 1e-8))
        spike_indices = np.where(z_scores > threshold)[0].tolist()

        return spike_indices

    def get_coverage_report(self) -> Dict[str, bool]:
        """Get report of what diagnostics have been collected."""
        return self.coverage.copy()

    def save_to_parquet(self):
        """Save all metrics to parquet file for efficient analysis."""
        if not self.metrics:
            return

        # Convert to DataFrame
        df_data = []
        for i in range(max(len(v) for v in self.metrics.values())):
            row = {"iteration": i}
            for key, values in self.metrics.items():
                if i < len(values):
                    row[key] = values[i]
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_parquet(self.parquet_file, index=False)

        self.logger.info(f"Saved {len(df)} diagnostic entries to {self.parquet_file}")

    def finalize(self):
        """Finalize logging and save all data."""
        self.save_to_parquet()

        # Save coverage report
        coverage_file = self.log_dir / f"{run_id}_coverage.json"
        with open(coverage_file, 'w') as f:
            json.dump(self.get_coverage_report(), f, indent=2)

        self.logger.info(f"Diagnostics finalized for {self.run_id}")


class EventAnalyzer:
    """Analyze events and correlations in diagnostics data."""

    def __init__(self, diagnostics_df: pd.DataFrame):
        self.df = diagnostics_df

    def analyze_exploitability_spikes(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Analyze exploitability spikes and their leading indicators."""
        if "exploitability" not in self.df.columns:
            return {"error": "exploitability not found in diagnostics"}

        # Detect spikes
        exploit_values = self.df["exploitability"].values
        spike_indices = self._detect_spikes(exploit_values, threshold)

        if not spike_indices:
            return {"spikes_found": 0}

        # Analyze leading indicators for each spike
        spike_analysis = []
        for spike_idx in spike_indices:
            window_before = max(0, spike_idx - 20)
            window_after = min(len(self.df), spike_idx + 10)

            spike_data = {
                "spike_iteration": spike_idx,
                "spike_value": exploit_values[spike_idx],
                "window_before": window_before,
                "window_after": window_after
            }

            # Analyze leading indicators
            for col in self.df.columns:
                if col in ["iteration", "exploitability"]:
                    continue

                values_before = self.df[col].iloc[window_before:spike_idx].values
                values_spike = self.df[col].iloc[spike_idx]

                if len(values_before) > 0 and not np.isnan(values_before).all():
                    trend = np.polyfit(range(len(values_before)), values_before, 1)[0]
                    spike_analysis.append({
                        "metric": col,
                        "pre_spike_trend": trend,
                        "spike_value": values_spike,
                        "correlation": np.corrcoef(values_before, [spike_idx - i for i in range(len(values_before))])[0,1] if len(values_before) > 1 else 0.0
                    })

            spike_analysis.append(spike_data)

        return {
            "spikes_found": len(spike_indices),
            "spike_iterations": spike_indices,
            "spike_analysis": spike_analysis
        }

    def _detect_spikes(self, values: np.ndarray, threshold: float) -> List[int]:
        """Detect spikes using z-score threshold."""
        if len(values) < 10:
            return []

        # Compute rolling statistics
        window = min(20, len(values) // 4)
        rolling_mean = pd.Series(values).rolling(window=window).mean().fillna(method='bfill').values
        rolling_std = pd.Series(values).rolling(window=window).std().fillna(method='bfill').values

        # Detect spikes
        z_scores = np.abs((values - rolling_mean) / (rolling_std + 1e-8))
        spike_indices = np.where(z_scores > threshold)[0].tolist()

        return spike_indices

    def compute_correlation_matrix(self, metrics: List[str]) -> pd.DataFrame:
        """Compute correlation matrix for specified metrics."""
        available_metrics = [m for m in metrics if m in self.df.columns]

        if len(available_metrics) < 2:
            return pd.DataFrame()

        return self.df[available_metrics].corr()

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for all metrics."""
        summary = {}

        for col in self.df.columns:
            if col in ["iteration", "timestamp", "run_id"]:
                continue

            values = self.df[col].dropna()
            if len(values) > 0:
                summary[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q50": float(values.quantile(0.50)),
                    "q75": float(values.quantile(0.75)),
                    "count": len(values)
                }

        return summary