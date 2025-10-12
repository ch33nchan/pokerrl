"""Unified metrics logging utility for Dual RL Poker experiments.

This module provides a standardized way to log metrics across all algorithms
and experiments, with support for both CSV and TensorBoard logging.
"""

import csv
import os
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np


class MetricsLogger:
    """Unified metrics logger for consistent logging across experiments.

    Supports:
    - CSV logging for structured data storage
    - TensorBoard logging for visualization
    - Automatic file management and organization
    - Metric aggregation and statistics
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        algorithm: str,
        log_format: str = "csv",
        tensorboard_dir: Optional[str] = None,
        flush_interval: int = 10,
    ):
        """Initialize metrics logger.

        Args:
            output_path: Base directory for logging outputs
            experiment_name: Name of the experiment
            algorithm: Algorithm name (e.g., "armac", "nfsp", "psro")
            log_format: Logging format ("csv", "tensorboard", or "both")
            tensorboard_dir: Directory for TensorBoard logs (if using TensorBoard)
            flush_interval: Number of writes between automatic flushes
        """
        self.output_path = Path(output_path)
        self.experiment_name = experiment_name
        self.algorithm = algorithm
        self.log_format = log_format
        self.flush_interval = flush_interval
        self.write_count = 0

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers = ["timestamp", "iteration", "metric", "value", "metadata"]

        if log_format in ["csv", "both"]:
            self._init_csv_logging()

        # Initialize TensorBoard logging
        self.tensorboard_writer = None
        if log_format in ["tensorboard", "both"] and tensorboard_dir:
            self._init_tensorboard_logging(tensorboard_dir)

        # Metric aggregation
        self.metric_buffers: Dict[str, List[float]] = {}
        self.aggregation_window = 100

        self.logger_info = {
            "start_time": time.time(),
            "total_metrics": 0,
            "unique_metrics": set(),
        }

    def _init_csv_logging(self):
        """Initialize CSV logging."""
        csv_filename = f"{self.experiment_name}_{self.algorithm}_metrics.csv"
        self.csv_file_path = self.output_path / csv_filename

        # Check if file exists to determine if we need headers
        file_exists = self.csv_file_path.exists()

        self.csv_file = open(self.csv_file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write headers if new file
        if not file_exists:
            self.csv_writer.writerow(self.csv_headers)
            self.csv_file.flush()

    def _init_tensorboard_logging(self, tensorboard_dir: str):
        """Initialize TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = Path(tensorboard_dir) / self.experiment_name / self.algorithm
            tb_dir.mkdir(parents=True, exist_ok=True)

            self.tensorboard_writer = SummaryWriter(
                log_dir=str(tb_dir), flush_secs=30, max_queue=1000
            )
        except ImportError:
            print("TensorBoard not available. Using CSV logging only.")
            self.log_format = "csv"

    def log_metric(
        self,
        metric: str,
        value: Union[float, int, np.ndarray],
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aggregate: bool = True,
    ):
        """Log a metric value.

        Args:
            metric: Name of the metric
            value: Metric value (float, int, or numpy array)
            iteration: Training iteration (defaults to auto-increment)
            metadata: Additional metadata dictionary
            aggregate: Whether to include in running statistics
        """
        timestamp = time.time()

        # Handle array values
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value.item())
            else:
                # Log each element of array separately
                for i, v in enumerate(value.flatten()):
                    self.log_metric(
                        f"{metric}_{i}", float(v), iteration, metadata, aggregate
                    )
                return

        # Convert to float
        try:
            value = float(value)
        except (ValueError, TypeError):
            print(
                f"Warning: Could not convert metric '{metric}' value to float: {value}"
            )
            return

        # Auto-increment iteration if not provided
        if iteration is None:
            iteration = self.logger_info["total_metrics"]

        # Prepare metadata string
        metadata_str = ""
        if metadata:
            metadata_str = str(metadata)

        # Update logger info
        self.logger_info["total_metrics"] += 1
        self.logger_info["unique_metrics"].add(metric)

        # Log to CSV
        if self.csv_writer and self.log_format in ["csv", "both"]:
            self.csv_writer.writerow(
                [timestamp, iteration, metric, value, metadata_str]
            )

        # Log to TensorBoard
        if self.tensorboard_writer and self.log_format in ["tensorboard", "both"]:
            self.tensorboard_writer.add_scalar(metric, value, iteration)

        # Update aggregation buffers
        if aggregate:
            if metric not in self.metric_buffers:
                self.metric_buffers[metric] = []

            self.metric_buffers[metric].append(value)

            # Keep buffer size limited
            if len(self.metric_buffers[metric]) > self.aggregation_window:
                self.metric_buffers[metric] = self.metric_buffers[metric][
                    -self.aggregation_window :
                ]

        # Periodic flush
        self.write_count += 1
        if self.write_count % self.flush_interval == 0:
            self.flush()

    def log_metrics_batch(
        self,
        metrics: Dict[str, Union[float, int, np.ndarray]],
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            iteration: Training iteration
            metadata: Additional metadata dictionary
        """
        for metric, value in metrics.items():
            self.log_metric(metric, value, iteration, metadata)

    def log_training_step(
        self,
        iteration: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
        runtime: Optional[float] = None,
    ):
        """Log a complete training step.

        Args:
            iteration: Training iteration number
            losses: Dictionary of loss values
            metrics: Additional metrics dictionary
            runtime: Runtime for this step in seconds
        """
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log_metric(f"loss/{loss_name}", loss_value, iteration)

        # Log additional metrics
        if metrics:
            for metric_name, metric_value in metrics.items():
                self.log_metric(f"metrics/{metric_name}", metric_value, iteration)

        # Log runtime
        if runtime is not None:
            self.log_metric("runtime/step_time", runtime, iteration)

    def log_evaluation(
        self,
        iteration: int,
        exploitability: float,
        nash_conv: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log evaluation results.

        Args:
            iteration: Evaluation iteration
            exploitability: Exploitability value
            nash_conv: Nash convergence value
            additional_metrics: Additional evaluation metrics
        """
        self.log_metric("eval/exploitability", exploitability, iteration)

        if nash_conv is not None:
            self.log_metric("eval/nash_conv", nash_conv, iteration)

        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                self.log_metric(f"eval/{metric_name}", metric_value, iteration)

    def get_metric_statistics(
        self, metric: str, window: Optional[int] = None
    ) -> Dict[str, float]:
        """Get running statistics for a metric.

        Args:
            metric: Metric name
            window: Window size for statistics (uses default if None)

        Returns:
            Dictionary with statistics (mean, std, min, max, count)
        """
        if metric not in self.metric_buffers or len(self.metric_buffers[metric]) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        values = self.metric_buffers[metric]
        if window is not None and window < len(values):
            values = values[-window:]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def get_all_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all logged metrics."""
        stats = {}
        for metric in self.metric_buffers:
            stats[metric] = self.get_metric_statistics(metric)
        return stats

    def flush(self):
        """Flush all pending writes."""
        if self.csv_file:
            self.csv_file.flush()

        if self.tensorboard_writer:
            self.tensorboard_writer.flush()

    def close(self):
        """Close the logger and clean up resources."""
        self.flush()

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None

        # Log summary statistics
        self._log_summary_statistics()

    def _log_summary_statistics(self):
        """Log summary statistics to a separate file."""
        summary_path = (
            self.output_path / f"{self.experiment_name}_{self.algorithm}_summary.txt"
        )

        with open(summary_path, "w") as f:
            f.write(f"Experiment Summary: {self.experiment_name}\n")
            f.write(f"Algorithm: {self.algorithm}\n")
            f.write(f"Total Metrics Logged: {self.logger_info['total_metrics']}\n")
            f.write(f"Unique Metrics: {len(self.logger_info['unique_metrics'])}\n")
            f.write(
                f"Duration: {time.time() - self.logger_info['start_time']:.2f} seconds\n"
            )
            f.write("\nMetric Statistics:\n")

            for metric in sorted(self.metric_buffers.keys()):
                stats = self.get_metric_statistics(metric)
                f.write(
                    f"  {metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                    f"min={stats['min']:.6f}, max={stats['max']:.6f}, count={stats['count']}\n"
                )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ExperimentLogger:
    """High-level experiment logger that manages multiple algorithm loggers."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "results",
        log_format: str = "csv",
        tensorboard_dir: Optional[str] = "logs/tensorboard",
    ):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Base output directory
            log_format: Logging format
            tensorboard_dir: TensorBoard log directory
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.log_format = log_format
        self.tensorboard_dir = tensorboard_dir
        self.algorithm_loggers: Dict[str, MetricsLogger] = {}

    def get_algorithm_logger(self, algorithm: str) -> MetricsLogger:
        """Get or create a logger for a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            MetricsLogger instance for the algorithm
        """
        if algorithm not in self.algorithm_loggers:
            self.algorithm_loggers[algorithm] = MetricsLogger(
                output_path=self.output_dir,
                experiment_name=self.experiment_name,
                algorithm=algorithm,
                log_format=self.log_format,
                tensorboard_dir=self.tensorboard_dir,
            )

        return self.algorithm_loggers[algorithm]

    def close_all(self):
        """Close all algorithm loggers."""
        for logger in self.algorithm_loggers.values():
            logger.close()
        self.algorithm_loggers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()
