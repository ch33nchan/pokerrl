"""Logging utilities for the Dual RL Poker project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                console: bool = True) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Specialized logger for experiment tracking."""

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create main log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Setup structured logging
        self.logger = setup_logging(
            log_level="INFO",
            log_file=str(self.log_file),
            console=True
        )

        # Initialize metrics storage
        self.metrics_log = []
        self.start_time = datetime.now()

    def log_experiment_start(self, config: dict):
        """Log experiment start.

        Args:
            config: Experiment configuration
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("=" * 60)

        # Log configuration
        self.logger.info("Configuration:")
        self.logger.info(json.dumps(config, indent=2))

        # Save configuration
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Configuration saved to {config_file}")

    def log_iteration(self, iteration: int, metrics: dict):
        """Log iteration metrics.

        Args:
            iteration: Current iteration
            metrics: Metrics dictionary
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()

        # Add to metrics log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'elapsed_time': elapsed,
            **metrics
        }
        self.metrics_log.append(log_entry)

        # Log to console
        self.logger.info(
            f"Iteration {iteration:4d} | "
            f"Time: {elapsed:7.1f}s | "
            f"Loss: {metrics.get('loss', 'N/A'):8.4f} | "
            f"Exploitability: {metrics.get('exploitability', 'N/A'):8.4f} | "
            f"Buffer: {metrics.get('buffer_size', 'N/A'):6d}"
        )

        # Log detailed metrics
        if 'extra_metrics' in metrics:
            for key, value in metrics['extra_metrics'].items():
                self.logger.debug(f"  {key}: {value}")

    def log_evaluation(self, iteration: int, eval_metrics: dict):
        """Log evaluation results.

        Args:
            iteration: Current iteration
            eval_metrics: Evaluation metrics
        """
        self.logger.info(f"Evaluation at iteration {iteration}:")
        self.logger.info(f"  Exploitability: {eval_metrics.get('exploitability', 'N/A'):.4f}")
        self.logger.info(f"  NashConv: {eval_metrics.get('nash_conv', 'N/A'):.4f}")
        self.logger.info(f"  MC Mean Reward: {eval_metrics.get('mc_mean_reward', 'N/A'):.4f}")
        self.logger.info(f"  MC Std Reward: {eval_metrics.get('mc_std_reward', 'N/A'):.4f}")

    def log_experiment_end(self, final_results: dict):
        """Log experiment completion.

        Args:
            final_results: Final experiment results
        """
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()

        self.logger.info("=" * 60)
        self.logger.info(f"Experiment completed: {self.experiment_name}")
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info("=" * 60)

        # Log final results
        self.logger.info("Final Results:")
        for key, value in final_results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

        # Save metrics log
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2, default=str)
        self.logger.info(f"Metrics saved to {metrics_file}")

        # Save final results
        results_file = self.log_dir / f"{self.experiment_name}_final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        self.logger.info(f"Final results saved to {results_file}")

    def log_error(self, error: Exception, context: str = ""):
        """Log error with context.

        Args:
            error: Exception that occurred
            context: Context description
        """
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.exception("Full traceback:")

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger.

        Returns:
            Logger instance
        """
        return self.logger


def get_experiment_logger(experiment_name: str, log_dir: str = "logs") -> ExperimentLogger:
    """Create an experiment logger.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, log_dir)