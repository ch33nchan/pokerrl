"""Configuration loader utility for Dual RL Poker experiments.

This module provides a standardized way to load, validate, and manage
experiment configurations with support for inheritance, defaults, and
environment-specific overrides.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import logging


@dataclass
class ExperimentConfig:
    """Configuration for an entire experiment."""

    # Basic experiment settings
    name: str = "dual_rl_poker_default"
    seed: int = 42
    device: str = "cpu"
    log_level: str = "INFO"

    # Game settings
    game_name: str = "kuhn_poker"
    num_players: int = 2

    # Training settings
    iterations: int = 500
    eval_every: int = 25
    save_every: int = 100
    batch_size: int = 2048
    replay_window: int = 10
    buffer_size: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gradient_clip: float = 5.0

    # Network settings
    network_type: str = "mlp"
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    dropout: float = 0.0

    # Evaluation settings
    num_episodes: int = 1000
    head_to_head_episodes: int = 5000
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Algorithm-specific settings
    algorithms: dict = field(default_factory=dict)

    # Logging settings
    log_dir: str = "logs"
    save_dir: str = "results"
    tensorboard: bool = True
    print_frequency: int = 25
    save_frequency: int = 100

    # Reproducibility
    deterministic: bool = True
    torch_deterministic: bool = True
    benchmark: bool = False


@dataclass
class ARMACConfig:
    """Configuration specific to ARMAC algorithm."""

    buffer_size: int = 10000
    policy_replay_size: int = 5
    update_frequency: int = 1
    gamma: float = 0.99
    regret_weight: float = 0.1
    lambda_mode: str = "adaptive"  # "fixed" or "adaptive"
    lambda_alpha: float = 0.5
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    regret_lr: float = 1e-3


@dataclass
class NFSPConfig:
    """Configuration specific to NFSP algorithm."""

    eta: float = 0.1  # Anticipatory parameter
    alpha: float = 0.01  # RL learning rate
    beta: float = 0.01  # SL learning rate
    gamma: float = 0.99  # Discount factor
    buffer_size: int = 20000
    batch_size: int = 256
    policy_update_freq: int = 1
    sl_update_freq: int = 1
    hidden_dims: list = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1


@dataclass
class PSROConfig:
    """Configuration specific to PSRO algorithm."""

    population_size: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs_per_iteration: int = 100
    update_frequency: int = 10
    gamma: float = 0.99
    hidden_dims: list = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1


class ConfigLoader:
    """Configuration loader with inheritance and validation."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.logger = logging.getLogger(__name__)

        # Default configurations
        self.defaults = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default configurations."""
        # Experiment defaults
        self.defaults["experiment"] = asdict(ExperimentConfig())

        # Algorithm defaults
        self.defaults["armac"] = asdict(ARMACConfig())
        self.defaults["nfsp"] = asdict(NFSPConfig())
        self.defaults["psro"] = asdict(PSROConfig())

    def load_config(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        algorithm: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load and merge configuration.

        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            algorithm: Algorithm-specific configuration to load
            overrides: Override values

        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        config = self.defaults["experiment"].copy()

        # Load from file if provided
        if config_path:
            file_config = self._load_config_file(config_path)
            config = self._merge_configs(config, file_config)

        # Load from dictionary if provided
        if config_dict:
            config = self._merge_configs(config, config_dict)

        # Load algorithm-specific configuration
        if algorithm:
            if algorithm in self.defaults:
                alg_config = self.defaults[algorithm].copy()
                # Add under algorithms key
                if "algorithms" not in config:
                    config["algorithms"] = {}
                config["algorithms"][algorithm] = alg_config
            else:
                self.logger.warning(f"Unknown algorithm: {algorithm}")

        # Apply overrides
        if overrides:
            config = self._merge_configs(config, overrides)

        # Validate configuration
        self._validate_config(config)

        return config

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        # If relative path, look in config directory
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == ".json":
                    return json.load(f) or {}
                else:
                    raise ValueError(
                        f"Unsupported config file format: {config_path.suffix}"
                    )
        except Exception as e:
            raise RuntimeError(f"Error loading config file {config_path}: {e}")

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration values.

        Args:
            config: Configuration to validate
        """
        # Validate basic fields
        required_fields = ["name", "game_name", "iterations"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate value ranges
        if config["iterations"] <= 0:
            raise ValueError("iterations must be positive")

        if config["eval_every"] <= 0:
            raise ValueError("eval_every must be positive")

        if config["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")

        # Validate game name
        valid_games = ["kuhn_poker", "leduc_poker"]
        if config["game_name"] not in valid_games:
            self.logger.warning(f"Unknown game: {config['game_name']}")

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if config["device"] not in valid_devices:
            self.logger.warning(f"Unknown device: {config['device']}")

        # Validate algorithms if present
        if "algorithms" in config:
            for alg_name, alg_config in config["algorithms"].items():
                self._validate_algorithm_config(alg_name, alg_config)

    def _validate_algorithm_config(self, algorithm: str, config: Dict[str, Any]):
        """Validate algorithm-specific configuration.

        Args:
            algorithm: Algorithm name
            config: Algorithm configuration
        """
        if algorithm == "armac":
            if "lambda_mode" in config:
                if config["lambda_mode"] not in ["fixed", "adaptive"]:
                    raise ValueError(f"Invalid lambda_mode: {config['lambda_mode']}")

            if "lambda_alpha" in config:
                if not 0 <= config["lambda_alpha"] <= 10:
                    raise ValueError("lambda_alpha must be in [0, 10]")

        elif algorithm == "nfsp":
            if "eta" in config:
                if not 0 <= config["eta"] <= 1:
                    raise ValueError("eta must be in [0, 1]")

        elif algorithm == "psro":
            if "population_size" in config:
                if config["population_size"] <= 0:
                    raise ValueError("population_size must be positive")

    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file.

        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                if output_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif output_path.suffix.lower() == ".json":
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported output format: {output_path.suffix}")

            self.logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Error saving config to {output_path}: {e}")

    def get_config_template(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get a configuration template.

        Args:
            algorithm: Algorithm to include template for

        Returns:
            Configuration template
        """
        template = self.defaults["experiment"].copy()

        if algorithm and algorithm in self.defaults:
            template["algorithms"] = {algorithm: self.defaults[algorithm].copy()}

        return template

    def list_available_configs(self) -> list:
        """List available configuration files.

        Returns:
            List of configuration file names
        """
        if not self.config_dir.exists():
            return []

        config_files = []
        for file_path in self.config_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".yaml",
                ".yml",
                ".json",
            ]:
                config_files.append(str(file_path.relative_to(self.config_dir)))

        return sorted(config_files)


# Global config loader instance
_global_config_loader = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """Get global config loader instance.

    Args:
        config_dir: Configuration directory

    Returns:
        ConfigLoader instance
    """
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_dir)
    return _global_config_loader


def load_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load configuration using global config loader.

    Args:
        config_path: Path to configuration file
        config_dict: Configuration dictionary
        algorithm: Algorithm-specific configuration
        overrides: Override values
        config_dir: Configuration directory

    Returns:
        Loaded configuration
    """
    loader = get_config_loader(config_dir)
    return loader.load_config(config_path, config_dict, algorithm, overrides)


def save_config(
    config: Dict[str, Any], output_path: str, config_dir: Optional[str] = None
):
    """Save configuration using global config loader.

    Args:
        config: Configuration to save
        output_path: Output path
        config_dir: Configuration directory
    """
    loader = get_config_loader(config_dir)
    loader.save_config(config, output_path)
