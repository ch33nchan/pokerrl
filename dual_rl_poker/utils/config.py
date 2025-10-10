"""Configuration loading and management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Set default values for missing keys
        config = _set_defaults(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        raise RuntimeError(f"Error saving configuration: {e}")


def _set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default values for configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with defaults applied
    """
    defaults = {
        'experiment': {
            'name': 'dual_rl_poker_experiment',
            'seed': 42,
            'device': 'cpu',
            'log_level': 'INFO'
        },
        'game': {
            'name': 'kuhn_poker',
            'num_players': 2
        },
        'training': {
            'iterations': 500,
            'eval_every': 25,
            'save_every': 100,
            'batch_size': 2048,
            'replay_window': 10,
            'buffer_size': 10000,
            'learning_rate': 3e-4,
            'weight_decay': 0.0,
            'gradient_clip': 5.0
        },
        'network': {
            'type': 'mlp',
            'hidden_dims': [64, 64],
            'dropout': 0.0
        },
        'evaluation': {
            'num_episodes': 1000,
            'head_to_head_episodes': 5000,
            'bootstrap_samples': 1000,
            'confidence_level': 0.95
        },
        'algorithms': {
            'deep_cfr': {
                'advantage_memory_size': 10000,
                'strategy_memory_size': 10000,
                'external_sampling': True
            },
            'sd_cfr': {
                'memory_size': 10000,
                'external_sampling': True
            },
            'armac': {
                'buffer_size': 10000,
                'policy_replay_size': 5,
                'update_frequency': 1,
                'gamma': 0.99
            }
        },
        'optimizer': {
            'type': 'adam',
            'lr': 3e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.0
        },
        'logging': {
            'log_dir': 'logs',
            'save_dir': 'results',
            'tensorboard': False,
            'print_frequency': 25,
            'save_frequency': 100
        },
        'reproducibility': {
            'deterministic': True,
            'torch_deterministic': True,
            'benchmark': False
        }
    }

    return _merge_configs(defaults, config)


def _merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries.

    Args:
        default: Default configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['experiment', 'game', 'training', 'network', 'algorithms']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate training parameters
    training = config['training']
    if training['iterations'] <= 0:
        raise ValueError("Training iterations must be positive")
    if training['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
    if training['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")

    # Validate game configuration
    game = config['game']
    valid_games = ['kuhn_poker', 'leduc_poker']
    if game['name'] not in valid_games:
        raise ValueError(f"Invalid game: {game['name']}. Must be one of {valid_games}")

    # Validate algorithm configuration
    algorithms = config['algorithms']
    valid_algorithms = ['deep_cfr', 'sd_cfr', 'armac']
    for alg in algorithms:
        if alg not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {alg}. Must be one of {valid_algorithms}")

    return True


def get_algorithm_config(config: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
    """Get configuration for a specific algorithm.

    Args:
        config: Global configuration
        algorithm_name: Name of the algorithm

    Returns:
        Algorithm-specific configuration
    """
    algorithm_config = config['algorithms'].get(algorithm_name, {})

    # Merge with global training and network configs
    merged_config = {
        'training': config['training'].copy(),
        'network': config['network'].copy(),
        'optimizer': config['optimizer'].copy(),
        'experiment': config['experiment'].copy(),
        'game': config['game'].copy(),
        'logging': config['logging'].copy(),
        'reproducibility': config['reproducibility'].copy(),
        'evaluation': config['evaluation'].copy()
    }

    # Apply algorithm-specific overrides
    merged_config.update(algorithm_config)

    return merged_config