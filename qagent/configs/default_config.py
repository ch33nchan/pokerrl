"""
Default configuration settings for the poker reinforcement learning project.
This file centralizes parameters for the agent, environment, and training process,
allowing for easy management and reproduction of experiments.
"""

import torch

# --- General Training Configuration ---
TRAINING_CONFIG = {
    "use_wandb": True,   # Whether to use Weights & Biases for logging
    "use_gpu": True,     # Whether to use GPU if available
    "seed": 42,
    "log_dir": "logs",
    "checkpoint_dir": "checkpoints",
    "checkpoint_freq": 10000, # Save model checkpoint every 10000 episodes
}

# --- Agent Configuration ---
AGENT_CONFIG = {
    "lr": 1e-4,          # Learning rate for the Adam optimizer
    "gamma": 0.99,       # Discount factor for future rewards
    "entropy_beta": 0.01, # Coefficient for the entropy bonus in the loss function
    "hidden_size": 256,  # Number of neurons in hidden layers
    "device": "cuda" if torch.cuda.is_available() and TRAINING_CONFIG['use_gpu'] else "cpu",
}

# --- Environment Configuration ---
ENV_CONFIG = {
    "initial_stack": 1000,
    "small_blind": 5,
    "big_blind": 10,
    "min_raise": 10,
    "max_stage_raises": 4, # Increased from 3 for more complex scenarios
}

# --- Curriculum Learning Configuration ---
# Defines the stages of training, the opponent for each stage, and the number of episodes.
CURRICULUM = [
    {
        "stage_name": "Stage 1: Vs. Random",
        "opponent": "RandomBot",
        "episodes": 100_000,
    },
    {
        "stage_name": "Stage 2: Vs. Tight-Aggressive",
        "opponent": "TightAggressiveBot",
        "episodes": 200_000,
    },
    {
        "stage_name": "Stage 3: Vs. Loose-Passive",
        "opponent": "LoosePassiveBot",
        "episodes": 200_000,
    },
    {
        "stage_name": "Stage 4: Mixed Opponent Pool",
        "opponent": "MixedStrategyBot",
        "episodes": 500_000,
    },
    # Stage 5 (Self-play) will require a more complex Population-Based Training loop
    # and is deferred as a secondary task per the action plan.
]

# --- Evaluation Configuration ---
EVALUATION_CONFIG = {
    "episodes": 10_000, # Number of episodes to run for each evaluation matchup
    "num_seeds": 5,     # Number of independent runs for statistical significance
}

# --- Statistical Analysis Configuration ---
STATISTICAL_CONFIG = {
    "bootstrap_samples": 10_000,
    "confidence_level": 0.95,
    "p_value_threshold": 0.05, # Alpha for significance testing
}

# --- Project Configuration ---
PROJECT_CONFIG = {
    "wandb_project": "poker-rl-research",
    "wandb_entity": "tbsrinivas-x", # Replace with your wandb entity
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
    "log_interval": 1000,       # Log average reward every 1000 episodes
    "checkpoint_interval": 10000, # Save a checkpoint every 10000 episodes
}

# --- Opponent Mapping ---
from qagent.opponents.bots import (
    RandomBot,
    CallBot,
    TightAggressiveBot,
    LoosePassiveBot,
    MixedStrategyBot,
    AdaptiveBot
)

OPPONENTS = {
    "RandomBot": RandomBot,
    "CallBot": CallBot,
    "TightAggressiveBot": TightAggressiveBot,
    "LoosePassiveBot": LoosePassiveBot,
    "MixedStrategyBot": MixedStrategyBot,
    "AdaptiveBot": AdaptiveBot,
}
