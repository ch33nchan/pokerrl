"""
Configurations for baseline training runs (no curriculum, specialists).
"""

# Total episodes for all baseline runs to ensure fair comparison
TOTAL_EPISODES = 1_000_000

# --- No-Curriculum Baseline ---
# Trains the agent for the full duration against a randomly sampled opponent each episode.
NO_CURRICULUM_CONFIG = {
    "run_name": "baseline_no_curriculum",
    "total_episodes": TOTAL_EPISODES,
    # Opponents to sample from each episode
    "opponent_pool": ['RandomBot', 'TightAggressiveBot', 'LoosePassiveBot'],
    "checkpoint_dir": "checkpoints/no_curriculum",
    "final_model_dir": "checkpoints/final_model_no_curriculum"
}

# --- Specialist Baselines ---
# Train against a single opponent for the entire duration.

# 1. Specialist trained only against the aggressive bot
SPECIALIST_CONFIG_AGGRESSIVE = {
    "run_name": "baseline_specialist_aggressive",
    "total_episodes": TOTAL_EPISODES,
    "opponent": "TightAggressiveBot",
    "checkpoint_dir": "checkpoints/specialist_aggressive",
    "final_model_dir": "checkpoints/final_model_specialist_aggressive"
}

# 2. Specialist trained only against a passive bot
SPECIALIST_CONFIG_PASSIVE = {
    "run_name": "baseline_specialist_passive",
    "total_episodes": TOTAL_EPISODES,
    "opponent": "LoosePassiveBot",
    "checkpoint_dir": "checkpoints/specialist_passive",
    "final_model_dir": "checkpoints/final_model_specialist_passive"
}
