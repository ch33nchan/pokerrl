"""
Runs Manifest Schema for Deep CFR Architecture Study

This module defines the comprehensive manifest structure for tracking all experimental runs
with complete configuration, results, and diagnostics information.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import json
from datetime import datetime

@dataclass
class RunManifest:
    """Complete manifest entry for a single experimental run."""

    # Core identification
    run_id: str
    game: str  # "kuhn_poker" or "leduc_holdem"
    traversal: str  # "external_sampling" (fixed for main results)
    architecture: str  # "baseline", "lstm_opt", "lstm_no_hist", "lstm_no_emb", "transformer"
    seed: int

    # Model capacity
    params_count: int
    flops_est: float  # FLOPs per forward pass

    # Training configuration
    optimizer_cfg: Dict[str, Any]
    replay_cfg: Dict[str, Any]
    update_cadence: int  # iterations between neural updates

    # Training schedule
    iterations: int  # 400 for both games
    eval_every: int  # 20 iterations

    # Results metrics
    final_exploitability: float
    steps_to_threshold: Optional[float] = None  # steps to reach target threshold T
    time_to_threshold: Optional[float] = None  # wall-clock time to reach T
    wall_clock_s: float = 0.0  # total wall-clock time

    # NashConv (if available)
    final_nashconv: Optional[float] = None

    # Environment info
    openspiel_version: str = "1.6.3"
    python_version: str = "3.9.6"
    commit_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Diagnostics coverage
    diagnostics_coverage: Dict[str, bool] = field(default_factory=dict)

    # Additional metrics
    peak_memory_mb: Optional[float] = None
    convergence_iteration: Optional[int] = None

    # Head-to-head EV results (filled later)
    ev_vs_tabular: Optional[float] = None
    ev_vs_deepcfr: Optional[float] = None
    ev_vs_sdcfr: Optional[float] = None

    # Status tracking
    status: str = "pending"  # "pending", "running", "completed", "failed"
    error_message: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for a complete experimental study."""

    # Target thresholds T for each game
    threshold_targets: Dict[str, float] = field(default_factory=lambda: {
        "kuhn_poker": 0.1,  # mBB/100
        "leduc_holdem": 2.5  # mBB/100
    })

    # Architecture configurations
    architectures: List[str] = field(default_factory=lambda: [
        "baseline", "lstm_opt", "lstm_no_hist", "lstm_no_emb"
    ])

    # Games to run
    games: List[str] = field(default_factory=lambda: [
        "kuhn_poker", "leduc_holdem"
    ])

    # Seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: list(range(20)))

    # Training parameters
    iterations: int = 400
    eval_every: int = 20
    update_cadence: int = 10

    # Fixed traversal scheme
    traversal: str = "external_sampling"


def create_manifest_entry(
    run_id: str,
    game: str,
    architecture: str,
    seed: int,
    config: ExperimentConfig,
    **kwargs
) -> RunManifest:
    """Create a new manifest entry with default configuration."""

    # Default optimizer configuration
    default_optimizer_cfg = {
        "type": "adam",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0
    }

    # Default replay configuration
    default_replay_cfg = {
        "buffer_size": 50000,
        "batch_size": 128,
        "sampling": "uniform"
    }

    # Architecture-specific parameters
    arch_params = {
        "baseline": {"params_count": 5040, "flops_est": 5040},  # ~5K parameters
        "lstm_opt": {"params_count": 12640, "flops_est": 25280},  # ~12K parameters
        "lstm_no_hist": {"params_count": 8960, "flops_est": 17920},  # ~9K parameters
        "lstm_no_emb": {"params_count": 10240, "flops_est": 20480},  # ~10K parameters
        "transformer": {"params_count": 15360, "flops_est": 30720},  # ~15K parameters
    }

    arch_info = arch_params.get(architecture, arch_params["baseline"])

    return RunManifest(
        run_id=run_id,
        game=game,
        traversal=config.traversal,
        architecture=architecture,
        seed=seed,
        params_count=arch_info["params_count"],
        flops_est=arch_info["flops_est"],
        optimizer_cfg=kwargs.get("optimizer_cfg", default_optimizer_cfg),
        replay_cfg=kwargs.get("replay_cfg", default_replay_cfg),
        update_cadence=config.update_cadence,
        iterations=config.iterations,
        eval_every=config.eval_every,
        # Results to be filled after training
        final_exploitability=float('nan'),
        steps_to_threshold=None,
        time_to_threshold=None,
        wall_clock_s=float('nan'),
        openspiel_version="1.6.3",
        python_version="3.9.6",
        **kwargs
    )


def generate_run_manifests(config: ExperimentConfig) -> List[RunManifest]:
    """Generate all manifest entries for a complete experimental study."""
    manifests = []

    for game in config.games:
        for architecture in config.architectures:
            for seed in config.seeds:
                run_id = f"{game}_{architecture}_{seed:03d}"
                manifest = create_manifest_entry(run_id, game, architecture, seed, config)
                manifests.append(manifest)

    return manifests


def save_manifests(manifests: List[RunManifest], filepath: str):
    """Save manifests to JSON file."""
    data = [manifest.__dict__ for manifest in manifests]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_manifests(filepath: str) -> List[RunManifest]:
    """Load manifests from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [RunManifest(**item) for item in data]


if __name__ == "__main__":
    # Generate initial manifest for the study
    config = ExperimentConfig()
    manifests = generate_run_manifests(config)

    # Save to experiments directory
    import os
    os.makedirs("experiments", exist_ok=True)
    save_manifests(manifests, "experiments/runs_manifest.json")

    print(f"Generated {len(manifests)} manifest entries")
    print(f"Games: {config.games}")
    print(f"Architectures: {config.architectures}")
    print(f"Seeds per config: {len(config.seeds)}")
    print(f"Total runs: {len(manifests)}")