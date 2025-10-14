"""Deterministic replay system for ARMAC scheduler verification.

This module implements a comprehensive deterministic replay system that
stores and replays training episodes with exact numerical precision for
verification and debugging purposes.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import hashlib
import pickle
from dataclasses import dataclass, asdict
import time


@dataclass
class ReplayStep:
    """Single step in a deterministic replay."""

    t: int
    s: Dict[str, Any]  # Minimal serializable state
    actor_logits: List[float]
    regret_logits: List[float]
    scheduler_logits: Optional[List[float]]  # None for continuous mode
    lambda_val: float
    k_choice: Optional[int]  # For discrete mode
    action_sampled: int
    reward: float
    done: bool
    legal_actions: List[int]


@dataclass
class ReplayEpisode:
    """Complete episode for deterministic replay."""

    run_id: str
    seed: int
    env: str
    config: Dict[str, Any]
    deck_order: Optional[List[int]]  # For card games
    trajectory: List[ReplayStep]
    rng_state: str  # Serialized RNG state
    timestamp: float
    total_reward: float
    episode_length: int


class DeterministicReplayWriter:
    """Writes deterministic replay data in JSONL format."""

    def __init__(self, replay_dir: str = "replays"):
        """Initialize replay writer.

        Args:
            replay_dir: Directory to store replay files
        """
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.file_handle = None

    def start_new_file(self, run_id: str) -> str:
        """Start a new replay file.

        Args:
            run_id: Unique identifier for this run

        Returns:
            Path to the created file
        """
        timestamp = int(time.time())
        filename = f"{run_id}_{timestamp}.jsonl"
        filepath = self.replay_dir / filename

        if self.file_handle:
            self.file_handle.close()

        self.file_handle = open(filepath, "w")
        self.current_file = str(filepath)
        return self.current_file

    def write_episode(self, episode: ReplayEpisode):
        """Write a single episode to the current file.

        Args:
            episode: Episode data to write
        """
        if self.file_handle is None:
            raise RuntimeError("Must call start_new_file() first")

        # Convert to dict and ensure JSON serializable
        episode_dict = asdict(episode)

        # Convert tensors to lists if any
        episode_dict = self._make_json_serializable(episode_dict)

        # Write as JSON line
        json_line = json.dumps(episode_dict, separators=(",", ":"))
        self.file_handle.write(json_line + "\n")
        self.file_handle.flush()

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def close(self):
        """Close the current file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class DeterministicReplayReader:
    """Reads and verifies deterministic replay data."""

    def __init__(self, replay_file: str):
        """Initialize replay reader.

        Args:
            replay_file: Path to replay file
        """
        self.replay_file = Path(replay_file)
        if not self.replay_file.exists():
            raise FileNotFoundError(f"Replay file not found: {replay_file}")

    def read_episodes(self) -> List[ReplayEpisode]:
        """Read all episodes from the replay file.

        Returns:
            List of episodes
        """
        episodes = []

        with open(self.replay_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    episode_dict = json.loads(line.strip())
                    episode = self._dict_to_episode(episode_dict)
                    episodes.append(episode)
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

        return episodes

    def _dict_to_episode(self, episode_dict: Dict[str, Any]) -> ReplayEpisode:
        """Convert dictionary to ReplayEpisode.

        Args:
            episode_dict: Dictionary representation

        Returns:
            ReplayEpisode instance
        """
        # Convert trajectory steps
        trajectory = []
        for step_dict in episode_dict["trajectory"]:
            step = ReplayStep(**step_dict)
            trajectory.append(step)

        episode_dict["trajectory"] = trajectory
        return ReplayEpisode(**episode_dict)


class ReplayVerifier:
    """Verifies replay determinism by recomputing outputs."""

    def __init__(self, tolerance: float = 1e-6):
        """Initialize verifier.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance

    def verify_episode(
        self,
        episode: ReplayEpisode,
        actor_network: torch.nn.Module,
        regret_network: torch.nn.Module,
        scheduler_network: Optional[torch.nn.Module] = None,
        state_encoder_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Verify a single episode against model outputs.

        Args:
            episode: Episode to verify
            actor_network: Actor network
            regret_network: Regret network
            scheduler_network: Optional scheduler network
            state_encoder_fn: Function to encode states

        Returns:
            Verification results
        """
        results = {
            "run_id": episode.run_id,
            "episode_length": episode.episode_length,
            "verified_steps": 0,
            "failed_steps": 0,
            "max_error": 0.0,
            "error_details": [],
        }

        networks = {
            "actor": actor_network,
            "regret": regret_network,
            "scheduler": scheduler_network,
        }

        # Set networks to eval mode
        for net in networks.values():
            if net is not None:
                net.eval()

        with torch.no_grad():
            for step_idx, step in enumerate(episode.trajectory):
                try:
                    # Encode state if encoder provided
                    if state_encoder_fn:
                        state_tensor = state_encoder_fn(step.s)
                    else:
                        # Simple default encoding - assume state has encoding
                        if "encoding" in step.s:
                            state_tensor = torch.tensor(
                                step.s["encoding"], dtype=torch.float32
                            )
                        else:
                            # Skip verification for this step
                            continue

                    # Recompute actor logits
                    recomputed_actor = networks["actor"](state_tensor)
                    if isinstance(recomputed_actor, dict):
                        recomputed_actor = recomputed_actor.get(
                            "logits", recomputed_actor.get("action_probs")
                        )

                    # Recompute regret logits
                    recomputed_regret = networks["regret"](state_tensor)
                    if isinstance(recomputed_regret, dict):
                        recomputed_regret = recomputed_regret.get(
                            "logits", recomputed_regret.get("action_probs")
                        )

                    # Compare tensors
                    actor_error = self._compare_tensors(
                        step.actor_logits, recomputed_actor, "actor_logits", step_idx
                    )
                    regret_error = self._compare_tensors(
                        step.regret_logits, recomputed_regret, "regret_logits", step_idx
                    )

                    # Verify scheduler if available
                    scheduler_error = 0.0
                    if (
                        networks["scheduler"] is not None
                        and step.scheduler_logits is not None
                    ):
                        recomputed_scheduler = networks["scheduler"](state_tensor)
                        if isinstance(recomputed_scheduler, dict):
                            if "logits" in recomputed_scheduler:
                                recomputed_scheduler = recomputed_scheduler["logits"]
                            else:
                                # For continuous mode, compare lambda directly
                                if "lambda" in recomputed_scheduler:
                                    recomputed_scheduler = recomputed_scheduler[
                                        "lambda"
                                    ]

                        scheduler_error = self._compare_tensors(
                            step.scheduler_logits,
                            recomputed_scheduler,
                            "scheduler_logits",
                            step_idx,
                        )

                    max_step_error = max(actor_error, regret_error, scheduler_error)
                    results["max_error"] = max(results["max_error"], max_step_error)

                    if max_step_error <= self.tolerance:
                        results["verified_steps"] += 1
                    else:
                        results["failed_steps"] += 1
                        results["error_details"].append(
                            {
                                "step": step_idx,
                                "actor_error": actor_error,
                                "regret_error": regret_error,
                                "scheduler_error": scheduler_error,
                                "max_error": max_step_error,
                            }
                        )

                except Exception as e:
                    results["failed_steps"] += 1
                    results["error_details"].append(
                        {"step": step_idx, "error": str(e), "type": "exception"}
                    )

        # Success criteria
        results["success"] = (
            results["failed_steps"] == 0 and results["max_error"] <= self.tolerance
        )

        return results

    def _compare_tensors(
        self,
        original: Union[List[float], torch.Tensor],
        recomputed: Union[List[float], torch.Tensor],
        name: str,
        step_idx: int,
    ) -> float:
        """Compare two tensors and return max absolute error.

        Args:
            original: Original tensor values
            recomputed: Recomputed tensor values
            name: Tensor name for logging
            step_idx: Step index for logging

        Returns:
            Maximum absolute error
        """
        # Convert to numpy arrays
        if isinstance(original, (list, tuple)):
            original = np.array(original, dtype=np.float32)
        elif isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()

        if isinstance(recomputed, (list, tuple)):
            recomputed = np.array(recomputed, dtype=np.float32)
        elif isinstance(recomputed, torch.Tensor):
            recomputed = recomputed.detach().cpu().numpy()

        # Ensure same shape
        if original.shape != recomputed.shape:
            print(
                f"Warning: Shape mismatch for {name} at step {step_idx}: "
                f"original={original.shape}, recomputed={recomputed.shape}"
            )
            # Try to squeeze singleton dimensions
            original = original.squeeze()
            recomputed = recomputed.squeeze()

        # Compute absolute error
        abs_error = np.abs(original - recomputed)
        max_error = np.max(abs_error)

        if max_error > self.tolerance:
            print(f"Error in {name} at step {step_idx}: max_error={max_error:.2e}")

        return max_error


def create_replay_writer(config: Dict[str, Any]) -> DeterministicReplayWriter:
    """Factory function to create replay writer from config.

    Args:
        config: Configuration dictionary

    Returns:
        DeterministicReplayWriter instance
    """
    replay_config = config.get("deterministic_replay", {})
    replay_dir = replay_config.get("replay_dir", "replays")

    return DeterministicReplayWriter(replay_dir=replay_dir)


def create_replay_verifier(config: Dict[str, Any]) -> ReplayVerifier:
    """Factory function to create replay verifier from config.

    Args:
        config: Configuration dictionary

    Returns:
        ReplayVerifier instance
    """
    replay_config = config.get("deterministic_replay", {})
    tolerance = replay_config.get("tolerance", 1e-6)

    return ReplayVerifier(tolerance=tolerance)
