"""Meta-regret module for ARMAC discrete scheduler training.

This module implements the meta-regret manager that maintains regrets over
scheduler choices and performs regret-matching updates for discrete mode
training as specified in the plan.
"""

import numpy as np
import torch
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional, Hashable, Callable
import json
import time


class MetaRegretManager:
    """Meta-regret manager for discrete scheduler training.

    This class maintains cumulative regrets over scheduler choices and
    implements regret-matching to select actions in the discrete mode.
    Enhanced with LRU eviction, EMA smoothing, and robustness measures.
    """

    def __init__(
        self,
        K: int,
        state_key_func: Callable,
        decay: float = 0.99,
        initial_regret: float = 0.0,
        regret_min: float = 0.0,
        smoothing_factor: float = 1e-6,
        max_states: int = 10000,
        util_clip: float = 5.0,
        regret_clip: float = 10.0,
        lru_evict_batch: int = 100,
    ):
        """Initialize meta-regret manager.

        Args:
            K: Number of discrete choices (bins)
            state_key_func: Function to compute state key from observation
            decay: EMA decay factor for utility averaging
            initial_regret: Initial regret value
            regret_min: Minimum regret value (positive part)
            smoothing_factor: Small constant for numerical stability
            max_states: Maximum number of states to store (LRU eviction)
            util_clip: Clipping threshold for utility EMAs
            regret_clip: Clipping threshold for regrets
            lru_evict_batch: Number of states to evict at once when max reached
        """
        self.K = K
        self.state_key_func = state_key_func
        self.decay = decay
        self.initial_regret = initial_regret
        self.regret_min = regret_min
        self.smoothing_factor = smoothing_factor
        self.max_states = max_states
        self.util_clip = util_clip
        self.regret_clip = regret_clip
        self.lru_evict_batch = lru_evict_batch

        # Storage with LRU tracking
        self.regrets = OrderedDict()
        self.util_emas = OrderedDict()
        self.action_counts = OrderedDict()
        self.last_access = OrderedDict()

        # Statistics tracking
        self.total_updates = 0
        self.state_visits = defaultdict(int)
        self.eviction_count = 0
        self.last_eviction_time = time.time()

    def record(
        self,
        state_key: Hashable,
        k_choice: int,
        utility: float,
        learning_rate: float = 1.0,
    ) -> Dict[str, float]:
        """Record a scheduler choice and its utility.

        Args:
            state_key: Hashable representation of state or cluster
            k_choice: Chosen discrete action (bin index)
            utility: Observed utility for this choice
            learning_rate: Learning rate for regret updates

        Returns:
            Dictionary with update statistics
        """
        if not (0 <= k_choice < self.K):
            raise ValueError(f"k_choice {k_choice} out of range [0, {self.K - 1}]")

        # Check for eviction before initializing state to avoid race condition
        if len(self.regrets) >= self.max_states and state_key not in self.regrets:
            self._evict_lru_states()
            # Debug: print eviction info
            if len(self.regrets) > self.max_states:
                print(
                    f"Warning: After eviction, still have {len(self.regrets)} states (max: {self.max_states})"
                )

        # Initialize state if not present
        self._ensure_state_exists(state_key)

        # Update access time for LRU
        self.last_access[state_key] = time.time()

        # Update utility EMA with clipping
        old_util_ema = self.util_emas[state_key][k_choice]
        new_util_ema = self.decay * old_util_ema + (1 - self.decay) * utility
        self.util_emas[state_key][k_choice] = np.clip(
            new_util_ema, -self.util_clip, self.util_clip
        )

        # Update action count
        self.action_counts[state_key][k_choice] += 1

        # Compute average utility across all actions
        util_mean = np.mean(self.util_emas[state_key])

        # Regret matching update with clipping
        regret_increment = self.util_emas[state_key][k_choice] - util_mean
        new_regret = (
            self.regrets[state_key][k_choice] + learning_rate * regret_increment
        )
        self.regrets[state_key][k_choice] = np.clip(
            new_regret, self.regret_min, self.regret_clip
        )

        # Apply positive part constraint
        self.regrets[state_key] = np.maximum(self.regrets[state_key], self.regret_min)

        # Update statistics
        self.total_updates += 1
        self.state_visits[state_key] += 1

        return {
            "util_ema": self.util_emas[state_key][k_choice],
            "util_mean": util_mean,
            "regret_increment": regret_increment,
            "regret_value": self.regrets[state_key][k_choice],
            "total_updates": self.total_updates,
            "num_states": len(self.regrets),
        }

    def get_action_probs(self, state_key: Hashable) -> np.ndarray:
        """Get action probabilities via regret matching.

        Args:
            state_key: Hashable representation of state or cluster

        Returns:
            Probability distribution over K actions
        """
        # Initialize state if not present
        self._ensure_state_exists(state_key)

        # Update access time for LRU
        self.last_access[state_key] = time.time()

        # Check for eviction if needed (after ensuring state exists)
        if len(self.regrets) > self.max_states:
            self._evict_lru_states()

        # If state was evicted during eviction, re-initialize it
        if state_key not in self.regrets:
            self._ensure_state_exists(state_key)

        # Positive part of regrets
        g = np.maximum(self.regrets[state_key], self.regret_min)

        # Add smoothing to avoid zero probabilities
        g = g + self.smoothing_factor

        # Normalize to get probabilities
        s = g.sum()
        if s <= 0:
            # Uniform policy if all regrets are zero/negative
            return np.ones(self.K) / self.K

        return g / s

    def sample_action(
        self, state_key: Hashable, stochastic: bool = True
    ) -> Tuple[int, np.ndarray]:
        """Sample an action using regret-matching policy.

        Args:
            state_key: Hashable representation of state or cluster
            stochastic: Whether to sample stochastically or take argmax

        Returns:
            Tuple of (chosen_action, action_probabilities)
        """
        probs = self.get_action_probs(state_key)

        if stochastic:
            action = np.random.choice(self.K, p=probs)
        else:
            action = np.argmax(probs)

        return action, probs

    def _ensure_state_exists(self, state_key: Hashable):
        """Ensure state exists in all tracking dictionaries.

        Args:
            state_key: State key to initialize
        """
        if state_key not in self.regrets:
            self.regrets[state_key] = np.full(self.K, self.initial_regret)
            self.util_emas[state_key] = np.zeros(self.K)
            self.action_counts[state_key] = np.zeros(self.K, dtype=int)
            self.last_access[state_key] = time.time()

    def _evict_lru_states(self):
        """Evict least recently used states to maintain memory bounds."""
        if len(self.regrets) <= self.max_states:
            return

        # Sort by last access time
        sorted_states = sorted(self.last_access.items(), key=lambda x: x[1])

        # Evict batch size - be more aggressive to stay under the limit
        evict_count = min(
            self.lru_evict_batch,
            len(sorted_states),
            max(len(sorted_states) - self.max_states + 1, self.lru_evict_batch),
        )
        evicted_keys = [state_key for state_key, _ in sorted_states[:evict_count]]

        # Remove from all dictionaries
        for state_key in evicted_keys:
            self.regrets.pop(state_key, None)
            self.util_emas.pop(state_key, None)
            self.action_counts.pop(state_key, None)
            self.last_access.pop(state_key, None)
            self.state_visits.pop(state_key, None)

        self.eviction_count += len(evicted_keys)
        self.last_eviction_time = time.time()

    def get_eviction_stats(self) -> Dict[str, Any]:
        """Get eviction statistics.

        Returns:
            Dictionary with eviction stats
        """
        return {
            "eviction_count": self.eviction_count,
            "last_eviction_time": self.last_eviction_time,
            "current_states": len(self.regrets),
            "max_states": self.max_states,
            "utilization": len(self.regrets) / self.max_states,
        }

    def get_regret_stats(self, state_key: Hashable) -> Dict[str, Any]:
        """Get statistics for a given state.

        Args:
            state_key: Hashable representation of state or cluster

        Returns:
            Dictionary with regret statistics
        """
        regrets = self.regrets[state_key]
        utils = self.util_emas[state_key]
        counts = self.action_counts[state_key]
        probs = self.get_action_probs(state_key)

        return {
            "regrets": regrets.copy(),
            "utilities": utils.copy(),
            "action_counts": counts.copy(),
            "probabilities": probs.copy(),
            "total_visits": self.state_visits[state_key],
            "regret_sum": regrets.sum(),
            "util_mean": utils.mean(),
            "util_std": utils.std() if counts.sum() > 1 else 0.0,
            "entropy": -(probs * np.log(probs + 1e-8)).sum(),
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all states.

        Returns:
            Dictionary with global statistics
        """
        all_regrets = []
        all_utils = []
        all_entropies = []

        for state_key in self.regrets.keys():
            regrets = self.regrets[state_key]
            utils = self.util_emas[state_key]
            probs = self.get_action_probs(state_key)

            all_regrets.extend(regrets)
            all_utils.extend(utils)
            all_entropies.append(-(probs * np.log(probs + 1e-8)).sum())

        stats = {
            "total_states": len(self.regrets),
            "total_updates": self.total_updates,
            "regret_mean": np.mean(all_regrets) if all_regrets else 0.0,
            "regret_std": np.std(all_regrets) if all_regrets else 0.0,
            "regret_min_global": np.min(all_regrets) if all_regrets else 0.0,
            "regret_max_global": np.max(all_regrets) if all_regrets else 0.0,
            "util_mean": np.mean(all_utils) if all_utils else 0.0,
            "util_std": np.std(all_utils) if all_utils else 0.0,
            "entropy_mean": np.mean(all_entropies) if all_entropies else 0.0,
            "entropy_std": np.std(all_entropies) if all_entropies else 0.0,
        }

        # Add eviction stats
        stats.update(self.get_eviction_stats())

        return stats

    def decay_regrets(self, decay_factor: float = 0.99):
        """Apply decay to all regrets.

        Args:
            decay_factor: Decay factor to apply
        """
        for state_key in self.regrets:
            self.regrets[state_key] *= decay_factor

    def reset_state(self, state_key: Hashable):
        """Reset regrets for a specific state.

        Args:
            state_key: State key to reset
        """
        if state_key in self.regrets:
            del self.regrets[state_key]
        if state_key in self.util_emas:
            del self.util_emas[state_key]
        if state_key in self.action_counts:
            del self.action_counts[state_key]
        if state_key in self.state_visits:
            del self.state_visits[state_key]

    def reset_all(self):
        """Reset all regrets and statistics."""
        self.regrets.clear()
        self.util_emas.clear()
        self.action_counts.clear()
        self.state_visits.clear()
        self.total_updates = 0

    def save_state(self, filepath: str):
        """Save the current state to a file.

        Args:
            filepath: Path to save the state
        """
        state_data = {
            "regrets": {str(k): v.tolist() for k, v in self.regrets.items()},
            "util_emas": {str(k): v.tolist() for k, v in self.util_emas.items()},
            "action_counts": {
                str(k): v.tolist() for k, v in self.action_counts.items()
            },
            "state_visits": {str(k): v for k, v in self.state_visits.items()},
            "last_access": {str(k): v for k, v in self.last_access.items()},
            "total_updates": self.total_updates,
            "eviction_count": self.eviction_count,
            "K": self.K,
            "decay": self.decay,
            "initial_regret": self.initial_regret,
            "regret_min": self.regret_min,
            "smoothing_factor": self.smoothing_factor,
            "max_states": self.max_states,
            "util_clip": self.util_clip,
            "regret_clip": self.regret_clip,
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, filepath: str):
        """Load state from a file.

        Args:
            filepath: Path to load the state from
        """
        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Restore data using OrderedDict for LRU tracking
        self.regrets = OrderedDict(
            (k, np.array(v)) for k, v in state_data["regrets"].items()
        )
        self.util_emas = OrderedDict(
            (k, np.array(v)) for k, v in state_data["util_emas"].items()
        )
        self.action_counts = OrderedDict(
            (k, np.array(v, dtype=int)) for k, v in state_data["action_counts"].items()
        )
        self.state_visits = defaultdict(int, state_data["state_visits"])
        self.last_access = OrderedDict(
            (k, v) for k, v in state_data.get("last_access", {}).items()
        )

        # Restore statistics
        self.total_updates = state_data.get("total_updates", 0)
        self.eviction_count = state_data.get("eviction_count", 0)

        # Restore parameters
        self.K = state_data["K"]
        self.decay = state_data["decay"]
        self.initial_regret = state_data["initial_regret"]
        self.regret_min = state_data["regret_min"]
        self.smoothing_factor = state_data["smoothing_factor"]
        self.max_states = state_data.get("max_states", 10000)
        self.util_clip = state_data.get("util_clip", 5.0)
        self.regret_clip = state_data.get("regret_clip", 10.0)


def compute_state_key_simple(state_encoding: torch.Tensor) -> str:
    """Simple state key function based on discretized encoding.

    Args:
        state_encoding: State encoding tensor

    Returns:
        String state key
    """
    # Discretize the encoding to create a hashable key
    if isinstance(state_encoding, torch.Tensor):
        # Convert to numpy and discretize
        encoding_np = state_encoding.detach().cpu().numpy()
        # Round to 3 decimal places and convert to string
        discretized = np.round(encoding_np, 3)
        return str(discretized.tolist())
    else:
        return str(state_encoding)


def compute_state_key_cluster(
    state_encoding: torch.Tensor, cluster_centers: np.ndarray, k: int = 10
) -> int:
    """Cluster-based state key function.

    Args:
        state_encoding: State encoding tensor
        cluster_centers: Cluster centers for discretization
        k: Number of clusters to consider

    Returns:
        Cluster index as state key
    """
    if isinstance(state_encoding, torch.Tensor):
        encoding_np = state_encoding.detach().cpu().numpy()
    else:
        encoding_np = np.array(state_encoding)

    # Find nearest cluster centers
    distances = np.linalg.norm(
        cluster_centers[:, : encoding_np.shape[-1]] - encoding_np, axis=1
    )
    nearest_indices = np.argpartition(distances, min(k, len(distances)))[:k]

    # Return the index of the nearest cluster
    return int(nearest_indices[0])


def create_meta_regret_manager(config: dict) -> MetaRegretManager:
    """Factory function to create meta-regret manager from config.

    Args:
        config: Configuration dictionary

    Returns:
        MetaRegretManager instance
    """
    meta_config = config.get("meta_regret", {})

    K = meta_config.get("K", 5)
    decay = meta_config.get("decay", 0.99)
    initial_regret = meta_config.get("initial_regret", 0.0)
    regret_min = meta_config.get("regret_min", 0.0)
    smoothing_factor = meta_config.get("smoothing_factor", 1e-6)
    max_states = meta_config.get("max_states", 10000)
    util_clip = meta_config.get("util_clip", 5.0)
    regret_clip = meta_config.get("regret_clip", 10.0)
    lru_evict_batch = meta_config.get("lru_evict_batch", 100)

    # Use enhanced state key manager if available
    try:
        from algs.scheduler.utils.state_keying import create_state_key_manager

        state_key_func = create_state_key_manager(config)
    except ImportError:
        # Fallback to simple state key function
        state_key_func = compute_state_key_simple

    return MetaRegretManager(
        K=K,
        state_key_func=state_key_func,
        decay=decay,
        initial_regret=initial_regret,
        regret_min=regret_min,
        smoothing_factor=smoothing_factor,
        max_states=max_states,
        util_clip=util_clip,
        regret_clip=regret_clip,
        lru_evict_batch=lru_evict_batch,
    )
