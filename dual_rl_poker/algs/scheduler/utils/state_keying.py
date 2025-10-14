"""State keying utilities for meta-regret manager.

This module implements various state representation strategies for the
meta-regret manager, allowing configurable granularity levels.
"""

import numpy as np
import torch
from typing import Hashable, Dict, Any, Optional
from sklearn.cluster import KMeans
import pickle
import os


def compute_state_key(s: Dict[str, Any], level: int = 1) -> str:
    """Compute state key at specified granularity level.

    Args:
        s: State dictionary with fields: pot, stacks, player_pos, round, embedding
        level: Granularity level (0=coarse, 1=medium, 2=fine)

    Returns:
        Hashable state key string
    """
    if level == 0:
        # Coarse: game_stage + player_pos + round
        return f"{s.get('round', 0)}_{s.get('player_pos', 0)}"

    elif level == 1:
        # Medium: include discretized pot/stack ratios, pot size bucket
        pot_bucket = int(min(s.get("pot", 0) / 10, 9))
        stack_ratio = s.get("stack_ratio", 0)
        stack_bucket = int(min(max(stack_ratio * 10, 0), 9))
        return f"{s.get('round', 0)}_{s.get('player_pos', 0)}_pb{pot_bucket}_sb{stack_bucket}"

    else:
        # Fine: hash embedding via clustering
        embedding = s.get("embedding", None)
        if embedding is not None:
            if isinstance(embedding, torch.Tensor):
                embedding_np = embedding.detach().cpu().numpy()
            else:
                embedding_np = np.array(embedding)

            # Simple discretization for now - could use k-means clustering
            discretized = np.round(embedding_np * 100).astype(int)
            return f"emb_{hash(discretized.tobytes()) % 1000000}"
        else:
            # Fallback to medium level
            return compute_state_key(s, level=1)


class StateKeyManager:
    """Manages state key computation with clustering support."""

    def __init__(
        self,
        level: int = 1,
        n_clusters: int = 100,
        cluster_file: Optional[str] = None,
        update_clusters: bool = True,
    ):
        """Initialize state key manager.

        Args:
            level: Granularity level (0=coarse, 1=medium, 2=fine)
            n_clusters: Number of clusters for fine-grained keying
            cluster_file: Path to saved cluster centers
            update_clusters: Whether to update clusters online
        """
        self.level = level
        self.n_clusters = n_clusters
        self.cluster_file = cluster_file
        self.update_clusters = update_clusters

        self.kmeans = None
        self.embeddings_buffer = []
        self.cluster_update_freq = 1000
        self.update_counter = 0

        # Load existing clusters if available
        if cluster_file and os.path.exists(cluster_file):
            try:
                with open(cluster_file, "rb") as f:
                    cluster_data = pickle.load(f)
                    self.kmeans = cluster_data["kmeans"]
                    self.level = cluster_data["level"]
                    self.n_clusters = cluster_data["n_clusters"]
            except Exception as e:
                print(f"Warning: Could not load cluster file: {e}")

    def __call__(self, s: Dict[str, Any]) -> Hashable:
        """Compute state key for given state.

        Args:
            s: State dictionary

        Returns:
            Hashable state key
        """
        if self.level < 2:
            return compute_state_key(s, self.level)

        # Fine-grained clustering
        embedding = s.get("embedding", None)
        if embedding is None:
            return compute_state_key(s, 1)

        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.detach().cpu().numpy().flatten()
        else:
            embedding_np = np.array(embedding).flatten()

        # Initialize clusters if needed
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            # Use random initialization initially
            self.kmeans.cluster_centers_ = np.random.randn(
                self.n_clusters, len(embedding_np)
            )

        # Update cluster buffer
        if self.update_clusters:
            self.embeddings_buffer.append(embedding_np)
            self.update_counter += 1

            # Update clusters periodically
            if (
                self.update_counter % self.cluster_update_freq == 0
                and len(self.embeddings_buffer) >= self.n_clusters
            ):
                self._update_clusters()

        # Get cluster assignment
        cluster_id = self.kmeans.predict(embedding_np.reshape(1, -1))[0]
        return f"cl{cluster_id:03d}"

    def _update_clusters(self):
        """Update K-means clusters with buffered embeddings."""
        if len(self.embeddings_buffer) >= self.n_clusters:
            embeddings_array = np.array(self.embeddings_buffer)
            try:
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters, random_state=42, n_init=10
                )
                self.kmeans.fit(embeddings_array)
                self.embeddings_buffer = []  # Clear buffer

                # Save clusters if path provided
                if self.cluster_file:
                    self.save_clusters()
            except Exception as e:
                print(f"Warning: Cluster update failed: {e}")

    def save_clusters(self):
        """Save cluster centers to file."""
        if self.kmeans is not None and self.cluster_file:
            try:
                os.makedirs(os.path.dirname(self.cluster_file), exist_ok=True)
                cluster_data = {
                    "kmeans": self.kmeans,
                    "level": self.level,
                    "n_clusters": self.n_clusters,
                }
                with open(self.cluster_file, "wb") as f:
                    pickle.dump(cluster_data, f)
            except Exception as e:
                print(f"Warning: Could not save cluster file: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics.

        Returns:
            Dictionary with clustering stats
        """
        stats = {
            "level": self.level,
            "n_clusters": self.n_clusters,
            "update_counter": self.update_counter,
            "buffer_size": len(self.embeddings_buffer),
            "clusters_initialized": self.kmeans is not None,
        }

        if self.kmeans is not None:
            stats["cluster_centers_shape"] = self.kmeans.cluster_centers_.shape

        return stats


def create_state_key_manager(config: Dict[str, Any]) -> StateKeyManager:
    """Factory function to create state key manager from config.

    Args:
        config: Configuration dictionary

    Returns:
        StateKeyManager instance
    """
    keying_config = config.get("state_keying", {})

    level = keying_config.get("level", 1)
    n_clusters = keying_config.get("n_clusters", 100)
    cluster_file = keying_config.get("cluster_file", None)
    update_clusters = keying_config.get("update_clusters", True)

    return StateKeyManager(
        level=level,
        n_clusters=n_clusters,
        cluster_file=cluster_file,
        update_clusters=update_clusters,
    )
