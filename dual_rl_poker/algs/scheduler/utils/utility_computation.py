"""Utility signal computation for scheduler training.

This module implements various strategies for computing utility signals
that guide the discrete scheduler's meta-regret learning process.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


class UtilitySignalComputer:
    """Computes utility signals for discrete scheduler training."""

    def __init__(
        self,
        utility_type: str = "advantage_based",
        gamma: float = 0.99,
        baseline_window: int = 100,
        advantage_window: int = 10,
        min_samples: int = 5,
    ):
        """Initialize utility signal computer.

        Args:
            utility_type: Type of utility computation ('immediate', 'advantage_based', 'hybrid')
            gamma: Discount factor for return computation
            baseline_window: Window size for moving average baseline
            advantage_window: Window for advantage estimation
            min_samples: Minimum samples required for reliable estimates
        """
        self.utility_type = utility_type
        self.gamma = gamma
        self.baseline_window = baseline_window
        self.advantage_window = advantage_window
        self.min_samples = min_samples

        # Tracking buffers
        self.return_buffer = deque(maxlen=baseline_window)
        self.advantage_buffer = deque(maxlen=advantage_window)
        self.episode_returns = []

    def compute_scheduler_utility(
        self,
        trajectory: List[Dict[str, Any]],
        decision_index: int,
        value_fn: Optional[callable] = None,
        critic_network: Optional[torch.nn.Module] = None,
        state_encoding_fn: Optional[callable] = None,
    ) -> float:
        """Compute utility signal for a scheduler decision.

        Args:
            trajectory: List of step dictionaries from an episode
            decision_index: Index of the decision in the trajectory
            value_fn: Optional value function for baseline
            critic_network: Optional critic network for baseline
            state_encoding_fn: Optional function to encode states for critic

        Returns:
            Utility signal for the scheduler choice
        """
        if self.utility_type == "immediate":
            return self._compute_immediate_utility(trajectory, decision_index, value_fn)
        elif self.utility_type == "advantage_based":
            return self._compute_advantage_based_utility(
                trajectory, decision_index, critic_network, state_encoding_fn
            )
        elif self.utility_type == "hybrid":
            imm_util = self._compute_immediate_utility(
                trajectory, decision_index, value_fn
            )
            adv_util = self._compute_advantage_based_utility(
                trajectory, decision_index, critic_network, state_encoding_fn
            )
            return 0.5 * imm_util + 0.5 * adv_util
        else:
            raise ValueError(f"Unknown utility_type: {self.utility_type}")

    def _compute_immediate_utility(
        self,
        trajectory: List[Dict[str, Any]],
        decision_index: int,
        value_fn: Optional[callable] = None,
    ) -> float:
        """Compute utility based on discounted return from decision.

        Args:
            trajectory: Episode trajectory
            decision_index: Decision index
            value_fn: Optional value function for baseline

        Returns:
            Utility signal (discounted return - baseline)
        """
        # Compute discounted return from decision_index to end
        G = 0.0
        for t, step in enumerate(trajectory[decision_index:]):
            reward = step.get("reward", 0.0)
            G += (self.gamma**t) * reward

        # Track returns for baseline
        self.return_buffer.append(G)

        # Compute baseline
        if value_fn is not None:
            decision_state = trajectory[decision_index].get("s", None)
            if decision_state is not None:
                baseline = value_fn(decision_state)
            else:
                baseline = np.mean(self.return_buffer) if self.return_buffer else 0.0
        else:
            baseline = (
                np.mean(self.return_buffer)
                if len(self.return_buffer) >= self.min_samples
                else 0.0
            )

        return G - baseline

    def _compute_advantage_based_utility(
        self,
        trajectory: List[Dict[str, Any]],
        decision_index: int,
        critic_network: Optional[torch.nn.Module] = None,
        state_encoding_fn: Optional[callable] = None,
    ) -> float:
        """Compute utility based on expected advantage of mixed policy.

        Args:
            trajectory: Episode trajectory
            decision_index: Decision index
            critic_network: Critic network for Q-value estimation
            state_encoding_fn: Function to encode states

        Returns:
            Expected advantage utility
        """
        if critic_network is None or state_encoding_fn is None:
            # Fallback to immediate utility
            return self._compute_immediate_utility(trajectory, decision_index)

        decision_step = trajectory[decision_index]
        state = decision_step.get("s", None)

        if state is None:
            return 0.0

        # Encode state for critic
        state_tensor = state_encoding_fn(state)

        with torch.no_grad():
            # Get Q-values for all actions
            critic_output = critic_network(state_tensor)

            if isinstance(critic_output, dict):
                q_values = critic_output.get(
                    "q_values", critic_output.get("values", None)
                )
            else:
                q_values = critic_output

            if q_values is None:
                return 0.0

            if q_values.shape[-1] == 1:
                # If critic outputs single value, use it directly
                v = q_values.item()
                return v  # Simple utility based on state value
            else:
                # Compute expected advantage under mixed policy
                mixed_policy = decision_step.get("mixed_policy", None)
                if mixed_policy is None:
                    return 0.0

                # Convert to tensor if needed
                if isinstance(mixed_policy, np.ndarray):
                    mixed_policy = torch.tensor(mixed_policy, dtype=torch.float32)

                # Compute expected value and advantage
                q_values = q_values.squeeze()
                expected_q = (mixed_policy * q_values).sum()
                v = expected_q  # For single state, expected Q is the value

                # Expected advantage (zero for single state)
                advantage = (mixed_policy * (q_values - v)).sum()

                return advantage.item()

    def compute_episode_metrics(
        self, trajectory: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute metrics for the entire episode.

        Args:
            trajectory: Episode trajectory

        Returns:
            Dictionary with episode metrics
        """
        total_reward = sum(step.get("reward", 0.0) for step in trajectory)
        episode_length = len(trajectory)

        # Store episode return
        self.episode_returns.append(total_reward)

        # Keep only recent episodes
        if len(self.episode_returns) > 100:
            self.episode_returns = self.episode_returns[-100:]

        metrics = {
            "total_reward": total_reward,
            "episode_length": episode_length,
            "avg_reward_per_step": total_reward / max(episode_length, 1),
        }

        # Add rolling statistics
        if len(self.episode_returns) >= 10:
            recent_returns = self.episode_returns[-10:]
            metrics.update(
                {
                    "avg_return_last_10": np.mean(recent_returns),
                    "std_return_last_10": np.std(recent_returns),
                }
            )

        if len(self.return_buffer) >= 10:
            metrics.update(
                {
                    "baseline_return": np.mean(self.return_buffer),
                    "baseline_std": np.std(self.return_buffer),
                }
            )

        return metrics

    def reset_buffers(self):
        """Reset all tracking buffers."""
        self.return_buffer.clear()
        self.advantage_buffer.clear()
        self.episode_returns.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get utility computation statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "utility_type": self.utility_type,
            "gamma": self.gamma,
            "baseline_window": self.baseline_window,
            "return_buffer_size": len(self.return_buffer),
            "episode_returns_count": len(self.episode_returns),
        }

        if self.return_buffer:
            stats.update(
                {
                    "return_buffer_mean": np.mean(self.return_buffer),
                    "return_buffer_std": np.std(self.return_buffer),
                    "return_buffer_min": np.min(self.return_buffer),
                    "return_buffer_max": np.max(self.return_buffer),
                }
            )

        if self.episode_returns:
            stats.update(
                {
                    "episode_returns_mean": np.mean(self.episode_returns),
                    "episode_returns_std": np.std(self.episode_returns),
                    "episode_returns_min": np.min(self.episode_returns),
                    "episode_returns_max": np.max(self.episode_returns),
                }
            )

        return stats


def create_utility_computer(config: Dict[str, Any]) -> UtilitySignalComputer:
    """Factory function to create utility computer from config.

    Args:
        config: Configuration dictionary

    Returns:
        UtilitySignalComputer instance
    """
    utility_config = config.get("utility_computation", {})

    utility_type = utility_config.get("utility_type", "advantage_based")
    gamma = utility_config.get("gamma", 0.99)
    baseline_window = utility_config.get("baseline_window", 100)
    advantage_window = utility_config.get("advantage_window", 10)
    min_samples = utility_config.get("min_samples", 5)

    return UtilitySignalComputer(
        utility_type=utility_type,
        gamma=gamma,
        baseline_window=baseline_window,
        advantage_window=advantage_window,
        min_samples=min_samples,
    )


# Legacy function for backward compatibility
def compute_scheduler_utility(
    trajectory: List[Dict[str, Any]],
    decision_index: int,
    value_fn: Optional[callable] = None,
    gamma: float = 0.99,
    baseline_window: int = 100,
) -> float:
    """Legacy function for immediate utility computation.

    Args:
        trajectory: Episode trajectory
        decision_index: Decision index
        value_fn: Optional value function
        gamma: Discount factor
        baseline_window: Baseline window size

    Returns:
        Utility signal
    """
    computer = UtilitySignalComputer(
        utility_type="immediate",
        gamma=gamma,
        baseline_window=baseline_window,
    )
    return computer._compute_immediate_utility(trajectory, decision_index, value_fn)
