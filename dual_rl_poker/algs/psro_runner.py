"""Policy Space Response Oracles (PSRO) implementation for poker games.

PSRO maintains a population of strategies and iteratively adds best responses
to the current population mixture, converging to Nash equilibrium.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

from algs.base import BaseAlgorithm, TrainingState, ExperienceBuffer
from utils.logging import get_experiment_logger
from eval.policy_adapter import PolicyMetadata


class PSRONetwork(nn.Module):
    """Neural network for PSRO policy learning."""

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_dims: List[int] = [128, 128],
        dropout: float = 0.1,
    ):
        """Initialize PSRO network.

        Args:
            input_size: Size of information state encoding
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build policy network
        layers = []
        prev_size = input_size

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_size, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_dim

        layers.append(nn.Linear(prev_size, num_actions))
        self.policy_network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy network.

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Policy logits [batch_size, num_actions]
        """
        return self.policy_network(x)


class Strategy:
    """Represents a learned strategy in PSRO."""

    def __init__(self, network: PSRONetwork, policy_dict: Dict[str, np.ndarray]):
        """Initialize strategy.

        Args:
            network: Neural network representing the strategy
            policy_dict: Dictionary mapping info states to action probabilities
        """
        self.network = network
        self.policy_dict = policy_dict
        self.fitness_scores = []

    def get_policy(self, info_state_key: str, legal_actions: List[int]) -> np.ndarray:
        """Get action probabilities for an info state."""
        if info_state_key in self.policy_dict:
            policy = self.policy_dict[info_state_key]
            # Apply legal action mask
            legal_mask = np.zeros(len(policy))
            legal_mask[legal_actions] = 1.0
            masked_policy = policy * legal_mask

            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                masked_policy[legal_actions] = 1.0 / len(legal_actions)

            return masked_policy
        else:
            # Fallback to uniform
            return np.full(len(legal_actions), 1.0 / len(legal_actions))


class PSROAlgorithm(BaseAlgorithm):
    """Policy Space Response Oracles algorithm implementation.

    PSRO maintains a population of strategies and iteratively:
    1. Computes meta-strategy (mixture over population)
    2. Trains best response to meta-strategy
    3. Adds best response to population
    """

    def __init__(self, game_wrapper, config: Dict[str, Any]):
        """Initialize PSRO algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Algorithm configuration
        """
        super().__init__(game_wrapper, config)

        self.experiment_logger = get_experiment_logger("psro")
        self.logger = self.experiment_logger.get_logger()

        # PSRO specific parameters
        self.population_size = config.get("population_size", 5)
        self.lr = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 256)
        self.epochs_per_iteration = config.get("epochs_per_iteration", 100)
        self.update_frequency = config.get("update_frequency", 10)
        self.gamma = config.get("gamma", 0.99)

        # Strategy population
        self.strategies: List[Strategy] = []
        self.meta_strategy: np.ndarray = np.array([])  # Weights over strategies
        self.current_iteration = 0

        # Training data
        self.experience_buffer = deque(maxlen=10000)
        self.best_response_network = None
        self.best_response_optimizer = None

        # Initialize networks
        encoding_size = game_wrapper.get_encoding_size()
        num_actions = game_wrapper.num_actions
        hidden_dims = config.get(
            "hidden_dims", config.get("network", {}).get("hidden_dims", [128, 128])
        )

        # Create initial random strategy
        initial_network = PSRONetwork(
            input_size=encoding_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=config.get("dropout", 0.1),
        )
        initial_policy = self._create_random_policy()
        initial_strategy = Strategy(initial_network, initial_policy)
        self.strategies.append(initial_strategy)

        # Initialize best response network
        self.best_response_network = PSRONetwork(
            input_size=encoding_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=0.0,  # No dropout during training
        )
        self.best_response_optimizer = torch.optim.Adam(
            self.best_response_network.parameters(), lr=self.lr
        )

        self.logger.info("Initialized PSRO algorithm")
        self.logger.info(f"Population size: {self.population_size}")
        self.logger.info(
            f"Network parameters: {sum(p.numel() for p in self.best_response_network.parameters())}"
        )

    def _create_network(self):
        """PSRO manages its own network architecture."""
        return None

    def train_step(self) -> Dict[str, float]:
        """Perform one PSRO training step."""
        state = self.train_iteration()
        return {
            "loss": state.loss,
            "gradient_norm": state.gradient_norm,
            "wall_time": state.wall_time,
        }

    def train_iteration(self) -> TrainingState:
        """Perform one PSRO training iteration."""
        start_time = time.time()

        # Step 1: Update meta-strategy using rectified Nash
        self._update_meta_strategy()

        # Step 2: Train best response to current meta-strategy
        best_response_loss = self._train_best_response()

        # Step 3: Add trained best response to population (if space allows)
        if len(self.strategies) < self.population_size:
            self._add_best_response_to_population()

        # Step 4: Collect new experience using meta-strategy
        self._collect_self_play_experience()

        # Calculate statistics
        iteration_time = time.time() - start_time

        self.current_iteration += 1

        return TrainingState(
            iteration=self.current_iteration,
            loss=best_response_loss,
            buffer_size=len(self.experience_buffer),
            wall_time=iteration_time,
            gradient_norm=0.0,  # TODO: Compute actual gradient norm
            extra_metrics={
                "population_size": len(self.strategies),
                "best_response_loss": best_response_loss,
                "meta_strategy_entropy": self._compute_meta_strategy_entropy(),
                "num_trajectories": self.batch_size // 10,
            },
        )

    def _update_meta_strategy(self):
        """Update meta-strategy using rectified Nash."""
        if len(self.strategies) <= 1:
            self.meta_strategy = np.array([1.0])
            return

        # Compute payoff matrix (simplified - using exploitability estimates)
        n = len(self.strategies)
        payoff_matrix = np.zeros((n, n))

        # For each strategy pair, estimate payoffs
        for i, strategy_i in enumerate(self.strategies):
            for j, strategy_j in enumerate(self.strategies):
                # Simplified payoff estimation
                payoff_matrix[i, j] = self._estimate_payoff(strategy_i, strategy_j)

        # Compute rectified Nash using linear programming
        # This is a simplified version - full implementation would use LP solver
        self.meta_strategy = self._compute_rectified_nash(payoff_matrix)

    def _estimate_payoff(self, strategy1: Strategy, strategy2: Strategy) -> float:
        """Estimate payoff between two strategies."""
        # Simplified payoff estimation
        # In practice, this would involve running episodes between strategies
        return random.uniform(-1, 1)  # Placeholder

    def _compute_rectified_nash(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Compute rectified Nash equilibrium from payoff matrix."""
        # Simplified implementation - would use linear programming in practice
        n = payoff_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Use uniform distribution as fallback
        meta_strategy = np.ones(n) / n

        # Simple iterative improvement
        for _ in range(10):
            # Compute expected payoffs
            expected_payoffs = payoff_matrix @ meta_strategy

            # Update strategy (rectified)
            new_strategy = np.maximum(expected_payoffs, 0)
            if new_strategy.sum() > 0:
                new_strategy = new_strategy / new_strategy.sum()
            else:
                new_strategy = np.ones(n) / n

            meta_strategy = 0.9 * meta_strategy + 0.1 * new_strategy

        return meta_strategy

    def _train_best_response(self) -> float:
        """Train best response to current meta-strategy."""
        if len(self.experience_buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs_per_iteration):
            if len(self.experience_buffer) < self.batch_size:
                break

            # Sample batch
            batch = random.sample(list(self.experience_buffer), self.batch_size)

            # Prepare training data
            info_states = torch.FloatTensor([t["info_state"] for t in batch])
            actions = torch.LongTensor([t["action"] for t in batch])
            rewards = torch.FloatTensor([t["reward"] for t in batch])
            dones = torch.FloatTensor([t["done"] for t in batch])
            next_info_states = torch.FloatTensor([t["next_info_state"] for t in batch])

            # Current Q-values
            self.best_response_optimizer.zero_grad()
            current_q = self.best_response_network(info_states)
            current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze()

            # Next Q-values (target network)
            with torch.no_grad():
                next_q = self.best_response_network(next_info_states)
                next_q_values = next_q.max(dim=1)[0]
                target_q = rewards + self.gamma * (1 - dones) * next_q_values

            # Compute loss
            loss = F.mse_loss(current_q_values, target_q)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.best_response_network.parameters(), 1.0)
            self.best_response_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _add_best_response_to_population(self):
        """Add trained best response to population."""
        # Create policy dictionary from best response network
        policy_dict = self._extract_policy_from_network(self.best_response_network)

        # Create new strategy
        new_strategy = Strategy(self.best_response_network.cpu(), policy_dict.copy())

        self.strategies.append(new_strategy)

        # Reset best response network for next iteration
        encoding_size = self.game_wrapper.get_encoding_size()
        num_actions = self.game_wrapper.num_actions
        hidden_dims = self.config.get(
            "hidden_dims", self.config.get("network", {}).get("hidden_dims", [128, 128])
        )

        self.best_response_network = PSRONetwork(
            input_size=encoding_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=0.0,
        )
        self.best_response_optimizer = torch.optim.Adam(
            self.best_response_network.parameters(), lr=self.lr
        )

        self.logger.info(
            f"Added new strategy to population. Size: {len(self.strategies)}"
        )

    def _extract_policy_from_network(
        self, network: PSRONetwork
    ) -> Dict[str, np.ndarray]:
        """Extract policy dictionary from neural network."""
        policy_dict = {}

        # Sample info states from experience buffer
        if len(self.experience_buffer) > 0:
            sample_transitions = random.sample(
                list(self.experience_buffer), min(1000, len(self.experience_buffer))
            )

            for transition in sample_transitions:
                info_state_key = transition["info_state_key"]
                legal_actions = transition["legal_actions"]

                # Get policy from network
                with torch.no_grad():
                    info_tensor = torch.FloatTensor(transition["info_state"]).unsqueeze(
                        0
                    )
                    logits = self.best_response_network(info_tensor)
                    policy_probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # Apply legal action mask
                legal_mask = np.zeros(len(policy_probs))
                legal_mask[legal_actions] = 1.0
                masked_policy = policy_probs * legal_mask

                if masked_policy.sum() > 0:
                    masked_policy = masked_policy / masked_policy.sum()
                else:
                    masked_policy[legal_actions] = 1.0 / len(legal_actions)

                policy_dict[info_state_key] = masked_policy

        return policy_dict

    def _collect_self_play_experience(self):
        """Collect experience using current meta-strategy."""
        num_episodes = max(1, self.batch_size // 10)

        for _ in range(num_episodes):
            trajectory = self._generate_meta_strategy_episode()
            self._process_trajectory(trajectory)

    def _generate_meta_strategy_episode(self) -> List[Dict[str, Any]]:
        """Generate episode using meta-strategy."""
        trajectory = []
        state = self.game_wrapper.new_initial_state()

        while not state.is_terminal():
            current_player = state.current_player()
            legal_actions = state.legal_actions()

            # Get info state
            info_state = self.game_wrapper.encode_state(state, current_player)
            info_state_key = self.game_wrapper.get_info_state_key(state, current_player)

            # Sample strategy from meta-strategy
            if len(self.strategies) > 0 and len(self.meta_strategy) > 0:
                strategy_idx = np.random.choice(
                    len(self.strategies), p=self.meta_strategy
                )
                selected_strategy = self.strategies[strategy_idx]
                action_probs = selected_strategy.get_policy(
                    info_state_key, legal_actions
                )
            else:
                # Fallback to uniform
                action_probs = np.full(len(legal_actions), 1.0 / len(legal_actions))

            action = np.random.choice(legal_actions, p=action_probs)

            # Store transition
            transition = {
                "info_state": info_state,
                "info_state_key": info_state_key,
                "legal_actions": legal_actions,
                "action": action,
                "action_probs": action_probs,
                "player": current_player,
                "iteration": self.current_iteration,
            }
            trajectory.append(transition)

            # Apply action
            state.apply_action(action)

        # Fill in rewards and next states
        for i, transition in enumerate(trajectory):
            transition["reward"] = state.returns()[transition["player"]]
            transition["done"] = i == len(trajectory) - 1

            if i < len(trajectory) - 1:
                transition["next_info_state"] = trajectory[i + 1]["info_state"]
            else:
                # Terminal state - use dummy next state
                transition["next_info_state"] = np.zeros_like(transition["info_state"])

        return trajectory

    def _process_trajectory(self, trajectory: List[Dict[str, Any]]):
        """Process trajectory and add to experience buffer."""
        for transition in trajectory:
            self.experience_buffer.append(transition)

    def _compute_meta_strategy_entropy(self) -> float:
        """Compute entropy of meta-strategy."""
        if len(self.meta_strategy) == 0:
            return 0.0

        # Remove zero probabilities
        non_zero_probs = self.meta_strategy[self.meta_strategy > 0]
        if len(non_zero_probs) == 0:
            return 0.0

        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs + 1e-8))
        return entropy

    def _create_random_policy(self) -> Dict[str, np.ndarray]:
        """Create a random policy for initialization."""
        # This would be populated during training
        return {}

    def get_policy(self, player: int) -> callable:
        """Get the current meta-strategy policy for a player."""

        def policy(info_state_key: str, legal_actions: List[int]) -> np.ndarray:
            if len(self.strategies) == 0 or len(self.meta_strategy) == 0:
                return np.full(len(legal_actions), 1.0 / len(legal_actions))

            # Sample strategy from meta-strategy
            strategy_idx = np.random.choice(len(self.strategies), p=self.meta_strategy)
            selected_strategy = self.strategies[strategy_idx]

            return selected_strategy.get_policy(info_state_key, legal_actions)

        return policy

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current meta-strategy."""
        from eval.openspiel_evaluator import OpenSpielExactEvaluator

        evaluator = OpenSpielExactEvaluator(self.game_wrapper.game_name)
        metadata = PolicyMetadata(method="psro", iteration=self.current_iteration)

        def policy_fn(
            player_id: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            return self.get_policy(player_id)(info_state, list(legal_actions))

        policy_adapter = evaluator.build_policy(policy_fn, metadata=metadata)

        # Evaluate exploitability
        results = evaluator.evaluate_policy(policy_adapter)

        return {
            "exploitability": results.get("exploitability", 0.0),
            "nash_conv": results.get("nash_conv", 0.0),
            "iteration": self.current_iteration,
            "population_size": len(self.strategies),
        }

    def save_checkpoint(self, path: str):
        """Save PSRO checkpoint."""
        checkpoint = {
            "current_iteration": self.current_iteration,
            "strategies": [
                {
                    "network_state": strategy.network.state_dict(),
                    "policy_dict": strategy.policy_dict,
                    "fitness_scores": strategy.fitness_scores,
                }
                for strategy in self.strategies
            ],
            "meta_strategy": self.meta_strategy,
            "best_response_network_state": self.best_response_network.state_dict(),
            "best_response_optimizer_state": self.best_response_optimizer.state_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved PSRO checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load PSRO checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.current_iteration = checkpoint["current_iteration"]
        self.meta_strategy = checkpoint["meta_strategy"]

        # Load strategies
        self.strategies = []
        for strategy_data in checkpoint["strategies"]:
            network = PSRONetwork(
                input_size=self.game_wrapper.get_encoding_size(),
                num_actions=self.game_wrapper.num_actions,
                hidden_dims=self.config.get(
                    "hidden_dims",
                    self.config.get("network", {}).get("hidden_dims", [128, 128]),
                ),
            )
            network.load_state_dict(strategy_data["network_state"])
            strategy = Strategy(network, strategy_data["policy_dict"])
            strategy.fitness_scores = strategy_data["fitness_scores"]
            self.strategies.append(strategy)

        # Load best response network
        self.best_response_network.load_state_dict(
            checkpoint["best_response_network_state"]
        )
        self.best_response_optimizer.load_state_dict(
            checkpoint["best_response_optimizer_state"]
        )

        self.logger.info(f"Loaded PSRO checkpoint from {path}")

    def get_policy_adapter(self):
        """Return a PolicyAdapter instance for the current meta-strategy."""

        def policy_fn(
            player: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            del player
            if len(self.strategies) == 0 or len(self.meta_strategy) == 0:
                return np.full(
                    len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64
                )

            # Use meta-strategy mixture
            mixed_policy = np.zeros(self.game_wrapper.num_actions)
            for strategy, weight in zip(self.strategies, self.meta_strategy):
                strategy_policy = strategy.get_policy(info_state, legal_actions)
                # Map back to full action space
                full_policy = np.zeros(self.game_wrapper.num_actions)
                full_policy[legal_actions] = strategy_policy
                mixed_policy += weight * full_policy

            # Extract legal actions
            legal_mixed_policy = mixed_policy[legal_actions]
            total = legal_mixed_policy.sum()
            if total <= 0:
                return np.full(
                    len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64
                )
            return (legal_mixed_policy / total).astype(np.float64)

        from eval.openspiel_evaluator import create_evaluator

        evaluator = create_evaluator(self.game_wrapper.game_name)
        return evaluator.build_policy(
            policy_fn, PolicyMetadata(method="psro", iteration=self.current_iteration)
        )
