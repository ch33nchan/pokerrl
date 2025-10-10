"""Base classes for all algorithms."""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque

from games.base import GameWrapper
from nets.base import BaseNetwork


@dataclass
class TrainingState:
    """Training state information for logging and analysis."""
    iteration: int = 0
    loss: float = 0.0
    exploitability: float = 0.0
    nash_conv: float = 0.0
    wall_time: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    buffer_size: int = 0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'loss': self.loss,
            'exploitability': self.exploitability,
            'nash_conv': self.nash_conv,
            'wall_time': self.wall_time,
            'gradient_norm': self.gradient_norm,
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            **self.extra_metrics
        }


class ExperienceBuffer:
    """Experience buffer for storing game trajectories."""

    def __init__(self, max_size: int = 10000):
        """Initialize buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_experiences = 0

    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer.

        Args:
            experience: Experience dictionary
        """
        self.buffer.append(experience)
        self.total_experiences += 1

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample experiences from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class BaseAlgorithm(ABC):
    """Base class for all learning algorithms."""

    def __init__(self, game_wrapper: GameWrapper, config: Dict[str, Any]):
        """Initialize algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Configuration dictionary
        """
        self.game_wrapper = game_wrapper
        self.game = game_wrapper.game
        self.num_players = game_wrapper.num_players
        self.num_actions = self.game.num_distinct_actions()
        self.config = config

        # Training parameters
        self.iteration = 0
        self.device = torch.device(config.get('device', 'cpu'))
        self.batch_size = config.get('batch_size', 2048)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gradient_clip = config.get('gradient_clip', 5.0)

        # Initialize network
        self.network = self._create_network()

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.0)
        )

        # Experience buffers
        self.regret_buffer = ExperienceBuffer(config.get('regret_buffer_size', 10000))
        self.strategy_buffer = ExperienceBuffer(config.get('strategy_buffer_size', 10000))

        # Training history
        self.training_history: List[TrainingState] = []

        # Timing
        self.start_time = time.time()

    @abstractmethod
    def _create_network(self) -> BaseNetwork:
        """Create the neural network for this algorithm.

        Returns:
            Network instance
        """
        pass

    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def get_policy(self, player: int):
        """Get current policy for a player.

        Args:
            player: Player index

        Returns:
            Policy function
        """
        pass

    def _external_sampling_traversal(self) -> List[Dict[str, Any]]:
        """Perform external sampling traversal of the game.

        Returns:
            List of experiences from the traversal
        """
        experiences = []
        state = self.game.new_initial_state()
        self._traverse_state(state, experiences, np.ones(1))
        return experiences

    def _traverse_state(self, state, experiences: List[Dict[str, Any]],
                        reach_prob: np.ndarray):
        """Recursive traversal for external sampling.

        Args:
            state: Current game state
            experiences: List to collect experiences
            reach_prob: Reach probability for current player
        """
        if state.is_terminal():
            return

        if state.is_chance_node():
            # Sample chance action
            action = np.random.choice(state.legal_actions(),
                                     p=[state.chance_outcomes()[a][1]
                                        for a in state.legal_actions()])
            new_state = state.child(action)
            self._traverse_state(new_state, experiences, reach_prob)
            return

        current_player = state.current_player()
        legal_actions = state.legal_actions()

        # Get information state
        info_state = self.game_wrapper.encode_state(state)
        legal_actions_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        legal_actions_mask[legal_actions] = True

        # Sample action according to current strategy
        with torch.no_grad():
            network_output = self.network(
                torch.tensor(info_state, dtype=torch.float32).unsqueeze(0).to(self.device),
                legal_actions_mask.unsqueeze(0).to(self.device)
            )

            if 'policy' in network_output:
                action_probs = network_output['policy'].cpu().numpy()[0]
            else:
                action_probs = np.ones(len(legal_actions)) / len(legal_actions)

        # Ensure probabilities match legal actions
        action_probs = np.array([action_probs[a] for a in legal_actions])
        action_probs = action_probs / action_probs.sum()

        # Sample action
        action_idx = np.random.choice(len(legal_actions), p=action_probs)
        action = legal_actions[action_idx]

        # Store experience
        experiences.append({
            'info_state': info_state,
            'legal_actions': legal_actions,
            'legal_actions_mask': legal_actions_mask.numpy(),
            'action': action,
            'action_idx': action_idx,
            'player': current_player,
            'reach_prob': reach_prob[0]
        })

        # Continue traversal
        new_state = state.child(action)
        self._traverse_state(new_state, experiences, reach_prob)

    def _compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute training loss from batch.

        Args:
            batch: Batch of experiences

        Returns:
            Loss tensor
        """
        # Extract batch data
        info_states = torch.stack([torch.tensor(exp['info_state'], dtype=torch.float32)
                                 for exp in batch]).to(self.device)
        legal_actions_masks = torch.stack([torch.tensor(exp['legal_actions_mask'], dtype=torch.bool)
                                         for exp in batch]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)

        # Forward pass
        network_output = self.network(info_states, legal_actions_masks)

        # Compute loss based on algorithm type
        if self.config.get('algorithm_type') == 'regret':
            # Regret matching loss (MSE)
            if 'advantages' in network_output:
                advantages = network_output['advantages']
                target_advantages = self._compute_regret_targets(batch)
                loss = nn.functional.mse_loss(advantages, target_advantages)
            else:
                raise ValueError("Advantages not found in network output")
        elif self.config.get('algorithm_type') == 'strategy':
            # Strategy loss (cross-entropy)
            if 'policy' in network_output:
                policy = network_output['policy']
                target_policy = self._compute_strategy_targets(batch)
                loss = -(target_policy * torch.log(policy + 1e-8)).sum(dim=-1).mean()
            else:
                raise ValueError("Policy not found in network output")
        else:
            raise ValueError(f"Unknown algorithm type: {self.config.get('algorithm_type')}")

        return loss

    def _compute_regret_targets(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute regret targets for training.

        Args:
            batch: Batch of experiences

        Returns:
            Target advantages tensor
        """
        # Simple implementation - in practice, this would compute
        # actual counterfactual regret values
        batch_size = len(batch)
        num_actions = self.num_actions
        targets = torch.zeros(batch_size, num_actions, device=self.device)

        for i, exp in enumerate(batch):
            action = exp['action']
            # Simple regret target (would be computed from CFR in practice)
            targets[i, action] = 1.0  # Placeholder

        return targets

    def _compute_strategy_targets(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute strategy targets for training.

        Args:
            batch: Batch of experiences

        Returns:
            Target policy tensor
        """
        batch_size = len(batch)
        num_actions = self.num_actions
        targets = torch.zeros(batch_size, num_actions, device=self.device)

        for i, exp in enumerate(batch):
            legal_actions = exp['legal_actions']
            # Uniform distribution over legal actions (placeholder)
            for action in legal_actions:
                targets[i, action] = 1.0 / len(legal_actions)

        return targets

    def train_iteration(self) -> TrainingState:
        """Perform one training iteration.

        Returns:
            Training state with metrics
        """
        start_time = time.time()

        # Generate experiences via external sampling
        experiences = self._external_sampling_traversal()

        # Add experiences to buffers
        for exp in experiences:
            if self.config.get('algorithm_type') == 'regret':
                self.regret_buffer.add(exp)
            else:
                self.strategy_buffer.add(exp)

        # Sample batch and compute loss
        if self.config.get('algorithm_type') == 'regret':
            batch = self.regret_buffer.sample(self.batch_size)
        else:
            batch = self.strategy_buffer.sample(self.batch_size)

        if len(batch) > 0:
            loss = self._compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Compute gradient norm
            total_norm = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        else:
            loss = torch.tensor(0.0)
            total_norm = 0.0

        self.iteration += 1
        wall_time = time.time() - start_time

        # Create training state
        training_state = TrainingState(
            iteration=self.iteration,
            loss=loss.item(),
            wall_time=wall_time,
            gradient_norm=total_norm,
            learning_rate=self.learning_rate,
            buffer_size=len(self.regret_buffer) if self.config.get('algorithm_type') == 'regret'
                        else len(self.strategy_buffer),
            extra_metrics={
                'algorithm_type': self.config.get('algorithm_type'),
                'batch_size': len(batch)
            }
        )

        self.training_history.append(training_state)
        return training_state

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy.

        Returns:
            Dictionary with evaluation metrics
        """
        # This would be implemented using the OpenSpiel evaluator
        # For now, return placeholder metrics
        return {
            'exploitability': 0.1,
            'nash_conv': 0.2,
            'sampled_return': 0.0
        }