"""Base neural network classes for poker agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np


class BaseNetwork(nn.Module, ABC):
    """Base class for all neural networks in the project."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        """Initialize base network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.num_parameters = self._count_parameters()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Dictionary with network outputs
        """
        pass

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, batch_size: int = 1) -> int:
        """Estimate FLOPs for forward pass.

        Args:
            batch_size: Batch size

        Returns:
            Estimated FLOPs
        """
        # Simple estimation based on parameters
        # Each parameter requires roughly 2 operations (multiply + add)
        return self.num_parameters * 2 * batch_size

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture information for logging.

        Returns:
            Dictionary with architecture details
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_parameters": self.num_parameters,
            "estimated_flops": self.estimate_flops(),
            "network_type": self.__class__.__name__
        }


class PolicyHead(nn.Module):
    """Policy head that outputs action probabilities."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        """Initialize policy head.

        Args:
            input_dim: Input feature dimension
            num_actions: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor, legal_actions_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features
            legal_actions_mask: Boolean mask for legal actions

        Returns:
            Action probabilities (sum to 1)
        """
        logits = self.layers(x)

        if legal_actions_mask is not None:
            # Set logits of illegal actions to very negative value
            logits = logits.masked_fill(~legal_actions_mask, -1e9)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs


class AdvantageHead(nn.Module):
    """Advantage head that outputs advantage values for actions."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        """Initialize advantage head.

        Args:
            input_dim: Input feature dimension
            num_actions: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Advantage values for each action
        """
        advantages = self.layers(x)
        return advantages


class CriticHead(nn.Module):
    """Critic head that outputs state values."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Initialize critic head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            State value (scalar)
        """
        value = self.layers(x)
        return value.squeeze(-1)  # Remove last dimension


class StandardMLP(nn.Module):
    """Standard MLP body for neural networks."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int, dropout: float = 0.0):
        """Initialize residual block.

        Args:
            dim: Feature dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        return x + self.net(x)