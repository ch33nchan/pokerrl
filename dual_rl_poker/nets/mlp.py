"""MLP network implementations for Deep CFR."""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base import BaseNetwork, PolicyHead, AdvantageHead, StandardMLP


class MLPNetwork(BaseNetwork):
    """MLP network with separate policy and advantage heads for Deep CFR."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list = None):
        """Initialize MLP network.

        Args:
            input_dim: Dimension of input features
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__(input_dim, hidden_dims or [64, 64])
        self.num_actions = num_actions

        # Shared body
        self.body = StandardMLP(input_dim, self.hidden_dims, dropout=0.0)

        # Separate heads
        self.policy_head = PolicyHead(self.hidden_dims[-1], num_actions)
        self.advantage_head = AdvantageHead(self.hidden_dims[-1], num_actions)

    def forward(self, x: torch.Tensor, legal_actions_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            legal_actions_mask: Boolean mask for legal actions of shape (batch_size, num_actions)

        Returns:
            Dictionary with 'policy' and 'advantages' keys
        """
        # Shared representation
        features = self.body(x)

        # Policy output (probabilities)
        policy = self.policy_head(features, legal_actions_mask)

        # Advantage output (values)
        advantages = self.advantage_head(features)

        return {
            'policy': policy,
            'advantages': advantages,
            'features': features
        }


class DeepCFRNetwork(BaseNetwork):
    """Deep CFR network with regret and strategy networks."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list = None):
        """Initialize Deep CFR network.

        Args:
            input_dim: Dimension of input features
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__(input_dim, hidden_dims or [64, 64])
        self.num_actions = num_actions

        # Regret network (for advantage prediction)
        self.regret_body = StandardMLP(input_dim, self.hidden_dims, dropout=0.0)
        self.regret_head = AdvantageHead(self.hidden_dims[-1], num_actions)

        # Strategy network (for policy prediction)
        self.strategy_body = StandardMLP(input_dim, self.hidden_dims, dropout=0.0)
        self.strategy_head = PolicyHead(self.hidden_dims[-1], num_actions)

    def forward(self, x: torch.Tensor, network_type: str = 'regret',
                legal_actions_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            network_type: Either 'regret' or 'strategy'
            legal_actions_mask: Boolean mask for legal actions

        Returns:
            Dictionary with appropriate network outputs
        """
        if network_type == 'regret':
            features = self.regret_body(x)
            advantages = self.regret_head(features)
            return {
                'advantages': advantages,
                'features': features,
                'network_type': 'regret'
            }
        elif network_type == 'strategy':
            features = self.strategy_body(x)
            policy = self.strategy_head(features, legal_actions_mask)
            return {
                'policy': policy,
                'features': features,
                'network_type': 'strategy'
            }
        else:
            raise ValueError(f"Invalid network_type: {network_type}")


class SDCFRNetwork(BaseNetwork):
    """Self-Play Deep CFR network."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list = None):
        """Initialize SD-CFR network.

        Args:
            input_dim: Dimension of input features
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__(input_dim, hidden_dims or [64, 64])
        self.num_actions = num_actions

        # Single network that outputs both policy and advantages
        self.body = StandardMLP(input_dim, self.hidden_dims, dropout=0.0)
        self.policy_head = PolicyHead(self.hidden_dims[-1], num_actions)
        self.advantage_head = AdvantageHead(self.hidden_dims[-1], num_actions)

        # Additional head for value estimation
        from .base import CriticHead
        self.value_head = CriticHead(self.hidden_dims[-1])

    def forward(self, x: torch.Tensor, legal_actions_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            legal_actions_mask: Boolean mask for legal actions

        Returns:
            Dictionary with all outputs
        """
        features = self.body(x)

        return {
            'policy': self.policy_head(features, legal_actions_mask),
            'advantages': self.advantage_head(features),
            'value': self.value_head(features),
            'features': features
        }


class TabularNetwork(BaseNetwork):
    """Tabular network for small games (baseline)."""

    def __init__(self, input_dim: int, num_actions: int, table_size: int = None):
        """Initialize tabular network.

        Args:
            input_dim: Dimension of input features (for hashing)
            num_actions: Number of possible actions
            table_size: Size of lookup table
        """
        super().__init__(input_dim, [])
        self.num_actions = num_actions
        self.table_size = table_size or 10000

        # Initialize tables
        self.register_buffer('policy_table', torch.zeros(self.table_size, num_actions))
        self.register_buffer('advantage_table', torch.zeros(self.table_size, num_actions))
        self.register_buffer('count_table', torch.zeros(self.table_size))

    def forward(self, x: torch.Tensor, legal_actions_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            legal_actions_mask: Boolean mask for legal actions

        Returns:
            Dictionary with tabular outputs
        """
        # Simple hashing from input to table index
        indices = self._hash_inputs(x) % self.table_size

        policy = self.policy_table[indices]
        advantages = self.advantage_table[indices]

        # Apply legal actions mask
        if legal_actions_mask is not None:
            policy = policy.masked_fill(~legal_actions_mask, 0)
            advantages = advantages.masked_fill(~legal_actions_mask, 0)

        # Normalize policy
        policy_sums = policy.sum(dim=-1, keepdim=True)
        policy = policy / (policy_sums + 1e-8)

        return {
            'policy': policy,
            'advantages': advantages,
            'indices': indices
        }

    def _hash_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Simple hash function for inputs.

        Args:
            x: Input tensor

        Returns:
            Hash indices
        """
        # Simple sum-based hashing (for demonstration)
        # In practice, you'd want a better hashing scheme
        hash_values = torch.sum(x * torch.arange(1, x.shape[-1] + 1, device=x.device), dim=-1)
        return hash_values.long()

    def update_table(self, indices: torch.Tensor, policy_updates: torch.Tensor,
                    advantage_updates: torch.Tensor):
        """Update table entries.

        Args:
            indices: Table indices to update
            policy_updates: Policy update values
            advantage_updates: Advantage update values
        """
        self.policy_table[indices] = policy_updates
        self.advantage_table[indices] = advantage_updates
        self.count_table[indices] += 1


class ARMACNetwork(BaseNetwork):
    """ARMAC network for Actor-Critic with Regret Matching."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list = None, network_type: str = 'actor'):
        """Initialize ARMAC network.

        Args:
            input_dim: Dimension of input features
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            network_type: One of 'actor', 'critic', 'regret'
        """
        super().__init__(input_dim, hidden_dims or [64, 64])
        self.num_actions = num_actions
        self.network_type = network_type

        # Shared body
        self.body = StandardMLP(input_dim, self.hidden_dims, dropout=0.0)

        # Network-specific heads
        if network_type == 'actor':
            self.policy_head = PolicyHead(self.hidden_dims[-1], num_actions)
        elif network_type == 'critic':
            self.value_head = CriticHead(self.hidden_dims[-1])
        elif network_type == 'regret':
            self.regret_head = AdvantageHead(self.hidden_dims[-1], num_actions)
        else:
            raise ValueError(f"Invalid network_type: {network_type}")

    def forward(self, x: torch.Tensor, legal_actions_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            legal_actions_mask: Boolean mask for legal actions (for actor and regret)

        Returns:
            Dictionary with network outputs
        """
        features = self.body(x)

        if self.network_type == 'actor':
            action_probs = self.policy_head(features, legal_actions_mask)
            return {
                'action_probs': action_probs,
                'features': features
            }
        elif self.network_type == 'critic':
            value = self.value_head(features)
            return {
                'value': value.unsqueeze(-1),  # Add dimension for consistency
                'features': features
            }
        elif self.network_type == 'regret':
            regrets = self.regret_head(features)
            # Apply legal actions mask if provided
            if legal_actions_mask is not None:
                regrets = regrets.masked_fill(~legal_actions_mask, 0)
            return {
                'action_probs': torch.softmax(regrets, dim=-1),  # Return as probabilities
                'regrets': regrets,
                'features': features
            }