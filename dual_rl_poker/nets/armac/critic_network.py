"""Critic network for ARMAC algorithm.

This module implements the critic network that estimates Q-values
for state-action pairs in the ARMAC algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class CriticNetwork(nn.Module):
    """Critic network for Q-value approximation in ARMAC.

    The critic network takes information state encodings and outputs
    Q-values for each possible action.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: list = [64, 64],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize critic network.

        Args:
            state_dim: Dimension of information state encoding
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer - outputs Q-values for all actions
        layers.append(nn.Linear(input_dim, num_actions))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network.

        Args:
            states: Information state tensors [batch_size, state_dim]

        Returns:
            Q-values for all actions [batch_size, num_actions]
        """
        return self.network(states)

    def get_q_values(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get Q-values with legal actions masking.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Q-values [batch_size, num_actions]
        """
        q_values = self.forward(states)

        if legal_actions_mask is not None:
            # Set Q-values of illegal actions to very negative value
            mask_float = legal_actions_mask.float()
            masked_q_values = q_values * mask_float + (1 - mask_float) * (-1e9)
            return masked_q_values
        else:
            return q_values

    def get_action_q_values(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Get Q-values for specific actions.

        Args:
            states: Information state tensors [batch_size, state_dim]
            actions: Action indices [batch_size]

        Returns:
            Q-values for specified actions [batch_size]
        """
        q_values = self.forward(states)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    def get_value_function(
        self,
        states: torch.Tensor,
        policy_probs: Optional[torch.Tensor] = None,
        legal_actions_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get state value function V(s) = Q(s,a) * π(a|s).

        Args:
            states: Information state tensors [batch_size, state_dim]
            policy_probs: Policy probabilities [batch_size, num_actions]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            State values [batch_size]
        """
        q_values = self.get_q_values(states, legal_actions_mask)

        if policy_probs is not None:
            # Use provided policy
            if legal_actions_mask is not None:
                policy_probs = policy_probs * legal_actions_mask.float()
                policy_probs = policy_probs / (
                    policy_probs.sum(dim=-1, keepdim=True) + 1e-8
                )

            values = (q_values * policy_probs).sum(dim=-1)
        else:
            # Use max Q-value as value estimate
            if legal_actions_mask is not None:
                mask_float = legal_actions_mask.float()
                masked_q_values = q_values * mask_float + (1 - mask_float) * (-1e9)
                values = masked_q_values.max(dim=-1)[0]
            else:
                values = q_values.max(dim=-1)[0]

        return values


def create_critic_network(config: Dict[str, Any]) -> CriticNetwork:
    """Factory function to create critic network from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        CriticNetwork instance
    """
    state_dim = config.get("state_dim", 10)
    num_actions = config.get("num_actions", 2)
    hidden_dims = config.get("hidden_dims", [64, 64])
    dropout = config.get("dropout", 0.0)
    activation = config.get("activation", "relu")

    return CriticNetwork(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )
