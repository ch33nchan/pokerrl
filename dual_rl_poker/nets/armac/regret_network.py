"""Regret network for ARMAC algorithm.

This module implements the regret network that estimates counterfactual regrets
for state-action pairs in the ARMAC algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class RegretNetwork(nn.Module):
    """Regret network for counterfactual regret approximation in ARMAC.

    The regret network takes information state encodings and outputs
    regret values for each possible action, which are then used
    for regret-matching policy updates.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: list = [64, 64],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize regret network.

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

        # Output layer - outputs regret values for all actions
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
        """Forward pass through regret network.

        Args:
            states: Information state tensors [batch_size, state_dim]

        Returns:
            Regret values for all actions [batch_size, num_actions]
        """
        return self.network(states)

    def get_regret_values(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get regret values with legal actions masking.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Regret values [batch_size, num_actions]
        """
        regret_values = self.forward(states)

        if legal_actions_mask is not None:
            # Set regret values of illegal actions to zero
            mask_float = legal_actions_mask.float()
            masked_regret_values = regret_values * mask_float
            return masked_regret_values
        else:
            return regret_values

    def get_positive_regret(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get positive regret values only (for regret matching).

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Positive regret values [batch_size, num_actions]
        """
        regret_values = self.get_regret_values(states, legal_actions_mask)
        return torch.clamp(regret_values, min=0.0)

    def get_action_regret(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Get regret values for specific actions.

        Args:
            states: Information state tensors [batch_size, state_dim]
            actions: Action indices [batch_size]

        Returns:
            Regret values for specified actions [batch_size]
        """
        regret_values = self.forward(states)
        return regret_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    def compute_regret_matching_policy(
        self,
        states: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute regret-matching policy from regret values.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]
            temperature: Temperature parameter for soft regret matching

        Returns:
            Regret-matching policy probabilities [batch_size, num_actions]
        """
        positive_regret = self.get_positive_regret(states, legal_actions_mask)

        if legal_actions_mask is not None:
            mask_float = legal_actions_mask.float()

            # Compute regret sum for normalization
            regret_sum = positive_regret.sum(dim=-1, keepdim=True)

            # Uniform policy for states with no positive regret
            uniform_policy = mask_float / (mask_float.sum(dim=-1, keepdim=True) + 1e-8)

            # Regret-matching policy
            if temperature != 1.0:
                positive_regret = positive_regret / temperature

            regret_policy = torch.where(
                regret_sum > 1e-8, positive_regret / (regret_sum + 1e-8), uniform_policy
            )
        else:
            regret_sum = positive_regret.sum(dim=-1, keepdim=True)
            regret_policy = torch.where(
                regret_sum > 1e-8,
                positive_regret / (regret_sum + 1e-8),
                torch.ones_like(positive_regret) / self.num_actions,
            )

        return regret_policy

    def get_mean_regret_magnitude(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get mean absolute regret magnitude.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Mean absolute regret value
        """
        regret_values = self.get_regret_values(states, legal_actions_mask)
        return torch.mean(torch.abs(regret_values)).item()


def create_regret_network(config: Dict[str, Any]) -> RegretNetwork:
    """Factory function to create regret network from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        RegretNetwork instance
    """
    state_dim = config.get("state_dim", 10)
    num_actions = config.get("num_actions", 2)
    hidden_dims = config.get("hidden_dims", [64, 64])
    dropout = config.get("dropout", 0.0)
    activation = config.get("activation", "relu")

    return RegretNetwork(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )
