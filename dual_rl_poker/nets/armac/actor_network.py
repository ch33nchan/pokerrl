"""Actor network for ARMAC algorithm.

This module implements the actor network that outputs policy probabilities
for actions given information states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class ActorNetwork(nn.Module):
    """Actor network for policy approximation in ARMAC.

    The actor network takes information state encodings and outputs
    action probabilities for each legal action.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: list = [64, 64],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize actor network.

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

        # Output layer
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
        """Forward pass through actor network.

        Args:
            states: Information state tensors [batch_size, state_dim]

        Returns:
            Action logits [batch_size, num_actions]
        """
        return self.network(states)

    def get_action_probs(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get action probabilities with legal actions masking.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Action probabilities [batch_size, num_actions]
        """
        logits = self.forward(states)

        if legal_actions_mask is not None:
            # Apply legal actions mask
            masked_logits = logits + (1 - legal_actions_mask.float()) * (-1e9)
            return F.softmax(masked_logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def get_log_probs(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get log action probabilities.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Log action probabilities [batch_size, num_actions]
        """
        logits = self.forward(states)

        if legal_actions_mask is not None:
            # Apply legal actions mask
            masked_logits = logits + (1 - legal_actions_mask.float()) * (-1e9)
            return F.log_softmax(masked_logits, dim=-1)
        else:
            return F.log_softmax(logits, dim=-1)

    def sample_action(
        self, states: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.

        Args:
            states: Information state tensors [batch_size, state_dim]
            legal_actions_mask: Legal actions mask [batch_size, num_actions]

        Returns:
            Tuple of (actions, log_probs)
        """
        log_probs = self.get_log_probs(states, legal_actions_mask)

        # Sample actions
        probs = torch.exp(log_probs)
        actions = torch.multinomial(probs.view(-1, self.num_actions), 1).view(-1)

        # Get log probs of sampled actions
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        return actions, selected_log_probs


def create_actor_network(config: Dict[str, Any]) -> ActorNetwork:
    """Factory function to create actor network from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        ActorNetwork instance
    """
    state_dim = config.get("state_dim", 10)
    num_actions = config.get("num_actions", 2)
    hidden_dims = config.get("hidden_dims", [64, 64])
    dropout = config.get("dropout", 0.0)
    activation = config.get("activation", "relu")

    return ActorNetwork(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )
