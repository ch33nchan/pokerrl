import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class ActorCriticAgent(nn.Module):
    """
    A standard Actor-Critic neural network architecture.
    This class defines a simple yet effective model for reinforcement learning,
    featuring a shared body with separate heads for policy (actor) and value (critic) estimation.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        """
        Initializes the ActorCriticAgent.
        Args:
            state_dim: The dimensionality of the state space.
            action_dim: The number of possible actions.
            hidden_dim: The number of units in the hidden layers.
        """
        super().__init__()
        
        # Shared layers process the input state
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # The policy head (Actor) outputs logits for each action
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # The value head (Critic) outputs a single value estimating the state's worth
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.
        Args:
            state: A tensor representing the current state of the environment.
        Returns:
            A tuple containing:
            - policy_logits: Raw, unnormalized scores for each action.
            - value: A single value estimating the utility of the state.
        """
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(next(self.parameters()).device)
            
        # Pass the state through the shared layers
        shared_features = self.shared_layer(state)
        
        # Get policy logits and state value from the respective heads
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return policy_logits, value

