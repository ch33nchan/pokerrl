import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MultiObjectiveLoss(nn.Module):
    """
    A more rigorous multi-objective loss function for the poker agent.

    This loss function addresses the feedback by:
    1.  Using standard, well-defined Policy Loss (Cross-Entropy) and Value Loss (MSE).
    2.  Introducing an Entropy Bonus to encourage exploration and prevent premature convergence,
        a standard technique in modern RL.
    3.  Removing the ill-defined "commitment" and "deception" losses.
    """
    def __init__(self, 
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        super().__init__()
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def forward(self,
                policy_logits: torch.Tensor,
                value_preds: torch.Tensor,
                actions: torch.Tensor,
                returns: torch.Tensor,
                advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates the multi-objective loss for an Actor-Critic agent.

        Args:
            policy_logits: Raw output from the policy head of the network.
            value_preds: Raw output from the value head of the network.
            actions: The actions taken by the agent during the trajectory.
            returns: The discounted returns (G_t) calculated from the episode.
            advantages: The advantage function (e.g., GAE).

        Returns:
            A dictionary containing the total loss and its individual components.
        """
        # 1. Policy Loss (Actor)
        # The advantages are used to weight the loss, so that actions that led to 
        # better-than-expected outcomes are encouraged.
        policy_loss = F.cross_entropy(policy_logits, actions, reduction='none')
        actor_loss = (policy_loss * advantages.detach()).mean()

        # 2. Value Loss (Critic)
        # Measures how well the critic is estimating the value of states.
        # It's the mean squared error between the predicted value and the actual returns.
        # Ensure value_preds is the same shape as returns to avoid broadcasting errors
        critic_loss = F.mse_loss(value_preds.view_as(returns), returns)

        # 3. Entropy Bonus (Exploration)
        # Encourages the policy to be more stochastic, promoting exploration.
        policy_dist = F.softmax(policy_logits, dim=-1)
        log_policy_dist = F.log_softmax(policy_logits, dim=-1)
        entropy = -(policy_dist * log_policy_dist).sum(dim=-1).mean()
        
        # We subtract the entropy bonus from the loss, which is equivalent to maximizing entropy.
        entropy_loss = -self.entropy_coef * entropy

        # Total Loss
        total_loss = (
            actor_loss + 
            self.value_loss_coef * critic_loss + 
            entropy_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': actor_loss,
            'value_loss': critic_loss,
            'entropy_bonus': -entropy_loss, # Report as a positive bonus for logging
        }