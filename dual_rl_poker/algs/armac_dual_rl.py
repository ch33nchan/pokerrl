"""
ARMAC dual RL advantage computation and regret-matching updates.

This module implements the core ARMAC dual RL functionality:
- Proper advantage computation: A(I,a) = q_θ(I,a) - Σ_a' π(a'|I)q_θ(I,a')
- Regret-matching policy updates: π_{t+1}(a|I) ∝ max(A(I,a), 0)
- Actor/regret mixture with sweepable weight λ
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class ARMACDualRL:
    """ARMAC dual RL advantage computation and policy updates."""

    def __init__(
        self,
        num_actions: int,
        mixture_weight: float = 0.1,
        lambda_mode: str = "fixed",
        lambda_alpha: float = 0.5,
    ):
        """
        Initialize ARMAC dual RL.

        Args:
            num_actions: Number of actions in the game
            mixture_weight: Weight λ for mixing actor and regret policies
            lambda_mode: Mode for lambda computation ("fixed" or "adaptive")
            lambda_alpha: Alpha parameter for adaptive lambda computation
        """
        self.num_actions = num_actions
        self.mixture_weight = mixture_weight
        self.lambda_mode = lambda_mode
        self.lambda_alpha = lambda_alpha

        # For adaptive lambda scheduling
        self.avg_regret_loss = 0.0
        self.avg_policy_loss = 0.0

    def compute_lambda_t(
        self, avg_regret_loss: float, avg_policy_loss: float, alpha: float = 0.5
    ) -> float:
        """
        Compute adaptive lambda based on loss differences.

        Args:
            avg_regret_loss: Average regret loss
            avg_policy_loss: Average policy loss
            alpha: Scaling factor for adaptation

        Returns:
            Adaptive lambda value in [0, 1]
        """
        diff = avg_regret_loss - avg_policy_loss
        import torch

        return torch.sigmoid(torch.tensor(alpha * diff)).item()

    def update_loss_averages(
        self, current_regret_loss: float, current_policy_loss: float, beta: float = 0.9
    ):
        """
        Update exponential moving averages for losses.

        Args:
            current_regret_loss: Current regret loss
            current_policy_loss: Current policy loss
            beta: EMA decay factor
        """
        self.avg_regret_loss = (
            beta * self.avg_regret_loss + (1 - beta) * current_regret_loss
        )
        self.avg_policy_loss = (
            beta * self.avg_policy_loss + (1 - beta) * current_policy_loss
        )

    def compute_advantages(
        self,
        info_states: torch.Tensor,
        critic_network: torch.nn.Module,
        actor_network: torch.nn.Module,
        legal_actions_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ARMAC advantages: A(I,a) = q_θ(I,a) - Σ_a' π(a'|I)q_θ(I,a').

        Args:
            info_states: Information state encodings [batch_size, state_dim]
            critic_network: Critic network for Q-value estimation
            actor_network: Actor network for policy π(a|I)
            legal_actions_masks: Legal actions masks [batch_size, num_actions]

        Returns:
            Advantages for all actions [batch_size, num_actions]
        """
        with torch.no_grad():
            # Get Q-values for all actions: q_θ(I,a)
            critic_output = critic_network(info_states)
            if isinstance(critic_output, dict):
                # Try different possible keys for the value output
                q_values = critic_output.get(
                    "q_values",
                    critic_output.get("values", critic_output.get("value", None)),
                )

                if q_values is None:
                    raise ValueError(
                        "Critic network output does not contain 'q_values', 'values', or 'value' key"
                    )

                if q_values.shape[-1] == 1:
                    # If critic outputs single value, expand to all actions
                    # This is a simplification - ideally we'd have action-conditional critic
                    q_values = q_values.expand(-1, self.num_actions)
            else:
                q_values = critic_output
                if q_values.shape[-1] == 1:
                    q_values = q_values.expand(-1, self.num_actions)

            # Get policy probabilities: π(a|I)
            actor_output = actor_network(info_states)
            if isinstance(actor_output, dict):
                policy_logits = actor_output.get(
                    "action_probs", actor_output.get("policy")
                )
            else:
                policy_logits = actor_output

            # Apply legal actions mask and normalize
            legal_mask_float = legal_actions_masks.float()
            masked_logits = policy_logits + (1 - legal_mask_float) * (-1e9)
            policy_probs = F.softmax(masked_logits, dim=-1)

            # Compute baseline: Σ_a' π(a'|I)q_θ(I,a')
            baseline = torch.sum(policy_probs * q_values, dim=-1, keepdim=True)

            # Compute advantages: A(I,a) = q_θ(I,a) - baseline
            advantages = q_values - baseline

            # Apply legal actions mask to advantages
            advantages = advantages * legal_mask_float

        return advantages

    def regret_matching_policy(
        self, advantages: torch.Tensor, legal_actions_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regret-matching policy: π_{t+1}(a|I) ∝ max(A(I,a), 0).

        Args:
            advantages: Advantage values [batch_size, num_actions]
            legal_actions_masks: Legal actions masks [batch_size, num_actions]

        Returns:
            Regret-matching policy probabilities [batch_size, num_actions]
        """
        # Positive regrets only
        positive_advantages = torch.clamp(advantages, min=0.0)

        # Apply legal actions mask
        legal_mask_float = legal_actions_masks.float()
        masked_advantages = positive_advantages * legal_mask_float

        # Normalize to probabilities
        regret_sum = masked_advantages.sum(dim=-1, keepdim=True)
        uniform_policy = legal_mask_float / (
            legal_mask_float.sum(dim=-1, keepdim=True) + 1e-8
        )

        regret_policy = torch.where(
            regret_sum > 1e-8, masked_advantages / (regret_sum + 1e-8), uniform_policy
        )

        return regret_policy

    def mixed_policy_update(
        self,
        actor_policy: torch.Tensor,
        regret_policy: torch.Tensor,
        legal_actions_masks: torch.Tensor,
        mixture_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute mixed policy: π_mixed = (1-λ)π_actor + λπ_regret.

        Args:
            actor_policy: Actor policy probabilities [batch_size, num_actions]
            regret_policy: Regret-matching policy [batch_size, num_actions]
            legal_actions_masks: Legal actions masks [batch_size, num_actions]
            mixture_weight: Optional override for mixture weight λ

        Returns:
            Mixed policy probabilities [batch_size, num_actions]
        """
        if mixture_weight is None:
            if self.lambda_mode == "adaptive":
                mixture_weight = self.compute_lambda_t(
                    self.avg_regret_loss, self.avg_policy_loss, self.lambda_alpha
                )
            else:
                mixture_weight = self.mixture_weight

        # Ensure both policies are properly normalized
        legal_mask_float = legal_actions_masks.float()

        # Normalize actor policy
        actor_masked = actor_policy * legal_mask_float
        actor_sum = actor_masked.sum(dim=-1, keepdim=True)
        uniform_policy = legal_mask_float / (
            legal_mask_float.sum(dim=-1, keepdim=True) + 1e-8
        )
        actor_normalized = torch.where(
            actor_sum > 1e-8, actor_masked / (actor_sum + 1e-8), uniform_policy
        )

        # Mix policies
        mixed_policy = (
            1 - mixture_weight
        ) * actor_normalized + mixture_weight * regret_policy

        # Final normalization
        mixed_sum = mixed_policy.sum(dim=-1, keepdim=True)
        mixed_policy = torch.where(
            mixed_sum > 1e-8, mixed_policy / (mixed_sum + 1e-8), uniform_policy
        )

        return mixed_policy

    def compute_policy_loss(
        self,
        actor_logits: torch.Tensor,
        target_policy: torch.Tensor,
        legal_actions_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy loss for actor network toward target policy.

        Args:
            actor_logits: Raw actor network logits [batch_size, num_actions]
            target_policy: Target policy probabilities [batch_size, num_actions]
            legal_actions_masks: Legal actions masks [batch_size, num_actions]

        Returns:
            Cross-entropy loss scalar
        """
        # Apply legal actions mask to logits
        legal_mask_float = legal_actions_masks.float()
        masked_logits = actor_logits + (1 - legal_mask_float) * (-1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        # Cross-entropy loss
        loss = -(target_policy * log_probs).sum(dim=-1).mean()

        return loss


def create_armac_dual_rl(num_actions: int, config: Dict[str, Any]) -> ARMACDualRL:
    """
    Factory function to create ARMAC dual RL instance.

    Args:
        num_actions: Number of actions in the game
        config: Configuration dictionary

    Returns:
        ARMACDualRL instance
    """
    mixture_weight = config.get("regret_weight", config.get("mixture_weight", 0.1))
    lambda_mode = config.get("lambda_mode", "fixed")
    lambda_alpha = config.get("lambda_alpha", 0.5)
    return ARMACDualRL(num_actions, mixture_weight, lambda_mode, lambda_alpha)
