"""
ARMAC dual RL advantage computation and regret-matching updates.

This module implements the core ARMAC dual RL functionality:
- Proper advantage computation: A(I,a) = q_θ(I,a) - Σ_a' π(a'|I)q_θ(I,a')
- Regret-matching policy updates: π_{t+1}(a|I) ∝ max(A(I,a), 0)
- Actor/regret mixture with per-instance lambda from scheduler
- Integration with scheduler and meta-regret components
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from algs.scheduler.scheduler import Scheduler, compute_scheduler_input
from algs.scheduler.policy_mixer import PolicyMixer
from algs.scheduler.meta_regret import MetaRegretManager, compute_state_key_simple


class ARMACDualRL:
    """ARMAC dual RL advantage computation and policy updates with scheduler integration."""

    def __init__(
        self,
        num_actions: int,
        config: Dict[str, Any],
        scheduler: Optional[Scheduler] = None,
        policy_mixer: Optional[PolicyMixer] = None,
        meta_regret: Optional[MetaRegretManager] = None,
    ):
        """
        Initialize ARMAC dual RL with scheduler components.

        Args:
            num_actions: Number of actions in the game
            config: Configuration dictionary
            scheduler: Optional scheduler network
            policy_mixer: Optional policy mixer
            meta_regret: Optional meta-regret manager
        """
        self.num_actions = num_actions
        self.config = config

        # Scheduler components
        self.scheduler = scheduler
        self.policy_mixer = policy_mixer
        self.meta_regret = meta_regret

        # Legacy support for fixed lambda
        self.mixture_weight = config.get("mixture_weight", 0.1)
        self.lambda_mode = config.get("lambda_mode", "fixed")
        self.lambda_alpha = config.get("lambda_alpha", 2.0)

        # For adaptive lambda scheduling
        self.avg_regret_loss = 0.0
        self.avg_policy_loss = 0.0

        # Enhanced adaptive tracking
        self.regret_loss_history = []
        self.policy_loss_history = []
        self.lambda_history = []
        self.performance_trend = 0.0
        self.iteration_count = 0

        # Scheduler-specific tracking
        self.scheduler_inputs = []
        self.scheduler_choices = []
        self.scheduler_utilities = []

    def compute_lambda_t(
        self, avg_regret_loss: float, avg_policy_loss: float, alpha: float = 2.0
    ) -> float:
        """
        Compute adaptive lambda based on loss differences with enhanced adaptation.

        Args:
            avg_regret_loss: Average regret loss
            avg_policy_loss: Average policy loss
            alpha: Scaling factor for adaptation

        Returns:
            Adaptive lambda value in [0, 1]
        """
        diff = avg_regret_loss - avg_policy_loss
        import torch

        # More aggressive adaptation with larger alpha
        base_lambda = torch.sigmoid(torch.tensor(alpha * diff)).item()

        # Add trend-based adjustment
        trend_adjustment = 0.0
        if len(self.regret_loss_history) >= 5:
            # Compute recent trends
            recent_regret_trend = np.mean(np.diff(self.regret_loss_history[-3:]))
            recent_policy_trend = np.mean(np.diff(self.policy_loss_history[-3:]))

            # If regret loss is decreasing faster than policy loss, increase lambda
            if recent_regret_trend < recent_policy_trend:
                trend_adjustment = 0.1  # Boost lambda to leverage regret learning
            elif recent_policy_trend < recent_regret_trend:
                trend_adjustment = -0.1  # Reduce lambda to focus on actor

        # Add iteration-based annealing for exploration
        annealing_factor = min(1.0, self.iteration_count / 100.0)
        exploration_noise = (
            (1.0 - annealing_factor) * 0.05 * np.sin(self.iteration_count * 0.1)
        )

        final_lambda = np.clip(
            base_lambda + trend_adjustment + exploration_noise, 0.0, 1.0
        )

        return final_lambda

    def update_loss_averages(
        self, current_regret_loss: float, current_policy_loss: float, beta: float = 0.7
    ):
        """
        Update exponential moving averages for losses with more responsive tracking.

        Args:
            current_regret_loss: Current regret loss
            current_policy_loss: Current policy loss
            beta: EMA decay factor (reduced for more responsiveness)
        """
        # Update history for trend analysis
        self.regret_loss_history.append(current_regret_loss)
        self.policy_loss_history.append(current_policy_loss)

        # Keep only recent history
        max_history = 20
        if len(self.regret_loss_history) > max_history:
            self.regret_loss_history = self.regret_loss_history[-max_history:]
            self.policy_loss_history = self.policy_loss_history[-max_history:]

        # More responsive EMA with lower beta
        self.avg_regret_loss = (
            beta * self.avg_regret_loss + (1 - beta) * current_regret_loss
        )
        self.avg_policy_loss = (
            beta * self.avg_policy_loss + (1 - beta) * current_policy_loss
        )

        self.iteration_count += 1

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
        Compute mixed policy: π_mixed = (1-λ)π_actor + λπ_regret with enhanced adaptive mixing.

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
                # Track lambda history for analysis
                self.lambda_history.append(mixture_weight)
                if len(self.lambda_history) > 100:
                    self.lambda_history = self.lambda_history[-100:]
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

        # Enhanced mixing with adaptive weighting based on policy confidence
        if self.lambda_mode == "adaptive" and len(self.lambda_history) >= 5:
            # Compute policy diversity to guide mixing
            policy_diversity = torch.sum(
                torch.abs(actor_normalized - regret_policy), dim=-1, keepdim=True
            )

            # When policies are very different, be more decisive with lambda
            diversity_factor = torch.tanh(policy_diversity * 2.0)

            # Adjust mixture weight based on diversity
            adjusted_mixture_weight = (
                mixture_weight + 0.1 * diversity_factor.mean().item()
            )
            adjusted_mixture_weight = np.clip(adjusted_mixture_weight, 0.0, 1.0)
        else:
            adjusted_mixture_weight = mixture_weight

        # Mix policies with enhanced weighting
        mixed_policy = (
            1 - adjusted_mixture_weight
        ) * actor_normalized + adjusted_mixture_weight * regret_policy

        # Final normalization with entropy regularization for exploration
        mixed_sum = mixed_policy.sum(dim=-1, keepdim=True)
        mixed_policy = torch.where(
            mixed_sum > 1e-8, mixed_policy / (mixed_sum + 1e-8), uniform_policy
        )

        # Add small entropy bonus for better exploration in early training
        if self.iteration_count < 50:
            entropy = -torch.sum(
                mixed_policy * torch.log(mixed_policy + 1e-8), dim=-1, keepdim=True
            )
            entropy_bonus = 0.01 * entropy / self.num_actions
            mixed_policy = mixed_policy + entropy_bonus * uniform_policy
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

    def compute_scheduler_lambda(
        self,
        state_encoding: torch.Tensor,
        actor_logits: torch.Tensor,
        regret_logits: torch.Tensor,
        iteration: int,
        additional_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute lambda values using scheduler network.

        Args:
            state_encoding: Base state encoding
            actor_logits: Actor policy logits
            regret_logits: Regret head logits
            iteration: Current training iteration
            additional_features: Optional additional features

        Returns:
            Tuple of (lambda_values, scheduler_output_dict)
        """
        if self.scheduler is None:
            # Fallback to legacy adaptive lambda
            if self.lambda_mode == "adaptive":
                lambda_val = self.compute_lambda_t(
                    self.avg_regret_loss, self.avg_policy_loss, self.lambda_alpha
                )
                scheduler_out = {
                    "mode": "continuous",
                    "lambda": torch.full(
                        (state_encoding.size(0), 1),
                        lambda_val,
                        device=state_encoding.device,
                    ),
                }
                return (
                    scheduler_out["lambda"].squeeze(-1),
                    scheduler_out,
                )
            else:
                scheduler_out = {
                    "mode": "continuous",
                    "lambda": torch.full(
                        (state_encoding.size(0), 1),
                        self.mixture_weight,
                        device=state_encoding.device,
                    ),
                }
                return (
                    scheduler_out["lambda"].squeeze(-1),
                    scheduler_out,
                )

        # Compute scheduler input
        scheduler_input = compute_scheduler_input(
            state_encoding, actor_logits, regret_logits, iteration, additional_features
        )

        # Get scheduler output in standardized format
        scheduler_out = self.scheduler(scheduler_input)

        # Extract lambda values
        if scheduler_out["mode"] == "continuous":
            lambda_vals = scheduler_out["lambda"].squeeze(-1)
        else:
            # For discrete mode, convert to lambda values
            from algs.scheduler.policy_mixer import discrete_logits_to_lambda

            lambda_vals, _ = discrete_logits_to_lambda(
                scheduler_out["logits"],
                self.scheduler.k_bins,
                hard=not scheduler_out["logits"].requires_grad,
                tau=self.scheduler.temperature,
            )
            lambda_vals = lambda_vals.squeeze(-1)

        return lambda_vals, scheduler_out

    def mixed_policy_with_scheduler(
        self,
        state_encoding: torch.Tensor,
        actor_logits: torch.Tensor,
        regret_logits: torch.Tensor,
        legal_actions_masks: torch.Tensor,
        iteration: int,
        additional_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute mixed policy using scheduler for per-instance lambda.

        Args:
            state_encoding: Base state encoding
            actor_logits: Actor policy logits
            regret_logits: Regret head logits
            legal_actions_masks: Legal actions masks
            iteration: Current training iteration
            additional_features: Optional additional features

        Returns:
            Tuple of (mixed_policy, metadata_dict)
        """
        # Get lambda values from scheduler
        lambda_vals, scheduler_out = self.compute_scheduler_lambda(
            state_encoding, actor_logits, regret_logits, iteration, additional_features
        )

        # Mix policies using policy mixer
        if self.policy_mixer is not None:
            mixed_policy = self.policy_mixer.mix(
                actor_logits, regret_logits, scheduler_out
            )
            mixing_stats = self.policy_mixer.compute_mixing_stats(
                actor_logits, regret_logits, scheduler_out
            )
        else:
            # Fallback to legacy mixing
            mixed_policy = self.mixed_policy_update(
                F.softmax(actor_logits, dim=-1),
                self.regret_matching_policy(
                    self.compute_advantages(
                        state_encoding, None, actor_logits, legal_actions_masks
                    ),
                    legal_actions_masks,
                ),
                legal_actions_masks,
                lambda_vals.mean().item() if lambda_vals.dim() > 0 else lambda_vals,
            )
            mixing_stats = {}

        # Store scheduler data for meta-regret updates
        self.scheduler_inputs.append(
            scheduler_out.get("scheduler_inputs", torch.zeros_like(state_encoding))
            .detach()
            .cpu()
        )

        # Store discrete choices for meta-regret
        if scheduler_out["mode"] == "discrete":
            self.scheduler_choices.append(
                scheduler_out.get("lambda_idx", torch.zeros(len(lambda_vals)))
                .detach()
                .cpu()
            )
        else:
            self.scheduler_choices.append(lambda_vals.detach().cpu())

        metadata = {
            "lambda_values": lambda_vals,
            "scheduler_output": scheduler_out,
            "mixing_stats": mixing_stats,
        }

        return mixed_policy, metadata

    def update_meta_regret(
        self,
        state_encodings: torch.Tensor,
        scheduler_choices: torch.Tensor,
        utilities: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Update meta-regret manager with scheduler outcomes.

        Args:
            state_encodings: State encodings for which choices were made
            scheduler_choices: Scheduler choices (lambda values or discrete indices)
            utilities: Observed utilities for these choices

        Returns:
            Dictionary with update statistics
        """
        if self.meta_regret is None:
            return {}

        update_stats = []
        for i in range(state_encodings.size(0)):
            state_key = self.meta_regret.state_key_func(state_encodings[i])

            # Convert continuous lambda to discrete choice if needed
            if self.policy_mixer and self.policy_mixer.discrete:
                # Already discrete
                k_choice = scheduler_choices[i].item()
            else:
                # Discretize continuous lambda
                if self.policy_mixer and self.policy_mixer.lambda_bins is not None:
                    # Find nearest bin
                    lambda_val = scheduler_choices[i].item()
                    bins = self.policy_mixer.lambda_bins
                    k_choice = torch.argmin(torch.abs(bins - lambda_val)).item()
                else:
                    # Default to 5 bins
                    lambda_val = scheduler_choices[i].item()
                    k_choice = min(int(lambda_val * 5), 4)

            utility = utilities[i].item()
            stats = self.meta_regret.record(state_key, k_choice, utility)
            update_stats.append(stats)

        self.scheduler_utilities.extend(utilities.tolist())

        return {
            "num_updates": len(update_stats),
            "avg_regret_increment": np.mean(
                [s["regret_increment"] for s in update_stats]
            ),
            "avg_utility": utilities.mean().item(),
            "global_stats": self.meta_regret.get_global_stats(),
        }

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        stats = {
            "iteration": self.iteration_count,
            "lambda_mode": self.lambda_mode,
            "avg_regret_loss": self.avg_regret_loss,
            "avg_policy_loss": self.avg_policy_loss,
        }

        if self.scheduler is not None:
            stats["scheduler_type"] = (
                "discrete" if self.scheduler.discrete else "continuous"
            )
            stats["num_bins"] = self.scheduler.get_num_bins()
            stats["temperature"] = self.scheduler.temperature

        if self.policy_mixer is not None:
            stats["mixer_discrete"] = self.policy_mixer.discrete
            stats["mixer_temperature"] = self.policy_mixer.current_temperature

        if self.meta_regret is not None:
            stats.update(self.meta_regret.get_global_stats())

        # Lambda statistics
        if self.scheduler_choices:
            all_lambdas = torch.cat(self.scheduler_choices)
            stats["lambda_mean"] = all_lambdas.mean().item()
            stats["lambda_std"] = all_lambdas.std().item()
            stats["lambda_min"] = all_lambdas.min().item()
            stats["lambda_max"] = all_lambdas.max().item()

        return stats


def create_armac_dual_rl(
    num_actions: int,
    config: Dict[str, Any],
    scheduler_components: Optional[Dict] = None,
) -> ARMACDualRL:
    """
    Factory function to create ARMAC dual RL instance with scheduler components.

    Args:
        num_actions: Number of actions in the game
        config: Configuration dictionary
        scheduler_components: Optional dict with scheduler, policy_mixer, meta_regret

    Returns:
        ARMACDualRL instance
    """
    if scheduler_components is None:
        # Legacy mode - extract parameters from config
        if isinstance(config, dict):
            mixture_weight = config.get(
                "regret_weight", config.get("mixture_weight", 0.1)
            )
            lambda_mode = config.get("lambda_mode", "fixed")
            lambda_alpha = config.get("lambda_alpha", 0.5)
        else:
            # Assume config is already a properly formatted dict
            mixture_weight = config.get("mixture_weight", 0.1)
            lambda_mode = config.get("lambda_mode", "fixed")
            lambda_alpha = config.get("lambda_alpha", 0.5)

        # Create a minimal config dict for legacy mode
        legacy_config = {
            "mixture_weight": mixture_weight,
            "lambda_mode": lambda_mode,
            "lambda_alpha": lambda_alpha,
        }

        return ARMACDualRL(
            num_actions=num_actions,
            config=legacy_config,
            scheduler=None,
            policy_mixer=None,
            meta_regret=None,
        )
    else:
        # New scheduler mode
        return ARMACDualRL(
            num_actions=num_actions,
            config=config,
            scheduler=scheduler_components.get("scheduler"),
            policy_mixer=scheduler_components.get("policy_mixer"),
            meta_regret=scheduler_components.get("meta_regret"),
        )
