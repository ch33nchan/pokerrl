"""Policy mixer module for ARMAC dual-learning framework.

This module implements the mixing of actor and regret policies using
per-instance lambda values from the scheduler. Supports both continuous
and discrete lambda modes.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np


def regret_policy_from_regret_logits(regret_logits: torch.Tensor) -> torch.Tensor:
    """Convert regret logits to regret-matching policy.

    Args:
        regret_logits: Regret head logits of shape (batch_size, num_actions)

    Returns:
        Regret-matching policy probabilities
    """
    # Apply positive part and normalize
    regrets = F.relu(regret_logits)
    regrets_sum = regrets.sum(dim=-1, keepdim=True)

    # Handle zero regrets case
    policy = torch.where(
        regrets_sum > 0,
        regrets / (regrets_sum + 1e-8),
        torch.ones_like(regrets) / regrets.size(-1),
    )

    return policy


def renormalize(policy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Renormalize policy to ensure valid probability distribution.

    Args:
        policy: Policy tensor
        eps: Small epsilon for numerical stability

    Returns:
        Renormalized policy
    """
    policy_sum = policy.sum(dim=-1, keepdim=True)
    return policy / (policy_sum + eps)


def discrete_logits_to_lambda(
    logits: torch.Tensor,
    lambda_bins: torch.Tensor,
    hard: bool = False,
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert discrete scheduler logits to lambda values and indices.

    Args:
        logits: Tensor[B, K] - scheduler logits
        lambda_bins: Tensor[K] - lambda bin values
        hard: bool, if True use argmax to return indices (non-diff)
        tau: Temperature for Gumbel-softmax

    Returns:
        tuple: (lambda_vals Tensor[B,1], idx LongTensor[B])
    """
    # Ensure same device
    lambda_bins = lambda_bins.to(logits.device)
    assert lambda_bins.device == logits.device, (
        "Device mismatch between lambda_bins and logits"
    )
    assert logits.dim() == 2, f"Expected 2D logits, got {logits.dim()}D"
    assert lambda_bins.dim() == 1, f"Expected 1D lambda_bins, got {lambda_bins.dim()}D"

    # Soft selection during training: Gumbel-softmax
    if not hard:
        probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # [B,K]
        # Expected lambda value (float per batch)
        lambda_vals = probs @ lambda_bins.float()  # [B]
        idx = probs.argmax(dim=-1)  # For logging/bookkeeping
    else:
        # Inference / hard selection
        idx = torch.argmax(logits, dim=-1)  # LongTensor[B]
        lambda_vals = lambda_bins.float()[idx]  # [B]

    return lambda_vals.view(-1, 1), idx.long()


def mix_policies(
    actor_logits: torch.Tensor,
    regret_logits: torch.Tensor,
    scheduler_out: dict,
    lambda_bins: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mix actor and regret policies using scheduler output.

    Args:
        actor_logits: Actor policy logits of shape (batch_size, num_actions)
        regret_logits: Regret head logits of shape (batch_size, num_actions)
        scheduler_out: Dict from Scheduler.forward with standardized format
        lambda_bins: Lambda bin values for discrete mode

    Returns:
        Mixed policy probabilities
    """
    # Convert logits to probabilities
    pi_actor = F.softmax(actor_logits, dim=-1)
    pi_regret = regret_policy_from_regret_logits(regret_logits)

    if scheduler_out["mode"] == "continuous":
        lam = scheduler_out["lambda"].to(pi_actor.device)  # [B,1]
        # Ensure broadcast shape
        lam = lam.unsqueeze(-1) if lam.dim() == 1 else lam  # make [B,1]
    else:
        # Discrete mode
        logits = scheduler_out["logits"]
        assert lambda_bins is not None, "lambda_bins required for discrete mode"
        lam_vals, idxs = discrete_logits_to_lambda(
            logits, lambda_bins, hard=not logits.requires_grad, tau=1.0
        )
        lam = lam_vals.to(pi_actor.device)  # [B,1]

    # Mix policies
    pi_mix = lam * pi_actor + (1.0 - lam) * pi_regret

    # Renormalize to ensure valid probability distribution
    return renormalize(pi_mix)


class PolicyMixer:
    """Policy mixer class for ARMAC framework.

    This class handles the mixing of actor and regret policies using
    scheduler outputs, with support for both continuous and discrete modes.
    """

    def __init__(
        self,
        discrete: bool = False,
        lambda_bins: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        temperature_decay: float = 0.99,
        min_temperature: float = 0.1,
    ):
        """Initialize policy mixer.

        Args:
            discrete: Whether to use discrete lambda mode
            lambda_bins: Lambda bin values for discrete mode
            temperature_decay: Temperature decay rate for discrete mode
            min_temperature: Minimum temperature value
        """
        self.discrete = discrete
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.current_temperature = 1.0

        if discrete and lambda_bins is not None:
            if isinstance(lambda_bins, (list, np.ndarray)):
                self.lambda_bins = torch.tensor(lambda_bins, dtype=torch.float32)
            else:
                self.lambda_bins = lambda_bins
        else:
            self.lambda_bins = None

    def mix(
        self,
        actor_logits: torch.Tensor,
        regret_logits: torch.Tensor,
        scheduler_out: dict,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Mix policies using scheduler output.

        Args:
            actor_logits: Actor policy logits
            regret_logits: Regret head logits
            scheduler_out: Dict from Scheduler.forward with standardized format
            temperature: Optional temperature override

        Returns:
            Mixed policy probabilities
        """
        if temperature is not None:
            current_temp = temperature
        else:
            current_temp = self.current_temperature

        return mix_policies(
            actor_logits=actor_logits,
            regret_logits=regret_logits,
            scheduler_out=scheduler_out,
            lambda_bins=self.lambda_bins,
        )

    def update_temperature(self):
        """Update temperature for discrete mode annealing."""
        if self.discrete:
            self.current_temperature = max(
                self.min_temperature, self.current_temperature * self.temperature_decay
            )

    def set_temperature(self, temperature: float):
        """Set temperature manually.

        Args:
            temperature: Temperature value
        """
        self.current_temperature = temperature

    def get_effective_lambda(
        self,
        scheduler_out: dict,
    ) -> torch.Tensor:
        """Get effective lambda values for analysis.

        Args:
            scheduler_out: Dict from Scheduler.forward with standardized format

        Returns:
            Effective lambda values in [0, 1]
        """
        if scheduler_out["mode"] == "continuous":
            # Continuous mode
            lam = scheduler_out["lambda"]
            if lam.dim() > 1:
                return lam.squeeze(-1)
            return lam
        else:
            # Discrete mode
            if self.lambda_bins is None:
                raise ValueError("lambda_bins required for discrete mode")

            logits = scheduler_out["logits"]
            lambda_vals, _ = discrete_logits_to_lambda(
                logits, self.lambda_bins, hard=True, tau=1.0
            )
            return lambda_vals.squeeze(-1)

    def compute_mixing_stats(
        self,
        actor_logits: torch.Tensor,
        regret_logits: torch.Tensor,
        scheduler_out: dict,
    ) -> dict:
        """Compute mixing statistics for analysis.

        Args:
            actor_logits: Actor policy logits
            regret_logits: Regret head logits
            scheduler_out: Dict from Scheduler.forward with standardized format

        Returns:
            Dictionary of mixing statistics
        """
        with torch.no_grad():
            pi_actor = F.softmax(actor_logits, dim=-1)
            pi_regret = regret_policy_from_regret_logits(regret_logits)
            pi_mix = self.mix(actor_logits, regret_logits, scheduler_out)

            # Effective lambda values
            lambda_eff = self.get_effective_lambda(scheduler_out)

            # Policy divergences
            kl_actor_regret = F.kl_div(
                F.log_softmax(actor_logits, dim=-1), pi_regret, reduction="batchmean"
            )
            kl_mix_actor = F.kl_div(
                torch.log(pi_mix + 1e-8), pi_actor, reduction="batchmean"
            )
            kl_mix_regret = F.kl_div(
                torch.log(pi_mix + 1e-8), pi_regret, reduction="batchmean"
            )

            # Entropy values
            entropy_actor = (
                -(pi_actor * F.log_softmax(actor_logits, dim=-1)).sum(dim=-1).mean()
            )
            entropy_regret = (
                -(pi_regret * torch.log(pi_regret + 1e-8)).sum(dim=-1).mean()
            )
            entropy_mix = -(pi_mix * torch.log(pi_mix + 1e-8)).sum(dim=-1).mean()

            return {
                "lambda_mean": lambda_eff.mean().item(),
                "lambda_std": lambda_eff.std().item(),
                "lambda_min": lambda_eff.min().item(),
                "lambda_max": lambda_eff.max().item(),
                "kl_actor_regret": kl_actor_regret.item(),
                "kl_mix_actor": kl_mix_actor.item(),
                "kl_mix_regret": kl_mix_regret.item(),
                "entropy_actor": entropy_actor.item(),
                "entropy_regret": entropy_regret.item(),
                "entropy_mix": entropy_mix.item(),
                "temperature": self.current_temperature,
            }


def create_policy_mixer(config: dict) -> PolicyMixer:
    """Factory function to create policy mixer from config.

    Args:
        config: Configuration dictionary

    Returns:
        PolicyMixer instance
    """
    mixer_config = config.get("policy_mixer", {})

    discrete = mixer_config.get("discrete", False)
    lambda_bins = mixer_config.get("lambda_bins", None)
    temperature_decay = mixer_config.get("temperature_decay", 0.99)
    min_temperature = mixer_config.get("min_temperature", 0.1)

    return PolicyMixer(
        discrete=discrete,
        lambda_bins=lambda_bins,
        temperature_decay=temperature_decay,
        min_temperature=min_temperature,
    )
