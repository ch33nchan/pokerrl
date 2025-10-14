"""Scheduler module for ARMAC dual-learning framework.

This module implements the scheduler network that outputs per-instance mixing
coefficients (lambda) for combining actor and regret policies. Supports both
continuous and discrete modes as specified in the plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np


class Scheduler(nn.Module):
    """Scheduler network that outputs mixing coefficients.

    The scheduler takes state embeddings z(s) and outputs either:
    - Continuous lambda values in [0, 1]
    - Discrete choice among K bins for regret matching
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Tuple[int, int] = (64, 32),
        k_bins: Optional[Union[list, np.ndarray]] = None,
        temperature: float = 1.0,
        use_gumbel: bool = True,
    ):
        """Initialize scheduler network.

        Args:
            input_dim: Dimension of scheduler input z(s)
            hidden: Hidden layer sizes
            k_bins: If provided, discretize lambda into these bins
            temperature: Temperature for Gumbel-softmax
            use_gumbel: Whether to use Gumbel-softmax for discrete mode
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Robustness measures
        self.lam_clamp_eps = 1e-3
        self.warmup_iters = 0
        self.init_lambda = 0.5
        self.regularization_config = {}

        # Build MLP
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1) if i < len(hidden) - 1 else nn.Identity(),
                ]
            )
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        if k_bins is None:
            # Continuous mode
            self.out = nn.Linear(prev_dim, 1)
            self.discrete = False
            self.k_bins = None
        else:
            # Discrete mode
            self.out = nn.Linear(prev_dim, len(k_bins))
            self.k_bins = torch.tensor(k_bins, dtype=torch.float32)
            self.discrete = True

    def forward(self, z: torch.Tensor, hard: bool = False) -> dict:
        """Forward pass with standardized output format.

        Args:
            z: State embedding tensor of shape (batch_size, input_dim)
            hard: Whether to use hard discretization (only for discrete mode)

        Returns:
            Dict with standardized format:
            - continuous: {"mode": "continuous", "lambda": Tensor(B,1)}
            - discrete: {"mode": "discrete", "logits": Tensor(B,K), "lambda_idx": LongTensor(B)}
        """
        # Input validation
        assert z.dim() == 2, f"Expected 2D input, got {z.dim()}D"
        assert z.size(-1) == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {z.size(-1)}"
        )

        h = self.mlp(z)

        if self.discrete:
            logits = self.out(h)  # shape: [B, K]

            # Ensure logits are on the same device as input
            logits = logits.to(z.device)

            if self.training and self.use_gumbel and not hard:
                # Use Gumbel-softmax during training - return expected lambda
                return {"mode": "discrete", "logits": logits}
            else:
                # Use regular softmax during inference or hard selection
                probs = F.softmax(logits / self.temperature, dim=-1)
                lambda_idx = torch.argmax(probs, dim=-1)  # LongTensor[B]
                return {"mode": "discrete", "logits": logits, "lambda_idx": lambda_idx}
        else:
            # Continuous mode - output sigmoid with clamping
            lam = torch.sigmoid(self.out(h))  # shape: [B, 1]
            lam = lam.view(-1, 1)  # ensure [B,1] shape

            # Apply clamping for numerical stability
            lam = lam.clamp(min=self.lam_clamp_eps, max=1.0 - self.lam_clamp_eps)

            return {"mode": "continuous", "lambda": lam}

    def get_lambda_values(self, z: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Get actual lambda values using standardized forward output.

        For discrete mode, this converts the discrete choice to the corresponding
        lambda bin value. For continuous mode, this returns the raw output.

        Args:
            z: State embedding tensor
            hard: Whether to use hard discretization

        Returns:
            Lambda values tensor of shape [B]
        """
        scheduler_out = self.forward(z, hard=hard)

        if scheduler_out["mode"] == "discrete":
            # Use discrete_logits_to_lambda helper
            from algs.scheduler.policy_mixer import discrete_logits_to_lambda

            lambda_vals, lambda_idx = discrete_logits_to_lambda(
                scheduler_out["logits"],
                self.k_bins,
                hard=hard or not self.training,
                tau=self.temperature,
            )
            return lambda_vals.squeeze(-1)  # Return [B]
        else:
            # Continuous mode
            return scheduler_out["lambda"].squeeze(-1)  # Return [B]

    def get_discrete_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Get discrete probabilities (only for discrete mode).

        Args:
            z: State embedding tensor

        Returns:
            Probability distribution over bins
        """
        if not self.discrete:
            raise ValueError("get_discrete_probs only valid for discrete scheduler")

        h = self.mlp(z)
        logits = self.out(h)
        return F.softmax(logits, dim=-1)

    def set_temperature(self, temperature: float):
        """Set temperature for discrete mode.

        Args:
            temperature: Temperature value
        """
        self.temperature = temperature

    def get_num_bins(self) -> Optional[int]:
        """Get number of bins for discrete mode.

        Returns:
            Number of bins or None if continuous
        """
        return len(self.k_bins) if self.discrete else None

    def set_warmup(self, warmup_iters: int, init_lambda: float = 0.5):
        """Set warmup parameters for scheduler.

        Args:
            warmup_iters: Number of iterations to freeze scheduler
            init_lambda: Initial lambda value during warmup
        """
        self.warmup_iters = warmup_iters
        self.init_lambda = init_lambda

    def set_clamping(self, eps: float = 1e-3):
        """Set clamping epsilon for lambda values.

        Args:
            eps: Small epsilon to avoid extreme values
        """
        self.lam_clamp_eps = eps

    def set_regularization(self, config: dict):
        """Set regularization parameters.

        Args:
            config: Dictionary with regularization parameters
        """
        self.regularization_config = config

    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for scheduler parameters.

        Returns:
            Regularization loss tensor
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if not self.regularization_config:
            return loss

        # L2 regularization
        beta_l2 = self.regularization_config.get("beta_l2", 0.0)
        if beta_l2 > 0:
            l2_loss = sum(torch.sum(p**2) for p in self.parameters())
            loss += beta_l2 * l2_loss

        # Entropy regularization (only for discrete mode)
        if self.discrete:
            beta_ent = self.regularization_config.get("beta_ent", 0.0)
            if beta_ent > 0:
                # This would need logits from the forward pass
                # For now, return zero - this should be computed during training
                pass

        return loss

    def forward_with_warmup(
        self, z: torch.Tensor, iteration: int, hard: bool = False
    ) -> dict:
        """Forward pass with warmup support.

        Args:
            z: State embedding tensor
            iteration: Current training iteration
            hard: Whether to use hard discretization

        Returns:
            Standardized scheduler output
        """
        # Check warmup condition
        if iteration < self.warmup_iters:
            batch_size = z.size(0)
            if self.discrete:
                # During warmup, return uniform distribution
                logits = torch.zeros(batch_size, len(self.k_bins), device=z.device)
                return {"mode": "discrete", "logits": logits}
            else:
                # Return initial lambda
                lam = torch.full(
                    (batch_size, 1), self.init_lambda, device=z.device, dtype=z.dtype
                )
                lam = lam.clamp(min=self.lam_clamp_eps, max=1.0 - self.lam_clamp_eps)
                return {"mode": "continuous", "lambda": lam}

        # Normal forward pass
        return self.forward(z, hard=hard)


def create_scheduler(config: dict, input_dim: int) -> Scheduler:
    """Factory function to create scheduler from config.

    Args:
        config: Configuration dictionary
        input_dim: Input dimension

    Returns:
        Scheduler instance
    """
    scheduler_config = config.get("scheduler", {})

    # Extract parameters
    hidden = tuple(scheduler_config.get("hidden", [64, 32]))
    k_bins = scheduler_config.get("k_bins", None)
    temperature = scheduler_config.get("temperature", 1.0)
    use_gumbel = scheduler_config.get("use_gumbel", True)

    return Scheduler(
        input_dim=input_dim,
        hidden=hidden,
        k_bins=k_bins,
        temperature=temperature,
        use_gumbel=use_gumbel,
    )


# Utility functions for scheduler input computation
def compute_scheduler_input(
    state_encoding: torch.Tensor,
    actor_logits: torch.Tensor,
    regret_logits: torch.Tensor,
    iteration: int,
    additional_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scheduler input z(s) from various features.

    Args:
        state_encoding: Base state encoding
        actor_logits: Actor policy logits
        regret_logits: Regret head logits
        iteration: Current training iteration
        additional_features: Optional additional features

    Returns:
        Scheduler input tensor
    """
    features = []

    # Base state encoding
    features.append(state_encoding)

    # Policy disagreement metrics
    actor_probs = F.softmax(actor_logits, dim=-1)
    regret_probs = F.softmax(regret_logits, dim=-1)

    # KL divergence between policies
    kl_div = F.kl_div(
        F.log_softmax(actor_logits, dim=-1), regret_probs, reduction="none"
    ).sum(dim=-1, keepdim=True)
    features.append(kl_div)

    # Policy entropy
    actor_entropy = -(actor_probs * F.log_softmax(actor_logits, dim=-1)).sum(
        dim=-1, keepdim=True
    )
    regret_entropy = -(regret_probs * F.log_softmax(regret_logits, dim=-1)).sum(
        dim=-1, keepdim=True
    )

    features.extend([actor_entropy, regret_entropy])

    # Iteration feature (normalized)
    iter_norm = torch.tensor(
        [iteration / 1000.0], device=state_encoding.device, dtype=state_encoding.dtype
    )
    iter_norm = iter_norm.expand(state_encoding.size(0), -1)
    features.append(iter_norm)

    # Additional features if provided
    if additional_features is not None:
        features.append(additional_features)

    return torch.cat(features, dim=-1)
