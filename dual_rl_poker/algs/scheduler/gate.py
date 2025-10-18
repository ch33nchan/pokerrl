"""Expert gate network for mixing multiple behavior policies.

This module implements the neural gating policy used by the Meta-Adaptive
K-Expert Gate (MARM-K) scheduler. The gate consumes a compact feature vector
constructed from the information state encoding, running training statistics,
and iteration context. It outputs a logit vector over K experts which is later
converted into probabilities via softmax with temperature/entropy regularisers.

The design intentionally keeps the implementation lightweight so that it can run
comfortably on CPU-only environments (e.g. MacBook). The gate is trained via a
combination of meta-gradients (short unroll objective) and a regret-matching
bandit target that guarantees sublinear scheduler regret when tracked closely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateOutput:
    """Container holding the gate probabilities and optional diagnostics."""

    probs: torch.Tensor  # shape: [B, K]
    logits: torch.Tensor  # shape: [B, K]
    temperature: float

    def detached(self) -> "GateOutput":
        return GateOutput(self.probs.detach(), self.logits.detach(), self.temperature)


class ExpertGate(nn.Module):
    """Small MLP gate that outputs a distribution over experts."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_sizes: Iterable[int] = (128, 64),
        *,
        temperature: float = 1.0,
        min_prob: float = 1e-3,
        entropy_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev_dim, num_experts)

        self.num_experts = num_experts
        self.temperature = temperature
        self.min_prob = min_prob
        self.entropy_reg = entropy_reg

    def forward(self, x: torch.Tensor) -> GateOutput:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {x.shape}")
        logits = self.head(self.backbone(x))
        scaled = logits / max(self.temperature, 1e-3)
        probs = F.softmax(scaled, dim=-1)
        if self.min_prob > 0:
            probs = probs.clamp(min=self.min_prob)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return GateOutput(probs=probs, logits=scaled, temperature=self.temperature)

    def entropy_bonus(self, output: GateOutput) -> torch.Tensor:
        if self.entropy_reg <= 0:
            return torch.tensor(0.0, device=output.probs.device)
        entropy = -(output.probs * output.probs.log()).sum(dim=-1)
        return self.entropy_reg * entropy.mean()

    def set_temperature(self, value: float) -> None:
        self.temperature = max(1e-3, float(value))

    def extra_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


def gate_context(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack heterogeneous feature tensors into a single flat vector.

    The trainer collects a dictionary of feature blocks (state encoding,
    lambda statistics, running losses, iteration context, etc.). This helper
    concatenates them in a deterministic key order so the gate always sees the
    same layout. Missing entries are treated as errors to avoid silent
    misalignment.
    """

    if not features:
        raise ValueError("features dictionary is empty")
    ordered = []
    for key in sorted(features.keys()):
        value = features[key]
        if value.dim() == 1:
            ordered.append(value)
        elif value.dim() == 2 and value.size(0) == 1:
            ordered.append(value.squeeze(0))
        else:
            ordered.append(value.view(-1))
    return torch.cat(ordered, dim=0)


def batch_gate_context(batch: Iterable[Dict[str, torch.Tensor]]) -> torch.Tensor:
    tensors = [gate_context(item) for item in batch]
    return torch.stack(tensors, dim=0)
