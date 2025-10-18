"""Regret-matching bandit target for the expert gate."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import torch


@dataclass
class BanditStats:
    cumulative_regret: torch.Tensor
    utilities: torch.Tensor
    count: int


class GateBandit:
    """Maintains per-cluster regret tables and produces soft targets."""

    def __init__(
        self,
        num_experts: int,
        *,
        clip: float = 5.0,
        device: torch.device | None = None,
        ema_decay: float = 0.99,
    ) -> None:
        self.num_experts = num_experts
        self.clip = clip
        self.device = device or torch.device("cpu")
        self.ema_decay = ema_decay
        self._regret: Dict[str, torch.Tensor] = {}
        self._avg_util: Dict[str, torch.Tensor] = {}
        self._history: Dict[str, Deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=64))

    def _get_regret(self, cluster: str) -> torch.Tensor:
        if cluster not in self._regret:
            self._regret[cluster] = torch.zeros(self.num_experts, device=self.device)
            self._avg_util[cluster] = torch.zeros(self.num_experts, device=self.device)
        return self._regret[cluster]

    def observe(self, cluster: str, utilities: torch.Tensor) -> None:
        if utilities.dim() != 1 or utilities.numel() != self.num_experts:
            raise ValueError("utilities must be 1D with length num_experts")
        utilities = utilities.to(self.device)
        self._history[cluster].append(utilities.detach())
        regrets = self._get_regret(cluster)
        avg_util = self._avg_util[cluster]
        avg_util.mul_(self.ema_decay).add_(utilities * (1 - self.ema_decay))
        regrets.add_(utilities - avg_util)
        if self.clip > 0:
            regrets.clamp_(min=-self.clip, max=self.clip)

    def target(self, cluster: str) -> torch.Tensor:
        regrets = self._get_regret(cluster)
        positive = torch.clamp(regrets, min=0.0)
        if positive.sum() <= 1e-8:
            dist = torch.full_like(positive, 1.0 / self.num_experts)
        else:
            dist = positive / positive.sum()
        return dist

    def diagnostics(self, cluster: str) -> Dict[str, float]:
        regrets = self._get_regret(cluster)
        history = list(self._history[cluster])
        if history:
            stacked = torch.stack(history)
            util_mean = float(stacked.mean().item())
        else:
            util_mean = 0.0
        return {
            "regret_norm": float(regrets.norm().item()),
            "avg_utility": util_mean,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "regret": {k: v.clone() for k, v in self._regret.items()},
            "avg": {k: v.clone() for k, v in self._avg_util.items()},
        }

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        regret = state.get("regret", {})
        avg = state.get("avg", {})
        self._regret = {k: v.to(self.device) for k, v in regret.items()}
        self._avg_util = {k: v.to(self.device) for k, v in avg.items()}
