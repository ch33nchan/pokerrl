"""Meta-objective utilities for training the expert gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch

from dual_rl_poker.tools.approx_br import ApproxBestResponse


@dataclass
class MetaBatchItem:
    features: torch.Tensor
    gate_probs: torch.Tensor
    cluster: str
    expert_policies: Sequence[Dict[int, float]]


class MetaObjective:
    """Combines rollout-based utilities with differentiable gate loss."""

    def __init__(
        self,
        approx_br: ApproxBestResponse,
        *,
        kl_weight: float = 0.1,
    ) -> None:
        self.approx_br = approx_br
        self.kl_weight = kl_weight

    def evaluate(
        self,
        batch: Sequence[MetaBatchItem],
        gate: Callable[[torch.Tensor], torch.Tensor],
        bandit_targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not batch:
            return torch.tensor(0.0, dtype=torch.float32)
        losses = []
        for item in batch:
            with torch.no_grad():
                base_policy = self._mix_policy(item.gate_probs, item.expert_policies)
                expert_wrappers = [self._wrap_policy(policy) for policy in item.expert_policies]
                utilities = self.approx_br.evaluate_experts(base_policy, expert_wrappers)
            logits = gate(item.features.unsqueeze(0))
            log_probs = torch.log_softmax(logits, dim=-1)
            utilities_tensor = torch.tensor(utilities, dtype=torch.float32, device=log_probs.device)
            policy_loss = -(utilities_tensor.detach() * log_probs.squeeze(0)).sum()
            target = bandit_targets[item.cluster].to(log_probs.device)
            kl = torch.nn.functional.kl_div(log_probs, target.unsqueeze(0), reduction="batchmean")
            losses.append(policy_loss + self.kl_weight * kl)
        return torch.stack(losses).mean()

    def _wrap_policy(self, policy: Dict[int, float]) -> Callable[[object, int], Dict[int, float]]:
        def fn(_: object, __: int) -> Dict[int, float]:
            return policy

        return fn

    def _mix_policy(
        self,
        gate_probs: torch.Tensor,
        expert_policies: Sequence[Dict[int, float]],
    ) -> Callable[[object, int], Dict[int, float]]:
        mix: Dict[int, float] = {}
        for prob, expert in zip(gate_probs.tolist(), expert_policies):
            for action, p in expert.items():
                mix[action] = mix.get(action, 0.0) + prob * p
        norm = sum(mix.values()) or 1.0
        mix = {a: p / norm for a, p in mix.items()}
        return self._wrap_policy(mix)
