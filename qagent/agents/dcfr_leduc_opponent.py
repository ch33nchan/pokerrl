"""Opponent-model variant of the Leduc DCFR trainer."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qagent.agents.dcfr_leduc import DCFRTrainer


class OpponentModelNet(nn.Module):
    """Predicts opponent mixed strategies from info set encodings."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(x)
        return self.softmax(logits)


class DCFROpponentModelTrainer(DCFRTrainer):
    """Extends `DCFRTrainer` with an opponent modelling auxiliary network."""

    def __init__(
        self,
        *args,
        opponent_model_weight: float = 0.5,
        opponent_model_lr: float = 5e-4,
        opponent_model_hidden_dim: int = 128,
        opponent_memory_size: int = 200_000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.opponent_model = OpponentModelNet(self.info_set_dim, self.num_actions, opponent_model_hidden_dim).to(
            self.device
        )
        self.opponent_optimizer = optim.Adam(self.opponent_model.parameters(), lr=opponent_model_lr)
        self.opponent_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.opponent_memory_size = opponent_memory_size
        self.opponent_model_weight = opponent_model_weight

        self.opponent_parameter_count = sum(p.numel() for p in self.opponent_model.parameters())
        self.opponent_losses: List[float] = []
        self.opponent_grad_norms: List[float] = []

    # ------------------------------------------------------------------
    # CFR traversal override with opponent-modelling adjustments
    # ------------------------------------------------------------------
    def _cfr_traverse(self, state: Dict[str, object], t: int, pi_p: float, pi_o: float) -> float:
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, 0)

        player = self.game.get_current_player(state)
        infoset = self._build_infoset(state)
        info_tensor = self.encoder.encode_infoset(self.game, infoset).unsqueeze(0).to(self.device)
        info_vector = info_tensor.squeeze(0).detach().cpu().numpy()

        legal_mask = self._legal_action_mask(state)
        if legal_mask.sum() == 0:
            return self.game.get_payoff(state, 0)

        strategy = self._strategy_from_tensor(info_tensor, legal_mask)

        if player != 0:
            # store the baseline CFR-derived strategy for training the opponent model
            self._add_to_opponent_memory(info_vector.copy(), strategy.copy())
            with torch.no_grad():
                opponent_prediction = self.opponent_model(info_tensor).cpu().numpy().flatten()
            opponent_prediction *= legal_mask
            if opponent_prediction.sum() > 0:
                opponent_prediction /= opponent_prediction.sum()
            else:
                opponent_prediction = legal_mask / legal_mask.sum()
            strategy = (1.0 - self.opponent_model_weight) * strategy + self.opponent_model_weight * opponent_prediction

        util = np.zeros(self.num_actions, dtype=np.float32)
        node_util = 0.0

        for action in range(self.num_actions):
            if legal_mask[action] == 0:
                continue

            next_state = self.game.get_next_state(dict(state), action)
            if player == 0:
                util[action] = self._cfr_traverse(next_state, t, pi_p * strategy[action], pi_o)
            else:
                util[action] = self._cfr_traverse(next_state, t, pi_p, pi_o * strategy[action])
            node_util += strategy[action] * util[action]

        if player == 0:
            regrets = (util - node_util) * legal_mask
            if self.regret_noise_std > 0:
                noise = np.random.normal(0.0, self.regret_noise_std, size=regrets.shape).astype(np.float32)
                regrets = regrets + noise
            with torch.no_grad():
                current_regrets = self.regret_net(info_tensor).cpu().numpy().flatten()
            new_regrets = current_regrets + pi_o * regrets
            self._add_to_memory(self.regret_memory, (info_vector.copy(), new_regrets))

            with torch.no_grad():
                current_strategy = self.strategy_net(info_tensor).cpu().numpy().flatten()
            current_strategy *= legal_mask
            if current_strategy.sum() > 0:
                current_strategy /= current_strategy.sum()
            else:
                current_strategy = legal_mask / legal_mask.sum()

            update_term = pi_p * strategy
            new_strategy = current_strategy + update_term
            self._add_to_memory(self.strategy_memory, (info_vector.copy(), new_strategy))

        return float(node_util)

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------
    def _update_networks(self, update_threshold: int, iteration: int) -> Optional[Dict[str, float]]:
        metrics = super()._update_networks(update_threshold, iteration)

        if metrics is not None:
            metrics.setdefault("regret_param_count", float(self.parameter_counts["regret"]))
            metrics.setdefault("strategy_param_count", float(self.parameter_counts["strategy"]))
            metrics.setdefault("iteration_wall_clock_sec", float(0.0))
            metrics.setdefault("cumulative_wall_clock_sec", float(0.0))

        opponent_metrics = self._update_opponent_model()
        if opponent_metrics:
            if metrics is None:
                metrics = {"iteration": float(iteration)}
            metrics.update({f"opponent_{k}": v for k, v in opponent_metrics.items()})
            metrics.setdefault("opponent_param_count", float(self.opponent_parameter_count))

        return metrics

    def _update_opponent_model(self, batch_size: int = 256) -> Dict[str, float]:
        if not self.opponent_memory:
            return {}

        sample_size = min(len(self.opponent_memory), batch_size)
        batch = random.sample(self.opponent_memory, sample_size)
        infoset_vectors, target_strategies = zip(*batch)

        inputs = torch.from_numpy(np.stack(infoset_vectors)).float().to(self.device)
        targets = torch.from_numpy(np.stack(target_strategies)).float().to(self.device)

        predictions = self.opponent_model(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.opponent_optimizer.zero_grad()
        loss.backward()
        grad_norm = self._gradient_norm(self.opponent_model.parameters())
        self.opponent_optimizer.step()

        epsilon = 1e-8
        with torch.no_grad():
            target_probs = targets.clamp_min(epsilon)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            predicted_probs = predictions.clamp_min(epsilon)
            predicted_probs = predicted_probs / predicted_probs.sum(dim=-1, keepdim=True)
            kl_div = (target_probs * (torch.log(target_probs) - torch.log(predicted_probs))).sum(dim=-1)
            kl_mean = float(kl_div.mean().item())
            variance = float(torch.var(targets).item())

        self.opponent_losses.append(loss.item())
        self.opponent_grad_norms.append(grad_norm)

        return {
            "loss": float(loss.item()),
            "grad_norm": grad_norm,
            "kl": kl_mean,
            "target_variance": variance,
        }

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def _add_to_opponent_memory(self, info_vector: np.ndarray, target_strategy: np.ndarray) -> None:
        if len(self.opponent_memory) >= self.opponent_memory_size:
            self.opponent_memory.pop(random.randint(0, len(self.opponent_memory) - 1))
        self.opponent_memory.append((info_vector, target_strategy))

    @staticmethod
    def _gradient_norm(parameters) -> float:
        grads = [p.grad.detach().flatten() for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        stacked = torch.cat(grads)
        return float(torch.linalg.vector_norm(stacked).item())


__all__ = ["DCFROpponentModelTrainer"]
