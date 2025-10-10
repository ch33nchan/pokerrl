"""Deep CFR trainer for Kuhn Poker using the unified environment API."""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qagent.encoders import FlatInfoSetEncoder
from qagent.environments.base import BaseGameEnv, InfoSet
from qagent.environments.kuhn_poker import KuhnEnv
from qagent.utils.logging import TrainingLogger


class RegretNet(nn.Module):
    """Two-layer feed-forward network that outputs cumulative regrets."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.output(x)


class StrategyNet(nn.Module):
    """Two-layer feed-forward network producing a probability distribution."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.softmax(self.output(x))


class DCFRTrainer:
    """Deep CFR trainer operating on environments that implement `BaseGameEnv`."""

    def __init__(
        self,
        game: Optional[BaseGameEnv] = None,
        learning_rate: float = 1e-4,
        memory_size: int = 1_000_000,
        grad_clip: float | None = 1.0,
        lr_scheduler_gamma: float | None = None,
        log_dir: str | None = None,
        log_prefix: str = "kuhn",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game: BaseGameEnv = game if game is not None else KuhnEnv()
        self.info_set_size = self.game.get_obs_dim()
        self.num_actions = self.game.num_actions()

        self.regret_net = RegretNet(self.info_set_size, self.num_actions)
        self.strategy_net = StrategyNet(self.info_set_size, self.num_actions)

        self.optimizer_regret = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.optimizer_strategy = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.regret_scheduler = (
            optim.lr_scheduler.ExponentialLR(self.optimizer_regret, gamma=lr_scheduler_gamma)
            if lr_scheduler_gamma is not None
            else None
        )
        self.strategy_scheduler = (
            optim.lr_scheduler.ExponentialLR(self.optimizer_strategy, gamma=lr_scheduler_gamma)
            if lr_scheduler_gamma is not None
            else None
        )

        self.regret_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.strategy_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.memory_size = memory_size

        self.encoder = FlatInfoSetEncoder()
        self.grad_clip = grad_clip
        self.log_history: List[Dict[str, float]] = []
        self.logger: TrainingLogger | None = None
        if log_dir is not None:
            self.logger = TrainingLogger(
                output_dir=log_dir,
                base_filename=f"{log_prefix}_metrics",
                write_csv=log_to_csv,
                write_jsonl=log_to_jsonl,
                overwrite=True,
            )

    def _add_to_memory(self, memory: List[Tuple[np.ndarray, np.ndarray]], data: Tuple[np.ndarray, np.ndarray]) -> None:
        if len(memory) >= self.memory_size:
            memory.pop(random.randint(0, len(memory) - 1))
        memory.append(data)

    def _apply_gradient_clipping(self, parameters: Iterable[torch.nn.parameter.Parameter]) -> float:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        if self.grad_clip is not None and self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().flatten() for p in params]))
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm)

    def _kl_divergence(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            preds = predictions.clamp_min(1e-8)
            target_probs = targets.clone()
            sums = target_probs.sum(dim=1, keepdim=True)
            mask = sums > 0
            if mask.any():
                target_probs[mask] = target_probs[mask] / sums[mask]
                target_probs = target_probs.clamp_min(1e-8)
                kl = torch.zeros(target_probs.size(0), device=preds.device)
                kl[mask.flatten()] = torch.sum(
                    target_probs[mask] * (torch.log(target_probs[mask]) - torch.log(preds[mask])),
                    dim=1,
                )
                return float(kl[mask.flatten()].mean().item())
            return 0.0

    def get_strategy(self, info_tensor: torch.Tensor, legal_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            regrets = self.regret_net(info_tensor).cpu().numpy().flatten()
        positive_regrets = np.maximum(regrets, 0.0) * legal_mask
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        legal_count = legal_mask.sum()
        if legal_count == 0:
            raise RuntimeError("No legal actions available while computing strategy.")
        return legal_mask / legal_count

    def get_average_strategy(self, info_tensor: torch.Tensor, legal_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            strategy = self.strategy_net(info_tensor).cpu().numpy().flatten()
        masked = strategy * legal_mask
        total = masked.sum()
        if total > 0:
            return masked / total
        legal_count = legal_mask.sum()
        if legal_count == 0:
            raise RuntimeError("No legal actions available while computing average strategy.")
        return legal_mask / legal_count

    def _update_networks(self, update_threshold: int, iteration: int) -> Dict[str, float] | None:
        if len(self.regret_memory) < update_threshold or len(self.strategy_memory) < update_threshold:
            return None

        metrics: Dict[str, float] = {"iteration": float(iteration)}

        regret_metrics = self._update_regret_network()
        metrics.update({f"regret_{k}": v for k, v in regret_metrics.items()})

        strategy_metrics = self._update_strategy_network()
        metrics.update({f"strategy_{k}": v for k, v in strategy_metrics.items()})

        if self.regret_scheduler is not None:
            self.regret_scheduler.step()
        if self.strategy_scheduler is not None:
            self.strategy_scheduler.step()

        metrics["regret_lr"] = self.optimizer_regret.param_groups[0]["lr"]
        metrics["strategy_lr"] = self.optimizer_strategy.param_groups[0]["lr"]
        metrics["regret_mem_size"] = float(len(self.regret_memory))
        metrics["strategy_mem_size"] = float(len(self.strategy_memory))

        self.log_history.append(metrics)
        if self.logger is not None:
            self.logger.log(metrics)
        return metrics

    def _update_regret_network(self) -> Dict[str, float]:
        batch = random.sample(self.regret_memory, min(len(self.regret_memory), 256))
        inputs = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
        targets = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)

        self.optimizer_regret.zero_grad()
        predictions = self.regret_net(inputs)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        grad_norm = self._apply_gradient_clipping(self.regret_net.parameters())
        self.optimizer_regret.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
        }

    def _update_strategy_network(self) -> Dict[str, float]:
        batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), 256))
        inputs = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
        targets = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)

        self.optimizer_strategy.zero_grad()
        predictions = self.strategy_net(inputs)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        grad_norm = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.optimizer_strategy.step()

        kl_div = self._kl_divergence(predictions.detach(), targets)

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "kl": kl_div,
        }

    def train(
        self,
        iterations: int,
        update_threshold: int = 50,
        checkpoint_dir: str = "checkpoints_kuhn_baseline",
        checkpoint_interval: int = 10_000,
    ) -> None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for t in range(1, iterations + 1):
            root = self.game.get_initial_state()
            state = self.game.sample_chance_outcome(root)
            self._cfr_traverse(state, t, 1.0, 1.0)

            if t % update_threshold == 0:
                metrics = self._update_networks(update_threshold, t)
                if metrics:
                    lr_regret = metrics.get("regret_lr", self.optimizer_regret.param_groups[0]["lr"])
                    lr_strategy = metrics.get("strategy_lr", self.optimizer_strategy.param_groups[0]["lr"])
                    msg = (
                        f"Iteration {t}/{iterations} | "
                        f"RegretLoss {metrics.get('regret_loss', 0.0):.4f} | "
                        f"StrategyLoss {metrics.get('strategy_loss', 0.0):.4f} | "
                        f"RegGrad {metrics.get('regret_grad_norm', 0.0):.2f} | "
                        f"StratGrad {metrics.get('strategy_grad_norm', 0.0):.2f} | "
                        f"KL {metrics.get('strategy_kl', 0.0):.4f} | "
                        f"LR(R/S) {lr_regret:.2e}/{lr_strategy:.2e}"
                    )
                else:
                    msg = (
                        f"Iteration {t}/{iterations}: Networks updated. Regret Mem: {len(self.regret_memory)}, "
                        f"Strategy Mem: {len(self.strategy_memory)}"
                    )
                print(msg, end='\r')
                if metrics and self.logger is not None:
                    self.logger.log(metrics)

            if t % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{t}.pt")
                torch.save(self.strategy_net.state_dict(), path)
                print(f"\nSaved checkpoint to {path}")

        if self.logger is not None:
            self.logger.close()

    def _cfr_traverse(self, state: Dict[str, object], t: int, pi_p: float, pi_o: float) -> float:
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, 0)

        player = self.game.get_current_player(state)
        infoset = self._build_infoset(state)
        info_tensor = self.encoder.encode_infoset(self.game, infoset).unsqueeze(0)
        info_vector = info_tensor.squeeze(0).cpu().numpy()

        legal_mask = self._legal_action_mask(state)
        if legal_mask.sum() == 0:
            return self.game.get_payoff(state, 0)

        strategy = self.get_strategy(info_tensor, legal_mask)

        util = np.zeros(self.num_actions, dtype=np.float32)
        node_util = 0.0

        for action in range(self.num_actions):
            if legal_mask[action] == 0:
                continue

            next_state = self.game.get_next_state(state, action)
            if player == 0:
                util[action] = self._cfr_traverse(next_state, t, pi_p * strategy[action], pi_o)
            else:
                util[action] = self._cfr_traverse(next_state, t, pi_p, pi_o * strategy[action])
            node_util += strategy[action] * util[action]

        if player == 0:
            regrets = (util - node_util) * legal_mask
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

    def _build_infoset(self, state: Dict[str, object]) -> InfoSet:
        metadata = {
            "cards": state.get("cards"),
            "history": state["history"],
            "player": self.game.get_current_player(state),
            "terminal": state.get("terminal", False),
        }
        key = self.game.get_state_string(state)
        return InfoSet(key=key, metadata=metadata)

    def _legal_action_mask(self, state: Dict[str, object]) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        legal_actions = self.game.get_legal_actions(state)
        for action in legal_actions:
            mask[action] = 1.0
        return mask
