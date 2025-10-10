"""DCFR Leduc ablation without card embeddings, with diagnostics instrumentation."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from qagent.environments.leduc_holdem import LeducHoldem
from qagent.utils.logging import TrainingLogger


NUM_ACTIONS = 3
MAX_HISTORY_LEN = 10


class OneHotLSTMAgentNet(nn.Module):
    """Sequence encoder that consumes one-hot card vectors."""

    def __init__(self, card_vector_size: int, lstm_hidden_size: int = 64, fc_hidden_size: int = 128) -> None:
        super().__init__()
        self.card_vector_size = card_vector_size
        self.history_lstm = nn.LSTM(1, lstm_hidden_size, batch_first=True)
        feature_dim = card_vector_size * 2 + lstm_hidden_size
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, NUM_ACTIONS),
        )

    def forward(
        self,
        private_card_oh: torch.Tensor,
        public_card_oh: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.history_lstm(history)
        history_features = hidden.squeeze(0)
        features = torch.cat([private_card_oh, public_card_oh, history_features], dim=-1)
        return self.layers(features)


class DCFROneHotAblationTrainer:
    """DCFR trainer variant without card embeddings (one-hot inputs)."""

    def __init__(
        self,
        game: Optional[LeducHoldem] = None,
        learning_rate: float = 5e-4,
        lstm_hidden_size: int = 64,
        fc_hidden_size: int = 128,
        grad_clip: Optional[float] = 1.0,
        log_dir: Optional[str] = None,
        log_prefix: str = "leduc_lstm_onehot",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game = game if game is not None else LeducHoldem()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        card_vector_size = self.game.num_ranks

        self.regret_net = OneHotLSTMAgentNet(card_vector_size, lstm_hidden_size, fc_hidden_size).to(self.device)
        self.strategy_net = OneHotLSTMAgentNet(card_vector_size, lstm_hidden_size, fc_hidden_size).to(self.device)

        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
        self.grad_clip = grad_clip

        self.regret_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []
        self.strategy_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []

        self.gradient_norms: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.training_losses: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.target_variances: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.kl_history: List[float] = []
        self.training_start_time: Optional[float] = None
        self.numeric_stability_events: Dict[str, int] = {
            "regret_grad_clipped": 0,
            "strategy_grad_clipped": 0,
        }
        self.advantage_stats: List[Dict[str, float]] = []
        self.strategy_stats: List[Dict[str, float]] = []
        self._parquet_records: List[Dict[str, float]] = []
        self._parquet_path: Optional[Path] = Path(log_dir) / f"{log_prefix}_diagnostics.parquet" if log_dir else None

        self.parameter_counts = {
            "regret": sum(p.numel() for p in self.regret_net.parameters()),
            "strategy": sum(p.numel() for p in self.strategy_net.parameters()),
        }

        self.log_history: List[Dict[str, float]] = []
        self.logger: Optional[TrainingLogger] = None
        if log_dir is not None:
            self.logger = TrainingLogger(
                output_dir=log_dir,
                base_filename=f"{log_prefix}_metrics",
                write_csv=log_to_csv,
                write_jsonl=log_to_jsonl,
                overwrite=True,
            )

    @staticmethod
    def _clone_inputs(
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        private_card, public_card, history = inputs
        return (
            private_card.detach().clone(),
            public_card.detach().clone(),
            history.detach().clone(),
        )

    def _to_one_hot(self, card_idx: Optional[int]) -> torch.Tensor:
        vec = torch.zeros(1, self.game.num_ranks, device=self.device)
        if card_idx is not None:
            vec[0, card_idx % self.game.num_ranks] = 1.0
        return vec

    def _encode_info_set(self, state: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        private_cards = state.get("private_cards")
        if private_cards is None:
            raise ValueError("Private cards not yet dealt; cannot encode infoset")

        current_player = int(state.get("current_player", 0))
        private_card_oh = self._to_one_hot(int(private_cards[current_player]))

        board_card = state.get("board_card")
        public_card_oh = self._to_one_hot(int(board_card) if board_card is not None else None)

        history_tensor = torch.zeros(1, MAX_HISTORY_LEN, 1, device=self.device)
        for idx, entry in enumerate(state.get("history", ())[:MAX_HISTORY_LEN]):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                _, action = entry
            else:
                action = entry
            history_tensor[0, idx, 0] = float(action)

        return private_card_oh, public_card_oh, history_tensor

    def _apply_gradient_clipping(self, parameters) -> Tuple[float, bool]:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0, False
        if self.grad_clip is not None and self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            clipped = float(norm) > self.grad_clip if isinstance(norm, torch.Tensor) else norm > self.grad_clip
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().flatten() for p in params]))
            clipped = False
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm), bool(clipped)

    @staticmethod
    def _kl_divergence(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        eps = 1e-8
        preds = predictions.clamp_min(eps)
        targets = targets.clamp_min(eps)
        kl = torch.sum(targets * (torch.log(targets) - torch.log(preds)), dim=-1)
        return float(kl.mean().item())

    def get_strategy(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        legal_actions: List[int],
    ) -> np.ndarray:
        private_card_oh, public_card_oh, history = inputs
        logits = self.regret_net(private_card_oh, public_card_oh, history)
        regrets = torch.relu(logits.squeeze(0))
        mask = torch.zeros(NUM_ACTIONS, device=self.device)
        mask[legal_actions] = 1
        regrets = regrets * mask
        total = regrets.sum()
        if total > 0:
            probs = regrets / total
        else:
            if mask.sum() == 0:
                probs = torch.full((NUM_ACTIONS,), 1.0 / NUM_ACTIONS, device=self.device)
            else:
                probs = mask / mask.sum()
        return probs.detach().cpu().numpy()

    def get_average_strategy(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        legal_actions: List[int],
    ) -> np.ndarray:
        private_card_oh, public_card_oh, history = inputs
        logits = self.strategy_net(private_card_oh, public_card_oh, history).squeeze(0)
        if not legal_actions:
            return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
        mask = torch.full_like(logits, -np.inf)
        mask[legal_actions] = 0.0
        probs = torch.softmax(logits + mask, dim=-1)
        return probs.detach().cpu().numpy()

    def train(
        self,
        n_iterations: int,
        update_threshold: int = 100,
        batch_size: int = 128,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5_000,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()

        for iteration in range(1, n_iterations + 1):
            iteration_start = time.perf_counter()
            for player_to_train in range(self.game.num_players):
                self._traverse_tree_iterative(player_to_train)

            if iteration % update_threshold == 0:
                metrics: Dict[str, float] = {"iteration": float(iteration)}
                regret_metrics = self._update_regret_net(batch_size, iteration)
                if regret_metrics is not None:
                    metrics.update({k: float(v) for k, v in regret_metrics.items()})
                strategy_metrics = self._update_strategy_net(batch_size, iteration)
                if strategy_metrics is not None:
                    metrics.update({k: float(v) for k, v in strategy_metrics.items()})

                metrics["regret_mem_size"] = float(len(self.regret_memory))
                metrics["strategy_mem_size"] = float(len(self.strategy_memory))
                metrics["regret_param_count"] = float(self.parameter_counts["regret"])
                metrics["strategy_param_count"] = float(self.parameter_counts["strategy"])
                metrics["regret_lr"] = float(self.regret_optimizer.param_groups[0]["lr"])
                metrics["strategy_lr"] = float(self.strategy_optimizer.param_groups[0]["lr"])
                metrics["iteration_wall_clock_sec"] = float(time.perf_counter() - iteration_start)
                metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)

                self.log_history.append(metrics)
                if self.logger is not None:
                    self.logger.log(metrics)
                self._parquet_records.append(metrics)

                self.regret_memory.clear()
                self.strategy_memory.clear()

            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{iteration}.pt")
                torch.save(self.strategy_net.state_dict(), path)

        if self.logger is not None:
            self.logger.close()
        if self._parquet_path and self._parquet_records:
            self._parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(self._parquet_records)
            try:
                df.to_parquet(self._parquet_path, index=False)
            except (ImportError, ModuleNotFoundError, ValueError) as err:
                fallback_path = self._parquet_path.with_suffix(".csv")
                df.to_csv(fallback_path, index=False)
                print(
                    f"Warning: Unable to write diagnostics Parquet ({err}). "
                    f"Diagnostics saved as CSV to {fallback_path}."
                )

    def _traverse_tree_iterative(self, player_to_train: int) -> None:
        stack = [(self.game.get_initial_state(), "entry")]
        return_values: Dict[str, float] = {}
        sampled_info: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray] | int | str] = {}

        while stack:
            state, phase = stack.pop()
            state_key = self.game.get_state_string(state)

            if self.game.is_terminal(state):
                return_values[state_key] = self.game.get_payoff(state, player_to_train)
                continue

            if phase == "entry":
                stack.append((state, "exit"))
                if self.game.is_chance_node(state):
                    outcome = self.game.sample_chance_outcome(state)
                    if isinstance(outcome, list):
                        _, next_state = random.choice(outcome)
                    elif isinstance(outcome, tuple) and len(outcome) == 2:
                        _, next_state = outcome
                    else:
                        raise RuntimeError("Unexpected chance outcome format")
                    sampled_info[state_key] = self.game.get_state_string(next_state)
                    stack.append((next_state, "entry"))
                    continue

                current_player = self.game.get_current_player(state)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    return_values[state_key] = self.game.get_payoff(state, player_to_train)
                    continue

                inputs = self._encode_info_set(state)

                if current_player == player_to_train:
                    strategy = self.get_strategy(inputs, legal_actions)
                    sampled_info[state_key] = (self._clone_inputs(inputs), strategy)
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        stack.append((next_state, "entry"))
                else:
                    strategy = self.get_average_strategy(inputs, legal_actions)
                    action = int(np.random.choice(np.arange(NUM_ACTIONS), p=strategy))
                    sampled_info[state_key] = action
                    next_state = self.game.get_next_state(state, action)
                    stack.append((next_state, "entry"))

            else:
                if self.game.is_chance_node(state):
                    child_key = sampled_info[state_key]
                    return_values[state_key] = return_values.get(child_key, 0.0)
                    continue

                current_player = self.game.get_current_player(state)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    continue

                if current_player == player_to_train:
                    stored_inputs, strategy = sampled_info[state_key]  # type: ignore[misc]
                    action_utils = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    node_util = 0.0
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        child_key = self.game.get_state_string(next_state)
                        action_utils[action] = return_values.get(child_key, 0.0)
                        node_util += strategy[action] * action_utils[action]

                    regrets = action_utils - node_util
                    self.regret_memory.append((stored_inputs, regrets.astype(np.float32)))
                    self.strategy_memory.append((stored_inputs, strategy.astype(np.float32)))
                    return_values[state_key] = node_util
                else:
                    action = int(sampled_info[state_key])
                    next_state = self.game.get_next_state(state, action)
                    child_key = self.game.get_state_string(next_state)
                    return_values[state_key] = return_values.get(child_key, 0.0)

    def _update_regret_net(self, batch_size: int, iteration: int) -> Optional[Dict[str, float]]:
        if not self.regret_memory:
            return None
        memory_batch = random.sample(self.regret_memory, min(len(self.regret_memory), batch_size))
        inputs_list, target_regrets_np = zip(*memory_batch)

        private_cards = torch.cat([item[0] for item in inputs_list]).to(self.device)
        public_cards = torch.cat([item[1] for item in inputs_list]).to(self.device)
        histories = torch.cat([item[2] for item in inputs_list]).to(self.device)

        target_regrets = torch.from_numpy(np.array(target_regrets_np)).float().to(self.device)
        predicted_regrets = self.regret_net(private_cards, public_cards, histories)

        loss = nn.functional.mse_loss(predicted_regrets, target_regrets)
        self.regret_optimizer.zero_grad()
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.regret_net.parameters())
        self.regret_optimizer.step()

        variance = float(torch.var(target_regrets).item())
        self.gradient_norms["regret"].append(grad_norm)
        self.training_losses["regret"].append(loss.item())
        self.target_variances["regret"].append(variance)
        if clipped:
            self.numeric_stability_events["regret_grad_clipped"] += 1

        advantages = target_regrets.detach().cpu().numpy()
        self.advantage_stats.append(
            {
                "iteration": float(iteration),
                "phase": "regret",
                "mean": float(np.mean(advantages)),
                "std": float(np.std(advantages)),
                "p50": float(np.percentile(advantages, 50)),
                "p90": float(np.percentile(advantages, 90)),
                "p99": float(np.percentile(advantages, 99)),
            }
        )

        return {
            "regret_loss": loss.item(),
            "regret_grad_norm": grad_norm,
            "regret_target_variance": variance,
        }

    def _update_strategy_net(self, batch_size: int, iteration: int) -> Optional[Dict[str, float]]:
        if not self.strategy_memory:
            return None
        memory_batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), batch_size))
        inputs_list, target_strategies_np = zip(*memory_batch)

        private_cards = torch.cat([item[0] for item in inputs_list]).to(self.device)
        public_cards = torch.cat([item[1] for item in inputs_list]).to(self.device)
        histories = torch.cat([item[2] for item in inputs_list]).to(self.device)

        target_strategies = torch.from_numpy(np.array(target_strategies_np)).float().to(self.device)
        predicted_logits = self.strategy_net(private_cards, public_cards, histories)

        loss = nn.functional.mse_loss(predicted_logits, target_strategies)
        self.strategy_optimizer.zero_grad()
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.strategy_optimizer.step()

        predicted_probs = torch.softmax(predicted_logits.detach(), dim=-1)
        target_probs = target_strategies.detach()
        target_sums = target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_probs = target_probs / target_sums
        kl_mean = self._kl_divergence(predicted_probs, target_probs)
        variance = float(torch.var(target_strategies).item())

        self.gradient_norms["strategy"].append(grad_norm)
        self.training_losses["strategy"].append(loss.item())
        self.target_variances["strategy"].append(variance)
        self.kl_history.append(kl_mean)
        if clipped:
            self.numeric_stability_events["strategy_grad_clipped"] += 1

        strategy_np = predicted_probs.cpu().numpy()
        self.strategy_stats.append(
            {
                "iteration": float(iteration),
                "mean": float(np.mean(strategy_np)),
                "std": float(np.std(strategy_np)),
                "p50": float(np.percentile(strategy_np, 50)),
                "p90": float(np.percentile(strategy_np, 90)),
                "p99": float(np.percentile(strategy_np, 99)),
            }
        )

        return {
            "strategy_loss": loss.item(),
            "strategy_grad_norm": grad_norm,
            "strategy_kl": kl_mean,
            "strategy_target_variance": variance,
        }


__all__ = ["DCFROneHotAblationTrainer"]
