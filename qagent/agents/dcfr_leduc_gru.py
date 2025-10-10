"""GRU-based Deep CFR trainer for Leduc Hold'em."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from qagent.environments.base import InfoSet
from qagent.environments.leduc_holdem import LeducEnv
from qagent.utils.logging import TrainingLogger


CARD_EMBEDDING_DIM = 10
GRU_HIDDEN_SIZE = 64
FC_HIDDEN_SIZE = 128
NUM_ACTIONS = 3
MAX_HISTORY_LEN = 10


class GRUAgentNet(nn.Module):
    """Sequence encoder combining card embeddings with a GRU history encoder."""

    def __init__(
        self,
        num_ranks: int,
        embedding_dim: int = CARD_EMBEDDING_DIM,
        gru_hidden_size: int = GRU_HIDDEN_SIZE,
        num_gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.private_card_embedding = nn.Embedding(num_embeddings=num_ranks, embedding_dim=embedding_dim)
        self.board_card_embedding = nn.Embedding(num_embeddings=num_ranks + 1, embedding_dim=embedding_dim)
        self.history_gru = nn.GRU(
            input_size=NUM_ACTIONS,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        combined_features_size = embedding_dim * 2 + gru_hidden_size
        self.fc_net = nn.Sequential(
            nn.Linear(combined_features_size, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
        )
        self.hidden_size = gru_hidden_size

    def forward(self, private_card: torch.Tensor, board_card: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        private_emb = self.private_card_embedding(private_card)
        board_emb = self.board_card_embedding(board_card)
        if history.size(1) == 0:
            batch_size = private_card.size(0)
            history_features = torch.zeros(batch_size, self.hidden_size, device=private_card.device)
        else:
            _, hidden_states = self.history_gru(history)
            history_features = hidden_states[-1]

        combined = torch.cat([private_emb, board_emb, history_features], dim=1)
        return self.fc_net(combined)


class DCFRGRUTrainer:
    """Deep CFR trainer using a GRU-based regret/strategy network."""

    def __init__(
        self,
        game: Optional[LeducEnv] = None,
        learning_rate: float = 5e-4,
        embedding_dim: int = CARD_EMBEDDING_DIM,
        gru_hidden_size: int = GRU_HIDDEN_SIZE,
        num_gru_layers: int = 1,
        grad_clip: Optional[float] = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "leduc_gru",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game = game if game is not None else LeducEnv()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.regret_net = GRUAgentNet(
            num_ranks=self.game.num_ranks,
            embedding_dim=embedding_dim,
            gru_hidden_size=gru_hidden_size,
            num_gru_layers=num_gru_layers,
        ).to(self.device)
        self.strategy_net = GRUAgentNet(
            num_ranks=self.game.num_ranks,
            embedding_dim=embedding_dim,
            gru_hidden_size=gru_hidden_size,
            num_gru_layers=num_gru_layers,
        ).to(self.device)

        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.num_actions = NUM_ACTIONS
        self.regret_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []
        self.strategy_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []
        self.grad_clip = grad_clip

        self.gradient_norms: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.training_losses: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.kl_history: List[float] = []
        self.target_variances: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.iteration_wall_clock: List[float] = []
        self.training_start_time: Optional[float] = None
        self.parameter_counts: Dict[str, int] = {
            "regret": sum(p.numel() for p in self.regret_net.parameters()),
            "strategy": sum(p.numel() for p in self.strategy_net.parameters()),
        }
        self.log_history: List[Dict[str, float]] = []
        self.numeric_stability_events: Dict[str, int] = {
            "regret_grad_clipped": 0,
            "strategy_grad_clipped": 0,
        }
        self.advantage_stats: List[Dict[str, float]] = []
        self.strategy_stats: List[Dict[str, float]] = []
        self._parquet_records: List[Dict[str, float]] = []
        self._parquet_path: Optional[Path] = Path(log_dir) / f"{log_prefix}_diagnostics.parquet" if log_dir else None

        self.logger: Optional[TrainingLogger] = None
        if log_dir is not None:
            self.logger = TrainingLogger(
                output_dir=log_dir,
                base_filename=f"{log_prefix}_metrics",
                write_csv=log_to_csv,
                write_jsonl=log_to_jsonl,
                overwrite=True,
            )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_info_set(self, state: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_player = self.game.get_current_player(state)

        private_card_rank = state["private_cards"][current_player] % self.game.num_ranks
        private_card_tensor = torch.tensor([private_card_rank], dtype=torch.long)

        board_card = state.get("board_card")
        if board_card is None:
            board_tensor = torch.tensor([self.game.num_ranks], dtype=torch.long)
        else:
            board_tensor = torch.tensor([board_card % self.game.num_ranks], dtype=torch.long)

        history_vec = np.zeros((1, MAX_HISTORY_LEN, NUM_ACTIONS), dtype=np.float32)
        for idx, action in enumerate(state["history"][:MAX_HISTORY_LEN]):
            history_vec[0, idx, action] = 1.0
        history_tensor = torch.from_numpy(history_vec)
        return private_card_tensor, board_tensor, history_tensor

    def _prepare_inputs(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(t.to(self.device) for t in inputs)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------
    def _apply_gradient_clipping(self, parameters: Iterable[torch.nn.parameter.Parameter]) -> Tuple[float, bool]:
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

    def _kl_divergence(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        eps = 1e-8
        preds = predictions.clamp_min(eps)
        targets = targets.clamp_min(eps)
        kl = torch.sum(targets * (torch.log(targets) - torch.log(preds)), dim=-1)
        return float(kl.mean().item())

    def get_strategy(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], legal_actions: List[int]) -> np.ndarray:
        with torch.no_grad():
            regrets = self.regret_net(*self._prepare_inputs(inputs))

        positive_regrets = torch.relu(regrets.squeeze(0))
        mask = torch.zeros(NUM_ACTIONS, device=self.device)
        mask[legal_actions] = 1
        positive_regrets *= mask

        total = torch.sum(positive_regrets)
        if total > 0:
            strategy = positive_regrets / total
        else:
            strategy = torch.zeros(NUM_ACTIONS, device=self.device)
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy.detach().cpu().numpy()

    def get_average_strategy(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], legal_actions: List[int]) -> np.ndarray:
        with torch.no_grad():
            logits = self.strategy_net(*self._prepare_inputs(inputs)).squeeze(0)
            mask = torch.full_like(logits, -np.inf)
            mask[legal_actions] = 0
            probs = torch.softmax(logits + mask, dim=0)
        return probs.detach().cpu().numpy()

    def get_average_policy_net(self) -> nn.Module:
        return self.strategy_net

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        n_iterations: int,
        update_threshold: int = 100,
        batch_size: int = 256,
        checkpoint_dir: str = "checkpoints_gru",
        checkpoint_interval: int = 5_000,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()

        iterator = range(1, n_iterations + 1)
        pbar = tqdm(iterator, total=n_iterations, desc="GRU(DCFR)", leave=True) if tqdm else iterator
        for iteration in pbar:
            iteration_start = time.perf_counter()
            if iteration % 1_000 == 0:
                print(f"GRU Agent: Iteration {iteration}/{n_iterations}...")

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
                metrics["iteration_wall_clock_sec"] = float(time.perf_counter() - iteration_start)
                metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)
                metrics["regret_lr"] = float(self.regret_optimizer.param_groups[0]["lr"])
                metrics["strategy_lr"] = float(self.strategy_optimizer.param_groups[0]["lr"])

                message = (
                    f"it {iteration}/{n_iterations} | "
                    f"RegretLoss {metrics.get('regret_loss', 0.0):.4f} | "
                    f"StrategyLoss {metrics.get('strategy_loss', 0.0):.4f} | "
                    f"RegGrad {metrics.get('regret_grad_norm', 0.0):.2f} | "
                    f"StratGrad {metrics.get('strategy_grad_norm', 0.0):.2f}"
                )
                if tqdm:
                    pbar.set_postfix_str(message)
                else:
                    print(message, end="\r")

                self.log_history.append(metrics)
                if self.logger is not None:
                    self.logger.log(metrics)
                self._parquet_records.append(metrics)

                self.regret_memory.clear()
                self.strategy_memory.clear()

            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{iteration}.pt")
                torch.save(self.get_average_policy_net().state_dict(), path)
                print(f"\nGRU Agent: Saved checkpoint to {path}")

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

    # ------------------------------------------------------------------
    # MCCFR traversal
    # ------------------------------------------------------------------
    def _traverse_tree_iterative(self, player_to_train: int) -> None:
        stack = [(self.game.get_initial_state(), "entry")]
        return_values: Dict[str, float] = {}
        sampled_info: Dict[str, Union[np.ndarray, float, int]] = {}

        while stack:
            state, phase = stack.pop()
            state_key = self.game.get_state_string(state)

            if self.game.is_terminal(state):
                return_values[state_key] = self.game.get_payoff(state, player_to_train)
                continue

            is_chance = self.game.is_chance_node(state)

            if phase == "entry":
                stack.append((state, "exit"))

                if is_chance:
                    outcome = self.game.sample_chance_outcome(state)
                    if isinstance(outcome, list):
                        _, next_state = random.choice(outcome)
                    elif isinstance(outcome, tuple) and len(outcome) == 2:
                        _, next_state = outcome
                    else:
                        raise RuntimeError("Unexpected format from sample_chance_outcome")

                    sampled_info[state_key] = self.game.get_state_string(next_state)
                    stack.append((next_state, "entry"))
                    continue

                current_player = self.game.get_current_player(state)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    stack.pop()
                    return_values[state_key] = self.game.get_payoff(state, player_to_train)
                    continue

                inputs = self._encode_info_set(state)

                if current_player == player_to_train:
                    strategy = self.get_strategy(inputs, legal_actions)
                    sampled_info[state_key] = strategy
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        stack.append((next_state, "entry"))
                else:
                    strategy = self.get_average_strategy(inputs, legal_actions)
                    action = int(np.random.choice(np.arange(NUM_ACTIONS), p=strategy))
                    sampled_info[state_key] = action
                    next_state = self.game.get_next_state(state, action)
                    stack.append((next_state, "entry"))

            else:  # phase == "exit"
                if is_chance:
                    child_key = sampled_info[state_key]
                    return_values[state_key] = return_values.get(child_key, 0.0)
                    continue

                current_player = self.game.get_current_player(state)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    continue

                inputs = self._encode_info_set(state)

                if current_player == player_to_train:
                    strategy = sampled_info[state_key]
                    action_utils = np.zeros(NUM_ACTIONS)
                    node_util = 0.0
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        child_key = self.game.get_state_string(next_state)
                        action_utils[action] = return_values.get(child_key, 0.0)
                        node_util += strategy[action] * action_utils[action]

                    regrets = action_utils - node_util
                    self.regret_memory.append((inputs, regrets))
                    self.strategy_memory.append((inputs, strategy))
                    return_values[state_key] = node_util
                else:
                    action = int(sampled_info[state_key])
                    next_state = self.game.get_next_state(state, action)
                    child_key = self.game.get_state_string(next_state)
                    return_values[state_key] = return_values.get(child_key, 0.0)

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------
    def _update_regret_net(self, batch_size: int, iteration: int) -> Optional[Dict[str, float]]:
        if not self.regret_memory:
            return None
        memory_batch = random.sample(self.regret_memory, min(len(self.regret_memory), batch_size))

        inputs_list, target_regrets_np = zip(*memory_batch)
        private_cards = torch.cat([item[0] for item in inputs_list]).to(self.device)
        board_cards = torch.cat([item[1] for item in inputs_list]).to(self.device)
        histories = torch.cat([item[2] for item in inputs_list]).to(self.device)

        target_regrets = torch.from_numpy(np.array(target_regrets_np, dtype=np.float32)).to(self.device)
        predicted_regrets = self.regret_net(private_cards, board_cards, histories)

        loss = nn.functional.mse_loss(predicted_regrets, target_regrets)
        self.regret_optimizer.zero_grad()
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.regret_net.parameters())
        self.regret_optimizer.step()

        self.gradient_norms["regret"].append(grad_norm)
        self.training_losses["regret"].append(loss.item())
        variance = float(torch.var(target_regrets).item())
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

        inputs_list, target_strategy_np = zip(*memory_batch)
        private_cards = torch.cat([item[0] for item in inputs_list]).to(self.device)
        board_cards = torch.cat([item[1] for item in inputs_list]).to(self.device)
        histories = torch.cat([item[2] for item in inputs_list]).to(self.device)

        target_strategy = torch.from_numpy(np.array(target_strategy_np, dtype=np.float32)).to(self.device)
        predicted_logits = self.strategy_net(private_cards, board_cards, histories)

        loss = nn.functional.mse_loss(predicted_logits, target_strategy)
        self.strategy_optimizer.zero_grad()
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.strategy_optimizer.step()

        predicted_probs = torch.softmax(predicted_logits.detach(), dim=-1)
        target_probs = target_strategy.detach()
        target_sums = target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_probs = target_probs / target_sums
        kl = self._kl_divergence(predicted_probs, target_probs)
        variance = float(torch.var(target_strategy).item())

        self.gradient_norms["strategy"].append(grad_norm)
        self.training_losses["strategy"].append(loss.item())
        self.target_variances["strategy"].append(variance)
        self.kl_history.append(kl)
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
            "strategy_kl": kl,
            "strategy_target_variance": variance,
        }