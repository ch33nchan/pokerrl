"""
Deep CFR (DCFR) Trainer for Leduc Hold'em using a Transformer-based network.
This adapts the Transformer architecture from Kuhn Poker to the more complex
information set structure of Leduc Hold'em.
"""

import math
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

from qagent.environments.leduc_holdem import LeducEnv
from qagent.utils.logging import TrainingLogger

# --- Constants ---
NUM_ACTIONS = 3
MAX_HISTORY_LEN = 10 # Max actions in a round

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=15):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerStrategyNet(nn.Module):
    """Transformer-based encoder for DCFR betting history."""

    def __init__(
        self,
        num_ranks: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model = embedding_dim

        self.private_card_embedding = nn.Embedding(num_ranks, embedding_dim)
        self.board_card_embedding = nn.Embedding(num_ranks + 1, embedding_dim)  # +1 for 'no card'
        self.action_embedding = nn.Embedding(NUM_ACTIONS + 1, embedding_dim)  # +1 for padding

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(d_model, NUM_ACTIONS)

    def forward(self, private_card, board_card, history):
        private_emb = self.private_card_embedding(private_card).unsqueeze(1)
        board_emb = self.board_card_embedding(board_card).unsqueeze(1)
        history_emb = self.action_embedding(history)
        
        # Concatenate embeddings to form the input sequence
        # [private_card, board_card, action_1, action_2, ...]
        full_sequence = torch.cat([private_emb, board_emb, history_emb], dim=1)
        
        seq_with_pos = self.pos_encoder(full_sequence)
        transformer_output = self.transformer_encoder(seq_with_pos)
        
        # Use the output of the first token ([CLS] style) as the sequence representation
        cls_output = transformer_output[:, 0, :]
        logits = self.fc_out(cls_output)
        return logits

class DCFRTransformerTrainer:
    """Transformer-based Deep CFR trainer for Leduc Hold'em."""

    def __init__(
        self,
        game: Optional[LeducEnv] = None,
        learning_rate: float = 0.001,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        grad_clip: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "leduc_transformer",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game = game if game is not None else LeducEnv()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.regret_net = TransformerStrategyNet(
            num_ranks=self.game.num_ranks,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
        ).to(self.device)
        self.strategy_net = TransformerStrategyNet(
            num_ranks=self.game.num_ranks,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
        ).to(self.device)

        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.num_actions = NUM_ACTIONS
        self.regret_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []
        self.strategy_memory: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]] = []
        self.grad_clip = grad_clip

        self.gradient_norms: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.training_losses: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.target_variances: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.kl_history: List[float] = []
        self.iteration_wall_clock: List[float] = []
        self.training_start_time: Optional[float] = None
        self.parameter_counts: Dict[str, int] = {
            "regret": sum(p.numel() for p in self.regret_net.parameters()),
            "strategy": sum(p.numel() for p in self.strategy_net.parameters()),
        }
        self.log_history: List[Dict[str, float]] = []
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

    def _encode_info_set(self, state):
        """Encodes the Leduc info set into tensors for the Transformer."""
        # Handle different state representations from training and evaluation
        if "private_cards" in state:
            current_player = self.game.get_current_player(state)
            private_card_val = state["private_cards"][current_player]
        else:
            private_card_val = state.get("private_card", 0)

        private_card_rank = int(private_card_val % self.game.num_ranks)
        private_card_tensor = torch.tensor([private_card_rank], dtype=torch.long)

        board_card_val = state.get("board_card")
        if board_card_val is None:
            board_card_tensor = torch.tensor([self.game.num_ranks], dtype=torch.long)
        else:
            board_card_rank = int(board_card_val % self.game.num_ranks)
            board_card_tensor = torch.tensor([board_card_rank], dtype=torch.long)

        history_actions = list(state.get("history", ()))
        history_seq: List[int] = []
        for item in history_actions[-MAX_HISTORY_LEN:]:
            if isinstance(item, tuple):
                _, action = item
            else:
                action = item
            history_seq.append(int(action) + 1)
        padded_history = np.zeros(MAX_HISTORY_LEN, dtype=int)
        if history_seq:
            padded_history[: len(history_seq)] = history_seq

        history_tensor = torch.tensor(padded_history, dtype=torch.long).unsqueeze(0)

        return private_card_tensor, board_card_tensor, history_tensor

    def _prepare_inputs(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(t.to(self.device) for t in inputs)  # type: ignore[misc]

    def get_strategy(self, inputs, legal_actions: list) -> np.ndarray:
        with torch.no_grad():
            regrets = self.regret_net(*self._prepare_inputs(inputs))
        
        positive_regrets = torch.relu(regrets.squeeze(0))
        
        legal_mask = torch.zeros(NUM_ACTIONS, device=self.device)
        if legal_actions:
            legal_mask[legal_actions] = 1
        positive_regrets *= legal_mask

        sum_positive_regrets = torch.sum(positive_regrets)
        
        if sum_positive_regrets > 0:
            strategy = positive_regrets / sum_positive_regrets
        else:
            strategy = torch.zeros(NUM_ACTIONS, device=self.device)
            if legal_actions:
                strategy[legal_actions] = 1.0 / len(legal_actions)
        
        return strategy.detach().cpu().numpy()

    def get_average_strategy(self, inputs, legal_actions: list) -> np.ndarray:
        with torch.no_grad():
            logits = self.strategy_net(*self._prepare_inputs(inputs)).squeeze(0)
            legal_mask = torch.full_like(logits, -np.inf)
            legal_mask[legal_actions] = 0
            masked_logits = logits + legal_mask
            probs = torch.softmax(masked_logits, dim=0)
        return probs.detach().cpu().numpy()

    def train(
        self,
        n_iterations: int,
        update_threshold: int = 100,
        batch_size: int = 128,
        checkpoint_dir: str = "checkpoints_transformer",
        checkpoint_interval: int = 10_000,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()

        iterator = range(1, n_iterations + 1)
        pbar = tqdm(iterator, total=n_iterations, desc="Transformer(DCFR)", leave=True) if tqdm else iterator
        for iteration in pbar:
            iteration_start = time.perf_counter()
            if iteration % 1000 == 0:
                print(f"Transformer Agent: Iteration {iteration}/{n_iterations}...")

            for player_to_train in range(self.game.num_players):
                self._traverse_tree_iterative(player_to_train)

            if iteration % update_threshold == 0:
                metrics: Dict[str, float] = {"iteration": float(iteration)}
                regret_metrics = self._update_regret_net(batch_size)
                if regret_metrics is not None:
                    metrics.update({k: float(v) for k, v in regret_metrics.items()})
                strategy_metrics = self._update_strategy_net(batch_size)
                if strategy_metrics is not None:
                    metrics.update({k: float(v) for k, v in strategy_metrics.items()})

                metrics["regret_mem_size"] = float(len(self.regret_memory))
                metrics["strategy_mem_size"] = float(len(self.strategy_memory))
                metrics["regret_param_count"] = float(self.parameter_counts["regret"])
                metrics["strategy_param_count"] = float(self.parameter_counts["strategy"])
                metrics["iteration_wall_clock_sec"] = float(time.perf_counter() - iteration_start)
                metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)

                msg = (
                    f"it {iteration}/{n_iterations} | "
                    f"RegretLoss {metrics.get('regret_loss', 0.0):.4f} | "
                    f"StrategyLoss {metrics.get('strategy_loss', 0.0):.4f} | "
                    f"RegGrad {metrics.get('regret_grad_norm', 0.0):.2f} | "
                    f"StratGrad {metrics.get('strategy_grad_norm', 0.0):.2f}"
                )
                if tqdm:
                    pbar.set_postfix_str(msg)
                else:
                    print(msg, end='\r')

                self.log_history.append(metrics)
                if self.logger is not None:
                    self.logger.log(metrics)
                self._parquet_records.append(metrics)

                self.regret_memory.clear()
                self.strategy_memory.clear()

            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{iteration}.pt")
                torch.save(self.strategy_net.state_dict(), path)
                print(f"\nTransformer Agent: Saved checkpoint to {path}")

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
        stack: List[Tuple[Dict[str, object], str]] = [(self.game.get_initial_state(), "entry")]
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
                    action_utils = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    node_util = 0.0
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        child_key = self.game.get_state_string(next_state)
                        action_utils[action] = return_values.get(child_key, 0.0)
                        node_util += strategy[action] * action_utils[action]

                    regrets = action_utils - node_util
                    stored_inputs = tuple(t.clone() for t in inputs)
                    self.regret_memory.append((stored_inputs, regrets.astype(np.float32)))
                    self.strategy_memory.append((stored_inputs, strategy.astype(np.float32)))
                    return_values[state_key] = node_util
                else:
                    action = int(sampled_info[state_key])
                    next_state = self.game.get_next_state(state, action)
                    child_key = self.game.get_state_string(next_state)
                    return_values[state_key] = return_values.get(child_key, 0.0)

    def _apply_gradient_clipping(self, parameters: Iterable[torch.nn.parameter.Parameter]) -> float:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        if self.grad_clip is not None and self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().flatten() for p in params]))
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm)

    def _update_regret_net(self, batch_size: int) -> Optional[Dict[str, float]]:
        if not self.regret_memory:
            return None
        memory_batch = random.sample(self.regret_memory, min(len(self.regret_memory), batch_size))

        inputs_list, target_regrets_np = zip(*memory_batch)

        private_cards = torch.cat([item[0] for item in inputs_list])
        board_cards = torch.cat([item[1] for item in inputs_list])
        histories = torch.cat([item[2] for item in inputs_list])

        target_regrets_batch = torch.from_numpy(np.array(target_regrets_np)).float()
        predicted_regrets = self.regret_net(private_cards, board_cards, histories)

        loss = nn.functional.mse_loss(predicted_regrets, target_regrets_batch)

        self.regret_optimizer.zero_grad()
        loss.backward()
        grad_norm = self._apply_gradient_clipping(self.regret_net.parameters())
        self.regret_optimizer.step()

        variance = float(torch.var(target_regrets_batch).item())
        self.gradient_norms["regret"].append(grad_norm)
        self.training_losses["regret"].append(loss.item())
        self.target_variances["regret"].append(variance)

        return {
            "regret_loss": loss.item(),
            "regret_grad_norm": grad_norm,
            "regret_target_variance": variance,
        }

    def _update_strategy_net(self, batch_size: int) -> Optional[Dict[str, float]]:
        if not self.strategy_memory:
            return None
        memory_batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), batch_size))

        inputs_list, target_strategies_np = zip(*memory_batch)

        private_cards = torch.cat([item[0] for item in inputs_list])
        board_cards = torch.cat([item[1] for item in inputs_list])
        histories = torch.cat([item[2] for item in inputs_list])

        target_strategy_batch = torch.from_numpy(np.array(target_strategies_np)).float()
        predicted_logits = self.strategy_net(private_cards, board_cards, histories)

        loss = nn.functional.mse_loss(predicted_logits, target_strategy_batch)

        self.strategy_optimizer.zero_grad()
        loss.backward()
        grad_norm = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.strategy_optimizer.step()

        epsilon = 1e-8
        with torch.no_grad():
            target_probs = target_strategy_batch.clamp_min(epsilon)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            predicted_probs = torch.softmax(predicted_logits, dim=-1).clamp_min(epsilon)
            kl_div = (target_probs * (torch.log(target_probs) - torch.log(predicted_probs))).sum(dim=-1)
            kl_mean = float(kl_div.mean().item())
            variance = float(torch.var(target_strategy_batch).item())

        self.gradient_norms["strategy"].append(grad_norm)
        self.training_losses["strategy"].append(loss.item())
        self.target_variances["strategy"].append(variance)
        self.kl_history.append(kl_mean)

        return {
            "strategy_loss": loss.item(),
            "strategy_grad_norm": grad_norm,
            "strategy_kl": kl_mean,
            "strategy_target_variance": variance,
        }

    def get_average_policy_net(self):
        return self.strategy_net
