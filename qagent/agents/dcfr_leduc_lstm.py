"""
An experimental Deep Counterfactual Regret Minimization (DCFR) agent for
Leduc Hold'em that uses an LSTM to encode the betting history.

This agent is part of a study to determine if sequence-aware encoders can
improve sample efficiency and final performance compared to standard one-hot
encodings.
"""
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

from qagent.environments.leduc_holdem import LeducHoldem
from qagent.environments.base import InfoSet
from qagent.utils.logging import TrainingLogger

# --- Constants ---
# Network parameters
CARD_EMBEDDING_DIM = 10
LSTM_HIDDEN_SIZE = 32
COMBINED_FEATURES_SIZE = CARD_EMBEDDING_DIM * 2 + LSTM_HIDDEN_SIZE
FC_HIDDEN_SIZE = 128
NUM_ACTIONS = 3
MAX_HISTORY_LEN = 10 # Max actions in a round

class LSTMAgentNet(nn.Module):
    """
    A neural network for the DCFR agent that uses an LSTM for history.
    """
    def __init__(self, embedding_dim=10, lstm_hidden_size=32, num_lstm_layers=1):
        super(LSTMAgentNet, self).__init__()
        # Card embeddings
        self.private_card_embedding = nn.Embedding(LeducHoldem.num_ranks, embedding_dim)
        self.board_card_embedding = nn.Embedding(LeducHoldem.num_ranks + 1, embedding_dim) # +1 for 'no card'

        # History LSTM
        self.history_lstm = nn.LSTM(
            input_size=NUM_ACTIONS, 
            hidden_size=lstm_hidden_size, 
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Fully connected head
        combined_features_size = embedding_dim * 2 + lstm_hidden_size
        self.fc_net = nn.Sequential(
            nn.Linear(combined_features_size, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS)
        )

    def forward(self, private_card, board_card, history):
        private_card_emb = self.private_card_embedding(private_card)
        board_card_emb = self.board_card_embedding(board_card)

        # LSTM expects (batch, seq_len, input_size)
        _, (h_n, _) = self.history_lstm(history)

        # Use the hidden state from the final LSTM layer for each batch element
        lstm_out = h_n[-1]

        combined = torch.cat([private_card_emb, board_card_emb, lstm_out], dim=1)
        return self.fc_net(combined)

class DCFRTrainer:
    """
    Trains a DCFR agent for Leduc Hold'em using an LSTM-based network.
    """

    def __init__(
        self,
        game: LeducHoldem = None,
        learning_rate: float = 0.0005,
        embedding_dim: int = 32,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        grad_clip: Optional[float] = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "leduc_lstm",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game = game if game is not None else LeducHoldem()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = NUM_ACTIONS
        self.regret_net = LSTMAgentNet(embedding_dim=embedding_dim, lstm_hidden_size=lstm_hidden_size, num_lstm_layers=num_lstm_layers).to(self.device)
        self.strategy_net = LSTMAgentNet(embedding_dim=embedding_dim, lstm_hidden_size=lstm_hidden_size, num_lstm_layers=num_lstm_layers).to(self.device)
        
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
        
        self.regret_memory = []
        self.strategy_memory = []
        self.grad_clip = grad_clip

        self.gradient_norms: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.training_losses: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.kl_history: List[float] = []
        self.target_variances: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.iteration_exploitability: List[Tuple[int, float]] = []
        self.convergence_data: List[Dict[str, object]] = []
        self.log_history: List[Dict[str, float]] = []
        self.training_start_time: Optional[float] = None
        self.total_elapsed: float = 0.0

        self.parameter_counts: Dict[str, int] = {
            "regret": sum(p.numel() for p in self.regret_net.parameters()),
            "strategy": sum(p.numel() for p in self.strategy_net.parameters()),
        }
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

    def _encode_info_set(self, state):
        """
        Encodes the information set into multiple tensors for the LSTM network.
        Returns (private_card_tensor, board_card_tensor, history_tensor)
        """
        current_player = self.game.get_current_player(state)
        
        # 1. Private card
        private_card_rank = state['private_cards'][current_player] % self.game.num_ranks
        private_card_tensor = torch.tensor([private_card_rank], dtype=torch.long)

        # 2. Board card
        if state['board_card'] is not None:
            board_card_rank = state['board_card'] % self.game.num_ranks
            board_card_tensor = torch.tensor([board_card_rank], dtype=torch.long)
        else:
            # Use the extra embedding index for 'no card'
            board_card_tensor = torch.tensor([self.game.num_ranks], dtype=torch.long)

        # 3. Betting history
        history_vec = np.zeros((1, MAX_HISTORY_LEN, NUM_ACTIONS))
        history_len = len(state['history'])
        for i, action in enumerate(state['history']):
            if i < MAX_HISTORY_LEN:
                history_vec[0, i, action] = 1
        history_tensor = torch.from_numpy(history_vec).float()

        return private_card_tensor, board_card_tensor, history_tensor

    def _prepare_inputs(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(t.to(self.device) for t in inputs)  # type: ignore[misc]

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
            legal_mask = torch.full_like(logits, -np.inf, device=self.device)
            legal_mask[legal_actions] = 0
            masked_logits = logits + legal_mask
            probs = torch.softmax(masked_logits, dim=0)
        return probs.detach().cpu().numpy()

    def train(
        self,
        n_iterations: int,
        update_threshold: int = 100,
        batch_size: int = 256,
        checkpoint_dir: str = "checkpoints_lstm",
        checkpoint_interval: int = 5000,
    ) -> None:
        """Main training loop for DCFR using External Sampling MCCFR."""
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()

        iterator = range(1, n_iterations + 1)
        pbar = tqdm(iterator, total=n_iterations, desc="LSTM(DCFR)", leave=True) if tqdm else iterator
        for iteration in pbar:
            iteration_start = time.perf_counter()
            if (tqdm is None) and (iteration % 1000 == 0):
                print(f"LSTM Agent: Starting iteration {iteration}/{n_iterations}...")

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

                iteration_wall_clock = time.perf_counter() - iteration_start
                metrics["iteration_wall_clock_sec"] = float(iteration_wall_clock)
                metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)

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
                    print(message, end='\r')

                self.log_history.append(metrics)
                if self.logger is not None:
                    self.logger.log(metrics)
                self._parquet_records.append(metrics)

                self.regret_memory.clear()
                self.strategy_memory.clear()

            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{iteration}.pt")
                torch.save(self.get_average_policy_net().state_dict(), path)
                print(f"\nLSTM Agent: Saved checkpoint to {path}")

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

    def _build_infoset(self, state):
        """Constructs an `InfoSet` compatible with exploitability evaluation."""
        metadata = {
            "private_cards": tuple(state.get("private_cards", ())),
            "board_card": state.get("board_card"),
            "history": tuple(state.get("history", ())),
            "round": state.get("round", 0),
            "player": self.game.get_current_player(state),
            "deck": tuple(state.get("deck", ())),
            "pot": tuple(state.get("pot_contributions", (1, 1))),
            "folded": state.get("folded_player"),
            "bets": tuple(state.get("bets", (1, 1))),
        }

        board = metadata["board_card"]
        board_str = "N" if board is None else str(board % self.game.num_ranks)
        private_cards = metadata["private_cards"]
        player = metadata["player"]
        if private_cards:
            private_card_rank = private_cards[player] % self.game.num_ranks
        else:
            private_card_rank = -1

        key = (
            f"P{player}|C{private_card_rank}|"
            f"B{board_str}|R{metadata['round']}|H{len(metadata['history'])}"
        )

        return InfoSet(key=key, metadata=metadata)

    def _traverse_tree_iterative(self, player_to_train):
        stack = [(self.game.get_initial_state(), 'entry')]
        return_values = {}
        sampled_info = {}

        while stack:
            state, phase = stack.pop()
            state_key = self.game.get_state_string(state)

            if self.game.is_terminal(state):
                return_values[state_key] = self.game.get_payoff(state, player_to_train)
                continue

            is_chance = self.game.is_chance_node(state)
            
            if phase == 'entry':
                stack.append((state, 'exit'))

                if is_chance:
                    outcome = self.game.sample_chance_outcome(state)
                    if isinstance(outcome, list):
                        _, next_state = random.choice(outcome)
                    elif isinstance(outcome, tuple) and len(outcome) == 2:
                        _, next_state = outcome
                    else:
                        raise RuntimeError("Unexpected format from sample_chance_outcome")
                    sampled_info[state_key] = self.game.get_state_string(next_state)
                    stack.append((next_state, 'entry'))
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
                        stack.append((next_state, 'entry'))
                else:
                    strategy = self.get_average_strategy(inputs, legal_actions)
                    action = np.random.choice(np.arange(NUM_ACTIONS), p=strategy)
                    sampled_info[state_key] = action
                    next_state = self.game.get_next_state(state, action)
                    stack.append((next_state, 'entry'))

            else:  # phase == 'exit'
                if is_chance:
                    child_key = sampled_info[state_key]
                    return_values[state_key] = return_values.get(child_key, 0)
                    continue
                
                current_player = self.game.get_current_player(state)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    continue

                inputs = self._encode_info_set(state)

                if current_player == player_to_train:
                    strategy = sampled_info[state_key]
                    action_utils = np.zeros(NUM_ACTIONS)
                    node_util = 0
                    for action in legal_actions:
                        next_state = self.game.get_next_state(state, action)
                        child_key = self.game.get_state_string(next_state)
                        action_utils[action] = return_values.get(child_key, 0)
                        node_util += strategy[action] * action_utils[action]

                    regrets = action_utils - node_util
                    self.regret_memory.append((inputs, regrets))
                    self.strategy_memory.append((inputs, strategy))
                    return_values[state_key] = node_util
                else:
                    action = sampled_info[state_key]
                    next_state = self.game.get_next_state(state, action)
                    child_key = self.game.get_state_string(next_state)
                    return_values[state_key] = return_values.get(child_key, 0)

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

    def get_average_policy_net(self):
        return self.strategy_net

def main():
    game = LeducHoldem()
    trainer = DCFRTrainer(game)
    
    print("Starting DCFR-LSTM training for Leduc Hold'em...")
    trainer.train(n_iterations=50000, update_threshold=50)
    print("Training complete.")
    
    final_path = "leduc_dcfr_lstm_strategy_net.pt"
    torch.save(trainer.get_average_policy_net().state_dict(), final_path)
    print(f"Saved final strategy network to {final_path}")

if __name__ == "__main__":
    main()
