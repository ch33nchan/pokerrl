"""
Deep CFR (Counterfactual Regret Minimization) agent for Kuhn Poker, using an LSTM
to process the sequence of betting actions.

This implementation uses a multi-input neural network. The card is processed by an
embedding layer, and the betting history is processed by an LSTM layer. The outputs
are then combined and fed into a feed-forward network to produce the strategy.
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qagent.utils.logging import TrainingLogger

class StrategyNet(nn.Module):
    """
    A multi-input neural network for Kuhn Poker using an LSTM for history.
    - Input 1: Card (processed via Embedding)
    - Input 2: Betting history (processed via LSTM)
    """
    def __init__(self, num_actions, card_embedding_dim=16, hidden_dim=64, lstm_hidden_dim=32):
        super(StrategyNet, self).__init__()
        self.card_embedding = nn.Embedding(4, card_embedding_dim) # 3 cards + 1 for padding
        self.lstm = nn.LSTM(input_size=card_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # We need an embedding for the actions in the history sequence
        self.action_embedding = nn.Embedding(3, card_embedding_dim) # 0:pad, 1:pass, 2:bet

        self.fc1 = nn.Linear(card_embedding_dim + lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, card_input, history_input):
        card_embedded = self.card_embedding(card_input)
        
        history_embedded = self.action_embedding(history_input)
        lstm_out, _ = self.lstm(history_embedded)
        
        # Use the last hidden state of the LSTM
        lstm_out_last = lstm_out[:, -1, :]
        
        # Concatenate card embedding with LSTM output
        combined = torch.cat((card_embedded, lstm_out_last), dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.output_layer(x)
        return self.softmax(x)
class DCFRTrainer:
    """
    Manages the training process for the LSTM-based Deep CFR agent on Kuhn Poker.
    Enhanced with gradient tracking and comprehensive diagnostics.
    """
    def __init__(
        self,
        game,
        learning_rate: float = 1e-4,
        memory_size: int = 1_000_000,
        grad_clip: float | None = 1.0,
        log_dir: str | None = None,
        log_prefix: str = "kuhn_lstm",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ):
        self.game = game
        self.num_actions = game.num_actions

        self.regret_net = StrategyNet(self.num_actions)
        self.strategy_net = StrategyNet(self.num_actions)
        
        self.optimizer_regret = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.optimizer_strategy = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.regret_memory = []
        self.strategy_memory = []
        self.memory_size = memory_size

        self.grad_clip = grad_clip

        # Training diagnostics
        self.gradient_norms = {0: [], 1: []}  # Track for both players conceptually
        self.training_losses = {0: [], 1: []}
        self.iteration_exploitability = []
        self.convergence_data = []
        self.log_history: list[Dict[str, float]] = []

        self.logger: TrainingLogger | None = None
        if log_dir is not None:
            self.logger = TrainingLogger(
                output_dir=log_dir,
                base_filename=f"{log_prefix}_metrics",
                write_csv=log_to_csv,
                write_jsonl=log_to_jsonl,
                overwrite=True,
            )

    def _add_to_memory(self, memory, data):
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

    def _encode_info_set(self, state):
        """Uses the game's specific LSTM info set encoding."""
        return self.game.get_lstm_info_set(state)

    def get_strategy(self, inputs):
        """Calculates the current strategy from regrets using the LSTM network."""
        card_input, history_input = inputs
        regrets = self.regret_net(card_input, history_input).detach().numpy().flatten()
        positive_regrets = np.maximum(regrets, 0)
        
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def get_average_strategy(self, inputs, legal_actions_mask):
        """Gets the policy from the averaged strategy network."""
        card_input, history_input = inputs
        strategy = self.strategy_net(card_input, history_input).detach().numpy().flatten()
        
        masked_strategy = strategy * legal_actions_mask
        strategy_sum = np.sum(masked_strategy)
        if strategy_sum > 0:
            return masked_strategy / strategy_sum
        else:
            num_legal_actions = np.sum(legal_actions_mask)
            return legal_actions_mask / num_legal_actions if num_legal_actions > 0 else np.zeros_like(strategy)

    def _update_networks(self, update_threshold: int) -> Optional[Dict[str, float]]:
        """Train the networks on replay data and return metrics."""
        if len(self.regret_memory) < update_threshold or len(self.strategy_memory) < update_threshold:
            return None

        # --- Train Regret Network ---
        self.optimizer_regret.zero_grad()
        batch = random.sample(self.regret_memory, min(len(self.regret_memory), 256))
        card_inputs = torch.cat([item[0][0] for item in batch])
        history_inputs = torch.cat([item[0][1] for item in batch])
        target_regrets = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)

        predicted_regrets = self.regret_net(card_inputs, history_inputs)
        loss_regret = nn.MSELoss()(predicted_regrets, target_regrets)
        loss_regret.backward()

        grad_norm_regret = self._apply_gradient_clipping(self.regret_net.parameters())
        self.gradient_norms[0].append(grad_norm_regret)
        self.training_losses[0].append(loss_regret.item())

        self.optimizer_regret.step()

        # --- Train Strategy Network ---
        self.optimizer_strategy.zero_grad()
        batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), 256))
        card_inputs = torch.cat([item[0][0] for item in batch])
        history_inputs = torch.cat([item[0][1] for item in batch])
        target_strategies = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)

        predicted_strategies = self.strategy_net(card_inputs, history_inputs)
        loss_strategy = nn.MSELoss()(predicted_strategies, target_strategies)
        loss_strategy.backward()

        grad_norm_strategy = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.gradient_norms[1].append(grad_norm_strategy)
        self.training_losses[1].append(loss_strategy.item())

        self.optimizer_strategy.step()

        metrics = {
            "regret_loss": loss_regret.item(),
            "strategy_loss": loss_strategy.item(),
            "regret_grad_norm": grad_norm_regret,
            "strategy_grad_norm": grad_norm_strategy,
            "regret_mem_size": len(self.regret_memory),
            "strategy_mem_size": len(self.strategy_memory),
        }
        self.log_history.append(metrics)
        if self.logger is not None:
            self.logger.log(metrics)
        return metrics

    def train(self, iterations, update_threshold=50, checkpoint_dir="checkpoints_kuhn_lstm", checkpoint_interval=10000):
        """Main training loop with enhanced tracking."""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        for t in range(1, iterations + 1):
            iteration_data = {'iteration': t}

            state = self.game.sample_chance_outcome({})
            self._cfr_traverse(state, t, 1.0, 1.0)

            if t % update_threshold == 0:
                metrics = self._update_networks(update_threshold)
                if metrics is not None:
                    print(
                        f"Iteration {t}/{iterations} | RegretLoss {metrics['regret_loss']:.4f} | "
                        f"StrategyLoss {metrics['strategy_loss']:.4f} | RegGrad {metrics['regret_grad_norm']:.2f} | "
                        f"StratGrad {metrics['strategy_grad_norm']:.2f}",
                        end='\r',
                    )

            # Periodic evaluation and logging
            if t % 1000 == 0:
                exploitability = self._estimate_exploitability()
                self.iteration_exploitability.append((t, exploitability))
                
                iteration_data.update({
                    'exploitability': exploitability,
                    'memory_sizes': {
                        'regret': len(self.regret_memory),
                        'strategy': len(self.strategy_memory)
                    },
                    'recent_gradient_norms': {
                        'regret': np.mean(self.gradient_norms[0][-10:]) if self.gradient_norms[0] else 0,
                        'strategy': np.mean(self.gradient_norms[1][-10:]) if self.gradient_norms[1] else 0
                    }
                })
                
                self.convergence_data.append(iteration_data)
                print(f"Iteration {t}: Exploitability = {exploitability:.6f}")
            
            # Save checkpoints
            if t % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"regret_net_iter_{t}.pt")
                torch.save(self.regret_net.state_dict(), path)
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{t}.pt")
                torch.save(self.strategy_net.state_dict(), path)
                print(f"\nSaved checkpoint to {path}")

        if self.logger is not None:
            self.logger.close()
    
    def _estimate_exploitability(self):
        """Estimate current exploitability (simplified)."""
        if not self.regret_memory:
            return float('inf')
        
        # Simple estimate based on recent regret magnitudes
        recent_regrets = [np.abs(item[1]).mean() for item in self.regret_memory[-100:]]
        return np.mean(recent_regrets) if recent_regrets else float('inf')
    
    def get_training_diagnostics(self):
        """Return comprehensive training diagnostics."""
        return {
            'gradient_norms': self.gradient_norms,
            'training_losses': self.training_losses,
            'exploitability_over_time': self.iteration_exploitability,
            'convergence_data': self.convergence_data,
            'final_memory_sizes': {
                'regret': len(self.regret_memory),
                'strategy': len(self.strategy_memory)
            }
        }
    
    def save_model(self, player: int, path: str):
        """Save trained model."""
        print(f"Saving LSTM strategy network for Player {player} to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.strategy_net.state_dict(), path)
        print("Model saved.")

    def _cfr_traverse(self, state, t, pi_p, pi_o):
        """Recursive traversal function for external sampling CFR."""
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, 0)

        player = self.game.get_current_player(state)
        inputs = self._encode_info_set(state)
        
        strategy = self.get_strategy(inputs)
        legal_actions = self.game.get_legal_actions(state)
        
        strategy = strategy * legal_actions
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy /= strategy_sum
        else:
            strategy = legal_actions / np.sum(legal_actions)

        util = np.zeros(self.num_actions)
        node_util = 0

        for action in range(self.num_actions):
            if legal_actions[action] == 0:
                continue
            
            next_state = self.game.get_next_state(state, action)
            
            if player == 0:
                util[action] = self._cfr_traverse(next_state, t, pi_p * strategy[action], pi_o)
            else:
                util[action] = self._cfr_traverse(next_state, t, pi_p, pi_o * strategy[action])
            
            node_util += strategy[action] * util[action]

        if player == 0:
            regrets = util - node_util
            current_regrets = self.regret_net(*inputs).detach().numpy().flatten()
            new_regrets = current_regrets + pi_o * regrets
            self._add_to_memory(self.regret_memory, (inputs, new_regrets))
            
            current_strategy = self.strategy_net(*inputs).detach().numpy().flatten()
            update_term = pi_p * strategy
            new_strategy = current_strategy + update_term
            self._add_to_memory(self.strategy_memory, (inputs, new_strategy))

        return node_util
