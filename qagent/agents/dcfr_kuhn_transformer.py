"""
Deep CFR (DCFR) Trainer for Kuhn Poker using a Transformer-based network.

This agent uses a Transformer encoder to process the sequence of betting actions,
offering a different approach to sequence modeling compared to the LSTM.
It's designed to be part of the comparative study of different neural
network architectures for Deep CFR.
"""

import os
import random
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qagent.environments.kuhn_poker import KuhnPoker
from qagent.utils.logging import TrainingLogger

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=10):
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

class StrategyNet(nn.Module):
    """
    A Transformer-based network for approximating strategies in Kuhn Poker.
    It takes the private card and the betting history as separate inputs.
    """
    def __init__(self, game: KuhnPoker, embedding_dim=32, nhead=2, nlayers=2, dropout=0.1):
        super(StrategyNet, self).__init__()
        self.num_actions = game.get_num_actions()
        d_model = embedding_dim

        # Embeddings
        self.card_embedding = nn.Embedding(game.NUM_CARDS, embedding_dim)
        # Action embedding: 0=pad, 1=pass, 2=bet
        self.action_embedding = nn.Embedding(3, embedding_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        # Output layer
        self.fc_out = nn.Linear(d_model, self.num_actions)

    def forward(self, inputs):
        card_tensor, history_tensor = inputs
        
        # Embed the inputs
        card_embed = self.card_embedding(card_tensor) # (batch, 1, card_embedding_dim)
        history_embed = self.action_embedding(history_tensor) # (batch, seq_len, action_embedding_dim)
        
        # Combine embeddings: prepend card embedding to the history sequence
        # The card acts like a special [CLS] token
        full_sequence = torch.cat([card_embed, history_embed], dim=1) # (batch, seq_len + 1, d_model)
        
        # Add positional encoding
        seq_with_pos = self.pos_encoder(full_sequence)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(seq_with_pos) # (batch, seq_len + 1, d_model)
        
        # We take the output corresponding to the first token (the card)
        # as the aggregated representation of the whole sequence.
        cls_output = transformer_output[:, 0, :] # (batch, d_model)
        
        # Final logits
        logits = self.fc_out(cls_output)
        
        return logits

class DCFRTrainer:
    """The main training class for the Transformer-based DCFR agent with gradient clipping."""

    def __init__(
        self,
        game: KuhnPoker,
        learning_rate: float = 0.001,
        gradient_clip_norm: float | None = 1.0,
        layer_norm_type: str = 'post',
        memory_size: int = 100000,
        log_dir: Optional[str] = None,
        log_prefix: str = 'kuhn_transformer',
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
    ) -> None:
        self.game = game
        self.strategy_net = StrategyNet(game)
        self.optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
        self.gradient_clip_norm = gradient_clip_norm
        self.layer_norm_type = layer_norm_type
        
        # Apply gradient clipping if specified
        self.use_gradient_clipping = gradient_clip_norm is not None
        
        # Memory management
        self.info_set_map = {}
        self.regret_memory = defaultdict(lambda: np.zeros(self.game.num_actions))
        self.strategy_memory = []
        self.memory_size = memory_size
        
        # Training diagnostics
        self.gradient_norms = {0: [], 1: []}
        self.training_losses = {0: [], 1: []}
        self.iteration_exploitability = []
        self.convergence_data = []
        self.gradient_explosion_events = []
        self.strategy_memory = defaultdict(lambda: np.zeros(self.game.num_actions))
        self.memory_counter = 0
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

    def _get_info_set_id(self, inputs):
        """Generates a unique, hashable ID for an information set."""
        # Create a tuple of tensor data to use as a dictionary key
        key = (
            tuple(inputs[0].numpy().flatten()), 
            tuple(inputs[1].numpy().flatten())
        )
        if key not in self.info_set_map:
            self.info_set_map[key] = self.memory_counter
            self.memory_counter += 1
        return self.info_set_map[key]

    def _encode_info_set(self, state: dict):
        """Encodes the state into tensors suitable for the Transformer network."""
        player = self.game.get_current_player(state)
        card = state['cards'][player]
        history = state['history']

        card_tensor = torch.tensor([card], dtype=torch.long).unsqueeze(0)

        action_map = {'p': 1, 'b': 2}
        history_seq = [action_map.get(a, 0) for a in history]
        if not history_seq:
            history_seq = [0] # Padded start token
        
        history_tensor = torch.tensor(history_seq, dtype=torch.long).unsqueeze(0)

        return card_tensor, history_tensor

    def get_strategy(self, inputs):
        """Get a strategy from the network for a given info set."""
        with torch.no_grad():
            logits = self.strategy_net(inputs)
            # Apply softmax to get probabilities
            probabilities = nn.functional.softmax(logits, dim=1).squeeze(0).numpy()
        return probabilities

    def get_average_strategy(self, inputs, legal_actions_mask):
        """Get the average strategy from the network."""
        strategy = self.get_strategy(inputs)
        strategy[~legal_actions_mask] = 0
        norm_sum = np.sum(strategy)
        if norm_sum > 0:
            strategy /= norm_sum
        else:
            # Fallback for all-zero case
            strategy[legal_actions_mask] = 1.0 / np.sum(legal_actions_mask)
        return strategy

    def train(self, iterations, update_threshold, checkpoint_dir, checkpoint_interval):
        """Main training loop with enhanced gradient tracking and clipping."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for i in range(1, iterations + 1):
            iteration_data = {'iteration': i}
            
            initial_state = self.game.sample_chance_outcome({})
            for player_to_train in range(self.game.num_players):
                self._walk_tree(initial_state, player_to_train, 1.0, 1.0)

            if i % update_threshold == 0 and len(self.regret_memory) > 100:
                metrics = self._update_network_with_tracking()
                if metrics is not None:
                    iteration_data.update(metrics)
                    if metrics.get('gradient_clipped'):
                        self.gradient_explosion_events.append(i)
                    print(
                        f"Iteration {i}/{iterations} | Loss {metrics['loss']:.4f} | Grad {metrics['gradient_norm']:.2f} "
                        f"| Clipped {metrics['gradient_clipped']} | Mem {metrics['memory_size']}",
                        end='\r',
                    )
            
            # Periodic evaluation
            if i % 1000 == 0:
                exploitability = self._estimate_exploitability()
                self.iteration_exploitability.append((i, exploitability))
                iteration_data['exploitability'] = exploitability
                self.convergence_data.append(iteration_data)
                print(f"Iteration {i}: Exploitability = {exploitability:.6f}, Grad Explosions: {len(self.gradient_explosion_events)}")

            if i % checkpoint_interval == 0:
                print(
                    f"\nIteration {i}/{iterations}: Regret Mem {len(self.regret_memory)}, "
                    f"Strategy Mem {len(self.strategy_memory)}"
                )
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{i}.pt")
                torch.save(self.strategy_net.state_dict(), path)
                print(f"Saved checkpoint to {path}")

        if self.logger is not None:
            self.logger.close()

    def _walk_tree(self, state, training_player, p0, p1):
        """Recursive tree traversal for CFR."""
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, training_player)

        current_player = self.game.get_current_player(state)
        inputs = self._encode_info_set(state)
        legal_actions = self.game.get_legal_actions(state)
        
        strategy = self.get_strategy(inputs)
        
        # Mask illegal actions
        legal_mask = np.zeros(self.game.num_actions, dtype=bool)
        legal_mask[legal_actions] = True
        strategy[~legal_mask] = 0
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy /= strategy_sum
        else:
            strategy[legal_mask] = 1.0 / np.sum(legal_mask)

        if current_player == training_player:
            node_util = 0
            action_utils = np.zeros(self.game.num_actions)

            for action in legal_actions:
                next_state = self.game.get_next_state(state, action)
                action_utils[action] = self._walk_tree(next_state, training_player, p0 * strategy[action], p1)
                node_util += strategy[action] * action_utils[action]

            # Accumulate regrets and strategy
            info_set_id = self._get_info_set_id(inputs)
            regrets = action_utils - node_util
            self.regret_memory[info_set_id] += regrets
            self.strategy_memory[info_set_id] += p0 * strategy
            return node_util
        else:
            # Opponent's turn
            next_action = np.random.choice(np.arange(self.game.num_actions), p=strategy)
            next_state = self.game.get_next_state(state, next_action)
            return self._walk_tree(next_state, training_player, p0, p1 * strategy[next_action])

    def _update_network_with_tracking(self) -> Optional[Dict[str, float]]:
        """Sample from memory and update the strategy network with gradient tracking."""
        if not self.info_set_map:
            return None

        # Create a dataset from the collected regrets
        info_set_keys = list(self.regret_memory.keys())
        
        # Sample a minibatch of info sets
        sample_size = min(len(info_set_keys), 512)  # Reduced batch size for stability
        sampled_keys = random.sample(info_set_keys, sample_size)

        # Retrieve the tensors and regrets for the minibatch
        key_to_input = {v: k for k, v in self.info_set_map.items()}
        
        card_tensors = []
        history_tensors = []
        regret_targets = []

        for key in sampled_keys:
            if key in key_to_input:
                inputs = key_to_input[key]
                card_tensors.append(inputs[0])
                history_tensors.append(inputs[1])
                
                # Convert cumulative regrets to strategy via regret matching
                regrets = self.regret_memory[key]
                positive_regrets = np.maximum(regrets, 0)
                regret_sum = np.sum(positive_regrets)
                if regret_sum > 0:
                    strategy = positive_regrets / regret_sum
                else:
                    strategy = np.ones(self.game.num_actions) / self.game.num_actions
                regret_targets.append(strategy)

        if not card_tensors:
            return None

        # Stack tensors
        card_batch = torch.cat(card_tensors, dim=0)
        history_batch = torch.cat(history_tensors, dim=0)
        targets = torch.tensor(np.array(regret_targets), dtype=torch.float32)

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.strategy_net((card_batch, history_batch))
        predictions = nn.functional.softmax(logits, dim=1)
        
        # Use KL divergence loss for probability distributions
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(logits, dim=1), 
            targets, 
            reduction='batchmean'
        )
        
        # Backward pass
        loss.backward()

        grad_norm = self._apply_gradient_clipping(self.strategy_net.parameters())
        gradient_clipped = self.use_gradient_clipping and self.gradient_clip_norm is not None and grad_norm > self.gradient_clip_norm

        self.gradient_norms[0].append(grad_norm)
        self.training_losses[0].append(loss.item())

        self.optimizer.step()

        metrics = {
            'loss': loss.item(),
            'gradient_norm': grad_norm,
            'gradient_clipped': gradient_clipped,
            'memory_size': len(self.regret_memory),
        }
        self.log_history.append(metrics)
        if self.logger is not None:
            self.logger.log(metrics)
        return metrics

    def _apply_gradient_clipping(self, parameters: Iterable[torch.nn.parameter.Parameter]) -> float:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        if self.use_gradient_clipping and self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_norm)
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().flatten() for p in params]))
        if isinstance(norm, torch.Tensor):
            norm_val = float(norm.item())
        else:
            norm_val = float(norm)
        return norm_val

    def _update_network(self):
        """Legacy method for compatibility."""
        metrics = self._update_network_with_tracking()
        return metrics['loss'] if metrics is not None else 0.0
    
    def _estimate_exploitability(self):
        """Estimate current exploitability."""
        if not self.regret_memory:
            return float('inf')
        
        # Simple estimate based on regret magnitudes
        total_regret = 0
        for regrets in self.regret_memory.values():
            total_regret += np.sum(np.abs(regrets))
        
        return total_regret / len(self.regret_memory) if self.regret_memory else float('inf')
    
    def get_training_diagnostics(self):
        """Return comprehensive training diagnostics."""
        return {
            'gradient_norms': self.gradient_norms,
            'training_losses': self.training_losses,
            'exploitability_over_time': self.iteration_exploitability,
            'convergence_data': self.convergence_data,
            'gradient_explosion_events': self.gradient_explosion_events,
            'gradient_clip_norm': self.gradient_clip_norm,
            'final_memory_sizes': {'regret': len(self.regret_memory)}
        }
    
    def save_model(self, player: int, path: str):
        """Save trained model."""
        print(f"Saving Transformer strategy network for Player {player} to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.strategy_net.state_dict(), path)
        print("Model saved.")
