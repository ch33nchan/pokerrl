"""
Deep CFR GRU implementation for Kuhn Poker.
This provides a simpler recurrent baseline compared to LSTM for the architectural study.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from tqdm import tqdm
import random

from qagent.environments.kuhn_poker import KuhnPoker

# Constants
INFO_SET_SIZE = 12  # Information set representation size
NUM_ACTIONS = 2     # Pass/Check, Bet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUNet(nn.Module):
    """
    GRU-based neural network for Kuhn Poker Deep CFR.
    Uses GRU for sequential processing with fewer parameters than LSTM.
    """
    def __init__(self, input_size=INFO_SET_SIZE, hidden_size=64, output_size=NUM_ACTIONS):
        super(GRUNet, self).__init__()
        
        # Sequential processing layers
        self.gru = nn.GRU(
            input_size=1,  # Process one element at a time
            hidden_size=hidden_size//2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined processing
        combined_size = hidden_size + hidden_size//2
        self.output_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
        
        self.hidden_size = hidden_size//2

    def forward(self, x):
        """
        Forward pass processing both sequential and static features.
        """
        batch_size = x.size(0)
        
        # Process sequential part through GRU
        # Convert input to sequence for GRU processing
        x_seq = x.unsqueeze(-1)  # [batch, features, 1]
        x_seq = x_seq.transpose(1, 2)  # [batch, 1, features]
        
        # Initialize hidden state
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(x.device)
        
        # GRU processing
        gru_out, _ = self.gru(x_seq, h0)
        gru_features = gru_out[:, -1, :]  # Take last output
        
        # Process static features
        static_features = self.feature_net(x)
        
        # Combine features
        combined = torch.cat([static_features, gru_features], dim=1)
        
        # Final output
        return self.output_net(combined)

class DeepCFRGRUTrainer:
    """
    Deep CFR trainer using GRU architecture for Kuhn Poker.
    Includes gradient tracking and comprehensive diagnostics.
    """
    def __init__(self, game: KuhnPoker, num_iterations: int = 1000, 
                 num_traversals: int = 500, memory_size: int = 100000,
                 hidden_size: int = 64, learning_rate: float = 0.001):
        self.game = game
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.memory_size = memory_size
        
        # Memory buffers
        self.regret_memory: Dict[int, List] = {p: [] for p in range(game.num_players)}
        self.strategy_memory: Dict[int, List] = {p: [] for p in range(game.num_players)}
        
        # Networks
        self.regret_nets = {p: GRUNet(hidden_size=hidden_size).to(DEVICE) for p in range(game.num_players)}
        self.avg_strategy_net = {p: GRUNet(hidden_size=hidden_size).to(DEVICE) for p in range(game.num_players)}
        
        # Optimizers
        self.regret_optimizers = {p: optim.Adam(self.regret_nets[p].parameters(), lr=learning_rate) 
                                 for p in range(game.num_players)}
        self.avg_strategy_optimizers = {p: optim.Adam(self.avg_strategy_net[p].parameters(), lr=learning_rate)
                                      for p in range(game.num_players)}
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Training diagnostics
        self.gradient_norms = {p: [] for p in range(game.num_players)}
        self.training_losses = {p: [] for p in range(game.num_players)}
        self.iteration_exploitability = []
        self.convergence_data = []
        
    def train(self):
        """Main training loop with comprehensive logging."""
        print(f"Starting Deep CFR GRU training for {self.num_iterations} iterations...")
        
        for t in tqdm(range(1, self.num_iterations + 1), desc="Deep CFR GRU Iterations"):
            iteration_data = {'iteration': t}
            
            # Self-play traversals
            for _ in range(self.num_traversals):
                for player in range(self.game.num_players):
                    initial_state = self.game.get_initial_state()
                    chance_state = self.game.sample_chance_outcome(initial_state)
                    self._traverse_game(chance_state, player, t)
            
            # Network training with gradient tracking
            iteration_grad_norms = {}
            iteration_losses = {}
            
            for player in range(self.game.num_players):
                regret_loss, regret_grad_norm = self._update_network_with_tracking(
                    self.regret_nets[player], 
                    self.regret_optimizers[player], 
                    self.regret_memory[player],
                    player, 'regret'
                )
                
                strategy_loss, strategy_grad_norm = self._update_network_with_tracking(
                    self.avg_strategy_net[player],
                    self.avg_strategy_optimizers[player],
                    self.strategy_memory[player], 
                    player, 'strategy'
                )
                
                iteration_grad_norms[player] = {
                    'regret': regret_grad_norm,
                    'strategy': strategy_grad_norm
                }
                iteration_losses[player] = {
                    'regret': regret_loss,
                    'strategy': strategy_loss
                }
            
            # Store iteration data
            iteration_data.update({
                'gradient_norms': iteration_grad_norms,
                'losses': iteration_losses,
                'memory_sizes': {p: len(self.regret_memory[p]) for p in range(self.game.num_players)}
            })
            
            self.convergence_data.append(iteration_data)
            
            # Periodic evaluation
            if t % 500 == 0:
                exploitability = self._evaluate_exploitability()
                self.iteration_exploitability.append((t, exploitability))
                print(f"Iteration {t}: Exploitability = {exploitability:.6f}")
        
        print("GRU training complete.")
    
    def _traverse_game(self, state, traversing_player: int, iteration: int):
        """Traverse game tree for training data collection."""
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, traversing_player)
        
        current_player = self.game.get_current_player(state)
        info_set_vector = self.game.get_info_set_vector(state)
        
        # Get strategy from regret network
        regret_net = self.regret_nets[current_player]
        regret_net.eval()
        with torch.no_grad():
            regrets = regret_net(torch.FloatTensor(info_set_vector).unsqueeze(0).to(DEVICE))
        
        # Regret matching to get strategy
        positive_regrets = torch.clamp(regrets, min=0)
        regret_sum = torch.sum(positive_regrets)
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = torch.ones(NUM_ACTIONS).to(DEVICE) / NUM_ACTIONS
        
        strategy = strategy.squeeze().cpu().numpy()
        
        # Store strategy for training
        self.strategy_memory[current_player].append((info_set_vector, strategy))
        if len(self.strategy_memory[current_player]) > self.memory_size:
            self.strategy_memory[current_player].pop(0)
        
        # Sample action and continue traversal
        if current_player == traversing_player:
            # Traverse all actions for counterfactual computation
            action_utilities = np.zeros(NUM_ACTIONS)
            for action in range(NUM_ACTIONS):
                new_state = self.game.take_action(state, action)
                action_utilities[action] = self._traverse_game(new_state, traversing_player, iteration)
            
            # Compute regrets
            strategy_utility = np.sum(strategy * action_utilities)
            regrets = action_utilities - strategy_utility
            
            # Store regret for training
            self.regret_memory[current_player].append((info_set_vector, regrets))
            if len(self.regret_memory[current_player]) > self.memory_size:
                self.regret_memory[current_player].pop(0)
            
            return strategy_utility
        else:
            # Sample action according to strategy
            action = np.random.choice(NUM_ACTIONS, p=strategy)
            new_state = self.game.take_action(state, action)
            return self._traverse_game(new_state, traversing_player, iteration)
    
    def _update_network_with_tracking(self, net, optimizer, memory, player, network_type, batch_size=128):
        """Update network with gradient norm and loss tracking."""
        if len(memory) < batch_size:
            return 0.0, 0.0  # Return zero loss and gradient norm
        
        net.train()
        
        # Sample batch
        batch = random.sample(memory, batch_size)
        info_sets, targets = zip(*batch)
        
        info_sets_tensor = torch.FloatTensor(np.array(info_sets)).to(DEVICE)
        targets_tensor = torch.FloatTensor(np.array(targets)).to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = net(info_sets_tensor)
        loss = self.mse_loss(predictions, targets_tensor)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Store diagnostics
        self.gradient_norms[player].append(total_norm)
        self.training_losses[player].append(loss.item())
        
        # Update parameters
        optimizer.step()
        
        return loss.item(), total_norm
    
    def _evaluate_exploitability(self):
        """Compute exploitability of current strategy."""
        # Simplified exploitability computation for Kuhn Poker
        # This is a placeholder - full implementation would compute exact exploitability
        total_regret = 0
        
        for player in range(self.game.num_players):
            if len(self.regret_memory[player]) > 0:
                recent_regrets = [abs(r) for _, regrets in self.regret_memory[player][-100:] for r in regrets]
                if recent_regrets:
                    total_regret += np.mean(recent_regrets)
        
        return total_regret / self.game.num_players
    
    def get_training_diagnostics(self):
        """Return comprehensive training diagnostics."""
        return {
            'gradient_norms': self.gradient_norms,
            'training_losses': self.training_losses,
            'exploitability_over_time': self.iteration_exploitability,
            'convergence_data': self.convergence_data,
            'final_memory_sizes': {p: len(self.regret_memory[p]) for p in range(self.game.num_players)}
        }
    
    def save_model(self, player: int, path: str):
        """Save trained model."""
        print(f"Saving GRU strategy network for Player {player} to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.avg_strategy_net[player].state_dict(), path)
        print("Model saved.")
    
    def get_avg_strategy(self):
        """Return average strategy networks."""
        return {p: self.avg_strategy_net[p] for p in range(self.game.num_players)}

if __name__ == '__main__':
    print("Setting up Deep CFR GRU for Kuhn Poker...")
    
    game = KuhnPoker()
    
    # Initialize and run trainer
    trainer = DeepCFRGRUTrainer(
        game=game,
        num_iterations=10000,
        num_traversals=150,
        memory_size=50000,
        hidden_size=64,
        learning_rate=0.001
    )
    
    trainer.train()
    
    # Save diagnostics
    diagnostics = trainer.get_training_diagnostics()
    print(f"Training completed. Final gradient norms: {[np.mean(norms[-10:]) if norms else 0 for norms in diagnostics['gradient_norms'].values()]}")
    
    # Save model
    trainer.save_model(player=0, path="models/deep_cfr_gru_kuhn_p0.pt")