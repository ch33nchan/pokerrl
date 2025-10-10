import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random

from qagent.games.kuhn_poker import KuhnPoker, INFO_SET_SIZE, NUM_ACTIONS

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Neural Network for Regret and Strategy ---

class RegretNet(nn.Module):
    """
    A simple feed-forward neural network to approximate regrets and strategy.
    The network takes an information set vector as input and outputs:
    1. The cumulative regrets for not taking each action.
    2. The current iteration's strategy.
    """
    def __init__(self, input_size=INFO_SET_SIZE, hidden_size=128, output_size=NUM_ACTIONS):
        super(RegretNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

# --- Deep CFR Trainer ---

class DeepCFRTrainer:
    """
    Trains a poker agent using Deep Counterfactual Regret Minimization (Deep CFR).

    This implementation uses two main networks:
    1. Regret Network: Trained over all iterations to approximate cumulative regrets.
    2. Strategy Network: Trained at each iteration `t` to approximate the strategy at `t`.
    
    """
    def __init__(self, game: KuhnPoker, num_iterations: int = 1000, num_traversals: int = 500, memory_size: int = 100000):
        self.game = game
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals

        # Memory buffers for regret and strategy training
        self.regret_memory: Dict[int, List] = {p: [] for p in range(game.num_players)}
        self.strategy_memory: Dict[int, List] = {p: [] for p in range(game.num_players)}
        self.memory_size = memory_size

        # Diagnostics containers
        self.iteration_metrics: List[Dict[str, float]] = []
        self.training_start_time: Optional[float] = None

        self.parameter_counts: Dict[str, int] = {
            "regret": sum(p.numel() for net in self.regret_nets.values() for p in net.parameters()),
            "strategy": sum(p.numel() for net in self.avg_strategy_net.values() for p in net.parameters()),
        }

    def train(self):
        """
        Main training loop for Deep CFR.
        """
        print(f"Starting Deep CFR training for {self.num_iterations} iterations...")
        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()
        for t in tqdm(range(1, self.num_iterations + 1), desc="Deep CFR Iterations"):
            iteration_start = time.perf_counter()
            # 1. Self-Play: Traverse the game tree to generate training data
            for _ in range(self.num_traversals):
                for player in range(self.game.num_players):
                    # We traverse the tree from the perspective of each player
                    initial_state = self.game.get_initial_state()
                    chance_state = self.game.sample_chance_outcome(initial_state)
                    self._traverse_game(chance_state, player, t)

            # 2. Network Training: Update networks using data from memory buffers
            iteration_metrics: Dict[str, float] = {"iteration": float(t)}
            for player in range(self.game.num_players):
                regret_metrics = self._update_network(
                    net=self.regret_nets[player],
                    optimizer=self.regret_optimizers[player],
                    memory=self.regret_memory[player],
                    label=f"regret_p{player}"
                )
                if regret_metrics is not None:
                    iteration_metrics.update(regret_metrics)

                strategy_metrics = self._update_network(
                    net=self.avg_strategy_net[player],
                    optimizer=self.avg_strategy_optimizers[player],
                    memory=self.strategy_memory[player],
                    label=f"strategy_p{player}"
                )
                if strategy_metrics is not None:
                    iteration_metrics.update(strategy_metrics)

            iteration_metrics["regret_param_count"] = float(self.parameter_counts["regret"])
            iteration_metrics["strategy_param_count"] = float(self.parameter_counts["strategy"])
            iteration_metrics["iteration_wall_clock_sec"] = float(time.perf_counter() - iteration_start)
            iteration_metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)
            self.iteration_metrics.append(iteration_metrics)

        print("Training complete.")

    def _traverse_game(self, state, traversing_player: int, iteration: int):
        """
{{ ... }}
                new_cumulative_regrets = prev_cumulative_regrets + regrets
                self.regret_memory[traversing_player].append((info_set_vector, new_cumulative_regrets))

        return node_util

    def _update_network(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        memory: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 128,
        label: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Updates a network using a batch of data from its memory buffer.
        """
        if len(memory) < batch_size:
            return None  # Not enough data to train

        net.train() # Set to training mode

        batch = random.sample(memory, batch_size)
        info_sets, targets = zip(*batch)
        
        info_sets_tensor = torch.FloatTensor(np.array(info_sets)).to(DEVICE)
        targets_tensor = torch.FloatTensor(np.array(target)).to(DEVICE)
{{ ... }}
        optimizer.zero_grad()
        predictions = net(info_sets_tensor)
        loss = self.mse_loss(predictions, targets_tensor)
        loss.backward()
        grad_norm = self._compute_gradient_norm(net)
        optimizer.step()

        target_variance = float(np.var(targets))

        if label is not None:
            if label.startswith("regret"):
                player_idx = int(label.split("p")[-1])
                self.regret_gradient_norms[player_idx].append(grad_norm)
                self.regret_losses[player_idx].append(loss.item())
                self.target_variances[f"regret_{player_idx}"].append(target_variance)
            elif label.startswith("strategy"):
                player_idx = int(label.split("p")[-1])
                self.strategy_gradient_norms[player_idx].append(grad_norm)
                self.strategy_losses[player_idx].append(loss.item())

        metrics = {
            f"{label}_loss": float(loss.item()) if label else float(loss.item()),
            f"{label}_grad_norm": float(grad_norm) if label else float(grad_norm),
            f"{label}_target_variance": float(target_variance) if label else float(target_variance),
        }

        return metrics

    def get_avg_strategy(self):
        """
        Returns the final, averaged strategy for inference.
        """
        return {p: self.avg_strategy_net[p] for p in range(self.game.num_players)}

    @staticmethod
    def _compute_gradient_norm(model: nn.Module) -> float:
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return float(total_norm ** 0.5)

    def save_model(self, player: int, path: str):
        """Saves the average strategy network for a player."""
        print(f"Saving average strategy network for Player {player} to {path}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.avg_strategy_net[player].state_dict(), path)
        print("Model saved.")


if __name__ == '__main__':
    print("Setting up Deep CFR for Kuhn Poker...")
    
    game = KuhnPoker()
    
    # Initialize and run the trainer
    trainer = DeepCFRTrainer(
        game=game,
        num_iterations=100000,
        num_traversals=150,
        memory_size=100000
    )
    
    trainer.train() 
    
    print("\nDeep CFR training finished.")
    
    # Save the trained model for Player 0
    MODEL_SAVE_PATH = "models/deep_cfr_avg_strategy_net_p0.pt"
    trainer.save_model(player=0, path=MODEL_SAVE_PATH)

    print("Next steps would be to evaluate the trained agent.")
    # Example: Get the trained networks
    final_policies = trainer.get_avg_strategy()
    print(f"Trained policy network for Player 0: {final_policies[0]}")
