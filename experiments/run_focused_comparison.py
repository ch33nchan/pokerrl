#!/usr/bin/env python3
"""
Focused Architecture Comparison for Deep CFR Study

This script compares different neural network architectures for Deep CFR
on Kuhn poker with proper implementation and meaningful differences.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pyspiel

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Game configuration
GAME_NAME = "kuhn_poker"
ITERATIONS = 500
EVAL_EVERY = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.01
BUFFER_SIZE = 10000

# Architecture configurations with intentional differences
ARCHITECTURES = {
    "baseline": {
        "hidden_sizes": [64, 64],
        "activation": "relu",
        "learning_rate": 0.01,
        "description": "Standard MLP baseline"
    },
    "wide": {
        "hidden_sizes": [128, 128],
        "activation": "relu",
        "learning_rate": 0.01,
        "description": "Wider network for more capacity"
    },
    "deep": {
        "hidden_sizes": [64, 64, 64],
        "activation": "relu",
        "learning_rate": 0.01,
        "description": "Deeper network with more layers"
    },
    "fast": {
        "hidden_sizes": [32, 32],
        "activation": "relu",
        "learning_rate": 0.02,  # Faster learning
        "description": "Smaller network with faster learning"
    }
}

class DeepCFRArchitecture:
    """Deep CFR with configurable architecture."""

    def __init__(self, game: pyspiel.Game, arch_config: Dict[str, Any], seed: int = 0):
        self.game = game
        self.arch_config = arch_config
        self.seed = seed

        # Game parameters
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()
        self.state_size = game.information_state_tensor_size()

        # Networks
        self.regret_net = self._create_network()
        self.strategy_net = self._create_network()

        # Optimizers with architecture-specific learning rates
        lr = arch_config.get("learning_rate", LEARNING_RATE)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr)

        # Experience buffers
        self.regret_buffer = []
        self.strategy_buffer = []

        # Training metrics
        self.training_step = 0
        self.losses = {"regret": [], "strategy": []}

    def _create_network(self) -> nn.Module:
        """Create network based on architecture configuration."""
        hidden_sizes = self.arch_config.get("hidden_sizes", [64, 64])
        activation = self.arch_config.get("activation", "relu")

        layers = []
        input_size = self.state_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())

            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, self.num_actions))

        return nn.Sequential(*layers)

    def _encode_state(self, state: pyspiel.State) -> torch.Tensor:
        """Encode state to tensor."""
        info_state = state.information_state_tensor()
        return torch.tensor(info_state, dtype=torch.float32)

    def _get_strategy(self, state: pyspiel.State, use_regret: bool = True) -> np.ndarray:
        """Get strategy from network."""
        state_tensor = self._encode_state(state)

        with torch.no_grad():
            if use_regret:
                output = self.regret_net(state_tensor)
                # Apply regret matching
                regrets = torch.clamp(output, min=0)
            else:
                output = self.strategy_net(state_tensor)
                regrets = output

            strategy = torch.softmax(regrets, dim=-1)

        # Mask illegal actions
        legal_actions = state.legal_actions()
        legal_mask = torch.zeros(self.num_actions)
        legal_mask[legal_actions] = 1.0

        strategy = strategy * legal_mask
        strategy = strategy / torch.sum(strategy) if torch.sum(strategy) > 0 else legal_mask / len(legal_actions)

        return strategy.numpy()

    def train(self, iterations: int, eval_every: int = 25) -> Dict[str, List[float]]:
        """Train Deep CFR with architecture-specific configuration."""
        results = {
            "iterations": [],
            "exploitability": [],
            "nashconv": [],
            "regret_loss": [],
            "strategy_loss": []
        }

        print(f"Training {self.arch_config['description']}")

        for it in range(iterations):
            player = it % self.num_players

            # External sampling traversal
            self._traverse_and_collect(self.game.new_initial_state(), player, 1.0)

            # Update networks
            if it % 10 == 0 and len(self.regret_buffer) > BATCH_SIZE:
                regret_loss, strategy_loss = self._update_networks()
                self.losses["regret"].append(regret_loss)
                self.losses["strategy"].append(strategy_loss)

            # Initialize loss variables
            current_regret_loss = self.losses["regret"][-1] if self.losses["regret"] else 0.0
            current_strategy_loss = self.losses["strategy"][-1] if self.losses["strategy"] else 0.0

            # Evaluation
            if it % eval_every == 0:
                exp, nash = self._evaluate_strategy()

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)
                results["regret_loss"].append(current_regret_loss)
                results["strategy_loss"].append(current_strategy_loss)

                print(f"Iteration {it}: Exploitability = {exp:.4f}")

        return results

    def _traverse_and_collect(self, state: pyspiel.State, player: int, reach_prob: float):
        """External sampling traversal with data collection."""
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome = random.choice([o[0] for o in outcomes])
            probs = [o[1] for o in outcomes]
            outcome_idx = np.random.choice(range(len(outcomes)), p=probs)
            state.apply_action(outcomes[outcome_idx][0])
            return self._traverse_and_collect(state, player, reach_prob * probs[outcome_idx])

        current_player = state.current_player()
        legal_actions = state.legal_actions()

        # Get strategy
        if current_player == player:
            strategy = self._get_strategy(state, use_regret=True)
        else:
            strategy = self._get_strategy(state, use_regret=False)

        if current_player == player:
            # Collect training samples
            state_tensor = self._encode_state(state)
            info_set = state.information_state_string(current_player)

            for action in legal_actions:
                next_state = state.clone()
                next_state.apply_action(action)
                cf_value = self._traverse_and_collect(next_state, player, reach_prob * strategy[action])

                # Store regret sample
                self.regret_buffer.append({
                    "state": state_tensor.numpy().copy(),
                    "info_set": info_set,
                    "action": action,
                    "cf_value": cf_value,
                    "reach_prob": reach_prob
                })

            # Store strategy sample
            self.strategy_buffer.append({
                "state": state_tensor.numpy().copy(),
                "info_set": info_set,
                "strategy": strategy.copy(),
                "reach_prob": reach_prob
            })

        # Sample action and continue
        if len(legal_actions) > 0:
            action = np.random.choice(legal_actions, p=strategy[legal_actions])
            state.apply_action(action)
            return self._traverse_and_collect(state, player, reach_prob)
        else:
            return 0.0

    def _update_networks(self) -> Tuple[float, float]:
        """Update networks with architecture-specific learning."""
        regret_loss = 0.0
        strategy_loss = 0.0

        # Update regret network
        if len(self.regret_buffer) > BATCH_SIZE:
            batch = random.sample(self.regret_buffer, BATCH_SIZE)

            states = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
            actions = torch.tensor([item["action"] for item in batch], dtype=torch.long)
            cf_values = torch.tensor([item["cf_value"] for item in batch], dtype=torch.float32)

            regrets = self.regret_net(states)
            selected_regrets = regrets[range(len(actions)), actions]

            regret_loss = nn.MSELoss()(selected_regrets, cf_values)

            self.regret_optimizer.zero_grad()
            regret_loss.backward()
            self.regret_optimizer.step()

        # Update strategy network
        if len(self.strategy_buffer) > BATCH_SIZE:
            batch = random.sample(self.strategy_buffer, BATCH_SIZE)

            states = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
            target_strategies = torch.tensor([item["strategy"] for item in batch], dtype=torch.float32)

            pred_strategies = torch.softmax(self.strategy_net(states), dim=-1)
            strategy_loss = nn.MSELoss()(pred_strategies, target_strategies)

            self.strategy_optimizer.zero_grad()
            strategy_loss.backward()
            self.strategy_optimizer.step()

        # Trim buffers
        if len(self.regret_buffer) > BUFFER_SIZE:
            self.regret_buffer = self.regret_buffer[-BUFFER_SIZE:]
        if len(self.strategy_buffer) > BUFFER_SIZE:
            self.strategy_buffer = self.strategy_buffer[-BUFFER_SIZE:]

        self.training_step += 1
        return regret_loss.item() if regret_loss else 0.0, strategy_loss.item() if strategy_loss else 0.0

    def _evaluate_strategy(self) -> Tuple[float, float]:
        """Evaluate current strategy using Monte Carlo simulation."""
        try:
            # Create policy from strategy network
            def policy_bot(state: pyspiel.State) -> int:
                strategy = self._get_strategy(state, use_regret=False)
                legal_actions = state.legal_actions()
                if len(legal_actions) == 0:
                    return 0
                probs = strategy[legal_actions]
                probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(len(legal_actions)) / len(legal_actions)
                return np.random.choice(legal_actions, p=probs)

            # Monte Carlo evaluation
            nash_conv = 0.0
            num_simulations = 500  # Reduced for speed

            for _ in range(num_simulations):
                state = self.game.new_initial_state()
                while not state.is_terminal():
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        probs = [o[1] for o in outcomes]
                        outcome_idx = np.random.choice(range(len(outcomes)), p=probs)
                        state.apply_action(outcomes[outcome_idx][0])
                    else:
                        action = policy_bot(state)
                        state.apply_action(action)

                nash_conv += abs(state.returns()[0])  # Game value for player 0

            nash_conv /= num_simulations
            exploitability = nash_conv  # For 2-player zero-sum games

            return exploitability, nash_conv

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), float('inf')


def run_architecture_comparison():
    """Run comparison of different architectures."""
    print("Starting Focused Deep CFR Architecture Study")
    print(f"Game: {GAME_NAME}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Architectures: {list(ARCHITECTURES.keys())}")

    # Load game
    game = pyspiel.load_game(GAME_NAME)

    # Store results
    all_results = {}

    # Run each architecture
    for arch_name, arch_config in ARCHITECTURES.items():
        print(f"\n=== Testing {arch_name}: {arch_config['description']} ===")

        # Create agent with architecture
        agent = DeepCFRArchitecture(game, arch_config, seed=SEED)

        # Train
        start_time = time.time()
        results = agent.train(ITERATIONS, EVAL_EVERY)
        end_time = time.time()

        # Store results
        results.update({
            "architecture": arch_name,
            "config": arch_config,
            "wall_clock_s": end_time - start_time,
            "final_exploitability": results["exploitability"][-1] if results["exploitability"] else float('inf'),
            "final_nashconv": results["nashconv"][-1] if results["nashconv"] else float('inf'),
            "parameter_count": sum(p.numel() for p in agent.regret_net.parameters()),
            "losses": agent.losses
        })

        all_results[arch_name] = results

        print(f"Final exploitability: {results['final_exploitability']:.4f}")
        print(f"Training time: {results['wall_clock_s']:.2f}s")
        print(f"Parameters: {results['parameter_count']}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/focused_architecture_comparison.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        converted_results = convert_numpy(all_results)
        json.dump(converted_results, f, indent=2)

    # Print summary comparison
    print(f"\n=== Architecture Comparison Summary ===")
    print(f"{'Architecture':<12} {'Final Exploitability':<20} {'Time (s)':<10} {'Parameters':<12}")
    print("-" * 60)

    for arch_name, results in all_results.items():
        exp = results["final_exploitability"]
        time_s = results["wall_clock_s"]
        params = results["parameter_count"]
        print(f"{arch_name:<12} {exp:<20.4f} {time_s:<10.2f} {params:<12}")

    # Find best architecture
    best_arch = min(all_results.keys(), key=lambda k: all_results[k]["final_exploitability"])
    print(f"\nBest architecture: {best_arch} ({ARCHITECTURES[best_arch]['description']})")
    print(f"Best exploitability: {all_results[best_arch]['final_exploitability']:.4f}")

    print(f"\nResults saved to: results/focused_architecture_comparison.json")
    return all_results


if __name__ == "__main__":
    results = run_architecture_comparison()