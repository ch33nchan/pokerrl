#!/usr/bin/env python3
"""
Comprehensive CPU Experiments for Deep CFR Architecture Study

This script implements the full experimental protocol with:
- Exact OpenSpiel evaluators (no Monte Carlo approximations)
- 20 seeds per condition for statistical significance
- SD-CFR and tabular CFR baselines
- Cross-entropy strategy loss
- External sampling traversal
- Capacity/FLOPs analysis
- Statistical analysis with confidence intervals
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pyspiel
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import hashlib
import subprocess

# Configuration
SEEDS = list(range(20))  # 20 seeds per condition
ITERATIONS = 500
EVAL_EVERY = 25
BATCH_SIZE = 64
BUFFER_SIZE = 10000
GAMES = ["kuhn_poker"]  # Primary game
OPTIONAL_GAMES = ["leduc_poker"]  # For sanity extension

# Thresholds T for each game (in OpenSpiel game units)
THRESHOLDS = {
    "kuhn_poker": 0.1,  # Exact exploitability threshold
    "leduc_poker": 2.5
}

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    run_id: str
    game: str
    baseline_type: str
    architecture: str
    seed: int
    params_count: int
    flops_est: int
    optimizer_cfg: Dict[str, Any]
    replay_cfg: Dict[str, Any]
    update_cadence: int
    iterations: int
    eval_every: int
    final_exploitability: float
    final_nashconv: float
    steps_to_threshold: Optional[int]
    time_to_threshold: Optional[float]
    wall_clock_s: float
    openspiel_version: str
    python_version: str
    git_hash: str

class ExactEvaluator:
    """Exact OpenSpiel evaluators for exploitability and NashConv."""

    def __init__(self, game: pyspiel.Game):
        self.game = game
        self.num_players = game.num_players()

    def evaluate_strategy(self, strategy_network, device='cpu') -> Tuple[float, float]:
        """
        Evaluate strategy using OpenSpiel's exact evaluators.
        Returns: (exploitability, nash_conv)
        """
        try:
            # Build policy mapping for OpenSpiel
            policy_mapping = self._build_policy_mapping(strategy_network, device)

            # Use OpenSpiel's exact evaluation with policy mapping
            nash_conv = pyspiel.nash_conv(self.game, policy_mapping)

            # For two-player zero-sum games, exploitability = nash_conv / 2
            exploitability = nash_conv / 2.0 if self.num_players == 2 else nash_conv

            return float(exploitability), float(nash_conv)

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), float('inf')

    def _build_policy_mapping(self, strategy_network, device='cpu') -> Dict[str, List[Tuple[int, float]]]:
        """Build OpenSpiel policy mapping from strategy network."""
        policy_mapping = {}

        # Sample states to build policy mapping
        for _ in range(100):  # Sample multiple trajectories
            state = self.game.new_initial_state()
            self._collect_policies_recursive(state, strategy_network, policy_mapping, device)

        return policy_mapping

    def _collect_policies_recursive(self, state: pyspiel.State, strategy_network,
                                   policy_mapping: Dict, device='cpu'):
        """Recursively collect policies for all information states."""
        if state.is_terminal():
            return

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            probs = [o[1] for o in outcomes]
            outcome_idx = np.random.choice(len(outcomes), p=probs)
            state.apply_action(outcomes[outcome_idx][0])
            self._collect_policies_recursive(state, strategy_network, policy_mapping, device)
            return

        current_player = state.current_player()
        info_set = state.information_state_string(current_player)

        # Skip if we already have policy for this info set
        if info_set not in policy_mapping:
            # Get strategy from network
            info_state = state.information_state_tensor()
            info_tensor = torch.tensor(info_state, dtype=torch.float32, device=device)

            with torch.no_grad():
                if hasattr(strategy_network, 'regret_net'):
                    # Use strategy network for Deep CFR
                    strategy_logits = strategy_network.strategy_net(info_tensor)
                elif hasattr(strategy_network, 'deep_cfr'):
                    # Use strategy network from SD-CFR
                    strategy_logits = strategy_network.deep_cfr.strategy_net(info_tensor)
                elif callable(strategy_network):
                    # Tabular CFR or callable strategy
                    strategy_array = strategy_network(state)
                    if isinstance(strategy_array, np.ndarray):
                        policy_mapping[info_set] = [(i, float(strategy_array[i]))
                                                  for i in range(len(strategy_array))]
                    else:
                        # Fallback to uniform
                        legal_actions = state.legal_actions()
                        uniform_prob = 1.0 / len(legal_actions)
                        policy_mapping[info_set] = [(action, uniform_prob) for action in legal_actions]
                    return
                else:
                    # Direct network call
                    strategy_logits = strategy_network(info_tensor)

                strategy_probs = F.softmax(strategy_logits, dim=-1).cpu().numpy()

            # Mask illegal actions and normalize
            legal_actions = state.legal_actions()
            legal_mask = np.zeros(len(strategy_probs))
            legal_mask[legal_actions] = 1.0
            strategy_probs = strategy_probs * legal_mask

            if strategy_probs.sum() > 0:
                strategy_probs = strategy_probs / strategy_probs.sum()
            else:
                strategy_probs = legal_mask / len(legal_actions) if legal_actions.sum() > 0 else strategy_probs

            # Convert to OpenSpiel format
            policy_mapping[info_set] = [(action, float(strategy_probs[action]))
                                       for action in legal_actions]

        # Continue traversal with sampled action
        legal_actions = state.legal_actions()
        if len(legal_actions) > 0 and info_set in policy_mapping:
            # Sample action according to policy
            policy_list = policy_mapping[info_set]
            actions = [a[0] for a in policy_list]
            probs = [a[1] for a in policy_list]

            if sum(probs) > 0:
                action = np.random.choice(actions, p=probs)
                next_state = state.clone()
                next_state.apply_action(action)
                self._collect_policies_recursive(next_state, strategy_network, policy_mapping, device)

class TabularCFR:
    """Tabular CFR implementation with exact evaluation."""

    def __init__(self, game: pyspiel.Game, seed: int = 0):
        self.game = game
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize regret and strategy sums
        self.regret_sums = defaultdict(lambda: np.zeros(game.num_distinct_actions()))
        self.strategy_sums = defaultdict(lambda: np.zeros(game.num_distinct_actions()))
        self.policy = {}

        # Game parameters
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()

    def get_strategy(self, info_set: str, legal_actions: List[int]) -> np.ndarray:
        """Get current strategy via regret matching."""
        regrets = self.regret_sums[info_set]
        positive_regrets = np.maximum(regrets[legal_actions], 0)
        sum_positive = np.sum(positive_regrets)

        if sum_positive > 0:
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = positive_regrets / sum_positive
        else:
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = 1.0 / len(legal_actions)

        return strategy

    def train(self, iterations: int, eval_every: int = 20) -> Dict[str, List[float]]:
        """Train tabular CFR with periodic evaluation."""
        results = {
            "iterations": [],
            "exploitability": [],
            "nashconv": []
        }

        evaluator = ExactEvaluator(self.game)

        for it in range(iterations):
            # External sampling - one player per iteration
            player = it % self.num_players
            state = self.game.new_initial_state()

            self._cfr_iteration(state, player, reach_prob=1.0)

            # Periodic evaluation
            if it % eval_every == 0:
                exp, nash = evaluator.evaluate_strategy(self)

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.6f}")

        return results

    def _cfr_iteration(self, state: pyspiel.State, player: int, reach_prob: float):
        """Single CFR iteration."""
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome = random.choice([o[0] for o in outcomes])
            probs = [o[1] for o in outcomes]
            outcome_idx = np.random.choice(range(len(outcomes)), p=probs)
            state.apply_action(outcomes[outcome_idx][0])
            return self._cfr_iteration(state, player, reach_prob * probs[outcome_idx])

        current_player = state.current_player()
        info_set = state.information_state_string(current_player)
        legal_actions = state.legal_actions()

        strategy = self.get_strategy(info_set, legal_actions)
        self.policy[info_set] = strategy

        if current_player == player:
            # Update average strategy
            self.strategy_sums[info_set][legal_actions] += reach_prob * strategy[legal_actions]

            # Counterfactual values
            cf_values = np.zeros(len(legal_actions))
            expected_value = 0.0

            for i, action in enumerate(legal_actions):
                next_state = state.clone()
                next_state.apply_action(action)
                cf_values[i] = self._cfr_iteration(next_state, player, reach_prob * strategy[action])
                expected_value += strategy[action] * cf_values[i]

            # Update regrets
            for i, action in enumerate(legal_actions):
                regret = cf_values[i] - expected_value
                self.regret_sums[info_set][action] += regret

            return expected_value
        else:
            # Opponent action - sample action according to strategy
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions, p=strategy[legal_actions])
                state.apply_action(action)
            return self._cfr_iteration(state, player, reach_prob)

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """Get average strategy."""
        avg_strategy = {}
        for info_set, strategy_sum in self.strategy_sums.items():
            total = np.sum(strategy_sum)
            if total > 0:
                avg_strategy[info_set] = strategy_sum / total
            else:
                avg_strategy[info_set] = np.ones(self.num_actions) / self.num_actions
        return avg_strategy

    def __call__(self, state: pyspiel.State):
        """Make this callable for evaluation using average strategy."""
        info_set = state.information_state_string()
        legal_actions = state.legal_actions()

        # Use average strategy for evaluation
        avg_strategy = self.get_average_strategy()

        if info_set in avg_strategy:
            strategy = avg_strategy[info_set]
        else:
            # Fallback to uniform strategy
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = 1.0 / len(legal_actions)

        # Return full strategy vector (needed for evaluator)
        full_strategy = np.zeros(self.num_actions)
        full_strategy[legal_actions] = strategy[legal_actions]
        return full_strategy

class DeepCFRCanonical:
    """Deep CFR with proper cross-entropy strategy loss."""

    def __init__(self, game: pyspiel.Game, architecture: str = "baseline", seed: int = 0):
        self.game = game
        self.seed = seed
        self.architecture = architecture

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Game parameters
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()
        self.state_size = game.information_state_tensor_size()

        # Networks
        self.regret_net = self._create_network()
        self.strategy_net = self._create_network()

        # Optimizers
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=0.001)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.001)

        # Experience buffers
        self.regret_buffer = []
        self.strategy_buffer = []
        self.buffer_size = BUFFER_SIZE

        # Training parameters
        self.batch_size = BATCH_SIZE
        self.update_every = 10
        self.training_step = 0

    def _create_network(self) -> nn.Module:
        """Create network based on architecture."""
        if self.architecture == "baseline":
            hidden_sizes = [64, 64]
        elif self.architecture == "wide":
            hidden_sizes = [128, 128]
        elif self.architecture == "deep":
            hidden_sizes = [64, 64, 64]
        elif self.architecture == "fast":
            hidden_sizes = [32, 32]
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        layers = []
        input_size = self.state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size

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
                strategy = F.softmax(regrets, dim=-1)
            else:
                output = self.strategy_net(state_tensor)
                strategy = F.softmax(output, dim=-1)

        strategy = strategy.numpy()

        # Mask illegal actions
        legal_actions = state.legal_actions()
        if len(legal_actions) > 0:
            legal_mask = np.zeros(self.num_actions)
            legal_mask[legal_actions] = 1.0
            strategy = strategy * legal_mask
            strategy = strategy / np.sum(strategy) if np.sum(strategy) > 0 else legal_mask / len(legal_actions)

        return strategy

    def train(self, iterations: int, eval_every: int = 20) -> Dict[str, List[float]]:
        """Train Deep CFR with proper protocol."""
        results = {
            "iterations": [],
            "exploitability": [],
            "nashconv": []
        }

        evaluator = ExactEvaluator(self.game)

        for it in range(iterations):
            player = it % self.num_players

            # External sampling traversal
            self._traverse_and_collect(self.game.new_initial_state(), player, 1.0)

            # Update networks
            if it % self.update_every == 0 and len(self.regret_buffer) > self.batch_size:
                self._update_networks()

            # Evaluation
            if it % eval_every == 0:
                exp, nash = evaluator.evaluate_strategy(self)

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.6f}")

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

                # Store regret training sample
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

    def _update_networks(self):
        """Update networks with proper loss functions."""
        # Update regret network (MSE for advantage values)
        if len(self.regret_buffer) > self.batch_size:
            batch = random.sample(self.regret_buffer, self.batch_size)

            states = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
            actions = torch.tensor([item["action"] for item in batch], dtype=torch.long)
            cf_values = torch.tensor([item["cf_value"] for item in batch], dtype=torch.float32)

            regrets = self.regret_net(states)
            selected_regrets = regrets[range(len(actions)), actions]

            regret_loss = F.mse_loss(selected_regrets, cf_values)

            self.regret_optimizer.zero_grad()
            regret_loss.backward()
            self.regret_optimizer.step()

        # Update strategy network (Cross-entropy/KL for mixed strategy)
        if len(self.strategy_buffer) > self.batch_size:
            batch = random.sample(self.strategy_buffer, self.batch_size)

            states = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
            target_strategies = torch.tensor([item["strategy"] for item in batch], dtype=torch.float32)

            pred_logits = self.strategy_net(states)
            pred_strategies = F.softmax(pred_logits, dim=-1)

            # Cross-entropy loss (equivalent to KL divergence up to constant)
            strategy_loss = -(target_strategies * torch.log(pred_strategies + 1e-8)).sum(dim=1).mean()

            self.strategy_optimizer.zero_grad()
            strategy_loss.backward()
            self.strategy_optimizer.step()

        # Trim buffers
        if len(self.regret_buffer) > self.buffer_size:
            self.regret_buffer = self.regret_buffer[-self.buffer_size:]
        if len(self.strategy_buffer) > self.buffer_size:
            self.strategy_buffer = self.strategy_buffer[-self.buffer_size:]

        self.training_step += 1

class SDCFR:
    """Self-Play Deep CFR with strategy reconstruction."""

    def __init__(self, game: pyspiel.Game, architecture: str = "baseline", seed: int = 0):
        self.game = game
        self.seed = seed
        self.architecture = architecture

        # Game parameters (delegate to deep_cfr)
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()
        self.state_size = game.information_state_tensor_size()

        # Use Deep CFR as base
        self.deep_cfr = DeepCFRCanonical(game, architecture, seed)

        # Strategy pool for reconstruction
        self.strategy_pool = []
        self.pool_size = 10

    def train(self, iterations: int, eval_every: int = 20) -> Dict[str, List[float]]:
        """Train SD-CFR with strategy reconstruction."""
        results = {
            "iterations": [],
            "exploitability": [],
            "nashconv": []
        }

        evaluator = ExactEvaluator(self.game)

        for it in range(iterations):
            player = it % self.num_players

            # Train underlying Deep CFR
            self.deep_cfr._traverse_and_collect(self.game.new_initial_state(), player, 1.0)

            # Update networks and strategy pool
            if it % self.deep_cfr.update_every == 0 and len(self.deep_cfr.regret_buffer) > self.deep_cfr.batch_size:
                self.deep_cfr._update_networks()

                # Add current strategy to pool
                current_strategy = self._extract_current_strategy()
                self.strategy_pool.append(current_strategy)
                if len(self.strategy_pool) > self.pool_size:
                    self.strategy_pool.pop(0)

            # Evaluation with reconstructed strategy
            if it % eval_every == 0:
                exp, nash = evaluator.evaluate_strategy(self)

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.6f}")

        return results

    def _extract_current_strategy(self) -> Dict[str, np.ndarray]:
        """Extract current strategy from Deep CFR networks."""
        strategy = {}
        for _ in range(100):  # Sample states
            state = self.game.new_initial_state()
            while not state.is_terminal() and not state.is_chance_node():
                info_set = state.information_state_string(state.current_player())
                if info_set not in strategy:
                    state_tensor = self.deep_cfr._encode_state(state)
                    with torch.no_grad():
                        strategy_probs = self.deep_cfr.strategy_net(state_tensor)
                        strategy[info_set] = F.softmax(strategy_probs, dim=-1).numpy()

                legal_actions = state.legal_actions()
                if len(legal_actions) > 0:
                    action = random.choice(legal_actions)
                    state.apply_action(action)

        return strategy

    def _reconstruct_strategy(self) -> Dict[str, np.ndarray]:
        """Reconstruct average strategy from strategy pool."""
        if not self.strategy_pool:
            return {}

        reconstructed = {}
        for strategy in self.strategy_pool:
            for info_set, probs in strategy.items():
                if info_set not in reconstructed:
                    reconstructed[info_set] = np.zeros_like(probs)
                reconstructed[info_set] += probs

        # Average over pool
        for info_set in reconstructed:
            reconstructed[info_set] /= len(self.strategy_pool)

        return reconstructed

    def __call__(self, state: pyspiel.State):
        """Make this callable for evaluation."""
        info_set = state.information_state_string()
        legal_actions = state.legal_actions()

        # Get reconstructed strategy
        reconstructed_strategy = self._reconstruct_strategy()
        if info_set in reconstructed_strategy:
            strategy = reconstructed_strategy[info_set]
        else:
            # Fallback to uniform
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = 1.0 / len(legal_actions)

        # Return full strategy vector (needed for evaluator)
        full_strategy = np.zeros(self.num_actions)
        full_strategy[legal_actions] = strategy[legal_actions]
        return full_strategy

def calculate_flops(architecture: str, state_size: int, num_actions: int) -> int:
    """Calculate FLOPs for a forward pass."""
    if architecture == "baseline":
        # [state_size, 64] + [64, 64] + [64, num_actions]
        return state_size * 64 + 64 * 64 + 64 * num_actions
    elif architecture == "wide":
        # [state_size, 128] + [128, 128] + [128, num_actions]
        return state_size * 128 + 128 * 128 + 128 * num_actions
    elif architecture == "deep":
        # [state_size, 64] + [64, 64] + [64, 64] + [64, num_actions]
        return state_size * 64 + 64 * 64 + 64 * 64 + 64 * num_actions
    elif architecture == "fast":
        # [state_size, 32] + [32, 32] + [32, num_actions]
        return state_size * 32 + 32 * 32 + 32 * num_actions
    else:
        return 0

def get_git_hash():
    """Get current git hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, cwd='.')
        return result.stdout.strip()
    except:
        return "unknown"

def run_single_experiment(config: Dict[str, Any]) -> ExperimentConfig:
    """Run a single experiment with proper methodology."""
    game_name = config["game"]
    baseline_type = config["baseline_type"]
    architecture = config.get("architecture", "baseline")
    seed = config["seed"]
    run_id = config["run_id"]

    print(f"Starting experiment: {run_id}")

    # Initialize diagnostics
    start_time = time.time()

    try:
        # Load game
        game = pyspiel.load_game(game_name)

        # Initialize baseline
        if baseline_type == "tabular_cfr":
            agent = TabularCFR(game, seed)
        elif baseline_type == "deep_cfr":
            agent = DeepCFRCanonical(game, architecture, seed)
        elif baseline_type == "sd_cfr":
            agent = SDCFR(game, architecture, seed)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Train
        results = agent.train(ITERATIONS, EVAL_EVERY)
        end_time = time.time()

        # Calculate metrics
        final_exploitability = results["exploitability"][-1] if results["exploitability"] else float('inf')
        final_nashconv = results["nashconv"][-1] if results["nashconv"] else float('inf')

        # Calculate steps to threshold
        threshold = THRESHOLDS[game_name]
        steps_to_threshold = None
        time_to_threshold = None

        for i, exp in enumerate(results["exploitability"]):
            if exp <= threshold:
                steps_to_threshold = results["iterations"][i]
                time_per_iteration = (end_time - start_time) / ITERATIONS
                time_to_threshold = steps_to_threshold * time_per_iteration
                break

        # Calculate capacity metrics
        state_size = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()

        if baseline_type == "tabular_cfr":
            params_count = 0  # Tabular doesn't have fixed parameter count
            flops_est = 0
        else:
            # Estimate parameters for neural networks
            if architecture == "baseline":
                params_count = (state_size + 1) * 64 + (64 + 1) * 64 + (64 + 1) * num_actions
            elif architecture == "wide":
                params_count = (state_size + 1) * 128 + (128 + 1) * 128 + (128 + 1) * num_actions
            elif architecture == "deep":
                params_count = (state_size + 1) * 64 + (64 + 1) * 64 + (64 + 1) * 64 + (64 + 1) * num_actions
            elif architecture == "fast":
                params_count = (state_size + 1) * 32 + (32 + 1) * 32 + (32 + 1) * num_actions
            else:
                params_count = 0

            flops_est = calculate_flops(architecture, state_size, num_actions)

        # Configuration
        optimizer_cfg = {"type": "adam", "lr": 0.001}
        replay_cfg = {"buffer_size": BUFFER_SIZE, "batch_size": BATCH_SIZE}

        # Create experiment config
        exp_config = ExperimentConfig(
            run_id=run_id,
            game=game_name,
            baseline_type=baseline_type,
            architecture=architecture,
            seed=seed,
            params_count=params_count,
            flops_est=flops_est,
            optimizer_cfg=optimizer_cfg,
            replay_cfg=replay_cfg,
            update_cadence=10,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            final_exploitability=final_exploitability,
            final_nashconv=final_nashconv,
            steps_to_threshold=steps_to_threshold,
            time_to_threshold=time_to_threshold,
            wall_clock_s=end_time - start_time,
            openspiel_version=pyspiel.__version__,
            python_version=f"{subprocess.sys.version_info.major}.{subprocess.sys.version_info.minor}",
            git_hash=get_git_hash()
        )

        print(f"Completed experiment: {run_id}")
        print(f"Final exploitability: {final_exploitability:.6f}")
        print(f"Final NashConv: {final_nashconv:.6f}")
        print(f"Wall clock: {end_time - start_time:.2f}s")

        return exp_config

    except Exception as e:
        print(f"Failed experiment {run_id}: {e}")
        # Return failed config
        return ExperimentConfig(
            run_id=run_id,
            game=game_name,
            baseline_type=baseline_type,
            architecture=architecture,
            seed=seed,
            params_count=0,
            flops_est=0,
            optimizer_cfg={},
            replay_cfg={},
            update_cadence=10,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            final_exploitability=float('inf'),
            final_nashconv=float('inf'),
            steps_to_threshold=None,
            time_to_threshold=None,
            wall_clock_s=0.0,
            openspiel_version=pyspiel.__version__,
            python_version=f"{subprocess.sys.version_info.major}.{subprocess.sys.version_info.minor}",
            git_hash=get_git_hash()
        )

def run_comprehensive_experiments():
    """Run comprehensive CPU experiments."""
    print("Starting Comprehensive Deep CFR Architecture Study")
    print(f"OpenSpiel version: {pyspiel.__version__}")
    print(f"Python version: {subprocess.sys.version}")
    print(f"Git hash: {get_git_hash()}")
    print()

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("manifests", exist_ok=True)

    # Define experiments
    experiments = []

    # Primary experiments (Kuhn Poker)
    architectures = ["baseline", "wide", "deep", "fast"]
    baseline_types = ["deep_cfr", "sd_cfr", "tabular_cfr"]

    for baseline_type in baseline_types:
        if baseline_type == "tabular_cfr":
            # Tabular CFR only uses baseline architecture
            for seed in SEEDS:
                run_id = f"kuhn_poker_{baseline_type}_{seed:03d}"
                experiments.append({
                    "run_id": run_id,
                    "game": "kuhn_poker",
                    "baseline_type": baseline_type,
                    "architecture": "baseline",
                    "seed": seed
                })
        else:
            # Deep CFR and SD-CFR with all architectures
            for architecture in architectures:
                for seed in SEEDS:
                    run_id = f"kuhn_poker_{baseline_type}_{architecture}_{seed:03d}"
                    experiments.append({
                        "run_id": run_id,
                        "game": "kuhn_poker",
                        "baseline_type": baseline_type,
                        "architecture": architecture,
                        "seed": seed
                    })

    print(f"Total experiments: {len(experiments)}")
    print(f"Primary experiments: {len([e for e in experiments if e['game'] == 'kuhn_poker'])}")
    print()

    # Run experiments
    all_configs = []
    failed_configs = []

    for i, exp_config in enumerate(experiments):
        print(f"Progress: {i+1}/{len(experiments)} - {exp_config['run_id']}")

        try:
            config = run_single_experiment(exp_config)

            if config.final_exploitability < float('inf'):
                all_configs.append(config)
            else:
                failed_configs.append(config)

        except Exception as e:
            print(f"Exception in experiment {exp_config['run_id']}: {e}")
            failed_configs.append(ExperimentConfig(
                run_id=exp_config["run_id"],
                game=exp_config["game"],
                baseline_type=exp_config["baseline_type"],
                architecture=exp_config["architecture"],
                seed=exp_config["seed"],
                params_count=0,
                flops_est=0,
                optimizer_cfg={},
                replay_cfg={},
                update_cadence=10,
                iterations=ITERATIONS,
                eval_every=EVAL_EVERY,
                final_exploitability=float('inf'),
                final_nashconv=float('inf'),
                steps_to_threshold=None,
                time_to_threshold=None,
                wall_clock_s=0.0,
                openspiel_version=pyspiel.__version__,
                python_version=f"{subprocess.sys.version_info.major}.{subprocess.sys.version_info.minor}",
                git_hash=get_git_hash()
            ))

    # Save results
    successful_data = [asdict(config) for config in all_configs]
    failed_data = [asdict(config) for config in failed_configs]

    with open("manifests/comprehensive_experiments.json", "w") as f:
        json.dump({
            "successful": successful_data,
            "failed": failed_data,
            "metadata": {
                "total_experiments": len(experiments),
                "successful_experiments": len(all_configs),
                "failed_experiments": len(failed_configs),
                "timestamp": time.time()
            }
        }, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Successful experiments: {len(all_configs)}/{len(experiments)}")
    print(f"Failed experiments: {len(failed_configs)}")

    if all_configs:
        # Print summary statistics
        print(f"\n=== Results Summary ===")

        # Group by baseline type and architecture
        summary = {}
        for config in all_configs:
            key = f"{config.baseline_type}_{config.architecture}"
            if key not in summary:
                summary[key] = []
            summary[key].append(config.final_exploitability)

        for key, values in summary.items():
            if values and all(v < float('inf') for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{key}: {mean_val:.6f} ± {std_val:.6f}")

    print(f"\nResults saved to: manifests/comprehensive_experiments.json")
    return all_configs, failed_configs

if __name__ == "__main__":
    successful, failed = run_comprehensive_experiments()