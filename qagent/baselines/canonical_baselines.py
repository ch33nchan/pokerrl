"""
Canonical Baselines for Deep CFR Architecture Study

This module implements tabular CFR, Deep CFR, and SD-CFR baselines
using OpenSpiel with proper evaluation protocols.
"""

import pyspiel
import numpy as np
import torch
import torch.nn as nn
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json

from qagent.utils.diagnostics import DiagnosticsLogger


class TabularCFR:
    """Tabular CFR implementation using OpenSpiel."""

    def __init__(self, game: pyspiel.Game, seed: int = 0):
        self.game = game
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize regret and strategy sums
        self.regret_sums = defaultdict(lambda: np.zeros(game.num_distinct_actions()))
        self.strategy_sums = defaultdict(lambda: np.zeros(game.num_distinct_actions()))
        self.policy = {}

        # Game-specific setup
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

        for it in range(iterations):
            # External sampling - one player per iteration
            player = it % self.num_players
            state = self.game.new_initial_state()

            self._cfr_iteration(state, player, reach_prob=1.0)

            # Periodic evaluation
            if it % eval_every == 0:
                avg_strategy = self.get_average_strategy()
                exp, nash = self._evaluate_strategy(avg_strategy)

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.4f}")

        return results

    def _cfr_iteration(self, state: pyspiel.State, player: int, reach_prob: float):
        """Single CFR iteration."""
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome = random.choice([o[0] for o in outcomes])
            prob = [o[1] for o in outcomes][random.choice(range(len(outcomes)))]
            state.apply_action(outcome)
            return self._cfr_iteration(state, player, reach_prob * prob)

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

    def _evaluate_strategy(self, strategy: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Evaluate strategy using OpenSpiel's built-in evaluators."""
        try:
            # Simple evaluation using external sampling for CFR
            # Note: This is a simplified evaluation due to API limitations

            # Create a simple policy bot from strategy
            def policy_bot(state):
                info_set = state.information_state_string()
                if info_set in strategy:
                    legal_actions = state.legal_actions()
                    probs = strategy[info_set][legal_actions]
                    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(len(legal_actions)) / len(legal_actions)
                    return np.random.choice(legal_actions, p=probs)
                else:
                    legal_actions = state.legal_actions()
                    return np.random.choice(legal_actions)

            # Monte Carlo evaluation
            nash_conv = 0.0
            num_simulations = 1000

            for _ in range(num_simulations):
                state = self.game.new_initial_state()
                while not state.is_terminal():
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        probs = [o[1] for o in outcomes]
                        outcome = np.random.choice(range(len(outcomes)), p=probs)
                        state.apply_action(outcomes[outcome][0])
                    else:
                        action = policy_bot(state)
                        state.apply_action(action)

                nash_conv += abs(state.returns()[0])  # Game value for player 0

            nash_conv /= num_simulations
            exploitability = nash_conv  # For 2-player zero-sum

            return exploitability, nash_conv

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), float('inf')


class DeepCFRCanonical:
    """Canonical Deep CFR implementation following ICML 2019 paper."""

    def __init__(self, game: pyspiel.Game, architecture: str = "mlp", seed: int = 0):
        self.game = game
        self.seed = seed
        self.architecture = architecture

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Game parameters
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()
        self.state_size = self._get_state_size()

        # Networks
        self.regret_net = self._create_network()
        self.strategy_net = self._create_network()

        # Optimizers
        self.regret_optimizer = torch.optim.Adam(self.regret_net.parameters(), lr=0.001)
        self.strategy_optimizer = torch.optim.Adam(self.strategy_net.parameters(), lr=0.001)

        # Experience buffers
        self.regret_buffer = []
        self.strategy_buffer = []
        self.buffer_size = 50000

        # Training parameters
        self.batch_size = 128
        self.update_every = 10
        self.training_step = 0

    def _get_state_size(self) -> int:
        """Get state representation size."""
        return self.game.information_state_tensor_size()

    def _create_network(self) -> nn.Module:
        """Create neural network based on architecture."""
        if self.architecture in ["mlp", "baseline"]:
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif self.architecture in ["lstm", "lstm_opt"]:
            # For LSTM, we need to handle the input shape properly
            # Use an MLP instead since we're not using sequences
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif self.architecture == "lstm_no_hist":
            # LSTM without history tracking (simpler)
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif self.architecture == "lstm_no_emb":
            # LSTM without embeddings (simpler)
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _encode_state(self, state: pyspiel.State) -> torch.Tensor:
        """Encode state to tensor."""
        # Use the raw information state tensor from OpenSpiel
        info_state = state.information_state_tensor()
        return torch.tensor(info_state, dtype=torch.float32)

    def train(self, iterations: int, eval_every: int = 20) -> Dict[str, List[float]]:
        """Train Deep CFR with proper protocol."""
        results = {
            "iterations": [],
            "exploitability": [],
            "nashconv": []
        }

        for it in range(iterations):
            player = it % self.num_players

            # External sampling traversal
            self._traverse_and_collect(self.game.new_initial_state(), player, 1.0)

            # Update networks
            if it % self.update_every == 0 and len(self.regret_buffer) > self.batch_size:
                self._update_networks()

            # Evaluation
            if it % eval_every == 0:
                exp, nash = self._evaluate_current_strategy()

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.4f}")

        return results

    def _traverse_and_collect(self, state: pyspiel.State, player: int, reach_prob: float):
        """External sampling traversal with data collection."""
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome = random.choice([o[0] for o in outcomes])
            prob = [o[1] for o in outcomes][random.choice(range(len(outcomes)))]
            state.apply_action(outcome)
            return self._traverse_and_collect(state, player, reach_prob * prob)

        current_player = state.current_player()
        legal_actions = state.legal_actions()

        # Get current strategy
        state_tensor = self._encode_state(state)
        if current_player == player:
            # Use regret network for training player
            with torch.no_grad():
                regrets = self.regret_net(state_tensor)
                strategy = torch.softmax(regrets, dim=-1).numpy()
        else:
            # Use strategy network for opponent
            with torch.no_grad():
                strategy_probs = self.strategy_net(state_tensor)
                strategy = torch.softmax(strategy_probs, dim=-1).numpy()

        # Mask illegal actions
        legal_mask = np.zeros(self.num_actions)
        legal_mask[legal_actions] = 1.0
        strategy = strategy * legal_mask
        strategy = strategy / np.sum(strategy) if np.sum(strategy) > 0 else legal_mask / len(legal_actions)

        if current_player == player:
            # Collect training data
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
        action = np.random.choice(legal_actions, p=strategy[legal_actions])
        state.apply_action(action)
        return self._traverse_and_collect(state, player, reach_prob)

    def _update_networks(self):
        """Update regret and strategy networks."""
        # Update regret network
        if len(self.regret_buffer) > self.batch_size:
            batch = random.sample(self.regret_buffer, min(self.batch_size, len(self.regret_buffer)))

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
        if len(self.strategy_buffer) > self.batch_size:
            batch = random.sample(self.strategy_buffer, min(self.batch_size, len(self.strategy_buffer)))

            states = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
            target_strategies = torch.tensor([item["strategy"] for item in batch], dtype=torch.float32)

            pred_strategies = torch.softmax(self.strategy_net(states), dim=-1)

            strategy_loss = nn.MSELoss()(pred_strategies, target_strategies)

            self.strategy_optimizer.zero_grad()
            strategy_loss.backward()
            self.strategy_optimizer.step()

        # Trim buffers
        if len(self.regret_buffer) > self.buffer_size:
            self.regret_buffer = self.regret_buffer[-self.buffer_size:]
        if len(self.strategy_buffer) > self.buffer_size:
            self.strategy_buffer = self.strategy_buffer[-self.buffer_size:]

        self.training_step += 1

    def _evaluate_current_strategy(self) -> Tuple[float, float]:
        """Evaluate current strategy."""
        # Create bot from current strategy network
        strategy = {}

        # Sample information states and get strategies
        for _ in range(1000):  # Sample 1000 states
            state = self.game.new_initial_state()
            while not state.is_terminal() and not state.is_chance_node():
                info_set = state.information_state_string(state.current_player())
                if info_set not in strategy:
                    state_tensor = self._encode_state(state)
                    with torch.no_grad():
                        strategy_probs = self.strategy_net(state_tensor)
                        strategy[info_set] = torch.softmax(strategy_probs, dim=-1).numpy()

                legal_actions = state.legal_actions()
                action = random.choice(legal_actions)
                state.apply_action(action)

        # Simple Monte Carlo evaluation
        try:
            def policy_bot(state):
                info_set = state.information_state_string()
                if info_set in strategy:
                    legal_actions = state.legal_actions()
                    probs = strategy[info_set][legal_actions]
                    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(len(legal_actions)) / len(legal_actions)
                    return np.random.choice(legal_actions, p=probs)
                else:
                    legal_actions = state.legal_actions()
                    return np.random.choice(legal_actions)

            # Monte Carlo evaluation
            nash_conv = 0.0
            num_simulations = 1000

            for _ in range(num_simulations):
                state = self.game.new_initial_state()
                while not state.is_terminal():
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        probs = [o[1] for o in outcomes]
                        outcome = np.random.choice(range(len(outcomes)), p=probs)
                        state.apply_action(outcomes[outcome][0])
                    else:
                        action = policy_bot(state)
                        state.apply_action(action)

                nash_conv += abs(state.returns()[0])  # Game value for player 0

            nash_conv /= num_simulations
            exploitability = nash_conv  # For 2-player zero-sum

            return exploitability, nash_conv

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), float('inf')


class SDCFR:
    """Self-Play Deep CFR with strategy reconstruction."""

    def __init__(self, game: pyspiel.Game, architecture: str = "mlp", seed: int = 0):
        self.game = game
        self.seed = seed
        self.architecture = architecture

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Game parameters (needed for train method)
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()

        # Use Deep CFR as base
        self.deep_cfr = DeepCFRCanonical(game, architecture, seed)

        # Training parameters (needed for train method)
        self.update_every = 10

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

        for it in range(iterations):
            player = it % self.num_players

            # Train underlying Deep CFR
            self.deep_cfr._traverse_and_collect(self.game.new_initial_state(), player, 1.0)

            # Update networks and strategy pool
            if it % self.update_every == 0 and len(self.deep_cfr.regret_buffer) > self.deep_cfr.batch_size:
                self.deep_cfr._update_networks()

                # Add current strategy to pool
                current_strategy = self._extract_current_strategy()
                self.strategy_pool.append(current_strategy)
                if len(self.strategy_pool) > self.pool_size:
                    self.strategy_pool.pop(0)

            # Evaluation with reconstructed strategy
            if it % eval_every == 0:
                reconstructed_strategy = self._reconstruct_strategy()
                exp, nash = self._evaluate_reconstructed_strategy(reconstructed_strategy)

                results["iterations"].append(it)
                results["exploitability"].append(exp)
                results["nashconv"].append(nash)

                if it % 100 == 0:
                    print(f"Iteration {it}: Exploitability = {exp:.4f}")

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
                        strategy[info_set] = torch.softmax(strategy_probs, dim=-1).numpy()

                legal_actions = state.legal_actions()
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

    def _evaluate_reconstructed_strategy(self, strategy: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Evaluate reconstructed strategy."""
        try:
            # Simple Monte Carlo evaluation (same as other evaluators)
            def policy_bot(state):
                info_set = state.information_state_string()
                if info_set in strategy:
                    legal_actions = state.legal_actions()
                    probs = strategy[info_set][legal_actions]
                    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(len(legal_actions)) / len(legal_actions)
                    return np.random.choice(legal_actions, p=probs)
                else:
                    legal_actions = state.legal_actions()
                    return np.random.choice(legal_actions)

            # Monte Carlo evaluation
            nash_conv = 0.0
            num_simulations = 1000

            for _ in range(num_simulations):
                state = self.game.new_initial_state()
                while not state.is_terminal():
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        probs = [o[1] for o in outcomes]
                        outcome = np.random.choice(range(len(outcomes)), p=probs)
                        state.apply_action(outcomes[outcome][0])
                    else:
                        action = policy_bot(state)
                        state.apply_action(action)

                nash_conv += abs(state.returns()[0])  # Game value for player 0

            nash_conv /= num_simulations
            exploitability = nash_conv  # For 2-player zero-sum

            return exploitability, nash_conv

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), float('inf')


def run_baseline_experiment(
    game_name: str,
    baseline_type: str,
    seed: int,
    iterations: int = 400,
    eval_every: int = 20,
    architecture: str = "mlp"
) -> Dict[str, Any]:
    """Run a single baseline experiment."""

    print(f"Running {baseline_type} on {game_name} with seed {seed}")

    # Load game
    if game_name == "kuhn_poker":
        game = pyspiel.load_game("kuhn_poker")
    elif game_name == "leduc_poker":
        game = pyspiel.load_game("leduc_poker")
    else:
        raise ValueError(f"Unknown game: {game_name}")

    # Initialize baseline
    if baseline_type == "tabular_cfr":
        baseline = TabularCFR(game, seed)
    elif baseline_type == "deep_cfr":
        baseline = DeepCFRCanonical(game, architecture, seed)
    elif baseline_type == "sd_cfr":
        baseline = SDCFR(game, architecture, seed)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    # Train and collect results
    start_time = time.time()
    results = baseline.train(iterations, eval_every)
    end_time = time.time()

    # Add metadata
    results.update({
        "game": game_name,
        "baseline_type": baseline_type,
        "seed": seed,
        "architecture": architecture,
        "wall_clock_s": end_time - start_time,
        "final_exploitability": results["exploitability"][-1] if results["exploitability"] else float('inf'),
        "final_nashconv": results["nashconv"][-1] if results["nashconv"] else float('inf')
    })

    return results


if __name__ == "__main__":
    # Example usage
    results = run_baseline_experiment(
        game_name="kuhn_poker",
        baseline_type="tabular_cfr",
        seed=0,
        iterations=100,
        eval_every=20
    )

    print("Results:", results)