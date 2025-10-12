"""Neural Fictitious Self-Play (NFSP) implementation for poker games.

NFSP combines reinforcement learning with supervised learning to approximate
Nash equilibrium strategies in imperfect information games.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

from algs.base import BaseAlgorithm, TrainingState, ExperienceBuffer
from utils.logging import get_experiment_logger
from eval.policy_adapter import PolicyMetadata


class NFSPNetwork(nn.Module):
    """Neural network for NFSP policy and value prediction."""

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_dims: List[int] = [128, 128],
        dropout: float = 0.1,
    ):
        """Initialize NFSP network.

        Args:
            input_size: Size of information state encoding
            num_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build shared layers
        layers = []
        prev_size = input_size

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_size, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Policy head (for RL)
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, num_actions), nn.Softmax(dim=-1)
        )

        # Q-value head (for RL)
        self.value_head = nn.Sequential(nn.Linear(prev_size, num_actions))

        # Supervised learning head (for SL policy)
        self.sl_head = nn.Sequential(
            nn.Linear(prev_size, num_actions), nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through NFSP network.

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Dictionary containing policy, value, and sl policy outputs
        """
        shared_features = self.shared_layers(x)

        policy = self.policy_head(shared_features)
        values = self.value_head(shared_features)
        sl_policy = self.sl_head(shared_features)

        return {"policy": policy, "values": values, "sl_policy": sl_policy}


class NFSPAlgorithm(BaseAlgorithm):
    """Neural Fictitious Self-Play algorithm implementation.

    NFSP maintains two learning systems:
    1. Reinforcement learning: learns best response to opponent strategies
    2. Supervised learning: learns average strategy over time
    """

    def __init__(self, game_wrapper, config: Dict[str, Any]):
        """Initialize NFSP algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Algorithm configuration
        """
        super().__init__(game_wrapper, config)

        self.experiment_logger = get_experiment_logger("nfsp")
        self.logger = self.experiment_logger.get_logger()

        # NFSP specific parameters
        self.eta = config.get("eta", 0.1)  # Anticipatory parameter
        self.alpha = config.get("alpha", 0.01)  # Learning rate for RL
        self.beta = config.get("beta", 0.01)  # Learning rate for SL
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.buffer_size = config.get("buffer_size", 20000)
        self.batch_size = config.get("batch_size", 256)
        self.policy_update_freq = config.get("policy_update_freq", 1)
        self.sl_update_freq = config.get("sl_update_freq", 1)

        # Experience buffers
        self.rl_buffer = deque(maxlen=self.buffer_size)
        self.sl_buffer = deque(maxlen=self.buffer_size)

        # Strategy tracking
        self.iteration = 0
        self.strategy_history = []
        self.current_strategy = {}

        # Initialize networks
        encoding_size = game_wrapper.get_encoding_size()
        num_actions = game_wrapper.num_actions
        hidden_dims = config.get(
            "hidden_dims", config.get("network", {}).get("hidden_dims", [128, 128])
        )

        self.network = NFSPNetwork(
            input_size=encoding_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=config.get("dropout", 0.1),
        )

        # Target network for stability
        self.target_network = NFSPNetwork(
            input_size=encoding_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=0.0,
        )

        # Initialize target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizers
        self.rl_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.alpha)
        self.sl_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.beta)

        # Training statistics
        self.total_steps = 0
        self.losses = {
            "rl_loss": deque(maxlen=100),
            "sl_loss": deque(maxlen=100),
            "total_loss": deque(maxlen=100),
        }

        self.logger.info("Initialized NFSP algorithm")
        self.logger.info(
            f"Network parameters: {sum(p.numel() for p in self.network.parameters())}"
        )

    def _create_network(self):
        """NFSP manages its own network architecture."""
        return None

    def train_step(self) -> Dict[str, float]:
        """Perform one NFSP training step."""
        state = self.train_iteration()
        return {
            "loss": state.loss,
            "gradient_norm": state.gradient_norm,
            "wall_time": state.wall_time,
        }

    def train_iteration(self) -> TrainingState:
        """Perform one NFSP training iteration."""
        start_time = time.time()

        # Step 1: Generate self-play experience
        trajectories = self._generate_self_play_trajectories()

        # Step 2: Store experiences in buffers
        self._store_experiences(trajectories)

        # Step 3: Update networks
        rl_loss = self._update_rl_network()
        sl_loss = self._update_sl_network()

        # Step 4: Update target network
        self._update_target_network()

        # Step 5: Update current strategy
        self._update_current_strategy()

        # Calculate statistics
        iteration_time = time.time() - start_time
        total_loss = rl_loss + sl_loss

        self.iteration += 1

        return TrainingState(
            iteration=self.iteration,
            loss=total_loss,
            buffer_size=len(self.rl_buffer) + len(self.sl_buffer),
            wall_time=iteration_time,
            gradient_norm=0.0,  # TODO: Compute actual gradient norm
            extra_metrics={
                "rl_loss": rl_loss,
                "sl_loss": sl_loss,
                "num_trajectories": len(trajectories),
                "eta": self.eta,
                "strategy_size": len(self.current_strategy),
            },
        )

    def _generate_self_play_trajectories(self) -> List[List[Dict[str, Any]]]:
        """Generate self-play trajectories using current strategy."""
        trajectories = []
        num_episodes = max(1, self.batch_size // 10)

        for _ in range(num_episodes):
            trajectory = self._generate_episode()
            trajectories.append(trajectory)

        return trajectories

    def _generate_episode(self) -> List[Dict[str, Any]]:
        """Generate a single episode trajectory."""
        trajectory = []
        state = self.game_wrapper.new_initial_state()

        while not state.is_terminal():
            current_player = state.current_player()
            legal_actions = state.legal_actions()

            # Get info state
            info_state = self.game_wrapper.encode_state(state, current_player)
            info_state_key = self.game_wrapper.get_info_state_key(state, current_player)

            # Choose action based on current policy
            action_probs = self._get_action_probabilities(
                info_state, legal_actions, info_state_key
            )
            action = np.random.choice(legal_actions, p=action_probs[legal_actions])

            # Store transition
            transition = {
                "info_state": info_state,
                "info_state_key": info_state_key,
                "legal_actions": legal_actions,
                "action": action,
                "action_probs": action_probs,
                "player": current_player,
                "iteration": self.iteration,
            }
            trajectory.append(transition)

            # Apply action
            state.apply_action(action)

        # Fill in rewards
        for transition in trajectory:
            player = transition["player"]
            transition["reward"] = state.returns()[player]

        return trajectory

    def _get_action_probabilities(
        self, info_state: np.ndarray, legal_actions: List[int], info_state_key: str
    ) -> np.ndarray:
        """Get action probabilities using current NFSP strategy."""
        # Decide whether to use RL or SL policy based on eta
        use_rl_policy = random.random() < self.eta

        with torch.no_grad():
            info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
            network_output = self.network(info_tensor)

            if use_rl_policy:
                # Use RL policy (best response)
                policy = network_output["policy"].squeeze().cpu().numpy()
            else:
                # Use SL policy (average strategy)
                policy = network_output["sl_policy"].squeeze().cpu().numpy()

        # Apply legal action mask
        legal_mask = np.zeros(len(policy))
        legal_mask[legal_actions] = 1.0
        masked_policy = policy * legal_mask

        # Normalize
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            # Fallback to uniform
            masked_policy[legal_actions] = 1.0 / len(legal_actions)

        return masked_policy

    def _store_experiences(self, trajectories: List[List[Dict[str, Any]]]):
        """Store experiences in RL and SL buffers."""
        for trajectory in trajectories:
            # Store in RL buffer (for learning best response)
            for transition in trajectory:
                self.rl_buffer.append(transition)

            # Store in SL buffer (for learning average strategy)
            if len(self.strategy_history) > 0:
                # Use strategy from previous iteration
                previous_strategy = self.strategy_history[-1]

                for transition in trajectory:
                    info_state_key = transition["info_state_key"]
                    if info_state_key in previous_strategy:
                        sl_transition = transition.copy()
                        sl_transition["target_policy"] = previous_strategy[
                            info_state_key
                        ]
                        self.sl_buffer.append(sl_transition)

    def _update_rl_network(self) -> float:
        """Update reinforcement learning network."""
        if len(self.rl_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.rl_buffer), self.batch_size)

        # Prepare training data
        info_states = torch.FloatTensor([t["info_state"] for t in batch])
        actions = torch.LongTensor([t["action"] for t in batch])
        rewards = torch.FloatTensor([t["reward"] for t in batch])

        # Get current Q-values
        self.rl_optimizer.zero_grad()
        network_output = self.network(info_states)
        current_q = network_output["values"].gather(1, actions.unsqueeze(1)).squeeze()

        # Get target Q-values (using target network)
        with torch.no_grad():
            target_output = self.target_network(info_states)
            target_q = target_output["values"].max(dim=1)[0]
            target_values = rewards + self.gamma * target_q

        # Compute loss
        rl_loss = F.mse_loss(current_q, target_values)

        # Backward pass
        rl_loss.backward()
        self.rl_optimizer.step()

        return rl_loss.item()

    def _update_sl_network(self) -> float:
        """Update supervised learning network."""
        if len(self.sl_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.sl_buffer), self.batch_size)

        # Prepare training data
        info_states = torch.FloatTensor([t["info_state"] for t in batch])
        target_policies = torch.FloatTensor([t["target_policy"] for t in batch])

        # Get current policy
        self.sl_optimizer.zero_grad()
        network_output = self.network(info_states)
        current_policy = network_output["sl_policy"]

        # Compute cross-entropy loss
        sl_loss = F.cross_entropy(current_policy, target_policies)

        # Backward pass
        sl_loss.backward()
        self.sl_optimizer.step()

        return sl_loss.item()

    def _update_target_network(self):
        """Update target network with current network weights."""
        tau = 0.005  # Soft update parameter
        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _update_current_strategy(self):
        """Update current strategy based on network predictions."""
        # Sample some info states from RL buffer to update strategy
        if len(self.rl_buffer) > 0:
            sample_transitions = random.sample(
                list(self.rl_buffer), min(1000, len(self.rl_buffer))
            )

            new_strategy = {}
            for transition in sample_transitions:
                info_state_key = transition["info_state_key"]
                legal_actions = transition["legal_actions"]

                # Get SL policy for this state
                with torch.no_grad():
                    info_tensor = torch.FloatTensor(transition["info_state"]).unsqueeze(
                        0
                    )
                    network_output = self.network(info_tensor)
                    policy = network_output["sl_policy"].squeeze().cpu().numpy()

                # Apply legal action mask
                legal_mask = np.zeros(len(policy))
                legal_mask[legal_actions] = 1.0
                masked_policy = policy * legal_mask

                if masked_policy.sum() > 0:
                    masked_policy = masked_policy / masked_policy.sum()
                else:
                    masked_policy[legal_actions] = 1.0 / len(legal_actions)

                new_strategy[info_state_key] = masked_policy

            self.current_strategy = new_strategy
            self.strategy_history.append(new_strategy.copy())

    def get_policy(self, player: int) -> callable:
        """Get the current policy for a player."""

        def policy(info_state_key: str, legal_actions: List[int]) -> np.ndarray:
            # Get information state encoding
            info_state = self.game_wrapper.encode_info_state_key(info_state_key, player)

            # Get action probabilities from SL policy
            with torch.no_grad():
                info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
                network_output = self.network(info_tensor)
                policy_probs = network_output["sl_policy"].squeeze().cpu().numpy()

            # Apply legal action mask
            legal_mask = np.zeros(len(policy_probs))
            legal_mask[legal_actions] = 1.0
            masked_policy = policy_probs * legal_mask

            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                masked_policy[legal_actions] = 1.0 / len(legal_actions)

            return masked_policy

        return policy

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        from eval.openspiel_evaluator import OpenSpielExactEvaluator

        evaluator = OpenSpielExactEvaluator(self.game_wrapper.game_name)
        metadata = PolicyMetadata(method="nfsp", iteration=self.iteration)

        def policy_fn(
            player_id: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            return self.get_policy(player_id)(info_state, list(legal_actions))

        policy_adapter = evaluator.build_policy(policy_fn, metadata=metadata)

        # Evaluate exploitability
        results = evaluator.evaluate_policy(policy_adapter)

        return {
            "exploitability": results.get("exploitability", 0.0),
            "nash_conv": results.get("nash_conv", 0.0),
            "iteration": self.iteration,
        }

    def save_checkpoint(self, path: str):
        """Save NFSP checkpoint."""
        checkpoint = {
            "iteration": self.iteration,
            "network_state": self.network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "rl_optimizer_state": self.rl_optimizer.state_dict(),
            "sl_optimizer_state": self.sl_optimizer.state_dict(),
            "current_strategy": self.current_strategy,
            "strategy_history": self.strategy_history,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved NFSP checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load NFSP checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.iteration = checkpoint["iteration"]
        self.network.load_state_dict(checkpoint["network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.rl_optimizer.load_state_dict(checkpoint["rl_optimizer_state"])
        self.sl_optimizer.load_state_dict(checkpoint["sl_optimizer_state"])
        self.current_strategy = checkpoint["current_strategy"]
        self.strategy_history = checkpoint["strategy_history"]

        self.logger.info(f"Loaded NFSP checkpoint from {path}")

    def get_policy_adapter(self):
        """Return a PolicyAdapter instance for the current strategy."""

        def policy_fn(
            player: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            del player
            strategy = self.current_strategy.get(info_state)
            if strategy is None or strategy.sum() <= 0:
                return np.full(
                    len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64
                )
            legal_strategy = strategy[legal_actions]
            total = legal_strategy.sum()
            if total <= 0:
                return np.full(
                    len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64
                )
            return (legal_strategy / total).astype(np.float64)

        from eval.openspiel_evaluator import create_evaluator

        evaluator = create_evaluator(self.game_wrapper.game_name)
        return evaluator.build_policy(
            policy_fn, PolicyMetadata(method="nfsp", iteration=self.iteration)
        )
