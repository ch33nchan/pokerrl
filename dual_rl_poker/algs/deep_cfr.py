"""Deep Counterfactual Regret Minimization implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import time

from algs.base import BaseAlgorithm, TrainingState, ExperienceBuffer
from nets.mlp import DeepCFRNetwork
from eval.openspiel_evaluator import create_evaluator, PolicyMetadata


class DeepCFRAlgorithm(BaseAlgorithm):
    """Deep Counterfactual Regret Minimization algorithm with external sampling.

    Follows the ICML 2019 Deep CFR paper with separate advantage and strategy networks.
    """

    def __init__(self, game_wrapper, config: Dict[str, Any]):
        """Initialize Deep CFR algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Configuration dictionary
        """
        raise NotImplementedError(
            "DeepCFRAlgorithm is not yet fully implemented for this repository. "
            "Use the tabular ARMAC or CFR pipelines for reproducible results."
        )

        # Deep CFR specific parameters
        self.advantage_memory_size = config.get('advantage_memory_size', 10000)
        self.strategy_memory_size = config.get('strategy_memory_size', 10000)
        self.replay_window = config.get('replay_window', 10)

        # Separate networks for regret and strategy
        self.regret_network = DeepCFRNetwork(
            input_dim=game_wrapper.encoder.encoding_size,
            num_actions=self.num_actions,
            hidden_dims=config.get('hidden_dims', [64, 64])
        ).to(self.device)

        self.strategy_network = DeepCFRNetwork(
            input_dim=game_wrapper.encoder.encoding_size,
            num_actions=self.num_actions,
            hidden_dims=config.get('hidden_dims', [64, 64])
        ).to(self.device)

        # Separate optimizers
        self.regret_optimizer = torch.optim.Adam(
            self.regret_network.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.0)
        )

        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.0)
        )

    # Separate buffers
        self.regret_buffer = ExperienceBuffer(self.advantage_memory_size)
        self.strategy_buffer = ExperienceBuffer(self.strategy_memory_size)

        # Current regrets and strategies
        self.regrets: Dict[str, np.ndarray] = {}
        self.strategies: Dict[str, np.ndarray] = {}
        self.cumulative_strategies: Dict[str, np.ndarray] = {}

        # Evaluation setup
        self.evaluator = create_evaluator(game_wrapper.game_name)

    def _create_network(self):
        """Create network (overridden to handle separate networks)."""
        return None  # Networks are created separately in __init__

    def train_iteration(self) -> TrainingState:
        """Perform one Deep CFR training iteration.

        Returns:
            Training state with metrics
        """
        start_time = time.time()

        # Step 1: Generate trajectories with external sampling
        trajectories = self._collect_trajectories()

        # Step 2: Update regret network
        regret_metrics = self._update_regret_network(trajectories)

        # Step 3: Update strategy network
        strategy_metrics = self._update_strategy_network(trajectories)

        # Step 4: Update cumulative strategies
        self._update_cumulative_strategies(trajectories)

        self.iteration += 1
        wall_time = time.time() - start_time

        # Create training state
        training_state = TrainingState(
            iteration=self.iteration,
            loss=regret_metrics.get('loss', 0.0) + strategy_metrics.get('loss', 0.0),
            wall_time=wall_time,
            gradient_norm=max(
                regret_metrics.get('gradient_norm', 0.0),
                strategy_metrics.get('gradient_norm', 0.0)
            ),
            learning_rate=self.learning_rate,
            buffer_size=len(self.regret_buffer) + len(self.strategy_buffer),
            extra_metrics={
                'algorithm': 'deep_cfr',
                'regret_loss': regret_metrics.get('loss', 0.0),
                'strategy_loss': strategy_metrics.get('loss', 0.0),
                'regret_buffer_size': len(self.regret_buffer),
                'strategy_buffer_size': len(self.strategy_buffer),
                'trajectories_collected': len(trajectories),
                'num_info_states': len(self.regrets)
            }
        )

        self.training_history.append(training_state)
        return training_state

    def _collect_trajectories(self) -> List[Dict[str, Any]]:
        """Collect trajectories using external sampling.

        Returns:
            List of trajectory experiences
        """
        trajectories = []
        state = self.game.new_initial_state()
        self._collect_trajectory(state, trajectories, np.ones(1))
        return trajectories

    def _collect_trajectory(self, state, trajectories: List[Dict[str, Any]],
                           reach_prob: np.ndarray):
        """Recursively collect trajectory with external sampling.

        Args:
            state: Current game state
            trajectories: List to collect experiences
            reach_prob: Reach probability for current player
        """
        if state.is_terminal():
            return

        if state.is_chance_node():
            # Sample chance outcome
            outcomes = state.chance_outcomes()
            probs = [outcome[1] for outcome in outcomes]
            chosen = np.random.choice(len(outcomes), p=probs)
            action = outcomes[chosen][0]
            new_state = state.child(action)
            self._collect_trajectory(new_state, trajectories, reach_prob)
            return

        current_player = state.current_player()
        legal_actions = state.legal_actions()

        # Get information state
        info_state = self.game_wrapper.encode_state(state)
        info_state_str = state.information_state_string(current_player)
        legal_actions_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        legal_actions_mask[legal_actions] = True

        # Get current strategy
        strategy = self._get_current_strategy(info_state_str, legal_actions)

        # Sample action according to strategy
        action_probs = np.array([strategy[a] for a in legal_actions])
        action_probs = action_probs / action_probs.sum()
        action_idx = np.random.choice(len(legal_actions), p=action_probs)
        action = legal_actions[action_idx]

        # Store experience
        trajectories.append({
            'info_state': info_state,
            'info_state_str': info_state_str,
            'legal_actions': legal_actions,
            'legal_actions_mask': legal_actions_mask.numpy(),
            'action': action,
            'action_idx': action_idx,
            'player': current_player,
            'reach_prob': reach_prob[0],
            'strategy': strategy.copy(),
            'behavior_policy': strategy.copy()
        })

        # Continue trajectory
        new_state = state.child(action)
        self._collect_trajectory(new_state, trajectories, reach_prob)

    def _update_regret_network(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update regret network using collected trajectories.

        Args:
            trajectories: List of trajectory experiences

        Returns:
            Dictionary with training metrics
        """
        if len(trajectories) == 0:
            return {'loss': 0.0, 'gradient_norm': 0.0}

        # Add experiences to regret buffer
        for traj in trajectories:
            # Compute regret targets
            regret_targets = self._compute_regret_targets(traj)
            traj['regret_targets'] = regret_targets
            self.regret_buffer.add(traj)

        # Sample batch and train
        batch = self.regret_buffer.sample(self.batch_size)
        if len(batch) == 0:
            return {'loss': 0.0, 'gradient_norm': 0.0}

        # Prepare batch data
        info_states = torch.stack([torch.tensor(t['info_state'], dtype=torch.float32)
                                 for t in batch]).to(self.device)
        legal_actions_masks = torch.stack([torch.tensor(t['legal_actions_mask'], dtype=torch.bool)
                                         for t in batch]).to(self.device)
        regret_targets = torch.stack([torch.tensor(t['regret_targets'], dtype=torch.float32)
                                    for t in batch]).to(self.device)

        # Forward pass through regret network
        self.regret_optimizer.zero_grad()
        network_output = self.regret_network(info_states, network_type='regret')
        predicted_advantages = network_output['advantages']

        if self.diagnostics is not None:
            self.diagnostics.log_advantage_statistics(
                predicted_advantages.detach().cpu(),
                iteration=self.iteration
            )

        if self.diagnostics is not None:
            with torch.no_grad():
                legal_mask_float = legal_actions_masks.float()
                positive_advantages = torch.clamp(predicted_advantages.detach(), min=0.0)
                masked_advantages = positive_advantages * legal_mask_float
                mass = masked_advantages.sum(dim=-1, keepdim=True)
                uniform_policy = legal_mask_float / (legal_mask_float.sum(dim=-1, keepdim=True) + 1e-8)
                current_policy = torch.where(
                    mass > 1e-8,
                    masked_advantages / (mass + 1e-8),
                    uniform_policy
                )
                behavior_policy = torch.stack([
                    torch.tensor(t['behavior_policy'], dtype=torch.float32)
                    for t in batch
                ])
                self.diagnostics.log_policy_kl(
                    current_policy.cpu(),
                    behavior_policy,
                    legal_mask_float.cpu(),
                    iteration=self.iteration,
                )

        # MSE loss for regret matching
        loss = nn.functional.mse_loss(predicted_advantages, regret_targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        total_norm = 0
        if self.gradient_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.regret_network.parameters(), self.gradient_clip
            ).item()

        self.regret_optimizer.step()

        return {'loss': loss.item(), 'gradient_norm': total_norm}

    def _update_strategy_network(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update strategy network using collected trajectories.

        Args:
            trajectories: List of trajectory experiences

        Returns:
            Dictionary with training metrics
        """
        if len(trajectories) == 0:
            return {'loss': 0.0, 'gradient_norm': 0.0}

        # Add experiences to strategy buffer
        for traj in trajectories:
            # Compute strategy targets (from positive regrets)
            strategy_targets = self._compute_strategy_targets(traj['info_state_str'])
            traj['strategy_targets'] = strategy_targets
            self.strategy_buffer.add(traj)

        # Sample batch and train
        batch = self.strategy_buffer.sample(self.batch_size)
        if len(batch) == 0:
            return {'loss': 0.0, 'gradient_norm': 0.0}

        # Prepare batch data
        info_states = torch.stack([torch.tensor(t['info_state'], dtype=torch.float32)
                                 for t in batch]).to(self.device)
        legal_actions_masks = torch.stack([torch.tensor(t['legal_actions_mask'], dtype=torch.bool)
                                         for t in batch]).to(self.device)
        strategy_targets = torch.stack([torch.tensor(t['strategy_targets'], dtype=torch.float32)
                                       for t in batch]).to(self.device)

        # Forward pass through strategy network
        self.strategy_optimizer.zero_grad()
        network_output = self.strategy_network(info_states, network_type='strategy')
        predicted_policy = network_output['policy']

        if self.diagnostics is not None:
            with torch.no_grad():
                legal_mask_float = legal_actions_masks.float()
                masked_policy = predicted_policy.detach() * legal_mask_float
                policy_mass = masked_policy.sum(dim=-1, keepdim=True)
                uniform_policy = legal_mask_float / (legal_mask_float.sum(dim=-1, keepdim=True) + 1e-8)
                normalized_policy = torch.where(
                    policy_mass > 1e-8,
                    masked_policy / (policy_mass + 1e-8),
                    uniform_policy
                )
                behavior_policy = torch.stack([
                    torch.tensor(t['behavior_policy'], dtype=torch.float32)
                    for t in batch
                ])
                self.diagnostics.log_policy_kl(
                    normalized_policy.cpu(),
                    behavior_policy,
                    legal_mask_float.cpu(),
                    iteration=self.iteration,
                )

        # Cross-entropy loss for strategy matching toward normalized positive regrets
        # Apply legal actions mask to strategy targets
        legal_mask_float = legal_actions_masks.float()
        masked_strategy_targets = strategy_targets * legal_mask_float
        
        # Normalize strategy targets
        target_sum = masked_strategy_targets.sum(dim=-1, keepdim=True)
        normalized_targets = torch.where(
            target_sum > 1e-8,
            masked_strategy_targets / (target_sum + 1e-8),
            legal_mask_float / (legal_mask_float.sum(dim=-1, keepdim=True) + 1e-8)
        )
        
        # Apply softmax to policy logits with masking
        masked_logits = predicted_policy + (1 - legal_mask_float) * (-1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        
        # Cross-entropy loss
        loss = -(normalized_targets * log_probs).sum(dim=-1).mean()

        # Backward pass
        loss.backward()

        # Gradient clipping
        total_norm = 0
        if self.gradient_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.strategy_network.parameters(), self.gradient_clip
            ).item()

        self.strategy_optimizer.step()

        return {'loss': loss.item(), 'gradient_norm': total_norm}

    def _compute_regret_targets(self, trajectory: Dict[str, Any]) -> np.ndarray:
        """Compute regret targets for a trajectory step.

        Args:
            trajectory: Single trajectory experience

        Returns:
            Regret targets for all actions
        """
        info_state_str = trajectory['info_state_str']
        legal_actions = trajectory['legal_actions']
        action = trajectory['action']

        # Get current regrets for this info state
        current_regrets = self.regrets.get(info_state_str, np.zeros(self.num_actions))

        # Simple regret computation (placeholder - in practice would compute
        # actual counterfactual regrets)
        regret_targets = current_regrets.copy()

        # Add regret for taken action (positive or negative)
        # This is a simplified version
        if action < len(regret_targets):
            regret_targets[action] += 0.1  # Placeholder regret update

        # Store updated regrets
        self.regrets[info_state_str] = regret_targets

        return regret_targets

    def _compute_strategy_targets(self, info_state_str: str) -> np.ndarray:
        """Compute strategy targets from normalized positive-regret distribution.

        Args:
            info_state_str: Information state string

        Returns:
            Strategy probability distribution (normalized positive regrets)
        """
        regrets = self.regrets.get(info_state_str, np.zeros(self.num_actions))
        positive_regrets = np.maximum(regrets, 0)

        if positive_regrets.sum() > 0:
            # Normalize positive regrets to create strategy distribution
            strategy = positive_regrets / positive_regrets.sum()
        else:
            # Uniform distribution if no positive regrets
            strategy = np.ones(self.num_actions) / self.num_actions

        return strategy

    def _update_cumulative_strategies(self, trajectories: List[Dict[str, Any]]):
        """Update cumulative strategy averages.

        Args:
            trajectories: List of trajectory experiences
        """
        for traj in trajectories:
            info_state_str = traj['info_state_str']
            strategy = traj['strategy']

            if info_state_str not in self.cumulative_strategies:
                self.cumulative_strategies[info_state_str] = strategy.copy()
            else:
                # Running average
                self.cumulative_strategies[info_state_str] = (
                    0.9 * self.cumulative_strategies[info_state_str] +
                    0.1 * strategy
                )

    def _get_current_strategy(self, info_state_str: str, legal_actions: List[int]) -> np.ndarray:
        """Get current strategy for an information state.

        Args:
            info_state_str: Information state string
            legal_actions: List of legal action indices

        Returns:
            Strategy probability distribution
        """
        strategy = self.cumulative_strategies.get(info_state_str)
        if strategy is None:
            strategy = np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        legal_mask = np.zeros_like(strategy)
        legal_mask[legal_actions] = strategy[legal_actions]
        total = legal_mask[legal_actions].sum()
        if total <= 0:
            legal_probs = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
        else:
            legal_probs = (legal_mask[legal_actions] / total).astype(np.float64)

        # Construct full strategy vector for storage
        full_strategy = np.zeros_like(strategy)
        full_strategy[legal_actions] = legal_probs
        return full_strategy

    def get_policy(self, player: int):
        """Get current policy for evaluation.

        Args:
            player: Player index

        Returns:
            Policy function for OpenSpiel evaluator
        """
        def policy_function(info_state: str, legal_actions: List[int]) -> np.ndarray:
            """Policy function returning action probabilities.

            Args:
                info_state: Information state string
                legal_actions: List of legal action indices

            Returns:
                Action probabilities
            """
            strategy = self._get_current_strategy(info_state, legal_actions)
            return strategy

        return policy_function

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy using exact OpenSpiel evaluator.

        Returns:
            Dictionary with evaluation metrics
        """
        policy = self.get_policy_adapter()
        result = self.evaluator.evaluate(
            policy,
            metadata=PolicyMetadata(method="deep_cfr", iteration=self.iteration),
        )

        return {
            'nash_conv': result.nash_conv,
            'exploitability': result.exploitability,
            'mean_value': result.mean_value,
            'player_0_value': result.player_0_value,
            'player_1_value': result.player_1_value,
            'num_info_states': result.info_state_count,
            'openspiel_version': result.openspiel_version,
            'python_version': result.python_version,
        }

    def get_policy_adapter(self):
        """Return a PolicyAdapter instance for the current averaged strategy."""

        def policy_fn(player: int, info_state: str, legal_actions: List[int]) -> np.ndarray:
            del player
            strategy = self.cumulative_strategies.get(info_state)
            if strategy is None or strategy.sum() <= 0:
                return np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
            legal_strategy = strategy[legal_actions]
            total = legal_strategy.sum()
            if total <= 0:
                return np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
            return (legal_strategy / total).astype(np.float64)

        return self.evaluator.build_policy(
            policy_fn,
            metadata=PolicyMetadata(method="deep_cfr", iteration=self.iteration),
        )

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """Get average strategy for analysis.

        Returns:
            Dictionary mapping info states to strategies
        """
        return self.cumulative_strategies.copy()

    def get_regrets(self) -> Dict[str, np.ndarray]:
        """Get current regrets for analysis.

        Returns:
            Dictionary mapping info states to regrets
        """
        return self.regrets.copy()

    def train_step(self) -> Dict[str, float]:
        """Train step for compatibility with base class.

        Returns:
            Dictionary with training metrics
        """
        state = self.train_iteration()
        return {
            'loss': state.loss,
            'gradient_norm': state.gradient_norm,
            'wall_time': state.wall_time
        }
