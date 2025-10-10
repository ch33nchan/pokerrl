"""Self-Play Deep CFR (SD-CFR) algorithm implementation.

SD-CFR extends Deep CFR with improved self-play dynamics and more stable
training through regret accumulation and strategy updates.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

from algs.base import BaseAlgorithm, TrainingState, ExperienceBuffer
from utils.logging import get_experiment_logger


class SDCFRAlgorithm(BaseAlgorithm):
    """Self-Play Deep CFR algorithm implementation.

    SD-CFR improves upon standard Deep CFR through:
    - Enhanced self-play dynamics with proper opponent modeling
    - Stabilized regret accumulation across iterations
    - Improved strategy network training with better sampling
    - Adaptive learning rates and exploration schedules
    """

    def __init__(self, game_wrapper, config: Dict[str, Any]):
        """Initialize SD-CFR algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Algorithm configuration
        """
        super().__init__(game_wrapper, config)

        self.logger = get_experiment_logger("sd_cfr")

        # SD-CFR specific parameters
        self.regret_buffer_size = config.get('regret_buffer_size', 10000)
        self.regret_learning_rate = config.get('regret_learning_rate', 1e-3)
        self.regret_update_frequency = config.get('regret_update_frequency', 1)
        self.regret_decay = config.get('regret_decay', 0.99)
        self.strategy_smoothing = config.get('strategy_smoothing', 0.1)

        # Initialize neural networks
        from nets.mlp import DeepCFRNetwork
        encoding_size = game_wrapper.get_encoding_size()
        num_actions = game_wrapper.num_actions

        self.regret_network = DeepCFRNetwork(
            encoding_size,
            num_actions,
            config['network']['hidden_dims']
        )
        # SD-CFR reconstructs strategy from regret network (no separate strategy network)
        # This ensures fair comparison with Deep CFR

        # Initialize optimizer for regret network only
        self.regret_optimizer = torch.optim.Adam(
            self.regret_network.parameters(),
            lr=self.regret_learning_rate,
            weight_decay=config['training'].get('weight_decay', 0.0)
        )

        # Experience replay buffer (only regret buffer needed)
        self.regret_buffer = deque(maxlen=self.regret_buffer_size)

        # Regret accumulation and strategy tracking
        self.cumulative_regrets = defaultdict(lambda: np.zeros(num_actions))
        self.cumulative_strategies = defaultdict(lambda: np.zeros(num_actions))
        self.strategy_counts = defaultdict(int)

        # Training statistics
        self.iteration_count = 0
        self.total_trajectories = 0

        # Exploration parameters
        self.initial_epsilon = config.get('initial_epsilon', 0.5)
        self.final_epsilon = config.get('final_epsilon', 0.01)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 1000)

        self.logger.info("Initialized SD-CFR algorithm with regret reconstruction")
        self.logger.info(f"Regret network parameters: {sum(p.numel() for p in self.regret_network.parameters())}")
        self.logger.info("Strategy reconstruction from regret network (no separate strategy network)")

    def train_iteration(self) -> TrainingState:
        """Perform one SD-CFR training iteration.

        Returns:
            Training state with loss and buffer information
        """
        start_time = time.time()

        # Step 1: Generate self-play trajectories with external sampling
        trajectories = self._collect_self_play_trajectories()

        # Step 2: Update regret network with sampled data
        regret_metrics = self._update_regret_network_sd()

        # Step 3: Update cumulative strategies with smoothing
        self._update_cumulative_strategies_sd(trajectories)

        # Step 4: Apply regret decay for stability
        self._apply_regret_decay()

        # Calculate training statistics
        iteration_time = time.time() - start_time

        training_state = TrainingState(
            iteration=self.iteration_count,
            loss=regret_metrics['loss'],
            regret_loss=regret_metrics['loss'],
            strategy_loss=0.0,  # No separate strategy network
            buffer_size=len(self.regret_buffer),
            wall_time=iteration_time,
            gradient_norm=regret_metrics['gradient_norm'],
            extra_metrics={
                'num_trajectories': len(trajectories),
                'regret_buffer_size': len(self.regret_buffer),
                'epsilon': self._get_current_epsilon(),
                'avg_regret_norm': np.mean([np.linalg.norm(r) for r in self.cumulative_regrets.values() if len(r) > 0]),
                'algorithm': 'SD-CFR'
            }
        )

        self.iteration_count += 1
        return training_state

    def _collect_self_play_trajectories(self) -> List[Dict[str, Any]]:
        """Collect self-play trajectories using external sampling.

        Returns:
            List of trajectory data
        """
        trajectories = []
        batch_size = self.config['training']['batch_size']

        for _ in range(batch_size):
            trajectory = self._generate_self_play_trajectory()
            if trajectory:
                trajectories.extend(trajectory)

        self.total_trajectories += len(trajectories)
        return trajectories

    def _generate_self_play_trajectory(self) -> List[Dict[str, Any]]:
        """Generate a single self-play trajectory using external sampling.

        Returns:
            List of transition data
        """
        trajectory = []
        state = self.game_wrapper.get_initial_state()

        while not self.game_wrapper.is_terminal(state):
            current_player = self.game_wrapper.get_current_player(state)
            if current_player == -1:  # Chance node
                state = self.game_wrapper.sample_chance_action(state)
                continue

            # Get legal actions
            legal_actions = self.game_wrapper.get_legal_actions(state)
            if len(legal_actions) <= 1:
                action = legal_actions[0] if legal_actions else None
                state = self.game_wrapper.make_action(state, current_player, action)
                continue

            # Encode information state
            info_state = self.game_wrapper.encode_state(state)
            info_state_key = self.game_wrapper.get_info_state_key(state, current_player)

            # Get action probabilities using current strategy network with exploration
            action_probs = self._get_exploratory_strategy(info_state, legal_actions)

            # Sample action according to strategy
            action = np.random.choice(legal_actions, p=action_probs)

            # Store transition for training
            transition = {
                'info_state': info_state,
                'info_state_key': info_state_key,
                'legal_actions': legal_actions,
                'action': action,
                'action_probs': action_probs,
                'player': current_player,
                'iteration': self.iteration_count
            }
            trajectory.append(transition)

            # Apply action to get next state
            state = self.game_wrapper.make_action(state, current_player, action)

        # Calculate returns for each player in the trajectory
        final_rewards = self.game_wrapper.get_rewards(state)
        self._assign_trajectory_returns(trajectory, final_rewards)

        return trajectory

    def _get_exploratory_strategy(self, info_state: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """Get action probabilities with epsilon-exploration using regret reconstruction.

        Args:
            info_state: Encoded information state
            legal_actions: List of legal action indices

        Returns:
            Action probabilities with exploration
        """
        # Get current epsilon for exploration
        epsilon = self._get_current_epsilon()

        # Get base strategy from regret network using reconstruction
        base_probs = self._reconstruct_strategy_from_regrets(info_state, legal_actions)

        # Apply epsilon-exploration
        legal_mask = np.zeros(len(base_probs))
        legal_mask[legal_actions] = 1.0

        action_probs = (1 - epsilon) * base_probs + epsilon * (legal_mask / legal_mask.sum())

        return action_probs

    def _reconstruct_strategy_from_regrets(self, info_state: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """Reconstruct strategy from regret predictions using regret matching.

        Args:
            info_state: Encoded information state
            legal_actions: List of legal action indices

        Returns:
            Action probabilities from reconstructed strategy
        """
        with torch.no_grad():
            info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
            network_output = self.regret_network(info_tensor)
            predicted_regrets = network_output['action_values'].squeeze().cpu().numpy()

        # Apply positive regret transformation
        positive_regrets = np.maximum(predicted_regrets, 0)

        # Apply legal action mask
        masked_regrets = np.zeros_like(positive_regrets)
        masked_regrets[legal_actions] = positive_regrets[legal_actions]

        # Regret matching: normalize positive regrets
        if masked_regrets.sum() > 0:
            strategy = masked_regrets / masked_regrets.sum()
        else:
            # Uniform strategy if no positive regrets
            strategy = np.zeros(len(positive_regrets))
            strategy[legal_actions] = 1.0 / len(legal_actions)

        return strategy

    def _get_current_epsilon(self) -> float:
        """Get current exploration epsilon.

        Returns:
            Current epsilon value
        """
        decay_progress = min(self.iteration_count, self.epsilon_decay_steps)
        return self.initial_epsilon * (self.final_epsilon / self.initial_epsilon) ** (decay_progress / self.epsilon_decay_steps)

    def _assign_trajectory_returns(self, trajectory: List[Dict[str, Any]], final_rewards: Dict[int, float]):
        """Assign returns to trajectory transitions for counterfactual reasoning.

        Args:
            trajectory: List of trajectory transitions
            final_rewards: Final rewards for each player
        """
        # Calculate cumulative returns for each player
        player_returns = {p: 0.0 for p in range(self.game_wrapper.num_players)}

        # Process trajectory in reverse order
        for transition in reversed(trajectory):
            player = transition['player']

            # Update return for current player
            player_returns[player] = final_rewards[player]

            # Store immediate reward and return for regret calculation
            transition['immediate_reward'] = final_rewards[player]
            transition['return'] = player_returns[player]

            # The return for other players remains the same at this step
            for other_player in range(self.game_wrapper.num_players):
                if other_player != player:
                    transition[f'return_player_{other_player}'] = player_returns[other_player]

    def _update_regret_network_sd(self) -> Dict[str, float]:
        """Update regret network using SD-CFR specific sampling.

        Returns:
            Training metrics
        """
        if len(self.regret_buffer) < self.config['training']['batch_size']:
            return {'loss': 0.0, 'gradient_norm': 0.0}

        # Sample regret training data with importance sampling
        batch_size = min(self.config['training']['batch_size'], len(self.regret_buffer))
        regret_batch = random.sample(list(self.regret_buffer), batch_size)

        # Prepare training data
        info_states = torch.FloatTensor([t['info_state'] for t in regret_batch])
        legal_actions_mask = torch.FloatTensor([t['legal_actions_mask'] for t in regret_batch])
        target_regrets = torch.FloatTensor([t['target_regrets'] for t in regret_batch])

        # Forward pass
        self.regret_optimizer.zero_grad()
        network_output = self.regret_network(info_states)
        predicted_regrets = network_output['action_values']

        # Apply legal actions mask
        masked_pred_regrets = predicted_regrets * legal_actions_mask

        # Calculate MSE loss for regret prediction
        regret_loss = nn.MSELoss()(masked_pred_regrets, target_regrets)

        # Backward pass with gradient clipping
        regret_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.regret_network.parameters(),
            self.config['training'].get('gradient_clip', 5.0)
        )

        self.regret_optimizer.step()

        return {
            'loss': regret_loss.item(),
            'gradient_norm': gradient_norm.item()
        }

    
    def _update_cumulative_strategies_sd(self, trajectories: List[Dict[str, Any]]):
        """Update cumulative strategies with smoothing for stability.

        Args:
            trajectories: List of trajectory transitions
        """
        for transition in trajectories:
            info_state_key = transition['info_state_key']
            legal_actions = transition['legal_actions']
            action_probs = transition['action_probs']

            # Update cumulative strategy with smoothing
            for i, action in enumerate(legal_actions):
                self.cumulative_strategies[info_state_key][action] += action_probs[i]

            self.strategy_counts[info_state_key] += 1

            # Calculate regrets and add to regret buffer
            self._calculate_and_store_regrets(transition)

    def _calculate_and_store_regrets(self, transition: Dict[str, Any]):
        """Calculate counterfactual regrets using proper CFR methodology.

        Args:
            transition: Single trajectory transition
        """
        info_state_key = transition['info_state_key']
        player = transition['player']
        action = transition['action']
        legal_actions = transition['legal_actions']
        immediate_return = transition['return']

        # Get current average strategy for this information state
        if info_state_key in self.cumulative_strategies:
            cum_strategy = self.cumulative_strategies[info_state_key]
            total_strategy = cum_strategy.sum()
            if total_strategy > 0:
                avg_strategy = cum_strategy / total_strategy
            else:
                avg_strategy = np.zeros(self.game_wrapper.num_actions)
                avg_strategy[legal_actions] = 1.0 / len(legal_actions)
        else:
            avg_strategy = np.zeros(self.game_wrapper.num_actions)
            avg_strategy[legal_actions] = 1.0 / len(legal_actions)

        # For external sampling CFR, the counterfactual value is the immediate return
        # since we're already conditioning on the player's information state
        cf_value = immediate_return

        # Calculate regrets: r_i(a) = u_i(a) - u_i(σ)
        # where u_i(a) is the counterfactual value of taking action a
        # and u_i(σ) is the counterfactual value of following current strategy
        regrets = np.zeros(self.game_wrapper.num_actions)

        # Expected value under current strategy
        ev_current = np.sum(avg_strategy[legal_actions] * cf_value)

        for legal_action in legal_actions:
            # In external sampling, regret is simply the difference between
            # taking action a and the expected value under current strategy
            regrets[legal_action] = cf_value - ev_current

        # Update cumulative regrets
        self.cumulative_regrets[info_state_key] += regrets

        # Apply positive regret transformation and add to buffer
        positive_regrets = np.maximum(regrets, 0)
        if np.sum(positive_regrets) > 0:
            self._add_to_regret_buffer(transition, positive_regrets)

    def _add_to_regret_buffer(self, transition: Dict[str, Any], regrets: np.ndarray):
        """Add transition to regret buffer for training.

        Args:
            transition: Trajectory transition
            regrets: Positive regrets for each action
        """
        legal_actions = transition['legal_actions']
        legal_actions_mask = np.zeros(self.game_wrapper.num_actions)
        legal_actions_mask[legal_actions] = 1.0

        buffer_entry = {
            'info_state': transition['info_state'],
            'legal_actions_mask': legal_actions_mask,
            'target_regrets': regrets,
            'player': transition['player']
        }

        self.regret_buffer.append(buffer_entry)

    
    def _apply_regret_decay(self):
        """Apply regret decay for training stability."""
        decay_factor = self.regret_decay

        for info_state_key in self.cumulative_regrets:
            self.cumulative_regrets[info_state_key] *= decay_factor

    def get_policy(self, player: int) -> callable:
        """Get the current policy for a player using regret reconstruction.

        Args:
            player: Player index

        Returns:
            Policy function that maps info states to action probabilities
        """
        def policy(info_state_key: str, legal_actions: List[int]) -> np.ndarray:
            # Get information state encoding
            info_state = self.game_wrapper.encode_info_state_key(info_state_key, player)

            # Get action probabilities from regret network using reconstruction
            action_probs = self._reconstruct_strategy_from_regrets(info_state, legal_actions)

            return action_probs

        return policy

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """Get the average strategy across all information states.

        Returns:
            Dictionary mapping info state keys to action probabilities
        """
        avg_strategy = {}

        for info_state_key, cum_strategy in self.cumulative_strategies.items():
            total = cum_strategy.sum()
            if total > 0:
                avg_strategy[info_state_key] = cum_strategy / total
            else:
                # Uniform strategy for unseen states
                avg_strategy[info_state_key] = np.ones(self.game_wrapper.num_actions) / self.game_wrapper.num_actions

        return avg_strategy

    def get_regrets(self) -> Dict[str, np.ndarray]:
        """Get the current regrets for all information states.

        Returns:
            Dictionary mapping info state keys to regret values
        """
        return dict(self.cumulative_regrets)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the current policy using OpenSpiel evaluators.

        Returns:
            Evaluation metrics
        """
        from eval.evaluator import OpenSpielEvaluator

        evaluator = OpenSpielEvaluator(self.game_wrapper)

        # Get policies for all players
        policies = {}
        for player in range(self.game_wrapper.num_players):
            policies[player] = self.get_policy(player)

        # Run evaluation
        eval_metrics = evaluator.evaluate_with_diagnostics(policies, num_episodes=100)

        return eval_metrics

    def save_checkpoint(self, path: str):
        """Save algorithm checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'iteration': self.iteration_count,
            'regret_network_state': self.regret_network.state_dict(),
            'regret_optimizer_state': self.regret_optimizer.state_dict(),
            'cumulative_regrets': dict(self.cumulative_regrets),
            'cumulative_strategies': dict(self.cumulative_strategies),
            'strategy_counts': dict(self.strategy_counts),
            'config': self.config
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved SD-CFR checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load algorithm checkpoint.

        Args:
            path: Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')

        self.iteration_count = checkpoint['iteration']
        self.regret_network.load_state_dict(checkpoint['regret_network_state'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer_state'])
        self.cumulative_regrets = defaultdict(lambda: np.zeros(self.game_wrapper.num_actions),
                                            checkpoint['cumulative_regrets'])
        self.cumulative_strategies = defaultdict(lambda: np.zeros(self.game_wrapper.num_actions),
                                               checkpoint['cumulative_strategies'])
        self.strategy_counts = defaultdict(int, checkpoint['strategy_counts'])

        self.logger.info(f"Loaded SD-CFR checkpoint from {path} (iteration {self.iteration_count})")