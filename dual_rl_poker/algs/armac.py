"""ARMAC-style Dual RL algorithm implementation.

ARMAC (Actor-Critic with Regret Matching) combines policy gradient methods
with regret-based learning for improved performance in sequential games.
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
from algs.armac_dual_rl import ARMACDualRL


class ARMACAlgorithm(BaseAlgorithm):
    """ARMAC-style Actor-Critic with Regret Matching algorithm.

    ARMAC combines:
    - Actor-Critic architecture for policy and value learning
    - Regret matching for strategic exploration
    - Dual learning dynamics for stable training
    - Adaptive opponent modeling
    """

    def __init__(self, game_wrapper, config: Dict[str, Any]):
        """Initialize ARMAC algorithm.

        Args:
            game_wrapper: Game wrapper instance
            config: Algorithm configuration
        """
        super().__init__(game_wrapper, config)

        self.experiment_logger = get_experiment_logger("armac")
        self.logger = self.experiment_logger.get_logger()

        # ARMAC specific parameters
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        self.regret_lr = config.get("regret_lr", 1e-3)
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.tau = config.get("tau", 0.005)  # Soft update parameter
        self.regret_weight = config.get("regret_weight", 0.1)
        self.lambda_mode = config.get("lambda_mode", "fixed")
        self.lambda_alpha = config.get("lambda_alpha", 0.5)
        self.policy_update_frequency = config.get("policy_update_frequency", 1)
        self.value_update_frequency = config.get("value_update_frequency", 1)
        self.gradient_clip = config.get(
            "gradient_clip", config.get("training", {}).get("gradient_clip", 5.0)
        )

        # Initialize networks
        from nets.mlp import ARMACNetwork

        encoding_size = game_wrapper.get_encoding_size()
        num_actions = game_wrapper.num_actions

        # Actor and Critic networks
        hidden_dims = config.get(
            "hidden_dims", config.get("network", {}).get("hidden_dims", [64, 64])
        )

        self.actor = ARMACNetwork(
            encoding_size, num_actions, hidden_dims, network_type="actor"
        )

        self.critic = ARMACNetwork(
            encoding_size,
            1,  # Value output
            hidden_dims,
            network_type="critic",
        )

        # Target networks for stability
        self.actor_target = ARMACNetwork(
            encoding_size, num_actions, hidden_dims, network_type="actor"
        )

        self.critic_target = ARMACNetwork(
            encoding_size, 1, hidden_dims, network_type="critic"
        )

        # Regret network for strategic guidance
        self.regret_network = ARMACNetwork(
            encoding_size, num_actions, hidden_dims, network_type="regret"
        )

        # Initialize target networks
        self._update_target_networks(tau=1.0)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.regret_optimizer = torch.optim.Adam(
            self.regret_network.parameters(), lr=self.regret_lr
        )

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)

        # Regret tracking
        self.cumulative_regrets = defaultdict(lambda: np.zeros(num_actions))
        self.regret_counts = defaultdict(int)

        # Training statistics
        self.iteration_count = 0
        self.total_steps = 0

        # Exploration parameters
        self.initial_noise_scale = config.get("initial_noise_scale", 0.5)
        self.final_noise_scale = config.get("final_noise_scale", 0.01)
        self.noise_decay_steps = config.get("noise_decay_steps", 1000)

        # ARMAC dual RL module with adaptive lambda support
        self.armac_dual_rl = ARMACDualRL(
            num_actions=num_actions,
            mixture_weight=self.regret_weight,
            lambda_mode=self.lambda_mode,
            lambda_alpha=self.lambda_alpha,
        )

        self.logger.info("Initialized ARMAC algorithm")
        self.logger.info(
            f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}"
        )
        self.logger.info(
            f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}"
        )
        self.logger.info(
            f"Regret parameters: {sum(p.numel() for p in self.regret_network.parameters())}"
        )

    def _create_network(self):
        """ARMAC manages dedicated actor/critic networks explicitly."""
        return None

    def train_step(self) -> Dict[str, float]:
        state = self.train_iteration()
        return {
            "loss": state.loss,
            "gradient_norm": state.gradient_norm,
            "wall_time": state.wall_time,
        }

    def _update_target_networks(self, tau: float = 1.0):
        """Soft update target networks.

        Args:
            tau: Soft update parameter
        """
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_iteration(self) -> TrainingState:
        """Perform one ARMAC training iteration.

        Returns:
            Training state with loss and buffer information
        """
        start_time = time.time()

        # Step 1: Collect self-play experience
        trajectories = self._collect_armac_trajectories()

        # Step 2: Update regret network
        regret_metrics = self._update_regret_network()

        # Step 3: Update critic network
        critic_metrics = self._update_critic_network()

        # Step 4: Update actor network with regret guidance
        actor_metrics = self._update_actor_network()

        # Step 5: Update adaptive lambda if enabled
        current_lambda = self.regret_weight
        if self.lambda_mode == "adaptive":
            # Update loss averages for adaptive lambda
            self.armac_dual_rl.update_loss_averages(
                regret_metrics["loss"], actor_metrics["policy_gradient_loss"]
            )
            current_lambda = self.armac_dual_rl.compute_lambda_t(
                self.armac_dual_rl.avg_regret_loss,
                self.armac_dual_rl.avg_policy_loss,
                self.lambda_alpha,
            )

        # Step 6: Soft update target networks
        self._update_target_networks(self.tau)

        # Calculate training statistics
        iteration_time = time.time() - start_time
        total_loss = (
            actor_metrics["loss"] + critic_metrics["loss"] + regret_metrics["loss"]
        )

        training_state = TrainingState(
            iteration=self.iteration_count,
            loss=total_loss,
            buffer_size=len(self.replay_buffer),
            wall_time=iteration_time,
            gradient_norm=(
                actor_metrics["gradient_norm"]
                + critic_metrics["gradient_norm"]
                + regret_metrics["gradient_norm"]
            ),
            extra_metrics={
                "strategy_loss": actor_metrics["loss"],
                "value_loss": critic_metrics["loss"],
                "regret_loss": regret_metrics["loss"],
                "policy_gradient_loss": actor_metrics["policy_gradient_loss"],
                "num_trajectories": len(trajectories),
                "noise_scale": self._get_current_noise_scale(),
                "avg_regret_norm": np.mean(
                    [np.linalg.norm(r) for r in self.cumulative_regrets.values()]
                )
                if self.cumulative_regrets
                else 0.0,
                "current_lambda": current_lambda,
                "avg_regret_loss": self.armac_dual_rl.avg_regret_loss,
                "avg_policy_loss": self.armac_dual_rl.avg_policy_loss,
            },
        )

        self.iteration_count += 1
        return training_state

    def _collect_armac_trajectories(self) -> List[Dict[str, Any]]:
        """Collect trajectories using ARMAC exploration strategy.

        Returns:
            List of trajectory transitions
        """
        trajectories = []
        episodes_per_iteration = max(
            1, self.batch_size // 10
        )  # Adjust based on game length

        for _ in range(episodes_per_iteration):
            episode_trajectory = self._generate_armac_episode()
            trajectories.extend(episode_trajectory)

        return trajectories

    def _generate_armac_episode(self) -> List[Dict[str, Any]]:
        """Generate a single episode using ARMAC policy.

        Returns:
            List of episode transitions
        """
        trajectory = []
        state = self.game_wrapper.get_initial_state()
        episode_step = 0

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

            # Get action probabilities from actor with regret guidance
            action_probs = self._get_armac_strategy(
                info_state, legal_actions, info_state_key
            )

            # Extract probabilities for legal actions only
            legal_action_probs = action_probs[legal_actions]

            # Sample action
            action = np.random.choice(legal_actions, p=legal_action_probs)

            # Store transition
            transition = {
                "info_state": info_state,
                "info_state_key": info_state_key,
                "legal_actions": legal_actions,
                "legal_actions_mask": self._legal_actions_mask(legal_actions),
                "action": action,
                "action_probs": action_probs,
                "player": current_player,
                "episode_step": episode_step,
                "iteration": self.iteration_count,
            }
            trajectory.append(transition)

            # Apply action
            next_state = self.game_wrapper.make_action(state, current_player, action)
            state = next_state
            episode_step += 1

        # Assign rewards and bootstrap values
        final_rewards = self.game_wrapper.get_rewards(state)
        self._assign_armac_returns(trajectory, final_rewards)

        # Add to replay buffer
        for transition in trajectory:
            self.replay_buffer.append(transition)

        return trajectory

    def _get_armac_strategy(
        self, info_state: np.ndarray, legal_actions: List[int], info_state_key: str
    ) -> np.ndarray:
        """Get ARMAC action probabilities with improved regret matching integration.

        Args:
            info_state: Encoded information state
            legal_actions: List of legal actions
            info_state_key: Information state key for regret lookup

        Returns:
            Action probabilities
        """
        # Get policy from actor network
        with torch.no_grad():
            info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
            actor_output = self.actor(info_tensor)
            policy_probs = actor_output["action_probs"].squeeze().cpu().numpy()

        # Apply legal action mask to policy
        legal_mask = np.zeros(len(policy_probs))
        legal_mask[legal_actions] = 1.0
        masked_policy = policy_probs * legal_mask
        if masked_policy.sum() > 0:
            policy_probs = masked_policy / masked_policy.sum()
        else:
            policy_probs = legal_mask / legal_mask.sum()

        # Get regret matching strategy
        if info_state_key in self.cumulative_regrets:
            regrets = self.cumulative_regrets[info_state_key]
            # Regret matching: σ(a) = max(r(a), 0) / Σ_b max(r(b), 0)
            positive_regrets = np.maximum(regrets[legal_actions], 0)
            if np.sum(positive_regrets) > 0:
                regret_probs = np.zeros(len(policy_probs))
                regret_probs[legal_actions] = positive_regrets / np.sum(
                    positive_regrets
                )
            else:
                # If no positive regrets, use uniform distribution over legal actions
                regret_probs = np.zeros(len(policy_probs))
                regret_probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            # No regrets yet, use uniform distribution
            regret_probs = np.zeros(len(policy_probs))
            regret_probs[legal_actions] = 1.0 / len(legal_actions)

        # Adaptive regret weight based on training progress
        adaptive_regret_weight = self.regret_weight * min(
            1.0, self.iteration_count / 1000.0
        )

        # Combine policy and regret matching using adaptive weighting
        combined_probs = (
            1 - adaptive_regret_weight
        ) * policy_probs + adaptive_regret_weight * regret_probs

        # Apply exploration noise (decreasing over time)
        noise_scale = self._get_current_noise_scale()
        if noise_scale > 0.01:  # Only add noise when it's significant
            noise = np.random.normal(0, noise_scale, len(combined_probs))
            noisy_probs = combined_probs + noise
            # Ensure non-negative and renormalize
            final_probs = np.maximum(noisy_probs * legal_mask, 0)
        else:
            final_probs = combined_probs * legal_mask

        # Final normalization
        if final_probs.sum() > 0:
            final_probs = final_probs / final_probs.sum()
        else:
            # Uniform fallback
            final_probs = legal_mask / legal_mask.sum()

        return final_probs

    def _legal_actions_mask(self, legal_actions: List[int]) -> np.ndarray:
        mask = np.zeros(self.game_wrapper.num_actions, dtype=np.float32)
        mask[legal_actions] = 1.0
        return mask

    def _get_current_noise_scale(self) -> float:
        """Get current exploration noise scale.

        Returns:
            Current noise scale
        """
        decay_progress = min(self.total_steps, self.noise_decay_steps)
        return self.initial_noise_scale * (
            self.final_noise_scale / self.initial_noise_scale
        ) ** (decay_progress / self.noise_decay_steps)

    def _assign_armac_returns(
        self, trajectory: List[Dict[str, Any]], final_rewards: Dict[int, float]
    ):
        """Assign returns and advantages to trajectory transitions using proper TD and advantage computation.

        Args:
            trajectory: List of trajectory transitions
            final_rewards: Final rewards for each player
        """
        # Process trajectory in reverse order for proper TD bootstrapping
        next_states = {p: None for p in range(self.game_wrapper.num_players)}
        next_values = {p: 0.0 for p in range(self.game_wrapper.num_players)}

        for transition in reversed(trajectory):
            player = transition["player"]
            info_state = transition["info_state"]
            action = transition["action"]
            legal_actions = transition["legal_actions"]

            # Store immediate reward
            transition["reward"] = final_rewards[player]

            # Get current value estimate for this state
            with torch.no_grad():
                info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
                value_output = self.critic(info_tensor)
                current_value = value_output["value"].item()

            # Store current value
            transition["value"] = current_value

            # Calculate TD target: r + γ * V(s')
            td_target = final_rewards[player] + self.gamma * next_values[player]

            # Calculate TD error: δ = r + γ * V(s') - V(s)
            transition["td_error"] = td_target - current_value

            # Calculate advantage: A = Q(s,a) - V(s)
            # We use TD target as Q estimate for simplicity
            transition["advantage"] = td_target - current_value

            # Update next value for next iteration
            next_values[player] = current_value

            # Store legal actions mask
            legal_mask = np.zeros(self.game_wrapper.num_actions)
            legal_mask[legal_actions] = 1.0
            transition["legal_actions_mask"] = legal_mask

            # One-hot encode action
            action_one_hot = np.zeros(self.game_wrapper.num_actions)
            action_one_hot[action] = 1.0
            transition["action_one_hot"] = action_one_hot

            # Calculate regrets for regret network update
            self._update_regrets(transition, final_rewards[player])

        self.total_steps += len(trajectory)

    def _update_regrets(self, transition: Dict[str, Any], reward: float):
        """Update cumulative regrets using proper counterfactual regret computation.

        Args:
            transition: Trajectory transition
            reward: Reward received
        """
        info_state_key = transition["info_state_key"]
        legal_actions = transition["legal_actions"]
        action = transition["action"]
        value = transition["value"]  # Current value estimate from critic
        advantage = transition["advantage"]  # Advantage computed earlier

        # Compute counterfactual regrets
        # r_i(a) = Q_i(s,a) - V_i(s) = advantage for taken action
        # For other actions, we need to estimate counterfactual values

        regrets = np.zeros(self.game_wrapper.num_actions)

        # For the taken action: use the advantage directly
        if advantage is not None:
            regrets[action] = advantage
        else:
            # Fallback to simple reward-based regret
            regrets[action] = reward - value

        # For other legal actions: estimate counterfactual regrets
        # This is simplified - in full CFR, we would need to traverse game tree
        for legal_action in legal_actions:
            if legal_action != action:
                # Estimate regret for unchosen actions
                # Since we don't have true counterfactual values, use a fraction of negative reward
                # This encourages exploration of alternatives
                if reward > 0:
                    # If we got a positive reward, other actions might have been worse
                    regrets[legal_action] = -reward * 0.3
                else:
                    # If we got a negative reward, other actions might have been better
                    regrets[legal_action] = abs(reward) * 0.3

        # Apply positive regret transformation (regret matching only uses positive regrets)
        positive_regrets = np.maximum(regrets, 0)

        # Update cumulative regrets with decay for stability
        decay_factor = 0.99  # Regret decay for stability
        self.cumulative_regrets[info_state_key] = (
            self.cumulative_regrets[info_state_key] * decay_factor + positive_regrets
        )
        self.regret_counts[info_state_key] += 1

    def _update_regret_network(self) -> Dict[str, float]:
        """Update regret network using stored regrets.

        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "gradient_norm": 0.0}

        # Sample batch
        batch = random.sample(
            list(self.replay_buffer), min(self.batch_size, len(self.replay_buffer))
        )

        # Prepare training data
        info_states = torch.FloatTensor([t["info_state"] for t in batch])
        legal_actions_mask = torch.FloatTensor([t["legal_actions_mask"] for t in batch])

        # Calculate target regrets from cumulative regrets
        target_regrets = []
        for transition in batch:
            info_state_key = transition["info_state_key"]
            if info_state_key in self.cumulative_regrets:
                regrets = self.cumulative_regrets[info_state_key]
                # Apply positive transformation and normalize
                positive_regrets = np.maximum(regrets, 0)
                if np.sum(positive_regrets) > 0:
                    target_regrets.append(positive_regrets / np.sum(positive_regrets))
                else:
                    target_regrets.append(np.ones(len(regrets)) / len(regrets))
            else:
                target_regrets.append(
                    np.ones(self.game_wrapper.num_actions)
                    / self.game_wrapper.num_actions
                )

        target_regrets = torch.FloatTensor(np.array(target_regrets))

        # Update regret network
        self.regret_optimizer.zero_grad()
        regret_output = self.regret_network(info_states)
        predicted_regrets = regret_output["action_probs"]

        # Apply legal actions mask
        masked_pred = predicted_regrets * legal_actions_mask

        # Calculate loss
        regret_loss = F.mse_loss(masked_pred, target_regrets)

        # Backward pass
        regret_loss.backward()
        training_cfg = self.config.get("training", {})
        gradient_clip = training_cfg.get("gradient_clip", self.gradient_clip)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.regret_network.parameters(), gradient_clip
        )
        self.regret_optimizer.step()

        return {"loss": regret_loss.item(), "gradient_norm": gradient_norm.item()}

    def _update_critic_network(self) -> Dict[str, float]:
        """Update critic network using TD learning.

        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "gradient_norm": 0.0}

        # Sample batch
        batch = random.sample(
            list(self.replay_buffer), min(self.batch_size, len(self.replay_buffer))
        )

        # Prepare training data
        info_states = torch.FloatTensor([t["info_state"] for t in batch])
        rewards = torch.FloatTensor([t["reward"] for t in batch])
        td_errors = torch.FloatTensor([t["td_error"] for t in batch])

        # Get current value estimates
        self.critic_optimizer.zero_grad()
        value_output = self.critic(info_states)
        predicted_values = value_output["value"].squeeze()

        # Calculate TD loss
        target_values = rewards + self.gamma * td_errors
        critic_loss = F.mse_loss(predicted_values, target_values.detach())

        # Backward pass
        critic_loss.backward()
        training_cfg = self.config.get("training", {})
        gradient_clip = training_cfg.get("gradient_clip", self.gradient_clip)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), gradient_clip
        )
        self.critic_optimizer.step()

        return {"loss": critic_loss.item(), "gradient_norm": gradient_norm.item()}

    def _update_actor_network(self) -> Dict[str, float]:
        """Update actor network using advantage-based policy gradient with regret guidance.

        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "gradient_norm": 0.0}

        # Sample batch
        batch = random.sample(
            list(self.replay_buffer), min(self.batch_size, len(self.replay_buffer))
        )

        # Prepare training data
        info_states = torch.FloatTensor([t["info_state"] for t in batch])
        legal_actions_mask = torch.FloatTensor([t["legal_actions_mask"] for t in batch])
        actions = torch.LongTensor([t["action"] for t in batch])

        # Use ARMAC dual RL for proper advantage computation
        advantages = self.armac_dual_rl.compute_advantages(
            info_states, self.critic, self.actor, legal_actions_mask
        )

        # Get selected advantages
        selected_advantages = advantages.gather(1, actions.unsqueeze(1)).squeeze()

        # Get policy from actor
        self.actor_optimizer.zero_grad()
        actor_output = self.actor(info_states)
        action_probs = actor_output["action_probs"]

        # Apply legal actions mask and normalize
        masked_probs = action_probs * legal_actions_mask
        normalized_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Calculate policy gradient loss using advantages
        selected_probs = normalized_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_gradient_loss = -(
            torch.log(selected_probs + 1e-8) * selected_advantages.detach()
        ).mean()

        # Add entropy regularization
        entropy = (
            -(normalized_probs * torch.log(normalized_probs + 1e-8)).sum(dim=1).mean()
        )
        entropy_bonus = 0.01 * entropy

        # Add regret matching loss
        regret_policy = self.armac_dual_rl.regret_matching_policy(
            advantages, legal_actions_mask
        )
        regret_loss = F.mse_loss(normalized_probs, regret_policy.detach())
        regret_loss *= 0.1  # Scale down regret loss

        # Total actor loss
        actor_loss = policy_gradient_loss - entropy_bonus + regret_loss

        # Backward pass
        actor_loss.backward()
        training_cfg = self.config.get("training", {})
        gradient_clip = training_cfg.get("gradient_clip", self.gradient_clip)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), gradient_clip
        )
        self.actor_optimizer.step()

        return {
            "loss": actor_loss.item(),
            "gradient_norm": gradient_norm.item(),
            "policy_gradient_loss": policy_gradient_loss.item(),
            "entropy_bonus": entropy_bonus.item(),
            "regret_loss": regret_loss.item(),
        }

    def get_policy(self, player: int) -> callable:
        """Get the current policy for a player.

        Args:
            player: Player index

        Returns:
            Policy function
        """

        def policy(info_state_key: str, legal_actions: List[int]) -> np.ndarray:
            # Get information state encoding
            info_state = self.game_wrapper.encode_info_state_key(info_state_key, player)

            # Get action probabilities from actor (no exploration during evaluation)
            with torch.no_grad():
                info_tensor = torch.FloatTensor(info_state).unsqueeze(0)
                actor_output = self.actor(info_tensor)
                action_probs = actor_output["action_probs"].squeeze().cpu().numpy()

            # Apply legal action mask
            legal_mask = np.zeros(len(action_probs))
            legal_mask[legal_actions] = 1.0

            # Normalize for legal actions
            masked_probs = action_probs * legal_mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                masked_probs = legal_mask / legal_mask.sum()

            return masked_probs

        return policy

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """Get the average strategy (uses current actor policy).

        Returns:
            Dictionary mapping info state keys to action probabilities
        """
        # For ARMAC, we use the current actor policy as the average strategy
        avg_strategy = {}

        # Sample some common information states from the replay buffer
        if len(self.replay_buffer) > 0:
            sample_transitions = random.sample(
                list(self.replay_buffer), min(1000, len(self.replay_buffer))
            )

            for transition in sample_transitions:
                info_state_key = transition["info_state_key"]
                if info_state_key not in avg_strategy:
                    legal_actions = transition["legal_actions"]
                    action_probs = self.get_policy(transition["player"])(
                        info_state_key, legal_actions
                    )
                    avg_strategy[info_state_key] = action_probs

        return avg_strategy

    def get_regrets(self) -> Dict[str, np.ndarray]:
        """Get current regrets.

        Returns:
            Dictionary mapping info state keys to regret values
        """
        return dict(self.cumulative_regrets)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the current policy.

        Returns:
            Evaluation metrics
        """
        from eval.openspiel_evaluator import OpenSpielExactEvaluator

        evaluator = OpenSpielExactEvaluator(self.game_wrapper.game_name)
        metadata = PolicyMetadata(method="armac", iteration=self.iteration_count)

        def policy_fn(
            player_id: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            return self.get_policy(player_id)(info_state, list(legal_actions))

        policy_adapter = evaluator.build_policy(policy_fn, metadata=metadata)
        result = evaluator.evaluate(policy_adapter, metadata=metadata)

        return {
            "nash_conv": result.nash_conv,
            "exploitability": result.exploitability,
            "player_0_value": result.player_0_value,
            "player_1_value": result.player_1_value,
            "mean_value": result.mean_value,
            "info_state_count": result.info_state_count,
            "openspiel_version": result.openspiel_version,
            "python_version": result.python_version,
        }

    def save_checkpoint(self, path: str):
        """Save algorithm checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "iteration": self.iteration_count,
            "total_steps": self.total_steps,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_target_state": self.actor_target.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "regret_network_state": self.regret_network.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "regret_optimizer_state": self.regret_optimizer.state_dict(),
            "cumulative_regrets": dict(self.cumulative_regrets),
            "regret_counts": dict(self.regret_counts),
            "config": self.config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved ARMAC checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load algorithm checkpoint.

        Args:
            path: Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")

        self.iteration_count = checkpoint["iteration"]
        self.total_steps = checkpoint.get("total_steps", 0)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.regret_network.load_state_dict(checkpoint["regret_network_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
        self.regret_optimizer.load_state_dict(checkpoint["regret_optimizer_state"])
        self.cumulative_regrets = defaultdict(
            lambda: np.zeros(self.game_wrapper.num_actions),
            checkpoint["cumulative_regrets"],
        )
        self.regret_counts = defaultdict(int, checkpoint["regret_counts"])

        self.logger.info(
            f"Loaded ARMAC checkpoint from {path} (iteration {self.iteration_count})"
        )

    def get_policy_adapter(self):
        """Return a PolicyAdapter instance for the current averaged strategy."""

        def policy_fn(
            player: int, info_state: str, legal_actions: List[int]
        ) -> np.ndarray:
            del player
            strategy = self.get_average_strategy().get(info_state)
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
            policy_fn,
            metadata=PolicyMetadata(method="armac", iteration=self.iteration_count),
        )
