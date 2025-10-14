"""Comprehensive scheduler training module with meta-regret integration.

This module implements the complete training loop for the discrete scheduler,
including utility computation, meta-regret updates, and KL loss optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import time

from algs.scheduler.meta_regret import MetaRegretManager
from algs.scheduler.utils.utility_computation import create_utility_computer
from algs.scheduler.utils.state_keying import create_state_key_manager


class SchedulerTrainer:
    """Comprehensive trainer for discrete scheduler with meta-regret learning."""

    def __init__(
        self,
        scheduler: torch.nn.Module,
        meta_regret: MetaRegretManager,
        config: Dict[str, Any],
        state_key_manager: Optional[Any] = None,
        utility_computer: Optional[Any] = None,
    ):
        """Initialize scheduler trainer.

        Args:
            scheduler: Scheduler network
            meta_regret: Meta-regret manager
            config: Training configuration
            state_key_manager: Optional state key manager
            utility_computer: Optional utility computer
        """
        self.scheduler = scheduler
        self.meta_regret = meta_regret
        self.config = config

        # Initialize components
        if state_key_manager is None:
            state_key_manager = create_state_key_manager(config)
        self.state_key_manager = state_key_manager

        if utility_computer is None:
            utility_computer = create_utility_computer(config)
        self.utility_computer = utility_computer

        # Training parameters
        self.scheduler_lr = config.get("scheduler_lr", 1e-4)
        self.use_scheduler = scheduler is not None and scheduler.discrete

        # Gumbel-softmax parameters
        self.gumbel_tau_start = config.get("gumbel_tau_start", 1.0)
        self.gumbel_tau_end = config.get("gumbel_tau_end", 0.1)
        self.gumbel_anneal_iters = config.get("gumbel_anneal_iters", 50000)

        # Optimizer
        if self.use_scheduler:
            self.scheduler_optimizer = torch.optim.Adam(
                self.scheduler.parameters(), lr=self.scheduler_lr
            )

        # Training statistics
        self.training_stats = {
            "scheduler_loss": [],
            "meta_regret_updates": 0,
            "utility_signals": [],
            "temperature_history": [],
            "lambda_entropy": [],
        }

        self.current_iteration = 0

    def update_temperature(self):
        """Update Gumbel-softmax temperature with annealing."""
        if self.current_iteration >= self.gumbel_anneal_iters:
            tau = self.gumbel_tau_end
        else:
            progress = self.current_iteration / self.gumbel_anneal_iters
            tau = (
                self.gumbel_tau_start * (1 - progress) + self.gumbel_tau_end * progress
            )

        self.scheduler.set_temperature(tau)
        self.training_stats["temperature_history"].append(tau)
        return tau

    def process_trajectory_batch(
        self,
        trajectories: List[List[Dict[str, Any]]],
        scheduler_outputs: List[List[Dict[str, Any]]],
        state_encodings: List[List[torch.Tensor]],
        iteration: int,
    ) -> Dict[str, Any]:
        """Process a batch of trajectories for scheduler training.

        Args:
            trajectories: List of episode trajectories
            scheduler_outputs: List of scheduler outputs per episode
            state_encodings: List of state encodings per episode
            iteration: Current training iteration

        Returns:
            Dictionary with training statistics
        """
        if not self.use_scheduler:
            return {"scheduler_loss": 0.0, "meta_regret_updates": 0}

        batch_stats = {
            "scheduler_loss": 0.0,
            "meta_regret_updates": 0,
            "utility_signals_processed": 0,
            "lambda_entropy": 0.0,
        }

        # Update temperature
        self.update_temperature()
        self.current_iteration = iteration

        # Collect scheduler data for meta-regret updates
        all_state_keys = []
        all_k_choices = []
        all_utilities = []
        all_scheduler_logits = []

        # Process each trajectory
        for traj_idx, (trajectory, sched_outputs, state_encs) in enumerate(
            zip(trajectories, scheduler_outputs, state_encodings)
        ):
            for step_idx, (step, sched_out, state_enc) in enumerate(
                zip(trajectory, sched_outputs, state_encs)
            ):
                if sched_out["mode"] != "discrete":
                    continue

                # Compute utility for this decision
                try:
                    utility = self.utility_computer.compute_scheduler_utility(
                        trajectory=trajectory,
                        decision_index=step_idx,
                        critic_network=None,  # Could pass critic if available
                        state_encoding_fn=lambda s: state_enc,
                    )
                except Exception as e:
                    print(f"Warning: Utility computation failed: {e}")
                    utility = 0.0

                # Compute state key
                state_info = {
                    "embedding": state_enc,
                    "round": step.get("round", 0),
                    "player_pos": step.get("player_pos", 0),
                    "pot": step.get("pot", 0),
                    "stack_ratio": step.get("stack_ratio", 0),
                }
                state_key = self.state_key_manager(state_info)

                # Get discrete choice
                k_choice = sched_out.get("lambda_idx", 0)
                if isinstance(k_choice, torch.Tensor):
                    k_choice = k_choice.item()

                # Store for meta-regret update
                all_state_keys.append(state_key)
                all_k_choices.append(k_choice)
                all_utilities.append(utility)
                all_scheduler_logits.append(sched_out["logits"])

                batch_stats["utility_signals_processed"] += 1

        # Update meta-regret with all collected data
        for state_key, k_choice, utility in zip(
            all_state_keys, all_k_choices, all_utilities
        ):
            update_stats = self.meta_regret.record(
                state_key=state_key,
                k_choice=k_choice,
                utility=utility,
                learning_rate=0.1,  # Could be configurable
            )
            batch_stats["meta_regret_updates"] += 1

        # Compute scheduler training loss if we have logits
        if all_scheduler_logits:
            scheduler_loss = self._compute_scheduler_loss(
                all_scheduler_logits, all_state_keys
            )
            batch_stats["scheduler_loss"] = scheduler_loss.item()

            # Update scheduler
            self.scheduler_optimizer.zero_grad()
            scheduler_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.scheduler.parameters(), max_norm=1.0)

            self.scheduler_optimizer.step()

            # Compute entropy for logging
            with torch.no_grad():
                avg_entropy = self._compute_average_entropy(all_scheduler_logits)
                batch_stats["lambda_entropy"] = avg_entropy

        # Update training statistics
        self.training_stats["scheduler_loss"].append(batch_stats["scheduler_loss"])
        self.training_stats["meta_regret_updates"] += batch_stats["meta_regret_updates"]
        self.training_stats["utility_signals"].extend(all_utilities)
        self.training_stats["lambda_entropy"].append(batch_stats["lambda_entropy"])

        return batch_stats

    def _compute_scheduler_loss(
        self, scheduler_logits: List[torch.Tensor], state_keys: List[Any]
    ) -> torch.Tensor:
        """Compute KL loss for scheduler using desired probabilities from meta-regret.

        Args:
            scheduler_logits: List of scheduler logits from forward pass
            state_keys: Corresponding state keys

        Returns:
            KL divergence loss
        """
        if not scheduler_logits:
            return torch.tensor(0.0, device=next(self.scheduler.parameters()).device)

        # Stack logits
        all_logits = torch.stack(scheduler_logits)  # [B, K]
        device = all_logits.device

        # Get desired probabilities from meta-regret
        desired_probs_list = []
        for state_key in state_keys:
            desired_probs = self.meta_regret.get_action_probs(state_key)
            desired_probs_list.append(torch.tensor(desired_probs, dtype=torch.float32))

        if not desired_probs_list:
            return torch.tensor(0.0, device=device)

        desired_probs_batch = torch.stack(desired_probs_list).to(device)  # [B, K]

        # Ensure desired_probs sums to 1
        desired_probs_batch = desired_probs_batch / (
            desired_probs_batch.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Compute KL divergence loss
        log_probs = F.log_softmax(all_logits, dim=-1)
        loss = F.kl_div(log_probs, desired_probs_batch, reduction="batchmean")

        # Add regularization if configured
        reg_loss = self.scheduler.compute_regularization_loss()
        total_loss = loss + reg_loss

        return total_loss

    def _compute_average_entropy(self, scheduler_logits: List[torch.Tensor]) -> float:
        """Compute average entropy of scheduler policy.

        Args:
            scheduler_logits: List of scheduler logits

        Returns:
            Average entropy
        """
        if not scheduler_logits:
            return 0.0

        entropies = []
        for logits in scheduler_logits:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropies.append(entropy.mean().item())

        return np.mean(entropies) if entropies else 0.0

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dictionary with training statistics
        """
        stats = {
            "current_iteration": self.current_iteration,
            "training_stats": self.training_stats.copy(),
            "current_temperature": self.scheduler.temperature
            if self.use_scheduler
            else 0.0,
            "meta_regret_stats": self.meta_regret.get_global_stats(),
        }

        # Add recent averages
        if self.training_stats["scheduler_loss"]:
            recent_losses = self.training_stats["scheduler_loss"][-100:]
            stats["recent_avg_loss"] = np.mean(recent_losses)
            stats["recent_std_loss"] = np.std(recent_losses)

        if self.training_stats["utility_signals"]:
            recent_utilities = self.training_stats["utility_signals"][-1000:]
            stats["recent_avg_utility"] = np.mean(recent_utilities)
            stats["recent_std_utility"] = np.std(recent_utilities)

        if self.training_stats["lambda_entropy"]:
            recent_entropies = self.training_stats["lambda_entropy"][-100:]
            stats["recent_avg_entropy"] = np.mean(recent_entropies)

        return stats

    def save_checkpoint(self, filepath: str, include_meta_regret: bool = True):
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            include_meta_regret: Whether to include meta-regret state
        """
        checkpoint = {
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.use_scheduler
            else None,
            "scheduler_optimizer_state_dict": (
                self.scheduler_optimizer.state_dict() if self.use_scheduler else None
            ),
            "training_stats": self.training_stats,
            "current_iteration": self.current_iteration,
            "config": self.config,
        }

        if include_meta_regret:
            meta_regret_path = filepath.replace(".pt", "_meta_regret.json")
            self.meta_regret.save_state(meta_regret_path)
            checkpoint["meta_regret_path"] = meta_regret_path

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, include_meta_regret: bool = True):
        """Load training checkpoint.

        Args:
            filepath: Path to load checkpoint
            include_meta_regret: Whether to load meta-regret state
        """
        checkpoint = torch.load(filepath, map_location="cpu")

        # Load scheduler state
        if self.use_scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scheduler_optimizer.load_state_dict(
                checkpoint["scheduler_optimizer_state_dict"]
            )

        # Load training stats
        self.training_stats = checkpoint["training_stats"]
        self.current_iteration = checkpoint["current_iteration"]

        # Load meta-regret state
        if include_meta_regret and "meta_regret_path" in checkpoint:
            try:
                self.meta_regret.load_state(checkpoint["meta_regret_path"])
            except Exception as e:
                print(f"Warning: Failed to load meta-regret state: {e}")

    def reset_stats(self):
        """Reset training statistics."""
        self.training_stats = {
            "scheduler_loss": [],
            "meta_regret_updates": 0,
            "utility_signals": [],
            "temperature_history": [],
            "lambda_entropy": [],
        }
        self.current_iteration = 0


def create_scheduler_trainer(
    scheduler: torch.nn.Module,
    meta_regret: MetaRegretManager,
    config: Dict[str, Any],
) -> SchedulerTrainer:
    """Factory function to create scheduler trainer.

    Args:
        scheduler: Scheduler network
        meta_regret: Meta-regret manager
        config: Configuration dictionary

    Returns:
        SchedulerTrainer instance
    """
    return SchedulerTrainer(
        scheduler=scheduler,
        meta_regret=meta_regret,
        config=config,
    )
