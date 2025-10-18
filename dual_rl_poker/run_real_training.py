"""Neural-style ARMAC training loop using pure Python data structures.

This script trains actor/critic/regret tables with adaptive λ on Kuhn or Leduc
Poker. It produces real trajectories, adapts λ with a logistic scheduler, and
computes exploitability via OpenSpiel's tabular utilities. The implementation
avoids third-party numeric libraries so it remains portable in restricted
environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import pyspiel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

from algs.cfr.cfr_head import TabularCFRHead
from algs.meta.meta_objective import MetaBatchItem, MetaObjective
from algs.scheduler.gate import ExpertGate, gate_context
from algs.scheduler.gate_bandit import GateBandit
from algs.scheduler.scheduler import Scheduler, compute_scheduler_input
from dual_rl_poker.tools.approx_br import ApproxBestResponse
from utils.manifest_manager import ManifestManager


@dataclass
class EpisodeStep:
    info_state: str
    legal_actions: Tuple[int, ...]
    action: int
    player: int


@dataclass
class Episode:
    steps: List[EpisodeStep]
    returns: Tuple[float, float]
    lambdas: List[float]


@dataclass
class IterationMetrics:
    iteration: int
    exploitability: float
    nash_conv: float
    actor_loss: float
    average_lambda: float
    mean_episode_length: float
    mean_return_player0: float
    mean_return_player1: float
    wall_time_sec: float
    scheduler_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "iteration": self.iteration,
            "exploitability": self.exploitability,
            "nash_conv": self.nash_conv,
            "actor_loss": self.actor_loss,
            "average_lambda": self.average_lambda,
            "mean_episode_length": self.mean_episode_length,
            "mean_return_player0": self.mean_return_player0,
            "mean_return_player1": self.mean_return_player1,
            "wall_time_sec": self.wall_time_sec,
            "scheduler_loss": self.scheduler_loss,
        }


@dataclass
class GateTrace:
    state_idx: int
    cluster: str
    features: torch.Tensor
    probs: torch.Tensor
    expert_policies: List[Dict[int, float]]
    legal_actions: Tuple[int, ...]


def masked_softmax(logits: Dict[int, float], legal_actions: Sequence[int]) -> Dict[int, float]:
    if not legal_actions:
        return {}
    max_logit = max(logits[a] for a in legal_actions)
    exp_vals = [math.exp(logits[a] - max_logit) for a in legal_actions]
    denom = sum(exp_vals) + 1e-8
    return {a: val / denom for a, val in zip(legal_actions, exp_vals)}


class AdamTable:
    def __init__(self, rows: int, cols: int, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [[0.0 for _ in range(cols)] for _ in range(rows)]
        self.v = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def step(self, params: List[List[float]], grads: List[List[float]]) -> None:
        self.t += 1
        lr = self.lr
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        t = self.t
        for r in range(len(params)):
            row_params = params[r]
            row_grads = grads[r]
            row_m = self.m[r]
            row_v = self.v[r]
            for c in range(len(row_params)):
                g = row_grads[c]
                row_m[c] = b1 * row_m[c] + (1 - b1) * g
                row_v[c] = b2 * row_v[c] + (1 - b2) * (g * g)
                m_hat = row_m[c] / (1 - b1 ** t)
                v_hat = row_v[c] / (1 - b2 ** t)
                row_params[c] -= lr * m_hat / (math.sqrt(v_hat) + eps)


class NeuralARMACTrainer:
    def __init__(
        self,
        game_name: str,
        *,
        seed: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        regret_lr: float = 1e-3,
        episodes_per_iteration: int = 64,
        buffer_size: int = 50000,
        batch_size: int = 256,
        lambda_alpha: float = 2.0,
        loss_beta: float = 0.7,
        backend: str = "pyspiel",
        experts: Sequence[str] = ("actor", "regret", "ra", "explore", "cfr"),
        meta_unroll: int = 16,
        br_budget: int = 64,
        gate_lr: float = 1e-3,
        gate_kl_weight: float = 0.1,
        handoff_tau: float = 0.15,
        handoff_patience: int = 3,
        state_cluster: str = "round+position+pot",
    ) -> None:
        if game_name not in {"kuhn_poker", "leduc_poker"}:
            raise ValueError("Supported games: kuhn_poker, leduc_poker")

        self.game = pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError("Only two-player zero-sum games are supported")

        random.seed(seed)
        self.backend = backend
        self.use_rust_backend = backend == "rust"
        self.game_name = game_name
        self._rust_utils = None
        self._rust_env = None
        self.device = torch.device("cpu")
        if self.use_rust_backend:
            try:
                from utils import rust_env as rust_env_utils
            except ImportError as exc:
                raise ImportError(
                    "Rust backend requested but utils.rust_env is unavailable. "
                    "Build the Rust module before using --backend rust."
                ) from exc
            self._rust_utils = rust_env_utils
            self._rust_env = rust_env_utils.RustEnvWrapper(game_name, seed=seed)

        self.info_state_encoding_cache: Dict[str, List[float]] = {}
        self.info_states, self.info_state_actions = self._enumerate_info_states()
        self.info_state_index = {s: i for i, s in enumerate(self.info_states)}
        self.num_states = len(self.info_states)
        self.num_actions = self.game.num_distinct_actions()

        # Parameter tables (logits / values)
        self.actor_logits = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        self.critic_table = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        self.regret_table = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]

        self.actor_opt = AdamTable(self.num_states, self.num_actions, lr=actor_lr)
        self.critic_opt = AdamTable(self.num_states, self.num_actions, lr=critic_lr)
        self.regret_opt = AdamTable(self.num_states, self.num_actions, lr=regret_lr)

        self.episodes_per_iteration = episodes_per_iteration
        self.buffer: Deque[Dict[str, object]] = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.lambda_alpha = lambda_alpha
        self.loss_beta = loss_beta
        self.current_lambda = 0.5
        self.avg_actor_loss = 0.0
        self.avg_regret_loss = 0.0
        self.avg_scheduler_loss = 0.0
        self.scheduler_kappa = 5.0
        self.scheduler_training_buffer: List[Tuple[torch.Tensor, float]] = []
        self.iteration_count = 0

        # Expert gate configuration -------------------------------------------------
        expert_list = [exp.strip() for exp in experts if exp and exp.strip()]
        if not expert_list:
            raise ValueError("At least one expert must be provided for MARM-K gate")
        self.expert_names: Tuple[str, ...] = tuple(expert_list)
        self.num_experts = len(self.expert_names)
        self.actor_index = self.expert_names.index("actor") if "actor" in self.expert_names else None
        self.regret_index = self.expert_names.index("regret") if "regret" in self.expert_names else None
        self.cfr_index = self.expert_names.index("cfr") if "cfr" in self.expert_names else None
        self.risk_temperature = 0.5
        self.explore_epsilon = 0.2
        self.meta_unroll = meta_unroll

        dummy_features = self._build_gate_features(self.info_states[0])
        self.gate = ExpertGate(
            input_dim=dummy_features.numel(),
            num_experts=self.num_experts,
            temperature=1.0,
            entropy_reg=1e-3,
        )
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=gate_lr)
        self.gate_bandit = GateBandit(self.num_experts, device=self.device)
        self.approx_br = ApproxBestResponse(self.game, budget=br_budget, seed=seed)
        self.meta_objective = MetaObjective(self.approx_br, kl_weight=gate_kl_weight)
        self.gate_traces: List[GateTrace] = []
        self.state_cluster_spec = state_cluster
        self.iteration_clusters: set[str] = set()
        self.cluster_history: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=handoff_patience))
        self.handoff_tau = handoff_tau
        self.handoff_patience = handoff_patience
        self.frozen_clusters: Dict[str, int] = {}
        self.cfr_head = TabularCFRHead(self.num_states, self.num_actions) if self.cfr_index is not None else None

        # Backwards compatibility scheduler (kept for lambda diagnostics) -----------
        state_dim = len(next(iter(self.info_state_encoding_cache.values())))
        dummy_state = torch.zeros(1, state_dim, dtype=torch.float32, device=self.device)
        dummy_actor = torch.zeros(1, self.num_actions, dtype=torch.float32, device=self.device)
        dummy_regret = torch.zeros(1, self.num_actions, dtype=torch.float32, device=self.device)
        dummy_input = compute_scheduler_input(dummy_state, dummy_actor, dummy_regret, iteration=0)
        self.scheduler_input_dim = dummy_input.shape[-1]
        self.scheduler = Scheduler(input_dim=self.scheduler_input_dim)
        self.scheduler.train()
        self.scheduler_optimizer = torch.optim.Adam(self.scheduler.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Game traversal helpers
    # ------------------------------------------------------------------
    def _enumerate_info_states(self) -> Tuple[List[str], Dict[str, Tuple[int, ...]]]:
        cache: Dict[str, Tuple[int, ...]] = {}
        stack = [self.game.new_initial_state()]
        while stack:
            state = stack.pop()
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    stack.append(state.child(action))
                continue
            player = state.current_player()
            info_state = state.information_state_string(player)
            if info_state not in cache:
                cache[info_state] = tuple(state.legal_actions())
                tensor = state.information_state_tensor(player)
                self.info_state_encoding_cache[info_state] = list(tensor)
            for action in state.legal_actions():
                stack.append(state.child(action))
        info_states = sorted(cache.keys())
        return info_states, cache

    def _build_gate_features(self, info_state: str) -> torch.Tensor:
        state_idx = self.info_state_index[info_state]
        state_tensor = torch.tensor(
            self.info_state_encoding_cache[info_state], dtype=torch.float32, device=self.device
        )
        actor_logits_tensor = torch.tensor(
            self.actor_logits[state_idx], dtype=torch.float32, device=self.device
        )
        regret_tensor = torch.tensor(
            self.regret_table[state_idx], dtype=torch.float32, device=self.device
        )
        scheduler_input = compute_scheduler_input(
            state_tensor.unsqueeze(0),
            actor_logits_tensor.unsqueeze(0),
            regret_tensor.unsqueeze(0),
            self.iteration_count,
        )
        feature_dict = {
            "state": state_tensor,
            "scheduler": scheduler_input.squeeze(0),
            "lambda": torch.tensor([self.current_lambda], dtype=torch.float32, device=self.device),
            "iter": torch.tensor([self.iteration_count / max(1, self.meta_unroll)], dtype=torch.float32, device=self.device),
        }
        return gate_context(feature_dict)

    def _cluster_key(self, info_state: str) -> str:
        if self.state_cluster_spec == "round+position+pot":
            tokens = info_state.split(" ")
            prefix = tokens[0] if tokens else info_state
            player = tokens[1] if len(tokens) > 1 else "p"
            bucket = tokens[2] if len(tokens) > 2 else "b"
            return f"{prefix}:{player}:{bucket}"
        return info_state.split(":", 1)[0]

    def _gate_forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.gate(features).logits

    def _expert_policies(
        self, state_idx: int, info_state: str, legal_actions: Sequence[int]
    ) -> Dict[str, Dict[int, float]]:
        actor_logits = self.actor_logits[state_idx]
        actor_policy = masked_softmax(actor_logits, legal_actions)

        regrets = self.regret_table[state_idx]
        positive = {a: max(regrets[a], 0.0) for a in legal_actions}
        pos_sum = sum(positive.values())
        if pos_sum > 1e-8:
            regret_policy = {a: val / pos_sum for a, val in positive.items()}
        else:
            regret_policy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        temp_logits = {a: actor_logits[a] / max(self.risk_temperature, 1e-3) for a in legal_actions}
        risk_policy = masked_softmax(temp_logits, legal_actions)

        explore_policy = {
            a: (1 - self.explore_epsilon) * actor_policy[a] + self.explore_epsilon / len(legal_actions)
            for a in legal_actions
        }

        policies: Dict[str, Dict[int, float]] = {
            "actor": actor_policy,
            "regret": regret_policy,
            "ra": risk_policy,
            "explore": explore_policy,
        }

        if self.cfr_head is not None:
            policies["cfr"] = self.cfr_head.policy(state_idx, legal_actions)

        return {name: policies[name] for name in self.expert_names if name in policies}

    def _compute_gate_utilities(self, trace: GateTrace) -> torch.Tensor:
        critic_values = self.critic_table[trace.state_idx]
        utilities = []
        for policy in trace.expert_policies:
            value = sum(policy[a] * critic_values[a] for a in trace.legal_actions)
            utilities.append(value)
        mixed_value = sum(
            trace.probs[idx].item() * sum(policy[a] * critic_values[a] for a in trace.legal_actions)
            for idx, policy in enumerate(trace.expert_policies)
        )
        tensor = torch.tensor(utilities, dtype=torch.float32)
        tensor -= mixed_value
        return tensor

    def _update_gate(self) -> float:
        if not self.gate_traces:
            return 0.0

        bandit_targets: Dict[str, torch.Tensor] = {}
        meta_batch: List[MetaBatchItem] = []
        for trace in self.gate_traces:
            utilities = self._compute_gate_utilities(trace)
            self.gate_bandit.observe(trace.cluster, utilities)

        for trace in self.gate_traces:
            if trace.cluster not in bandit_targets:
                bandit_targets[trace.cluster] = self.gate_bandit.target(trace.cluster)

        for trace in self.gate_traces:
            meta_batch.append(
                MetaBatchItem(
                    features=trace.features,
                    gate_probs=trace.probs,
                    cluster=trace.cluster,
                    expert_policies=trace.expert_policies,
                )
            )

        loss = self.meta_objective.evaluate(meta_batch, self._gate_forward, bandit_targets)
        self.gate_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gate.parameters(), 5.0)
        self.gate_optimizer.step()
        self.gate_traces.clear()
        return float(loss.item())

    def _apply_anytime_handoff(self, exploitability: float) -> None:
        for cluster in self.iteration_clusters:
            history = self.cluster_history[cluster]
            history.append(exploitability)
            if len(history) == self.handoff_patience and all(v <= self.handoff_tau for v in history):
                if cluster not in self.frozen_clusters:
                    self.frozen_clusters[cluster] = self.iteration_count
        self.iteration_clusters.clear()

    def _policy_components(
        self,
        state_idx: int,
        legal_actions: Sequence[int],
        *,
        record_trace: bool = False,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        info_state = self.info_states[state_idx]
        policies = self._expert_policies(state_idx, info_state, legal_actions)
        cluster = self._cluster_key(info_state)
        self.iteration_clusters.add(cluster)

        features = self._build_gate_features(info_state)
        gate_out = self.gate(features.unsqueeze(0))
        probs = gate_out.probs.squeeze(0)

        if cluster in self.frozen_clusters and self.cfr_index is not None:
            frozen = torch.zeros_like(probs)
            frozen[self.cfr_index] = 1.0
            probs = frozen

        actor_policy = policies.get("actor") or {
            a: 1.0 / len(legal_actions) for a in legal_actions
        }
        regret_policy = policies.get("regret") or actor_policy

        mixed = {a: 0.0 for a in legal_actions}
        expert_list = [policies[name] for name in self.expert_names]
        for weight, expert_policy in zip(probs.tolist(), expert_list):
            for action in legal_actions:
                mixed[action] += weight * expert_policy[action]
        normaliser = sum(mixed.values()) + 1e-8
        for action in mixed:
            mixed[action] /= normaliser

        if self.regret_index is not None:
            self.current_lambda = float(probs[self.regret_index].item())

        if record_trace:
            self.gate_traces.append(
                GateTrace(
                    state_idx=state_idx,
                    cluster=cluster,
                    features=features.detach(),
                    probs=probs.detach(),
                    expert_policies=expert_list,
                    legal_actions=tuple(legal_actions),
                )
            )

        return actor_policy, regret_policy, mixed

    def _collect_episode(self) -> Tuple[Episode, List[Dict[str, object]]]:
        if self.use_rust_backend:
            return self._collect_episode_rust()

        state = self.game.new_initial_state()
        steps: List[EpisodeStep] = []
        experiences: List[Dict[str, object]] = []
        lambdas: List[float] = []

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = random.choices(actions, weights=probs, k=1)[0]
                state = state.child(action)
                continue

            player = state.current_player()
            info_state = state.information_state_string(player)
            legal_actions = self.info_state_actions[info_state]
            state_idx = self.info_state_index[info_state]

            actor_policy, regret_policy, mixed_policy = self._policy_components(
                state_idx, legal_actions, record_trace=True
            )
            probs = [mixed_policy[a] for a in legal_actions]
            action = random.choices(legal_actions, weights=probs, k=1)[0]

            steps.append(EpisodeStep(info_state, legal_actions, action, player))
            lambdas.append(self.current_lambda)
            state = state.child(action)

            experiences.append(
                {
                    "state_idx": state_idx,
                    "legal_actions": legal_actions,
                    "action": action,
                    "player": player,
                    "mixed_policy": mixed_policy,
                }
            )

        returns = state.returns()
        for exp in experiences:
            exp["return"] = returns[exp["player"]]

        return Episode(steps, tuple(returns), lambdas), experiences

    def _collect_episode_rust(self) -> Tuple[Episode, List[Dict[str, object]]]:
        assert self._rust_env is not None and self._rust_utils is not None

        rust_env = self._rust_env
        episode_seed = random.randrange(0, 2**32)
        rust_env.reset(seed=episode_seed)
        rust_state = rust_env.get_state()

        state = self.game.new_initial_state()
        self._rust_utils._sync_initial_chance(state, self.game_name, rust_state)

        steps: List[EpisodeStep] = []
        experiences: List[Dict[str, object]] = []
        lambdas: List[float] = []

        while True:
            while state.is_chance_node() and not state.is_terminal():
                rust_state = rust_env.get_state()
                self._rust_utils._sync_chance_from_rust(state, self.game_name, rust_state)

            if state.is_terminal():
                break

            player = state.current_player()
            info_state = state.information_state_string(player)
            legal_actions = self.info_state_actions[info_state]
            state_idx = self.info_state_index[info_state]

            actor_policy, regret_policy, mixed_policy = self._policy_components(
                state_idx, legal_actions, record_trace=True
            )
            probs = [mixed_policy[a] for a in legal_actions]
            action = random.choices(legal_actions, weights=probs, k=1)[0]

            steps.append(EpisodeStep(info_state, legal_actions, action, player))
            lambdas.append(self.current_lambda)

            _, _, done = rust_env.step(action)
            state = state.child(action)
            experiences.append(
                {
                    "state_idx": state_idx,
                    "legal_actions": legal_actions,
                    "action": action,
                    "player": player,
                    "mixed_policy": mixed_policy,
                }
            )

            if done:
                break

        returns_vec = rust_env.rewards()
        returns = tuple(float(r) for r in returns_vec)
        for exp in experiences:
            exp["return"] = returns[exp["player"]]

        return Episode(steps, returns, lambdas), experiences

    def _sample_batch(self) -> List[Dict[str, object]] | None:
        if not self.buffer:
            return None
        bs = min(self.batch_size, len(self.buffer))
        return random.sample(self.buffer, bs)

    def _update_parameters(self) -> Tuple[float, float, float, float] | None:
        batch = self._sample_batch()
        if not batch:
            return None

        batch_size = len(batch)
        grad_actor = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        grad_critic = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        grad_regret = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        regret_losses: List[float] = []

        for sample in batch:
            idx = sample["state_idx"]
            legal_actions = sample["legal_actions"]
            action = sample["action"]
            returns = sample["return"]

            actor_policy, regret_policy, mixed_policy = self._policy_components(idx, legal_actions)
            critic_values = self.critic_table[idx]

            value_eval = sum(mixed_policy[a] * critic_values[a] for a in legal_actions)
            advantage = critic_values[action] - value_eval

            # Actor gradient
            for a in legal_actions:
                if a == action:
                    grad_actor[idx][a] += -(advantage) * (1 - actor_policy[a])
                else:
                    grad_actor[idx][a] += -(advantage) * (-actor_policy[a])

            # Critic gradient (squared error)
            grad_critic[idx][action] += 2.0 * (critic_values[action] - returns)

            target_regret = max(critic_values[action] - value_eval, 0.0)
            target_by_action = {
                a: max(critic_values[a] - value_eval, 0.0) for a in legal_actions
            }
            for a in legal_actions:
                target = target_by_action[a]
                grad_regret[idx][a] += 2.0 * (self.regret_table[idx][a] - target)

            actor_losses.append(-advantage * math.log(actor_policy[action] + 1e-8))
            critic_losses.append((critic_values[action] - returns) ** 2)
            regret_losses.append((self.regret_table[idx][action] - target_regret) ** 2)

            if self.cfr_head is not None:
                self.cfr_head.observe(
                    idx,
                    legal_actions,
                    {a: self.regret_table[idx][a] - target_by_action[a] for a in legal_actions},
                )
                self.cfr_head.accumulate_policy(idx, legal_actions, mixed_policy)

        inv_bs = 1.0 / batch_size
        for table in (grad_actor, grad_critic, grad_regret):
            for r in range(self.num_states):
                for c in range(self.num_actions):
                    table[r][c] *= inv_bs

        self.actor_opt.step(self.actor_logits, grad_actor)
        self.critic_opt.step(self.critic_table, grad_critic)
        self.regret_opt.step(self.regret_table, grad_regret)

        actor_loss_value = float(sum(actor_losses) / len(actor_losses)) if actor_losses else 0.0
        regret_loss_value = float(sum(regret_losses) / len(regret_losses)) if regret_losses else 0.0
        critic_loss_value = float(sum(critic_losses) / len(critic_losses)) if critic_losses else 0.0

        self.avg_actor_loss = self.loss_beta * self.avg_actor_loss + (1 - self.loss_beta) * actor_loss_value
        self.avg_regret_loss = self.loss_beta * self.avg_regret_loss + (1 - self.loss_beta) * regret_loss_value
        scheduler_loss_value = self._update_scheduler()
        self.avg_scheduler_loss = self.loss_beta * self.avg_scheduler_loss + (1 - self.loss_beta) * scheduler_loss_value

        diff = self.avg_regret_loss - self.avg_actor_loss
        self.current_lambda = float(max(0.05, min(0.95, 1.0 / (1.0 + math.exp(-self.lambda_alpha * diff)))))

        return actor_loss_value, regret_loss_value, critic_loss_value, scheduler_loss_value

    def _update_scheduler(self) -> float:
        if not self.scheduler_training_buffer:
            return 0.0
        inputs = torch.cat([inp.to(self.device).float() for inp, _ in self.scheduler_training_buffer], dim=0)
        targets = torch.tensor(
            [target for _, target in self.scheduler_training_buffer],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        preds = self.scheduler(inputs)
        if preds["mode"] == "continuous":
            lam_pred = preds["lambda"]
        else:
            lam_pred = F.softmax(preds["logits"], dim=-1) @ self.scheduler.k_bins.to(self.device)
        loss = F.mse_loss(lam_pred, targets)
        self.scheduler_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.scheduler.parameters(), 5.0)
        self.scheduler_optimizer.step()
        self.scheduler_training_buffer.clear()
        return float(loss.item())

    def _policy_distribution(self) -> Dict[str, List[float]]:
        table: Dict[str, List[float]] = {}
        for info_state, legal_actions in self.info_state_actions.items():
            idx = self.info_state_index[info_state]
            _, _, mixed = self._policy_components(idx, legal_actions)
            full = [0.0 for _ in range(self.num_actions)]
            for action in legal_actions:
                full[action] = mixed[action]
            table[info_state] = full
        return table

    def _evaluate_policy(self) -> Tuple[float, float, float, float]:
        policy_table = self._policy_distribution()
        tab_policy = pyspiel.TabularPolicy({k: [(a, probs[a]) for a in range(self.num_actions)] for k, probs in policy_table.items()})
        exploitability = float(pyspiel.exploitability(self.game, tab_policy))
        values = pyspiel.expected_returns(self.game.new_initial_state(), [tab_policy, tab_policy], -1, True)
        nash_conv = 2.0 * exploitability
        return nash_conv, exploitability, float(values[0]), float(values[1])

    def run_iteration(self) -> Tuple[IterationMetrics, Dict[str, List[float]]]:
        start = time.perf_counter()
        episodes: List[Episode] = []
        lambda_samples: List[float] = []
        self.gate_traces.clear()
        self.iteration_clusters.clear()

        for _ in range(self.episodes_per_iteration):
            episode, experiences = self._collect_episode()
            episodes.append(episode)
            lambda_samples.extend(episode.lambdas)
            for exp in experiences:
                self.buffer.append(exp)

        gate_loss_value = self._update_gate()
        update = self._update_parameters()
        if update is None:
            actor_loss_value = 0.0
            regret_loss_value = 0.0
            critic_loss_value = 0.0
            scheduler_loss_value = gate_loss_value
        else:
            actor_loss_value, regret_loss_value, critic_loss_value, scheduler_loss_value = update
            scheduler_loss_value = gate_loss_value if gate_loss_value else scheduler_loss_value

        self.iteration_count += 1
        nash_conv, exploitability, value_p0, value_p1 = self._evaluate_policy()
        self._apply_anytime_handoff(exploitability)

        mean_length = statistics.mean(len(ep.steps) for ep in episodes) if episodes else 0.0
        mean_lambda = statistics.mean(lambda_samples) if lambda_samples else self.current_lambda
        mean_return_p0 = statistics.mean(ep.returns[0] for ep in episodes) if episodes else 0.0
        mean_return_p1 = statistics.mean(ep.returns[1] for ep in episodes) if episodes else 0.0

        metrics = IterationMetrics(
            iteration=self.iteration_count,
            exploitability=exploitability,
            nash_conv=nash_conv,
            actor_loss=actor_loss_value,
            average_lambda=mean_lambda,
            mean_episode_length=mean_length,
            mean_return_player0=mean_return_p0,
            mean_return_player1=mean_return_p1,
            wall_time_sec=time.perf_counter() - start,
            scheduler_loss=scheduler_loss_value,
        )

        return metrics, {"lambda_samples": lambda_samples, "values": [value_p0, value_p1]}

    def average_strategy_table(self) -> Dict[str, List[float]]:
        return self._policy_distribution()


class CFRTrainer:
    def __init__(self, game_name: str) -> None:
        if game_name not in {"kuhn_poker", "leduc_poker"}:
            raise ValueError("Supported games: kuhn_poker, leduc_poker")
        self.game = pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError("CFR requires a two-player zero-sum game")
        self.solver = pyspiel.CFRSolver(self.game)
        self.iteration = 0

    def run_iteration(self) -> Tuple[IterationMetrics, Dict[str, List[float]]]:
        start = time.perf_counter()
        self.solver.evaluate_and_update_policy()
        self.iteration += 1
        policy = self.solver.average_policy()
        exploitability = float(pyspiel.exploitability(self.game, policy))
        nash_conv = 2.0 * exploitability
        values = pyspiel.expected_returns(
            self.game.new_initial_state(), [policy, policy], -1, True
        )
        metrics = IterationMetrics(
            iteration=self.iteration,
            exploitability=exploitability,
            nash_conv=nash_conv,
            actor_loss=0.0,
            average_lambda=0.0,
            mean_episode_length=0.0,
            mean_return_player0=float(values[0]),
            mean_return_player1=float(values[1]),
            wall_time_sec=time.perf_counter() - start,
            scheduler_loss=0.0,
        )
        return metrics, {"lambda_samples": []}

    def average_strategy_table(self) -> Dict[str, List[float]]:
        policy = self.solver.average_policy()
        mapping: Dict[str, List[float]] = {}
        stack = [self.game.new_initial_state()]
        visited: set[str] = set()
        while stack:
            state = stack.pop()
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    stack.append(state.child(action))
                continue
            player = state.current_player()
            info_state = state.information_state_string(player)
            if info_state not in visited:
                visited.add(info_state)
                probs_map = policy.action_probabilities(state)
                probs = [0.0 for _ in range(self.game.num_distinct_actions())]
                for action, prob in probs_map.items():
                    probs[action] = prob
                mapping[info_state] = probs
            for action in state.legal_actions():
                stack.append(state.child(action))
        return mapping


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARMAC poker agents (tabular neural-style)")
    parser.add_argument("--game", type=str, default="kuhn_poker", choices=["kuhn_poker", "leduc_poker"])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iteration", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lambda-alpha", type=float, default=2.0)
    parser.add_argument("--loss-beta", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="neural_tabular",
        choices=["neural_tabular", "cfr"],
        help="Training algorithm to run",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Root directory for experiment artefacts (default: results).",
    )
    parser.add_argument(
        "--manifest-path",
        type=pathlib.Path,
        default=pathlib.Path("results/manifest.csv"),
        help="CSV manifest that records a summary row for each training run.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default_experiment",
        help="Logical experiment bucket used to organize result directories.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional free-form label appended to result filenames for easier filtering.",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--experts",
        type=str,
        default="actor,regret,ra,explore,cfr",
        help="Comma separated list of experts for the MARM-K gate",
    )
    parser.add_argument("--meta-unroll", type=int, default=16)
    parser.add_argument("--br-budget", type=int, default=64)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--gate-kl-weight", type=float, default=0.1)
    parser.add_argument("--handoff-tau", type=float, default=0.15)
    parser.add_argument("--handoff-patience", type=int, default=3)
    parser.add_argument(
        "--state-cluster",
        type=str,
        default="round+position+pot",
        help="Clustering heuristic for gate bandit tables",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pyspiel",
        choices=["pyspiel", "rust"],
        help="State transition backend (default: pyspiel)",
    )
    return parser.parse_args(argv)


def run_training(opts: argparse.Namespace) -> Dict[str, object]:
    if opts.algorithm == "cfr":
        trainer: object = CFRTrainer(opts.game)
        policy_type = "cfr"
    else:
        trainer = NeuralARMACTrainer(
            opts.game,
            seed=opts.seed,
            actor_lr=opts.actor_lr,
            critic_lr=opts.critic_lr,
            regret_lr=opts.regret_lr,
            episodes_per_iteration=opts.episodes_per_iteration,
            buffer_size=opts.buffer_size,
            batch_size=opts.batch_size,
            lambda_alpha=opts.lambda_alpha,
            loss_beta=opts.loss_beta,
            backend=opts.backend,
            experts=[exp.strip() for exp in opts.experts.split(",") if exp.strip()],
            meta_unroll=opts.meta_unroll,
            br_budget=opts.br_budget,
            gate_lr=opts.gate_lr,
            gate_kl_weight=opts.gate_kl_weight,
            handoff_tau=opts.handoff_tau,
            handoff_patience=opts.handoff_patience,
            state_cluster=opts.state_cluster,
        )
        policy_type = "neural_tabular"

    training_history: List[Dict[str, float]] = []
    lambda_samples: List[float] = []

    iterable = range(1, opts.iterations + 1)
    progress = (
        tqdm(
            iterable,
            desc=f"{policy_type.upper()} {opts.game}",
            unit="iter",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else iterable
    )

    for iteration in progress:
        metrics, samples = trainer.run_iteration()
        training_history.append(metrics.as_dict())
        lambda_samples.extend(samples["lambda_samples"])

        if tqdm is not None:
            progress.set_postfix(
                {
                    "exploit": f"{metrics.exploitability:.4f}",
                    "nash": f"{metrics.nash_conv:.4f}",
                    "actor": f"{metrics.actor_loss:.4f}",
                    "lambda": f"{metrics.average_lambda:.3f}",
                    "sched": f"{metrics.scheduler_loss:.2e}",
                    "sec/it": f"{metrics.wall_time_sec:.2f}",
                },
                refresh=False,
            )
        elif iteration % max(1, opts.iterations // 10) == 0 or iteration == 1:
            print(
                f"[{iteration:04d}/{opts.iterations}] exploitability={metrics.exploitability:.4f} "
                f"NashConv={metrics.nash_conv:.4f} actor_loss={metrics.actor_loss:.4f} "
                f"avg_lambda={metrics.average_lambda:.3f} scheduler_loss={metrics.scheduler_loss:.2e} "
                f"sec/iter={metrics.wall_time_sec:.2f}"
            )

    if tqdm is not None:
        progress.close()

    total_wall_time = float(sum(entry["wall_time_sec"] for entry in training_history)) if training_history else 0.0
    final_metrics = training_history[-1] if training_history else {}

    summary = {
        "game": opts.game,
        "seed": opts.seed,
        "iterations": opts.iterations,
        "episodes_per_iteration": opts.episodes_per_iteration,
        "actor_lr": opts.actor_lr,
        "critic_lr": opts.critic_lr,
        "regret_lr": opts.regret_lr,
        "batch_size": opts.batch_size,
        "buffer_size": opts.buffer_size,
        "lambda_alpha": opts.lambda_alpha,
        "loss_beta": opts.loss_beta,
        "policy_type": policy_type,
        "training_history": training_history,
        "lambda_samples": lambda_samples,
        "average_strategy": trainer.average_strategy_table(),
        "total_wall_time_sec": total_wall_time,
        "final_metrics": final_metrics,
    }
    return summary


def _manifest_notes(opts: argparse.Namespace) -> str:
    notes = [item for item in (opts.run_label, opts.tag) if item]
    return ",".join(notes)


def save_results(summary: Dict[str, object], opts: argparse.Namespace) -> pathlib.Path:
    experiment = opts.experiment_name or "default_experiment"
    policy = summary["policy_type"]
    target_dir = (
        opts.output_dir
        / experiment
        / summary["game"]
        / (policy if isinstance(policy, str) else str(policy))
        / f"seed_{opts.seed}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    components = [
        summary["game"],
        policy,
        f"seed{opts.seed}",
        f"iter{summary['iterations']}",
        f"epi{summary['episodes_per_iteration']}",
        f"backend{opts.backend}",
    ]
    if opts.run_label:
        components.append(opts.run_label)
    if opts.tag:
        components.append(opts.tag)
    base_name = "_".join(str(part) for part in components) + f"_{timestamp}"

    json_path = target_dir / f"{base_name}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    history = summary.get("training_history", [])
    csv_path: Optional[pathlib.Path] = None
    if history:
        csv_path = target_dir / f"{base_name}_history.csv"
        fieldnames = list(history[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    manifest_run_id: Optional[str] = None
    try:
        manifest = ManifestManager(str(opts.manifest_path))
        final_metrics: Dict[str, float] = summary.get("final_metrics", {}) or {}
        manifest_config = {
            "iterations": summary.get("iterations"),
            "episodes_per_iteration": summary.get("episodes_per_iteration"),
            "actor_lr": summary.get("actor_lr"),
            "critic_lr": summary.get("critic_lr"),
            "regret_lr": summary.get("regret_lr"),
            "batch_size": summary.get("batch_size"),
            "buffer_size": summary.get("buffer_size"),
            "lambda_alpha": summary.get("lambda_alpha"),
            "loss_beta": summary.get("loss_beta"),
            "backend": opts.backend,
            "experts": opts.experts,
            "meta_unroll": opts.meta_unroll,
            "br_budget": opts.br_budget,
            "gate_lr": opts.gate_lr,
            "gate_kl_weight": opts.gate_kl_weight,
            "handoff_tau": opts.handoff_tau,
            "handoff_patience": opts.handoff_patience,
            "state_cluster": opts.state_cluster,
        }
        manifest_run_id = manifest.log_experiment(
            algorithm=str(policy),
            game=summary.get("game", "unknown"),
            config=manifest_config,
            seed=summary.get("seed", 0),
            iteration=summary.get("iterations", 0),
            nash_conv=float(final_metrics.get("nash_conv", 0.0)),
            exploitability=float(final_metrics.get("exploitability", 0.0)),
            wall_clock_time=float(summary.get("total_wall_time_sec", 0.0)),
            final_reward=float(final_metrics.get("mean_return_player0", 0.0)),
            parameters=0,
            flops_per_forward=0,
            training_flops=0,
            model_size_mb=0.0,
            notes=_manifest_notes(opts),
        )
    except Exception as exc:  # pragma: no cover - manifest is best-effort logging
        print(f"[WARN] Failed to update manifest at {opts.manifest_path}: {exc}")

    print(f"Results saved to {json_path}")
    if csv_path is not None:
        print(f"Per-iteration metrics saved to {csv_path}")
    if manifest_run_id is not None:
        print(
            "Manifest updated",
            f"(run_id={manifest_run_id}) at {opts.manifest_path}",
        )

    return json_path


def main(argv: Sequence[str] | None = None) -> None:
    opts = parse_args(argv)
    opts.output_dir = opts.output_dir.expanduser()
    opts.manifest_path = opts.manifest_path.expanduser()
    summary = run_training(opts)
    save_results(summary, opts)


if __name__ == "__main__":
    main()
