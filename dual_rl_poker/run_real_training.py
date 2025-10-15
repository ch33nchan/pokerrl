"""Neural-style ARMAC training loop using pure Python data structures.

This script trains actor/critic/regret tables with adaptive λ on Kuhn or Leduc
Poker. It produces real trajectories, adapts λ with a logistic scheduler, and
computes exploitability via OpenSpiel's tabular utilities. The implementation
avoids third-party numeric libraries so it remains portable in restricted
environments.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple

import pyspiel


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
        }


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
    ) -> None:
        if game_name not in {"kuhn_poker", "leduc_poker"}:
            raise ValueError("Supported games: kuhn_poker, leduc_poker")

        self.game = pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError("Only two-player zero-sum games are supported")

        random.seed(seed)

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
        self.iteration_count = 0

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
            for action in state.legal_actions():
                stack.append(state.child(action))
        info_states = sorted(cache.keys())
        return info_states, cache

    def _policy_components(self, state_idx: int, legal_actions: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        actor_logits = self.actor_logits[state_idx]
        actor_policy = masked_softmax(actor_logits, legal_actions)

        # Regret policy from non-negative regrets
        regrets = self.regret_table[state_idx]
        positive = {a: max(regrets[a], 0.0) for a in legal_actions}
        pos_sum = sum(positive.values())
        if pos_sum > 1e-8:
            regret_policy = {a: val / pos_sum for a, val in positive.items()}
        else:
            regret_policy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        mixed: Dict[int, float] = {}
        lam = self.current_lambda
        for action in legal_actions:
            mixed[action] = lam * regret_policy[action] + (1 - lam) * actor_policy[action]
        norm = sum(mixed.values()) + 1e-8
        for action in mixed:
            mixed[action] /= norm

        return actor_policy, regret_policy, mixed

    def _collect_episode(self) -> Tuple[Episode, List[Dict[str, object]]]:
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

            actor_policy, regret_policy, mixed_policy = self._policy_components(state_idx, legal_actions)
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

    def _sample_batch(self) -> List[Dict[str, object]] | None:
        if not self.buffer:
            return None
        bs = min(self.batch_size, len(self.buffer))
        return random.sample(self.buffer, bs)

    def _update_parameters(self) -> Tuple[float, float, float] | None:
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
            for a in legal_actions:
                target = max(critic_values[a] - value_eval, 0.0)
                grad_regret[idx][a] += 2.0 * (self.regret_table[idx][a] - target)

            actor_losses.append(-advantage * math.log(actor_policy[action] + 1e-8))
            critic_losses.append((critic_values[action] - returns) ** 2)
            regret_losses.append((self.regret_table[idx][action] - target_regret) ** 2)

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

        diff = self.avg_regret_loss - self.avg_actor_loss
        self.current_lambda = float(max(0.05, min(0.95, 1.0 / (1.0 + math.exp(-self.lambda_alpha * diff)))))

        return actor_loss_value, regret_loss_value, critic_loss_value

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

        for _ in range(self.episodes_per_iteration):
            episode, experiences = self._collect_episode()
            episodes.append(episode)
            lambda_samples.extend(episode.lambdas)
            for exp in experiences:
                self.buffer.append(exp)

        update = self._update_parameters()
        if update is None:
            actor_loss_value = 0.0
            regret_loss_value = 0.0
            critic_loss_value = 0.0
        else:
            actor_loss_value, regret_loss_value, critic_loss_value = update

        self.iteration_count += 1
        nash_conv, exploitability, value_p0, value_p1 = self._evaluate_policy()

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
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--tag", type=str, default="")
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
        )
        policy_type = "neural_tabular"

    training_history: List[Dict[str, float]] = []
    lambda_samples: List[float] = []

    for iteration in range(1, opts.iterations + 1):
        metrics, samples = trainer.run_iteration()
        training_history.append(metrics.as_dict())
        lambda_samples.extend(samples["lambda_samples"])

        if iteration % max(1, opts.iterations // 10) == 0 or iteration == 1:
            print(
                f"[{iteration:04d}/{opts.iterations}] exploitability={metrics.exploitability:.4f} "
                f"NashConv={metrics.nash_conv:.4f} actor_loss={metrics.actor_loss:.4f} "
                f"avg_lambda={metrics.average_lambda:.3f}"
            )

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
    }
    return summary


def save_results(summary: Dict[str, object], opts: argparse.Namespace) -> pathlib.Path:
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{opts.tag}" if opts.tag else ""
    timestamp = int(time.time())
    filename = f"{opts.game}_{summary['policy_type']}_seed{opts.seed}_{timestamp}{tag}.json"
    output_path = opts.output_dir / filename
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    opts = parse_args(argv)
    summary = run_training(opts)
    output_path = save_results(summary, opts)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
