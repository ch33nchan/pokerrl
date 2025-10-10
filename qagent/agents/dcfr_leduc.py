"""Deep CFR trainer for Leduc Hold'em using the unified environment API."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from qagent.encoders import FlatInfoSetEncoder
from qagent.environments.base import BaseGameEnv, InfoSet
from qagent.environments.leduc_holdem import LeducEnv, LeducState
from qagent.utils.logging import TrainingLogger


class RegretNet(nn.Module):
    """Two-layer feed-forward network that outputs cumulative regrets."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.output(x)


class StrategyNet(nn.Module):
    """Two-layer feed-forward network producing a probability distribution."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.softmax(self.output(x))


class DCFRTrainer:
    """Deep CFR trainer operating on environments that implement `BaseGameEnv`."""

    def __init__(
        self,
        game: Optional[BaseGameEnv] = None,
        learning_rate: float = 1e-4,
        memory_size: int = 1_000_000,
        hidden_dim: int = 256,
        grad_clip: Optional[float] = 1.0,
        lr_scheduler_gamma: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "leduc",
        log_to_csv: bool = True,
        log_to_jsonl: bool = False,
        regret_noise_std: float = 0.0,
    ) -> None:
        self.game: BaseGameEnv = game if game is not None else LeducEnv()
        self.info_set_dim = self.game.get_obs_dim()
        self.num_actions = self.game.num_actions()

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.regret_net = RegretNet(self.info_set_dim, self.num_actions, hidden_dim).to(self.device)
        self.strategy_net = StrategyNet(self.info_set_dim, self.num_actions, hidden_dim).to(self.device)

        self.optimizer_regret = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.optimizer_strategy = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.regret_scheduler = (
            optim.lr_scheduler.ExponentialLR(self.optimizer_regret, gamma=lr_scheduler_gamma)
            if lr_scheduler_gamma is not None
            else None
        )
        self.strategy_scheduler = (
            optim.lr_scheduler.ExponentialLR(self.optimizer_strategy, gamma=lr_scheduler_gamma)
            if lr_scheduler_gamma is not None
            else None
        )

        self.regret_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.strategy_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.memory_size = memory_size

        self.encoder = FlatInfoSetEncoder()
        self.grad_clip = grad_clip
        self.log_history: List[Dict[str, float]] = []
        self.logger: Optional[TrainingLogger] = None
        self.training_start_time: Optional[float] = None

        self.gradient_norms: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.training_losses: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.kl_history: List[float] = []
        self.target_variances: Dict[str, List[float]] = {"regret": [], "strategy": []}
        self.iteration_wall_clock: List[float] = []
        self.parameter_counts: Dict[str, int] = {
            "regret": sum(p.numel() for p in self.regret_net.parameters()),
            "strategy": sum(p.numel() for p in self.strategy_net.parameters()),
        }
        self.numeric_stability_events: Dict[str, int] = {
            "regret_grad_clipped": 0,
            "strategy_grad_clipped": 0,
            "advantage_normalized": 0,
        }
        self.advantage_stats: List[Dict[str, float]] = []
        self.reach_prob_bins: List[Dict[str, float]] = []
        self._parquet_records: List[Dict[str, float]] = []
        self._parquet_path: Optional[Path] = Path(log_dir) / f"{log_prefix}_diagnostics.parquet" if log_dir else None

        self.regret_noise_std = float(regret_noise_std)

        if log_dir is not None:
            self.logger = TrainingLogger(
                output_dir=log_dir,
                base_filename=f"{log_prefix}_metrics",
                write_csv=log_to_csv,
                write_jsonl=log_to_jsonl,
                overwrite=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        iterations: int,
        update_threshold: int = 100,
        checkpoint_dir: Optional[str] = "checkpoints_leduc_dcfr",
        checkpoint_interval: int = 50_000,
    ) -> None:
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if self.training_start_time is None:
            self.training_start_time = time.perf_counter()

        iterator = range(1, iterations + 1)
        pbar = tqdm(iterator, total=iterations, desc=f"DCFR({self.__class__.__name__})", leave=True) if tqdm else iterator
        for t in pbar:
            iteration_start = time.perf_counter()
            root_state = self.game.get_initial_state()
            traverser_state = self._sample_chance_state(root_state)
            self._cfr_traverse(traverser_state, t, 1.0, 1.0)

            if t % update_threshold == 0:
                metrics = self._update_networks(update_threshold, t)
                if metrics:
                    lr_regret = metrics.get("regret_lr", self.optimizer_regret.param_groups[0]["lr"])
                    lr_strategy = metrics.get("strategy_lr", self.optimizer_strategy.param_groups[0]["lr"])
                    metrics["iteration_wall_clock_sec"] = float(time.perf_counter() - iteration_start)
                    metrics["cumulative_wall_clock_sec"] = float(time.perf_counter() - self.training_start_time)
                    metrics["regret_param_count"] = float(self.parameter_counts["regret"])
                    metrics["strategy_param_count"] = float(self.parameter_counts["strategy"])
                    msg = (
                        f"Iteration {t}/{iterations} | "
                        f"RegretLoss {metrics.get('regret_loss', 0.0):.4f} | "
                        f"StrategyLoss {metrics.get('strategy_loss', 0.0):.4f} | "
                        f"RegGrad {metrics.get('regret_grad_norm', 0.0):.2f} | "
                        f"StratGrad {metrics.get('strategy_grad_norm', 0.0):.2f} | "
                        f"KL {metrics.get('strategy_kl', 0.0):.4f} | "
                        f"LR(R/S) {lr_regret:.2e}/{lr_strategy:.2e}"
                    )
                else:
                    msg = (
                        f"Iteration {t}/{iterations}: Networks updated. Regret Mem: {len(self.regret_memory)}, "
                        f"Strategy Mem: {len(self.strategy_memory)}"
                    )
                if tqdm:
                    pbar.set_postfix_str(msg.replace("Iteration ", "").replace(f"/{iterations}", ""))
                else:
                    print(msg, end='\r')

            if checkpoint_dir and t % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"strategy_net_iter_{t}.pt")
                torch.save(self.strategy_net.state_dict(), path)
                print(f"\nSaved checkpoint to {path}")

        if self.logger is not None:
            self.logger.close()
        if self._parquet_path and self._parquet_records:
            df = pd.DataFrame(self._parquet_records)
            try:
                df.to_parquet(self._parquet_path, index=False)
            except (ImportError, ModuleNotFoundError, ValueError) as err:
                fallback_path = self._parquet_path.with_suffix(".csv")
                df.to_csv(fallback_path, index=False)
                print(
                    f"Warning: Unable to write diagnostics Parquet ({err}). "
                    f"Diagnostics saved as CSV to {fallback_path}."
                )

    def get_average_strategy(self, state: Dict[str, object]) -> np.ndarray:
        infoset = self._build_infoset(state)
        info_tensor = self.encoder.encode_infoset(self.game, infoset).unsqueeze(0)
        legal_mask = self._legal_action_mask(state)
        return self._average_strategy_from_tensor(info_tensor, legal_mask)

    # ------------------------------------------------------------------
    # Core CFR logic
    # ------------------------------------------------------------------
    def _cfr_traverse(self, state: Dict[str, object], t: int, pi_p: float, pi_o: float) -> float:
        if self.game.is_terminal(state):
            return self.game.get_payoff(state, 0)

        player = self.game.get_current_player(state)
        infoset = self._build_infoset(state)
        info_tensor = self.encoder.encode_infoset(self.game, infoset).unsqueeze(0).to(self.device)
        info_vector = info_tensor.squeeze(0).detach().cpu().numpy()

        legal_mask = self._legal_action_mask(state)
        if legal_mask.sum() == 0:
            return self.game.get_payoff(state, 0)

        strategy = self._strategy_from_tensor(info_tensor, legal_mask)

        util = np.zeros(self.num_actions, dtype=np.float32)
        node_util = 0.0

        for action in range(self.num_actions):
            if legal_mask[action] == 0:
                continue

            next_state = self.game.get_next_state(dict(state), action)
            if player == 0:
                util[action] = self._cfr_traverse(next_state, t, pi_p * strategy[action], pi_o)
            else:
                util[action] = self._cfr_traverse(next_state, t, pi_p, pi_o * strategy[action])
            node_util += strategy[action] * util[action]

        if player == 0:
            regrets = (util - node_util) * legal_mask
            if self.regret_noise_std > 0:
                noise = np.random.normal(0.0, self.regret_noise_std, size=regrets.shape).astype(np.float32)
                regrets = regrets + noise
            with torch.no_grad():
                current_regrets = self.regret_net(info_tensor).cpu().numpy().flatten()
            new_regrets = current_regrets + pi_o * regrets
            self._add_to_memory(self.regret_memory, (info_vector.copy(), new_regrets))

            with torch.no_grad():
                current_strategy = self.strategy_net(info_tensor).cpu().numpy().flatten()
            current_strategy *= legal_mask
            if current_strategy.sum() > 0:
                current_strategy /= current_strategy.sum()
            else:
                current_strategy = legal_mask / legal_mask.sum()

            update_term = pi_p * strategy
            new_strategy = current_strategy + update_term
            self._add_to_memory(self.strategy_memory, (info_vector.copy(), new_strategy))

        return float(node_util)

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------
    def _update_networks(self, update_threshold: int, iteration: int) -> Optional[Dict[str, float]]:
        if len(self.regret_memory) < update_threshold or len(self.strategy_memory) < update_threshold:
            return None

        metrics: Dict[str, float] = {"iteration": float(iteration)}

        regret_metrics = self._update_regret_network(iteration)
        metrics.update({f"regret_{k}": v for k, v in regret_metrics.items()})

        strategy_metrics = self._update_strategy_network(iteration)
        metrics.update({f"strategy_{k}": v for k, v in strategy_metrics.items()})

        if self.regret_scheduler is not None:
            self.regret_scheduler.step()
        if self.strategy_scheduler is not None:
            self.strategy_scheduler.step()

        metrics["regret_lr"] = self.optimizer_regret.param_groups[0]["lr"]
        metrics["strategy_lr"] = self.optimizer_strategy.param_groups[0]["lr"]
        metrics["regret_mem_size"] = float(len(self.regret_memory))
        metrics["strategy_mem_size"] = float(len(self.strategy_memory))

        self.log_history.append(metrics)
        if self.logger is not None:
            self.logger.log(metrics)
        self._parquet_records.append(metrics)
        return metrics

    def _update_regret_network(self, iteration: int) -> Dict[str, float]:
        batch = random.sample(self.regret_memory, min(len(self.regret_memory), 256))
        inputs_np = np.stack([item[0] for item in batch])
        targets_np = np.stack([item[1] for item in batch])
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.device)

        self.optimizer_regret.zero_grad()
        predictions = self.regret_net(inputs)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.regret_net.parameters())
        self.optimizer_regret.step()

        variance = float(torch.var(targets).item())
        self.gradient_norms["regret"].append(grad_norm)
        self.training_losses["regret"].append(loss.item())
        self.target_variances["regret"].append(variance)
        if clipped:
            self.numeric_stability_events["regret_grad_clipped"] += 1

        advantages = targets.detach().cpu().numpy()
        self.advantage_stats.append(
            {
                "iteration": float(iteration),
                "phase": "regret",
                "mean": float(np.mean(advantages)),
                "std": float(np.std(advantages)),
                "p50": float(np.percentile(advantages, 50)),
                "p90": float(np.percentile(advantages, 90)),
                "p99": float(np.percentile(advantages, 99)),
            }
        )

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "target_variance": variance,
        }

    def _update_strategy_network(self, iteration: int) -> Dict[str, float]:
        batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), 256))
        inputs_np = np.stack([item[0] for item in batch])
        targets_np = np.stack([item[1] for item in batch])
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.device)

        self.optimizer_strategy.zero_grad()
        predictions = self.strategy_net(inputs)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        grad_norm, clipped = self._apply_gradient_clipping(self.strategy_net.parameters())
        self.optimizer_strategy.step()

        kl_div = self._kl_divergence(predictions.detach(), targets)
        variance = float(torch.var(targets).item())

        self.gradient_norms["strategy"].append(grad_norm)
        self.training_losses["strategy"].append(loss.item())
        self.target_variances["strategy"].append(variance)
        self.kl_history.append(kl_div)
        if clipped:
            self.numeric_stability_events["strategy_grad_clipped"] += 1

        strategies = predictions.detach().cpu().numpy()
        self.advantage_stats.append(
            {
                "iteration": float(iteration),
                "phase": "strategy",
                "mean": float(np.mean(strategies)),
                "std": float(np.std(strategies)),
                "p50": float(np.percentile(strategies, 50)),
                "p90": float(np.percentile(strategies, 90)),
                "p99": float(np.percentile(strategies, 99)),
            }
        )

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "kl": kl_div,
            "target_variance": variance,
        }

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------
    def _strategy_from_tensor(self, info_tensor: torch.Tensor, legal_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            regrets = self.regret_net(info_tensor.to(self.device)).cpu().numpy().flatten()
        positive_regrets = np.maximum(regrets, 0.0) * legal_mask
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        legal_count = legal_mask.sum()
        if legal_count == 0:
            raise RuntimeError("No legal actions available to compute strategy.")
        return legal_mask / legal_count

    def _average_strategy_from_tensor(self, info_tensor: torch.Tensor, legal_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            strategy = self.strategy_net(info_tensor.to(self.device)).cpu().numpy().flatten()
        masked = strategy * legal_mask
        total = masked.sum()
        if total > 0:
            return masked / total
        legal_count = legal_mask.sum()
        if legal_count == 0:
            raise RuntimeError("No legal actions available to compute average strategy.")
        return legal_mask / legal_count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _sample_chance_state(self, state: Dict[str, object]) -> Dict[str, object]:
        outcome = self.game.sample_chance_outcome(state)
        if isinstance(outcome, list):
            _, sampled = random.choice(outcome)
            return sampled
        if isinstance(outcome, tuple) and len(outcome) == 2:
            return outcome[1]
        return outcome

    def _build_infoset(self, state: Dict[str, object]) -> InfoSet:
        leduc_state = LeducState.from_dict(state)
        metadata = {
            "private_cards": leduc_state.private_cards,
            "board_card": leduc_state.board_card,
            "history": leduc_state.history,
            "round": leduc_state.round,
            "player": leduc_state.current_player,
            "deck": leduc_state.deck,
            "pot": leduc_state.pot_contributions,
            "folded": leduc_state.folded_player,
            "bets": leduc_state.bets,
        }
        key = self._infoset_key(metadata)
        return InfoSet(key=key, metadata=metadata)

    def _infoset_key(self, metadata: Dict[str, object]) -> str:
        board = metadata["board_card"]
        board_str = "N" if board is None else str(board % LeducEnv.num_ranks)
        private_cards: Sequence[int] = metadata["private_cards"]
        player: int = metadata["player"]
        return (
            f"P{player}|C{private_cards[player] % LeducEnv.num_ranks}|"
            f"B{board_str}|R{metadata['round']}|H{len(metadata['history'])}"
        )

    def _legal_action_mask(self, state: Dict[str, object]) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        for action in self.game.get_legal_actions(state):
            mask[action] = 1.0
        return mask

    def _add_to_memory(self, memory: List[Tuple[np.ndarray, np.ndarray]], data: Tuple[np.ndarray, np.ndarray]) -> None:
        if len(memory) >= self.memory_size:
            memory.pop(random.randint(0, len(memory) - 1))
        memory.append(data)

    def _apply_gradient_clipping(self, parameters: Iterable[torch.nn.parameter.Parameter]) -> Tuple[float, bool]:
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0, False
        if self.grad_clip is not None and self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            clipped = float(norm) > self.grad_clip if isinstance(norm, torch.Tensor) else norm > self.grad_clip
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().flatten() for p in params]))
            clipped = False
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm), bool(clipped)

    def _kl_divergence(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            preds = predictions.clamp_min(1e-8)
            target_probs = targets.clone()
            sums = target_probs.sum(dim=1, keepdim=True)
            valid_rows = (sums > 0).squeeze(1)
            if valid_rows.any():
                target_probs_valid = target_probs[valid_rows] / sums[valid_rows]
                target_probs_valid = target_probs_valid.clamp_min(1e-8)
                preds_valid = preds[valid_rows]
                kl_values = torch.sum(
                    target_probs_valid * (torch.log(target_probs_valid) - torch.log(preds_valid.clamp_min(1e-8))),
                    dim=1,
                )
                return float(kl_values.mean().item())
            return 0.0


__all__ = ["DCFRTrainer"]
