"""Ablation study orchestration for Leduc Hold'em DCFR agents."""

from __future__ import annotations

import argparse
import platform
import subprocess
import inspect
import json
import copy
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import yaml

from qagent.agents.dcfr_leduc import DCFRTrainer as BaselineAgent
from qagent.agents.dcfr_leduc_gru import DCFRGRUTrainer as GRUAgent
from qagent.agents.dcfr_leduc_lstm import DCFRTrainer as LSTMAgent
from qagent.agents.dcfr_leduc_lstm_no_embedding import DCFROneHotAblationTrainer as LSTMNoEmbeddingAgent
from qagent.agents.dcfr_leduc_lstm_no_history import DCFRAblationNoHistoryTrainer as LSTMNoHistoryAgent
from qagent.agents.dcfr_leduc_transformer import DCFRTransformerTrainer as TransformerAgent
from qagent.agents.dcfr_leduc_opponent import DCFROpponentModelTrainer as OpponentModelAgent
from qagent.analysis.plot_learning_curves import plot_learning_curves_ci
from qagent.evaluation.exploitability import calculate_exploitability
from qagent.environments.leduc_holdem import LeducHoldem

DEFAULT_NUM_RUNS = 2
DEFAULT_TRAINING_ITERATIONS = 20_000
DEFAULT_CHECKPOINT_INTERVAL = 5_000
DEFAULT_EVALUATION_INTERVAL = 5_000
DEFAULT_UPDATE_THRESHOLD = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_DEVICE = "cpu"
DEFAULT_PLOT_TITLE = "Leduc Hold'em: DCFR Ablation Study"
DEFAULT_PLOT_FILENAME = "exploitability_ablation_study.png"
DEFAULT_RANDOM_SEED = 42
DEFAULT_TRAVERSAL_SCHEME = "external_sampling"
DEFAULT_SUCCESS_THRESHOLD = 2.0
DEFAULT_OUTPUT_DIR = Path("runs/leduc_dcfr_ablation")

EXPERIMENT_NAME = "leduc_dcfr_ablation"

AGENT_CONFIG: Dict[str, Dict[str, object]] = {
    "baseline": {
        "agent_class": BaselineAgent,
        "name": "Baseline (One-Hot)",
        "init_params": {},
        "train_params": {},
    },
    "baseline_noise_001": {
        "agent_class": BaselineAgent,
        "name": "Baseline (σ=0.01)",
        "init_params": {"regret_noise_std": 0.01},
        "train_params": {},
    },
    "baseline_noise_005": {
        "agent_class": BaselineAgent,
        "name": "Baseline (σ=0.05)",
        "init_params": {"regret_noise_std": 0.05},
        "train_params": {},
    },
    "baseline_noise_010": {
        "agent_class": BaselineAgent,
        "name": "Baseline (σ=0.10)",
        "init_params": {"regret_noise_std": 0.10},
        "train_params": {},
    },
    "lstm_optimized": {
        "agent_class": LSTMAgent,
        "name": "LSTM (Optimized)",
        "init_params": {},
        "train_params": {},
    },
    "lstm_no_history": {
        "agent_class": LSTMNoHistoryAgent,
        "name": "LSTM (No History)",
        "init_params": {},
        "train_params": {},
    },
    "lstm_no_embedding": {
        "agent_class": LSTMNoEmbeddingAgent,
        "name": "LSTM (No Embedding)",
        "init_params": {},
        "train_params": {},
    },
    "gru": {
        "agent_class": GRUAgent,
        "name": "GRU (Sequence)",
        "init_params": {},
        "train_params": {},
    },
    "transformer": {
        "agent_class": TransformerAgent,
        "name": "Transformer (Self-Attention)",
        "init_params": {
            "embedding_dim": 128,
            "num_heads": 8,
            "num_encoder_layers": 2,
        },
        "train_params": {},
    },
    "opponent_model": {
        "agent_class": OpponentModelAgent,
        "name": "DCFR + Opponent Model",
        "init_params": {},
        "train_params": {},
    },
}


def _compute_parameter_count(trainer: Any) -> Optional[int]:
    total = 0
    found = False
    for attr in ("regret_net", "strategy_net", "policy_network", "advantage_network"):
        module = getattr(trainer, attr, None)
        if module is None:
            continue
        found = True
        total += sum(p.numel() for p in module.parameters())
    return int(total) if found else None


def _infer_optimizer_name(trainer: Any) -> Optional[str]:
    for attr in ("optimizer_regret", "optimizer", "optimizer_strategy"):
        opt = getattr(trainer, attr, None)
        if opt is not None:
            return opt.__class__.__name__
    return None


def _infer_learning_rate(trainer: Any) -> Optional[float]:
    for attr in ("optimizer_regret", "optimizer", "optimizer_strategy"):
        opt = getattr(trainer, attr, None)
        if opt is None:
            continue
        param_groups = getattr(opt, "param_groups", None)
        if not param_groups:
            continue
        lr = param_groups[0].get("lr")
        if lr is not None:
            return float(lr)
    return None


def _append_manifest_record(entry: Dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")


def _filter_kwargs(callable_obj, candidate_kwargs):
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in candidate_kwargs.items() if key in signature.parameters}


def train_agent(
    agent_class,
    agent_name: str,
    run_seed: int,
    checkpoint_dir: str,
    log_dir: Path,
    iterations: int,
    checkpoint_interval: int,
    update_threshold: int,
    batch_size: int,
    device: str,
    init_overrides: Optional[Dict[str, Any]] = None,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print(f"--- Starting Training for {agent_name}, Run Seed: {run_seed} ---")
    start_time = time.time()

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    game = LeducHoldem()
    init_candidates: Dict[str, Any] = {
        "log_dir": str(log_dir),
        "log_prefix": f"ablation_{agent_name.lower().replace(' ', '_')}_seed_{run_seed}",
        "log_to_csv": True,
        "log_to_jsonl": False,
        "device": device,
    }
    if init_overrides:
        init_candidates.update(init_overrides)
    init_kwargs = _filter_kwargs(agent_class.__init__, init_candidates)
    trainer = agent_class(game, **init_kwargs)

    train_signature = inspect.signature(trainer.train)
    train_kwargs = {}
    if "n_iterations" in train_signature.parameters:
        train_kwargs["n_iterations"] = iterations
    elif "iterations" in train_signature.parameters:
        train_kwargs["iterations"] = iterations
    if "update_threshold" in train_signature.parameters:
        train_kwargs["update_threshold"] = update_threshold
    if "batch_size" in train_signature.parameters:
        train_kwargs["batch_size"] = batch_size
    if "checkpoint_dir" in train_signature.parameters:
        train_kwargs["checkpoint_dir"] = checkpoint_dir
    if "checkpoint_interval" in train_signature.parameters:
        train_kwargs["checkpoint_interval"] = checkpoint_interval

    if train_overrides:
        train_kwargs.update(_filter_kwargs(trainer.train, train_overrides))

    trainer.train(**train_kwargs)

    end_time = time.time()
    wall_clock = end_time - start_time
    print(f"--- Training Complete for {agent_name}, Run Seed: {run_seed} in {wall_clock:.2f}s ---")

    metadata: Dict[str, Any] = {
        "training_time": wall_clock,
        "parameter_count": _compute_parameter_count(trainer),
        "optimizer": _infer_optimizer_name(trainer),
        "learning_rate": _infer_learning_rate(trainer),
        "device": str(getattr(trainer, "device", "unknown")),
    }
    return metadata


def evaluate_checkpoint(
    agent_class,
    checkpoint_path: str,
    iteration: int,
    device: str,
    init_overrides: Optional[Dict[str, Any]] = None,
):
    try:
        game = LeducHoldem()
        init_candidates: Dict[str, Any] = {"device": device}
        if init_overrides:
            init_candidates.update(init_overrides)
        init_kwargs = _filter_kwargs(agent_class.__init__, init_candidates)
        agent = agent_class(game, **init_kwargs)
        state_dict = torch.load(checkpoint_path, map_location=device)
        agent.strategy_net.load_state_dict(state_dict)
        agent.strategy_net.to(device)
        agent.strategy_net.eval()

        # Try evaluation; supports trainers with either `_build_infoset` or
        # `get_average_strategy` + `_encode_info_set` via PolicyWrapper logic.
        try:
            exploitability = calculate_exploitability(agent)
            return iteration, exploitability
        except Exception as eval_exc:
            print(
                f"Skipping exploitability evaluation for {agent_class.__name__} at iteration {iteration}: {eval_exc}"
            )
            return iteration, None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error evaluating checkpoint {checkpoint_path}: {exc}")
        return iteration, None


def evaluate_checkpoints_sequentially(
    agent_class,
    checkpoint_dir: str,
    evaluation_interval: int,
    device: str,
    init_overrides: Optional[Dict[str, Any]] = None,
):
    checkpoints: List[Tuple[int, str]] = []
    if not os.path.isdir(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return []

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt") and filename.startswith("strategy_net_iter_"):
            try:
                iteration = int(filename.split("_")[-1].split(".")[0])
                if iteration > 0 and iteration % evaluation_interval == 0:
                    checkpoints.append((iteration, os.path.join(checkpoint_dir, filename)))
            except (ValueError, IndexError):
                print(f"Could not parse iteration from filename: {filename}")

    checkpoints.sort(key=lambda item: item[0])

    results: List[Tuple[int, float]] = []
    for iteration, checkpoint_path in tqdm(
        checkpoints,
        desc=f"Evaluate {agent_class.__name__}",
        unit="ckpt",
        leave=False,
    ):
        iter_idx, exploitability = evaluate_checkpoint(
            agent_class,
            checkpoint_path,
            iteration,
            device,
            init_overrides=init_overrides,
        )
        if exploitability is not None:
            results.append((int(iter_idx), float(exploitability)))

    return results


def run_ablation(
    num_runs: int,
    iterations: int,
    checkpoint_interval: int,
    evaluation_interval: int,
    update_threshold: int,
    batch_size: int,
    device: str,
    plot_title: str,
    plot_output_filename: str,
    success_threshold: float,
    agent_keys: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    agent_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    selected_agents = _select_agents(agent_keys)
    selected_agents = _apply_agent_overrides(selected_agents, agent_overrides)

    random.seed(DEFAULT_RANDOM_SEED if random_seed is None else random_seed)

    output_root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "run_manifest_records.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    results_path = output_root / "ablation_study_results.json"
    if results_path.exists():
        results_path.unlink()

    all_run_data: Dict[str, List[List[Tuple[int, float]]]] = {key: [] for key in selected_agents.keys()}

    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    run_progress = tqdm(range(1, num_runs + 1), desc="Runs", unit="run")
    for run_idx in run_progress:
        run_seeds = {name: random.randint(0, int(1e6)) for name in selected_agents.keys()}

        # Persist reproducibility metadata for this run
        run_root = log_root / f"run_{run_idx}"
        run_root.mkdir(parents=True, exist_ok=True)
        meta_path = run_root / "reproducibility.json"
        git_hash = None
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".")
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_hash = None
        repro = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": device,
            "num_runs": num_runs,
            "iterations": iterations,
            "checkpoint_interval": checkpoint_interval,
            "evaluation_interval": evaluation_interval,
            "update_threshold": update_threshold,
            "batch_size": batch_size,
            "agents": list(selected_agents.keys()),
            "seeds": run_seeds,
            "git_commit": git_hash,
            "traversal_scheme": DEFAULT_TRAVERSAL_SCHEME,
            "success_threshold": success_threshold,
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(repro, fh, indent=2)
        agent_progress = tqdm(list(selected_agents.items()), desc=f"Run {run_idx}", unit="agent", leave=False)
        for agent_key, config in agent_progress:
            agent_progress.set_postfix(agent=config["name"])

            checkpoint_dir = output_root / f"checkpoints_ablation_{agent_key}_run_{run_idx}"
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            run_log_dir = run_root / agent_key
            if run_log_dir.exists():
                shutil.rmtree(run_log_dir)
            run_log_dir.mkdir(parents=True, exist_ok=True)

            train_metadata = train_agent(
                agent_class=config["agent_class"],
                agent_name=config["name"],
                run_seed=run_seeds[agent_key],
                checkpoint_dir=str(checkpoint_dir),
                log_dir=run_log_dir,
                iterations=iterations,
                checkpoint_interval=checkpoint_interval,
                update_threshold=update_threshold,
                batch_size=batch_size,
                device=device,
                init_overrides=config.get("init_params"),
                train_overrides=config.get("train_params"),
            )

            results = evaluate_checkpoints_sequentially(
                config["agent_class"],
                str(checkpoint_dir),
                evaluation_interval,
                device,
                init_overrides=config.get("init_params"),
            )
            all_run_data[agent_key].append(results)

            final_iteration = results[-1][0] if results else None
            final_exploitability = results[-1][1] if results else None
            success_flag = bool(
                final_exploitability is not None and final_exploitability <= success_threshold
            )
            notes: Optional[str] = None
            if not results:
                notes = "exploitability_unavailable"

            manifest_entry = {
                "run_id": f"{EXPERIMENT_NAME}-{agent_key}-run{run_idx}",
                "experiment": EXPERIMENT_NAME,
                "architecture": agent_key,
                "seed": int(run_seeds[agent_key]),
                "traversal_scheme": DEFAULT_TRAVERSAL_SCHEME,
                "params_count": train_metadata.get("parameter_count"),
                "optimizer": train_metadata.get("optimizer"),
                "lr": train_metadata.get("learning_rate"),
                "batch_size": batch_size,
                "replay_ratio": None,
                "steps": final_iteration if final_iteration is not None else iterations,
                "wall_clock_s": train_metadata.get("training_time"),
                "success_flag": success_flag,
                "final_exploitability": final_exploitability,
                "notes": notes,
                "commit_hash": git_hash,
            }
            _append_manifest_record(manifest_entry, manifest_path)

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(all_run_data, handle, indent=4)

    print(f"\nAggregated results from all runs saved to {results_path}")

    plot_learning_curves_ci(
        str(results_path),
        agent_names={key: config["name"] for key, config in selected_agents.items()},
        title=plot_title,
        output_filename=str(output_root / plot_output_filename),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Leduc DCFR ablation study with configurable runtime and device options.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--num-runs", type=int, default=None, help="Number of independent runs (seeds).")
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations per run.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Interval between saved checkpoints.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help="Evaluate checkpoints every N iterations.",
    )
    parser.add_argument(
        "--update-threshold",
        type=int,
        default=None,
        help="Traversals between network updates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for network updates (if supported).",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device (e.g., 'cpu', 'cuda').")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed controlling run scheduling and agent seeds.")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Exploitability threshold defining run success (default 2.0).",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Optional subset of agents to evaluate (keys: baseline, lstm_optimized, lstm_no_history, lstm_no_embedding).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where checkpoints, logs, manifests, and plots will be written.",
    )
    return parser.parse_args()


def _select_agents(agent_keys: Optional[List[str]]) -> Dict[str, Dict[str, object]]:
    if agent_keys is None:
        return {key: copy.deepcopy(value) for key, value in AGENT_CONFIG.items()}

    missing = [key for key in agent_keys if key not in AGENT_CONFIG]
    if missing:
        raise ValueError(f"Unknown agent keys requested: {', '.join(missing)}")

    return {key: copy.deepcopy(AGENT_CONFIG[key]) for key in agent_keys}


def _apply_agent_overrides(
    base_agents: Dict[str, Dict[str, object]],
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, object]]:
    if not overrides:
        return base_agents

    updated: Dict[str, Dict[str, object]] = {}
    for key, config in base_agents.items():
        merged = copy.deepcopy(config)
        override_cfg = overrides.get(key)
        if override_cfg:
            if "name" in override_cfg:
                merged["name"] = override_cfg["name"]
            for field in ("init_params", "train_params"):
                if field in override_cfg:
                    base_params = dict(merged.get(field, {}) or {})
                    override_params = override_cfg[field] or {}
                    base_params.update(override_params)
                    merged[field] = base_params
        updated[key] = merged
    return updated


def _load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping at the top level.")
    return data


def _resolve_parameters(
    cli_args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[List[str]], Dict[str, Dict[str, Any]]]:
    config_data: Dict[str, Any] = {}
    if cli_args.config:
        config_data = _load_config(cli_args.config)

    experiment_cfg: Dict[str, Any] = config_data.get("experiment", {}) if isinstance(config_data.get("experiment", {}), dict) else {}
    agents_cfg: Dict[str, Any] = config_data.get("agents", {}) if isinstance(config_data.get("agents", {}), dict) else {}

    def fetch_param(name: str, default: Any) -> Any:
        value = getattr(cli_args, name)
        if value is not None:
            return value
        return experiment_cfg.get(name, default)

    run_kwargs = {
        "num_runs": fetch_param("num_runs", DEFAULT_NUM_RUNS),
        "iterations": fetch_param("iterations", DEFAULT_TRAINING_ITERATIONS),
        "checkpoint_interval": fetch_param("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL),
        "evaluation_interval": fetch_param("evaluation_interval", DEFAULT_EVALUATION_INTERVAL),
        "update_threshold": fetch_param("update_threshold", DEFAULT_UPDATE_THRESHOLD),
        "batch_size": fetch_param("batch_size", DEFAULT_BATCH_SIZE),
        "device": fetch_param("device", DEFAULT_DEVICE),
        "plot_title": experiment_cfg.get("plot_title", DEFAULT_PLOT_TITLE),
        "plot_output_filename": experiment_cfg.get("plot_output_filename", DEFAULT_PLOT_FILENAME),
        "random_seed": fetch_param("random_seed", DEFAULT_RANDOM_SEED),
        "success_threshold": fetch_param("success_threshold", DEFAULT_SUCCESS_THRESHOLD),
        "output_dir": fetch_param("output_dir", str(DEFAULT_OUTPUT_DIR)),
    }

    agents_override: Optional[List[str]] = None
    if cli_args.agents is not None:
        agents_override = cli_args.agents
    elif "include" in agents_cfg and isinstance(agents_cfg.get("include"), list):
        agents_override = [str(agent) for agent in agents_cfg["include"]]

    agent_overrides: Dict[str, Dict[str, Any]] = {}
    for key, override_cfg in agents_cfg.items():
        if key == "include":
            continue
        if isinstance(override_cfg, dict):
            agent_overrides[str(key)] = override_cfg

    return run_kwargs, agents_override, agent_overrides


if __name__ == "__main__":
    cli_args = parse_args()
    run_kwargs, agents_override, agent_overrides = _resolve_parameters(cli_args)
    run_ablation(
        num_runs=int(run_kwargs["num_runs"]),
        iterations=int(run_kwargs["iterations"]),
        checkpoint_interval=int(run_kwargs["checkpoint_interval"]),
        evaluation_interval=int(run_kwargs["evaluation_interval"]),
        update_threshold=int(run_kwargs["update_threshold"]),
        batch_size=int(run_kwargs["batch_size"]),
        device=str(run_kwargs["device"]),
        plot_title=str(run_kwargs["plot_title"]),
        plot_output_filename=str(run_kwargs["plot_output_filename"]),
        success_threshold=float(run_kwargs["success_threshold"]),
        agent_keys=agents_override,
        random_seed=int(run_kwargs["random_seed"]),
        agent_overrides=agent_overrides,
        output_dir=Path(run_kwargs["output_dir"]),
    )
