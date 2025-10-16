"""Utilities for interacting with the compiled Rust poker environments.

This module keeps the PyO3 extension importable from a bare `cargo build`
and offers lightweight wrappers for parity checks against OpenSpiel.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# OpenSpiel import is deferred because the sandbox blocks OpenMP shared memory.
pyspiel = None  # type: ignore

_RUST_MODULE: ModuleType | None = None


class RustModuleNotBuiltError(ImportError):
    """Raised when the compiled PyO3 module cannot be located."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_library_paths() -> List[Path]:
    release_dir = _project_root() / "rust" / "target" / "release"
    if not release_dir.exists():
        return []

    patterns = (
        "dual_rl_poker_rust*.so",
        "dual_rl_poker_rust*.dylib",
        "dual_rl_poker_rust*.pyd",
        "libdual_rl_poker_rust*.so",
        "libdual_rl_poker_rust*.dylib",
    )

    files: List[Path] = []
    for pattern in patterns:
        files.extend(release_dir.glob(pattern))
    return sorted({path.resolve() for path in files})


def _build_rust_module() -> None:
    """Invoke `cargo build --release` to produce the shared library."""

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RustModuleNotBuiltError(
            "Rust module 'dual_rl_poker_rust' is missing and `cargo` is not available. "
            "Install Rust toolchain or build the extension manually."
        )

    project_root = _project_root()
    rust_dir = project_root / "rust"
    if not (rust_dir / "Cargo.toml").exists():
        raise RustModuleNotBuiltError(
            f"Rust module expected at {rust_dir}, but Cargo.toml is missing."
        )

    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")

    result = subprocess.run(
        [cargo, "build", "--release"],
        cwd=str(rust_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RustModuleNotBuiltError(
            "Failed to build Rust module via `cargo build --release`.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def load_rust_module(force_reload: bool = False) -> ModuleType:
    """Import the compiled PyO3 module that exposes the poker environments.

    The loader first attempts a standard import (`pip install`/`maturin build`
    workflows). If that fails, it looks for the shared library under
    `rust/target/release` so developers can run directly from a `cargo build`.
    """

    global _RUST_MODULE
    if not force_reload and _RUST_MODULE is not None:
        return _RUST_MODULE

    try:
        module = importlib.import_module("dual_rl_poker_rust")
    except ImportError as exc:
        last_error: Optional[BaseException] = exc
        build_attempted = False
        for path in _candidate_library_paths():
            loader = importlib.machinery.ExtensionFileLoader("dual_rl_poker_rust", str(path))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as load_error:  # pragma: no cover - rare edge cases
                last_error = load_error
                continue
            sys.modules["dual_rl_poker_rust"] = module
            _RUST_MODULE = module
            return module
        if not build_attempted:
            try:
                _build_rust_module()
            except RustModuleNotBuiltError as build_error:
                last_error = build_error
            else:
                build_attempted = True
                for path in _candidate_library_paths():
                    loader = importlib.machinery.ExtensionFileLoader("dual_rl_poker_rust", str(path))
                    spec = importlib.util.spec_from_loader(loader.name, loader)
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                    except Exception as load_error:  # pragma: no cover
                        last_error = load_error
                        continue
                    sys.modules["dual_rl_poker_rust"] = module
                    _RUST_MODULE = module
                    return module
        raise RustModuleNotBuiltError(
            "Rust module 'dual_rl_poker_rust' is missing and automatic build was unsuccessful. "
            "Run `cargo build --release` in the `rust/` directory and ensure the shared library "
            "is available under `rust/target/release/`."
        ) from last_error

    _RUST_MODULE = module
    return module


@dataclass
class EpisodeTrace:
    """Captures a single episode rollout for debugging purposes."""

    actions: List[int]
    rewards: Tuple[float, float]
    steps: int
    initial_state: Dict[str, object]
    final_state: Dict[str, object]


class RustEnvWrapper:
    """Convenience wrapper around the compiled Rust environments."""

    def __init__(self, game_name: str, *, seed: Optional[int] = None, module: ModuleType | None = None):
        module = module or load_rust_module()
        try:
            factory = {
                "kuhn_poker": module.KuhnPokerEnv,
                "leduc_poker": module.LeducPokerEnv,
            }[game_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported game: {game_name}") from exc

        self._module = module
        self._env = factory(seed)
        self._game_name = game_name

    # ------------------------------------------------------------------
    # Direct proxies to the PyO3 class
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        self._env.reset(seed)
        return self.observation()

    def observation(self):
        return self._env.observation()

    def legal_actions(self) -> List[int]:
        return list(self._env.legal_actions())

    def current_player(self) -> int:
        return int(self._env.current_player())

    def is_terminal(self) -> bool:
        return bool(self._env.is_terminal())

    def rewards(self):
        return self._env.rewards()

    def step(self, action: int):
        obs, reward, done = self._env.step(int(action))
        return obs, float(reward), bool(done)

    def info_state(self, player: int):
        return self._env.info_state(int(player))

    def game_info(self) -> Dict[str, object]:
        return dict(self._env.get_game_info())

    def get_state(self) -> Dict[str, object]:
        return dict(self._env.get_state())

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def run_episode(self, rng: random.Random) -> EpisodeTrace:
        """Roll an episode using uniform random actions."""

        actions: List[int] = []
        self.reset()
        initial_state = self.get_state()
        steps = 0
        while not self.is_terminal():
            legal = self.legal_actions()
            if not legal:
                break
            action = int(rng.choice(legal))
            actions.append(action)
            _, _, done = self.step(action)
            steps += 1
            if done:
                break
        reward_vec = self.rewards()
        return EpisodeTrace(
            actions=actions,
            rewards=(float(reward_vec[0]), float(reward_vec[1])),
            steps=steps,
            initial_state=dict(initial_state),
            final_state=self.get_state(),
        )


def make_env(game_name: str, seed: Optional[int] = None) -> RustEnvWrapper:
    """Factory that returns a reset-ready Rust environment."""

    return RustEnvWrapper(game_name, seed=seed)


# ----------------------------------------------------------------------
# OpenSpiel comparison helpers
# ----------------------------------------------------------------------

@dataclass
class ComparisonReport:
    game_name: str
    episodes: int
    mismatches: int
    max_reward_delta: float
    mean_steps: float
    rust_wall_time: float
    pyspiel_wall_time: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "game_name": self.game_name,
            "episodes": self.episodes,
            "mismatches": self.mismatches,
            "max_reward_delta": self.max_reward_delta,
            "mean_steps": self.mean_steps,
            "rust_wall_time": self.rust_wall_time,
            "pyspiel_wall_time": self.pyspiel_wall_time,
        }


def compare_with_openspiel(
    game_name: str,
    *,
    episodes: int = 256,
    seed: int = 0,
) -> ComparisonReport:
    """Run identical random rollouts in Rust and OpenSpiel to sanity-check parity."""

    global pyspiel
    if pyspiel is None:
        try:
            import pyspiel as _pyspiel  # type: ignore
        except ImportError as exc:
            raise ImportError("OpenSpiel (pyspiel) is required for parity comparisons.") from exc
        pyspiel = _pyspiel

    module = load_rust_module()
    rust_env = RustEnvWrapper(game_name, seed=seed, module=module)
    game = pyspiel.load_game(game_name)
    rng = random.Random(seed)
    chance_seed = rng.randrange(0, 2**32)

    mismatches = 0
    reward_deltas: List[float] = []
    step_counts: List[int] = []

    rust_wall = 0.0
    py_wall = 0.0

    for episode_idx in range(episodes):
        # Ensure deterministic but distinct reset seeds
        episode_seed = (chance_seed + episode_idx) % (2**32)
        rust_env.reset(seed=episode_seed)
        rust_state = rust_env.get_state()

        state = game.new_initial_state()
        try:
            _sync_initial_chance(state, game_name, rust_state)
        except ValueError:
            mismatches += 1
            continue

        steps = 0
        mismatch = False

        while True:
            # Advance OpenSpiel through any remaining chance nodes using Rust's perspective.
            while state.is_chance_node() and not mismatch:
                try:
                    rust_state = rust_env.get_state()
                    _sync_chance_from_rust(state, game_name, rust_state)
                except ValueError:
                    mismatch = True
                    break

            rust_terminal = rust_env.is_terminal()
            py_terminal = state.is_terminal()

            if rust_terminal or py_terminal:
                if rust_terminal != py_terminal:
                    mismatch = True
                break

            rust_actions = rust_env.legal_actions()
            py_actions = state.legal_actions()
            common_actions = [action for action in py_actions if action in rust_actions]

            if not common_actions:
                mismatch = True
                break

            action = rng.choice(common_actions)

            rust_start = time.perf_counter()
            _, _, _ = rust_env.step(action)
            rust_wall += time.perf_counter() - rust_start

            py_start = time.perf_counter()
            state.apply_action(action)
            py_wall += time.perf_counter() - py_start

            steps += 1

        if not rust_env.is_terminal() or not state.is_terminal():
            mismatch = True

        rust_rewards = rust_env.rewards()
        py_returns = state.returns() if state.is_terminal() else (float("nan"), float("nan"))

        reward_delta = max(
            abs(float(rust_rewards[0]) - float(py_returns[0])),
            abs(float(rust_rewards[1]) - float(py_returns[1])),
        )

        reward_deltas.append(reward_delta)
        step_counts.append(steps)

        if mismatch or reward_delta > 1e-6:
            mismatches += 1

    return ComparisonReport(
        game_name=game_name,
        episodes=episodes,
        mismatches=mismatches,
        max_reward_delta=max(reward_deltas) if reward_deltas else 0.0,
        mean_steps=float(statistics.mean(step_counts) if step_counts else 0.0),
        rust_wall_time=rust_wall,
        pyspiel_wall_time=py_wall,
    )


def _sync_initial_chance(state, game_name: str, rust_state: Dict[str, object]) -> None:
    """Apply the initial chance actions so OpenSpiel mirrors the Rust deal."""

    if game_name == "kuhn_poker":
        cards = rust_state.get("cards")
        if not isinstance(cards, list) or len(cards) != 2:
            raise ValueError("Rust state for Kuhn Poker did not expose cards.")
        for card in cards:
            _apply_chance_action(state, game_name, int(card))
    elif game_name == "leduc_poker":
        cards = rust_state.get("player_card_ids") or rust_state.get("player_cards")
        if not isinstance(cards, list) or len(cards) != 2:
            raise ValueError("Rust state for Leduc Poker did not expose card identifiers.")
        for raw in cards:
            if raw is None:
                raise ValueError("Missing player card in Rust state.")
            _apply_chance_action(state, game_name, int(raw))
    else:
        raise ValueError(f"Unsupported game for chance sync: {game_name}")


def _sync_chance_from_rust(state, game_name: str, rust_state: Dict[str, object]) -> None:
    """Advance OpenSpiel chance nodes using Rust's current state."""

    if not state.is_chance_node():
        return

    if game_name == "leduc_poker":
        community_card = rust_state.get("community_card_id")
        if community_card is None:
            community_card = rust_state.get("community_card")
        if community_card is None:
            raise ValueError("Rust state missing community_card during flop sync.")
        _apply_chance_action(state, game_name, int(community_card))
    elif game_name == "kuhn_poker":
        return  # No additional chance events after the initial deal.
    else:
        raise ValueError(f"No mid-hand chance sync defined for {game_name}.")


def _apply_chance_action(state, game_name: str, action: int) -> None:
    """Apply a chance outcome, tolerating rank-only identifiers for Leduc."""

    for candidate, _ in state.chance_outcomes():
        if candidate == action:
            state.apply_action(candidate)
            return

    if game_name == "leduc_poker" and action in (0, 1, 2):
        rank_to_candidates = {
            0: (0, 1),  # Jacks
            1: (2, 3),  # Queens
            2: (4, 5),  # Kings
        }
        desired = rank_to_candidates.get(action, ())
        for candidate, _ in state.chance_outcomes():
            if candidate in desired:
                state.apply_action(candidate)
                return

    raise ValueError(
        f"Chance action {action} not available in current OpenSpiel state for {game_name}."
    )


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate Rust poker envs against OpenSpiel.")
    parser.add_argument("game", choices=["kuhn_poker", "leduc_poker"], help="Game to compare")
    parser.add_argument("--episodes", type=int, default=128, help="Number of random rollouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    report = compare_with_openspiel(args.game, episodes=args.episodes, seed=args.seed)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        data = report.as_dict()
        print(f"Game: {data['game_name']}")
        print(f"Episodes: {data['episodes']}")
        print(f"Mismatches: {data['mismatches']}")
        print(f"Max reward delta: {data['max_reward_delta']:.3e}")
        print(f"Mean steps: {data['mean_steps']:.2f}")
        print(f"Rust wall time: {data['rust_wall_time']:.3f}s")
        print(f"OpenSpiel wall time: {data['pyspiel_wall_time']:.3f}s")


if __name__ == "__main__":
    _main()
