"""Bindings around the Rust poker environments (compiled via PyO3).

This module attempts to import the compiled extension module
`dual_rl_poker_rust`. If it cannot be imported (because the shared
library is missing or fails to load) a descriptive ImportError is
raised so the caller can fall back to the Python/OpenSpiel path.
"""

from __future__ import annotations

try:
    import dual_rl_poker_rust as rust
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Rust environment module 'dual_rl_poker_rust' could not be imported."
        " Build it with `cargo build --release` inside the `rust/` directory."
    ) from exc


def create_batch(game_name: str, batch_size: int, seed: int | None = None):
    cfg = rust.EnvConfig(seed=seed, batch_size=batch_size, max_steps=None, reward_scale=1.0, gamma=1.0, deterministic=False)
    if game_name == "kuhn_poker":
        return rust.create_kuhn_batch(batch_size, cfg)
    if game_name == "leduc_poker":
        return rust.create_leduc_batch(batch_size, cfg)
    raise ValueError(f"Unknown game: {game_name}")
