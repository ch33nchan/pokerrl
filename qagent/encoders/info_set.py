from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from qagent.environments.base import BaseGameEnv, InfoSet


@dataclass(frozen=True)
class EncodedInfoSet:
    """Container holding encoded representations for a single info-set."""

    flat: torch.Tensor
    history_one_hot: torch.Tensor | None = None
    history_indices: torch.Tensor | None = None


class FlatInfoSetEncoder:
    """Encodes the canonical info-set vector exposed by `BaseGameEnv`."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype

    def encode_env(self, env: BaseGameEnv) -> torch.Tensor:
        vector = env.get_info_set_vector()
        return torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(self.dtype)

    def encode_infoset(self, env: BaseGameEnv, infoset: InfoSet) -> torch.Tensor:
        vector = env.info_set_to_vector(infoset)
        return torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(self.dtype)

    def batch(self, envs: Iterable[BaseGameEnv]) -> torch.Tensor:
        encoded = [self.encode_env(env) for env in envs]
        return torch.stack(encoded, dim=0)


class HistoryOneHotEncoder:
    """Encodes the discrete action history as a one-hot matrix."""

    def __init__(self, max_length: int, dtype: torch.dtype = torch.float32) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.dtype = dtype

    def encode_sequence(self, history: Sequence[int], num_actions: int) -> torch.Tensor:
        history_array = np.zeros((self.max_length, num_actions), dtype=np.float32)
        truncated = history[-self.max_length :]
        offset = self.max_length - len(truncated)
        for idx, action in enumerate(truncated):
            if action < 0 or action >= num_actions:
                raise ValueError(f"Action index {action} outside [0, {num_actions})")
            history_array[offset + idx, action] = 1.0
        return torch.from_numpy(history_array).to(self.dtype)

    def encode_env(self, env: BaseGameEnv) -> torch.Tensor:
        history = list(env.get_action_sequence())
        return self.encode_sequence(history, env.num_actions())


class ActionIndexSequenceEncoder:
    """Encodes the action history as padded integer indices suitable for embeddings."""

    def __init__(self, max_length: int, pad_value: int | None = None) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.pad_value = pad_value

    def encode_sequence(self, history: Sequence[int], num_actions: int) -> torch.Tensor:
        pad_token = self.pad_value if self.pad_value is not None else num_actions
        sequence = np.full(self.max_length, pad_token, dtype=np.int64)
        truncated = history[-self.max_length :]
        sequence[-len(truncated) :] = truncated
        return torch.from_numpy(sequence)

    def encode_env(self, env: BaseGameEnv) -> torch.Tensor:
        history = list(env.get_action_sequence())
        return self.encode_sequence(history, env.num_actions())


def build_encoded_infoset(
    env: BaseGameEnv,
    flat_encoder: FlatInfoSetEncoder,
    history_one_hot: HistoryOneHotEncoder | None = None,
    history_indices: ActionIndexSequenceEncoder | None = None,
) -> EncodedInfoSet:
    """Utility to produce a full `EncodedInfoSet` bundle for the current env state."""

    flat_tensor = flat_encoder.encode_env(env)
    history_hot_tensor = history_one_hot.encode_env(env) if history_one_hot else None
    history_index_tensor = history_indices.encode_env(env) if history_indices else None
    return EncodedInfoSet(
        flat=flat_tensor,
        history_one_hot=history_hot_tensor,
        history_indices=history_index_tensor,
    )
