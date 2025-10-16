"""Utility modules for the Dual RL Poker project.

The optional model analysis helpers depend on PyTorch; keep the import guarded so
environments without torch can still access the rest of the utilities (e.g. the
Rust loaders) without failing.
"""

from typing import Any, Dict, Tuple

from .config import load_config, save_config
from .logging import setup_logging, get_experiment_logger
from .manifest_manager import ManifestManager

try:
    from .diagnostics import TrainingDiagnostics, DiagnosticAnalyzer
except Exception:  # pragma: no cover - torch is optional here too
    def _torch_unavailable_diag(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "PyTorch is not available; install torch to use utils.diagnostics helpers."
        )

    class TrainingDiagnostics:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _torch_unavailable_diag()

    class DiagnosticAnalyzer:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _torch_unavailable_diag()

try:
    from .model_analysis import (
        count_parameters,
        estimate_flops_per_forward,
        analyze_model_capacity,
        ModelCapacityLogger,
    )
except Exception:  # pragma: no cover - torch is optional in many sandboxes
    def _torch_unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "PyTorch is not available; install torch to use utils.model_analysis helpers."
        )

    count_parameters = _torch_unavailable  # type: ignore[assignment]
    estimate_flops_per_forward = _torch_unavailable  # type: ignore[assignment]

    def analyze_model_capacity(
        _models: Dict[str, object], _input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, Dict[str, Any]]:
        _torch_unavailable()

    class ModelCapacityLogger:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _torch_unavailable()


__all__ = [
    "load_config",
    "save_config",
    "setup_logging",
    "get_experiment_logger",
    "count_parameters",
    "estimate_flops_per_forward",
    "analyze_model_capacity",
    "ModelCapacityLogger",
    "TrainingDiagnostics",
    "DiagnosticAnalyzer",
    "ManifestManager",
]
