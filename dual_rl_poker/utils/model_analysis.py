"""Model analysis utilities for parameter counting and FLOPs estimation."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_per_forward(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate FLOPs per forward pass for a given model.

    This is a rough estimation based on the number of multiply-add operations
    in each layer type.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, ...)

    Returns:
        Estimated FLOPs per forward pass
    """
    def count_linear_ops(layer, input_shape):
        """Count operations for linear layer."""
        in_features = layer.in_features
        out_features = layer.out_features
        batch_size = input_shape[0]
        # Each output requires in_features multiplications and in_features-1 additions
        ops_per_output = in_features * 2 - 1  # multiply-add
        return batch_size * out_features * ops_per_output

    def count_activation_ops(layer, input_shape):
        """Count operations for activation functions."""
        batch_size = np.prod(input_shape)
        if isinstance(layer, nn.ReLU):
            # ReLU: max(0, x) - simple comparison
            return batch_size
        elif isinstance(layer, nn.Tanh):
            # Tanh: complex operations, estimate as 10 FLOPs
            return batch_size * 10
        elif isinstance(layer, nn.Sigmoid):
            # Sigmoid: exp, division, etc. - estimate as 15 FLOPs
            return batch_size * 15
        elif isinstance(layer, nn.Softmax):
            # Softmax: exp, sum, division
            batch_size = input_shape[0]
            num_classes = input_shape[-1]
            return batch_size * (num_classes * 10 + 5)
        else:
            return batch_size

    total_flops = 0
    current_shape = input_shape

    # Create dummy input to trace through model
    dummy_input = torch.randn(*current_shape)

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_flops, current_shape

            if isinstance(module, nn.Linear):
                ops = count_linear_ops(module, current_shape)
                total_flops += ops
                current_shape = (current_shape[0], module.out_features)

            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax)):
                ops = count_activation_ops(module, current_shape)
                total_flops += ops
                # Activation functions don't change shape

        return hook

    # Register hooks for all relevant layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax)):
            hooks.append(module.register_forward_hook(register_hook(module)))

    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops


def analyze_model_capacity(models: Dict[str, nn.Module],
                           input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict[str, Any]]:
    """Analyze model capacity including parameters and FLOPs.

    Args:
        models: Dictionary mapping model names to PyTorch models
        input_shapes: Dictionary mapping model names to input shapes

    Returns:
        Dictionary with capacity analysis for each model
    """
    analysis = {}

    for name, model in models.items():
        if name not in input_shapes:
            logger.warning(f"No input shape provided for model {name}, skipping")
            continue

        try:
            # Count parameters
            total_params = count_parameters(model)

            # Estimate FLOPs
            input_shape = input_shapes[name]
            flops_per_forward = estimate_flops_per_forward(model, input_shape)

            # Get memory usage estimate
            param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            input_size_mb = np.prod(input_shape) * 4 / (1024 * 1024)

            analysis[name] = {
                "total_parameters": total_params,
                "flops_per_forward": flops_per_forward,
                "parameter_size_mb": param_size_mb,
                "input_size_mb": input_size_mb,
                "memory_estimate_mb": param_size_mb + input_size_mb,
                "input_shape": list(input_shape)
            }

            logger.info(f"Model {name}: {total_params:,} params, "
                       f"{flops_per_forward:,} FLOPs/forward, "
                       f"{param_size_mb:.2f}MB parameters")

        except Exception as e:
            logger.error(f"Error analyzing model {name}: {e}")
            analysis[name] = {"error": str(e)}

    return analysis


def compute_training_flops(analysis: Dict[str, Dict[str, Any]],
                           batch_size: int,
                           num_iterations: int) -> Dict[str, Dict[str, Any]]:
    """Compute total FLOPs for training.

    Args:
        analysis: Model capacity analysis
        batch_size: Training batch size
        num_iterations: Number of training iterations

    Returns:
        Dictionary with training FLOPs computation
    """
    training_analysis = {}

    for name, model_analysis in analysis.items():
        if "error" in model_analysis:
            training_analysis[name] = model_analysis
            continue

        # Forward pass FLOPs (scaled by batch size if needed)
        base_batch_size = model_analysis["input_shape"][0]
        batch_scale = batch_size / base_batch_size if base_batch_size > 0 else 1
        forward_flops = int(model_analysis["flops_per_forward"] * batch_scale)

        # Backward pass typically costs ~2x forward pass
        backward_flops = forward_flops * 2

        # Total FLOPs per iteration
        flops_per_iteration = forward_flops + backward_flops

        # Total training FLOPs
        total_flops = flops_per_iteration * num_iterations

        training_analysis[name] = {
            **model_analysis,
            "forward_flops_per_iteration": forward_flops,
            "backward_flops_per_iteration": backward_flops,
            "total_flops_per_iteration": flops_per_iteration,
            "total_training_flops": total_flops,
            "batch_size": batch_size,
            "num_iterations": num_iterations
        }

    return training_analysis


def create_capacity_report(analysis: Dict[str, Dict[str, Any]],
                          output_file: str = None) -> str:
    """Create a formatted capacity analysis report.

    Args:
        analysis: Model capacity analysis
        output_file: Optional file to save report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("Model Capacity Analysis Report")
    lines.append("=" * 50)

    for name, model_analysis in analysis.items():
        lines.append(f"\n{name}:")
        lines.append("-" * len(name))

        if "error" in model_analysis:
            lines.append(f"  Error: {model_analysis['error']}")
            continue

        lines.append(f"  Parameters: {model_analysis['total_parameters']:,}")
        lines.append(f"  FLOPs/forward: {model_analysis['flops_per_forward']:,}")
        lines.append(f"  Parameter size: {model_analysis['parameter_size_mb']:.2f} MB")
        lines.append(f"  Input shape: {model_analysis['input_shape']}")

        if "total_training_flops" in model_analysis:
            lines.append(f"  Total training FLOPs: {model_analysis['total_training_flops']:,e}")
            lines.append(f"  Batch size: {model_analysis['batch_size']}")
            lines.append(f"  Iterations: {model_analysis['num_iterations']}")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Capacity report saved to {output_file}")

    return report


class ModelCapacityLogger:
    """Logger for tracking model capacity during training."""

    def __init__(self):
        self.capacity_data = {}
        self.training_data = {}

    def log_initial_capacity(self, models: Dict[str, nn.Module],
                           input_shapes: Dict[str, Tuple[int, ...]]):
        """Log initial model capacity.

        Args:
            models: Dictionary of models to analyze
            input_shapes: Input shapes for each model
        """
        analysis = analyze_model_capacity(models, input_shapes)
        self.capacity_data = analysis
        logger.info("Initial model capacity logged")

    def log_training_session(self, batch_size: int, num_iterations: int):
        """Log training session parameters.

        Args:
            batch_size: Batch size used
            num_iterations: Number of iterations
        """
        training_analysis = compute_training_flops(
            self.capacity_data, batch_size, num_iterations
        )
        self.training_data = training_analysis
        logger.info(f"Training session logged: batch_size={batch_size}, "
                   f"iterations={num_iterations}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all logged data.

        Returns:
            Dictionary with capacity and training summary
        """
        return {
            "capacity_analysis": self.capacity_data,
            "training_analysis": self.training_data
        }