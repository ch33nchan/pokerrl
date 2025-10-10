"""
Computational analysis utilities for model capacity and FLOPs estimation.

Provides exact parameter counting and FLOPs estimation for fair algorithm comparison:
- Parameter counting with detailed breakdown
- FLOPs estimation per forward/backward pass
- Memory usage analysis
- Training computational cost estimation
- Integration with enhanced manifest system
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import psutil
import time


@dataclass
class ModelCapacity:
    """Container for model capacity analysis."""
    # Parameter counts
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int

    # Memory usage
    parameter_size_mb: float
    activation_size_mb: float  # Estimated
    gradient_size_mb: float
    total_memory_mb: float

    # FLOPs estimation
    flops_per_forward: int
    flops_per_backward: int
    flops_per_update: int
    total_training_flops: int

    # Network architecture details
    layers: List[Dict[str, Any]]
    input_size: Tuple[int, ...]
    output_size: Tuple[int, ...]

    # Computational characteristics
    model_complexity_score: float
    inference_time_ms: float  # Measured


class ComputationalAnalyzer:
    """
    Computational analyzer for exact model capacity and FLOPs analysis.

    Provides detailed analysis for fair algorithm comparison as specified
    in the executive directive.
    """

    def __init__(self):
        """Initialize computational analyzer."""
        self.logger = logging.getLogger(__name__)

    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count parameters in a neural network model.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params
        }

    def estimate_flops_per_forward(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs per forward pass.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Estimated FLOPs per forward pass
        """
        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Hook to count operations
        flops = 0

        def hook_fn(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                # Linear layer: input_size * output_size * 2 (mult + add)
                in_features = input[0].shape[-1]
                out_features = output.shape[-1]
                batch_size = np.prod(input[0].shape[:-1])
                module_flops = batch_size * in_features * out_features * 2
                flops += module_flops

            elif isinstance(module, nn.Conv2d):
                # Conv2d: output_elements * kernel_elements * in_channels * 2
                in_channels = input[0].shape[1]
                out_channels = output.shape[1]
                kernel_size = module.kernel_size
                output_elements = np.prod(output.shape[2:])
                kernel_elements = np.prod(kernel_size)
                batch_size = input[0].shape[0]
                module_flops = batch_size * out_channels * in_channels * kernel_elements * output_elements * 2
                flops += module_flops

            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                # Activation functions: element-wise operations
                elements = np.prod(output.shape)
                flops += elements

            elif isinstance(module, nn.Dropout):
                # Dropout: element-wise operations
                elements = np.prod(output.shape)
                flops += elements

        # Register hooks
        hooks = []
        for module in model.modules():
            if list(module.children()):  # Skip container modules
                continue
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        try:
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return flops

    def estimate_flops_per_backward(self, forward_flops: int) -> int:
        """Estimate FLOPs per backward pass.

        Args:
            forward_flops: FLOPs per forward pass

        Returns:
            Estimated FLOPs per backward pass (typically ~2x forward)
        """
        # Backward pass is typically ~2x forward pass cost
        return forward_flops * 2

    def estimate_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...],
                              batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory usage for model and training.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (excluding batch)
            batch_size: Batch size

        Returns:
            Dictionary with memory usage in MB
        """
        # Parameter memory
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        # Activation memory (rough estimate)
        dummy_input = torch.randn((batch_size,) + input_shape)
        with torch.no_grad():
            output = model(dummy_input)

        activation_size = output.nelement() * output.element_size()

        # Gradient memory (same size as parameters for training)
        gradient_size = param_size

        # Total memory (with some overhead factor)
        total_memory = (param_size + activation_size + gradient_size) * 1.5

        return {
            'parameter_size_mb': param_size / (1024 * 1024),
            'activation_size_mb': activation_size / (1024 * 1024),
            'gradient_size_mb': gradient_size / (1024 * 1024),
            'total_memory_mb': total_memory / (1024 * 1024)
        }

    def measure_inference_time(self, model: nn.Module, input_shape: Tuple[int, ...],
                             num_trials: int = 100) -> float:
        """Measure actual inference time.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_trials: Number of trials for averaging

        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        dummy_input = torch.randn(input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(dummy_input)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_trials * 1000
        return avg_time_ms

    def analyze_model_capacity(self, models: Dict[str, nn.Module],
                             input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, ModelCapacity]:
        """Analyze capacity of multiple models.

        Args:
            models: Dictionary of model name to model
            input_shapes: Dictionary of model name to input shape

        Returns:
            Dictionary of model capacity analysis
        """
        results = {}

        for name, model in models.items():
            if name not in input_shapes:
                self.logger.warning(f"No input shape provided for model {name}")
                continue

            input_shape = input_shapes[name]

            # Count parameters
            param_counts = self.count_parameters(model)

            # Estimate FLOPs
            forward_flops = self.estimate_flops_per_forward(model, input_shape)
            backward_flops = self.estimate_flops_per_backward(forward_flops)
            update_flops = forward_flops + backward_flops

            # Estimate memory usage
            memory_usage = self.estimate_memory_usage(model, input_shape[1:])

            # Measure inference time
            try:
                inference_time = self.measure_inference_time(model, input_shape)
            except Exception as e:
                self.logger.warning(f"Could not measure inference time for {name}: {e}")
                inference_time = 0.0

            # Get network architecture details
            layers = self._extract_layer_info(model)

            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                param_counts['total_parameters'],
                forward_flops,
                len(layers)
            )

            # Create capacity object
            capacity = ModelCapacity(
                total_parameters=param_counts['total_parameters'],
                trainable_parameters=param_counts['trainable_parameters'],
                non_trainable_parameters=param_counts['non_trainable_parameters'],
                parameter_size_mb=memory_usage['parameter_size_mb'],
                activation_size_mb=memory_usage['activation_size_mb'],
                gradient_size_mb=memory_usage['gradient_size_mb'],
                total_memory_mb=memory_usage['total_memory_mb'],
                flops_per_forward=forward_flops,
                flops_per_backward=backward_flops,
                flops_per_update=update_flops,
                total_training_flops=update_flops,  # Per iteration
                layers=layers,
                input_size=input_shape,
                output_size=self._get_output_size(model, input_shape),
                model_complexity_score=complexity_score,
                inference_time_ms=inference_time
            )

            results[name] = capacity

        return results

    def _extract_layer_info(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Extract information about model layers.

        Args:
            model: PyTorch model

        Returns:
            List of layer information
        """
        layers = []

        for name, module in model.named_modules():
            if list(module.children()):  # Skip container modules
                continue

            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
            }

            # Add type-specific information
            if isinstance(module, nn.Linear):
                layer_info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                })
            elif isinstance(module, nn.Conv2d):
                layer_info.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'bias': module.bias is not None
                })

            layers.append(layer_info)

        return layers

    def _get_output_size(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get output size of model.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Output tensor shape
        """
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        return tuple(output.shape)

    def _calculate_complexity_score(self, num_parameters: int, flops: int, num_layers: int) -> float:
        """Calculate model complexity score.

        Args:
            num_parameters: Number of parameters
            flops: FLOPs per forward pass
            num_layers: Number of layers

        Returns:
            Complexity score (higher = more complex)
        """
        # Normalize and combine metrics
        param_score = np.log10(max(num_parameters, 1))
        flops_score = np.log10(max(flops, 1))
        layer_score = np.log10(max(num_layers, 1))

        # Weighted combination
        complexity_score = 0.4 * param_score + 0.4 * flops_score + 0.2 * layer_score
        return complexity_score

    def estimate_training_cost(self, capacity: ModelCapacity, iterations: int,
                             batch_size: int) -> Dict[str, Any]:
        """Estimate computational cost for training.

        Args:
            capacity: Model capacity analysis
            iterations: Number of training iterations
            batch_size: Batch size

        Returns:
            Training cost estimates
        """
        # FLOPs for entire training
        total_training_flops = capacity.flops_per_update * iterations

        # Time estimates (rough, based on typical hardware performance)
        # Assuming ~10 GFLOPs/s for CPU, ~1000 GFLOPs/s for GPU
        cpu_gflops_per_second = 10
        gpu_gflops_per_second = 1000

        cpu_time_hours = total_training_flops / (cpu_gflops_per_second * 1e9 * 3600)
        gpu_time_hours = total_training_flops / (gpu_gflops_per_second * 1e9 * 3600)

        # Memory estimates
        peak_memory_mb = capacity.total_memory_mb * 2  # With gradients and optimizer states

        return {
            'total_training_flops': total_training_flops,
            'cpu_time_hours': cpu_time_hours,
            'gpu_time_hours': gpu_time_hours,
            'peak_memory_mb': peak_memory_mb,
            'energy_estimate_kwh': gpu_time_hours * 0.3,  # Rough estimate
            'cost_per_iteration': capacity.flops_per_update,
            'cost_per_epoch': capacity.flops_per_update * (iterations // 10)  # Assuming 10 iterations per epoch
        }

    def compare_models(self, capacities: Dict[str, ModelCapacity]) -> Dict[str, Any]:
        """Compare multiple models on various metrics.

        Args:
            capacities: Dictionary of model capacity analyses

        Returns:
            Comparison results
        """
        if not capacities:
            return {}

        comparison = {
            'parameter_comparison': {},
            'flops_comparison': {},
            'memory_comparison': {},
            'complexity_ranking': [],
            'efficiency_analysis': {}
        }

        # Parameter comparison
        for name, capacity in capacities.items():
            comparison['parameter_comparison'][name] = capacity.total_parameters

        # FLOPs comparison
        for name, capacity in capacities.items():
            comparison['flops_comparison'][name] = capacity.flops_per_forward

        # Memory comparison
        for name, capacity in capacities.items():
            comparison['memory_comparison'][name] = capacity.total_memory_mb

        # Complexity ranking
        complexity_scores = [(name, cap.model_complexity_score) for name, cap in capacities.items()]
        complexity_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['complexity_ranking'] = complexity_scores

        # Efficiency analysis (performance per parameter)
        for name, capacity in capacities.items():
            if capacity.total_parameters > 0:
                efficiency = capacity.flops_per_forward / capacity.total_parameters
            else:
                efficiency = 0
            comparison['efficiency_analysis'][name] = efficiency

        return comparison

    def generate_capacity_report(self, capacities: Dict[str, ModelCapacity],
                                output_path: str) -> str:
        """Generate comprehensive capacity report.

        Args:
            capacities: Dictionary of model capacity analyses
            output_path: Path to save report

        Returns:
            Path to generated report
        """
        report_lines = []
        report_lines.append("MODEL CAPACITY ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of models: {len(capacities)}")
        report_lines.append("")

        # Individual model details
        for name, capacity in capacities.items():
            report_lines.append(f"MODEL: {name}")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Parameters: {capacity.total_parameters:,}")
            report_lines.append(f"Trainable Parameters: {capacity.trainable_parameters:,}")
            report_lines.append(f"FLOPs per Forward: {capacity.flops_per_forward:,}")
            report_lines.append(f"FLOPs per Update: {capacity.flops_per_update:,}")
            report_lines.append(f"Parameter Memory: {capacity.parameter_size_mb:.2f} MB")
            report_lines.append(f"Total Memory: {capacity.total_memory_mb:.2f} MB")
            report_lines.append(f"Complexity Score: {capacity.model_complexity_score:.2f}")
            report_lines.append(f"Inference Time: {capacity.inference_time_ms:.2f} ms")
            report_lines.append("")

        # Comparison
        if len(capacities) > 1:
            comparison = self.compare_models(capacities)
            report_lines.append("COMPARISON SUMMARY")
            report_lines.append("-" * 30)
            report_lines.append("Complexity Ranking:")
            for name, score in comparison['complexity_ranking']:
                report_lines.append(f"  {name}: {score:.2f}")
            report_lines.append("")

        # Write report
        report_content = "\n".join(report_lines)
        with open(output_path, 'w') as f:
            f.write(report_content)

        self.logger.info(f"Generated capacity report at {output_path}")
        return output_path

    def get_manifest_integration_data(self, capacities: Dict[str, ModelCapacity]) -> Dict[str, Any]:
        """Get data formatted for manifest integration.

        Args:
            capacities: Dictionary of model capacity analyses

        Returns:
            Formatted data for manifest
        """
        if not capacities:
            return {}

        # For single model or multi-model algorithms
        total_params = sum(cap.total_parameters for cap in capacities.values())
        total_flops = sum(cap.flops_per_forward for cap in capacities.values())

        return {
            'params_count': total_params,
            'flops_est': total_flops,
            'memory_usage_mb': sum(cap.total_memory_mb for cap in capacities.values()),
            'inference_time_ms': np.mean([cap.inference_time_ms for cap in capacities.values()]),
            'complexity_score': np.mean([cap.model_complexity_score for cap in capacities.values()]),
            'model_details': {
                name: {
                    'parameters': cap.total_parameters,
                    'flops': cap.flops_per_forward,
                    'memory_mb': cap.total_memory_mb
                } for name, cap in capacities.items()
            }
        }