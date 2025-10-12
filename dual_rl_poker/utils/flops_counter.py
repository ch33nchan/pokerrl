"""
FLOPs estimation utility for neural network architectures.

Provides approximate FLOP counts for MLP forward and backward passes
to enable fair capacity comparison across algorithms.
"""

from typing import List


def estimate_flops(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    batch_size: int = 1,
    include_backward: bool = True
) -> int:
    """
    Estimate FLOPs for a single forward (and optionally backward) pass through an MLP.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        batch_size: Batch size for computation
        include_backward: Whether to include backward pass FLOPs
        
    Returns:
        Estimated FLOPs per update
    """
    forward_flops = 0
    
    # Build layer dimensions
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    
    # Forward pass FLOPs
    for i in range(len(layer_dims) - 1):
        input_size = layer_dims[i]
        output_size = layer_dims[i + 1]
        
        # Matrix multiplication: batch_size * input_size * output_size
        # Plus bias addition: batch_size * output_size
        layer_flops = batch_size * (input_size * output_size + output_size)
        forward_flops += layer_flops
    
    total_flops = forward_flops
    
    # Backward pass approximately 2x forward pass FLOPs
    if include_backward:
        total_flops += 2 * forward_flops
    
    return total_flops


def estimate_algorithm_flops(algorithm_name: str, **kwargs) -> int:
    """
    Estimate FLOPs for specific algorithm architectures.
    
    Args:
        algorithm_name: Name of the algorithm
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Estimated FLOPs per update
    """
    if algorithm_name == "tabular_cfr":
        return 0  # No neural network operations
    
    elif algorithm_name == "deep_cfr":
        # Deep CFR has separate regret and strategy networks
        single_network_flops = estimate_flops(**kwargs)
        return 2 * single_network_flops  # Two networks
    
    elif algorithm_name == "sd_cfr":
        # SD-CFR has single regret network
        return estimate_flops(**kwargs)
    
    elif algorithm_name == "armac":
        # ARMAC has actor, critic, and regret networks
        single_network_flops = estimate_flops(**kwargs)
        return 3 * single_network_flops  # Three networks
    
    else:
        # Default to single network
        return estimate_flops(**kwargs)


def get_parameter_count(layer_dims: List[int]) -> int:
    """
    Calculate parameter count for MLP with given layer dimensions.
    
    Args:
        layer_dims: List of layer dimensions [input, hidden1, hidden2, ..., output]
        
    Returns:
        Total parameter count
    """
    param_count = 0
    
    for i in range(len(layer_dims) - 1):
        input_size = layer_dims[i]
        output_size = layer_dims[i + 1]
        
        # Weights: input_size * output_size
        # Biases: output_size
        layer_params = input_size * output_size + output_size
        param_count += layer_params
    
    return param_count


def get_algorithm_parameter_count(
    algorithm_name: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int
) -> int:
    """
    Get parameter count for specific algorithm architectures.
    
    Args:
        algorithm_name: Name of the algorithm
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        
    Returns:
        Total parameter count
    """
    if algorithm_name == "tabular_cfr":
        return 0
    
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    single_network_params = get_parameter_count(layer_dims)
    
    if algorithm_name == "deep_cfr":
        return 2 * single_network_params  # Regret + strategy networks
    elif algorithm_name == "sd_cfr":
        return single_network_params  # Single regret network
    elif algorithm_name == "armac":
        return 3 * single_network_params  # Actor + critic + regret networks
    else:
        return single_network_params