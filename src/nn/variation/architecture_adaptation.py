import torch
import torch.nn as nn
import logging
import math

from src.nn.variation.utils import add_specific_node


def _adapt_tensor_size(graph, node, current_size: int, target_size: int, target_user=None):
    """
    Helper function to adapt a tensor's size using repeat_interleave, circular padding, or adaptive pooling.
    
    Args:
        graph: The FX graph
        node: The node to adapt
        current_size: Current size (integer)
        target_size: Target size (integer)
        target_user: Optional specific node that should use the new node
    Returns:
        graph: The modified graph
        adapted_node: The node after adaptation
    """
    if current_size < target_size:
        length_multiplier = target_size // current_size
        
        if length_multiplier > 1:
            remainder = target_size % current_size
        
            # First repeat the tensor as many times as possible
            graph, repeat_node = add_specific_node(
                graph, 
                node, 
                torch.repeat_interleave, 
                kwargs={"repeats": length_multiplier, "dim": 1},
                target_user=target_user  # Intermediate node
            )
            logging.debug(f"Added repeat node {repeat_node.name} after node {node.name}, repeats: {length_multiplier}")

            if remainder > 0:
                # Then use circular padding for the remainder
                graph, adapted_node = add_specific_node(
                    graph, 
                    repeat_node, 
                    nn.CircularPad1d((0, remainder)),
                    target_user=target_user
                )
                logging.debug(f"Added circular pad node {adapted_node.name} after repeat node {repeat_node.name}, remainder: {remainder}")
            else:
                adapted_node = repeat_node
        else:
            # If we only need to wrap once, just use circular padding
            graph, adapted_node = add_specific_node(
                graph, 
                node, 
                nn.CircularPad1d((0, target_size - current_size)),
                target_user=target_user
            )
            logging.debug(f"Added circular pad node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")
    else:
        # Need to decrease size - use adaptive pooling
        graph, adapted_node = add_specific_node(
            graph, 
            node, 
            nn.AdaptiveAvgPool1d(target_size),
            target_user=target_user
        )
        logging.debug(f"Added adaptive avg pool node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")

    return graph, adapted_node

class ReshapeModule(nn.Module):
    """A PyTorch module for reshaping tensors to a specific target size."""
    
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, x):
        return x.reshape(-1, *self.target_size)

def adapt_node_shape(graph, node, current_size, target_size, target_user=None, try_linear_adapter=True):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
        try_linear_adapter: If True, try using a single linear layer instead of flatten->adapt->unflatten pattern
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Convert current_size and target_size to tuples if they are not already
    current_size = tuple(current_size)
    target_size = tuple(target_size)
    
    if current_size == target_size:
        return graph, node
    
    current_dims = len(current_size)
    target_dims = len(target_size)
    
    # Handle 1D to 1D case directly
    if current_dims == 1 and target_dims == 1:
        if try_linear_adapter:
            return add_specific_node(
                graph,
                node,
                nn.Linear(current_size[0], target_size[0]),
                target_user=target_user
            )
        return _adapt_tensor_size(graph, node, current_size[0], target_size[0], target_user=target_user)
    
    # Calculate total elements
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    
    # If total elements are the same, just reshape and return
    if current_total == target_total:
        return add_specific_node(
            graph,
            node,
            ReshapeModule(target_size),
            target_user=target_user
        )
    
    if try_linear_adapter and current_size[:-1] == target_size[:-1]:  # Only use linear adapter if all but last dims are the same
        return add_specific_node(
            graph,
            node,
            nn.Linear(current_size[-1], target_size[-1]),
            target_user=target_user
        )
    
    # Step 1: Flatten if starting from multi-dimensional (2+:1 or 2+:2+)
    if current_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Flatten(start_dim=1, end_dim=-1),
            target_user=target_user
        )
    
    # Step 2: Adapt tensor size (total elements differ, so this is always needed)
    graph, node = _adapt_tensor_size(
        graph, 
        node, 
        current_total, 
        target_total, 
        target_user=target_user
    )
    
    # Step 3: Unflatten if ending with multi-dimensional (1:2+ or 2+:2+)
    if target_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Unflatten(dim=1, unflattened_size=target_size),
            target_user=target_user
        )
    
    return graph, node