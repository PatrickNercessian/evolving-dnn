import torch
import torch.nn as nn
import logging

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