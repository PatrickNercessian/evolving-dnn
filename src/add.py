import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import random
from utils import get_unique_name

def add_linear(graph, reference_node, reference_node_output_shape: int):
    """
    Adds a linear layer to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        reference_node_output_shape: The output shape of the reference node
    Returns:
        graph: The modified graph
    """
    # Find a unique name for the new node
    name = get_unique_name(graph, 'linear')

    # Assign random shape to the new linear layer
    shape = (reference_node_output_shape, random.randint(1, 1000))

    # Add the linear layer to the graph
    graph.add_submodule(name, nn.Linear(shape[0], shape[1]))
    
    # Update the graph connections
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(reference_node,),
            kwargs={},
        )

    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)

    return graph, new_node

def add_pool(graph, reference_node, target_size: int):
    """
    Adds an adaptive pooling layer to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        target_size: The target output size for the pooling layer
    Returns:
        graph: The modified graph
    """
    # Create adaptive pooling layer with unique name
    name = get_unique_name(graph, 'adaptive_pool')
    pool_layer = nn.AdaptiveAvgPool1d(target_size)
    graph.add_submodule(name, pool_layer)
    
    # Add pooling node after reference_node
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(reference_node,),
            kwargs={},
        )
    
    # Update connections
    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)
    
    return graph, new_node

def add_repeat(graph, reference_node, input_size: int, target_size: int):
    """
    Adds a repeat/broadcast layer to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        input_size: The input size of the reference node
        target_size: The target output size
    Returns:
        graph: The modified graph
    """
    # Create repeat layer with unique name
    name = get_unique_name(graph, 'repeat')
    
    # Create repeat layer
    class RepeatLayer(nn.Module):
        def __init__(self, input_size, target_size):
            super().__init__()
            self.repeat_factor = target_size // input_size
            if target_size % input_size != 0:
                raise ValueError(f"Target size {target_size} must be divisible by input size {input_size}")
        
        def forward(self, x):
            return x.repeat(1, 1, self.repeat_factor)
    
    repeat_layer = RepeatLayer(input_size, target_size)
    graph.add_submodule(name, repeat_layer)
    
    # Add repeat node after reference_node
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(reference_node,),
            kwargs={},
        )
    
    # Update connections
    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)
    
    return graph, new_node
