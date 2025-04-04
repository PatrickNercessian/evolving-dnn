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
    
    # Calculate padding needed to reach target size
    padding = target_size - input_size
    
    # Create circular padding layer
    pad_layer = nn.CircularPad1d((0, padding))
    graph.add_submodule(name, pad_layer)
    
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

def add_skip(graph, reference_node, first_node):
    """
    Adds a skip connection to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        first_node: The node to skip
    Returns:
        graph: The modified graph
    """
    # Create skip connection with unique name
    name = get_unique_name(graph, 'skip')
    
    # Add skip connection node after reference_node
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_function(
            function_name=name,
            args=(reference_node, first_node),
            kwargs={},
        )
    
    # Update connections
    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)

    return graph, new_node

