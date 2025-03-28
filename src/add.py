import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import random

def add_linear(graph, reference_node, mutate: str = 'input'):
    """
    Adds a linear layer to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        mutate: The type of mutation to perform on the new linear layer input: 'input', 'output', or 'both'
    Returns:
        graph: The modified graph
    """
    # Get all nodes in the graph
    nodes = list(graph.graph.nodes)
    
    # Find a unique name for the new node
    name = f"node_{len(nodes)}"
    while name in graph.graph._modules:
        name = f"node_{len(nodes) + 1}"

    # Assign random shape to the new linear layer
    if mutate == 'input':
        shape = (random.randint(1, 1000), reference_node.shape[1])
    elif mutate == 'output':
        shape = (reference_node.shape[0], random.randint(1, 1000))
    elif mutate == 'both':
        shape = (random.randint(1, 1000), random.randint(1, 1000))
    else:
        raise ValueError("Invalid mutation type")

    # Add the linear layer to the graph
    graph.add_submodule(name, nn.Linear(shape[1], shape[0]))
    
    # Update the graph connections
    new_node = graph.graph.call_module(
                module_name=name,
                args=(reference_node,),
                kwargs={},
            )


    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)

    graph.graph.lint()
    graph.recompile()
        
    return graph, name
