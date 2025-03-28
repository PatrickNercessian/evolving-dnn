import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import random

def add_linear(graph, reference_node, reference_node_shape: tuple[int, int]):
    """
    Adds a linear layer to the graph after the reference node.

    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        reference_node_shape: The shape of the reference node
    Returns:
        graph: The modified graph
    """
    # Get all nodes in the graph
    nodes = list(graph.graph.nodes)
    
    # Find a unique name for the new node
    name = f"node_{len(nodes)}"
    # get list of all module names in graph
    module_names = [node.target for node in graph.graph.nodes if node.op == "call_module"]
    while name in module_names:
        name = f"node_{len(nodes) + 1}"


    # Assign random shape to the new linear layer
    shape = (reference_node_shape[0], random.randint(1, 1000))

    # Add the linear layer to the graph
    graph.add_submodule(name, nn.Linear(shape[0], shape[1]))
    
    # Update the graph connections
    new_node = graph.graph.call_module(
                module_name=name,
                args=(reference_node,),
                kwargs={},
            )


    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)

    return graph, new_node
