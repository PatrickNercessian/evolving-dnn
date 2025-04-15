import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import random


def find_required_shapes(node):
    """
    Finds the required shapes for a given node in the graph.
    
    Args:
        graph: The FX graph
        node: The node to find the required shapes for
    Returns:
        input_shape: The required shape of the input nodes, can be a list of shapes if the node has multiple inputs
        output_shape: The required shape of the output node
    """
    # if placeholder
    if node.op == 'placeholder':
        return (None, tuple(node.meta['tensor_meta'].shape))
    else:
        # required input shape of any given node is found in the meta of any node that feeds into it
        # check the node args for the nodes that feed into the current node
        def get_shape(arg):
            if isinstance(arg, torch.fx.Node):
                try:
                    return list(arg.meta['tensor_meta'].shape)
                except:
                    print(f"Error getting shape for node {arg}")
                    return None
            elif isinstance(arg, list):
                return [get_shape(sub_arg) for sub_arg in arg if isinstance(sub_arg, torch.fx.Node) or isinstance(sub_arg, list)]
            else:
                return None
        
        # Get shapes of all arguments
        input_shape = []
        for arg in node.args:
            # only consider the arg if it is a node or a list of nodes
            if isinstance(arg, torch.fx.Node) or isinstance(arg, list):
                shape = get_shape(arg)
                if shape is not None:
                    input_shape.append(shape)
        input_shape = input_shape[0] if len(input_shape) == 1 else input_shape if input_shape else None
        
        # required output shape can be found in the meta of the node
        output_shape = list(node.meta['tensor_meta'].shape)
        return (input_shape, output_shape)

def get_unique_name(graph, base_name: str) -> str:
    """
    Generates a unique name for a node in the graph by appending incrementing numbers.
    
    Args:
        graph: The FX graph
        base_name: The base name to use (e.g., 'linear', 'pool', 'repeat')
    Returns:
        str: A unique name for the node
    """
    # Get list of all module names in graph
    module_names = [node.target for node in graph.graph.nodes if node.op == "call_module"]
    
    # Try base name first
    name = base_name
    counter = 1
    
    # Keep incrementing counter until we find a unique name
    while name in module_names:
        name = f"{base_name}_{counter}"
        counter += 1
    
    return name

def add_specific_node(graph, reference_node, module):
    """
    Helper function to add a new node to the graph after the reference node.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        module: The PyTorch module to add
    Returns:
        graph: The modified graph
        new_node: The newly added node
    """
    name = get_unique_name(graph, module.__class__.__name__)
    graph.add_submodule(name, module)
    
    # Add node after reference_node
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
