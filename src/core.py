import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp


def get_graph(model):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace
    """
        
    # Symbolically trace the model to get computation graph
    graph = torch.fx.symbolic_trace(model)
    dict = {}
            
    # TODO: add shape propagation here instead of in shape_prop

    return graph

def shape_prop(graph, input_shape):
    """
    Propagate shapes through the graph to get tensor metadata for each node
    
    Args:
        graph: The FX graph
        input_shape: The input tensor shape (batch, seq_len, dim)
    """
    # Create example input
    example_input = torch.randn(input_shape)
    
    # Get the first node (should be placeholder/input)
    placeholder = next(iter(graph.graph.nodes))
    placeholder.meta['tensor_meta'] = {
        'dtype': example_input.dtype,
        'shape': input_shape,
        'requires_grad': example_input.requires_grad
    }
    
    # Run shape propagation
    ShapeProp(graph).propagate(example_input)
    
    return graph


def add_node(graph, node, module, name=None):
    """
    Adds a new node to the graph and handles module registration
    
    Args:
        graph: The FX graph
        node: The target node to insert after
        module: The new module to add
        name: Optional name for the module (default: auto-generated)
    Returns:
        graph: The modified graph
        name: The name of the added module
    """
    
    # TODO: modules should be pre-defined and dynamically built
    
    graph.add_submodule(name, module)
    # Add the node to the graph
    with graph.graph.inserting_after(node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(node,),
            kwargs={},
        )
        node.replace_all_uses_with(new_node)
        new_node.args = (node,)

    graph.graph.lint()
    graph.recompile()
        
    return graph, name

def remove_node(graph, node):
    """
    Removes a node from the graph and cleans up its module if it was dynamically added
    
    Args:
        graph: The FX graph
        node: The node to remove
    Returns:
        graph: The modified graph
    """

    # TODO: After a node is removed, the following nodes need to have their shape changed

    # Get the input node that feeds into this node
    input_node = node.args[0]
    
    # Reconnect all users of this node to its input before removing
    node.replace_all_uses_with(input_node)
    
    # Remove the node from the graph
    graph.graph.erase_node(node)
    
    # Remove the submodule from the graph
    if node.op == "call_module":
        submodule_name = node.target
        if hasattr(graph, submodule_name):
            delattr(graph, submodule_name)

    graph.graph.lint()
    graph.recompile()
    return graph







