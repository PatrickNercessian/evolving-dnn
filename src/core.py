import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from add import add_linear
from utils import find_required_shapes 


def get_graph(model, input_shape):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
        input_shape: tuple specifying input tensor shape (batch, seq_len, dim)
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace with shape information
    """
        
    # Symbolically trace the model to get computation graph
    graph = torch.fx.symbolic_trace(model)
    
    # Perform shape propagation if input_shape is provided
    if input_shape is not None:
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

def add_node(graph, reference_node, operation: str):
    """
    Adds a new node to the graph after the reference node.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        operation: The operation to be performed by the new node
    Returns:
        graph: The modified graph
    """

    # Add a linear layer to the graph
    if operation == 'linear':
        graph, input_node = add_linear(graph, reference_node)
    
    # Fix the connections
    adapt_connections(graph, input_node)

    return graph

def remove_node(graph, reference_node):
    """
    Removes a node from the graph
    
    Args:
        graph: The FX graph
        reference_node: The node to remove
    Returns:
        graph: The modified graph
    """
    input_node = reference_node.args[0]
    
    # Remove the node from the graph
    reference_node.replace_all_uses_with(input_node)
    graph.graph.erase_node(reference_node)
    
    # Delete the submodule
    delattr(graph, reference_node.target)

    # Lint and recompile the graph
    graph.graph.lint()
    graph.recompile()

    return graph, input_node

def adapt_connections(graph, node_to_adapt_from):
    """
    Adapts the connections to/from a node to ensure all connected nodes have compatible shapes.
    
    Args:
        graph: The FX graph
        node_to_adapt_from: The node whose connections need adaptation
    Returns:
        graph: The modified graph
    """

    # get the required shapes for the node to adapt from
    input_shape, output_shape = find_required_shapes(graph, node_to_adapt_from)

    # list of output shapes from input nodes
    input_node_output_shapes = []
    # list of input shapes from output nodes
    output_node_input_shapes = []
    # get the required shapes of any input nodes
    for input_node in node_to_adapt_from.args[0].args:
        previous_input_shape, previous_output_shape = find_required_shapes(graph, input_node)
        input_node_output_shapes.append(previous_output_shape)
    # get the required shapes of any output nodes
    for output_node in node_to_adapt_from.users:
        following_input_shape, following_output_shape = find_required_shapes(graph, output_node)
        output_node_input_shapes.append(following_input_shape)

    # check if the input node output shapes are compatible with the node to adapt from
    for input_node_output_shape in input_node_output_shapes:
        if input_node_output_shape != input_shape:
            pass

    # check if the output node input shapes are compatible with the node to adapt from
    for output_node_input_shape in output_node_input_shapes:
        if output_node_input_shape != output_shape:
            pass

