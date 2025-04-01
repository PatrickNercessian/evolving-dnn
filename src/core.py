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
    
    # Get required shapes before making any modifications
    input_shape, output_shape = find_required_shapes(graph, reference_node)
    
    # Add a linear layer to the graph
    if operation == 'linear':
        # get the shape of the reference node from metadata
        reference_node_shape = reference_node.meta['tensor_meta'].shape
        # add a linear layer to the graph
        graph, new_node = add_linear(graph, reference_node, reference_node_shape[-1])
        # get the shape of the new node from the module
        new_node_shape = (graph.get_submodule(new_node.target).in_features, 
                         graph.get_submodule(new_node.target).out_features)

        print(f"New node shape: {new_node_shape}")

    graph.graph.lint()
    graph.recompile()
        
    # Fix the connections using pre-computed shapes
    adapt_connections(graph, new_node, new_node_shape, input_shape, output_shape)

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

def adapt_connections(graph, new_node, new_node_shape: tuple[int, int], input_shape, output_shape):
    """
    Adapts the connections to/from a node to ensure all connected nodes have compatible shapes.
    
    Args:
        graph: The FX graph
        new_node: The node whose connections need adaptation
        new_node_shape: The shape of the new node
        input_shape: The shape of the input node (pre-computed)
        output_shape: The shape of the output node (pre-computed)
    Returns:
        graph: The modified graph
    """
    
    # check if the input node output shapes are compatible with the node to adapt from
    if input_shape is not None:
        if input_shape[-1] != new_node_shape[0]: # last dimension of input node output shape
            print(f"Input node output shape {input_shape} is not compatible with the node to adapt from {new_node_shape}")

    # check if the output node input shapes are compatible with the node to adapt from
    if output_shape is not None:
        if output_shape[-1] != new_node_shape[1]:
            print(f"Output node input shape {output_shape} is not compatible with the node to adapt from {new_node_shape}")

