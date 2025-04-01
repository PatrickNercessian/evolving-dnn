import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from add import add_linear, add_pool, add_repeat
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

def add_node(graph, reference_node, operation: str, **kwargs):
    """
    Adds a new node to the graph after the reference node.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        operation: The operation to be performed by the new node ('linear', 'pool', or 'repeat')
        **kwargs: Additional arguments for specific operations
            - For 'pool': target_size
            - For 'repeat': target_size
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
    
    # Add an adaptive pooling layer
    elif operation == 'pool':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for pool operation")
            
        graph, new_node = add_pool(graph, reference_node, target_size)
        
        # Get shape of the new node
        new_node_shape = (target_size, target_size)  # For 1D pooling, in/out are same
        
    # Add a repeat/broadcast layer
    elif operation == 'repeat':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for repeat operation")
            
        # Get input size from reference node
        input_size = reference_node.meta['tensor_meta'].shape[-1]
        
        graph, new_node = add_repeat(graph, reference_node, input_size, target_size)
        
        # Get shape of the new node
        new_node_shape = (target_size, target_size)  # For repeat, in/out are same

    graph.graph.lint()
    graph.recompile()
        
    # Fix the connections using pre-computed shapes
    adapt_connections(graph, new_node, new_node_shape, input_shape, output_shape)

    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    example_input = torch.randn(next(iter(graph.graph.nodes)).meta['tensor_meta'].shape)
    ShapeProp(graph).propagate(example_input)

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
            print(f"Input node output shape {input_shape[-1]} is not compatible with the node to adapt from {new_node_shape[0]}")
            
            # Handle input size mismatch by adapting the input node
            if input_shape[-1] > new_node_shape[0]:
                # Input is larger - add adaptive pooling to reduce size
                graph, new_node = add_pool(graph, new_node.args[0], new_node_shape[0])
                
            elif input_shape[-1] < new_node_shape[0]:
                # Input is smaller - add repeat/broadcast to increase size
                graph, new_node = add_repeat(graph, new_node.args[0], input_shape[-1], new_node_shape[0])

    # check if the output node input shapes are compatible with the node to adapt from
    if output_shape is not None:
        if output_shape[-1] != new_node_shape[1]:
            print(f"Output node input shape {output_shape[-1]} is not compatible with the node to adapt from {new_node_shape[1]}")
            
            # Handle output size mismatch
            if new_node_shape[1] > output_shape[-1]:
                # New node output is larger - add adaptive pooling
                graph, new_node = add_pool(graph, new_node, output_shape[-1])
                
            elif new_node_shape[1] < output_shape[-1]:
                # New node output is smaller - add repeat/broadcast
                graph, new_node = add_repeat(graph, new_node, new_node_shape[1], output_shape[-1])
    
    return graph




