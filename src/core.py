import random

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from individual_graph_module import IndividualGraphModule
from utils import find_required_shapes, add_specific_node, add_skip_connection, adapt_node_shape, add_branch_nodes


def get_graph(model: nn.Module, input_shape: tuple):

    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
        input_shape: tuple specifying input tensor shape (batch, seq_len, dim)
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace with shape information
    """
        
    # Symbolically trace the model to get computation graph
    graph = IndividualGraphModule(torch.fx.symbolic_trace(model))
    
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

def add_node(graph: torch.fx.GraphModule, reference_node: torch.fx.Node, operation: str, **kwargs):
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
    input_shape, output_shape = find_required_shapes(reference_node)

    # Add a linear layer to the graph
    if operation == 'linear':
        # get the shape of the reference node from metadata
        reference_node_shape = reference_node.meta['tensor_meta'].shape

        new_node_shape = (reference_node_shape[-1], random.randint(1, 1000))  # Assign random shape to the new linear layer
        print(f"New node shape: {new_node_shape}")
        graph, new_node = add_specific_node(graph, reference_node, nn.Linear(new_node_shape[0], new_node_shape[1]))

    # Add an adaptive pooling layer
    elif operation == 'pool':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for pool operation")
            
        graph, new_node = add_specific_node(graph, reference_node, nn.AdaptiveAvgPool1d(target_size))
        
        # Get shape of the new node
        new_node_shape = (target_size, target_size)  # For 1D pooling, in/out are same
        
    # Add a repeat/broadcast layer
    elif operation == 'repeat':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for repeat operation")
            
        # Get input size from reference node
        input_size = reference_node.meta['tensor_meta'].shape[-1]
        graph, new_node = add_specific_node(graph, reference_node, nn.CircularPad1d((0, target_size - input_size)))
        
        # Get shape of the new node
        new_node_shape = (target_size, target_size)  # For repeat, in/out are same

    # Add a flatten layer, that flattens every dimension except the batch dimension 
    elif operation == 'flatten':
        graph, new_node = add_specific_node(graph, reference_node, nn.Flatten(start_dim=1, end_dim=-1))
        new_node_shape = (reference_node.meta['tensor_meta'].shape[0], -1)

    # Add a skip connection
    elif operation == 'skip':
        first_node = kwargs.get('first_node')
        if first_node is None:
            raise ValueError("first_node must be provided for skip operation")
        second_node = reference_node
        graph, new_node = add_skip_connection(graph, second_node, first_node)
        new_node_shape = reference_node.meta['tensor_meta'].shape

    # Add branch node, that branches the input into two paths
    elif operation == 'branch':
        # Special case where we add two new nodes to the graph after the reference node BUT DO NOT CONNECT THEM TO USERS YET
        # We then add a skip connection between the two new nodes
        # Then we connect the skip connection to the users of the reference node
        # y = f(x) -> y = h(g(x), k(x))
        
        # Get the shape of the reference node from metadata
        reference_node_shape = reference_node.meta['tensor_meta'].shape
        branch_node_output_features = random.randint(1, 1000)

        # Create the branch modules with random shapes
        branch1_module = nn.Linear(reference_node_shape[-1], branch_node_output_features)
        branch2_module = nn.Linear(reference_node_shape[-1], branch_node_output_features)
        
        # Use the utility function to add branch nodes
        graph, new_node, new_node_shape = add_branch_nodes(graph, reference_node, branch1_module, branch2_module)

    graph.graph.lint()
    graph.recompile()
        
    # Fix the connections using pre-computed shapes
    adapt_connections(graph, new_node, new_node_shape, input_shape, output_shape)

    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    example_input = torch.randn(next(iter(graph.graph.nodes)).meta['tensor_meta'].shape)
    try:
        ShapeProp(graph).propagate(example_input)
    except Exception as e:
        print(f"Error during shape propagation: {e}")
        print(f"Graph: {graph}")

    return graph

def remove_node(graph: torch.fx.GraphModule, reference_node: torch.fx.Node):
    """
    Removes a node from the graph
    
    Args:
        graph: The FX graph
        reference_node: The node to remove
    Returns:
        graph: The modified graph
    """
    input_node = reference_node.args[0]
    
    # Get shapes before removing node
    output_left_shape = list(input_node.meta['tensor_meta'].shape)
    input_right_shape = reference_node.meta['tensor_meta'].shape
    
    # Remove the node from the graph
    reference_node.replace_all_uses_with(input_node)
    
    graph.graph.erase_node(reference_node)
    
    graph.delete_all_unused_submodules()

    # Adapt connections between input node and its new users
    # In this case, the new node is the input node, so new_node input is always correct already
    graph = adapt_connections(graph, new_node=input_node, 
                              new_node_shape=(output_left_shape[-1], output_left_shape[-1]), 
                              input_shape=output_left_shape, output_shape=input_right_shape)

    # Lint and recompile the graph
    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    example_input = torch.randn(next(iter(graph.graph.nodes)).meta['tensor_meta'].shape)
    ShapeProp(graph).propagate(example_input)

    return graph, input_node

def adapt_connections(
    graph: torch.fx.GraphModule,
    new_node: torch.fx.Node,
    new_node_shape: tuple | None,
    input_shape: tuple,
    output_shape: tuple
):
    """
    Adapts the connections to/from a node to ensure all connected nodes have compatible shapes.
    
    Args:
        graph: The FX graph
        new_node: The node whose connections need adaptation
        new_node_shape: The shape of the new node, can be None if the new node is a skip connection
        input_shape: The shape of the input node (pre-computed)
        output_shape: The shape of the output node (pre-computed)
    Returns:
        graph: The modified graph
    """
    
    # Special handling for skip connections (torch.add operations)
    if new_node.target == torch.add:
        # Get shapes of both input nodes
        first_node = new_node.args[0]
        second_node = new_node.args[1]
        first_shape = first_node.meta['tensor_meta'].shape
        second_shape = second_node.meta['tensor_meta'].shape
        
        # If shapes don't match, adapt both inputs to the larger shape
        if first_shape[-1] != second_shape[-1]:
            target_size = max(first_shape[-1], second_shape[-1])
            
            # Adapt first node if needed
            if first_shape[-1] != target_size:
                graph, first_node = adapt_node_shape(graph, first_node, first_shape[-1], target_size)
            
            # Adapt second node if needed
            if second_shape[-1] != target_size:
                graph, second_node = adapt_node_shape(graph, second_node, second_shape[-1], target_size)
            
            # Update the skip connection node's args
            new_node.args = (first_node, second_node)
            
            # Update new_node_shape to reflect the adapted size
            new_node_shape = (target_size, target_size)
    
    # For non-skip connections, handle input shape compatibility
    elif input_shape is not None:
        if input_shape[-1] != new_node_shape[0]: # last dimension of input node output shape
            print(f"Input node output shape {input_shape[-1]} is not compatible with the node to adapt from {new_node_shape[0]}")
            graph, new_node = adapt_node_shape(graph, new_node.args[0], input_shape[-1], new_node_shape[0])

    # Handle output shape compatibility for all nodes
    if output_shape is not None:
        if output_shape[-1] != new_node_shape[1]:
            print(f"Output node input shape {output_shape[-1]} is not compatible with the node to adapt from {new_node_shape[1]}")
            graph, new_node = adapt_node_shape(graph, new_node, new_node_shape[1], output_shape[-1])
    
    return graph
