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
        input_shape: tuple specifying input tensor shape (batch, seq_len, dim)  # SHAPE NOTE: input_shape includes batch dimension
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace with shape information
    """
        
    # Symbolically trace the model to get computation graph
    graph = IndividualGraphModule(torch.fx.symbolic_trace(model))
    
    # Perform shape propagation if input_shape is provided
    if input_shape is not None:
        # Create example input
        example_input = torch.randn(input_shape)  # SHAPE NOTE: Using full shape including batch dimension
        
        # Get the first node (should be placeholder/input)
        placeholder = next(iter(graph.graph.nodes))
        placeholder.meta['tensor_meta'] = {
            'dtype': example_input.dtype,
            'shape': input_shape,  # SHAPE NOTE: Storing full shape including batch dimension
            'requires_grad': example_input.requires_grad
        }
        
        # Run shape propagation
        ShapeProp(graph).propagate(example_input)  # SHAPE NOTE: Shape propagation uses full shape including batch dimension
    
    return graph

def add_node(graph: torch.fx.GraphModule, reference_node: torch.fx.Node, operation: str, **kwargs):
    """
    Adds a new node to the graph after the reference node.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        operation: The operation to be performed by the new node ('linear', 'pool', 'repeat', etc.)
        **kwargs: Additional arguments for specific operations
            - For 'pool': target_size
            - For 'repeat': target_size
            - For 'skip': first_node
    Returns:
        graph: The modified graph
    """
    
    # Get required shapes before making any modifications
    input_shape, output_shape = find_required_shapes(reference_node)  # SHAPE NOTE: Returns shapes with batch dimension included

    # Helper function to get feature dimensions (excluding batch dimension)
    def get_feature_dims(shape):
        if shape is None:
            return None
        # Skip the batch dimension (first dimension)
        if len(shape) > 1:
            return tuple(shape[1:])
        else:
            return tuple(shape)

    # Get feature dimensions from reference node (excluding batch)
    ref_feature_shape = get_feature_dims(reference_node.meta['tensor_meta'].shape)
    
    # Add a linear layer to the graph
    if operation == 'linear':
        # Use only the last feature dimension for input size
        # input_size = ref_feature_shape[-1]
        input_size = random.randint(1, 1000)
        
        output_size = random.randint(1, 1000)
        
        # Create separate input and output feature shapes
        # Make sure to create new tuples as tuples are immutable
        if len(ref_feature_shape) == 1:
            new_node_input_shape = (input_size,)
            new_node_output_shape = (output_size,)
        else:
            # Create a new tuple with the last dimension changed
            new_node_input_shape = tuple(list(ref_feature_shape[:-1]) + [input_size])
            new_node_output_shape = tuple(list(ref_feature_shape[:-1]) + [output_size])
        
        print(f"New linear layer: input={input_size}, output={output_size}")
        
        graph, new_node = add_specific_node(graph, reference_node, nn.Linear(input_size, output_size))

    # Add an adaptive pooling layer
    elif operation == 'pool':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for pool operation")

        graph, new_node = add_specific_node(graph, reference_node, nn.AdaptiveAvgPool1d(target_size))
        
        # Input and output shapes for pooling
        new_node_input_shape = ref_feature_shape
        # Create a new tuple with the last dimension changed
        if len(ref_feature_shape) == 1:
            new_node_output_shape = (target_size,)
        else:
            new_node_output_shape = tuple(list(ref_feature_shape[:-1]) + [target_size])
        
    # Add a repeat/broadcast layer
    elif operation == 'repeat':
        target_size = kwargs.get('target_size')
        if target_size is None:
            raise ValueError("target_size must be provided for repeat operation")
            
        # Get input size from reference node (last feature dimension)
        input_size = ref_feature_shape[-1]
        graph, new_node = add_specific_node(graph, reference_node, nn.CircularPad1d((0, target_size - input_size)))
        
        # Input and output shapes for repeat
        new_node_input_shape = ref_feature_shape
        # Create a new tuple with the last dimension changed
        if len(ref_feature_shape) == 1:
            new_node_output_shape = (target_size,)
        else:
            new_node_output_shape = tuple(list(ref_feature_shape[:-1]) + [target_size])

    # Add a flatten layer, that flattens every dimension except the batch dimension 
    elif operation == 'flatten':
        graph, new_node = add_specific_node(graph, reference_node, nn.Flatten(start_dim=1, end_dim=-1))
        
        # Calculate flattened feature size (product of all feature dimensions)
        flattened_size = 1
        for dim in ref_feature_shape:
            if isinstance(dim, int) and dim > 0:
                flattened_size *= dim
                
        # Input and output shapes for flatten
        new_node_input_shape = ref_feature_shape
        new_node_output_shape = (flattened_size,)

    # Add a skip connection
    elif operation == 'skip':
        first_node = kwargs.get('first_node')
        if first_node is None:
            raise ValueError("first_node must be provided for skip operation")
        second_node = reference_node
        graph, new_node = add_skip_connection(graph, second_node, first_node)
        
        # Get the feature shapes of both nodes
        first_features = get_feature_dims(first_node.meta['tensor_meta'].shape)
        second_features = get_feature_dims(second_node.meta['tensor_meta'].shape)
        
        # For a skip connection, we'll use the second node's features for input and output
        new_node_input_shape = second_features
        new_node_output_shape = second_features

    # Add branch node, that branches the input into two paths
    elif operation == 'branch':
        # Get the feature shape of the reference node, linear only for now
        input_size = ref_feature_shape[-1]
        
        # Create two new nodes with random output shapes
        branch1_out_size = random.randint(1, 1000)
        branch2_out_size = random.randint(1, 1000)
        
        # Create the branch modules
        branch1_module = nn.Linear(input_size, branch1_out_size)
        branch2_module = nn.Linear(input_size, branch2_out_size)
        
        # Use the utility function to add branch nodes
        graph, new_node, skip_output_shape = add_branch_nodes(graph, reference_node, branch1_module, branch2_module)
        
        # Get feature dimensions from skip connection output shape
        new_node_output_shape = get_feature_dims(skip_output_shape)
        
        # For branches, input shape is reference node features, output is from skip connection
        new_node_input_shape = ref_feature_shape

    graph.graph.lint()
    graph.recompile()
        
    # Fix the connections with clear input/output shape distinction
    adapt_connections(graph, new_node, 
                     parent_output_shape=input_shape,
                     new_node_input_features=new_node_input_shape,
                     new_node_output_features=new_node_output_shape,
                     child_input_shape=output_shape)

    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    example_input = torch.randn(next(iter(graph.graph.nodes)).meta['tensor_meta'].shape)  # SHAPE NOTE: Using full shape including batch dimension
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
    output_left_shape = list(input_node.meta['tensor_meta'].shape)  # SHAPE NOTE: Full shape with batch dimension
    input_right_shape = reference_node.meta['tensor_meta'].shape  # SHAPE NOTE: Full shape with batch dimension
    
    # Helper function to get feature dimensions (excluding batch dimension)
    def get_feature_dims(shape):
        if shape is None:
            return None
        # Skip the batch dimension (first dimension)
        if len(shape) > 1:
            return tuple(shape[1:])
        else:
            return tuple(shape)
    
    # Extract feature dimensions
    output_left_features = get_feature_dims(output_left_shape)
    
    # Remove the node from the graph
    reference_node.replace_all_uses_with(input_node)
    
    graph.graph.erase_node(reference_node)
    
    graph.delete_all_unused_submodules()

    # Adapt connections between input node and its new users with clear shape distinction
    graph = adapt_connections(graph, new_node=input_node, 
                             parent_output_shape=output_left_shape,
                             new_node_input_features=output_left_features,
                             new_node_output_features=output_left_features,
                             child_input_shape=input_right_shape)

    # Lint and recompile the graph
    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    example_input = torch.randn(next(iter(graph.graph.nodes)).meta['tensor_meta'].shape)  # SHAPE NOTE: Using full shape including batch dimension
    ShapeProp(graph).propagate(example_input)

    return graph, input_node

def adapt_connections(
    graph: torch.fx.GraphModule,
    new_node: torch.fx.Node,
    parent_output_shape: tuple,  # Full shape with batch dimension from parent node output
    new_node_input_features: tuple,  # Feature dimensions only (no batch) for new node input
    new_node_output_features: tuple,  # Feature dimensions only (no batch) for new node output
    child_input_shape: tuple  # Full shape with batch dimension required by child node
):
    """
    Adapts the connections to/from a node to ensure all connected nodes have compatible shapes.
    
    Args:
        graph: The FX graph
        new_node: The node whose connections need adaptation
        parent_output_shape: The shape output by the parent node (full shape with batch dimension)
        new_node_input_features: The input shape expected by new node (feature dimensions only, no batch)
        new_node_output_features: The output shape produced by new node (feature dimensions only, no batch)
        child_input_shape: The input shape expected by the child node (full shape with batch dimension)
    Returns:
        graph: The modified graph
    """
    
    # Helper function to extract feature dimensions (excluding batch dimension)
    def get_feature_dims(shape):
        if shape is None:
            return None
        # Skip the batch dimension (first dimension)
        if len(shape) > 1:
            return tuple(shape[1:])
        else:
            return tuple(shape)
    
    # Extract feature dimensions from parent and child shapes
    parent_features = get_feature_dims(parent_output_shape)
    child_features = get_feature_dims(child_input_shape)
    
    # Special handling for skip connections (torch.add operations)
    if new_node.target == torch.add:
        # Get shapes of both input nodes
        first_node = new_node.args[0]
        second_node = new_node.args[1]
        first_shape = first_node.meta['tensor_meta'].shape
        second_shape = second_node.meta['tensor_meta'].shape
        
        # Extract feature dimensions
        first_features = get_feature_dims(first_shape)
        second_features = get_feature_dims(second_shape)
        
        # Check if feature dimensions are compatible
        if first_features != second_features:
            # For skip connections, adapt output of first node to be compatible
            print(f"Skip connection shapes don't match: {first_features} vs {second_features}")
            
            graph, first_node = adapt_node_shape(graph, first_node, first_features, second_features)
            
            # Update the skip connection node's args
            new_node.args = (first_node, second_node)
    
    # For regular nodes, adapt parent-to-new-node connection
    else:
        # Always adapt all dimensions for full compatibility
        if parent_features != new_node_input_features:
            print(f"Parent output features {parent_features} don't match node input features {new_node_input_features}")
            graph, parent_node = adapt_node_shape(graph, new_node.args[0], parent_features, new_node_input_features)

    # Handle new-node-to-child connection
    if new_node_output_features != child_features:
        print(f"Node output features {new_node_output_features} don't match child input features {child_features}")
        graph, child_node = adapt_node_shape(graph, new_node, new_node_output_features, child_features)
    
    return graph
