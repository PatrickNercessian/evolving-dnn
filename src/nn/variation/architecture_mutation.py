import random
import math

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from src.nn.individual_graph_module import NeuralNetworkIndividualGraphModule
from src.nn.variation.utils import (
    node_has_shape, add_specific_node, add_skip_connection,
    adapt_node_shape, add_branch_nodes, get_feature_dims
)


def mutation_add_linear(individual):
    print("Starting mutation_add_linear")
    # Find a random node in the graph to add a linear layer after
    eligible_nodes = _get_eligible_nodes(individual)
    if not eligible_nodes:
        print("No eligible nodes for adding linear layer")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding linear layer after {reference_node.name}")
    
    # Get the output shape of the reference node if available
    if hasattr(reference_node, 'meta') and 'tensor_meta' in reference_node.meta:
        input_shape = reference_node.meta['tensor_meta'].shape
        # Use the feature dimension (last dimension) as input size
        input_size = input_shape[-1]
        # Choose a reasonable output size similar to input size
        output_size = random.randint(max(1, input_size // 2), input_size * 2)
        print(f"Using input_size={input_size}, output_size={output_size} from reference node shape")
    else:
        # Fallback to a safe small size if shape information is not available
        print("Shape information not available, using default sizes")
        input_size = 8
        output_size = 8
    
    # Add a linear layer
    individual.graph_module = _add_node(individual.graph_module, reference_node, 'linear', 
                                        input_size=input_size, output_size=output_size)
    # Adjust training config (you could add learning rate mutation here)
    if random.random() < 0.3:
        individual.train_config.learning_rate *= random.uniform(0.5, 1.5)
        individual.train_config.learning_rate = max(0.0001, min(0.1, individual.train_config.learning_rate))
    
    print("Completed mutation_add_linear")
    return individual


def mutation_add_relu(individual):
    print("Starting mutation_add_relu")
    # Find a random node in the graph to add a ReLU layer after
    eligible_nodes = _get_eligible_nodes(individual)
    if not eligible_nodes:
        print("No eligible nodes for adding ReLU")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding ReLU after {reference_node.name}")
    
    # Add a ReLU layer
    individual.graph_module = _add_node(individual.graph_module, reference_node, 'relu')
    print("Completed mutation_add_relu")
    return individual


def mutation_add_skip_connection(individual):
    print("Starting mutation_add_skip_connection")
    # Find two random nodes in the graph to connect
    eligible_nodes = _get_eligible_nodes(individual)
    
    if len(eligible_nodes) < 2:
        print("Not enough eligible nodes for skip connection")
        return individual
    
    # Pick two different nodes, ensuring first_node comes before second_node
    first_node = random.choice(eligible_nodes)
    later_nodes = [n for n in eligible_nodes if n != first_node and 
                    list(individual.graph_module.graph.nodes).index(n) > 
                    list(individual.graph_module.graph.nodes).index(first_node)]
    
    if not later_nodes:
        print("No eligible later nodes for skip connection")
        return individual
        
    second_node = random.choice(later_nodes)
    print(f"Adding skip connection from {first_node.name} to {second_node.name}")
    
    # Add skip connection
    individual.graph_module = _add_node(individual.graph_module, second_node, 'skip', first_node=first_node)
    print("Completed mutation_add_skip_connection")
    
    return individual


def mutation_add_branch(individual):
    print("Starting mutation_add_branch")
    # Find a random node in the graph to add branches after
    eligible_nodes = _get_eligible_nodes(individual)
    
    if not eligible_nodes:
        print("No eligible nodes for adding branches")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding branch after {reference_node.name}")
    
    # Get the shape information from the reference node
    if hasattr(reference_node, 'meta') and 'tensor_meta' in reference_node.meta:
        input_shape = reference_node.meta['tensor_meta'].shape
        input_size = input_shape[-1]
        # Choose reasonable output sizes for branches
        branch1_out_size = max(1, input_size // 2)
        branch2_out_size = max(1, input_size // 2)
    else:
        print("Shape information not available, using default sizes")
        input_size = 8
        branch1_out_size = 4
        branch2_out_size = 4
        
    # Branch out sizes are random ints
    branch1_out_size = random.randint(1, 500)
    branch2_out_size = random.randint(1, 500)
    
    # Add branch nodes
    individual.graph_module = _add_node(individual.graph_module, reference_node, 'branch',
                                        branch1_out_size=branch1_out_size,
                                        branch2_out_size=branch2_out_size)
    print("Completed mutation_add_branch")
    
    return individual


def mutation_remove_node(individual):
    print("Starting mutation_remove_node")
    # Find eligible nodes to remove (not input, output, or critical nodes)
    nodes = list(individual.graph_module.graph.nodes)
    possible_nodes = _get_eligible_nodes(individual, nodes)
    eligible_nodes = []

    for node in possible_nodes:
        # Skip placeholder (input) and output nodes
        if node.op in ['placeholder', 'output']:
            continue
        
        # Skip nodes that are skip connections or branch nodes (as per remove_node restrictions)
        if hasattr(node, 'target') and node.target in (torch.add, torch.cat, torch.mul):
            continue
            
        # Skip nodes that are the first node in a branch (have multiple users)
        if len(node.users) > 1:
            continue
            
        # Skip if this is the only non-input/output node (would break the graph)
        non_io_nodes = [n for n in nodes if n.op not in ['placeholder', 'output']]
        if len(non_io_nodes) <= 1:
            continue
            
        eligible_nodes.append(node)
    
    if not eligible_nodes:
        print("No eligible nodes for removal")
        return individual
    
    # Select a random node to remove
    node_to_remove = random.choice(eligible_nodes)
    print(f"Removing node: {node_to_remove.name}")
    
    # Remove the node
    individual.graph_module, remaining_node = _remove_node(individual.graph_module, node_to_remove)
    print(f"Successfully removed node {node_to_remove.name}, remaining node: {remaining_node.name}")
    print("Completed mutation_remove_node")
    return individual

def _get_eligible_nodes(individual, nodes=None):
    if nodes is None:
        nodes = list(individual.graph_module.graph.nodes)
    eligible_nodes = []
    for node in nodes:
        if node.op in ['placeholder', 'output'] or not node_has_shape(node):
            continue
        
        eligible_nodes.append(node)
    return eligible_nodes

def _add_node(graph: NeuralNetworkIndividualGraphModule, reference_node: torch.fx.Node, operation: str, **kwargs):
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
  
    # Get feature dimensions from reference node (excluding batch)
    ref_feature_shape = get_feature_dims(reference_node.meta['tensor_meta'].shape)
    
    # Add a linear layer to the graph
    if operation == 'linear':
        # Get input_size and output_size from kwargs if provided, otherwise use random values
        input_size = kwargs.get('input_size')
        output_size = kwargs.get('output_size')
        
        # If not provided in kwargs, try to use the ref_feature_shape or default to random
        if input_size is None:
            raise ValueError("input_size must be provided for linear layer")
                
        if output_size is None:
            raise ValueError("output_size must be provided for linear layer")
        
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
        flattened_size = math.prod(ref_feature_shape)
                
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
        # Get the feature shape of the reference node
        input_size = ref_feature_shape[-1]
        
        # Get branch output sizes from kwargs if provided
        branch1_out_size = kwargs.get('branch1_out_size')
        branch2_out_size = kwargs.get('branch2_out_size')
        
        if branch1_out_size is None:
            raise ValueError("branch1_out_size must be provided for branch operation")

        if branch2_out_size is None:
            raise ValueError("branch2_out_size must be provided for branch operation")
        
        print(f"Branch modules: input_size={input_size}, branch1_out={branch1_out_size}, branch2_out={branch2_out_size}")
        
        # Create the branch modules
        branch1_module = nn.Linear(input_size, branch1_out_size)
        branch2_module = nn.Linear(input_size, branch2_out_size)
        
        # Use the utility function to add branch nodes
        graph, new_node, skip_output_shape = add_branch_nodes(graph, reference_node, branch1_module, branch2_module)
        
        # Get feature dimensions from skip connection output shape
        new_node_output_shape = get_feature_dims(skip_output_shape)
        
        # For branches, input shape is reference node features, output is from skip connection
        new_node_input_shape = ref_feature_shape

    # Add a dropout layer
    elif operation == 'dropout':
        # Get dropout probability from kwargs if provided, otherwise use default
        prob = kwargs.get('prob')
        if prob is None:
            raise ValueError("prob must be provided for dropout operation")
        print(f"Adding dropout layer with probability {prob}")
        
        graph, new_node = add_specific_node(graph, reference_node, nn.Dropout(p=prob))

        new_node_input_shape = ref_feature_shape
        new_node_output_shape = ref_feature_shape

    # Add an activation function
    elif operation == 'relu':
        graph, new_node = add_specific_node(graph, reference_node, nn.ReLU())

        new_node_input_shape = ref_feature_shape
        new_node_output_shape = ref_feature_shape


    graph.graph.lint()
    graph.recompile()
        
    # Fix the connections with clear input/output shape distinction
    _adapt_connections(graph, new_node, 
                     parent_output_shape=reference_node.meta['tensor_meta'].shape,  # Use reference node's output shape as parent output
                     new_node_input_features=new_node_input_shape,
                     new_node_output_features=new_node_output_shape,
                     child_input_shape=reference_node.meta['tensor_meta'].shape)

    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    ShapeProp(graph).propagate(graph.example_input)

    return graph

def _remove_node(graph: NeuralNetworkIndividualGraphModule, reference_node: torch.fx.Node):
    """
    Removes a node from the graph, can't be a skip connection or branch node
    
    Args:
        graph: The FX graph
        reference_node: The node to remove
    Returns:
        graph: The modified graph
    """
    # Check if reference node is a skip connection or branch node
    if reference_node.target in (torch.add, torch.cat, torch.mul):
        raise ValueError("Reference node is a skip connection or branch node, can't be removed")
    if len(reference_node.args[0].users) > 1:
        raise ValueError("Reference node is first node in branch, can't be removed")
    # TODO: Implement different removals for skip connections and branch nodes


    input_node = reference_node.args[0]
    
    # Get shapes before removing node
    output_left_shape = input_node.meta['tensor_meta'].shape  # SHAPE NOTE: Full shape with batch dimension
    input_right_shape = reference_node.meta['tensor_meta'].shape  # SHAPE NOTE: Full shape with batch dimension
    
    # Extract feature dimensions
    output_left_features = get_feature_dims(output_left_shape)
    
    # Remove the node from the graph
    reference_node.replace_all_uses_with(input_node)
    
    graph.graph.erase_node(reference_node)
    
    graph.delete_all_unused_submodules()

    # Adapt connections between input node and its new users with clear shape distinction
    graph = _adapt_connections(graph, new_node=input_node, 
                             parent_output_shape=output_left_shape,
                             new_node_input_features=output_left_features,
                             new_node_output_features=output_left_features,
                             child_input_shape=input_right_shape)

    # Lint and recompile the graph
    graph.graph.lint()
    graph.recompile()

    # Run shape propagation again to update all shape metadata
    ShapeProp(graph).propagate(graph.example_input)

    return graph, input_node

def _adapt_connections(
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
    
    # Extract feature dimensions from parent and child shapes
    parent_features = get_feature_dims(parent_output_shape)
    child_features = get_feature_dims(child_input_shape)
    
    # Special handling for skip connections (torch.add operations)
    # TODO: Handle any kind of skip connection (e.g. torch.cat, torch.mul, etc.)
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
            # For skip connections, adapt output of first node to be compatible with second node
            print(f"Skip connection shapes don't match: {first_features} vs {second_features}")
            
            # Use target_user=new_node to only update the skip connection's use of first_node
            graph, adapted_first_node = adapt_node_shape(graph, first_node, first_features, second_features, target_user=new_node)
            
            # Update the skip connection node's args
            new_node.args = (second_node, adapted_first_node)
    
    # For regular nodes, adapt parent-to-new-node connection
    else:
        # Always adapt all dimensions for full compatibility
        if parent_features != new_node_input_features:
            print(f"Parent output features {parent_features} don't match node input features {new_node_input_features}")
            # Parent output features (128, 239) don't match node input features (2, 128, 239)
            graph, parent_node = adapt_node_shape(graph, new_node.args[0], parent_features, new_node_input_features)

    # Handle new-node-to-child connection
    if new_node_output_features != child_features:
        print(f"Node output features {new_node_output_features} don't match child input features {child_features}")
        graph, child_node = adapt_node_shape(graph, new_node, new_node_output_features, child_features)
    
    return graph
