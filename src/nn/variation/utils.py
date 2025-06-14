import math
import logging

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from ..individual_graph_module import NeuralNetworkIndividualGraphModule


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
    module_names = {node.target for node in graph.graph.nodes if node.op == "call_module"}
    
    # Try base name first
    name = base_name
    counter = 1
    
    # Keep incrementing counter until we find a unique name
    while name in module_names:
        name = f"{base_name}_{counter}"
        counter += 1
    
    return name

def add_specific_node(graph, reference_node, module_or_function, kwargs=None, target_user=None):
    """
    Helper function to add a new node to the graph after the reference node.
    Supports both PyTorch modules and functions.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        module_or_function: The PyTorch module or function to add
        kwargs: Optional keyword arguments for the function call (only used for functions)
        target_user: Optional specific node that should use the new node. If None, all users will be updated.
    Returns:
        graph: The modified graph
        new_node: The newly added node
    """
    if kwargs is None:
        kwargs = {}
        
    # Check if we're adding a module or a function
    if isinstance(module_or_function, nn.Module):
        # Add a module
        name = get_unique_name(graph, module_or_function.__class__.__name__)
        graph.add_submodule(name, module_or_function)
        
        # Add node after reference_node
        with graph.graph.inserting_after(reference_node):
            new_node = graph.graph.call_module(
                module_name=name,
                args=(reference_node,),
                kwargs=kwargs,
            )
    else:
        # Add a function
        with graph.graph.inserting_after(reference_node):
            new_node = graph.graph.call_function(
                module_or_function,
                args=(reference_node,) if not isinstance(reference_node, tuple) else reference_node,
                kwargs=kwargs,
            )
    
    # Update connections based on target_user
    if target_user is not None:
        # Find the index of reference_node in target_user's args
        for i, arg in enumerate(target_user.args):
            if arg is reference_node:
                # Update just this specific connection
                target_user.args = tuple(new_node if x is reference_node else x for x in target_user.args)
                break
    else:
        # Update all users of the reference node to use the new node
        reference_node.replace_all_uses_with(new_node)
    
    # Set the input of the new node
    if isinstance(reference_node, tuple):
        new_node.args = reference_node
    else:
        new_node.args = (reference_node,)
    
    logging.debug(f"Added node {new_node.name} after node {reference_node.name}")
    return graph, new_node

def add_skip_connection(graph, second_node, first_node, torch_function=torch.add):
    """
    Adds a skip connection to the graph after the reference node.
    
    Args:
        graph: The FX graph
        second_node: The node after which the new node will be inserted
        first_node: The node to skip
    Returns:
        graph: The modified graph
        new_node: The newly added node
    """
    
    # Verify first_node comes before second_node in graph
    first_node_found = False
    for node in graph.graph.nodes:
        if node == first_node:
            first_node_found = True
        elif node == second_node:
            if not first_node_found:
                raise ValueError("Skip connection first_node must come before second_node in graph")
            break

    # Add skip connection node after reference_node
    with graph.graph.inserting_after(second_node):
        new_node = graph.graph.call_function(
            torch_function,
            args=(first_node, second_node),
        )
    
    # Update connections
    second_node.replace_all_uses_with(new_node)
    new_node.args = (first_node, second_node)

    return graph, new_node

def _adapt_tensor_size(graph, node, current_size: int, target_size: int, target_user=None):
    """
    Helper function to adapt a tensor's size using repeat_interleave, circular padding, or adaptive pooling.
    
    Args:
        graph: The FX graph
        node: The node to adapt
        current_size: Current size (integer)
        target_size: Target size (integer)
        target_user: Optional specific node that should use the new node
    Returns:
        graph: The modified graph
        adapted_node: The node after adaptation
    """
    if current_size < target_size:
        length_multiplier = target_size // current_size
        
        if length_multiplier > 1:
            remainder = target_size % current_size
        
            # First repeat the tensor as many times as possible
            graph, repeat_node = add_specific_node(
                graph, 
                node, 
                torch.repeat_interleave, 
                kwargs={"repeats": length_multiplier, "dim": 1},
                target_user=target_user  # Intermediate node
            )
            logging.debug(f"Added repeat node {repeat_node.name} after node {node.name}, repeats: {length_multiplier}")

            if remainder > 0:
                # Then use circular padding for the remainder
                graph, adapted_node = add_specific_node(
                    graph, 
                    repeat_node, 
                    nn.CircularPad1d((0, remainder)),
                    target_user=target_user
                )
                logging.debug(f"Added circular pad node {adapted_node.name} after repeat node {repeat_node.name}, remainder: {remainder}")
            else:
                adapted_node = repeat_node
        else:
            # If we only need to wrap once, just use circular padding
            graph, adapted_node = add_specific_node(
                graph, 
                node, 
                nn.CircularPad1d((0, target_size - current_size)),
                target_user=target_user
            )
            logging.debug(f"Added circular pad node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")
    else:
        # Need to decrease size - use adaptive pooling
        graph, adapted_node = add_specific_node(
            graph, 
            node, 
            nn.AdaptiveAvgPool1d(target_size),
            target_user=target_user
        )
        logging.debug(f"Added adaptive avg pool node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")

    return graph, adapted_node

class ReshapeModule(nn.Module):
    """A PyTorch module for reshaping tensors to a specific target size."""
    
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, x):
        return x.reshape(-1, *self.target_size)

def adapt_node_shape(graph, node, current_size, target_size, target_user=None, try_linear_adapter=True):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
        try_linear_adapter: If True, try using a single linear layer instead of flatten->adapt->unflatten pattern
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Convert current_size and target_size to tuples if they are not already
    current_size = tuple(current_size)
    target_size = tuple(target_size)
    
    if current_size == target_size:
        return graph, node
    
    current_dims = len(current_size)
    target_dims = len(target_size)
    
    # Handle 1D to 1D case directly
    if current_dims == 1 and target_dims == 1:
        if try_linear_adapter:
            return add_specific_node(
                graph,
                node,
                nn.Linear(current_size, target_size),
                target_user=target_user
            )
        return _adapt_tensor_size(graph, node, current_size[0], target_size[0], target_user=target_user)
    
    # Calculate total elements
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    
    # If total elements are the same, just reshape and return
    if current_total == target_total:
        return add_specific_node(
            graph,
            node,
            ReshapeModule(target_size),
            target_user=target_user
        )
    
    if try_linear_adapter and current_size[:-1] == target_size[:-1]:  # Only use linear adapter if all but last dims are the same
        return add_specific_node(
            graph,
            node,
            nn.Linear(current_size[-1], target_size[-1]),
            target_user=target_user
        )
    
    # Step 1: Flatten if starting from multi-dimensional (2+:1 or 2+:2+)
    if current_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Flatten(start_dim=1, end_dim=-1),
            target_user=target_user
        )
    
    # Step 2: Adapt tensor size (total elements differ, so this is always needed)
    graph, node = _adapt_tensor_size(
        graph, 
        node, 
        current_total, 
        target_total, 
        target_user=target_user
    )
    
    # Step 3: Unflatten if ending with multi-dimensional (1:2+ or 2+:2+)
    if target_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Unflatten(dim=1, unflattened_size=target_size),
            target_user=target_user
        )
    
    return graph, node

def add_branch_nodes(graph: NeuralNetworkIndividualGraphModule, reference_node, branch1_module, branch2_module):
    """
    Adds two branch nodes in parallel after the reference node and connects them with a skip connection.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the branch nodes will be inserted
        branch1_module: The module for the first branch
        branch2_module: The module for the second branch
    Returns:
        graph: The modified graph
        new_node: The combined node after the branches
    """
    # Generate unique names for the branch nodes
    branch1_name = get_unique_name(graph, "branch1")
    branch2_name = get_unique_name(graph, "branch2")
    
    graph.add_submodule(branch1_name, branch1_module)
    graph.add_submodule(branch2_name, branch2_module)

    # Create the first branch node without replacing connections
    with graph.graph.inserting_after(reference_node):
        branch1_node = graph.graph.call_module(
            module_name=branch1_name, 
            args=(reference_node,)
        )
    
    # Create the second branch node without replacing connections
    with graph.graph.inserting_after(branch1_node):
        branch2_node = graph.graph.call_module(
            module_name=branch2_name, 
            args=(reference_node,)
        )
    
    # Run shape propagation to update metadata for the new nodes
    ShapeProp(graph).propagate(graph.example_input)
    
    # Infer the shapes of the branch nodes from the metadata, pass through get_feature_dims to remove batch dimension
    branch1_shape = get_feature_dims(branch1_node.meta['tensor_meta'].shape)
    branch2_shape = get_feature_dims(branch2_node.meta['tensor_meta'].shape)
    
    # Initialize variables to track the final nodes to use in skip connection
    final_branch1_node = branch1_node
    final_branch2_node = branch2_node
    
    # Adapt branch nodes if needed to ensure they have compatible shapes
    if branch1_shape != branch2_shape:    
        # Adapt first branch
        graph, final_branch1_node = adapt_node_shape(graph, branch1_node, branch1_shape, branch2_shape)

    # Run shape propagation to update metadata for the branch nodes
    ShapeProp(graph).propagate(graph.example_input)

    # Get the inferred output of the skip connection from the shape propagation
    skip_connection_output_shape = final_branch2_node.meta['tensor_meta'].shape

    # Create a skip connection between the two branch nodes using the final adapted nodes
    with graph.graph.inserting_after(final_branch2_node):
        new_node = graph.graph.call_function(
            torch.add,
            args=(final_branch1_node, final_branch2_node),
        )
    
    # Update connections - replace all uses of reference_node with new_node
    reference_node.replace_all_uses_with(new_node)
    # Reset the args of the two branch nodes
    branch1_node.args = (reference_node,)
    branch2_node.args = (reference_node,)

    return graph, new_node, skip_connection_output_shape

def get_feature_dims(shape):
    """
    Helper function to get feature dimensions (excluding batch dimension) from a shape tuple.
    
    Args:
        shape: A shape tuple that includes batch dimension
    Returns:
        tuple: Feature dimensions only (excluding batch dimension)
    """
    if shape is None:
        return None
    # Skip the batch dimension (first dimension)
    if len(shape) > 1:
        return tuple(shape[1:])
    else:
        return tuple(shape)

def node_has_shape(node: torch.fx.Node):
    return "tensor_meta" in node.meta and hasattr(node.meta["tensor_meta"], "shape")