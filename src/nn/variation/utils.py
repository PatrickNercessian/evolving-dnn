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
    
def adapt_node_shape(graph, node, current_size, target_size, target_user=None, adapt_type='gcf'):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
        adapt_type: Type of adaptation to use. Can be 'regular', 'linear', or 'gcf'
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Convert current_size and target_size to tuples if they are not already
    current_size = tuple(current_size)
    target_size = tuple(target_size)
    
    # If current_size = target_size, return the node
    if current_size == target_size:
        return graph, node
    
    # Get total elements of current_size and target_size
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    
    # If current_total = target_total, return a reshape node
    if current_total == target_total:
        return add_specific_node(
            graph,
            node,
            ReshapeModule(target_size),
            target_user=target_user
        )
    
    if adapt_type == 'gcf':
        return gcf_adapt_node_shape(graph, node, current_size, target_size, target_user)
    else:
        return adapt_node_shape_basic(graph, node, current_size, target_size, target_user, adapt_type)
    
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

def node_has_float_dtype(node: torch.fx.Node):
    """
    Check if a node has a float tensor dtype.
    
    Args:
        node: The FX node to check
    Returns:
        bool: True if the node has a float dtype, False otherwise
    """
    if not hasattr(node, 'meta') or 'tensor_meta' not in node.meta:
        return False
    
    tensor_dtype = node.meta['tensor_meta'].dtype
    # Check if dtype is one of the PyTorch float types
    float_dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16, 
                   torch.float8_e5m2, torch.float8_e4m3fn, torch.float, torch.double, torch.half, torch.bfloat16]
    return tensor_dtype in float_dtypes

def print_graph_debug_info(graph):
    """
    Prints every node in the graph, its name, op, shape (if available), args, and users for debugging.
    Args:
        graph: The FX GraphModule or Graph
    """
    logging.debug("\n==== GRAPH DEBUG INFO ====")
    for node in graph.graph.nodes if hasattr(graph, 'graph') else graph.nodes:
        name = getattr(node, 'name', str(node))
        op = getattr(node, 'op', 'unknown')
        # Shape info
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
            shape = node.meta['tensor_meta'].shape
        else:
            shape = 'N/A'
        # Args info
        args = []
        for arg in node.args:
            if hasattr(arg, 'name'):
                args.append(arg.name)
            elif isinstance(arg, (tuple, list)):
                args.append(str([a.name if hasattr(a, 'name') else repr(a) for a in arg]))
            else:
                args.append(repr(arg))
        # Users info
        users = [u.name for u in getattr(node, 'users', [])]
        logging.debug(f"Node: {name} | op: {op} | shape: {shape} | Args: {args} | Users: {users}")
    logging.debug("==== END GRAPH DEBUG INFO ====")
