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
            args=(second_node, first_node),
        )
    
    # Update connections
    second_node.replace_all_uses_with(new_node)
    new_node.args = (second_node, first_node)

    return graph, new_node

def adapt_tensor_size(graph, node, current_size, target_size, target_user=None):
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
        remainder = target_size % current_size
        
        if length_multiplier > 1:
            # First repeat the tensor as many times as possible
            graph, repeat_node = add_specific_node(
                graph, 
                node, 
                torch.repeat_interleave, 
                kwargs={"repeats": length_multiplier, "dim": 1},
                target_user=target_user  # Intermediate node
            )
            print(f"Added repeat node {repeat_node.name} after node {node.name}, repeats: {length_multiplier}")

            if remainder > 0:
                # Then use circular padding for the remainder
                graph, adapted_node = add_specific_node(
                    graph, 
                    repeat_node, 
                    nn.CircularPad1d((0, remainder)),
                    target_user=target_user
                )
                print(f"Added circular pad node {adapted_node.name} after repeat node {repeat_node.name}, remainder: {remainder}")
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
            print(f"Added circular pad node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")
    else:
        # Need to decrease size - use adaptive pooling
        graph, adapted_node = add_specific_node(
            graph, 
            node, 
            nn.AdaptiveAvgPool1d(target_size),
            target_user=target_user
        )
        print(f"Added adaptive avg pool node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")

    return graph, adapted_node

def adapt_node_shape(graph, node, current_size, target_size, target_user=None):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Convert current_size and target_size to tuples if they are not already
    current_size = tuple(current_size)
    target_size = tuple(target_size)
    
    if current_size == target_size:
        return graph, node
    
    if len(current_size) == 1:
        # For 1D tensors, directly adapt the size
        return adapt_tensor_size(graph, node, current_size[0], target_size[0], target_user)
    
    elif len(current_size) > 1:
        # calculate total size of target shape by multiplying all dimensions except the first
        target_total = math.prod(target_size)
        current_total = math.prod(current_size)

        # Add flatten node
        graph, flatten_node = add_specific_node(
            graph, 
            node, 
            nn.Flatten(start_dim=1, end_dim=-1),
            target_user=target_user  # Intermediate node
        )

        # Adapt the flattened tensor
        graph, adapted_node = adapt_tensor_size(
            graph, 
            flatten_node, 
            current_total, 
            target_total, 
            target_user=target_user  # Intermediate node
        )

        # Add unflatten node
        graph, unflatten_node = add_specific_node(
            graph, 
            adapted_node, 
            nn.Unflatten(dim=1, sizes=target_size),
            target_user=target_user  # Final node
        )

        return graph, unflatten_node

def add_branch_nodes(graph, reference_node, branch1_module, branch2_module):
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
    
    # Get the shape of the reference node from metadata
    reference_node_shape = reference_node.meta['tensor_meta'].shape
    
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
    placeholder_shape = next(iter(graph.graph.nodes)).meta['tensor_meta'].shape
    example_input = torch.randn(placeholder_shape)
    ShapeProp(graph).propagate(example_input)
    
    # Infer the shapes of the branch nodes from the metadata
    branch1_shape = tuple(branch1_node.meta['tensor_meta'].shape[1:])
    branch2_shape = tuple(branch2_node.meta['tensor_meta'].shape[1:])
    
    # Initialize variables to track the final nodes to use in skip connection
    final_branch1_node = branch1_node
    final_branch2_node = branch2_node
    
    # Adapt branch nodes if needed to ensure they have compatible shapes
    if branch1_shape != branch2_shape:    
        # Adapt first branch
        graph, final_branch1_node = adapt_node_shape(graph, branch1_node, branch1_shape, branch2_shape)

    # Run shape propagation to update metadata for the branch nodes
    placeholder_shape = next(iter(graph.graph.nodes)).meta['tensor_meta'].shape
    example_input = torch.randn(placeholder_shape)
    ShapeProp(graph).propagate(example_input)

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
