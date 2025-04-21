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

def add_specific_node(graph, reference_node, module):
    """
    Helper function to add a new node to the graph after the reference node.
    
    Args:
        graph: The FX graph
        reference_node: The node after which the new node will be inserted
        module: The PyTorch module to add
    Returns:
        graph: The modified graph
        new_node: The newly added node
    """
    name = get_unique_name(graph, module.__class__.__name__)
    graph.add_submodule(name, module)
    
    # Add node after reference_node
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(reference_node,),
            kwargs={},
        )
    
    # Update connections
    reference_node.replace_all_uses_with(new_node)
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

def adapt_node_shape(graph, node, current_size, target_size):
    """
    Adapts a node's output shape to match a target size using either adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output
        target_size: Desired size of the node's output
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    if current_size == target_size:
        return graph, node
        
    if current_size < target_size:
        # Need to increase size - use circular padding
        graph, adapted_node = add_specific_node(graph, node, nn.CircularPad1d((0, target_size - current_size)))
    else:
        # Need to decrease size - use adaptive pooling
        graph, adapted_node = add_specific_node(graph, node, nn.AdaptiveAvgPool1d(target_size))
        
    return graph, adapted_node

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
    example_input = torch.randn(reference_node_shape)
    ShapeProp(graph).propagate(example_input)
    
    # Infer the shapes of the branch nodes from the metadata
    branch1_shape = branch1_node.meta['tensor_meta'].shape
    branch2_shape = branch2_node.meta['tensor_meta'].shape
    
    # Initialize variables to track the final nodes to use in skip connection
    final_branch1_node = branch1_node
    final_branch2_node = branch2_node
    
    # Adapt branch nodes if needed to ensure they have compatible shapes
    if branch1_shape[-1] != branch2_shape[-1]:
        target_size = max(branch1_shape[-1], branch2_shape[-1])
        
        # Adapt first branch if needed
        if branch1_shape[-1] != target_size:
            graph, final_branch1_node = adapt_node_shape(graph, branch1_node, branch1_shape[-1], target_size)
        
        # Adapt second branch if needed
        if branch2_shape[-1] != target_size:
            graph, final_branch2_node = adapt_node_shape(graph, branch2_node, branch2_shape[-1], target_size)

    # Run shape propagation to update metadata for the branch nodes
    example_input = torch.randn(reference_node_shape)
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
