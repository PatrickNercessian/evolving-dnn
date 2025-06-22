import math
import logging

import torch
import torch.nn as nn
import torch.fx

from .utils import add_specific_node, get_unique_name, get_feature_dims
from ..individual_graph_module import NeuralNetworkIndividualGraphModule
from torch.fx.passes.shape_prop import ShapeProp

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

def _unflatten_linear_flatten(graph, node, adapt_shape_values: tuple[int, int, int], target_user=None):
    """
    Helper function to perform reshape-linear-flatten sequence.
    
    Args:
        graph: The FX graph
        node: The input node
        adapt_shape_values: Tuple of dimensions for reshaping
        target_user: Optional specific node that should use the output
    Returns:
        graph: The modified graph
        output_node: The node after reshape-linear-flatten
    """
    # Reshape
    graph, node = add_specific_node(
        graph,
        node,
        nn.Unflatten(dim=1, unflattened_size=(adapt_shape_values[0], adapt_shape_values[1])),
        target_user=target_user
    )

    # Add linear layer
    graph, node = add_specific_node(
        graph,
        node,
        nn.Linear(adapt_shape_values[1], adapt_shape_values[2], bias=False),
        target_user=target_user
    )

    # Flatten features
    graph, node = add_specific_node(
        graph,
        node,
        nn.Flatten(start_dim=1, end_dim=-1),
        target_user=target_user
    )
    
    return graph, node

def adapt_node_shape_basic(graph, node, current_size, target_size, target_user=None, adapt_type='regular'):
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
    current_dims = len(current_size)
    target_dims = len(target_size)
    
    # Handle 1D to 1D case directly
    if current_dims == 1 and target_dims == 1:
        if adapt_type == 'linear':
            return add_specific_node(
                graph,
                node,
                nn.Linear(current_size[0], target_size[0]),
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
    
    if adapt_type == 'linear' and current_size[:-1] == target_size[:-1]:  # Only use linear adapter if all but last dims are the same
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
    if adapt_type == 'linear':
        # If linear adapter is preferred, use linear layer
        graph, node = add_specific_node(
            graph,
            node,
            nn.Linear(current_total, target_total),
            target_user=target_user
        )
    else:
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

def gcf_adapt_node_shape(graph, node, current_size, target_size, target_user=None):
    """
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
    # Get total elements of current_size and target_size
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    logging.debug(f"Shape adaptation: {current_size} -> {target_size} (total elements: {current_total} -> {target_total})")
    
    # Determine if we're upsampling or downsampling
    is_upsampling = target_total > current_total
    logging.debug(f"Operation type: {'upsampling' if is_upsampling else 'downsampling'}")
    
    if is_upsampling:
        # For upsampling, work with target/current ratio
        feature_ratio = target_total / current_total
        r1 = math.floor(feature_ratio)
        r2 = math.ceil(feature_ratio)
        if r1 != r2:
            length_scale = int((target_total - (r2*current_total)) / (r1 - r2))
        else:
            length_scale = current_total
        r1_slice_length = length_scale
        r1_shape_values = (length_scale, 1, r1)
        r2_slice_length = current_total - r1_slice_length
        r2_shape_values = (r2_slice_length, 1, r2)
        logging.debug(f"Upsampling ratios: r1={r1}, r2={r2}, scale={length_scale}")
    else:
        # For downsampling, work with current/target ratio
        feature_ratio = current_total / target_total
        r1 = math.floor(feature_ratio)
        r2 = math.ceil(feature_ratio)
        # Only process if r1 != r2
        if r1 != r2:
            length_scale = int((current_total - (r2*target_total)) / (r1 - r2))
        else:
            length_scale = current_total
        r1_slice_length = r1 * length_scale
        r1_shape_values = (length_scale, r1, 1)
        r2_slice_length = current_total - r1_slice_length
        r2_shape_values = (r2_slice_length//r2, r2, 1)
        logging.debug(f"Downsampling ratios: r1={r1}, r2={r2}, scale={length_scale}")

    # If dims>1, flatten the node
    if len(current_size) > 1:
        logging.debug(f"Flattening node {node.name} (dims={len(current_size)})")
        graph, node = add_specific_node(
            graph,
            node,
            nn.Flatten(start_dim=1, end_dim=-1),
            target_user=target_user
        )
       
    if r1 != r2:
        # Create a slicing operation to get the first part
        logging.debug(f"Slicing first part: dim=1, start=0, length={r1_slice_length}")
        graph, r1_node = add_specific_node(
            graph,
            node,
            torch.narrow,
            kwargs={"dim": 1, "start": 0, "length": r1_slice_length},
            target_user=target_user
        )
        logging.debug(f"Added narrow node for r1 slice (length={r1_slice_length})")
    else:
        r1_node = node

    # Only apply reshape-linear-flatten if r1 > 1
    if r1 > 1:
        graph, r1_node = _unflatten_linear_flatten(graph, r1_node, r1_shape_values, target_user)
        logging.debug(f"Added reshape-linear-flatten sequence for r1 (shape={r1_shape_values})")
    
    # Chunking needed if r1 != r2
    if r1 != r2:
        graph, concat_node = add_specific_node(
            graph,
            r1_node,
            torch.cat,
            target_user=target_user
        )

        graph, r2_node = add_specific_node(
            graph,
            node,
            torch.narrow,
            kwargs={"dim": 1, "start": r1_slice_length, "length": r2_slice_length},
            target_user=concat_node
        )
        logging.debug(f"Added narrow node for r2 slice (length={r2_slice_length})")
        
        graph, r2_node = _unflatten_linear_flatten(graph, r2_node, r2_shape_values, concat_node)
        logging.debug(f"Added reshape-linear-flatten sequence for r2 (shape={r2_shape_values})")
        
        concat_node.args = ((r1_node, r2_node),-1)
        logging.debug("Added concatenation node")
    else:
        concat_node = r1_node
    
    # If target dimensions are greater than 1, unflatten to the target shape
    if len(target_size) > 1:
        graph, concat_node = add_specific_node(
            graph,
            concat_node,
            nn.Unflatten(dim=1, unflattened_size=target_size),
            target_user=target_user
        )
        logging.debug(f"Added unflatten node to target shape {target_size}")
    
    return graph, concat_node

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
    
# This has to be here to avoid circular logic
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