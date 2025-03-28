import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from itertools import zip_longest


def get_graph(model, input_shape=None):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
        input_shape: Optional tuple specifying input tensor shape (batch, seq_len, dim)
                    If provided, shape propagation will be performed
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

def shape_prop(graph, input_shape):
    """
    Propagate shapes through the graph to get tensor metadata for each node
    
    Args:
        graph: The FX graph
        input_shape: The input tensor shape (batch, seq_len, dim)
    """
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

def adapt_node_to_input(node, input_shape, mode='linear'):
    """
    Adapts a node's input dimensions to match the input shape
    
    Args:
        node: The node to adapt
        input_shape: The shape that should feed into this node
        mode: The adaptation method ('linear', 'broadcast', 'squeeze', 'unsqueeze')
    Returns:
        node: The adapted node
    """
    if mode == 'linear' and isinstance(node, nn.Linear):
        if node.in_features != input_shape[-1]:
            # Create new linear layer with matching input dimension
            new_node = nn.Linear(input_shape[-1], node.out_features)
            # Initialize weights using the original weights where possible
            with torch.no_grad():
                min_in = min(node.in_features, input_shape[-1])
                new_node.weight.data[:, :min_in] = node.weight.data[:, :min_in]
                if node.bias is not None:
                    new_node.bias.data[:] = node.bias.data[:]
            return new_node
    return node

def add_node(graph, reference_node, operation, operator_params: torch.Tensor = None, name=None):
    """
    Adds a new node to the graph and adapts it to fit the input shape.
    Does NOT modify existing nodes.
    
    Args:
        graph: The FX graph
        reference_node: The target node to insert after
        operation: Either a PyTorch module or a string representing an operator
        operator_params: The accompanying tensor/value for the operation (required for operators)
        name: Optional name for the module (default: auto-generated)
    Returns:
        graph: The modified graph
        name: The name of the added module/operator
    """
    if 'tensor_meta' not in reference_node.meta:
        raise ValueError("Input node missing shape metadata. Run shape_prop() first.")
    input_shape = reference_node.meta['tensor_meta'].shape
    
    if isinstance(operation, nn.Module):
        # Adapt the new module to match input shape
        operation = adapt_node_to_input(operation, input_shape)
        
        # Add to graph
        graph.add_submodule(name, operation)
        with graph.graph.inserting_after(reference_node):
            new_node = graph.graph.call_module(
                module_name=name,
                args=(reference_node,),
                kwargs={},
            )
    else:
        # Handle operators
        if operator_params is None:
            raise ValueError("operator_params is required for operators")
            
        # Adapt operator params to match input shape
        if operation in ['add', 'mul']:
            if len(operator_params.shape) < len(input_shape):
                # Add necessary dimensions to match input shape
                for _ in range(len(input_shape) - len(operator_params.shape)):
                    operator_params = operator_params.unsqueeze(0)
        elif operation == 'matmul':
            if operator_params.shape[0] != input_shape[-1]:
                raise ValueError(f"Matmul input dimension mismatch: {operator_params.shape[0]} vs {input_shape[-1]}")
        
        with graph.graph.inserting_after(reference_node):
            if operation == 'add':
                new_node = graph.graph.call_function(torch.add, args=(reference_node, operator_params))
            elif operation == 'mul':
                new_node = graph.graph.call_function(torch.mul, args=(reference_node, operator_params))
            elif operation == 'matmul':
                new_node = graph.graph.call_function(torch.matmul, args=(reference_node, operator_params))
            else:
                raise ValueError(f"Unsupported operation: {operation}")
    
    # Update the graph connections
    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)
    
    graph.graph.lint()
    graph.recompile()
    
    return graph, name

def adapt_connections(graph, node, target_shape):
    """
    Adapts the connections to/from a node to match a target shape.
    This modifies existing nodes in the graph to maintain shape compatibility.
    
    Args:
        graph: The FX graph
        node: The node whose connections need adaptation
        target_shape: The desired shape for the node's output
    Returns:
        graph: The modified graph
    """
    # Get all users of this node
    for user in node.users:
        if user.op == 'call_module':
            module = getattr(graph, user.target)
            if isinstance(module, nn.Linear):
                if module.in_features != target_shape[-1]:
                    # Create new linear layer with matching input dimension
                    new_module = nn.Linear(target_shape[-1], module.out_features)
                    # Initialize weights
                    with torch.no_grad():
                        min_in = min(module.in_features, target_shape[-1])
                        new_module.weight.data[:, :min_in] = module.weight.data[:, :min_in]
                        if module.bias is not None:
                            new_module.bias.data[:] = module.bias.data[:]
                    setattr(graph, user.target, new_module)
        
        elif user.op == 'call_function':
            if user.target in [torch.add, torch.mul]:
                # For element-wise operations, we need to ensure broadcasting works
                other_arg = user.args[1] if user.args[0] is node else user.args[0]
                if isinstance(other_arg, torch.Tensor):
                    # Reshape tensor to match target shape for broadcasting
                    reshaped_tensor = other_arg.view(*target_shape)
                    user.args = tuple(reshaped_tensor if arg is other_arg else arg for arg in user.args)
            
            elif user.target == torch.matmul:
                # For matmul, we need to ensure matrix dimensions match
                other_arg = user.args[1] if user.args[0] is node else user.args[0]
                if isinstance(other_arg, torch.Tensor):
                    if user.args[0] is node:  # node is first arg
                        if other_arg.shape[0] != target_shape[-1]:
                            raise ValueError(f"Cannot adapt matmul dimensions: {target_shape[-1]} vs {other_arg.shape[0]}")
                    else:  # node is second arg
                        if other_arg.shape[-1] != target_shape[-2]:
                            raise ValueError(f"Cannot adapt matmul dimensions: {other_arg.shape[-1]} vs {target_shape[-2]}")
    
    graph.graph.lint()
    graph.recompile()
    return graph

def remove_node(graph, reference_node):
    """
    Removes a node from the graph and cleans up its module if it was dynamically added.
    
    Args:
        graph: The FX graph
        reference_node: The node to remove
    Returns:
        graph: The modified graph
    """
    input_node = reference_node.args[0]
    
    # Remove the node
    reference_node.replace_all_uses_with(input_node)
    graph.graph.erase_node(reference_node)
    
    if reference_node.op == "call_module":
        submodule_name = reference_node.target
        if hasattr(graph, submodule_name):
            delattr(graph, submodule_name)

    graph.graph.lint()
    graph.recompile()
    
    return graph

def are_shapes_broadcastable(shape1, shape2):
    """Helper function to check if two shapes are broadcastable."""
    # Reverse shapes for easier comparison
    shape1 = list(reversed(shape1))
    shape2 = list(reversed(shape2))
    
    for s1, s2 in zip_longest(shape1, shape2, fillvalue=1):
        if s1 != 1 and s2 != 1 and s1 != s2:
            return False
    return True


