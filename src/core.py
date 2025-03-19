import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from itertools import zip_longest


def get_graph(model):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace
    """
        
    # Symbolically trace the model to get computation graph
    graph = torch.fx.symbolic_trace(model)
            
    # TODO: add shape propagation here instead of in shape_prop

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


def add_node(graph, node, operation, operator_params: torch.Tensor = None, name=None):
    """
    Adds a new node to the graph and handles both module registration and operators
    
    Args:
        graph: The FX graph
        node: The target node to insert after
        operation: Either a PyTorch module or a string representing an operator (e.g. 'add', 'mul')
        operator_params: The accompanying tensor/value for the operation (required for operators)
        name: Optional name for the module (default: auto-generated)
    Returns:
        graph: The modified graph
        name: The name of the added module/operator
    """
    # Get the shape of the input node from its metadata
    if 'tensor_meta' not in node.meta:
        raise ValueError("Input node missing shape metadata. Run shape_prop() first.")
    input_shape = node.meta['tensor_meta']['shape']
    
    if isinstance(operation, nn.Module):
        # Validate module shape compatibility
        if isinstance(operation, nn.Linear):
            if operation.in_features != input_shape[-1]:
                raise ValueError(f"Linear layer input dimension ({operation.in_features}) "
                              f"doesn't match input shape ({input_shape[-1]})")
        
        # Handle PyTorch modules
        graph.add_submodule(name, operation)
        with graph.graph.inserting_after(node):
            new_node = graph.graph.call_module(
                module_name=name,
                args=(node,),
                kwargs={},
            )
    else:
        # Handle operators (like add, mul, etc)
        if operator_params is None:
            raise ValueError("operator_params is required for matrix operations")
            
        # Validate operator shape compatibility
        op_shape = operator_params.shape
        if operation in ['add', 'mul']:
            # Check broadcasting compatibility
            if not are_shapes_broadcastable(input_shape, op_shape):
                raise ValueError(f"Shapes {input_shape} and {op_shape} are not broadcastable "
                              f"for {operation} operation")
        elif operation == 'matmul':
            # Check matrix multiplication compatibility
            if len(input_shape) < 2 or len(op_shape) < 2:
                raise ValueError("Inputs must have at least 2 dimensions for matmul")
            if input_shape[-1] != op_shape[-2]:
                raise ValueError(f"Incompatible dimensions for matmul: {input_shape} and {op_shape}")
            
        with graph.graph.inserting_after(node):
            if operation == 'add':
                new_node = graph.graph.call_function(torch.add, args=(node, operator_params))
            elif operation == 'mul':
                new_node = graph.graph.call_function(torch.mul, args=(node, operator_params))
            elif operation == 'matmul':
                new_node = graph.graph.call_function(torch.matmul, args=(node, operator_params))
            else:
                raise ValueError(f"Unsupported operation: {operation}")
    
    # Update the graph connections
    node.replace_all_uses_with(new_node)
    
    # Fix any self-references in the new node's args
    if isinstance(operation, nn.Module):
        new_node.args = (node,) # Assume that the module is a linear layer
    else:
        # For operators, replace any references to new_node with node in the args
        new_args = tuple(node if arg is new_node else arg for arg in new_node.args)
        new_node.args = new_args

    graph.graph.lint()
    graph.recompile()
        
    return graph, name

def remove_node(graph, node):
    """
    Removes a node from the graph and cleans up its module if it was dynamically added.
    Validates that the input node can be safely connected to the following nodes.
    
    Args:
        graph: The FX graph
        node: The node to remove
    Returns:
        graph: The modified graph
    Raises:
        ValueError: If shape validation fails or metadata is missing
    """
    # Get the input node that feeds into this node
    input_node = node.args[0]
    
    # Check that both nodes have shape metadata
    if 'tensor_meta' not in input_node.meta:
        raise ValueError("Input node missing shape metadata. Run shape_prop() first.")
    if 'tensor_meta' not in node.meta:
        raise ValueError("Node to remove missing shape metadata. Run shape_prop() first.")
        
    input_shape = input_node.meta['tensor_meta']['shape']
    
    # Check compatibility with all nodes that use the node being removed
    for user in node.users:
        if user.op == 'call_module':
            # Check module compatibility
            if hasattr(getattr(graph, user.target), 'in_features'):
                if input_shape[-1] != getattr(graph, user.target).in_features:
                    raise ValueError(f"Cannot remove node: Input shape {input_shape} is incompatible "
                                  f"with following layer's input dimension {getattr(graph, user.target).in_features}")
        
        elif user.op == 'call_function':
            # Check operator compatibility
            if user.target in [torch.add, torch.mul]:
                # For add/mul, check broadcasting compatibility with the other operand
                other_arg = user.args[1] if user.args[0] is node else user.args[0]
                if isinstance(other_arg, torch.fx.Node):
                    other_shape = other_arg.meta['tensor_meta']['shape']
                else:  # It's a tensor parameter
                    other_shape = other_arg.shape
                    
                if not are_shapes_broadcastable(input_shape, other_shape):
                    raise ValueError(f"Cannot remove node: Input shape {input_shape} is not broadcastable "
                                  f"with following operation's shape {other_shape}")
                    
            elif user.target == torch.matmul:
                # For matmul, check matrix multiplication compatibility
                other_arg = user.args[1] if user.args[0] is node else user.args[0]
                if isinstance(other_arg, torch.fx.Node):
                    other_shape = other_arg.meta['tensor_meta']['shape']
                else:  # It's a tensor parameter
                    other_shape = other_arg.shape
                    
                if len(input_shape) < 2 or len(other_shape) < 2:
                    raise ValueError("Cannot remove node: Matmul requires at least 2 dimensions")
                if user.args[0] is node:  # node is first arg
                    if input_shape[-1] != other_shape[-2]:
                        raise ValueError(f"Cannot remove node: Input shape {input_shape} is incompatible "
                                      f"with following matmul operation's shape {other_shape}")
                else:  # node is second arg
                    if other_shape[-1] != input_shape[-2]:
                        raise ValueError(f"Cannot remove node: Input shape {input_shape} is incompatible "
                                      f"with following matmul operation's shape {other_shape}")
    
    # If all validations pass, proceed with node removal
    node.replace_all_uses_with(input_node)
    graph.graph.erase_node(node)
    
    # Clean up module if necessary
    if node.op == "call_module":
        submodule_name = node.target
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







