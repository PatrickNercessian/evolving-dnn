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

def adapt_shape(graph, reference_node, target_shape, mode='linear'):
    """
    Adapts a node's output shape to match a target shape by modifying the module parameters
    
    Args:
        graph: The FX graph
        reference_node: The node whose output shape needs to be adapted
        target_shape: The desired output shape
        mode: The adaptation method ('linear', 'broadcast', 'squeeze', 'unsqueeze')
    Returns:
        reference_node: The node with the correct output shape
    """
    current_shape = reference_node.meta['tensor_meta'].shape
    
    if current_shape == target_shape:
        return reference_node
        
    if mode == 'linear':
        # Only works if reference node is a linear layer
        if reference_node.op == 'call_module':
            module_name = reference_node.target
            module = getattr(graph, module_name)
            
            if isinstance(module, nn.Linear):
                # Create a new linear layer with the desired output features
                in_features = module.in_features
                out_features = target_shape[-1]
                new_linear = nn.Linear(in_features, out_features)
                
                # Initialize the new weights using the original weights where possible
                with torch.no_grad():
                    min_out = min(module.out_features, out_features)
                    new_linear.weight.data[:min_out, :] = module.weight.data[:min_out, :]
                    if module.bias is not None:
                        new_linear.bias.data[:min_out] = module.bias.data[:min_out]
                
                # Replace the module in the graph
                setattr(graph, module_name, new_linear)
                return reference_node
            else:
                raise ValueError(f"Cannot adapt non-linear module with linear adaptation: {type(module)}")
        else:
            raise ValueError("Linear adaptation can only be applied to module nodes")
    
    elif mode == 'broadcast':
        # Add broadcasting dimension through unsqueeze and repeat
        if len(current_shape) < len(target_shape):
            # Add dimensions at the start
            diff = len(target_shape) - len(current_shape)
            for i in range(diff):
                reference_node = graph.graph.call_function(
                    torch.unsqueeze,
                    args=(reference_node, 0)
                )
        
        # Handle dimension sizes through repeat
        repeat_dims = []
        for c, t in zip(reference_node.meta['tensor_meta']['shape'], target_shape):
            repeat_dims.append(t if t != c else 1)
            
        reference_node = graph.graph.call_function(
            torch.repeat,
            args=(reference_node, *repeat_dims)
        )
        return reference_node
        
    elif mode == 'squeeze':
        # Remove extra dimensions
        while len(current_shape) > len(target_shape):
            reference_node = graph.graph.call_function(
                torch.squeeze,
                args=(reference_node, 0)
            )
        return reference_node
        
    elif mode == 'unsqueeze':
        # Add new dimensions
        while len(current_shape) < len(target_shape):
            reference_node = graph.graph.call_function(
                torch.unsqueeze,
                args=(reference_node, 0)
            )
        return reference_node
        
    raise ValueError(f"Unsupported shape adaptation mode: {mode}")

def add_node(graph, reference_node, operation, operator_params: torch.Tensor = None, name=None, adapt_direction='new'):
    """
    Adds a new node to the graph and handles both module registration and operators
    
    Args:
        graph: The FX graph
        reference_node: The target node to insert after
        operation: Either a PyTorch module or a string representing an operator
        operator_params: The accompanying tensor/value for the operation (required for operators)
        name: Optional name for the module (default: auto-generated)
        adapt_direction: Which node to adapt when shapes don't match ('new' or 'previous')
    Returns:
        graph: The modified graph
        name: The name of the added module/operator
    """
    if 'tensor_meta' not in reference_node.meta:
        raise ValueError("Input node missing shape metadata. Run shape_prop() first.")
    input_shape = reference_node.meta['tensor_meta'].shape
    
    if isinstance(operation, nn.Module):
        if isinstance(operation, nn.Linear):
            if operation.in_features != input_shape[-1]:
                if adapt_direction == 'previous':
                    # Adapt the previous node
                    reference_node = adapt_shape(graph, reference_node, (*input_shape[:-1], operation.in_features), mode='linear')
                else:
                    # Adapt the new layer
                    operation = nn.Linear(input_shape[-1], operation.out_features)
        
        # Handle PyTorch modules
        graph.add_submodule(name, operation)
        with graph.graph.inserting_after(reference_node):
            new_node = graph.graph.call_module(
                module_name=name,
                args=(reference_node,),
                kwargs={},
            )
    else:
        # Handle operators (like add, mul, etc)
        if operator_params is None:
            raise ValueError("operator_params is required for matrix operations")
            
        op_shape = operator_params.shape
        if operation in ['add', 'mul']:
            if not are_shapes_broadcastable(input_shape, op_shape):
                if adapt_direction == 'previous':
                    reference_node = adapt_shape(graph, reference_node, op_shape, mode='broadcast')
                else:
                    operator_params = operator_params.view(input_shape)
            
        elif operation == 'matmul':
            if len(input_shape) < 2 or len(op_shape) < 2:
                # Always add necessary dimensions regardless of direction
                if len(input_shape) < 2:
                    reference_node = adapt_shape(graph, reference_node, (*input_shape, 1), mode='unsqueeze')
                if len(op_shape) < 2:
                    operator_params = operator_params.unsqueeze(-1)
            
            if input_shape[-1] != op_shape[-2]:
                if adapt_direction == 'previous':
                    reference_node = adapt_shape(graph, reference_node, (*input_shape[:-1], op_shape[-2]), mode='linear')
                else:
                    # Create new operator_params with matching dimensions
                    new_params = nn.Linear(input_shape[-1], op_shape[-2])(operator_params)
                    operator_params = new_params
        
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
    
    # Fix any self-references in the new node's args
    if isinstance(operation, nn.Module):
        new_node.args = (reference_node,) # Assume that the module is a linear layer
    else:
        # For operators, replace any references to new_node with reference_node in the args
        new_args = tuple(reference_node if arg is new_node else arg for arg in new_node.args)
        new_node.args = new_args

    graph.graph.lint()
    graph.recompile()
        
        
    return graph, name

def remove_node(graph, reference_node, adapt_direction='previous'):
    """
    Removes a node from the graph and cleans up its module if it was dynamically added.
    
    Args:
        graph: The FX graph
        reference_node: The node to remove
        adapt_direction: Which nodes to adapt when shapes don't match ('previous' or 'following')
    Returns:
        graph: The modified graph
    """
    input_node = reference_node.args[0]
    
    if 'tensor_meta' not in input_node.meta or 'tensor_meta' not in reference_node.meta:
        raise ValueError("Nodes missing shape metadata. Run shape_prop() first.")
        
    input_shape = input_node.meta['tensor_meta'].shape
    
    print(f"Input shape: {input_shape}")
    
    # Track nodes that need adaptation
    nodes_to_adapt = []
    
    # Collect all shape mismatches
    for user in reference_node.users:
        if user.op == 'call_module':
            if hasattr(getattr(graph, user.target), 'in_features'):
                required_features = getattr(graph, user.target).in_features
                if input_shape[-1] != required_features:
                    # Wrap the required features in a proper shape tuple
                    target_shape = (*input_shape[:-1], required_features)
                    nodes_to_adapt.append((user, target_shape))
                    
        elif user.op == 'call_function':
            if user.target in [torch.add, torch.mul]:
                other_arg = user.args[1] if user.args[0] is reference_node else user.args[0]
                if isinstance(other_arg, torch.fx.Node):
                    other_shape = other_arg.meta['tensor_meta']['shape']
                    if not are_shapes_broadcastable(input_shape, other_shape):
                        nodes_to_adapt.append((user, other_shape))
                        
            elif user.target == torch.matmul:
                other_arg = user.args[1] if user.args[0] is reference_node else user.args[0]
                other_shape = other_arg.meta['tensor_meta']['shape']
                if user.args[0] is reference_node:  # reference_node is first arg
                    if input_shape[-1] != other_shape[-2]:
                        nodes_to_adapt.append((user, (*input_shape[:-1], other_shape[-2])))
                else:  # reference_node is second arg
                    if other_shape[-1] != input_shape[-2]:
                        nodes_to_adapt.append((user, (*input_shape[:-2], other_shape[-1], input_shape[-1])))
    print(f"Nodes to adapt: {nodes_to_adapt}")
    
    for user, target_shape in nodes_to_adapt:
        print(f"User: {user.op}, Target shape: {target_shape}")
        
    # Apply adaptations based on direction
    if adapt_direction == 'previous':
        # Adapt the input node to match all requirements
        for user, target_shape in nodes_to_adapt:
            input_node = adapt_shape(graph, input_node, target_shape, 
                                   mode='linear' if user.op == 'call_module' else 'broadcast')
    else:  # adapt_direction == 'following'
        # Adapt each following node individually
        for user, _ in nodes_to_adapt:
            if user.op == 'call_module':
                # Replace the module with one that matches input shape
                old_module = getattr(graph, user.target)
                new_module = nn.Linear(input_shape[-1], old_module.out_features)
                setattr(graph, user.target, new_module)
            elif user.op == 'call_function':
                # Insert adaptation layer before the operation
                adapted_node = adapt_shape(graph, input_node, input_shape, mode='broadcast')
                user.args = tuple(adapted_node if arg is reference_node else arg for arg in user.args)
    
    # Proceed with node removal
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


