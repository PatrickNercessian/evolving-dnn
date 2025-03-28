import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import random


def find_required_shapes(graph, node):
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
                return list(arg.meta['tensor_meta'].shape)
            elif isinstance(arg, list):
                return [get_shape(sub_arg) for sub_arg in arg]
            else:
                return arg
        
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
       

    # # if linear layer
    # if node.op == 'call_module' and isinstance(getattr(graph, node.target), nn.Linear):
    #     # get the pytorch linear layer
    #     linear_layer = getattr(graph, node.target)
    #     # get the input and output shapes
    #     input_shape = linear_layer.in_features
    #     output_shape = linear_layer.out_features
    #     return input_shape, output_shape
    # # if placeholder
    # elif node.op == 'placeholder':
    #     shape = list(node.meta['tensor_meta'].shape)
    #     return shape
    # # if relu
    # elif node.op == 'call_module' and isinstance(getattr(graph, node.target), nn.ReLU):
    #     # required input shape is the input of the user nodes
    #     user_node_inputs = []
    #     for user in node.users:
    #         user_node_inputs.append(find_required_shapes(graph, user))
    #     # the elements should be the same
    #     if not all(user_node_inputs[0] == user_node_inputs[i] for i in range(len(user_node_inputs))):
    #         raise ValueError(f"User node inputs are not the same for node {node.target}")
    #     # required output shape can be found in the meta of the node
    #     output_shape = list(node.meta['tensor_meta'].shape)[-1]
    #     return user_node_inputs[0], output_shape
    # # if cat
    # elif node.op == 'call_function' and node.target == torch.cat:
    #     # required input shapes are the input of the user nodes
    #     user_node_inputs = []
    #     for user in node.users:
    #         user_node_inputs.append(find_required_shapes(graph, user))
    #     return user_node_inputs
    # else:
    #     # required input shape of any given node is found in the meta of any node that feeds into it
    #     # check the node args for the node that feeds into the current node
    #     for arg in node.args:
    #         if isinstance(arg, torch.fx.Node):
    #             # get the last element of the shape of the node
    #             input_shape = list(arg.meta['tensor_meta'].shape)[-1]

    #     # required output shape can be found in the meta of the node
    #     output_shape = list(node.meta['tensor_meta'].shape)[-1]
    #     return input_shape, output_shape
