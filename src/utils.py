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
