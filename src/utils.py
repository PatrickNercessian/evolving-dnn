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
        input_shape: The required shape of the input node
        output_shape: The required shape of the output node
    """

    # if linear layer
    if node.op == 'call_module' and isinstance(getattr(graph, node.target), nn.Linear):
        # get the pytorch linear layer
        linear_layer = getattr(graph, node.target)
        # get the input and output shapes
        input_shape = linear_layer.in_features
        output_shape = linear_layer.out_features
        return input_shape, output_shape
    else:
        raise ValueError(f"Node {node.target} is not a linear layer")

