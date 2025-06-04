import logging

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from .individual_graph_module import NeuralNetworkIndividualGraphModule


def get_graph(model: nn.Module, input_shape: tuple|None = None, example_input: torch.Tensor|None = None):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
        input_shape: tuple specifying input tensor shape (batch, seq_len, dim)  # SHAPE NOTE: input_shape includes batch dimension
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace with shape information
    """
        
    # Symbolically trace the model to get computation graph
    if example_input is not None:
        graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model), example_input=example_input)
    else:
        graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model))
    
    # Perform shape propagation if input_shape is provided
    if input_shape is not None and example_input is None:
        # Create example input
        example_input = torch.randn(input_shape)  # SHAPE NOTE: Using full shape including batch dimension
        graph.example_input = example_input
        
    if example_input is not None:
        # Get the first node (should be placeholder/input)
        placeholder = next(iter(graph.graph.nodes))
        placeholder.meta['tensor_meta'] = {
            'dtype': example_input.dtype,
            'shape': example_input.shape,  # SHAPE NOTE: Storing full shape including batch dimension
            'requires_grad': example_input.requires_grad
        }
        
        # Run shape propagation
        logging.debug(f"example_input: {example_input}")
        ShapeProp(graph).propagate(example_input)  # SHAPE NOTE: Shape propagation uses full shape including batch dimension
    
    return graph
