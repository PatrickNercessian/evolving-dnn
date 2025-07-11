import logging
from ptflops import get_model_complexity_info
import torch
import copy

class FlopsAccumulator:
    def __init__(self):
        self.total_flops = 0

    def add(self, flops):
        self.total_flops += flops

    def log(self, logger=logging):
        logger.info(f"Total FLOPs used: {self.total_flops:,}")

def estimate_flops(model, example_input, batch_size=1):
    """
    Estimate FLOPs for a model using ptflops, given an example input tensor.
    Returns FLOPs per batch (forward + backward).
    """
    input_shape = tuple(example_input.shape[1:])  # Remove batch dimension
    input_dtype = example_input.dtype
    def input_constructor(input_shape):
        return torch.zeros((1, *input_shape), dtype=input_dtype)
    model_for_flops = copy.deepcopy(model)
    flops, params = get_model_complexity_info(
        model_for_flops, input_shape,
        as_strings=False, print_per_layer_stat=False, verbose=False,
        input_constructor=input_constructor
    )
    flops_per_batch = flops * batch_size * 2  # Forward + backward
    return flops_per_batch 