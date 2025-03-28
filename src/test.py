import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from add import add_linear
from utils import find_required_shapes 
from modelbank import CausalSelfAttention, simple_linear
from core import get_graph
from utils import find_required_shapes


# Initialize model and config
# class Config:
#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

# def setup_model():
#     config = Config(n_head=12, n_embd=768, block_size=1024, attn_pdrop=0.1, resid_pdrop=0.1, batch_size=1)
#     model = CausalSelfAttention(config)
#     # Get the graph and create a GraphModule
#     graph = get_graph(model, (1, 1024, 768))
#     return graph, model

# graph, model = setup_model()
model = simple_linear(10, 10)
graph = get_graph(model, (1, 10))


graph.graph.print_tabular()


# loop through graph nodes and test find_required_shapes
for i in graph.graph.nodes:
    print(i, end=" ")
    print(find_required_shapes(graph, i))










