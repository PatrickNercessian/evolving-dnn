import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from src.core import get_graph, add_node, remove_node, shape_prop
from src.visualization import visualize_graph

from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

config = CN()
config.vocab_size = 50257
# config.model_type = 'gpt2'
config.model_type = None
config.n_layer = 2
config.n_head = 2
config.n_embd = 768
config.block_size = 1024
config.embd_pdrop = 0.1
config.attn_pdrop = 0.1
config.resid_pdrop = 0.1
config.is_proxy_for_fx = True

model = GPT(config)

graph = get_graph(model)

print(graph.graph)

visualize_graph(graph, "model_graph", "graph.svg")
