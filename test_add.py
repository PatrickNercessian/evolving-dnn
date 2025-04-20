import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from src.utils import find_required_shapes 
from src.modelbank import CausalSelfAttention, simple_linear
from src.core import get_graph, add_node, remove_node, adapt_connections
from src.utils import find_required_shapes
import random

model = simple_linear(10, 3)
graph: torch.fx.GraphModule = get_graph(model, (1, 10))


graph.graph.print_tabular()


# loop through graph nodes and test find_required_shapes
for i in graph.graph.nodes:
    print(i, end=" ")
    print(find_required_shapes(i))

print("--------------------------------")


# get random node to add
node_to_add = random.choice(list(graph.graph.find_nodes(op="call_module")))
print(node_to_add)
add_node(graph, node_to_add, "linear", input_size=26, output_size=3)


graph.graph.print_tabular()

# loop through graph nodes and test find_required_shapes
for i in graph.graph.nodes:
    print(i, end=" ")
    print(find_required_shapes(i))

print("--------------------------------")

# generate code from graph and print it
print(graph.code)

print("--------------------------------")

# generate random tensor to test forward pass
example_input = torch.randn(1, 10)

# test forward pass
output = graph(example_input)
print(output)

