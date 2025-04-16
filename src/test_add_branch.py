import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from utils import find_required_shapes
from modelbank import simple_linear
from core import get_graph, add_node
import random

# Create a simple model with multiple layers
model = simple_linear(10, 10)
graph: torch.fx.GraphModule = get_graph(model, (1, 10))

# Print the initial graph structure
print("Initial Graph Structure:")
graph.graph.print_tabular()

# Print shapes for all nodes in the initial graph
print("\nInitial Node Shapes:")
for node in graph.graph.nodes:
    print(f"{node.name}: {find_required_shapes(node)}")

print("\n" + "-" * 40 + "\n")

# Find all module nodes in the graph
module_nodes = list(graph.graph.find_nodes(op="call_module"))
print(f"Found {len(module_nodes)} module nodes in the graph")

# Select a node to create branches from
if len(module_nodes) >= 1:
    # Select the first node to create branches from
    reference_node = module_nodes[0]
    
    print(f"Creating branches from {reference_node.name}")
    
    # Add branch nodes using add_node with 'branch' operation
    graph = add_node(graph, reference_node, "branch")
    
    # Print the updated graph structure
    print("\nUpdated Graph Structure:")
    graph.graph.print_tabular()
    
    # Print shapes for all nodes in the updated graph
    print("\nUpdated Node Shapes:")
    for node in graph.graph.nodes:
        print(f"{node.name}: {find_required_shapes(node)}")
    
    print("\n" + "-" * 40 + "\n")
    
    # Test forward pass
    print("Testing forward pass:")
    example_input = torch.randn(1, 10)
    
    # Test the graph with branch nodes
    output = graph(example_input)
    print(f"Output: {output}")
    
    # Print the generated code
    print("\nGenerated code:")
    print(graph.code)
    
else:
    print("Not enough module nodes to create branches") 