import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from src.nn.variation.utils import find_required_shapes, add_skip_connection
from src.tests.modelbank import simple_linear
from src.nn.core import get_graph, add_node
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

# Select two nodes to create a skip connection between
# We need to ensure the first node comes before the second node in the graph
if len(module_nodes) >= 2:
    # Sort nodes by their position in the graph
    sorted_nodes = []
    for node in graph.graph.nodes:
        if node in module_nodes:
            sorted_nodes.append(node)
    
    # Select the first and third nodes (if available) to create a skip connection
    first_node_idx = 0
    second_node_idx = min(2, len(sorted_nodes) - 1)
    
    first_node = sorted_nodes[first_node_idx]
    second_node = sorted_nodes[second_node_idx]
    
    print(f"Creating skip connection from {first_node.name} to {second_node.name}")
    
    # Method 1: Using add_node with "skip" operation
    print("\nMethod 1: Using add_node with 'skip' operation")
    graph = add_node(graph, second_node, "skip", first_node=first_node)
    
    # Print the updated graph structure
    print("\nUpdated Graph Structure:")
    graph.graph.print_tabular()
    
    # Print shapes for all nodes in the updated graph
    print("\nUpdated Node Shapes:")
    for node in graph.graph.nodes:
        print(f"{node.name}: {find_required_shapes(node)}")
    
    print("\n" + "-" * 40 + "\n")

    
    # Test forward pass with both graphs
    print("Testing forward pass with both graphs:")
    example_input = torch.randn(1, 10)
    
    # Test first graph
    output1 = graph(example_input)
    print(f"Output from Method 1: {output1}")
    
    
    # Print the generated code for both graphs
    print("\nGenerated code for Method 1:")
    print(graph.code)
    
else:
    print("Not enough module nodes to create a skip connection") 