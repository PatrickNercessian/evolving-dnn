import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from utils import find_required_shapes
from modelbank import simple_linear
from core import get_graph, add_node, remove_node
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

# First, add a new node to the graph to make it more interesting
if len(module_nodes) >= 1:
    # Select a random node to add after
    node_to_add_after = random.choice(module_nodes)
    print(f"Adding a new linear node after {node_to_add_after.name}")
    
    # Add a new linear node
    graph = add_node(graph, node_to_add_after, "linear")
    
    # Print the graph structure after adding a node
    print("\nGraph Structure After Adding Node:")
    graph.graph.print_tabular()
    
    # Print shapes for all nodes after adding
    print("\nNode Shapes After Adding:")
    for node in graph.graph.nodes:
        print(f"{node.name}: {find_required_shapes(node)}")
    
    print("\n" + "-" * 40 + "\n")
    
    # Now find all module nodes again (including the new one)
    module_nodes = list(graph.graph.find_nodes(op="call_module"))
    print(f"Found {len(module_nodes)} module nodes in the graph after adding")
    
    # Select a random node to remove (but not the first or last node)
    if len(module_nodes) >= 3:
        # Get nodes in order
        sorted_nodes = []
        for node in graph.graph.nodes:
            if node in module_nodes:
                sorted_nodes.append(node)
        
        # Select a node in the middle to remove
        node_to_remove_idx = len(sorted_nodes) // 2
        node_to_remove = sorted_nodes[node_to_remove_idx]
        
        print(f"Removing node {node_to_remove.name}")
        
        # Remove the node
        graph, new_node = remove_node(graph, node_to_remove)
        
        # Print the graph structure after removing a node
        print("\nGraph Structure After Removing Node:")
        graph.graph.print_tabular()
        
        # Print shapes for all nodes after removing
        print("\nNode Shapes After Removing:")
        for node in graph.graph.nodes:
            print(f"{node.name}: {find_required_shapes(node)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test forward pass
        print("Testing forward pass:")
        example_input = torch.randn(1, 10)
        
        # Test the graph
        output = graph(example_input)
        print(f"Output: {output}")
        
        # Print the generated code
        print("\nGenerated code:")
        print(graph.code)
    else:
        print("Not enough module nodes to safely remove one")
else:
    print("Not enough module nodes to add a new one") 