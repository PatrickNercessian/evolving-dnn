import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import get_graph
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

graph = get_graph(model, None)

# print(graph.graph)

# visualize_graph(graph, "model_graph", "graph.svg")

# anchor_node = random.choice(list(graph.graph.find_nodes(op="call_module")))

MAX_BOUNDARY_NODES = 10
MIN_NODES = 4
MAX_NODES = 32

def random_subgraph(graph_module, num_nodes):
    """
    This function returns a random subgraph of the given graph.

    Args:
        graph_module: The graph module to get the subgraph from.
        num_nodes: The number of nodes in the subgraph.

    Returns:
        A tuple of the candidate nodes, input boundary nodes, and output boundary nodes.
    """
    all_nodes = list(graph_module.graph.nodes)
    anchor_node = random.choice(all_nodes)
    subgraph_nodes = {anchor_node}
    frontier_nodes = [anchor_node]
    while frontier_nodes and len(subgraph_nodes) < num_nodes:
        current_node = frontier_nodes.pop()
        candidate_nodes = set()
        for neighbor_node in (*current_node.all_input_nodes, *current_node.users):
            print(neighbor_node.name)
            if neighbor_node not in subgraph_nodes and neighbor_node.op != "placeholder" and neighbor_node.op != "output" and not "cross_entropy" in neighbor_node.name and not "targets" in neighbor_node.name:
                candidate_nodes.add(neighbor_node)
        
        if len(subgraph_nodes) + len(candidate_nodes) <= num_nodes:
            for candidate_node in candidate_nodes:
                subgraph_nodes.add(candidate_node)
                frontier_nodes.append(candidate_node)

    # TODO these boundaries could still be placeholder ops, I think that's bad?
    # TODO oh wait, should the boundary nodes be the actual ones in the subgraph, rather than their neighbors?
    input_boundary_nodes = {parent_node for node in subgraph_nodes for parent_node in node.all_input_nodes
                          if parent_node not in subgraph_nodes}
    output_boundary_nodes = {user_node for node in subgraph_nodes for user_node in node.users
                           if user_node not in subgraph_nodes}
    return subgraph_nodes, input_boundary_nodes, output_boundary_nodes

subgraph_nodes = set()
# lowest_ratio_of_boundary_to_nodes = float('inf')
lowest_num_boundary_nodes = float('inf')
for i in range(20):
    num_nodes = random.randint(MIN_NODES, MAX_NODES)
    subgraph_nodes, input_boundary_nodes, output_boundary_nodes = random_subgraph(graph, num_nodes)
    # ratio_of_boundary_to_nodes = len(boundary_nodes) / len(candidate_nodes)
    # if len(boundary_nodes) <= MAX_BOUNDARY_NODES and ratio_of_boundary_to_nodes < lowest_ratio_of_boundary_to_nodes:
    #     lowest_ratio_of_boundary_to_nodes = ratio_of_boundary_to_nodes
    boundary_nodes = input_boundary_nodes | output_boundary_nodes
    if len(boundary_nodes) <= MAX_BOUNDARY_NODES and len(subgraph_nodes) > MIN_NODES and len(boundary_nodes) < lowest_num_boundary_nodes:
        lowest_num_boundary_nodes = len(boundary_nodes)
        best_subgraph_nodes = subgraph_nodes
        best_input_boundary_nodes = input_boundary_nodes
        best_output_boundary_nodes = output_boundary_nodes

print(best_subgraph_nodes)
print(best_input_boundary_nodes)
print(best_output_boundary_nodes)

# Extract node names for highlighting
subgraph_node_names = {node.name for node in best_subgraph_nodes}

# Visualize the graph with the subgraph highlighted
visualize_graph(graph, "model_graph_highlighted", "graph_highlighted.svg", highlight_nodes=subgraph_node_names)

# After your existing print statements, add:
print("\nPrinting weights of linear layers and layer norms:")

# Print weights for each node in best_subgraph_nodes
import torch
for node in best_subgraph_nodes:
    if node.op == 'call_module':  # Check if it's a module call
        target_module = graph.get_submodule(node.target)  # Get the actual module
        if isinstance(target_module, torch.nn.Linear):
            print(f"\nLinear layer {node.name}:")
            print(f"Weight shape: {target_module.weight.shape}")
            print(f"Weight stats: min={target_module.weight.min().item():.4f}, max={target_module.weight.max().item():.4f}, mean={target_module.weight.mean().item():.4f}")
            if target_module.bias is not None:
                print(f"Bias shape: {target_module.bias.shape}")
                print(f"Bias stats: min={target_module.bias.min().item():.4f}, max={target_module.bias.max().item():.4f}, mean={target_module.bias.mean().item():.4f}")
        
        elif isinstance(target_module, torch.nn.LayerNorm):
            print(f"\nLayerNorm {node.name}:")
            print(f"Weight shape: {target_module.weight.shape}")
            print(f"Weight stats: min={target_module.weight.min().item():.4f}, max={target_module.weight.max().item():.4f}, mean={target_module.weight.mean().item():.4f}")
            print(f"Bias shape: {target_module.bias.shape}")
            print(f"Bias stats: min={target_module.bias.min().item():.4f}, max={target_module.bias.max().item():.4f}, mean={target_module.bias.mean().item():.4f}")
