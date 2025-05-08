import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import get_graph
from src.visualization import visualize_graph
from src.utils import adapt_node_shape, get_unique_name

from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

import torch
import torch.fx
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

    # Find boundary nodes that are within the subgraph
    input_nodes = {node for node in subgraph_nodes 
                  if any(input_node not in subgraph_nodes for input_node in node.all_input_nodes)}
    output_nodes = {node for node in subgraph_nodes 
                   if any(user_node not in subgraph_nodes for user_node in node.users)}
    
    return subgraph_nodes, input_nodes, output_nodes

def find_potential_subgraph_connections(target_graph, subgraph_nodes, input_nodes, output_nodes):
    """
    Attempts to insert a subgraph into a target graph by finding compatible connection points.
    
    Args:
        target_graph: The graph to insert the subgraph into
        subgraph_nodes: Set of nodes in the subgraph
        input_nodes: Set of nodes in the subgraph that receive external inputs
        output_nodes: Set of nodes in the subgraph that provide external outputs
    
    Returns:
        A tuple of (candidate_inputs, candidate_outputs) where each element is a dict mapping
        boundary nodes to potential matching nodes in the target graph
    """
    target_graph_nodes = list(target_graph.graph.nodes)
    
    def are_shapes_adaptable(shape1, shape2):
        # Both shapes should have same number of dimensions
        if len(shape1) != len(shape2):
            return False
        # Only batch dimension (first dim) needs to match
        if shape1[0] != shape2[0]:
            return False
        return True
    
    def are_nodes_compatible(node1, node2):
        # Skip placeholder and output nodes
        if node1.op in ["placeholder", "output"] or node2.op in ["placeholder", "output"]:
            return False
            
        # Check tensor metadata
        if not hasattr(node1, 'tensor_meta') or not hasattr(node2, 'tensor_meta'):
            return False
            
        # Check tensor properties
        if node1.tensor_meta.dtype != node2.tensor_meta.dtype:
            return False
            
        # Check if shapes can be adapted
        if not are_shapes_adaptable(node1.tensor_meta.shape, node2.tensor_meta.shape):
            return False
            
        return True

    def get_candidates(boundary_nodes):
        all_candidates = {}
        for node in boundary_nodes:
            candidates = [n for n in target_graph_nodes if are_nodes_compatible(node, n)]
            if candidates:
                all_candidates[node] = candidates
        return all_candidates
    
    return get_candidates(input_nodes), get_candidates(output_nodes)

def _select_random_mapping(boundary_nodes, candidates_dict):
    """
    Randomly selects a compatible target node for each boundary node, avoiding clashes (no candidate is selected more than once).
    Args:
        boundary_nodes: Set of subgraph boundary nodes.
        candidates_dict: Dict mapping boundary nodes to lists of compatible target nodes.
    Returns:
        Dict mapping subgraph boundary node -> selected target node.
    """
    mapping = {}
    used_candidates = set()
    for node in boundary_nodes:
        candidates = [c for c in candidates_dict.get(node, []) if c not in used_candidates]
        if candidates:
            selected = random.choice(candidates)
            mapping[node] = selected
            used_candidates.add(selected)
    return mapping

def insert_subgraph(
    target_graph: torch.fx.GraphModule,
    subgraph_nodes: set[torch.fx.Node],
    input_mapping: dict[torch.fx.Node, torch.fx.Node],
    output_mapping: dict[torch.fx.Node, torch.fx.Node],
):
    """
    Inserts a subgraph into the target graph.
    Args:
        target_graph: The FX graph to insert into.
        subgraph_nodes: Set of nodes in the subgraph.
        input_boundary_nodes: Set of subgraph input boundary nodes.
        output_boundary_nodes: Set of subgraph output boundary nodes.
        input_mapping: Dict mapping subgraph input boundary node -> target node.
        output_mapping: Dict mapping subgraph output boundary node -> target node.
    Returns:
        Modified target_graph.
    """
    # 1. Copy modules if needed
    for node in subgraph_nodes:
        if node.op == "call_module":
            # Copy the module to the target graph if not already present
            module = node.graph.owning_module.get_submodule(node.target)
            if not hasattr(target_graph, node.target):  # TODO we should always add it, since we're doing names like linear3, and they might be different. But I need to do the unique name gen again
                target_graph.add_submodule(node.target, module)

    # 2. Map old nodes to new nodes in the target graph
    old_to_new = {}

    # 3. Insert nodes in topological order
    for node in subgraph_nodes:
        # Handle input boundary nodes
        if node in input_mapping:
            target_input = input_mapping[node]
            # Adapt shape if needed
            src_shape = node.tensor_meta.shape
            tgt_shape = target_input.tensor_meta.shape
            adapted_node = target_input
            if src_shape != tgt_shape:
                target_graph, adapted_node = adapt_node_shape(
                    target_graph,
                    node=target_input,
                    current_size=tgt_shape,
                    target_size=src_shape,
                    target_user=node
                )
            new_args = (adapted_node,)
        else:
            # Map args from old_to_new
            new_args = tuple(old_to_new.get(arg, arg) if isinstance(arg, torch.fx.Node) else arg for arg in node.args)

        # Insert node
        if node.op == "call_module":
            new_node = target_graph.graph.call_module(node.target, args=new_args, kwargs=node.kwargs)
        elif node.op == "call_function":
            new_node = target_graph.graph.call_function(node.target, args=new_args, kwargs=node.kwargs)
        elif node.op == "call_method":
            new_node = target_graph.graph.call_method(node.target, args=new_args, kwargs=node.kwargs)
        else:
            # For placeholder/output, skip (handled separately)
            continue

        old_to_new[node] = new_node

    # 4. For each output boundary node, replace the input of the mapped target node
    for sub_out, tgt_node in output_mapping.items():
        new_out_node = old_to_new[sub_out]
        # Replace the input of tgt_node with new_out_node
        new_args = tuple(new_out_node if arg == tgt_node.args[0] else arg for arg in tgt_node.args)
        tgt_node.args = new_args

    target_graph.graph.lint()
    target_graph.recompile()
    return target_graph

# TESTING STUFF BELOW
if __name__ == "__main__":
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

    example_input_shape = (1, 10)
    model1 = GPT(config)
    graph1 = get_graph(model1, example_input_shape)
    model2 = GPT(config)
    graph2 = get_graph(model2, example_input_shape)

    # print(graph.graph)

    visualize_graph(graph1, "model_graph", "graph.svg")

    subgraph_nodes = set()
    # lowest_ratio_of_boundary_to_nodes = float('inf')
    lowest_num_boundary_nodes = float('inf')
    for i in range(20):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        subgraph_nodes, input_boundary_nodes, output_boundary_nodes = random_subgraph(graph1, num_nodes)
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
    visualize_graph(graph1, "model_graph_highlighted", "graph_highlighted.svg", highlight_nodes=subgraph_node_names)

    # After your existing print statements, add:
    print("\nPrinting weights of linear layers and layer norms:")

    # Print weights for each node in best_subgraph_nodes
    for node in best_subgraph_nodes:
        if node.op == 'call_module':  # Check if it's a module call
            target_module = graph1.get_submodule(node.target)  # Get the actual module
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

    input_candidates, output_candidates = find_potential_subgraph_connections(graph2, best_subgraph_nodes, best_input_boundary_nodes, best_output_boundary_nodes)
    print(input_candidates)
    print(output_candidates)
    input_mapping = _select_random_mapping(best_input_boundary_nodes, input_candidates)
    output_mapping = _select_random_mapping(best_output_boundary_nodes, output_candidates)
    print(input_mapping)
    print(output_mapping)

    graph2 = insert_subgraph(graph2, best_subgraph_nodes, input_mapping, output_mapping)

    visualize_graph(graph2, "model_graph_highlighted", "graph_highlighted.svg", highlight_nodes=subgraph_node_names)
