from collections import deque, defaultdict
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
            # print(neighbor_node.name)
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

def find_subgraph_connections(target_graph_module: torch.fx.GraphModule, input_nodes: set[torch.fx.Node], output_nodes: set[torch.fx.Node]):
    """
    Attempts to insert a subgraph into a target graph by finding compatible connection points.
    
    Args:
        target_graph: The graph to insert the subgraph into
        input_nodes: Set of nodes in the subgraph that receive external inputs
        output_nodes: Set of nodes in the subgraph that provide external outputs
    
    Returns:
        A tuple of (candidate_inputs, candidate_outputs) where each element is a dict mapping
        boundary nodes to potential matching nodes in the target graph
    """
    target_graph_nodes = list(target_graph_module.graph.nodes)
    
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
        if "tensor_meta" not in node1.meta or "tensor_meta" not in node2.meta:
            return False
        
        if not hasattr(node1.meta["tensor_meta"], "dtype") or not hasattr(node2.meta["tensor_meta"], "dtype"):
            return False  # TODO can we handle this better? we were getting errors with split module nodes
            
        # Check tensor properties
        if node1.meta["tensor_meta"].dtype != node2.meta["tensor_meta"].dtype:
            return False
        
        # TODO why did I comment this out, I forgot? Do we want it?
        # if not hasattr(node1.meta["tensor_meta"], "shape") or not hasattr(node2.meta["tensor_meta"], "shape"):
        #     return False
            
        # Check if shapes can be adapted
        if not are_shapes_adaptable(node1.meta["tensor_meta"].shape, node2.meta["tensor_meta"].shape):
            return False
            
        return True

    def get_candidates(boundary_nodes):
        all_candidates = {}
        for node in boundary_nodes:
            candidates = [n for n in target_graph_nodes if are_nodes_compatible(node, n)]
            if candidates:
                all_candidates[node] = candidates
        return all_candidates
    
    print("input_nodes", input_nodes)
    print("output_nodes", output_nodes)
    input_mapping, _ = _select_random_mapping(input_nodes, get_candidates(input_nodes))
    output_mapping, topo_target_input_nodes = _select_random_mapping(
        output_nodes,
        get_candidates(output_nodes),
        target_graph_module.graph,
        nodes_before=set(node for nodes in input_mapping.values() for node in nodes)
    )
    return input_mapping, topo_target_input_nodes, output_mapping

def _select_random_mapping(
    boundary_nodes: set[torch.fx.Node],
    candidates_dict: dict[torch.fx.Node, list[torch.fx.Node]],
    target_graph: torch.fx.Graph|None = None,
    nodes_before: set[torch.fx.Node]|None = None
):
    """
    Randomly selects compatible target node(s) for each boundary node, avoiding clashes and incorrect topological order.
    Args:
        boundary_nodes: Set of subgraph boundary nodes.
        candidates_dict: Dict mapping boundary nodes to lists of compatible target nodes.
        target_graph (optional): The target graph. Should be provided if nodes_before is provided.
        nodes_before (optional): Set of nodes before the input boundary nodes. Should be provided if target_graph is provided.
    Returns:
        Dict mapping subgraph boundary node -> selected target node(s).
        Topologically last node in nodes_before.
    """
    visited_nodes, visited_nodes_before = _get_all_before_nodes(target_graph, nodes_before) if nodes_before else ([], [])
    visited_nodes_set = set(visited_nodes)

    mapping = {}
    used_candidates = set()
    for node in boundary_nodes:
        candidates = [c for c in candidates_dict.get(node, []) if c not in used_candidates and c not in visited_nodes_set]
        if candidates:
            selected = [random.choice(candidates)] if nodes_before else random.sample(candidates, k=len(node.args))

            used_candidates.update(selected)
            mapping[node] = selected
    return mapping, visited_nodes_before

def _get_all_before_nodes(target_graph: torch.fx.Graph, nodes_before: set[torch.fx.Node]):
    node_list = target_graph.nodes
    visited_nodes = []
    visited_nodes_before = []
    for node in node_list:
        visited_nodes.append(node)
        if node not in nodes_before:
            continue

        visited_nodes_before.append(node)
        if set(visited_nodes_before) == nodes_before:
            break

    return visited_nodes, visited_nodes_before

def insert_subgraph(
    target_graph: torch.fx.GraphModule,
    subgraph_nodes: set[torch.fx.Node],
    input_mapping: dict[torch.fx.Node, torch.fx.Node],
    topo_target_input_nodes: list[torch.fx.Node],
    output_mapping: dict[torch.fx.Node, torch.fx.Node],
):
    """
    Inserts a subgraph into the target graph.
    Args:
        target_graph: The FX graph to insert into.
        subgraph_nodes: Set of nodes in the subgraph.
        input_mapping: Dict mapping subgraph input boundary node -> target node(s).
        topo_target_input_nodes: List of target nodes for the input boundary nodes, in topological order.
        output_mapping: Dict mapping subgraph output boundary node -> target node.
    Returns:
        Modified target_graph.
    """
    # 1. Copy modules if needed
    new_node_names = set()
    module_name_map = {}
    for node in subgraph_nodes:
        if node.op == "call_module":
            # Copy the module to the target graph
            module = node.graph.owning_module.get_submodule(node.target)
            name = get_unique_name(target_graph, node.name)
            target_graph.add_submodule(name, module)
            new_node_names.add(name)
            module_name_map[node.target] = name

    # 2. Map old nodes to new nodes in the target graph
    old_to_new = {}

    # 3. Insert nodes in topological order
    topo_order = kanh_algo(subgraph_nodes)

    print("topo_order", topo_order)
    print("input_mapping", input_mapping)
    for i, node in enumerate(topo_order):
        print("inserting node", node)
        if node in input_mapping:  # Handle input boundary nodes
            target_inputs = input_mapping[node]
            # Adapt shape if needed
            src_shape = node.meta["tensor_meta"].shape
            adapted_nodes = []
            last_idx = 0
            for target_input in target_inputs:
                tgt_shape = target_input.meta["tensor_meta"].shape
                adapted_node = target_input
                if src_shape != tgt_shape:
                    target_graph, adapted_node = adapt_node_shape(
                        target_graph,
                        node=target_input,
                        current_size=tgt_shape,
                        target_size=src_shape,
                        target_user=node
                    )
                idx = topo_target_input_nodes.index(target_input)
                if idx >= last_idx:
                    last_idx = idx
                    after_node = adapted_node
                adapted_nodes.append(adapted_node)
            new_args = tuple(adapted_nodes)
            print("adapted_nodes", adapted_nodes)
            new_node = _insert_node(target_graph, after_node, node, new_args, module_name_map)
        else:
            # Map args from old_to_new
            new_args = tuple(old_to_new[arg] if isinstance(arg, torch.fx.Node) else arg for arg in node.args)
            after_node = old_to_new[topo_order[i-1]] if i > 0 else topo_target_input_nodes[-1]
            new_node = _insert_node(target_graph, after_node=after_node, node=node, new_args=new_args, module_name_map=module_name_map)

        if new_node:
            new_node_names.add(new_node.name)
        print("new_args", new_args)
        old_to_new[node] = new_node

    # 4. For each output boundary node, replace the input of the mapped target node
    for sub_out, tgt_nodes in output_mapping.items():
        tgt_node = tgt_nodes[0]
        new_out_node = old_to_new[sub_out]
        # Replace the input of tgt_node with new_out_node
        new_args = tuple(new_out_node if arg == tgt_node.args[0] else arg for arg in tgt_node.args)  # TODO do we need to track arg indices? This is just replacing the first arg. But our current node compatibility check doesn't look at args at all, just tensor shapes... I'm confused.
        tgt_node.args = new_args

    print("old_to_new", old_to_new)
    

    visualize_graph(target_graph, "model_graph2_highlighted99", "graph2_highlighted99.svg", highlight_nodes=new_node_names)
    target_graph.graph.lint()
    target_graph.recompile()
    return target_graph, new_node_names

def _insert_node(target_graph: torch.fx.GraphModule, after_node: torch.fx.Node, node: torch.fx.Node, new_args, module_name_map):
    def _insert_call(func):
        print("inserting after", after_node)
        target = module_name_map[node.target] if (node.op == "call_module" and node.target in module_name_map) else node.target
        with target_graph.graph.inserting_after(after_node):
            return func(target, args=new_args, kwargs=node.kwargs)
    if node.op == "call_module":
        return _insert_call(target_graph.graph.call_module)
    elif node.op == "call_function":
        return _insert_call(target_graph.graph.call_function)
    elif node.op == "call_method":
        return _insert_call(target_graph.graph.call_method)
    else:
        # For placeholder/output, skip (handled separately)
        return

def kanh_algo(subgraph_nodes: set[torch.fx.Node]) -> list[torch.fx.Node]:
    # Build dependency graph (only within subgraph_nodes)
    in_degree = {node: 0 for node in subgraph_nodes}
    dependents = defaultdict(list)
    for node in subgraph_nodes:
        for input_node in node.all_input_nodes:
            if input_node in subgraph_nodes:
                in_degree[node] += 1
                dependents[input_node].append(node)

    # Initialize queue with nodes having in-degree 0
    topo_queue = deque([node for node in subgraph_nodes if in_degree[node] == 0])
    topo_order = []
    while topo_queue:
        node = topo_queue.popleft()
        topo_order.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                topo_queue.append(dependent)
    return topo_order

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

    model1 = GPT(config)
    example_input = torch.randint(0, config.vocab_size, (1, 1024))
    print("example_input", example_input)
    graph1 = get_graph(model1, example_input=example_input)
    model2 = GPT(config)
    graph2 = get_graph(model2, example_input=example_input)

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

    # print(best_subgraph_nodes)
    # print(best_input_boundary_nodes)
    # print(best_output_boundary_nodes)

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

    input_mapping, topo_target_input_nodes, output_mapping = find_subgraph_connections(graph2, best_input_boundary_nodes, best_output_boundary_nodes)
    print(input_mapping)
    print(output_mapping)

    graph2, new_node_names = insert_subgraph(graph2, best_subgraph_nodes, input_mapping, topo_target_input_nodes, output_mapping)

    visualize_graph(graph2, "model_graph2_highlighted", "graph2_highlighted.svg", highlight_nodes=new_node_names)
