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

def random_subgraph(graph_module: torch.fx.GraphModule, num_nodes: int):
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
    while not _is_allowed_subgraph_node_type(anchor_node):
        print("WARNING: picked node with non-allowed type or name:", anchor_node.op, anchor_node.name)
        anchor_node = random.choice(all_nodes)
    subgraph_nodes = {anchor_node}
    frontier_nodes = [anchor_node]
    should_continue_past_num_nodes = False
    while frontier_nodes and (len(subgraph_nodes) < num_nodes or should_continue_past_num_nodes):
        current_node = frontier_nodes.pop()
        candidate_nodes = set()
        for neighbor_node in (*current_node.all_input_nodes, *current_node.users):
            # print(neighbor_node.name)
            if neighbor_node not in subgraph_nodes and _is_allowed_subgraph_node_type(neighbor_node):
                candidate_nodes.add(neighbor_node)
        
        if len(subgraph_nodes) + len(candidate_nodes) <= num_nodes:
            for candidate_node in candidate_nodes:
                subgraph_nodes.add(candidate_node)
                frontier_nodes.append(candidate_node)

    # Find boundary nodes that are within the subgraph
    input_mapping, output_mapping = {}, {}
    for node in subgraph_nodes:
        is_boundary = False
        input_mapping[node], output_mapping[node] = [], []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in subgraph_nodes:
                    input_mapping[node].append(arg)
                elif _node_has_shape(arg):
                    input_mapping[node].append(None)  # placeholder for target graph replacement arg
                else:
                    raise ValueError(f"WARNING: boundary node {node} NEIGHBOR arg {arg} has no shape, killing subgraph")
            else:
                input_mapping[node].append(arg)
        if not any(arg is None for arg in input_mapping[node]):
            del input_mapping[node]  # if all node inputs are in the subgraph, we don't need to keep the mapping
        else:
            is_boundary = True
        
        
        for user_node in node.users:
            if user_node in subgraph_nodes:
                output_mapping[node].append(user_node)
            elif _node_has_shape(user_node):
                output_mapping[node].append(None)  # placeholder for target graph replacement user
            else:
                raise ValueError(f"WARNING: boundary node {node} NEIGHBOR user_node {user_node} has no shape, killing subgraph")
        if all(user_node is not None for user_node in output_mapping[node]):
            del output_mapping[node]  # if all node outputs are in the subgraph, we don't need to keep the mapping
        else:
            is_boundary = True

        if is_boundary and not _node_has_shape(node):
            raise ValueError("WARNING: boundary node with no shape, killing subgraph", node)

    print("input_mapping before", input_mapping)
    print("output_mapping before", output_mapping)
    return subgraph_nodes, input_mapping, output_mapping

def _is_allowed_subgraph_node_type(node: torch.fx.Node):
    return node.op != "placeholder" and node.op != "output" and not "cross_entropy" in node.name and not "targets" in node.name

def _node_has_shape(node: torch.fx.Node):
    return "tensor_meta" in node.meta and hasattr(node.meta["tensor_meta"], "shape")

def find_subgraph_connections(
    target_graph_module: torch.fx.GraphModule,
    input_mapping: dict[torch.fx.Node, list[torch.fx.Node|None]],
    output_mapping: dict[torch.fx.Node, list[torch.fx.Node|None]]
):
    """
    Attempts to insert a subgraph into a target graph by finding compatible connection points.
    
    Args:
        target_graph: The graph to insert the subgraph into
        input_mapping: Dict mapping subgraph input boundary node -> target graph args. A None arg implies we need to select a compatible target arg.
        output_mapping: Dict mapping subgraph output boundary node -> target graph users. A None user implies we need to select a compatible target user.
    
    Returns:
        A tuple of (input_mapping, topo_target_input_nodes, output_mapping)
        input_mapping: Dict mapping subgraph input boundary node -> target graph args.
        topo_target_input_nodes: List of target nodes for the input boundary nodes, in topological order.
        output_mapping: Dict mapping subgraph output boundary node -> target graph users.
    """
    target_graph_nodes = list(target_graph_module.graph.nodes)
    
    def are_nodes_compatible(node1, node2):
        # Skip placeholder and output nodes
        if node1.op in ["placeholder", "output"] or node2.op in ["placeholder", "output"]:
            return False
            
        # Check tensor metadata
        if not _node_has_shape(node1) or not _node_has_shape(node2):
            return False
        
        if node1.meta["tensor_meta"].dtype != node2.meta["tensor_meta"].dtype:
            return False

        # Ensure batch dimension matches
        if node1.meta["tensor_meta"].shape[0] != node2.meta["tensor_meta"].shape[0]:
            return False
            
        return True

    def get_candidates(boundary_nodes):
        all_candidates = {}
        for node in boundary_nodes:  # TODO do we need each candidate list to be for an arg index?
            candidates = [n for n in target_graph_nodes if are_nodes_compatible(node, n)]
            if candidates:
                all_candidates[node] = candidates
            else:
                print("WARNING: no candidates found for node", node)
        return all_candidates
    
    input_mapping, _ = _select_random_mapping(input_mapping, get_candidates(input_mapping))
    output_mapping, topo_target_input_nodes = _select_random_mapping(
        output_mapping,
        get_candidates(output_mapping),
        target_graph_module.graph,
        nodes_before=set(node for nodes in input_mapping.values() for node in nodes)
    )
    return input_mapping, topo_target_input_nodes, output_mapping

def _select_random_mapping(
    boundary_nodes: dict[torch.fx.Node, list[torch.fx.Node|None]],
    candidates_dict: dict[torch.fx.Node, list[torch.fx.Node]],
    target_graph: torch.fx.Graph|None = None,
    nodes_before: set[torch.fx.Node]|None = None
):
    """
    Randomly selects compatible target node(s) for each boundary node, avoiding clashes (and incorrect topological order if selecting output nodes).
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

    used_candidates = set()
    for node, args_or_users in boundary_nodes.items():
        for i, arg_or_user in enumerate(args_or_users):  # TODO if these are users (meaning output_mapping), we don't necessarily need the same number of users in the target graph... But we are forcing it to be the same here.
            if arg_or_user is not None:
                continue
            candidates = [c for c in candidates_dict.get(node, []) if c not in used_candidates and c not in visited_nodes_set]
            if candidates:
                selected = random.choice(candidates)
                used_candidates.add(selected)
                boundary_nodes[node][i] = selected
            else:
                print("WARNING: no candidates found for node", node)
                raise ValueError("no candidates found for node", node)

    return boundary_nodes, visited_nodes_before

def _get_all_before_nodes(target_graph: torch.fx.Graph, nodes_before: set[torch.fx.Node]):
    node_list = target_graph.nodes
    visited_nodes, visited_nodes_before = [], []
    for node in node_list:
        visited_nodes.append(node)
        if node not in nodes_before:
            continue

        visited_nodes_before.append(node)
        if set(visited_nodes_before) == nodes_before:
            break

    return visited_nodes, visited_nodes_before

def insert_subgraph(
    target_graph_module: torch.fx.GraphModule,
    subgraph_nodes: set[torch.fx.Node],
    input_mapping: dict[torch.fx.Node, list[torch.fx.Node|list[torch.fx.Node]]],
    topo_target_input_nodes: list[torch.fx.Node],  # TODO ideally we can just sort the input_mapping to be topographical instead of needing this list
    output_mapping: dict[torch.fx.Node, list[torch.fx.Node|list[torch.fx.Node]]],
):
    """
    Inserts a subgraph into the target graph.
    Args:
        target_graph: The FX graph to insert into.
        subgraph_nodes: Set of nodes in the subgraph.
        input_mapping: Dict mapping subgraph input boundary node -> target node(s). If it's a list, it means we need to select one of the target nodes.
        topo_target_input_nodes: List of target nodes for the input boundary nodes, in topological order.
        output_mapping: Dict mapping subgraph output boundary node -> target node(s). If it's a list, it means we need to select one of the target nodes.
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
            name = get_unique_name(target_graph_module, node.target)
            target_graph_module.add_submodule(name, module)
            new_node_names.add(name)
            module_name_map[node.target] = name

    # 2. Map old nodes to new nodes in the target graph
    old_to_new = {}

    # 3. Insert nodes in topological order
    topo_order = kanh_algo(subgraph_nodes)

    print("topo_order", topo_order)
    print("input_mapping", input_mapping)
    print("output_mapping", output_mapping)
    for i, node in enumerate(topo_order):
        print("inserting node", node)
        after_node = old_to_new[topo_order[i-1]] if i > 0 else topo_target_input_nodes[-1]
        if node.op == "call_module":
            print("as", module_name_map[node.target])
        if node in input_mapping:  # Handle input boundary nodes
            target_inputs = []
            for j in range(len(input_mapping[node])):
                if input_mapping[node][j] in subgraph_nodes and isinstance(input_mapping[node][j], torch.fx.Node):
                    target_inputs.append(old_to_new[input_mapping[node][j]])
                else:
                    target_inputs.append(input_mapping[node][j])  # these are already in the target graph

            new_node = _insert_node(target_graph_module, after_node=after_node, node=node, new_args=tuple(target_inputs), module_name_map=module_name_map)

            # Adapt shape if needed
            for j, target_input in enumerate(target_inputs):
                target_graph_module, _ = adapt_node_shape(
                    target_graph_module,
                    node=target_input,
                    current_size=target_input.meta["tensor_meta"].shape,
                    target_size=node.args[j].meta["tensor_meta"].shape,
                    target_user=new_node
                )
        else:
            new_args = tuple(old_to_new[arg] if isinstance(arg, torch.fx.Node) else arg for arg in node.args)
            new_node = _insert_node(target_graph_module, after_node, node, new_args, module_name_map)

        if new_node:
            new_node_names.add(new_node.name)
        old_to_new[node] = new_node

    # 4. For each output boundary node, replace the input of the mapped target node
    for sub_out, users in output_mapping.items():
        print("sub_out", sub_out)
        for user in users:
            new_out_node = old_to_new[sub_out]
            # Replace the input of user with new_out_node
            # TODO should we do a random arg index here?
            tensor_meta = user.args[0].meta["tensor_meta"]
            try:
                first_arg_shape = tensor_meta.shape
            except:
                first_arg_shape = tensor_meta[0].shape  # Note: split nodes have a tuple of shapes. Maybe this is hacky?
            new_args = tuple([new_out_node, *user.args[1:]])  # TODO do we need to track arg indices? This is just replacing the first arg. But our current node compatibility check doesn't look at args at all, just tensor shapes... I'm confused.
            user.args = new_args

            target_graph_module, _ = adapt_node_shape(
                target_graph_module,
                node=new_out_node,
                current_size=sub_out.meta["tensor_meta"].shape,
                target_size=first_arg_shape,
                target_user=user
            )

    print("old_to_new", old_to_new)
    

    visualize_graph(target_graph_module, "model_graph2_highlighted99", "graph2_highlighted99.svg", highlight_nodes=new_node_names)
    target_graph_module.graph.lint()
    target_graph_module.recompile()
    return target_graph_module, new_node_names

def _insert_node(target_graph: torch.fx.GraphModule, after_node: torch.fx.Node, node: torch.fx.Node, new_args, module_name_map):
    print("inserting after", after_node)
    def _insert_call(func):
        target = module_name_map[node.target] if (node.op == "call_module" and node.target in module_name_map) else node.target
        with target_graph.graph.inserting_after(after_node):
            return func(target, args=new_args, kwargs=node.kwargs)
    if node.op == "call_module":
        return _insert_call(target_graph.graph.call_module)
    elif node.op == "call_function":
        return _insert_call(target_graph.graph.call_function)
    elif node.op == "call_method":
        return _insert_call(target_graph.graph.call_method)
    elif node.op == "get_attr":
        with target_graph.graph.inserting_after(after_node):
            return target_graph.graph.get_attr(node.target)
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

    # import copy
    # import time
    # a = time.time()
    # graph1_copy = copy.deepcopy(graph1)
    # graph2_copy = copy.deepcopy(graph2)
    # print("time taken to copy", time.time() - a)

    visualize_graph(graph1, "model_graph", "graph.svg")

    subgraph_nodes = set()
    # lowest_ratio_of_boundary_to_nodes = float('inf')
    lowest_num_boundary_nodes = float('inf')
    broken_subgraphs = 0
    import time
    for i in range(100):
        x = time.time()
        graph1_str = str(graph1.graph)
        graph2_str = str(graph2.graph)
        assert graph1_str == graph2_str
        try:
            num_nodes = random.randint(MIN_NODES, MAX_NODES)
            subgraph_nodes, input_boundary_nodes, output_boundary_nodes = random_subgraph(graph1, num_nodes)
            # ratio_of_boundary_to_nodes = len(boundary_nodes) / len(candidate_nodes)
            # if len(boundary_nodes) <= MAX_BOUNDARY_NODES and ratio_of_boundary_to_nodes < lowest_ratio_of_boundary_to_nodes:
            #     lowest_ratio_of_boundary_to_nodes = ratio_of_boundary_to_nodes
            num_boundary_nodes = len(input_boundary_nodes) + len(output_boundary_nodes)
            if num_boundary_nodes <= MAX_BOUNDARY_NODES and len(subgraph_nodes) > MIN_NODES and num_boundary_nodes < lowest_num_boundary_nodes:
                input_mapping, topo_target_input_nodes, output_mapping = find_subgraph_connections(graph2, input_boundary_nodes, output_boundary_nodes)
                lowest_num_boundary_nodes = num_boundary_nodes

                insert_subgraph_kwargs = {
                    "subgraph_nodes": subgraph_nodes,
                    "input_mapping": input_mapping,
                    "topo_target_input_nodes": topo_target_input_nodes,
                    "output_mapping": output_mapping
                }
            print("time taken", time.time() - x)
        except ValueError as e:
            print("WARNING: error finding subgraph", e)
            print("time taken for error", time.time() - x)
            broken_subgraphs += 1
    print("broken_subgraphs", broken_subgraphs)


# Extract node names for highlighting
subgraph_node_names = {node.name for node in insert_subgraph_kwargs["subgraph_nodes"]}

# Visualize the graph with the subgraph highlighted
visualize_graph(graph1, "model_graph_highlighted", "graph_highlighted.svg", highlight_nodes=subgraph_node_names)

graph2, new_node_names = insert_subgraph(graph2, **insert_subgraph_kwargs)

visualize_graph(graph2, "model_graph2_highlighted", "graph2_highlighted.svg", highlight_nodes=new_node_names)