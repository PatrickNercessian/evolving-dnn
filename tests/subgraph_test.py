import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nn.core import get_graph
from src.nn.visualization import visualize_graph
from src.nn.variation.architecture_crossover import random_subgraph, find_subgraph_connections, insert_subgraph

from src.mingpt_altered.model import GPT
from src.mingpt_altered.utils import CfgNode as CN

import torch
import torch.fx
MAX_BOUNDARY_NODES = 10
MIN_NODES = 4
MAX_NODES = 32

if __name__ == "__main__":
    config = CN()
    config.vocab_size = 50257
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 768
    config.block_size = 1024
    config.embd_pdrop = 0.1
    config.attn_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.is_proxy_for_fx = True

    model1 = GPT(config)
    example_input = torch.randint(0, config.vocab_size, (1, config.block_size))
    print("example_input", example_input)
    graph1 = get_graph(model1, example_input=example_input)
    model2 = GPT(config)
    graph2 = get_graph(model2, example_input=example_input)

    visualize_graph(graph1, "model_graph", "graph.svg")

    subgraph_nodes = set()
    # lowest_ratio_of_boundary_to_nodes = float('inf')
    lowest_num_boundary_nodes = float('inf')
    broken_subgraphs = 0
    for i in range(20):
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
                input_mapping, topo_target_input_nodes, output_mapping = find_subgraph_connections(graph2.graph, input_boundary_nodes, output_boundary_nodes)
                lowest_num_boundary_nodes = num_boundary_nodes

                insert_subgraph_kwargs = {
                    "subgraph_nodes": subgraph_nodes,
                    "input_mapping": input_mapping,
                    "topo_target_input_nodes": topo_target_input_nodes,
                    "output_mapping": output_mapping
                }
        except ValueError as e:
            print("WARNING: error finding subgraph", e)
            broken_subgraphs += 1
    print("broken_subgraphs", broken_subgraphs)


# Extract node names for highlighting
subgraph_node_names = {node.name for node in insert_subgraph_kwargs["subgraph_nodes"]}

# Visualize the graph with the subgraph highlighted
visualize_graph(graph1, "model_graph_highlighted", "graph_highlighted.svg", highlight_nodes=subgraph_node_names)

graph2, new_node_names = insert_subgraph(graph2, **insert_subgraph_kwargs)

visualize_graph(graph2, "model_graph2_highlighted", "graph2_highlighted.svg", highlight_nodes=new_node_names)