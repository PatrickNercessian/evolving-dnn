import pydot
from torch.fx.passes.graph_drawer import FxGraphDrawer

def visualize_graph(graph, name="model_graph", output_path=None, highlight_nodes=None):
    """
    Visualizes an FX graph using the FxGraphDrawer, optionally highlighting a subgraph.
    
    Args:
        graph: The FX graph to visualize
        name: Name for the graph visualization (default: "model_graph")
        output_path: Optional path to save the visualization (e.g., "graph.svg")
        highlight_nodes: An optional set of node names (strings) to highlight.
                         These nodes will have their style modified.
    Returns:
        The pydot graph object that can be further customized
    """
    # Create graph drawer
    drawer = FxGraphDrawer(graph, name)
    
    # Get the dot graph
    dot_graph = drawer.get_dot_graph()

    # Highlight nodes if specified
    if highlight_nodes:
        # Iterate through nodes in the graph and modify style if highlighted
        for node in dot_graph.get_node_list():
            # pydot node names might be quoted, strip quotes for comparison
            node_name = node.get_name().strip('"') 
            if node_name in highlight_nodes:
                node.set_style("filled")
                node.set_fillcolor("lightblue") # Example highlight color
                node.set_color("blue")
                node.set_fontcolor("blue")

        # The following code block for creating a cluster is removed
        # subgraph_cluster = pydot.Cluster(
        #     "highlighted_subgraph",  # Name of the cluster
        #     label="Highlighted Subgraph", 
        #     style="dashed",
        #     color="blue",
        #     fontcolor="blue",
        #     fontsize="16",
        #     labeljust="l" # Adjust label position if needed (l=left, r=right, c=center)
        # )
        
        # nodes_to_move = []
        # original_graph_nodes = dot_graph.get_node_list() # Get a copy to iterate while modifying
        # for node in original_graph_nodes:
        #     node_name = node.get_name().strip('"') # pydot might add quotes
        #     if node_name in highlight_nodes:
        #         # Remove node from main graph and add to cluster
        #         dot_graph.del_node(node_name) 
        #         subgraph_cluster.add_node(node)
                
        # dot_graph.add_subgraph(subgraph_cluster)
    
    # Save to file if output path is provided
    if output_path:
        if output_path.endswith('.svg'):
            dot_graph.write_svg(output_path)
        elif output_path.endswith('.png'):
            dot_graph.write_png(output_path)
        elif output_path.endswith('.pdf'):
            dot_graph.write_pdf(output_path)
        else:
            dot_graph.write(output_path)
    
    return dot_graph