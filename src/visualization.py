from torch.fx.passes.graph_drawer import FxGraphDrawer

def visualize_graph(graph, name="model_graph", output_path=None):
    """
    Visualizes an FX graph using the FxGraphDrawer
    
    Args:
        graph: The FX graph to visualize
        name: Name for the graph visualization (default: "model_graph")
        output_path: Optional path to save the visualization (e.g., "graph.svg")
    Returns:
        The pydot graph object that can be further customized
    """
    # Create graph drawer
    drawer = FxGraphDrawer(graph, name)
    
    # Get the dot graph
    dot_graph = drawer.get_dot_graph()
    
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