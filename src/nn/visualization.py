from torch.fx.passes.graph_drawer import FxGraphDrawer

import torch
import os
import logging

def visualize_graph(graph: torch.fx.GraphModule, name="model_graph", output_path=None, highlight_nodes=None):
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

def visualize_text_graph(graph_module):
    """
    Creates a text-based flowchart visualization of the graph module
    using box drawing characters for better clarity.
    
    Args:
        graph_module: The FX GraphModule to visualize
    
    Returns:
        str: A string representation of the graph visualization
    """
    output_lines = []
    output_lines.append("\nNeural Network Graph Visualization:")
    output_lines.append("═══════════════════════════════════")
    
    # Get nodes in topological order
    nodes = list(graph_module.graph.nodes)
    
    # Create a dictionary mapping nodes to their children
    node_to_children = {node: [] for node in nodes}
    # Create a dictionary mapping nodes to all their direct inputs
    node_to_inputs = {node: set(node.all_input_nodes) for node in nodes}
    
    # Build children relationships
    for node in nodes:
        for input_node in node.all_input_nodes:
            if input_node in node_to_children:
                node_to_children[input_node].append(node)
    
    # Find roots (nodes without inputs or only placeholder inputs)
    roots = [node for node in nodes if node.op == 'placeholder']
    
    # Track visited nodes and their paths
    visited = set()
    path_set = set()  # Tracks current path for branch detection
    
    # Find nodes with multiple inputs (merge points)
    merge_points = {node for node in nodes if len(node_to_inputs[node]) > 1}
    
    def get_node_info(node):
        """Get formatted node type and shape info"""
        # Get node type description
        if node.op == 'placeholder':
            type_str = "INPUT"
        elif node.op == 'output': 
            type_str = "OUTPUT"
        elif node.op == 'call_module':
            try:
                # Handle nested attribute access (e.g., 'transformer.wte')
                module = graph_module
                for attr in str(node.target).split('.'):
                    module = getattr(module, attr)
                type_str = type(module).__name__
            except AttributeError:
                type_str = f"Module({node.target})"
        elif node.op == 'call_function':
            type_str = node.target.__name__
        else:
            type_str = node.op
        
        # Get shape info
        shape_str = "unknown"
        if 'meta' in node.__dict__ and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            if hasattr(tensor_meta, 'shape'):
                # Standard case: tensor_meta has a shape attribute
                shape_str = f"{tensor_meta.shape}"
            elif isinstance(tensor_meta, (tuple, list)):
                # Case where tensor_meta is directly the shape
                shape_str = f"{tensor_meta}"
            elif hasattr(tensor_meta, '__iter__') and not isinstance(tensor_meta, str):
                # Case where tensor_meta is some other iterable
                try:
                    shape_str = f"{tuple(tensor_meta)}"
                except:
                    shape_str = f"{tensor_meta}"
            else:
                # Fallback: just convert to string
                shape_str = f"{tensor_meta}"
        
        return type_str, shape_str
    
    def print_node(node, depth=0, is_last=False, prefix="", branch_name=None):
        """
        Recursively print node and its children in a tree-like format
        branch_name: used to identify which branch we're in for clearer branching visualization
        """
        if node in path_set:  # Avoid cycles in the current path
            return
        
        type_str, shape_str = get_node_info(node)
        
        # Build the connection line prefix
        if depth > 0:
            if is_last:
                line = prefix + "└── "
                new_prefix = prefix + "    "
            else:
                line = prefix + "├── "
                new_prefix = prefix + "│   "
        else:
            line = ""
            new_prefix = ""
        
        # Format branch information if this is part of a branch
        branch_info = f" <{branch_name}>" if branch_name else ""
        
        # Format merge point information
        merge_info = " [MERGE POINT]" if node in merge_points and node in visited else ""
        
        # Only show full details first time we encounter node
        if node not in visited:
            output_lines.append(f"{line}{node.name}{branch_info} ({type_str}, shape={shape_str}){merge_info}")
            visited.add(node)
            
            # Add node to current path
            path_set.add(node)
            
            # Get all children
            children = node_to_children[node]
            
            # Handle branches: if there are multiple children, label each branch
            if len(children) > 1:
                output_lines.append(f"{new_prefix}│")
                output_lines.append(f"{new_prefix}┌─ BRANCH POINT ─┐")
                
                # Process each branch with a label
                for i, child in enumerate(children):
                    branch_label = f"Branch {i+1}"
                    is_final = (i == len(children) - 1)
                    
                    # Create branch-specific prefix
                    if is_final:
                        branch_prefix = new_prefix + "└── "
                        next_prefix = new_prefix + "    "
                    else:
                        branch_prefix = new_prefix + "├── "
                        next_prefix = new_prefix + "│   "
                    
                    output_lines.append(f"{branch_prefix}{branch_label}:")
                    print_node(child, depth + 2, is_final, next_prefix, branch_label)
            elif children:
                # Single child - no branching
                print_node(children[0], depth + 1, True, new_prefix)
            
            # Remove node from current path when backtracking
            path_set.remove(node)
        else:
            # For already visited nodes, just print reference
            output_lines.append(f"{line}{node.name}{branch_info}{merge_info} [↑ see above]")
    
    # Print from each root node
    for i, root in enumerate(roots):
        print_node(root)
        if i < len(roots) - 1:
            output_lines.append("")  # Add spacing between different roots
    
    output_lines.append("═══════════════════════════════════")
    return "\n".join(output_lines)

def log_best_individual(evolution, experiment_path, overwrite_logs=False):
    """
    Log detailed information about the best individual to a separate file
    
    Args:
        evolution: The evolution instance containing the best individual
        experiment_path: Path to the experiment directory
        overwrite_logs: Whether to overwrite existing log files (default: False)
    """
    if not evolution.best_individual:
        logging.warning("No best individual found to log")
        return
        
    best_individual_log_path = os.path.join(experiment_path, "best_individual.log")
    
    # Determine file mode based on overwrite_logs setting
    file_mode = 'w' if overwrite_logs else 'a'
    
    with open(best_individual_log_path, file_mode, encoding='utf-8') as f:
        # Add separator if appending to existing file
        if not overwrite_logs and os.path.exists(best_individual_log_path) and os.path.getsize(best_individual_log_path) > 0:
            f.write("\n" + "="*80 + "\n")
            f.write("NEW RUN SEPARATOR\n")
            f.write("="*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST INDIVIDUAL DETAILED LOG\n")
        f.write("="*80 + "\n\n")
        
        # Basic information
        f.write(f"Individual ID: {evolution.best_individual.id}\n")
        f.write(f"Fitness Score: {evolution.best_individual.fitness}\n")
        f.write(f"Generation: {getattr(evolution.best_individual, 'generation', 'Unknown')}\n\n")
        
        # Detailed node information
        f.write("ALL GRAPH NODES WITH DETAILS:\n")
        f.write("-" * 40 + "\n")
        
        for i, node in enumerate(evolution.best_individual.graph_module.graph.nodes):
            f.write(f"\nNode {i+1}: {node.name}\n")
            f.write(f"  Operation: {node.op}\n")
            f.write(f"  Target: {node.target}\n")
            
            # Args information
            if hasattr(node, 'args') and node.args:
                f.write(f"  Args: {node.args}\n")
            
            # Kwargs information  
            if hasattr(node, 'kwargs') and node.kwargs:
                f.write(f"  Kwargs: {node.kwargs}\n")
            
            # Meta information including tensor shapes
            if hasattr(node, 'meta') and node.meta:
                f.write(f"  Meta data:\n")
                for meta_key, meta_value in node.meta.items():
                    if meta_key == 'tensor_meta':
                        # Handle different types of tensor_meta
                        if hasattr(meta_value, 'shape'):
                            f.write(f"    {meta_key}.shape: {meta_value.shape}\n")
                            if hasattr(meta_value, 'dtype'):
                                f.write(f"    {meta_key}.dtype: {meta_value.dtype}\n")
                        elif isinstance(meta_value, (tuple, list)):
                            f.write(f"    {meta_key} (shape): {meta_value}\n")
                        else:
                            f.write(f"    {meta_key}: {meta_value}\n")
                    else:
                        f.write(f"    {meta_key}: {meta_value}\n")
            
            # Module information for call_module nodes
            if node.op == 'call_module':
                try:
                    # Handle nested attribute access (e.g., 'transformer.wte')
                    module = evolution.best_individual.graph_module
                    for attr in str(node.target).split('.'):
                        module = getattr(module, attr)
                    f.write(f"  Module Type: {type(module).__name__}\n")
                    f.write(f"  Module Details: {module}\n")
                except AttributeError:
                    f.write(f"  Module: Could not retrieve module details for {node.target}\n")
            
            # Input/output connections
            input_nodes = list(node.all_input_nodes)
            if input_nodes:
                f.write(f"  Input Nodes: {[n.name for n in input_nodes]}\n")
            
            users = list(node.users.keys())
            if users:
                f.write(f"  Output To: {[n.name for n in users]}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF BEST INDIVIDUAL LOG\n")
        f.write("="*80 + "\n")
    
    action = "overwritten" if overwrite_logs else "appended to"
    logging.info(f"Best individual details {action}: {best_individual_log_path}")