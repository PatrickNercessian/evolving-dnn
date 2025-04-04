import torch.nn as nn
import torch.fx
from utils import get_unique_name

def add(graph: torch.fx.GraphModule, reference_node: torch.fx.Node, module: nn.Module):
    """
    Adds a module to the graph after the reference node.
    """
    name = get_unique_name(graph, module.__class__.__name__)
    graph.add_submodule(name, module)
    
    # Add repeat node after reference_node
    with graph.graph.inserting_after(reference_node):
        new_node = graph.graph.call_module(
            module_name=name,
            args=(reference_node,),
            kwargs={},
        )
    
    # Update connections
    reference_node.replace_all_uses_with(new_node)
    new_node.args = (reference_node,)
    
    return graph, new_node
