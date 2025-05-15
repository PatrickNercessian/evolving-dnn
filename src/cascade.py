import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any, Union


class Cascade:
    """
    Handles cascading dimension changes throughout a neural network.
    Ensures dimensional compatibility throughout the graph when nodes are
    added, modified, or removed.
    """
    
    def __init__(self, graph: torch.fx.GraphModule):
        """
        Initialize with a graph.
        
        Args:
            graph: The FX graph module to work with
        """
        self.graph = graph
    
    def adapt_dimensions(self, 
                        node: torch.fx.Node, 
                        node_shape: tuple,
                        input_shape: Optional[tuple] = None,
                        output_shape: Optional[tuple] = None) -> torch.fx.GraphModule:
        """
        Adapt dimensions throughout the graph starting from a specific node.
        
        Args:
            node: The node where changes originated
            node_shape: The shape of the node (input_dim, output_dim)
            input_shape: The shape of the input node (pre-computed)
            output_shape: The shape of the output node (pre-computed)
            
        Returns:
            The modified graph
        """
        visited = set()
        self.graph = self._cascade_dimension_changes(
            node=node,
            node_shape=node_shape,
            input_shape=input_shape, 
            output_shape=output_shape,
            visited=visited
        )
        return self.graph
    
    def _cascade_dimension_changes(self,
                                 node: torch.fx.Node,
                                 node_shape: tuple,
                                 input_shape: Optional[tuple] = None,
                                 output_shape: Optional[tuple] = None,
                                 visited: Optional[Set[torch.fx.Node]] = None) -> torch.fx.GraphModule:
        """
        Internal recursive method that adjusts dimensions throughout the network.
        
        Args:
            node: The node where changes originated
            node_shape: The shape of the node (input_dim, output_dim)
            input_shape: The shape of the input node (pre-computed)
            output_shape: The shape of the output node (pre-computed)
            visited: Set of visited nodes to prevent cycles
            
        Returns:
            The modified graph
        """
        # Return if node has been visited to prevent cycles
        if node in visited:
            return self.graph
            
        # Add current node to visited
        visited.add(node)
        
        # Forward cascade (check children)
        self.graph = self._cascade_forward(node, node_shape, visited)
        
        # Backward cascade (check parents)
        self.graph = self._cascade_backward(node, node_shape, visited)
        
        # Recompile graph periodically to keep shape information up to date
        # Only do this at the root level of recursion to avoid excessive recompilations
        if len(visited) <= 1:
            self.graph.recompile()
        
        return self.graph
    
    def _cascade_forward(self, node, node_shape, visited):
        """Recursively check and adjust dimensions for children nodes"""
        from utils import add_specific_node
        
        # Get all children (users) of the current node
        children = list(node.users)
        
        # Process each child node
        for child in children:
            if child in visited:
                continue
            
            # Check if we need to adjust child's input to match parent's output
            child_input_dim = self._get_input_dim_from_parent(child, node)
            if child_input_dim is not None and child_input_dim != node_shape[1]:
                print(f"Child input {child_input_dim} doesn't match parent output {node_shape[1]}")
                
                # Add appropriate adapter
                if node_shape[1] > child_input_dim:
                    # Need to reduce dimension with pooling
                    self.graph, adapter = add_specific_node(self.graph, node, nn.AdaptiveAvgPool1d(child_input_dim))
                    self._replace_parent_in_child(child, node, adapter)
                else:
                    # Need to expand dimension with padding
                    self.graph, adapter = add_specific_node(self.graph, node, 
                                                      nn.CircularPad1d((0, child_input_dim - node_shape[1])))
                    self._replace_parent_in_child(child, node, adapter)
                
                # Get the updated shape of the child 
                child_shape = self._get_node_shape(child)
                
                # Continue cascade from this child
                self.graph = self._cascade_dimension_changes(child, child_shape, visited=visited)
        
        return self.graph
    
    def _cascade_backward(self, node, node_shape, visited):
        """Recursively check and adjust dimensions for parent nodes"""
        from utils import add_specific_node
        
        # Get all parents (args that are nodes) of the current node
        if not hasattr(node, 'args'):
            return self.graph
            
        parents = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
        
        # Process each parent node independently
        for parent in parents:
            if parent in visited:
                continue
            
            # Check if parent's output matches this node's input requirements
            parent_output_dim = self._get_output_dim(parent)
            if parent_output_dim is not None and parent_output_dim != node_shape[0]:
                print(f"Parent output {parent_output_dim} doesn't match child input {node_shape[0]}")
                
                # Add appropriate adapter
                if parent_output_dim > node_shape[0]:
                    # Need to reduce dimension with pooling
                    self.graph, adapter = add_specific_node(self.graph, parent, nn.AdaptiveAvgPool1d(node_shape[0]))
                    self._replace_parent_in_child(node, parent, adapter)
                else:
                    # Need to expand dimension with padding
                    self.graph, adapter = add_specific_node(self.graph, parent,
                                                     nn.CircularPad1d((0, node_shape[0] - parent_output_dim)))
                    self._replace_parent_in_child(node, parent, adapter)
                
                # Get the updated shape of the parent
                parent_shape = self._get_node_shape(parent)
                
                # Continue cascade from this parent
                self.graph = self._cascade_dimension_changes(parent, parent_shape, visited=visited)
        
        return self.graph

    # Helper methods for dimension analysis and node manipulation
    
    def _get_node_shape(self, node):
        """Get (input_dim, output_dim) tuple for a node"""
        input_dim = None
        if node.args and isinstance(node.args[0], torch.fx.Node):
            if hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
                shape = node.args[0].meta['tensor_meta'].shape
                if shape and len(shape) > 0:
                    input_dim = shape[-1]
    
        output_dim = None
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            shape = node.meta['tensor_meta'].shape
            if shape and len(shape) > 0:
                output_dim = shape[-1]
            
        if input_dim is None or output_dim is None:
            print(f"Warning: Could not determine full shape for node {node.name}")
            
        return (input_dim, output_dim)
    
    def _get_input_dim_from_parent(self, child, parent):
        """Get the input dimension that child receives from parent"""
        if hasattr(child, 'meta') and 'tensor_meta' in child.meta:
            for i, arg in enumerate(child.args):
                if arg == parent:
                    # Found the connection - for most ops, we need last dim
                    return child.meta['tensor_meta'].shape[-1]
        return None
    
    def _get_output_dim(self, node):
        """Get output dimension for a node"""
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            return node.meta['tensor_meta'].shape[-1]
        return None
    
    def _replace_parent_in_child(self, child, old_parent, new_parent):
        """Replace a parent node with a new one in a child node's args"""
        for i, arg in enumerate(child.args):
            if arg == old_parent:
                new_args = list(child.args)
                new_args[i] = new_parent
                child.args = tuple(new_args)
                break
