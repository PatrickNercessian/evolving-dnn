import copy
from typing import Dict, List, Set, Tuple, Optional, Any, Union


class Cascade:
    """
    Handles cascading dimension changes throughout a neural network.
    Ensures that when one layer's dimensions change, all connected layers
    are appropriately adjusted to maintain compatibility.
    """
    
    def __init__(self, network):
        """
        Initialize with a network representation.
        
        Args:
            network: The neural network object containing layers and connections
        """
        self.network = network
        
    def adjust_layer_dimensions(self, layer_id: str, new_input_dim: Optional[int] = None, 
                               new_output_dim: Optional[int] = None) -> None:
        """
        Adjust dimensions of a specific layer and propagate changes.
        
        Args:
            layer_id: Identifier for the layer to modify
            new_input_dim: New input dimension (if changing)
            new_output_dim: New output dimension (if changing)
        """
        # Get current layer
        layer = self.network.get_layer(layer_id)
        
        # Create new layer with adjusted dimensions
        new_layer = copy.deepcopy(layer)
        if new_input_dim is not None:
            new_layer.input_dim = new_input_dim
        if new_output_dim is not None:
            new_layer.output_dim = new_output_dim
            
        # Replace the layer
        self.replace_layer(layer_id, new_layer)
        
        # Propagate changes to connected layers
        self.propagate_changes(layer_id)
    
    def propagate_changes(self, start_layer_id: str, visited: Optional[Set[str]] = None) -> None:
        """
        Recursively adjust connected layers starting from the changed layer.
        
        Args:
            start_layer_id: Layer ID where changes originated
            visited: Set of already visited layers to prevent cycles
        """
        if visited is None:
            visited = set()
        
        if start_layer_id in visited:
            return
        
        visited.add(start_layer_id)
        start_layer = self.network.get_layer(start_layer_id)
        
        # Update children (forward propagation)
        children = self.get_layer_children(start_layer_id)
        for child_id in children:
            child = self.network.get_layer(child_id)
            if child.input_dim != start_layer.output_dim:
                # Create new child with adjusted input dimension
                new_child = copy.deepcopy(child)
                new_child.input_dim = start_layer.output_dim
                self.replace_layer(child_id, new_child)
            
            # Continue propagation
            self.propagate_changes(child_id, visited)
            
        # Update parents (backward propagation)
        parents = self.get_layer_parents(start_layer_id)
        for parent_id in parents:
            parent = self.network.get_layer(parent_id)
            if parent.output_dim != start_layer.input_dim:
                # Create new parent with adjusted output dimension
                new_parent = copy.deepcopy(parent)
                new_parent.output_dim = start_layer.input_dim
                self.replace_layer(parent_id, new_parent)
            
            # Continue propagation
            self.propagate_changes(parent_id, visited)
    
    def get_layer_parents(self, layer_id: str) -> List[str]:
        """
        Return layers that feed into this layer.
        
        Args:
            layer_id: Layer to find parents for
            
        Returns:
            List of parent layer IDs
        """
        # Placeholder - actual implementation depends on network representation
        return self.network.get_parent_layers(layer_id)
    
    def get_layer_children(self, layer_id: str) -> List[str]:
        """
        Return layers that this layer feeds into.
        
        Args:
            layer_id: Layer to find children for
            
        Returns:
            List of child layer IDs
        """
        # Placeholder - actual implementation depends on network representation
        return self.network.get_child_layers(layer_id)
    
    def replace_layer(self, layer_id: str, new_layer: Any) -> None:
        """
        Replace a layer with a new one, maintaining connections.
        
        Args:
            layer_id: ID of layer to replace
            new_layer: New layer object to insert
        """
        # Placeholder - would call network methods to replace the layer
        self.network.replace_layer(layer_id, new_layer)
    
    def adjust_network_for_insertion(self, parent_id: str, child_id: str, 
                                    new_layer: Any) -> None:
        """
        Insert a new layer between parent and child, adjusting dimensions.
        
        Args:
            parent_id: ID of parent layer
            child_id: ID of child layer
            new_layer: New layer to insert
        """
        parent = self.network.get_layer(parent_id)
        child = self.network.get_layer(child_id)
        
        # Adjust dimensions of new layer to fit between parent and child
        new_layer.input_dim = parent.output_dim
        new_layer.output_dim = child.input_dim
        
        # Insert the new layer
        new_layer_id = self.network.insert_layer(parent_id, child_id, new_layer)
        
        # No need to propagate changes as dimensions are already matched
        return new_layer_id
    
    def adjust_network_for_removal(self, layer_id: str) -> None:
        """
        Remove a layer and reconnect its parents to its children.
        
        Args:
            layer_id: ID of layer to remove
        """
        parents = self.get_layer_parents(layer_id)
        children = self.get_layer_children(layer_id)
        
        # Remove the layer
        self.network.remove_layer(layer_id)
        
        # Connect parents to children directly
        for parent_id in parents:
            for child_id in children:
                parent = self.network.get_layer(parent_id)
                child = self.network.get_layer(child_id)
                
                # Check if dimensions mismatch
                if parent.output_dim != child.input_dim:
                    # Adjust child's input to match parent's output
                    new_child = copy.deepcopy(child)
                    new_child.input_dim = parent.output_dim
                    self.replace_layer(child_id, new_child)
                    
                # Connect parent to child
                self.network.connect_layers(parent_id, child_id)
                
                # Propagate changes
                self.propagate_changes(child_id)