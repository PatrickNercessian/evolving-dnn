import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from core import get_graph, add_node
from utils import adapt_node_shape, remove_node_flexible
from torch.fx.passes.shape_prop import ShapeProp
import inspect
from torch.fx import Node
import time


class Cascade:
    """
    Handles cascading dimension changes throughout a neural network.
    Ensures dimensional compatibility throughout the graph when nodes are
    added, modified, or removed.
    """

    def __init__(self, graph: torch.fx.GraphModule, use_reshape: bool = True):
        """
        Initialize with a graph.

        Args:
            graph: The FX graph module to work with
            use_reshape: If True, repair by reshaping nodes; else, use adapters.
        """
        self.graph = graph
        self.use_reshape = use_reshape
        self.graph_input_shape = None

        # For dependency conflict detection and adapter tracking
        self.node_changes = defaultdict(dict)  # node_id -> {dim_type: (old_dim, new_dim, change_id)}
        self.change_counter = 0
        self.adapters = {}  # (parent_id, child_id, direction) -> adapter_node

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
        self.graph_input_shape = input_shape
        self.node_changes.clear()
        self.adapters.clear()
        self.change_counter = 0

        self.graph = self._cascade_bfs(
            node=node,
            node_shape=node_shape,
            input_shape=input_shape,
            output_shape=output_shape
        )
        if input_shape is not None:
            try:
                ShapeProp(self.graph).propagate(torch.zeros(*input_shape))
            except Exception as e:
                print(f"Warning: Shape propagation failed: {e}")
        return self.graph

    def _cascade_bfs(self,
                     node: torch.fx.Node,
                     node_shape: tuple,
                     input_shape: Optional[tuple] = None,
                     output_shape: Optional[tuple] = None) -> torch.fx.GraphModule:
        """
        BFS cascade with dependency conflict resolution.
        Each frontier entry: (node, direction, dim_type, target_dim, source_change_id)
        direction: 'forward' or 'backward'
        dim_type: 'input' or 'output'
        """
        frontier = deque()
        visited = set()

        input_dim, output_dim = node_shape
        self.change_counter += 1
        initial_change_id = self.change_counter

        frontier.append((node, 'forward', 'output', output_dim, initial_change_id))
        frontier.append((node, 'backward', 'input', input_dim, initial_change_id))

        while frontier:
            current_node, direction, dim_type, target_dim, source_change_id = frontier.popleft()
            visit_key = (id(current_node), direction, dim_type, target_dim)
            if visit_key in visited:
                continue
            visited.add(visit_key)

            # Dependency conflict detection
            if self._has_dependency_conflict(current_node, dim_type, target_dim, source_change_id):
                self._handle_dependency_conflict(current_node, direction, dim_type, target_dim, frontier, source_change_id)
                continue

            current_input_dim, current_output_dim = self._get_node_shape(current_node)
            current_dim = current_input_dim if dim_type == 'input' else current_output_dim
            if current_dim == target_dim:
                continue

            # Get neighbors
            if direction == 'forward':
                neighbors = list(current_node.users)
            else:
                neighbors = [arg for arg in current_node.args if isinstance(arg, Node)]

            # Process the current node if it needs changing
            changed_node = self._process_node_change(current_node, dim_type, target_dim, source_change_id)

            # Add neighbors to frontier based on the change made
            self._add_neighbors_to_frontier(changed_node, direction, neighbors, frontier, source_change_id)

        self.graph.recompile()
        return self.graph

    def _has_dependency_conflict(self, node: Node, dim_type: str, target_dim: int, source_change_id: int) -> bool:
        node_id = id(node)
        if node_id in self.node_changes and dim_type in self.node_changes[node_id]:
            prev_change = self.node_changes[node_id][dim_type]
            prev_new_dim, prev_change_id = prev_change[1], prev_change[2]
            return (target_dim != prev_new_dim and source_change_id != prev_change_id)
        return False

    def _handle_dependency_conflict(self, node: Node, direction: str, dim_type: str, target_dim: int, 
                                   frontier: deque, source_change_id: int):
        current_input_dim, current_output_dim = self._get_node_shape(node)
        current_dim = current_input_dim if dim_type == 'input' else current_output_dim

        if direction == 'forward':
            for child in list(node.users):
                if not is_shapeless_module(getattr(self.graph, child.target, None) if child.op == 'call_module' else None):
                    adapter_key = (id(node), id(child), 'forward')
                    if adapter_key not in self.adapters:
                        adapter = self._create_adapter(current_output_dim, target_dim)
                        self.adapters[adapter_key] = adapter
                        self._insert_adapter_between_nodes(node, child, adapter)
        else:
            for parent in [arg for arg in node.args if isinstance(arg, Node)]:
                if not is_shapeless_module(getattr(self.graph, parent.target, None) if parent.op == 'call_module' else None):
                    adapter_key = (id(parent), id(node), 'backward')
                    if adapter_key not in self.adapters:
                        parent_input_dim, parent_output_dim = self._get_node_shape(parent)
                        adapter = self._create_adapter(parent_output_dim, target_dim)
                        self.adapters[adapter_key] = adapter
                        self._insert_adapter_between_nodes(parent, node, adapter)

    def _create_adapter(self, input_dim: int, output_dim: int) -> str:
        # For 1D, use AdaptiveAvgPool1d or padding as a simple example
        if input_dim > output_dim:
            adapter_module = nn.AdaptiveAvgPool1d(output_dim)
        else:
            # Use a simple linear up-projection if needed
            adapter_module = nn.Linear(input_dim, output_dim)
        adapter_name = f"adapter_{self.change_counter}"
        self.change_counter += 1
        self.graph.add_module(adapter_name, adapter_module)
        return adapter_name

    def _insert_adapter_between_nodes(self, parent: Node, child: Node, adapter_module_name: str):
        with self.graph.graph.inserting_after(parent):
            adapter_node = self.graph.graph.call_module(adapter_module_name, args=(parent,))
        self.replace_parent_of_child(child, parent, adapter_node)

    def _process_node_change(self, node: Node, dim_type: str, target_dim: int, source_change_id: int) -> Node:
        current_input_dim, current_output_dim = self._get_node_shape(node)
        current_dim = current_input_dim if dim_type == 'input' else current_output_dim
        if current_dim == target_dim:
            return node
        node_id = id(node)
        self.node_changes[node_id][dim_type] = (current_dim, target_dim, source_change_id)
        if self.use_reshape and node.op == 'call_module':
            module = getattr(self.graph, node.target, None)
            if isinstance(module, nn.Linear):
                if dim_type == 'input':
                    new_node = self._replace_linear_node(node, new_in=target_dim)
                else:
                    new_node = self._replace_linear_node(node, new_out=target_dim)
                del self.node_changes[node_id]
                self.node_changes[id(new_node)][dim_type] = (current_dim, target_dim, source_change_id)
                return new_node
        return node

    def _add_neighbors_to_frontier(self, node: Node, direction: str, neighbors: List[Node], 
                                  frontier: deque, source_change_id: int):
        """
        Only add neighbors to the frontier if there is a dimension mismatch that needs fixing.
        Avoids unnecessary cascade and prevents conflicts.
        """
        node_input_dim, node_output_dim = self._get_node_shape(node)
        for neighbor in neighbors:
            neighbor_module = getattr(self.graph, neighbor.target, None) if neighbor.op == 'call_module' else None
            if is_shapeless_module(neighbor_module):
                # Shapeless modules preserve dimensions, continue cascade
                if direction == 'forward':
                    # Only add if neighbor's output doesn't match our output
                    neighbor_input_dim, neighbor_output_dim = self._get_node_shape(neighbor)
                    if neighbor_output_dim != node_output_dim:
                        frontier.append((neighbor, 'forward', 'output', node_output_dim, source_change_id))
                else:
                    neighbor_input_dim, neighbor_output_dim = self._get_node_shape(neighbor)
                    if neighbor_input_dim != node_input_dim:
                        frontier.append((neighbor, 'backward', 'input', node_input_dim, source_change_id))
            else:
                neighbor_input_dim, neighbor_output_dim = self._get_node_shape(neighbor)
                if direction == 'forward':
                    # Only add if neighbor's input doesn't match our output
                    if neighbor_input_dim != node_output_dim:
                        frontier.append((neighbor, 'backward', 'input', node_output_dim, source_change_id))
                else:
                    # Only add if neighbor's output doesn't match our input
                    if neighbor_output_dim != node_input_dim:
                        frontier.append((neighbor, 'forward', 'output', node_input_dim, source_change_id))

    def _get_node_shape(self, node):
        input_dim = None
        output_dim = None
        if node.op == 'call_module':
            try:
                module = getattr(self.graph, node.target, None)
                if module is not None:
                    if isinstance(module, nn.Linear):
                        input_dim = module.in_features
                        output_dim = module.out_features
                    elif isinstance(module, nn.Conv2d):
                        input_dim = module.in_channels
                        output_dim = module.out_channels
                    elif is_shapeless_module(module):
                        if node.args and isinstance(node.args[0], torch.fx.Node):
                            parent_shape = self._get_node_meta_shape(node.args[0])
                            if parent_shape and len(parent_shape) > 0:
                                input_dim = output_dim = parent_shape[-1]
            except AttributeError as e:
                print(f"Debug: AttributeError getting module for {node.name}: {e}")
        if input_dim is None and node.args and isinstance(node.args[0], torch.fx.Node):
            parent_shape = self._get_node_meta_shape(node.args[0])
            if parent_shape and len(parent_shape) > 0:
                input_dim = parent_shape[-1]
        if output_dim is None:
            node_shape = self._get_node_meta_shape(node)
            if node_shape and len(node_shape) > 0:
                output_dim = node_shape[-1]
        if (input_dim is None or output_dim is None) and node.op not in ['placeholder', 'output', 'get_attr']:
            print(f"Warning: Could not determine full shape for node {node.name}")
            if input_dim is None:
                input_dim = output_dim or 128
            if output_dim is None:
                output_dim = input_dim or 128
        return (input_dim, output_dim)

    def _get_node_meta_shape(self, node: Node) -> Optional[tuple]:
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            return node.meta['tensor_meta'].shape
        return None

    def replace_parent_of_child(self, child, old_parent, new_parent):
        new_args = tuple(new_parent if a is old_parent else a for a in child.args)
        child.args = new_args

    def replace_child_of_parent(self, parent, old_child, new_child):
        for user in list(parent.users):
            new_args = tuple(new_child if a is old_child else a for a in user.args)
            user.args = new_args

    def _replace_linear_node(self,
                         old_node: torch.fx.Node,
                         new_in: int = None,
                         new_out: int = None) -> torch.fx.Node:
        example = torch.zeros(*self.graph_input_shape) if self.graph_input_shape else None
        self.graph, new_node = reshape_node(
            self.graph, old_node,
            new_in_features=new_in,
            new_out_features=new_out,
            example_input=example
        )
        return new_node


def is_shapeless_module(mod):
    """
    Dynamically determines if a module is 'shapeless' (does not require shape args).
    Returns True if the module's __init__ does NOT have any typical shape arguments.
    """
    if mod is None:
        return True  # Defensive: treat unknown as shapeless

    # Common shape-related argument names
    shape_args = {'in_features', 'out_features', 'in_channels', 'out_channels', 'num_features', 'num_channels', 'features'}
    try:
        sig = inspect.signature(mod.__init__)
        param_names = set(sig.parameters.keys())
        param_names.discard('self')
        return len(shape_args & param_names) == 0
    except Exception:
        return True


def reshape_node(
    graph: torch.fx.GraphModule, 
    target_node: torch.fx.Node, 
    new_in_features: Optional[int] = None, 
    new_out_features: Optional[int] = None,
    example_input: Optional[torch.Tensor] = None
) -> Tuple[torch.fx.GraphModule, torch.fx.Node]:
    """
    Atomically replaces a node with a new node of the same type but new dimensions.
    Only replaces the node itself; does not adapt parents (let cascade handle that).
    """
    # 1. Clone the module with new dimensions
    old_mod = getattr(graph, target_node.target)
    if isinstance(old_mod, nn.Linear):
        in_f = new_in_features if new_in_features is not None else old_mod.in_features
        out_f = new_out_features if new_out_features is not None else old_mod.out_features
        new_mod = nn.Linear(in_f, out_f)
    else:
        raise NotImplementedError("reshape_node only supports nn.Linear for now.")
    
    # 2. Register the new module with unique name
    import time
    new_mod_name = f"{target_node.target}_reshaped_{int(time.time() * 1000000) % 1000000}"
    graph.add_module(new_mod_name, new_mod)
    
    # 3. Insert new node in the graph (use same args as old node)
    with graph.graph.inserting_after(target_node):
        new_node = graph.graph.call_module(new_mod_name, args=target_node.args, kwargs=target_node.kwargs)
    
    # 4. Redirect all users of the old node to the new node
    for user in list(target_node.users):
        user.replace_input_with(target_node, new_node)
    
    # 5. Remove the old node from the graph
    remove_node_flexible(graph, target_node)
    
    # 6. Recompile the graph
    graph.recompile()
    
    return graph, new_node
