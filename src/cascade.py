import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from utils import adapt_node_shape, remove_node_flexible
from torch.fx.passes.shape_prop import ShapeProp
import inspect
from torch.fx import Node
import time


class Cascade:
    """
    Handles cascading dimension changes throughout a neural network.
    Includes sophisticated cycle detection and resolution strategies.
    """

    def __init__(self, graph: torch.fx.GraphModule, use_reshape: bool = True, 
                 cycle_resolution: str = 'adapter'):
        """
        Args:
            graph: The FX graph module
            use_reshape: If True, reshape nodes; else use adapters
            cycle_resolution: How to handle cycles - 'adapter' or 'second_order'
        """
        self.graph = graph
        self.use_reshape = use_reshape
        self.cycle_resolution = cycle_resolution
        self.graph_input_shape = None
        
        # Tracking structures
        self.node_changes = defaultdict(dict)  # node_id -> {dim_type: (old, new, change_id)}
        self.change_counter = 0
        self.adapters = {}
        self.node_shapes = {}  # node_id -> (input_dim, output_dim)
        self.node_replacements = {}  # old_node_id -> new_node
        
        # Cycle detection
        self.cascade_order = []  # Track order of changes for cycle detection
        self.cascade_graph = defaultdict(set)  # Track what changes triggered what

    def adapt_dimensions(self,
                        node: torch.fx.Node,
                        node_shape: tuple,
                        input_shape: Optional[tuple] = None,
                        output_shape: Optional[tuple] = None) -> torch.fx.GraphModule:
        self.graph_input_shape = input_shape
        self.node_changes.clear()
        self.adapters.clear()
        self.change_counter = 0
        self.node_shapes.clear()
        self.node_replacements.clear()

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

    def _cascade_bfs(self, node, node_shape, input_shape=None, output_shape=None):
        """
        Enhanced BFS with cycle detection and resolution.
        Frontier entries: (node, direction, dim_type, target_dim, source_change_id, triggering_node_id)
        """
        frontier = deque()
        visited = {}  # (node_id, dim_type, target_dim) -> change_id
        
        # Initialize
        self._initialize_shape_tracking()
        input_dim, output_dim = node_shape
        self.change_counter += 1
        initial_change_id = self.change_counter
        
        # Update initial node shape
        self.node_shapes[id(node)] = (input_dim, output_dim)
        
        # Process initial node first
        if self._needs_change(node, 'output', output_dim):
            node = self._process_node_change(node, 'output', output_dim, initial_change_id, None)
        if self._needs_change(node, 'input', input_dim):
            node = self._process_node_change(node, 'input', input_dim, initial_change_id, None)
        
        # Add neighbors to frontier
        for neighbor in node.users:
            frontier.append((neighbor, 'backward', 'input', output_dim, initial_change_id, id(node)))
        for neighbor in [arg for arg in node.args if isinstance(arg, Node)]:
            frontier.append((neighbor, 'forward', 'output', input_dim, initial_change_id, id(node)))
        
        # BFS with cycle detection
        while frontier:
            current_node, direction, dim_type, target_dim, source_change_id, trigger_node_id = frontier.popleft()
            
            # Handle replacements
            current_node_id = id(current_node)
            if current_node_id in self.node_replacements:
                current_node = self.node_replacements[current_node_id]
                current_node_id = id(current_node)
            
            # Check if we've visited this exact change before
            visit_key = (current_node_id, dim_type, target_dim)
            if visit_key in visited:
                existing_change_id = visited[visit_key]
                if existing_change_id != source_change_id:
                    # Detected a cycle - different paths want different changes
                    cycle_info = self._detect_cycle(current_node, dim_type, target_dim, 
                                                   trigger_node_id, existing_change_id, source_change_id)
                    if cycle_info['is_cycle']:
                        self._handle_cycle(cycle_info, current_node, direction, dim_type, 
                                          target_dim, frontier, source_change_id)
                        continue
                else:
                    # Same change requested again, skip
                    continue
            
            visited[visit_key] = source_change_id
            
            # Get current shape
            current_shape = self._get_tracked_shape(current_node)
            if current_shape is None:
                continue
                
            current_input_dim, current_output_dim = current_shape
            current_dim = current_input_dim if dim_type == 'input' else current_output_dim
            
            if current_dim == target_dim:
                continue
            
            # Track cascade relationship
            self.cascade_graph[trigger_node_id].add((current_node_id, dim_type))
            
            # Process the change
            changed_node = self._process_node_change(current_node, dim_type, target_dim, 
                                                    source_change_id, trigger_node_id)
            
            # Get the updated shape after processing
            new_shape = self._get_tracked_shape(changed_node)
            if new_shape is None:
                continue
                
            new_input_dim, new_output_dim = new_shape
            
            # Add affected neighbors to frontier
            if dim_type == 'output' or new_input_dim != current_input_dim:
                # Output changed or input changed as side effect
                for neighbor in changed_node.users:
                    self._add_to_frontier_smart(changed_node, neighbor, 'forward', 
                                               frontier, source_change_id, visited)
            
            if dim_type == 'input' or new_output_dim != current_output_dim:
                # Input changed or output changed as side effect
                for neighbor in [arg for arg in changed_node.args if isinstance(arg, Node)]:
                    self._add_to_frontier_smart(changed_node, neighbor, 'backward', 
                                               frontier, source_change_id, visited)
        
        self.graph.recompile()
        return self.graph

    def _detect_cycle(self, node, dim_type, target_dim, trigger_node_id, 
                      existing_change_id, new_change_id):
        """Detect if we have a problematic cycle."""
        node_id = id(node)
        
        # Check if this node has been changed before
        if node_id in self.node_changes and dim_type in self.node_changes[node_id]:
            old_dim, prev_target, prev_change_id = self.node_changes[node_id][dim_type]
            
            # If the same dimension is being changed to a different value
            if prev_target != target_dim:
                # Trace back the cascade path to see if it forms a cycle
                path = self._trace_cascade_path(node_id, trigger_node_id)
                return {
                    'is_cycle': True,
                    'node': node,
                    'dim_type': dim_type,
                    'conflict': (prev_target, target_dim),
                    'path': path
                }
        
        return {'is_cycle': False}
    
    def _trace_cascade_path(self, start_node_id, end_node_id):
        """Trace the cascade path between two nodes."""
        # Simple BFS to find path in cascade graph
        queue = deque([(end_node_id, [end_node_id])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            if current == start_node_id:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for next_node, _ in self.cascade_graph.get(current, []):
                queue.append((next_node, path + [next_node]))
        
        return []
    
    def _handle_cycle(self, cycle_info, node, direction, dim_type, target_dim, 
                      frontier, source_change_id):
        """Handle detected cycles based on resolution strategy."""
        if self.cycle_resolution == 'adapter':
            # Insert adapter to break the cycle
            self._insert_cycle_breaking_adapter(node, dim_type, target_dim, cycle_info)
        elif self.cycle_resolution == 'second_order':
            # Allow second-order cascade with dampening
            self._handle_second_order_cascade(node, direction, dim_type, target_dim, 
                                             frontier, source_change_id, cycle_info)
    
    def _insert_cycle_breaking_adapter(self, node, dim_type, target_dim, cycle_info):
        """Insert an adapter to resolve the cycle without further cascading."""
        current_shape = self._get_tracked_shape(node)
        if not current_shape:
            return
            
        current_input, current_output = current_shape
        
        if dim_type == 'input':
            # Need to adapt input: insert adapter between parents and node
            for parent in [arg for arg in node.args if isinstance(arg, Node)]:
                parent_shape = self._get_tracked_shape(parent)
                if parent_shape:
                    adapter_key = (id(parent), id(node), 'cycle_break')
                    if adapter_key not in self.adapters:
                        adapter = self._create_adapter(parent_shape[1], target_dim)
                        self.adapters[adapter_key] = adapter
                        self._insert_adapter_between_nodes(parent, node, adapter)
        else:
            # Need to adapt output: insert adapter between node and children
            for child in node.users:
                child_shape = self._get_tracked_shape(child)
                if child_shape:
                    adapter_key = (id(node), id(child), 'cycle_break')
                    if adapter_key not in self.adapters:
                        adapter = self._create_adapter(current_output, target_dim)
                        self.adapters[adapter_key] = adapter
                        self._insert_adapter_between_nodes(node, child, adapter)
    
    def _handle_second_order_cascade(self, node, direction, dim_type, target_dim, 
                                     frontier, source_change_id, cycle_info):
        """Allow second-order cascades with cycle detection."""
        # Track that this is a second-order change
        node_id = id(node)
        cascade_depth = len(cycle_info.get('path', []))
        
        # Limit cascade depth to prevent infinite loops
        if cascade_depth > 10:  # Configurable depth limit
            print(f"Warning: Cascade depth limit reached at node {node.name}")
            self._insert_cycle_breaking_adapter(node, dim_type, target_dim, cycle_info)
            return
        
        # Allow the cascade to continue but mark it as second-order
        self.node_changes[node_id][f"{dim_type}_order"] = cascade_depth
        
        # Process the change normally
        changed_node = self._process_node_change(node, dim_type, target_dim, 
                                                 source_change_id, None)
        
        # Continue cascading with increased depth tracking
        new_shape = self._get_tracked_shape(changed_node)
        if new_shape:
            # Add neighbors with depth information
            for neighbor in changed_node.users:
                frontier.append((neighbor, 'backward', 'input', new_shape[1], 
                               source_change_id, id(changed_node)))
            for neighbor in [arg for arg in changed_node.args if isinstance(arg, Node)]:
                frontier.append((neighbor, 'forward', 'output', new_shape[0], 
                               source_change_id, id(changed_node)))
    
    def _add_to_frontier_smart(self, node, neighbor, direction, frontier, 
                               source_change_id, visited):
        """Add to frontier only if it won't create an immediate conflict."""
        node_shape = self._get_tracked_shape(node)
        if not node_shape:
            return
            
        if direction == 'forward':
            required_dim = node_shape[1]  # node's output
            target_dim_type = 'input'
        else:
            required_dim = node_shape[0]  # node's input
            target_dim_type = 'output'
        
        # Check if this would conflict with an existing visit
        visit_key = (id(neighbor), target_dim_type, required_dim)
        if visit_key not in visited or visited[visit_key] == source_change_id:
            frontier.append((neighbor, 'backward' if direction == 'forward' else 'forward',
                           target_dim_type, required_dim, source_change_id, id(node)))

    def _initialize_shape_tracking(self):
        """Initialize shape tracking from current graph state."""
        for node in self.graph.graph.nodes:
            if node.op == 'call_module':
                module = getattr(self.graph, node.target, None)
                if isinstance(module, nn.Linear):
                    self.node_shapes[id(node)] = (module.in_features, module.out_features)
                elif isinstance(module, nn.Conv2d):
                    self.node_shapes[id(node)] = (module.in_channels, module.out_channels)
                elif is_shapeless_module(module):
                    self.node_shapes[id(node)] = None
            elif node.op == 'placeholder':
                if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                    shape = node.meta['tensor_meta'].shape
                    if len(shape) > 0:
                        dim = shape[-1]
                        self.node_shapes[id(node)] = (dim, dim)

    def _get_tracked_shape(self, node):
        """Get shape from our tracking, handling shapeless nodes."""
        node_id = id(node)
        if node_id in self.node_shapes:
            shape = self.node_shapes[node_id]
            if shape is not None:
                return shape
        # For shapeless nodes, inherit from parent
        if node.op == 'call_module':
            module = getattr(self.graph, node.target, None)
            if is_shapeless_module(module) and node.args:
                parent = node.args[0]
                if isinstance(parent, Node):
                    parent_shape = self._get_tracked_shape(parent)
                    if parent_shape:
                        inherited_shape = (parent_shape[1], parent_shape[1])
                        self.node_shapes[node_id] = inherited_shape
                        return inherited_shape
        return None

    def _needs_change(self, node, dim_type, target_dim):
        """Check if a node needs dimension change."""
        shape = self._get_tracked_shape(node)
        if shape is None:
            return False
        current_dim = shape[0] if dim_type == 'input' else shape[1]
        return current_dim != target_dim

    def _process_node_change(self, node, dim_type, target_dim, source_change_id, triggering_node_id):
        """Process node change and update shape tracking."""
        current_shape = self._get_tracked_shape(node)
        if current_shape is None:
            return node

        current_input_dim, current_output_dim = current_shape
        current_dim = current_input_dim if dim_type == 'input' else current_output_dim

        if current_dim == target_dim:
            return node

        node_id = id(node)
        self.node_changes[node_id][dim_type] = (current_dim, target_dim, source_change_id)

        if self.use_reshape and node.op == 'call_module':
            module = getattr(self.graph, node.target, None)
            if isinstance(module, nn.Linear):
                new_input = target_dim if dim_type == 'input' else current_input_dim
                new_output = target_dim if dim_type == 'output' else current_output_dim
                new_node = self._replace_linear_node_tracked(node, new_input, new_output)
                old_node_id = id(node)
                new_node_id = id(new_node)
                self.node_replacements[old_node_id] = new_node
                self.node_shapes[new_node_id] = (new_input, new_output)
                del self.node_shapes[old_node_id]
                del self.node_changes[old_node_id]
                self.node_changes[new_node_id][dim_type] = (current_dim, target_dim, source_change_id)
                return new_node
        return node

    def _add_neighbor_to_frontier(self, node, neighbor, direction, frontier, source_change_id):
        """Add neighbor to frontier with appropriate dimension requirements."""
        node_shape = self._get_tracked_shape(node)
        if node_shape is None:
            return
        node_input, node_output = node_shape
        if direction == 'forward':
            required_dim = node_output
            frontier.append((neighbor, 'backward', 'input', required_dim, source_change_id))
        else:
            required_dim = node_input
            frontier.append((neighbor, 'forward', 'output', required_dim, source_change_id))

    def _has_dependency_conflict(self, node: Node, dim_type: str, target_dim: int, source_change_id: int) -> bool:
        node_id = id(node)
        if node_id in self.node_changes and dim_type in self.node_changes[node_id]:
            prev_change = self.node_changes[node_id][dim_type]
            prev_new_dim, prev_change_id = prev_change[1], prev_change[2]
            return (target_dim != prev_new_dim and source_change_id != prev_change_id)
        return False

    def _handle_dependency_conflict(self, node: Node, direction: str, dim_type: str, target_dim: int, 
                                   frontier: deque, source_change_id: int):
        current_shape = self._get_tracked_shape(node)
        if current_shape is None:
            return
        current_input_dim, current_output_dim = current_shape
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
                        parent_shape = self._get_tracked_shape(parent)
                        if parent_shape is None:
                            continue
                        parent_input_dim, parent_output_dim = parent_shape
                        adapter = self._create_adapter(parent_output_dim, target_dim)
                        self.adapters[adapter_key] = adapter
                        self._insert_adapter_between_nodes(parent, node, adapter)

    def _create_adapter(self, input_dim: int, output_dim: int) -> str:
        if input_dim > output_dim:
            adapter_module = nn.AdaptiveAvgPool1d(output_dim)
        else:
            adapter_module = nn.Linear(input_dim, output_dim)
        adapter_name = f"adapter_{self.change_counter}"
        self.change_counter += 1
        self.graph.add_module(adapter_name, adapter_module)
        return adapter_name

    def _insert_adapter_between_nodes(self, parent: Node, child: Node, adapter_module_name: str):
        with self.graph.graph.inserting_after(parent):
            adapter_node = self.graph.graph.call_module(adapter_module_name, args=(parent,))
        self.replace_parent_of_child(child, parent, adapter_node)

    def replace_parent_of_child(self, child, old_parent, new_parent):
        new_args = tuple(new_parent if a is old_parent else a for a in child.args)
        child.args = new_args

    def replace_child_of_parent(self, parent, old_child, new_child):
        for user in list(parent.users):
            new_args = tuple(new_child if a is old_child else a for a in user.args)
            user.args = new_args

    def _replace_linear_node_tracked(self, old_node, new_in, new_out):
        old_mod = getattr(self.graph, old_node.target)
        new_mod = nn.Linear(new_in, new_out)
        new_mod_name = f"{old_node.target}_reshaped_{int(time.time() * 1000000) % 1000000}"
        self.graph.add_module(new_mod_name, new_mod)
        parent_node = old_node.args[0] if old_node.args else None
        adapted_parent = parent_node
        if parent_node is not None and new_in != old_mod.in_features:
            parent_shape = self._get_tracked_shape(parent_node)
            if parent_shape and parent_shape[1] != new_in:
                self.graph, adapted_parent = adapt_node_shape(
                    self.graph, parent_node,
                    current_size=[parent_shape[1]],
                    target_size=[new_in]
                )
        with self.graph.graph.inserting_after(old_node):
            new_node = self.graph.graph.call_module(new_mod_name, args=(adapted_parent,), kwargs=old_node.kwargs)
        for user in list(old_node.users):
            user.replace_input_with(old_node, new_node)
        remove_node_flexible(self.graph, old_node)
        self.graph.recompile()
        return new_node

def is_shapeless_module(mod):
    if mod is None:
        return True
    shape_args = {'in_features', 'out_features', 'in_channels', 'out_channels', 'num_features', 'num_channels', 'features'}
    try:
        sig = inspect.signature(mod.__init__)
        param_names = set(sig.parameters.keys())
        param_names.discard('self')
        return len(shape_args & param_names) == 0
    except Exception:
        return True



TODO:Within this, we also need to ensure that we are not immediately overwriting a cascade step - for example when we go from A to B, B does not cause a cascade to A creating an infinite loop 

Currently, the naive case is that we can basically just change the input of B to match A. But, there might be a different layer type where the output of B is dependent on the input of B - in which case we will need to change B's children too. This is similar to propagating through the activation type layers like ReLU, where we need to reach the child's child. But, we only need to check the child's child if the output changes. 

One example of this might be a concatenation type layer. This is kind of a strange occurence because the reshaping could be fixed either by changing the output or the input of the concat node to match. Typically, for the concat node we would actually want to propagate forward instead of backwards, since it would make more sense to change the output. I.e. if the node is initially 100+100->200, we would do 150+100->250 instead of 150+50->200. There might be some other nodes, such as an elementwise add, that would cause a propagataion in both directions. For example, 100+100->100 would turn into 150+150->150 so we'd need a multidimensional reshape and a bidirectional  cascade. 