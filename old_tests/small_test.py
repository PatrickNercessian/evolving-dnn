import torch
import torch.nn as nn
import torch.fx
from core_old2 import get_graph, add_node, remove_node, shape_prop

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Create an instance of the model and get its graph
model = SimpleModel()
graph = get_graph(model)

# Define input shape
input_shape = (1, 10)

# Propagate shapes through the graph
shape_prop(graph, input_shape)

# Helper function to print graph structure
def print_graph(graph, title="Current Graph Structure"):
    print(f"\n{title}")
    print("-" * 50)
    for node in graph.graph.nodes:
        # Get shape info if available
        shape_info = ""
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            output_shape = node.meta['tensor_meta'].shape
            shape_info = f" - output shape: {output_shape}"
        
        # Get input shape from args
        input_shape_info = ""
        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                    input_shape = arg.meta['tensor_meta'].shape
                    input_shape_info = f" - input shape: {input_shape}"
                    break
        
        # Get args info
        args_info = ""
        if hasattr(node, 'args') and node.args:
            args_str = ", ".join([arg.name if hasattr(arg, 'name') else str(arg) for arg in node.args])
            args_info = f" - args: ({args_str})"
        
        print(f"{node.op} {node.name} -> {node.target}{input_shape_info}{shape_info}{args_info}")
    print("-" * 50)

# Print initial graph
print_graph(graph, "Initial Graph Structure")

# Test adding a new node
def test_add_new_node():
    global graph
    print("Testing add new node...")
    # Get the first node after input
    first_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and node.target == 'linear1':
            first_node = node
            break
    
    if not first_node:
        print("Could not find linear1 node")
        return

    # Add a new linear layer after the first node
    new_layer = nn.Linear(15, 20)  # Output dimension (20) matches linear1's output
    graph, _ = add_node(graph, first_node, new_layer, name='new_linear', adapt_direction='new')
    
    shape_prop(graph, input_shape)
    # New node should have its input dimension (15) changed to match linear1's output (20)
    print("New node added successfully.")
    print_graph(graph, "Graph After Adding New Node")

# Test adding a node by adapting the previous node
def test_add_previous_node():
    global graph
    print("Testing add previous node...")
    # Get the first node after input
    first_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and node.target == 'linear1':
            first_node = node
            break
    
    if not first_node:
        print("Could not find linear1 node")
        return

    new_layer = nn.Linear(25, 20) # Will feed into new_linear
    graph, _ = add_node(graph, first_node, new_layer, name='adapted_linear', adapt_direction='previous')
    shape_prop(graph, input_shape)
    # First layer will be adapted to have output shape match adapted_linear's input shape: was 20, is now 25
    print("Node added by adapting previous node successfully.")
    print_graph(graph, "Graph After Adapting Previous Node")

# Test removing a node
def test_remove_node():
    global graph
    print("Testing remove node...")
    # Get the node to remove
    node_to_remove = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and node.target == 'adapted_linear':
            node_to_remove = node
            break

    if node_to_remove:
        graph = remove_node(graph, node_to_remove, adapt_direction='previous')
        print("Node removed successfully.")
        print_graph(graph, "Graph After Removing Node")
    else:
        print("Could not find linear2 node")

# Run tests
test_add_new_node()
test_add_previous_node()
test_remove_node()