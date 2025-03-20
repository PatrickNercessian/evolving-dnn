import torch
import torch.nn as nn
import torch.fx
from core import get_graph, add_node, adapt_connections, shape_prop

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
        
        print(f"{node.op} {node.name} -> {node.target}{input_shape_info}{shape_info}")
    print("-" * 50)

def get_fresh_graph():
    """Helper function to get a fresh graph with shape propagation"""
    model = SimpleModel()
    input_shape = (1, 10)
    graph = get_graph(model)
    shape_prop(graph, input_shape)
    return graph, input_shape

def test_add_operator():
    print("\nTesting add operator...")
    graph, input_shape = get_fresh_graph()
    print_graph(graph, "Initial Graph Structure")
    
    # Get the first linear layer node
    linear1_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and node.target == 'linear1':
            linear1_node = node
            break
    
    if not linear1_node:
        print("Could not find linear1 node")
        return

    # Create a tensor to add
    add_tensor = torch.ones(1, 20)  # Match linear1 output shape including batch dimension
    graph, _ = add_node(graph, linear1_node, 'add', operator_params=add_tensor)
    shape_prop(graph, input_shape)
    print_graph(graph, "Graph After Adding Addition Operation")

def test_mul_operator():
    print("\nTesting multiplication operator...")
    graph, input_shape = get_fresh_graph()
    print_graph(graph, "Initial Graph Structure")
    
    # Get the first linear layer node
    linear1_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and node.target == 'linear1':
            linear1_node = node
            break

    # Create a tensor to multiply
    mul_tensor = torch.ones(1, 20) * 2.0  # Match linear1 output shape including batch dimension
    graph, _ = add_node(graph, linear1_node, 'mul', operator_params=mul_tensor)
    shape_prop(graph, input_shape)
    print_graph(graph, "Graph After Adding Multiplication Operation")

def test_matmul_operator():
    print("\nTesting matrix multiplication operator...")
    graph, input_shape = get_fresh_graph()
    print_graph(graph, "Initial Graph Structure")
    
    # Get both linear layer nodes
    linear1_node = None
    linear2_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module':
            if node.target == 'linear1':
                linear1_node = node
            elif node.target == 'linear2':
                linear2_node = node

    # First add matmul that changes dimensions
    matmul_tensor = torch.randn(20, 25)  # Change from 20 to 25 dimensions
    graph, _ = add_node(graph, linear1_node, 'matmul', operator_params=matmul_tensor)
    shape_prop(graph, input_shape)
    print_graph(graph, "Graph After Adding Matrix Multiplication")

    # Then adapt the connections to handle the new shape
    new_shape = (1, 25)  # New shape after matmul
    graph = adapt_connections(graph, linear1_node, new_shape)
    shape_prop(graph, input_shape)
    print_graph(graph, "Graph After Adapting Connections")

# Run tests
if __name__ == "__main__":
    test_add_operator()
    test_mul_operator()
    test_matmul_operator() 