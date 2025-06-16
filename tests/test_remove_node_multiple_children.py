import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import logging

# Import the functions we want to test
from src.nn.variation.architecture_mutation import _remove_node
from src.nn.individual_graph_module import NeuralNetworkIndividualGraphModule

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def create_test_model_single_child():
    """Creates a simple model where one node has a single child"""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 8)  # This will be removed
            self.layer2 = nn.Linear(8, 5)   # Single child
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    return TestModel()

def create_test_model_multiple_children():
    """Creates a model where one node has multiple children"""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_layer = nn.Linear(10, 8)  # This will be removed - has multiple children
            self.branch1 = nn.Linear(8, 3)
            self.branch2 = nn.Linear(8, 4)
            self.branch3 = nn.Linear(8, 6)
        
        def forward(self, x):
            shared = self.shared_layer(x)
            # Multiple children using the same shared output
            out1 = self.branch1(shared)
            out2 = self.branch2(shared) 
            out3 = self.branch3(shared)
            return out1, out2, out3
    
    return TestModel()

def get_graph_from_model(model, input_shape=(1, 10)):
    """Convert model to FX graph with shape propagation"""
    example_input = torch.randn(input_shape)
    graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model), example_input=example_input)
    ShapeProp(graph).propagate(example_input)
    return graph

def test_single_child_removal():
    """Test removing a node with a single child"""
    print("\n=== Testing Single Child Removal ===")
    
    model = create_test_model_single_child()
    graph = get_graph_from_model(model)
    
    print("Original graph nodes:")
    for node in graph.graph.nodes:
        print(f"  {node.name}: {node.op} - {node.target}")
    
    # Find the first linear layer to remove (layer1)
    layer1_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and 'layer1' in node.target:
            layer1_node = node
            break
    
    if layer1_node is None:
        print("Could not find layer1 node!")
        return
    
    print(f"\nRemoving node: {layer1_node.name}")
    print(f"Node users before removal: {[u.name for u in layer1_node.users]}")
    
    # Remove the node
    graph, remaining_node = _remove_node(graph, layer1_node)
    
    print(f"\nAfter removal nodes:")
    for node in graph.graph.nodes:
        print(f"  {node.name}: {node.op} - {node.target}")
    
    # Test that the graph still works
    example_input = torch.randn(1, 10)
    try:
        output = graph(example_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

def test_multiple_children_removal():
    """Test removing a node with multiple children"""
    print("\n=== Testing Multiple Children Removal ===")
    
    model = create_test_model_multiple_children()
    graph = get_graph_from_model(model)
    
    print("Original graph nodes:")
    for node in graph.graph.nodes:
        print(f"  {node.name}: {node.op} - {node.target}")
    
    # Find the shared layer to remove
    shared_layer_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and 'shared_layer' in node.target:
            shared_layer_node = node
            break
    
    if shared_layer_node is None:
        print("Could not find shared_layer node!")
        return
    
    print(f"\nRemoving node: {shared_layer_node.name}")
    print(f"Node users before removal: {[u.name for u in shared_layer_node.users]}")
    print(f"Number of children: {len(list(shared_layer_node.users))}")
    
    # Remove the node
    graph, remaining_node = _remove_node(graph, shared_layer_node)
    
    print(f"\nAfter removal nodes:")
    for node in graph.graph.nodes:
        print(f"  {node.name}: {node.op} - {node.target}")
    
    # Test that the graph still works
    example_input = torch.randn(1, 10)
    try:
        outputs = graph(example_input)
        print(f"Forward pass successful!")
        for i, output in enumerate(outputs):
            print(f"  Output {i+1} shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

def test_shape_information():
    """Test that shape information is preserved correctly"""
    print("\n=== Testing Shape Information ===")
    
    model = create_test_model_multiple_children()
    graph = get_graph_from_model(model)
    
    print("Node shapes before removal:")
    for node in graph.graph.nodes:
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            print(f"  {node.name}: {node.meta['tensor_meta'].shape}")
    
    # Find and remove shared layer
    shared_layer_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and 'shared_layer' in node.target:
            shared_layer_node = node
            break
    
    if shared_layer_node:
        graph, remaining_node = _remove_node(graph, shared_layer_node)
        
        print("\nNode shapes after removal:")
        for node in graph.graph.nodes:
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                print(f"  {node.name}: {node.meta['tensor_meta'].shape}")

if __name__ == "__main__":
    print("Testing Node Removal with Multiple Children")
    print("=" * 50)
    
    test_single_child_removal()
    test_multiple_children_removal()
    test_shape_information()
    
    print("\n" + "=" * 50)
    print("Tests completed!") 