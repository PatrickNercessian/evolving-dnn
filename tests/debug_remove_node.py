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

def debug_simple_case():
    """Debug the simplest possible case"""
    print("=== Debug Simple Case ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 8)  # This will be removed
            self.layer2 = nn.Linear(8, 5)   # Single child
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    model = SimpleModel()
    example_input = torch.randn(1, 10)
    graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model), example_input=example_input)
    ShapeProp(graph).propagate(example_input)
    
    print("Nodes and their shapes BEFORE removal:")
    for node in graph.graph.nodes:
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            print(f"  {node.name}: {node.op} - {node.target} - shape: {node.meta['tensor_meta'].shape}")
        else:
            print(f"  {node.name}: {node.op} - {node.target} - no shape")
    
    # Find the layer1 node
    layer1_node = None
    for node in graph.graph.nodes:
        if node.op == 'call_module' and 'layer1' in node.target:
            layer1_node = node
            break
    
    if layer1_node:
        print(f"\nAbout to remove: {layer1_node.name}")
        print(f"Children: {[u.name for u in layer1_node.users]}")
        print(f"Parent: {layer1_node.args[0].name if layer1_node.args else 'None'}")
        
        # Let's examine the shapes more closely
        feeding_node = layer1_node.args[0]
        print(f"Feeding node shape: {feeding_node.meta['tensor_meta'].shape}")
        print(f"Removed node shape: {layer1_node.meta['tensor_meta'].shape}")
        
        # Check what the children expect
        for user in layer1_node.users:
            print(f"Child {user.name} shape: {user.meta['tensor_meta'].shape}")
        
        print("\n=== REMOVING NODE ===")
        try:
            print("Calling _remove_node...")
            graph, remaining_node = _remove_node(graph, layer1_node)
            print(f"SUCCESS: Node removed successfully! Remaining node: {remaining_node.name}")
            
            print("\nNodes AFTER removal:")
            for node in graph.graph.nodes:
                if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                    print(f"  {node.name}: {node.op} - {node.target} - shape: {node.meta['tensor_meta'].shape}")
                else:
                    print(f"  {node.name}: {node.op} - {node.target} - no shape")
            
            # Test forward pass
            print("\n=== TESTING FORWARD PASS ===")
            test_input = torch.randn(1, 10)
            print(f"Input shape: {test_input.shape}")
            try:
                output = graph(test_input)
                print(f"Output shape: {output.shape}")
                print("Forward pass SUCCESSFUL!")
            except Exception as forward_error:
                print(f"Forward pass ERROR: {forward_error}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"ERROR during node removal: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Could not find layer1 node!")

if __name__ == "__main__":
    debug_simple_case() 