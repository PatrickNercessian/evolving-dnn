import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

# Minimal test to understand the cascade flow
class SimpleChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def trace_cascade_steps():
    """Trace through what should happen in the cascade"""
    print("=== CASCADE DEBUG TRACE ===\n")
    
    # Create model and graph
    model = SimpleChain()
    graph = torch.fx.symbolic_trace(model)
    input_shape = (4, 10)
    
    # Run initial shape propagation
    example_input = torch.randn(*input_shape)
    ShapeProp(graph).propagate(example_input)
    
    # Print initial state
    print("Initial Graph State:")
    for node in graph.graph.nodes:
        if node.op == 'call_module':
            module = getattr(graph, node.target)
            if isinstance(module, nn.Linear):
                print(f"  {node.name}: Linear({module.in_features}, {module.out_features})")
            else:
                print(f"  {node.name}: {type(module).__name__}")
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                print(f"    Output shape: {shape}")
    
    print("\n--- Cascade Goal: Change fc1 output from 20 to 15 ---")
    
    # Simulate what SHOULD happen in BFS order
    print("\nExpected BFS Processing Order:")
    print("1. Initial frontier:")
    print("   - (fc1, forward, output, 15)")
    print("   - (fc1, backward, input, 10)")
    
    print("\n2. Process fc1 forward:")
    print("   - fc1 needs output change: 20 -> 15")
    print("   - Replace fc1 with new Linear(10, 15)")
    print("   - Add neighbors to frontier:")
    print("     - relu is shapeless, add (relu, forward, output, 15)")
    
    print("\n3. Process fc1 backward:")
    print("   - fc1 input is already 10, no change needed")
    
    print("\n4. Process relu forward:")
    print("   - relu is shapeless, inherits shape")
    print("   - Add neighbors to frontier:")
    print("     - fc2 needs input change, add (fc2, backward, input, 15)")
    
    print("\n5. Process fc2 backward:")
    print("   - fc2 needs input change: 20 -> 15")
    print("   - Replace fc2 with new Linear(15, 5)")
    
    print("\n--- What's Actually Happening ---")
    print("The error '(4x15 and 20x5)' suggests:")
    print("1. fc1 was successfully replaced: outputs 15")
    print("2. fc2 was NOT yet replaced: still expects 20")
    print("3. Shape propagation runs BEFORE fc2 is updated")
    print("\nPossible causes:")
    print("- Shape propagation is called too early in the process")
    print("- The BFS isn't reaching fc2 before shape prop runs")
    print("- Node replacement triggers shape prop prematurely")

def test_reshape_node_isolation():
    """Test reshape_node in isolation to see when shape prop fails"""
    print("\n\n=== TESTING RESHAPE_NODE IN ISOLATION ===\n")
    
    model = SimpleChain()
    graph = torch.fx.symbolic_trace(model)
    input_shape = (4, 10)
    
    # Initial shape prop
    example_input = torch.randn(*input_shape)
    ShapeProp(graph).propagate(example_input)
    
    # Find fc1
    fc1_node = None
    for node in graph.graph.nodes:
        if node.name == 'fc1':
            fc1_node = node
            break
    
    print("Before reshape:")
    print(f"  fc1: {getattr(graph, 'fc1').in_features} -> {getattr(graph, 'fc1').out_features}")
    print(f"  fc2: {getattr(graph, 'fc2').in_features} -> {getattr(graph, 'fc2').out_features}")
    
    # Import the functions we need
    from cascade import reshape_node
    
    print("\nCalling reshape_node on fc1 (20 -> 15)...")
    try:
        graph, new_fc1 = reshape_node(graph, fc1_node, new_out_features=15, example_input=example_input)
        print("Reshape succeeded!")
        
        # Check the state
        print(f"\nAfter reshape:")
        for node in graph.graph.nodes:
            if node.op == 'call_module' and 'fc' in node.name:
                module = getattr(graph, node.target)
                print(f"  {node.name} ({node.target}): {module.in_features} -> {module.out_features}")
        
        # Try to run the graph
        print("\nTrying to run the reshaped graph...")
        try:
            output = graph(example_input)
            print(f"Success! Output shape: {output.shape}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print("This confirms fc2 wasn't updated!")
            
    except Exception as e:
        print(f"Reshape failed: {e}")

def test_cascade_bfs_order():
    """Test to verify BFS processing order"""
    print("\n\n=== TESTING CASCADE BFS ORDER ===\n")
    
    from cascade import Cascade
    
    model = SimpleChain()
    graph = torch.fx.symbolic_trace(model)
    input_shape = (4, 10)
    
    # Patch the cascade to log processing order
    class DebugCascade(Cascade):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.processing_log = []
        
        def _process_node_change(self, node, dim_type, target_dim, source_change_id):
            self.processing_log.append({
                'node': node.name,
                'dim_type': dim_type,
                'target_dim': target_dim,
                'op': 'process'
            })
            return super()._process_node_change(node, dim_type, target_dim, source_change_id)
        
        def _add_neighbors_to_frontier(self, node, direction, neighbors, frontier, source_change_id):
            for n in neighbors:
                self.processing_log.append({
                    'node': n.name,
                    'direction': direction,
                    'op': 'add_to_frontier'
                })
            super()._add_neighbors_to_frontier(node, direction, neighbors, frontier, source_change_id)
    
    # Find fc1
    fc1_node = None
    for node in graph.graph.nodes:
        if node.name == 'fc1':
            fc1_node = node
            break
    
    cascader = DebugCascade(graph, use_reshape=True)
    
    print("Running cascade to change fc1 output: 20 -> 15")
    try:
        graph = cascader.adapt_dimensions(
            node=fc1_node,
            node_shape=(10, 15),
            input_shape=input_shape
        )
        print("Cascade completed!")
    except Exception as e:
        print(f"Cascade failed: {e}")
    
    print("\nProcessing log:")
    for i, entry in enumerate(cascader.processing_log):
        print(f"{i+1}. {entry}")

# Run all tests
if __name__ == "__main__":
    trace_cascade_steps()
    test_reshape_node_isolation()
    test_cascade_bfs_order()