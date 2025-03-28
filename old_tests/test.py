import torch
import torch.nn as nn
from core import get_graph, add_node, remove_node, shape_prop
from modelbank import CausalSelfAttention

# Initialize model and config
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def setup_model():
    config = Config(n_head=12, n_embd=768, block_size=1024, attn_pdrop=0.1, resid_pdrop=0.1, batch_size=1)
    model = CausalSelfAttention(config)
    # Get the graph and create a GraphModule
    traced = get_graph(model)
    graph = torch.fx.GraphModule(model, traced.graph)
    # graph = get_graph(model)
    shape_prop(graph, (1, 1024, 768))
    return graph, model

def test_add_node_adapt_forward():
    print("\n=== Testing add_node with adapt_direction='new' ===")
    graph, model = setup_model()
    
    # Debug: Print all nodes in graph
    print("Available nodes:")
    for node in graph.graph.nodes:
        print(f"Node: {node.name}, op: {node.op}, target: {node.target}")
    
    # Get the c_proj node
    c_proj_nodes = list(graph.graph.find_nodes(op="call_module", target="c_proj"))
    if not c_proj_nodes:
        raise ValueError("Could not find c_proj node")
    c_proj_node = c_proj_nodes[0]
    
    # Debug: Print node metadata
    print("\nNode metadata:")
    print(f"Node name: {c_proj_node.name}")
    print(f"Node meta: {c_proj_node.meta}")
    
    # Create new linear layer with mismatched input dimension
    new_linear = nn.Linear(512, 768)  # Intentionally wrong input dim
    
    # Add node with forward adaptation (new layer will be adapted)
    try:
        graph, name = add_node(graph, c_proj_node, new_linear, name="test_linear", adapt_direction='new')
        print("\nNode added successfully")
    except Exception as e:
        print(f"\nError adding node: {e}")
        raise
    
    # Verify the modification
    print(f"Added layer name: {name}")
    print(f"New layer input dimension: {getattr(graph, name).in_features}")
    return graph

def test_add_node_adapt_backward():
    print("\n=== Testing add_node with adapt_direction='previous' ===")
    graph, model = setup_model()
    
    # Debug: Print all nodes in graph
    print("Available nodes before:")
    for node in graph.graph.nodes:
        print(f"Node: {node.name}, op: {node.op}, target: {node.target}")
        if 'tensor_meta' in node.meta and 'shape' in node.meta['tensor_meta']:
            print(f"  Shape: {node.meta['tensor_meta']['shape']}")
        else:
            print("  Shape: Not available")
    
    c_proj_node = list(graph.graph.find_nodes(op="call_module", target="c_proj"))[0]
    
    # Debug: Print c_proj node details
    print("\nc_proj node details:")
    print(f"Name: {c_proj_node.name}")
    print(f"Meta: {c_proj_node.meta}")
    print(f"Args: {c_proj_node.args}")
    print(f"Users: {list(c_proj_node.users)}")
    
    # Create new linear layer with mismatched input dimension
    new_linear = nn.Linear(512, 768)  # Intentionally wrong input dim
    print(f"\nNew linear layer dims: in={new_linear.in_features}, out={new_linear.out_features}")
    
    try:
        # Add node with backward adaptation (previous node will be adapted)
        graph, name = add_node(graph, c_proj_node, new_linear, name="test_linear", adapt_direction='previous')
        print("\nNode addition successful")
        print(f"Added layer name: {name}")
        
        # Debug: Print graph after modification
        print("\nNodes after modification:")
        for node in graph.graph.nodes:
            print(f"Node: {node.name}, op: {node.op}, target: {node.target}")
            if 'tensor_meta' in node.meta:
                print(f"  Shape: {node.meta['tensor_meta'].shape}")
    except Exception as e:
        print(f"\nError during node addition: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return graph

def test_remove_node_adapt_forward():
    print("\n=== Testing remove_node with adapt_direction='following' ===")
    graph, model = setup_model()
    
    # First add a node that we'll remove
    c_proj_node = list(graph.graph.find_nodes(op="call_module", target="c_proj"))[0]
    new_linear = nn.Linear(768, 512)  # Different output dim
    graph, name = add_node(graph, c_proj_node, new_linear, name="temp_linear")
    
    # Now remove the node, adapting following nodes
    temp_node = list(graph.graph.find_nodes(op="call_module", target="temp_linear"))[0]
    graph = remove_node(graph, temp_node, adapt_direction='following')
    
    print("Node removed with following nodes adapted")
    return graph

def test_remove_node_adapt_backward():
    print("\n=== Testing remove_node with adapt_direction='previous' ===")
    graph, model = setup_model()
    
    # First add a node that we'll remove
    c_proj_node = list(graph.graph.find_nodes(op="call_module", target="c_proj"))[0]
    new_linear = nn.Linear(768, 512)  # Different output dim
    graph, name = add_node(graph, c_proj_node, new_linear, name="temp_linear")
    
    # Now remove the node, adapting previous node
    temp_node = list(graph.graph.find_nodes(op="call_module", target="temp_linear"))[0]
    graph = remove_node(graph, temp_node, adapt_direction='previous')
    
    print("Node removed with previous node adapted")
    return graph

def test_forward_pass(graph):
    print("\n=== Testing forward pass ===")
    sample_input = torch.randn(1, 1024, 768)
    try:
        output = graph.forward(sample_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")

if __name__ == "__main__":
    # Test all cases
    graph1 = test_add_node_adapt_forward()
    test_forward_pass(graph1)
    
    graph2 = test_add_node_adapt_backward()
    test_forward_pass(graph2)
    
    graph3 = test_remove_node_adapt_forward()
    test_forward_pass(graph3)
    
    graph4 = test_remove_node_adapt_backward()
    test_forward_pass(graph4)


