import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from core import get_graph, add_node, remove_node, shape_prop, adapt_shape
from modelbank import CausalSelfAttention






# Initialize model
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

myconfig = Config(n_head=12, n_embd=768, block_size=1024, attn_pdrop=0.1, resid_pdrop=0.1, batch_size=1)
model = CausalSelfAttention(myconfig)

# Get the computation graph
graph = get_graph(model)




#Print graph
print(graph.graph)
shape_prop(graph, (1, 1024, 768))


# Get the tensor shape from the node we're inserting after
before_node = list(graph.graph.find_nodes(op="call_module",target="c_proj"))[0]

# Get the actual module
# Output the shape of the layer
print(f"Shape of the layer (c_proj): {model.c_proj.weight.shape}")
print(f"Module type: {type(model.c_proj)}")
print(before_node.meta['tensor_meta'].shape)
input_shape = before_node.meta['tensor_meta'].shape
embedding_dim = input_shape[-1]  # Get the last dimension which is the embedding dimension

# Create new linear layer with matching dimensions
new_linear = nn.Linear(embedding_dim, embedding_dim)  # Both input and output should be embedding_dim (768)

add_node(graph, before_node, new_linear, "new_linear")

#Print graph
print(graph.graph)

# 1. Run shape propagation again to verify tensor shapes flow correctly
shape_prop(graph, (1, 1024, 768))

# 2. Print the nodes and their tensor metadata to verify the flow
for node in graph.graph.nodes:
    print(f"\nNode: {node.name} (op: {node.op})")
    print(f"Target: {node.target}")
    try:
        print(f"Shape: {node.meta['tensor_meta'].shape}")
    except:
        print(f"Shape: unknown")

# 3. Verify the graph is valid
graph.graph.lint()  # This will raise an error if the graph is invalid

# 4. Test the modified model with sample input
sample_input = torch.randn(1, 1024, 768)  # Match your input shape
try:
    output = graph.forward(sample_input)
    print("\nTest forward pass successful!")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"\nError in forward pass: {e}")


