import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from core import get_graph, add_node, remove_node, shape_prop




# minGPT model slightly modified to be used with torch.fx, removed dynamic flow control
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.block_size = config.block_size
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        print(x)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        T = self.block_size
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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


