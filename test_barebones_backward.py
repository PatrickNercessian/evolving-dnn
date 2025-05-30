import torch
import torch.nn as nn
import torch.optim as optim

from src.individual_graph_module import IndividualGraphModule
from src.core import get_graph

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create a simple model and wrap it in a graph module
input_size = 2
hidden_size = 4
output_size = 1
model = SimpleModel(input_size, hidden_size, output_size)
graph_module = get_graph(model, input_shape=(1, input_size))

# Dummy data
X = torch.tensor([[1.0, 0.0]])
y = torch.tensor([[1.0]])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(graph_module.parameters(), lr=0.1)

# Save initial weights
initial_weight = None
for name, param in graph_module.named_parameters():
    if 'linear1.weight' in name:
        initial_weight = param.clone().detach()
        break

# Forward, loss, backward, step
optimizer.zero_grad()
output = graph_module(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()

# Check if weights changed
updated_weight = None
for name, param in graph_module.named_parameters():
    if 'linear1.weight' in name:
        updated_weight = param.clone().detach()
        break

print("Initial weight:\n", initial_weight)
print("Updated weight:\n", updated_weight)
print("Weights changed:", not torch.equal(initial_weight, updated_weight)) 