import os
import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.fx

from core import add_specific_node, get_graph
from cascade import Cascade


class SimpleModel(nn.Module):
    """Simple model for testing cascade functionality"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TestCascade(unittest.TestCase):
    """Simplified test cases for cascade functionality"""
    
    def test_basic_cascade(self):
        """Test basic cascade functionality with careful dimension control"""
        print("\nTesting basic cascade...")
        
        # Create a simple model
        model = SimpleModel()
        
        # Create input shape
        batch_size = 4
        input_size = 10
        input_shape = (batch_size, input_size)
        
        # Get the graph
        graph = get_graph(model, input_shape)
        
        # Create a sample input tensor
        sample_input = torch.randn(batch_size, input_size)
        
        # Get the original output as baseline
        original_output = graph(sample_input)
        print(f"Original output shape: {original_output.shape}")
        
        # Find the first linear layer
        fc1_node = None
        for node in graph.graph.nodes:
            if node.op == 'call_module' and node.name == 'fc1':
                fc1_node = node
                break
        
        self.assertIsNotNone(fc1_node, "Couldn't find fc1 module")
        
        # Create a cascade instance
        # To use reshaping for repairs:
        cascader = Cascade(graph, use_reshape=True)
        # To use adapters for repairs:
        # cascader = Cascade(graph, use_reshape=False)
        
        # Use a shape that would normally require adaptation
        new_hidden_size = 15  # Different from original 20
        
        try:
            # Apply cascade dimensionality changes
            graph = cascader.adapt_dimensions(
                node=fc1_node,
                node_shape=(input_size, new_hidden_size)
            )
            
            # Run the model with the same input
            new_output = graph(sample_input)
            
            print(f"New output shape: {new_output.shape}")
            
            # Check that output shape is maintained
            self.assertEqual(original_output.shape, new_output.shape,
                          "Output shape should be maintained after cascade")
            
            print("Basic cascade test passed!")
        except Exception as e:
            self.fail(f"Cascade test failed: {e}")


if __name__ == '__main__':
    unittest.main()