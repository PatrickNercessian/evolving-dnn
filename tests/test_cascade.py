import os
import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.fx

from core import add_specific_node, get_graph # Assuming add_specific_node is used elsewhere or can be removed if not
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
        """Test basic cascade functionality with careful dimension control."""
        for use_reshape_flag in [True, False]:
            with self.subTest(use_reshape=use_reshape_flag):
                print(f"\nTesting basic cascade with use_reshape={use_reshape_flag}...")
                
                # Create a simple model
                model = SimpleModel()
                
                # Create input shape
                batch_size = 4
                input_size = 10
                input_shape = (batch_size, input_size)
                
                # Get the graph
                graph = get_graph(model, input_shape)
                
                # Create a sample input tensor
                sample_input = torch.randn(*input_shape)
                
                # Get the original output as baseline
                original_output = graph(sample_input)
                
                # Find the first linear layer
                fc1_node = None
                for node_iter in graph.graph.nodes:
                    if node_iter.op == 'call_module' and node_iter.name == 'fc1':
                        fc1_node = node_iter
                        break
                self.assertIsNotNone(fc1_node, "Couldn't find fc1 module")
                
                # Create a cascade instance with the current strategy
                cascader = Cascade(graph, use_reshape=use_reshape_flag)
                
                # Define the new shape for fc1's output
                new_hidden_size = 15
                
                try:
                    # Apply cascade dimensionality changes
                    # Pass input_shape for internal ShapeProp calls in Cascade
                    graph = cascader.adapt_dimensions(
                        node=fc1_node,
                        node_shape=(input_size, new_hidden_size),
                        input_shape=input_shape 
                    )
                    
                    # Run the model with the same input
                    new_output = graph(sample_input)
                    
                    # Check that output shape is maintained
                    self.assertEqual(original_output.shape, new_output.shape,
                                  f"Output shape should be maintained after cascade (use_reshape={use_reshape_flag})")
                    
                    print(f"Basic cascade test passed for use_reshape={use_reshape_flag}!")
                except Exception as e:
                    self.fail(f"Basic cascade test failed for use_reshape={use_reshape_flag}: {e}")

    def test_cascade_functionality(self):
        """Test that the cascaded graph is functional for forward and backward passes."""
        for use_reshape_flag in [True, False]:
            with self.subTest(use_reshape=use_reshape_flag):
                print(f"\nTesting cascade graph functionality with use_reshape={use_reshape_flag}...")

                # Create a simple model
                model = SimpleModel()
                batch_size = 4
                input_size = 10
                input_shape = (batch_size, input_size)
                graph = get_graph(model, input_shape)
                sample_input = torch.randn(*input_shape, requires_grad=True)
                original_output = graph(sample_input.clone().detach()) # For shape comparison

                # Find the first linear layer
                fc1_node = None
                for node_iter in graph.graph.nodes:
                    if node_iter.op == 'call_module' and node_iter.name == 'fc1':
                        fc1_node = node_iter
                        break
                self.assertIsNotNone(fc1_node, "Couldn't find fc1 module")

                # Create a cascade instance with the current strategy
                cascader = Cascade(graph, use_reshape=use_reshape_flag)
                
                # Define the new shape for fc1's output
                new_hidden_size = 15
                
                try:
                    # Apply cascade dimensionality changes
                    # Pass input_shape for internal ShapeProp calls in Cascade
                    graph = cascader.adapt_dimensions(
                        node=fc1_node,
                        node_shape=(input_size, new_hidden_size),
                        input_shape=input_shape
                    )

                    # Forward pass
                    output = graph(sample_input)
                    # Check that output shape is maintained (or matches expected final shape)
                    self.assertEqual(output.shape, original_output.shape,
                                     f"Output shape mismatch after cascade (use_reshape={use_reshape_flag})")
                    # Or, more specifically for this model:
                    # self.assertEqual(output.shape[0], batch_size)
                    # self.assertEqual(output.shape[1], 5) 

                    # Backward pass
                    loss = output.sum()
                    loss.backward() 
                    self.assertIsNotNone(sample_input.grad, f"Gradients not computed (use_reshape={use_reshape_flag})")
                    print(f"Cascade graph functional for forward/backward with use_reshape={use_reshape_flag}.")
                except Exception as e:
                    self.fail(f"Cascade functionality test failed for use_reshape={use_reshape_flag}: {e}")

    def test_cascade_parent_adjustment(self):
        """Test cascade when parent node must be adjusted (input dimension change)."""
        for use_reshape_flag in [True, False]:
            with self.subTest(use_reshape=use_reshape_flag):
                print(f"\nTesting parent adjustment with use_reshape={use_reshape_flag}...")
                # Model: input -> fc1 -> act -> fc2
                model = SimpleModel()
                batch_size = 4
                input_size = 10
                input_shape = (batch_size, input_size)
                graph = get_graph(model, input_shape)
                sample_input = torch.randn(*input_shape)
                original_output = graph(sample_input)

                # Find fc2 node
                fc2_node = None
                for node_iter in graph.graph.nodes:
                    if node_iter.op == 'call_module' and node_iter.name == 'fc2':
                        fc2_node = node_iter
                        break
                self.assertIsNotNone(fc2_node, "Couldn't find fc2 module")

                # Cascade: change fc2's input to 8 (from 20), should propagate backward to fc1
                cascader = Cascade(graph, use_reshape=use_reshape_flag)
                new_fc2_in = 8
                graph = cascader.adapt_dimensions(
                    node=fc2_node,
                    node_shape=(new_fc2_in, 5),
                    input_shape=input_shape
                )
                new_output = graph(sample_input)
                self.assertEqual(original_output.shape, new_output.shape)
                print(f"Parent adjustment test passed for use_reshape={use_reshape_flag}!")

    def test_cascade_through_shapeless(self):
        """Test cascade traverses through shapeless nodes (e.g., ReLU, Dropout)."""
        class ModelWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout()
                self.fc2 = nn.Linear(20, 5)
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.drop(x)
                x = self.fc2(x)
                return x

        for use_reshape_flag in [True, False]:
            with self.subTest(use_reshape=use_reshape_flag):
                print(f"\nTesting cascade through shapeless nodes with use_reshape={use_reshape_flag}...")
                model = ModelWithDropout()
                batch_size = 4
                input_size = 10
                input_shape = (batch_size, input_size)
                graph = get_graph(model, input_shape)
                sample_input = torch.randn(*input_shape)
                original_output = graph(sample_input)

                # Find fc1 node
                fc1_node = None
                for node_iter in graph.graph.nodes:
                    if node_iter.op == 'call_module' and node_iter.name == 'fc1':
                        fc1_node = node_iter
                        break
                self.assertIsNotNone(fc1_node, "Couldn't find fc1 module")

                # Cascade: change fc1's output to 15, should propagate through ReLU and Dropout to fc2
                cascader = Cascade(graph, use_reshape=use_reshape_flag)
                new_hidden_size = 15
                graph = cascader.adapt_dimensions(
                    node=fc1_node,
                    node_shape=(input_size, new_hidden_size),
                    input_shape=input_shape
                )
                new_output = graph(sample_input)
                self.assertEqual(original_output.shape, new_output.shape)
                print(f"Cascade through shapeless nodes test passed for use_reshape={use_reshape_flag}!")


if __name__ == '__main__':
    unittest.main()