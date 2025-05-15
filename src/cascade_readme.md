# Cascade: Neural Network Dimension Compatibility System

## Overview

The `Cascade` class handles cascading dimension changes throughout a neural network's computational graph. It ensures dimensional compatibility when nodes are added, modified, or removed, automatically resolving mismatches by inserting appropriate adapter modules.

## Key Features

- **Bidirectional Propagation**: Recursively adjusts dimensions both forward (to children) and backward (to parents)
- **Automatic Adaptation**: Inserts appropriate adapter layers (pooling or padding) when dimensions don't match
- **Cycle Prevention**: Safely handles networks with complex topologies, including skip connections
- **FX Graph Compatible**: Works with PyTorch's FX Graph representation

## How It Works

The cascade mechanism works by recursively traversing the network graph in both directions:

1. **Forward Propagation**: Starting at a modified node, checks all children (users) to ensure they can accept the node's output dimensions
2. **Backward Propagation**: Ensures all parent nodes produce output that's compatible with the modified node's input dimensions
3. **Dimension Resolution**: When mismatches are found, appropriate adapters (pooling or padding) are inserted

## Usage

### Basic Usage

```python
import torch
from cascade import Cascade

# Create a cascade handler with your graph
cascader = Cascade(fx_graph)

# When a node's dimensions change, adapt the rest of the graph
modified_graph = cascader.adapt_dimensions(
    node=modified_node,
    node_shape=(input_dim, output_dim),
    input_shape=input_node_shape,  # Optional
    output_shape=output_node_shape  # Optional
)
```

### Functional API

```python
from cascade import cascade_dimension_changes

# Apply cascading dimension changes with the functional API
modified_graph = cascade_dimension_changes(
    graph=fx_graph,
    node=modified_node,
    node_shape=(input_dim, output_dim)
)
```

## Integration with Existing Code

The adapt_connections function in core.py can use this system:

```python
def adapt_connections(
    graph: torch.fx.GraphModule,
    new_node: torch.fx.Node,
    new_node_shape: tuple,
    input_shape: tuple,
    output_shape: tuple
):
    from cascade import cascade_dimension_changes
    
    return cascade_dimension_changes(
        graph=graph,
        node=new_node,
        node_shape=new_node_shape,
        input_shape=input_shape,
        output_shape=output_shape
    )
```

## Technical Details

### Dimension Handling Logic

For children nodes:

- If node output > child input: Add adaptive pooling
- If node output < child input: Add padding

For parent nodes:

- If parent output > node input: Add adaptive pooling
- If parent output < node input: Add padding

### Shape Extraction

The system extracts dimensions from the FX graph's metadata, which must be populated using ShapeProp or similar mechanisms before cascade can operate correctly.

## Limitations and Considerations

- Works best with networks where the last dimension represents the feature dimension
- Requires shape propagation to be run on the graph beforehand
- Currently handles basic dimensional compatibility; specialized operations may need custom handling in future versions
- Inserted adapters may affect model performance and should be considered during training

## Future Enhancements

- Special handling for operations like concatenation, matrix multiplication, etc.
- Support for more complex dimensional transformations
- Options to control adapter types (beyond pooling/padding)