import random
import torch
from src.core import add_node, remove_node
from src.utils import node_has_shape

def mutation_add_linear(individual):
    print("Starting mutation_add_linear")
    # Find a random node in the graph to add a linear layer after
    eligible_nodes = _get_eligible_nodes(individual)
    if not eligible_nodes:
        print("No eligible nodes for adding linear layer")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding linear layer after {reference_node.name}")
    
    # Get the output shape of the reference node if available
    if hasattr(reference_node, 'meta') and 'tensor_meta' in reference_node.meta:
        input_shape = reference_node.meta['tensor_meta'].shape
        # Use the feature dimension (last dimension) as input size
        input_size = input_shape[-1]
        # Choose a reasonable output size similar to input size
        output_size = random.randint(max(1, input_size // 2), input_size * 2)
        print(f"Using input_size={input_size}, output_size={output_size} from reference node shape")
    else:
        # Fallback to a safe small size if shape information is not available
        print("Shape information not available, using default sizes")
        input_size = 8
        output_size = 8
    
    # Add a linear layer
    individual.graph_module = add_node(individual.graph_module, reference_node, 'linear', 
                                        input_size=input_size, output_size=output_size)
    # Adjust training config (you could add learning rate mutation here)
    if random.random() < 0.3:
        individual.train_config.learning_rate *= random.uniform(0.5, 1.5)
        individual.train_config.learning_rate = max(0.0001, min(0.1, individual.train_config.learning_rate))
    
    print("Completed mutation_add_linear")
    return individual


def mutation_add_relu(individual):
    print("Starting mutation_add_relu")
    # Find a random node in the graph to add a ReLU layer after
    eligible_nodes = _get_eligible_nodes(individual)
    if not eligible_nodes:
        print("No eligible nodes for adding ReLU")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding ReLU after {reference_node.name}")
    
    # Add a ReLU layer
    individual.graph_module = add_node(individual.graph_module, reference_node, 'relu')
    print("Completed mutation_add_relu")
    return individual


def mutation_add_skip_connection(individual):
    print("Starting mutation_add_skip_connection")
    # Find two random nodes in the graph to connect
    eligible_nodes = _get_eligible_nodes(individual)
    
    if len(eligible_nodes) < 2:
        print("Not enough eligible nodes for skip connection")
        return individual
    
    # Pick two different nodes, ensuring first_node comes before second_node
    first_node = random.choice(eligible_nodes)
    later_nodes = [n for n in eligible_nodes if n != first_node and 
                    list(individual.graph_module.graph.nodes).index(n) > 
                    list(individual.graph_module.graph.nodes).index(first_node)]
    
    if not later_nodes:
        print("No eligible later nodes for skip connection")
        return individual
        
    second_node = random.choice(later_nodes)
    print(f"Adding skip connection from {first_node.name} to {second_node.name}")
    
    # Add skip connection
    individual.graph_module = add_node(individual.graph_module, second_node, 'skip', first_node=first_node)
    print("Completed mutation_add_skip_connection")
    
    return individual


def mutation_add_branch(individual):
    print("Starting mutation_add_branch")
    # Find a random node in the graph to add branches after
    eligible_nodes = _get_eligible_nodes(individual)
    
    if not eligible_nodes:
        print("No eligible nodes for adding branches")
        return individual
    
    reference_node = random.choice(eligible_nodes)
    print(f"Adding branch after {reference_node.name}")
    
    # Get the shape information from the reference node
    if hasattr(reference_node, 'meta') and 'tensor_meta' in reference_node.meta:
        input_shape = reference_node.meta['tensor_meta'].shape
        input_size = input_shape[-1]
        # Choose reasonable output sizes for branches
        branch1_out_size = max(1, input_size // 2)
        branch2_out_size = max(1, input_size // 2)
    else:
        print("Shape information not available, using default sizes")
        input_size = 8
        branch1_out_size = 4
        branch2_out_size = 4
        
    # Branch out sizes are random ints
    branch1_out_size = random.randint(1, 500)
    branch2_out_size = random.randint(1, 500)
    
    # Add branch nodes
    individual.graph_module = add_node(individual.graph_module, reference_node, 'branch',
                                        branch1_out_size=branch1_out_size,
                                        branch2_out_size=branch2_out_size)
    print("Completed mutation_add_branch")
    
    return individual


def mutation_remove_node(individual):
    print("Starting mutation_remove_node")
    # Find eligible nodes to remove (not input, output, or critical nodes)
    nodes = list(individual.graph_module.graph.nodes)
    possible_nodes = _get_eligible_nodes(individual, nodes)
    eligible_nodes = []

    for node in possible_nodes:
        # Skip placeholder (input) and output nodes
        if node.op in ['placeholder', 'output']:
            continue
        
        # Skip nodes that are skip connections or branch nodes (as per remove_node restrictions)
        if hasattr(node, 'target') and node.target in (torch.add, torch.cat, torch.mul):
            continue
            
        # Skip nodes that are the first node in a branch (have multiple users)
        if len(node.users) > 1:
            continue
            
        # Skip if this is the only non-input/output node (would break the graph)
        non_io_nodes = [n for n in nodes if n.op not in ['placeholder', 'output']]
        if len(non_io_nodes) <= 1:
            continue
            
        eligible_nodes.append(node)
    
    if not eligible_nodes:
        print("No eligible nodes for removal")
        return individual
    
    # Select a random node to remove
    node_to_remove = random.choice(eligible_nodes)
    print(f"Removing node: {node_to_remove.name}")
    
    # Remove the node
    individual.graph_module, remaining_node = remove_node(individual.graph_module, node_to_remove)
    print(f"Successfully removed node {node_to_remove.name}, remaining node: {remaining_node.name}")
    print("Completed mutation_remove_node")
    return individual

def _get_eligible_nodes(individual, nodes=None):
    if nodes is None:
        nodes = list(individual.graph_module.graph.nodes)
    eligible_nodes = []
    for node in nodes:
        if node.op in ['placeholder', 'output'] or not node_has_shape(node):
            continue
        
        eligible_nodes.append(node)
    return eligible_nodes