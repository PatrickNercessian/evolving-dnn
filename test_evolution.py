import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Imports completed")

from src.evolution import Evolution
from src.individual import Individual
from src.core import add_node, get_graph
from src.individual_graph_module import IndividualGraphModule
from mingpt.utils import CfgNode as CN

print("Custom imports completed")

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

print("Seeds set")

# Define a simple toy model
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

print("SimpleModel defined")

# Create a synthetic dataset (XOR problem)
def create_xor_dataset(num_samples=1000):
    X = np.random.randint(0, 2, (num_samples, 2)).astype(np.float32)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32).reshape(-1, 1)
    return torch.tensor(X), torch.tensor(y)

print("XOR dataset function defined")

# Create a weather dataset
def create_weather_dataset():
    print("Loading weather data...")
    # Read the CSV file
    df = pd.read_csv('weatherHistory.csv')
    
    # Select features and target
    # We'll predict temperature based on other features
    features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
    target = 'Temperature (C)'
    
    # Drop any rows with missing values
    df = df.dropna(subset=features + [target])
    
    # Create feature matrix X and target vector y
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    
    # Scale the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X)
    y = torch.tensor(y)
    
    return X, y, scaler_y

print("Weather dataset function defined")

# Define mutation functions
def mutation_add_linear(individual):
    print("Starting mutation_add_linear")
    try:
        # Find a random node in the graph to add a linear layer after
        nodes = list(individual.graph_module.graph.nodes)
        eligible_nodes = [n for n in nodes if n.op != 'output' and n.op != 'placeholder']
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
        try:
            individual.graph_module = add_node(individual.graph_module, reference_node, 'linear', 
                                             input_size=input_size, output_size=output_size)
        except Exception as e:
            print(f"Error adding linear layer: {e}")
            return individual
        
        # Adjust training config (you could add learning rate mutation here)
        if random.random() < 0.3:
            individual.train_config.learning_rate *= random.uniform(0.5, 1.5)
            individual.train_config.learning_rate = max(0.0001, min(0.1, individual.train_config.learning_rate))
        
        print("Completed mutation_add_linear")
    except Exception as e:
        print(f"Unexpected error in mutation_add_linear: {e}")
        traceback.print_exc()
    return individual

def mutation_add_relu(individual):
    print("Starting mutation_add_relu")
    try:
        # Find a random node in the graph to add a ReLU layer after
        nodes = list(individual.graph_module.graph.nodes)
        eligible_nodes = [n for n in nodes if n.op != 'output' and n.op != 'placeholder']
        if not eligible_nodes:
            print("No eligible nodes for adding ReLU")
            return individual
        
        reference_node = random.choice(eligible_nodes)
        print(f"Adding ReLU after {reference_node.name}")
        
        # Add a ReLU layer
        try:
            individual.graph_module = add_node(individual.graph_module, reference_node, 'relu')
            print("Completed mutation_add_relu")
        except Exception as e:
            print(f"Error adding ReLU: {e}")
    except Exception as e:
        print(f"Unexpected error in mutation_add_relu: {e}")
        traceback.print_exc()
    return individual

def mutation_add_skip_connection(individual):
    print("Starting mutation_add_skip_connection")
    try:
        # Find two random nodes in the graph to connect
        nodes = list(individual.graph_module.graph.nodes)
        eligible_nodes = [n for n in nodes if n.op != 'output' and n.op != 'placeholder']
        
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
        try:
            individual.graph_module = add_node(individual.graph_module, second_node, 'skip', first_node=first_node)
            print("Completed mutation_add_skip_connection")
        except Exception as e:
            print(f"Error adding skip connection: {e}")
            
    except Exception as e:
        print(f"Unexpected error in mutation_add_skip_connection: {e}")
        traceback.print_exc()
    return individual

def mutation_add_branch(individual):
    print("Starting mutation_add_branch")
    try:
        # Find a random node in the graph to add branches after
        nodes = list(individual.graph_module.graph.nodes)
        eligible_nodes = [n for n in nodes if n.op != 'output' and n.op != 'placeholder']
        
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
        try:
            individual.graph_module = add_node(individual.graph_module, reference_node, 'branch',
                                             branch1_out_size=branch1_out_size,
                                             branch2_out_size=branch2_out_size)
            print("Completed mutation_add_branch")
        except Exception as e:
            print(f"Error adding branch: {e}")
            
    except Exception as e:
        print(f"Unexpected error in mutation_add_branch: {e}")
        traceback.print_exc()
    return individual

print("Mutation functions defined")

# Fitness function - evaluate model performance
def fitness_function(individual):
    print(f"Evaluating fitness for individual {individual.id}")
    model = individual.graph_module
    
    try:
        X, y, scaler_y = create_weather_dataset()  # Get weather data
        
        criterion = nn.MSELoss()  # Use MSE loss for regression
        optimizer = optim.Adam(model.parameters(), lr=individual.train_config.learning_rate)
        
        # Mini training loop
        model.train()
        batch_size = individual.train_config.batch_size
        num_epochs = individual.train_config.num_epochs
        
        for epoch in range(num_epochs):
            epoch_losses = []
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            print(f"Individual {individual.id}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            mse = criterion(outputs, y).item()
            r2 = 1 - mse / torch.var(y).item()  # Calculate R² score
        
        print(f"Fitness (R² score) for individual {individual.id}: {r2}")
        return r2  # Return R² score as fitness
        
    except Exception as e:
        print(f"\nError during fitness evaluation for individual {individual.id}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nModel graph structure:")
        print(model.graph.print_tabular())
        print("\nFull stack trace:")
        traceback.print_exc()
        raise

print("Fitness function defined")

# Create initial population
def create_initial_population(pop_size=5):
    print(f"Creating initial population of size {pop_size}")
    population = []
    
    for i in range(pop_size):
        try:
            print(f"Creating individual {i}")
            # Create base model with parameters for weather prediction
            input_size = 5  # Number of weather features
            hidden_size = random.randint(8, 32)
            output_size = 1  # Temperature prediction
            
            model = SimpleModel(input_size, hidden_size, output_size)
            
            # Create graph module
            block_size = 128  # Not really used for this model but required
            print(f"Creating graph for individual {i}")
            graph_module = get_graph(model, input_shape=(32, input_size))  # Batch size of 32
            
            # Create training config
            train_config = CN()
            train_config.learning_rate = random.uniform(0.0001, 0.001)  # Smaller learning rates for stability
            train_config.batch_size = random.choice([32, 64, 128])  # Larger batches for stability
            train_config.num_epochs = 5  # Keep small for quick testing
            
            # Create individual
            individual = Individual(graph_module, train_config, i)
            population.append(individual)
            print(f"Individual {i} created")
        except Exception as e:
            print(f"Error creating individual {i}: {e}")
            traceback.print_exc()
            # Try again with a simpler model
            try:
                model = SimpleModel(5, 16, 1)  # Use fixed size for stability
                graph_module = get_graph(model, input_shape=(32, 5))
                train_config = CN()
                train_config.learning_rate = 0.0005
                train_config.batch_size = 64
                train_config.num_epochs = 5
                individual = Individual(graph_module, train_config, i)
                population.append(individual)
                print(f"Created fallback individual {i}")
            except Exception as e2:
                print(f"Failed to create fallback individual {i}: {e2}")
    
    print(f"Initial population creation complete with {len(population)} individuals")
    return population

print("Initial population function defined")

def print_population_graphs(population, generation=None):
    """Print the graph structure of all individuals in the population, including tensor shapes"""
    gen_str = f" in Generation {generation}" if generation is not None else ""
    print(f"\n=== Printing graphs for all individuals{gen_str} ===")
    for individual in population:
        print(f"\nIndividual {individual.id} Graph Structure:")
        print("----------------------------------------")
        for node in individual.graph_module.graph.nodes:
            shape_info = f" -> Shape: {node.meta['tensor_meta'].shape}" if 'meta' in node.__dict__ and 'tensor_meta' in node.meta else ""
            if node.op == 'placeholder':
                print(f"Input: {node.name}{shape_info}")
            elif node.op == 'output':
                print(f"Output: {node.name}{shape_info}")
            elif node.op == 'call_module':
                module = getattr(individual.graph_module, node.target)
                print(f"Layer: {node.name} ({type(module).__name__}){shape_info}")
            elif node.op == 'call_function':
                print(f"Operation: {node.name} ({node.target.__name__}){shape_info}")
            else:
                print(f"Node: {node.name} ({node.op}){shape_info}")
        print("----------------------------------------")

def test_evolution():
    print("Starting test_evolution function")
    print("Creating initial population...")
    population = create_initial_population(pop_size=2)
    
    if not population:
        print("Failed to create initial population. Exiting test.")
        return
    
    print("Setting up evolution...")
    # Setup evolution
    evolution = Evolution(
        population=population,
        fitness_fn=fitness_function,
        crossover_instead_of_mutation_rate=0.0,  # Disable crossover
        mutation_fns_and_probabilities=[
            (mutation_add_linear, 0.4),
            (mutation_add_relu, 0.2),
            (mutation_add_skip_connection, 0.2),
            (mutation_add_branch, 0.2),
        ],
        crossover_fns_and_probabilities=[],  # Empty crossover functions
        target_population_size=5,
        num_children_per_generation=3,
        block_size=128,
    )
    
    # Run evolution for a few generations
    print("Starting evolution...")
    try:
        evolution.run_evolution(num_generations=5)
        
        print("\nEvolution completed!")
        print(f"Best fitness: {evolution.best_fitness}")
        
        print("\n=== Final Population Graphs ===")
        print_population_graphs(evolution.population)
        
        if evolution.best_individual:
            print(f"Best individual ID: {evolution.best_individual.id}")
            print(f"Best config: {evolution.best_individual.train_config}")
            
            # Print the model architecture with shapes
            print("\nBest Model Architecture:")
            print("------------------------")
            for node in evolution.best_individual.graph_module.graph.nodes:
                if node.op == 'placeholder':
                    print(f"Input: {node.name} -> Shape: {node.meta['tensor_meta'].shape}")
                elif node.op == 'output':
                    print(f"Output: {node.name} -> Shape: {node.meta['tensor_meta'].shape}")
                elif node.op == 'call_module':
                    module = getattr(evolution.best_individual.graph_module, node.target)
                    print(f"Layer: {node.name} ({type(module).__name__}) -> Shape: {node.meta['tensor_meta'].shape}")
                elif node.op == 'call_function':
                    print(f"Operation: {node.name} ({node.target.__name__}) -> Shape: {node.meta['tensor_meta'].shape}")
        else:
            print("No best individual found.")
    except Exception as e:
        print(f"Error during evolution: {e}")
        traceback.print_exc()
    
    print("Test evolution function complete")

def test_base_model():
    """Test that the base model can learn weather prediction without evolution"""
    print("\n=== Testing Base Model ===")
    # Create a model with fixed parameters
    model = SimpleModel(input_size=5, hidden_size=16, output_size=1)  # 5 weather features -> 1 temperature prediction
    print(f"Created model: {model}")
    
    # Create dataset
    X, y, scaler_y = create_weather_dataset()
    print(f"Created dataset with {len(X)} samples")
    
    # Train the model
    criterion = nn.MSELoss()  # Use MSE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Mini training loop
    model.train()
    batch_size = 64
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        mse = criterion(outputs, y).item()
        r2 = 1 - mse / torch.var(y).item()
    
    print(f"Final R² score: {r2:.4f}")
    print("=== Base Model Test Complete ===\n")
    return r2 > 0.5  # Return True if model performs reasonably well

if __name__ == "__main__":
    print("Script started")
    try:
        test_evolution()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
    print("Script completed") 