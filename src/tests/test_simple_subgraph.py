import copy
import torch
import torch.nn as nn
from src.core import get_graph
from src.variation.architecture_crossover import random_subgraph, find_subgraph_connections, insert_subgraph

# Slightly more complex model for testing
class MoreComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial feature extraction layers
        self.fc1 = nn.Linear(20, 32)
        self.relu1 = nn.ReLU()
        
        # First residual block
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        
        # Second residual block
        self.fc4 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        # Final processing layers
        self.fc5 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(8, 4)
        self.relu4 = nn.ReLU()
        
        # Output layers with skip connections
        self.fc7 = nn.Linear(4, 2)
        self.skip1 = nn.Linear(4, 2)  # Skip from first block
        self.skip2 = nn.Linear(32, 2)  # Skip from second block
        self.skip3 = nn.Linear(4, 2)   # Skip from final processing

    def forward(self, x):
        # Initial feature extraction
        out = self.relu1(self.fc1(x))
        
        # First residual block
        residual1 = self.fc2(out)
        residual1 = self.dropout1(residual1)
        residual1 = self.fc3(residual1)
        residual1 = self.relu2(residual1)
        
        # Second residual block
        residual2 = self.fc4(residual1)
        residual2 = self.relu3(residual2)
        
        # Final processing
        out = self.fc5(residual2)
        out = self.dropout2(out)
        out = self.fc6(out)
        out = self.relu4(out)
        
        # Combine outputs with skip connections
        main_out = self.fc7(out)
        skip1_out = self.skip1(out)
        skip2_out = self.skip2(residual1)
        skip3_out = self.skip3(out)
        
        return main_out + skip1_out + skip2_out + skip3_out


def make_simple_model():
    return MoreComplexModel()

def test_subgraph_functions():
    # Create two simple models and their graphs
    model1 = make_simple_model()
    model2 = make_simple_model()
    example_input = torch.randn(1, 20)
    # Test example input on model1
    # print("Testing example input on model1:")
    # output = model1(example_input)
    # print(f"Output: {output}")

    # graph1 = get_graph(model1, example_input=example_input)
    # graph2 = get_graph(model2, example_input=example_input)

    # # Test random_subgraph
    # num_nodes = 3
    # subgraph_nodes, input_boundary_nodes, output_boundary_nodes = random_subgraph(graph1, num_nodes)
    # print(f"Subgraph nodes: {len(subgraph_nodes)}")
    # print(f"Input boundary nodes: {len(input_boundary_nodes)}")
    # print(f"Output boundary nodes: {len(output_boundary_nodes)}")

    # # Test find_subgraph_connections
    # input_mapping, topo_target_input_nodes, output_mapping = find_subgraph_connections(
    #     graph2.graph, input_boundary_nodes, output_boundary_nodes
    # )
    # print("Input mapping:", input_mapping)
    # print("Topo target input nodes:", topo_target_input_nodes)
    # print("Output mapping:", output_mapping)

    # # Test insert_subgraph
    # graph2_mod, new_node_names = insert_subgraph(
    #     graph2, subgraph_nodes, input_mapping, topo_target_input_nodes, output_mapping
    # )
    # print(f"Inserted subgraph. New node names: {new_node_names}")
    # print("Test completed successfully.")

    # # Test forward pass on the modified graph
    # print("Testing forward pass on modified graph:")
    # output = graph2_mod(example_input)
    # print(f"Forward pass output: {output}")

    # Minimal evolution framework test
    from src.individual import Individual
    from src.evolution import Evolution
    from mingpt.utils import CfgNode as CN
    from src.variation.hyperparam_variation import mutate_learning_rate
    from src.variation.architecture_crossover import crossover_subgraph

    def simple_fitness(ind):
        # Fitness is negative sum of output for a fixed input (just for demonstration)
        with torch.no_grad():
            out = ind.graph_module(example_input)
            return -out.sum().item()

    graph1 = get_graph(model1, example_input=example_input)

    # Create a small population of Individuals
    pop = []
    for i in range(4):
        train_config = CN()
        train_config.learning_rate = 0.01
        # Create individual with graph2_mod deepcopy
        pop.append(Individual(copy.deepcopy(graph1), train_config, i))

    evo = Evolution(
        population=pop,
        fitness_fn=simple_fitness,
        crossover_instead_of_mutation_rate=1.0,  # Only crossover
        mutation_fns_and_probabilities=[],
        crossover_fns_and_probabilities=[(crossover_subgraph, 1.0)],
        target_population_size=4,
        num_children_per_generation=2,
        block_size=8,
    )
    generations = 5
    evo.run_evolution(generations)
    print(f"Best fitness after {generations} generations: {evo.best_fitness}")
    print(f"Best individual output: {evo.best_individual.graph_module(example_input)}")

    # Print top 3 individuals and their fitness
    print("\nTop 3 individuals after evolution:")
    top3 = sorted(evo.population, key=lambda ind: ind.fitness, reverse=True)[:3]
    for idx, ind in enumerate(top3, 1):
        print(f"\nRank {idx} | Fitness: {ind.fitness}")
        print("Graph structure (node names and types):")
        for node in ind.graph_module.graph.nodes:
            if node.op == 'call_module':
                module = getattr(ind.graph_module, node.target)
                print(f"  {node.name}: {type(module).__name__}")
            elif node.op == 'call_function':
                print(f"  {node.name}: {getattr(node.target, '__name__', str(node.target))}")
            elif node.op == 'placeholder':
                print(f"  {node.name}: Input")
            elif node.op == 'output':
                print(f"  {node.name}: Output")

if __name__ == "__main__":
    test_subgraph_functions() 