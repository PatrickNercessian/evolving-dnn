import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TODO remove above

from src.nn.bpe import tokenize_string, VOCAB_SIZE
from src.gpt_evolution.initial_population import generate_initial_population
from src.nn.evaluate import calculate_fitness
from src.nn.individual import NeuralNetworkIndividual
from src.nn.evolution import NeuralNetworkEvolution
from src.nn.dataset import TextDataset
from src.nn.variation.hyperparam_variation import (
    mutate_batch_size, crossover_batch_size,
    mutate_learning_rate, crossover_learning_rate,
    mutate_learning_rate_scheduler, crossover_learning_rate_scheduler,
    mutate_optimizer_parameters, crossover_optimizer_parameters,
)
from src.nn.variation.architecture_mutation import mutation_add_linear, mutation_add_relu, mutation_add_skip_connection, mutation_add_branch, mutation_remove_node
from src.nn.variation.architecture_crossover import crossover_subgraph

import torch
import os

if __name__ == '__main__':
    print("Starting GPT evolution run")

    if os.path.exists('tokenized_data.pt'):
        data = torch.load('tokenized_data.pt')  # Save the tokenized data tensor to a file
    else:
        # Load, tokenize and save the input text
        with open('mingpt/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        data = tokenize_string(text)
        torch.save(data, 'tokenized_data.pt')

    BLOCK_SIZE = 128

    train_dataset = TextDataset(data, BLOCK_SIZE)

    TARGET_POPULATION_SIZE = 5
    NUM_CHILDREN_PER_GENERATION = 5

    gpt_config_params = {
        "block_size": BLOCK_SIZE,
        "layer_bounds": (2, 5),
        "head_bounds": (2, 5),
        "embed_bounds": (128, 512),
    }
    train_config_params = { "max_iters": 1, "device": "cpu" }

    val_loader = torch.utils.data.DataLoader(
        train_dataset,  # Using same dataset for validation for now
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    # Create a wrapper for calculate_fitness that only takes individual
    def fitness_wrapper(individual: NeuralNetworkIndividual) -> float:
        return calculate_fitness(
            individual,
            train_dataset,
            val_loader,
            device=train_config_params["device"],
        )
    
    EXPERIMENT_PATH = "experiments/test4"

    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    evolution = NeuralNetworkEvolution(
        population=generate_initial_population(
            TARGET_POPULATION_SIZE+NUM_CHILDREN_PER_GENERATION,  # start as if we have a bunch of children in order to perform selection
            VOCAB_SIZE,
            gpt_config_params,
            train_config_params,
        ),
        fitness_fn=fitness_wrapper,  # Now only takes individual as parameter
        crossover_instead_of_mutation_rate=0.5,
        mutation_fns_and_probabilities=[
            (mutate_batch_size, 0.2),
            (mutate_learning_rate, 0.2),
            (mutate_learning_rate_scheduler, 0.2),
            (mutate_optimizer_parameters, 0.2),
            (mutation_add_linear, 0.2),
            (mutation_add_relu, 0.2),
            (mutation_add_skip_connection, 0.2),
            (mutation_add_branch, 0.2),
            (mutation_remove_node, 0.2),

        ],
        crossover_fns_and_probabilities=[
            (crossover_subgraph, 0.3),
            (crossover_batch_size, 0.3),
            (crossover_learning_rate, 0.3),
            (crossover_learning_rate_scheduler, 0.3),
            (crossover_optimizer_parameters, 0.3),
        ],
        target_population_size=TARGET_POPULATION_SIZE,
        num_children_per_generation=NUM_CHILDREN_PER_GENERATION,
        experiment_path=EXPERIMENT_PATH,
    )
    evolution.run_evolution(10)
