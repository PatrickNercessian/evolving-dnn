import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TODO remove above

from src.bpe import tokenize_string, VOCAB_SIZE
from src.initial_population import generate_initial_population
from src.evaluate import calculate_fitness
from src.individual import Individual
from src.evolution import Evolution
from src.dataset import TextDataset
from src.hyperparam_variation import (
    mutate_batch_size, crossover_batch_size,
    mutate_learning_rate, crossover_learning_rate,
    mutate_learning_rate_scheduler, crossover_learning_rate_scheduler,
    mutate_optimizer_parameters, crossover_optimizer_parameters,
)
from src.subgraph import crossover_subgraph

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

    TARGET_POPULATION_SIZE = 2
    NUM_CHILDREN_PER_GENERATION = 2

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
    def fitness_wrapper(individual: Individual) -> float:
        return calculate_fitness(
            individual,
            train_dataset,
            val_loader,
            device=train_config_params["device"],
        )

    evolution = Evolution(
        population=generate_initial_population(
            TARGET_POPULATION_SIZE+NUM_CHILDREN_PER_GENERATION,  # start as if we have a bunch of children in order to perform selection
            VOCAB_SIZE,
            gpt_config_params,
            train_config_params,
        ),
        fitness_fn=fitness_wrapper,  # Now only takes individual as parameter
        crossover_instead_of_mutation_rate=1.0,
        mutation_fns_and_probabilities=[
            (mutate_batch_size, 0.3),
            (mutate_learning_rate, 0.3),
            (mutate_learning_rate_scheduler, 0.3),
            (mutate_optimizer_parameters, 0.3),
        ],
        crossover_fns_and_probabilities=[
            (crossover_subgraph, 1.0),
            (crossover_batch_size, 0.0),
            (crossover_learning_rate, 0.0),
            (crossover_learning_rate_scheduler, 0.0),
            (crossover_optimizer_parameters, 0.0),
        ],
        target_population_size=TARGET_POPULATION_SIZE,
        num_children_per_generation=NUM_CHILDREN_PER_GENERATION,
        block_size=BLOCK_SIZE  # TODO this shouldn't be part of the evolution class, not abstracted enough
    )
    evolution.run_evolution(10)
