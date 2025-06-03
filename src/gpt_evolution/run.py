import os
import argparse

import json
import logging

from src.gpt_evolution.initial_population import generate_initial_population
from src.gpt_evolution.helpers import set_random_seeds
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
from src.nn.variation.architecture_mutation import (
    mutation_add_linear, mutation_add_relu, mutation_add_skip_connection,
    mutation_add_branch, mutation_remove_node
)
from src.nn.variation.architecture_crossover import crossover_subgraph

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

VOCAB_SIZE = 2000
RANDOM_SEED = 42

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GPT Evolution experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/gpt_evolution/default_run_config.json",
        help="Path to the run configuration JSON file."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        run_config = json.load(f)

    tokenizer_config = run_config["tokenizer"]
    evolution_config = run_config["evolution"]
    training_config = run_config["training"]
    gpt_config = run_config["gpt_config"]

    experiment_path = evolution_config["experiment_path"]
    os.makedirs(experiment_path, exist_ok=True)

    log_file = os.path.join(experiment_path, "evolution_run.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    set_random_seeds(evolution_config["random_seed"])

    if os.path.exists(tokenizer_config["tokenizer_file"]):
        tokenizer = Tokenizer.from_file(tokenizer_config["tokenizer_file"])
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([tokenizer_config["input_file"]], trainer=BpeTrainer(vocab_size=tokenizer_config["vocab_size"]))
        tokenizer.save(tokenizer_config["tokenizer_file"])

    with open(tokenizer_config["input_file"], 'r', encoding='utf-8') as f:
        text = f.read()
    encoded_text = tokenizer.encode(text)
    data = torch.tensor(encoded_text.ids)

    train_dataset = TextDataset(data, gpt_config["block_size"])

    TARGET_POPULATION_SIZE = 5
    NUM_CHILDREN_PER_GENERATION = 5

    train_config_params = {
        "max_iters": training_config["max_iters"],
        "device": training_config["device"],
    }

    val_loader = torch.utils.data.DataLoader(
        train_dataset,  # Using same dataset for validation for now
        batch_size=training_config["validation_batch_size"],
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

    evolution = NeuralNetworkEvolution(
        population=generate_initial_population(
            TARGET_POPULATION_SIZE,
            VOCAB_SIZE,
            gpt_config,
            train_config_params,
        ),
        fitness_fn=fitness_wrapper,  # Now only takes individual as parameter
        crossover_instead_of_mutation_rate=evolution_config["crossover_instead_of_mutation_rate"],
        mutation_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["mutation_probabilities"].items()
        ],
        crossover_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["crossover_probabilities"].items()
        ],
        target_population_size=evolution_config["target_population_size"],
        num_children_per_generation=evolution_config["num_children_per_generation"],
        experiment_individuals_path=f"{experiment_path}/individuals",
    )
    evolution.run_evolution(evolution_config["num_generations"])
