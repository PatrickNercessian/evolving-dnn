import os
import argparse

import json
import logging

from ..gpt_evolution.initial_population import generate_initial_population
from ..gpt_evolution.helpers import set_random_seeds
from ..nn.evaluate import calculate_fitness
from ..nn.individual import NeuralNetworkIndividual
from ..nn.evolution import NeuralNetworkEvolution
from ..nn.variation.hyperparam_variation import (
    mutate_batch_size, crossover_batch_size,
    mutate_learning_rate, crossover_learning_rate,
    mutate_learning_rate_scheduler, crossover_learning_rate_scheduler,
    mutate_optimizer_parameters, crossover_optimizer_parameters,
)
from ..nn.variation.architecture_mutation import (
    mutation_add_linear, mutation_add_relu, mutation_add_skip_connection,
    mutation_add_branch, mutation_remove_node
)
from ..nn.variation.architecture_crossover import crossover_subgraph

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

VOCAB_SIZE = 2000
RANDOM_SEED = 42

def configure_logger(experiment_path):
    debug_log_file = os.path.join(experiment_path, "evolution_run_debug.log")
    info_log_file = os.path.join(experiment_path, "evolution_run.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Handler for DEBUG and above (all messages) - goes to debug file
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    
    # Handler for WARNING and above only - goes to warnings file
    info_handler = logging.FileHandler(info_log_file)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(console_handler)

    # Silence verbose loggers
    for logger_name in ["urllib3", "datasets", "huggingface_hub", "fsspec"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GPT Evolution experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/gpt_evolution/default_run_config.json",
        help="Path to the run configuration JSON file."
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="./default_experiment_path",
        help="Path to the experiment directory."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        run_config = json.load(f)
    experiment_path = args.experiment_path

    tokenizer_config = run_config["tokenizer"]
    evolution_config = run_config["evolution"]
    training_config = run_config["training"]
    gpt_config = run_config["gpt_config"]

    os.makedirs(experiment_path, exist_ok=True)

    configure_logger(experiment_path)

    set_random_seeds(evolution_config["random_seed"])

    load_dataset_constant_kwargs = {"path": tokenizer_config["dataset"], "name": tokenizer_config["dataset_name"], "streaming": True}
    if "data_files_prefixes" in tokenizer_config:
        suffix = tokenizer_config["data_files_suffix"]
        train_data_files = [f"{prefix}{suffix}" for prefix in tokenizer_config["data_files_prefixes"]["train"]]
        validation_data_files = [f"{prefix}{suffix}" for prefix in tokenizer_config["data_files_prefixes"]["validation"]]
        iterable_train_dataset = load_dataset(**load_dataset_constant_kwargs, split="train", data_dir=tokenizer_config["data_dir"], data_files=train_data_files)
        iterable_validation_dataset = load_dataset(**load_dataset_constant_kwargs, split="train", data_dir=tokenizer_config["data_dir"], data_files=validation_data_files)
    else:
        datasets = load_dataset(**load_dataset_constant_kwargs)
        iterable_train_dataset = datasets["train"]
        iterable_validation_dataset = datasets["validation"]

    tokenizer_filepath = os.path.join(experiment_path, tokenizer_config["tokenizer_filename"])
    if os.path.exists(tokenizer_filepath):
        logging.info("Loading tokenizer from file")
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        def text_generator():
            count = 0
            total_samples = tokenizer_config.get("tokenizer_training_samples", 10000)  # Default to 10k samples
            for example in iterable_train_dataset:
                if count >= total_samples:
                    break
                yield example["text"]
                count += 1
        
        tokenizer.train_from_iterator(text_generator(), trainer=BpeTrainer(vocab_size=tokenizer_config["vocab_size"]))
        tokenizer.save(tokenizer_filepath)

    train_config_params = {
        "max_iters": training_config["max_iters"],
        "device": training_config["device"],
    }

    # Create a wrapper for calculate_fitness that only takes individual
    def fitness_wrapper(individual: NeuralNetworkIndividual) -> float:
        return calculate_fitness(
            individual,
            iterable_train_dataset,
            iterable_validation_dataset,
            tokenizer,
            block_size=gpt_config["block_size"],
            device=train_config_params["device"],
        )

    evolution = NeuralNetworkEvolution(
        population=generate_initial_population(
            evolution_config["target_population_size"],
            tokenizer_config["vocab_size"],
            gpt_config,
            train_config_params,
        ),
        fitness_fn=fitness_wrapper,
        crossover_instead_of_mutation_rate=evolution_config["crossover_instead_of_mutation_rate"],
        mutation_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["mutation_probabilities"].items()
        ],
        crossover_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["crossover_probabilities"].items()
        ],
        target_population_size=evolution_config["target_population_size"],
        num_children_per_generation=evolution_config["num_children_per_generation"],
        experiment_path=experiment_path
    )
    evolution.run_evolution(evolution_config["num_generations"])
