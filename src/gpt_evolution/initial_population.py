import random

import torch

from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

from src.nn.core import get_graph
from src.nn.individual import NeuralNetworkIndividual

def generate_initial_population(
    population_size: int,
    vocab_size: int,
    gpt_config_params: dict,
    train_config_params: dict,
):
    """
    Generate an initial population of GPT models with random configurations
    
    Args:
        population_size (int): Number of models to generate
        vocab_size (int): Size of the vocabulary
        block_size (int): Maximum sequence length
        device (str): Device to place models on
        
    Returns:
        list: List of GPT models with random configurations
    """
    population = []
    
    for i in range(population_size):
        print(f"Generating individual {i+1} of {population_size}")
        model_config = create_random_gpt_config(vocab_size, **gpt_config_params)
        train_config = create_random_train_config(**train_config_params)
        print("model_config", model_config)
        print("train_config", train_config)
        example_input = torch.randint(0, model_config.vocab_size, (1, model_config.block_size))
        graph_module = get_graph(GPT(model_config), example_input=example_input)
        population.append(NeuralNetworkIndividual(i, graph_module=graph_module, train_config=train_config))

    return population

def create_random_gpt_config(
    vocab_size: int,
    block_size: int = 128,
    layer_bounds: tuple[int, int] = (3, 12),
    head_bounds: tuple[int, int] = (4, 12),
    embed_bounds: tuple[int, int] = (128, 768),
):
    """
    Create a random GPT configuration within specified bounds
    
    Args:
        vocab_size (int): Size of the vocabulary
        block_size (int): Maximum sequence length
        min_layers (int): Minimum number of transformer layers
        max_layers (int): Maximum number of transformer layers
        min_heads (int): Minimum number of attention heads
        max_heads (int): Maximum number of attention heads
        min_embed (int): Minimum embedding dimension
        max_embed (int): Maximum embedding dimension
    
    Returns:
        CN: Configuration object for GPT model
    """
    config = CN()

    # TODO should we do something like a normal distribution instead of equal probability?
    
    # Ensure n_embd is divisible by n_head
    config.n_head = random.randint(head_bounds[0], head_bounds[1])

    # Round embedding size to nearest multiple of n_head
    config.n_embd = random.randint(embed_bounds[0] // config.n_head, embed_bounds[1] // config.n_head) * config.n_head

    config.n_layer = random.randint(layer_bounds[0], layer_bounds[1])
    config.vocab_size = vocab_size
    config.block_size = block_size
    
    # Random dropout rates between 0.0 and 0.2
    config.embd_pdrop = random.uniform(0.0, 0.2)
    config.resid_pdrop = random.uniform(0.0, 0.2)
    config.attn_pdrop = random.uniform(0.0, 0.2)

    config.model_type = None
    config.is_proxy_for_fx = True
    
    return config

def create_random_train_config(
    batch_size_bounds=(32, 128),
    learning_rate_bounds=(1e-5, 1e-3),
    beta_1_bounds=(0.9, 0.95),
    beta_2_bounds=(0.95, 0.999),
    weight_decay_bounds=(0.0, 0.1),
    grad_norm_clip_bounds=(0.0, 1.0),
    max_iters=5000,
    device='auto',
):
    train_config = CN()
    train_config.device = device
    train_config.batch_size = random.randint(batch_size_bounds[0], batch_size_bounds[1])
    train_config.learning_rate = random.uniform(learning_rate_bounds[0], learning_rate_bounds[1])
    train_config.betas = (
        random.uniform(beta_1_bounds[0], beta_1_bounds[1]),
        random.uniform(beta_2_bounds[0], beta_2_bounds[1])
    )
    train_config.weight_decay = random.uniform(weight_decay_bounds[0], weight_decay_bounds[1])
    train_config.grad_norm_clip = random.uniform(grad_norm_clip_bounds[0], grad_norm_clip_bounds[1])
    train_config.num_workers = 0  # I checked: this takes <0.5% of the total train step time with num_workers=0
    train_config.max_iters = max_iters

    return train_config
