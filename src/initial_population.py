from mingpt.model import GPT, CN
import random

def generate_initial_population(
    population_size,
    vocab_size,
    block_size,
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
    
    for _ in range(population_size):
        config = _create_random_gpt_config(vocab_size, block_size)
        model = GPT(config)
        population.append(model)
    
    return population

def _create_random_gpt_config(
    vocab_size,
    block_size,
    min_layers=3,
    max_layers=12,
    min_heads=4,
    max_heads=12,
    min_embed=128,
    max_embed=768,
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
    n_head = random.randint(min_heads, max_heads)
    # Round embedding size to nearest multiple of n_head
    n_embd = random.randint(min_embed // n_head, max_embed // n_head) * n_head
    
    config.n_layer = random.randint(min_layers, max_layers)
    config.n_head = n_head
    config.n_embd = n_embd
    config.vocab_size = vocab_size
    config.block_size = block_size
    
    # Random dropout rates between 0.0 and 0.2
    config.embd_pdrop = random.uniform(0.0, 0.2)
    config.resid_pdrop = random.uniform(0.0, 0.2)
    config.attn_pdrop = random.uniform(0.0, 0.2)

    config.model_type = None
    config.is_proxy_for_fx = False
    
    return config
