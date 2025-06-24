import logging

import numpy as np
from ..individual import NeuralNetworkIndividual

def mutate_batch_size(individual: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Mutate batch size using a normal distribution centered on current value.
    Ensures the result stays within reasonable bounds.
    """
    current_batch_size = individual.train_config.batch_size
    # Standard deviation of 20% of current value
    std_dev = max(current_batch_size * 0.2, 1)
    new_batch_size = int(np.random.normal(current_batch_size, std_dev))
    # Ensure batch size stays within bounds (32, 128) and is even
    new_batch_size = max(32, min(128, new_batch_size))
    new_batch_size = (new_batch_size // 2) * 2  # Make even

    logging.debug(f"Mutated batch size from {current_batch_size} to {new_batch_size}")
    individual.train_config.batch_size = new_batch_size

def mutate_learning_rate(individual: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Mutate learning rate using a log-normal distribution to ensure positive values.
    """
    current_lr = individual.train_config.learning_rate
    # Use multiplicative noise with sigma = 0.2 (20% variation)
    log_lr = np.log(current_lr)
    new_lr = np.exp(np.random.normal(log_lr, 0.2))
    # Ensure learning rate stays within bounds (1e-5, 1e-3)
    new_lr = max(1e-5, min(1e-3, new_lr))

    logging.debug(f"Mutated learning rate from {current_lr} to {new_lr}")
    individual.train_config.learning_rate = float(new_lr)

def mutate_learning_rate_scheduler(individual: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Mutate learning rate scheduler parameters (beta1 and beta2).
    """
    current_beta1, current_beta2 = individual.train_config.betas
    
    # Mutate beta1 with small normal variation
    new_beta1 = np.random.normal(current_beta1, 0.01)
    new_beta1 = max(0.9, min(0.95, new_beta1))
    
    # Mutate beta2 with small normal variation
    new_beta2 = np.random.normal(current_beta2, 0.01)
    new_beta2 = max(0.95, min(0.999, new_beta2))
    
    logging.debug(f"Mutated learning rate scheduler betas for beta1 {current_beta1} and beta2 {current_beta2} to beta1 {new_beta1} and beta2 {new_beta2}")
    individual.train_config.betas = (float(new_beta1), float(new_beta2))

def mutate_optimizer_parameters(individual: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Mutate optimizer-related parameters (weight_decay and grad_norm_clip).
    """
    # Mutate weight decay
    current_wd = individual.train_config.weight_decay
    new_wd = np.random.normal(current_wd, 0.01)
    new_wd = max(0.0, min(0.1, new_wd))
    individual.train_config.weight_decay = float(new_wd)
    
    # Mutate gradient clipping
    current_clip = individual.train_config.grad_norm_clip
    new_clip = np.random.normal(current_clip, 0.1)
    new_clip = max(0.0, min(1.0, new_clip))

    logging.debug(f"Mutated optimizer parameters for weight_decay {current_wd} and grad_norm_clip {current_clip} to weight_decay {new_wd} and grad_norm_clip {new_clip}")
    individual.train_config.grad_norm_clip = float(new_clip)

def crossover_batch_size(parent1_copy: NeuralNetworkIndividual, parent2: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Average the batch size from two parents to create a child.
    """
    new_batch_size = int(0.5 * (parent1_copy.train_config.batch_size + parent2.train_config.batch_size))
    logging.debug(f"Crossover batch size from {parent1_copy.train_config.batch_size} and {parent2.train_config.batch_size} to {new_batch_size}")
    parent1_copy.train_config.batch_size = new_batch_size

def crossover_learning_rate(parent1_copy: NeuralNetworkIndividual, parent2: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Average the learning rate from two parents to create a child.
    """
    new_lr = 0.5 * (parent1_copy.train_config.learning_rate + parent2.train_config.learning_rate)
    logging.debug(f"Crossover learning rate from {parent1_copy.train_config.learning_rate} and {parent2.train_config.learning_rate} to {new_lr}")
    parent1_copy.train_config.learning_rate = new_lr

def crossover_learning_rate_scheduler(parent1_copy: NeuralNetworkIndividual, parent2: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Average the learning rate scheduler parameters from two parents to create a child.
    """
    new_beta1 = 0.5 * (parent1_copy.train_config.betas[0] + parent2.train_config.betas[0])
    new_beta2 = 0.5 * (parent1_copy.train_config.betas[1] + parent2.train_config.betas[1])
    logging.debug(f"Crossover learning rate scheduler betas for beta1s {parent1_copy.train_config.betas[0]} and {parent2.train_config.betas[0]} and beta2s {parent1_copy.train_config.betas[1]} and {parent2.train_config.betas[1]} to beta1 {new_beta1} and beta2 {new_beta2}")
    parent1_copy.train_config.betas = (new_beta1, new_beta2)

def crossover_optimizer_parameters(parent1_copy: NeuralNetworkIndividual, parent2: NeuralNetworkIndividual, **kwargs) -> NeuralNetworkIndividual:
    """
    Average the optimizer parameters from two parents to create a child.
    """
    new_wd = 0.5 * (parent1_copy.train_config.weight_decay + parent2.train_config.weight_decay)
    new_clip = 0.5 * (parent1_copy.train_config.grad_norm_clip + parent2.train_config.grad_norm_clip)
    logging.debug(f"Crossover optimizer parameters for weight_decay {parent1_copy.train_config.weight_decay} and {parent2.train_config.weight_decay} and grad_norm_clip {parent1_copy.train_config.grad_norm_clip} and {parent2.train_config.grad_norm_clip} to weight_decay {new_wd} and grad_norm_clip {new_clip}")
    parent1_copy.train_config.weight_decay = new_wd
    parent1_copy.train_config.grad_norm_clip = new_clip
