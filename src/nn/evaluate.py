import json
import logging
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

from ..mingpt_altered.trainer import Trainer
from .individual import NeuralNetworkIndividual
from .dataset import HuggingFaceIterableDataset
from .utils import estimate_flops

TOTAL_BATCHES_FOR_EVALUATION = 20

def calculate_fitness(
    individual: NeuralNetworkIndividual,
    iterable_train_dataset,
    iterable_test_dataset, 
    tokenizer,
    block_size: int,
    num_train_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    loss_log_frequency: int = 100,
    max_iter_timeout: float = 20.0,
    secondary_iter_timeout: float = 0.2,
) -> float:
    """
    Calculate fitness of a GPT model by training it and returning negative loss
    (negative because evolution maximizes fitness, but we want to minimize loss)
    
    Args:
        individual: The NeuralNetworkIndividual to evaluate
        iterable_train_dataset: HuggingFace iterable dataset for training. If empty list, skip training.
        iterable_test_dataset: HuggingFace iterable dataset for testing
        tokenizer: Tokenizer for encoding text
        num_train_steps: Number of training steps to perform
        device: Device to train on
        loss_log_frequency: How often to log training loss (every N iterations)
        max_iter_timeout: Maximum seconds per iteration before terminating
        secondary_iter_timeout: Secondary timeout for iterations that are too slow
        
    Returns:
        float: Fitness score (higher is better)
    """
    
    # Compute model FLOPs using ptflops
    max_flops = getattr(individual.train_config, 'max_flops', None)
    batch_size = getattr(individual.train_config, 'batch_size', 1)
    if max_flops is not None:
        example_input = getattr(individual.graph_module, 'example_input', None)
        if example_input is None:
            logging.error("No example_input found in individual.graph_module; cannot compute FLOPs.")
            return float('-inf')
        try:
            flops_per_batch = estimate_flops(individual.graph_module, example_input, batch_size)
        except Exception as e:
            logging.error(f"FLOPs calculation failed: {e}")
            return float('-inf')
        if flops_per_batch == 0:
            logging.error("Model has zero FLOPs, skipping.")
            return float('-inf')
        max_batches = int(max_flops // flops_per_batch)
        if max_batches < 1:
            logging.warning(f"Model exceeds max_flops for a single batch. Skipping training. FLOPs: {flops_per_batch}, max_flops: {max_flops}")
            return float('-inf')
        num_train_steps = int(min(num_train_steps, max_batches))
        individual.train_config.max_iters = num_train_steps
        logging.info(f"Model FLOPs per batch: {flops_per_batch:.2e}, batch size: {batch_size}, max allowed batches: {max_batches}, using {num_train_steps} steps.")

    # Only train if training dataset is provided
    if iterable_train_dataset:
        # Create train dataset
        train_dataset = HuggingFaceIterableDataset(
            iterable_train_dataset,
            tokenizer,
            block_size,
            max_samples=num_train_steps * batch_size  # Strict upper bound
        )
        
        # Run training
        trainer = Trainer(individual.train_config, individual.graph_module, train_dataset)
        def batch_end_callback(trainer):
            # Use timeout values passed from run config
            if trainer.iter_dt > max_iter_timeout:  # if it even has one that's this bad, just kill it
                raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
            if trainer.iter_num % loss_log_frequency == 0:  # Use configurable frequency
                logging.debug(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

                # # TODO better to do some averaging here instead of just checking 1/100
                # What's the point of this?
                if trainer.iter_dt > secondary_iter_timeout:  # Do it here so less likely that a random slow iteration will cause the entire train to fail
                    print("secondary_timeout", secondary_iter_timeout)
                    raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.set_callback('on_train_end', batch_end_callback)
        trainer.run()
    else:
        logging.info("No training data provided - skipping training phase")

    # Calculate perplexity on the validation set
    perplexity = calculate_perplexity(
        individual.graph_module, 
        iterable_test_dataset,
        tokenizer,
        block_size,
        device=device
    )

    individual.graph_module = individual.graph_module.to('cpu')  # Move the model back to CPU, since we're not going to run it again
    if device == 'cuda': torch.cuda.empty_cache()

    fitness = -perplexity  # negative perplexity as fitness (lower perplexity = better) so that we can go uppies :)
    
    return fitness

def calculate_perplexity(
    model: torch.nn.Module,
    iterable_test_dataset,
    tokenizer,
    block_size: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32
) -> float:
    """
    Calculate perplexity of a GPT model on the provided data
    
    Args:
        model: The GPT model to evaluate
        iterable_test_dataset: HuggingFace iterable dataset for testing
        tokenizer: Tokenizer for encoding text
        block_size: Sequence length for the model
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        
    Returns:
        float: Perplexity score (lower is better)
    """
    logging.debug(f"Calculating perplexity in device: {device}")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create test dataset
    test_dataset = HuggingFaceIterableDataset(
        iterable_test_dataset,
        tokenizer,
        block_size,
        max_samples=TOTAL_BATCHES_FOR_EVALUATION * batch_size  # Limit samples for evaluation
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,  # Important for iterable datasets
        pin_memory=True
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= TOTAL_BATCHES_FOR_EVALUATION:
                break
            idx, targets = batch
            idx, targets = idx.to(device), targets.to(device)
            
            # Forward pass
            logits = model(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Accumulate loss (weighted by number of tokens)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    logging.debug(f"avg_loss: {avg_loss}")
    
    # Perplexity is exp(average negative log likelihood)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    logging.debug(f"perplexity: {perplexity}")
    return perplexity
