import copy
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..mingpt_altered.trainer import Trainer
from .individual import NeuralNetworkIndividual
from .dataset import HuggingFaceIterableDataset

TOTAL_BATCHES_FOR_EVALUATION = 20

def calculate_fitness(
    individual: NeuralNetworkIndividual,
    iterable_train_dataset,
    iterable_test_dataset, 
    tokenizer,
    block_size: int,
    num_train_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Calculate fitness of a GPT model by training it and returning negative loss
    (negative because evolution maximizes fitness, but we want to minimize loss)
    
    Args:
        individual: The NeuralNetworkIndividual to evaluate
        iterable_train_dataset: HuggingFace iterable dataset for training
        iterable_test_dataset: HuggingFace iterable dataset for testing
        tokenizer: Tokenizer for encoding text
        num_train_steps: Number of training steps to perform
        device: Device to train on
        
    Returns:
        float: Fitness score (higher is better)
    """
        
    # Create train dataset
    train_dataset = HuggingFaceIterableDataset(
        iterable_train_dataset,
        tokenizer,
        block_size,
        max_samples=num_train_steps * individual.train_config.batch_size * 2  # Provide enough samples
    )
    
    # Run training
    trainer = Trainer(individual.train_config, individual.graph_module, train_dataset)
    def batch_end_callback(trainer):
        # TODO need to pick max iter_dt based on the model sizes.
        if trainer.iter_dt > 20000:  # if it even has one that's this bad, just kill it
            raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
        if trainer.iter_num % 100 == 0:
            logging.debug(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            # TODO better to do some averaging here instead of just checking 1/100
            if trainer.iter_dt > 200:  # Do it here so less likely that a random slow iteration will cause the entire train to fail
                raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_train_end', batch_end_callback)
    trainer.run()

    # Calculate perplexity on the validation set
    perplexity = calculate_perplexity(
        individual.graph_module, 
        iterable_test_dataset,
        tokenizer,
        block_size,
        device=device
    )

    original_device = next(individual.graph_module.parameters()).device if list(individual.graph_module.parameters()) else torch.device('cpu')
    if original_device.type == 'cpu':
        individual.graph_module.to('cpu')  # Move the model back to CPU, since we're not going to run it again

    # Return negative perplexity as fitness (lower perplexity = better)
    return -perplexity

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
