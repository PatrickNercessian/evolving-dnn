import torch
from mingpt.trainer import Trainer

from .individual import Individual

def calculate_fitness(
    individual: Individual,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    num_train_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Calculate fitness of a GPT model by training it and returning negative loss
    (negative because evolution maximizes fitness, but we want to minimize loss)
    
    Args:
        model: The GPT model to evaluate
        data_loader: DataLoader providing training examples
        num_train_steps: Number of training steps to perform
        device: Device to train on
        
    Returns:
        float: Fitness score (higher is better)
    """
    trainer = Trainer(individual.train_config, individual.graph_module, train_data_loader)
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    # Calculate perplexity on the validation set
    perplexity = calculate_perplexity(individual.graph_module, val_data_loader, device=device)

    # Return negative perplexity as fitness (lower perplexity = better)
    return -perplexity

def calculate_perplexity(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Calculate perplexity of a GPT model on the provided data
    
    Args:
        model: The GPT model to evaluate
        data_loader: DataLoader providing examples (inputs and targets)
        device: Device to evaluate on
        
    Returns:
        float: Perplexity score (lower is better)
    """
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    total_tokens = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch in data_loader:
            idx, targets = batch
            idx, targets = idx.to(device), targets.to(device)
            
            # Forward pass
            logits, loss = model(idx, targets)
            
            # Accumulate loss (weighted by number of tokens)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Perplexity is exp(average negative log likelihood)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity
