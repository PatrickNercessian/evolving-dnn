import torch
from mingpt.model import GPT
from mingpt.trainer import Trainer

def calculate_fitness(
    model: GPT,
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
    # Create a copy of the model to avoid changing the original
    config = model.get_default_config()
    config.merge_from_dict({
        'vocab_size': model.transformer.wte.weight.shape[0],
        'block_size': model.block_size,
        'n_layer': len(model.transformer.h),
        'n_head': model.transformer.h[0].attn.n_head,
        'n_embd': model.transformer.h[0].attn.n_embd,
        'is_proxy_for_fx': getattr(model, 'is_proxy_for_fx', False),
        'model_type': None,
    })
    model_copy = GPT(config)
    # Copy the weights
    model_copy.load_state_dict(model.state_dict())

    train_config = Trainer.get_default_config()
    train_config.max_iters = 5000
    train_config.batch_size = 32
    train_config.learning_rate = 3e-4
    train_config.num_workers = 0

    trainer = Trainer(train_config, model_copy, train_data_loader)
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    # Calculate perplexity on the validation set
    perplexity = calculate_perplexity(model_copy, val_data_loader, device=device)

    # Return negative perplexity as fitness (lower perplexity = better)
    return -perplexity

def calculate_perplexity(
    model: GPT,
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
