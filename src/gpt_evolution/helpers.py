import random

import numpy as np
import torch

def set_random_seeds(seed: int):
    """Set random seeds for all random number generators used in the project"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA operations
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True  # TODO revisit these for full runs
    # torch.backends.cudnn.benchmark = False