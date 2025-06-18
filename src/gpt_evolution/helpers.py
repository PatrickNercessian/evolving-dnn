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

def deep_merge_dicts(default_dict, override_dict):
    """
    Recursively merge two dictionaries, where override_dict values
    take precedence over default_dict values at the leaf level.
    
    Args:
        default_dict: The base dictionary with default values
        override_dict: The dictionary with override values
        
    Returns:
        A new dictionary with deep-merged values
    """
    import copy
    
    # Start with a deep copy of the default dict
    result = copy.deepcopy(default_dict)
    
    def _merge_recursive(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Both are dicts, merge recursively
                _merge_recursive(base[key], value)
            else:
                # Override the value (could be a new key or a leaf value)
                base[key] = copy.deepcopy(value)
    
    _merge_recursive(result, override_dict)
    return result

if __name__ == "__main__":
    default_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 4,
            "f": 5
        }
    }
    override_dict = {"b": 4, "d": {"e": 6, "g": 7}}    
    merged_dict = deep_merge_dicts(default_dict, override_dict)
    print(merged_dict)
