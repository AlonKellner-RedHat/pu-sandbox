"""Reproducibility utilities for setting random seeds across all libraries."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch CPU operations
    - PyTorch MPS operations (Apple Silicon)
    - PyTorch CUDA operations (NVIDIA GPUs)

    Also configures CUDNN for deterministic behavior and sets PYTHONHASHSEED.

    Args:
        seed: Random seed to use (typically 42 or similar)

    Note:
        MPS operations may have slight numerical differences compared to CUDA/CPU
        due to hardware-specific implementations, but results should be consistent
        across runs on the same device.

    Examples:
        >>> set_seed(42)
        >>> # Now all random operations will be deterministic
        >>> x = torch.randn(100, 100)  # Reproducible
        >>> y = np.random.rand(100)     # Reproducible
    """
    # Python random module
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # PyTorch CPU operations
    torch.manual_seed(seed)

    # PyTorch MPS operations (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # PyTorch CUDA operations (NVIDIA GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Configure CUDNN for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_current_seed() -> dict[str, Optional[int]]:
    """
    Get the current random seeds from all libraries.

    Returns:
        dict: Dictionary containing current seeds for each library

    Examples:
        >>> set_seed(42)
        >>> seeds = get_current_seed()
        >>> print(seeds)
        {'torch': 42, 'numpy': <seed_state>, ...}
    """
    seeds = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_initial_seed": torch.initial_seed(),
        # Note: Python random and NumPy don't expose current seed directly
        # We can only get their state, not the original seed value
    }

    if torch.cuda.is_available():
        seeds["cuda_initial_seed"] = torch.cuda.initial_seed()

    return seeds
