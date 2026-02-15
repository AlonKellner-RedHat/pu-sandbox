"""Pytest configuration and shared fixtures for PU learning tests."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in all tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def small_batch_size():
    """Small batch size for quick tests."""
    return 16


@pytest.fixture
def mnist_input_dim():
    """MNIST input dimension (28x28 flattened)."""
    return 784


@pytest.fixture
def hidden_dims():
    """Hidden layer dimensions for 6-layer MLP."""
    return [300, 300, 300, 300, 300]


@pytest.fixture
def class_prior():
    """Class prior π for PU learning."""
    return 0.5
