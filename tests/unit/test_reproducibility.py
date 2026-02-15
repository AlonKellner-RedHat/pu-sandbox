"""Tests for reproducibility utilities."""

import os
import random

import numpy as np
import pytest
import torch

from pu_learning.utils.reproducibility import get_current_seed, set_seed


def test_set_seed_makes_torch_deterministic():
    """Test that set_seed makes PyTorch operations deterministic."""
    set_seed(42)
    tensor1 = torch.randn(10, 10)

    set_seed(42)
    tensor2 = torch.randn(10, 10)

    assert torch.allclose(tensor1, tensor2, atol=1e-7)


def test_set_seed_makes_numpy_deterministic():
    """Test that set_seed makes NumPy operations deterministic."""
    set_seed(42)
    array1 = np.random.rand(10, 10)

    set_seed(42)
    array2 = np.random.rand(10, 10)

    assert np.allclose(array1, array2, atol=1e-7)


def test_set_seed_makes_python_random_deterministic():
    """Test that set_seed makes Python random module deterministic."""
    set_seed(42)
    numbers1 = [random.random() for _ in range(10)]

    set_seed(42)
    numbers2 = [random.random() for _ in range(10)]

    assert numbers1 == numbers2


def test_set_seed_sets_pythonhashseed():
    """Test that set_seed sets PYTHONHASHSEED environment variable."""
    set_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"

    set_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different random numbers."""
    set_seed(42)
    tensor1 = torch.randn(10, 10)

    set_seed(123)
    tensor2 = torch.randn(10, 10)

    # Should be different with high probability
    assert not torch.allclose(tensor1, tensor2, atol=1e-3)


def test_get_current_seed_returns_dict():
    """Test that get_current_seed returns a dictionary."""
    set_seed(42)
    seeds = get_current_seed()

    assert isinstance(seeds, dict)
    assert "pythonhashseed" in seeds
    assert "torch_initial_seed" in seeds


def test_reproducibility_with_mps_device():
    """Test that reproducibility works with MPS device."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = torch.device("mps")

    set_seed(42)
    x1 = torch.randn(100, 100, device=device)
    y1 = torch.randn(100, 100, device=device)
    z1 = (x1 @ y1).cpu()  # Move to CPU for comparison

    set_seed(42)
    x2 = torch.randn(100, 100, device=device)
    y2 = torch.randn(100, 100, device=device)
    z2 = (x2 @ y2).cpu()  # Move to CPU for comparison

    # MPS may have minor numerical differences, so use relaxed tolerance
    assert torch.allclose(z1, z2, atol=1e-5, rtol=1e-4)


def test_reproducibility_across_multiple_operations():
    """Test that reproducibility holds across multiple operations."""
    set_seed(42)
    results1 = []
    for _ in range(5):
        results1.append(torch.randn(10).sum().item())
        results1.append(np.random.rand())
        results1.append(random.random())

    set_seed(42)
    results2 = []
    for _ in range(5):
        results2.append(torch.randn(10).sum().item())
        results2.append(np.random.rand())
        results2.append(random.random())

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert abs(r1 - r2) < 1e-6
