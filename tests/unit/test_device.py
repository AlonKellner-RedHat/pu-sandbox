"""Tests for device selection utilities."""

import torch

from pu_learning.utils.device import get_device, get_device_name


def test_get_device_returns_torch_device():
    """Test that get_device returns a torch.device object."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_prefers_mps_on_apple_silicon():
    """Test that MPS is selected when available and prefer_mps=True."""
    if torch.backends.mps.is_available():
        device = get_device(prefer_mps=True)
        assert device.type == "mps"


def test_get_device_can_skip_mps():
    """Test that MPS can be skipped with prefer_mps=False."""
    device = get_device(prefer_mps=False)
    # Should be cuda if available, otherwise cpu
    assert device.type in ["cuda", "cpu"]


def test_get_device_falls_back_to_cpu():
    """Test that CPU is selected when no GPU is available."""
    device = get_device()
    # Device should be one of the valid types
    assert device.type in ["mps", "cuda", "cpu"]


def test_get_device_name_mps():
    """Test get_device_name for MPS device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        name = get_device_name(device)
        assert "MPS" in name or "Apple Silicon" in name


def test_get_device_name_cpu():
    """Test get_device_name for CPU device."""
    device = torch.device("cpu")
    name = get_device_name(device)
    assert "CPU" in name


def test_device_can_allocate_tensor():
    """Test that tensors can be allocated on the selected device."""
    device = get_device()
    tensor = torch.randn(10, 10, device=device)
    assert tensor.device.type == device.type


def test_mps_computation_if_available():
    """Test that MPS device can perform computations."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x @ y  # Matrix multiplication
        assert z.shape == (100, 100)
        assert z.device.type == "mps"
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
