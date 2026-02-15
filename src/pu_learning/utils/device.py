"""Device selection utilities for PyTorch (MPS/CUDA/CPU)."""

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Select the best available device for PyTorch operations.

    Device priority: MPS > CUDA > CPU

    Args:
        prefer_mps: If True, prefer MPS over CUDA when both are available.
            This is useful for Apple Silicon (M1/M2/M3/M4) which have MPS support.

    Returns:
        torch.device: The selected device (mps, cuda, or cpu)

    Examples:
        >>> device = get_device()  # Auto-selects best device
        >>> model = MyModel().to(device)
        >>> data = data.to(device)
    """
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration)")

    return device


def get_device_name(device: torch.device) -> str:
    """
    Get a human-readable name for the device.

    Args:
        device: PyTorch device

    Returns:
        str: Human-readable device name

    Examples:
        >>> device = get_device()
        >>> name = get_device_name(device)
        >>> print(f"Training on: {name}")
    """
    if device.type == "mps":
        return "MPS (Apple Silicon)"
    elif device.type == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(device)}"
    else:
        return "CPU"
