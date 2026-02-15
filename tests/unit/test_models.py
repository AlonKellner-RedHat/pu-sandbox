"""Unit tests for model architectures."""

import torch

from pu_learning.models import MLP6Layer
from pu_learning.utils.device import get_device


class TestMLP6Layer:
    """Tests for 6-layer MLP architecture."""

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        model = MLP6Layer()
        batch_size = 32
        input_dim = 784
        x = torch.randn(batch_size, input_dim)
        output = model(x)

        assert output.shape == (batch_size, 1), f"Expected output shape (32, 1), got {output.shape}"

    def test_output_range(self) -> None:
        """Test that output is in range [0, 1] due to sigmoid activation."""
        model = MLP6Layer()
        x = torch.randn(100, 784)
        output = model(x)

        assert torch.all(output >= 0) and torch.all(output <= 1), (
            f"Output must be in [0, 1], got min={output.min()}, max={output.max()}"
        )

    def test_parameter_count(self) -> None:
        """Test that model has expected number of parameters."""
        model = MLP6Layer()

        # Architecture: [784, 300, 300, 300, 300, 300, 1]
        # Layer 1: 784*300 + 300 = 235,500
        # Layer 2: 300*300 + 300 = 90,300
        # Layer 3: 300*300 + 300 = 90,300
        # Layer 4: 300*300 + 300 = 90,300
        # Layer 5: 300*300 + 300 = 90,300
        # Layer 6: 300*1 + 1 = 301
        # Total: 235,500 + 4*90,300 + 301 = 597,001
        expected_params = 597_001

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected_params, (
            f"Expected {expected_params} parameters, got {total_params}"
        )

    def test_initialization_no_nan_inf(self) -> None:
        """Test that weight initialization produces no NaN or Inf values."""
        model = MLP6Layer()

        for name, param in model.named_parameters():
            assert not torch.any(torch.isnan(param)), f"Parameter {name} contains NaN values"
            assert not torch.any(torch.isinf(param)), f"Parameter {name} contains Inf values"

    def test_model_on_mps_device(self) -> None:
        """Test that model can be moved to MPS device."""
        device = get_device()
        model = MLP6Layer().to(device)

        # Create input on same device
        x = torch.randn(16, 784, device=device)
        output = model(x)

        assert output.device.type == device.type, (
            f"Expected output on {device}, got {output.device}"
        )

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        model = MLP6Layer()
        x = torch.randn(8, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None, "Gradients should flow to input"

        # Check that all model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_batch_independence(self) -> None:
        """Test that predictions for different batch sizes are consistent."""
        model = MLP6Layer()
        model.eval()  # Set to eval mode for deterministic behavior

        # Create single sample
        x_single = torch.randn(1, 784)
        output_single = model(x_single)

        # Create batch with same sample repeated
        x_batch = x_single.repeat(5, 1)
        output_batch = model(x_batch)

        # All outputs should be identical
        assert torch.allclose(output_batch, output_single.repeat(5, 1), atol=1e-6), (
            "Model should produce identical outputs for identical inputs in batch"
        )

    def test_different_batch_sizes(self) -> None:
        """Test that model handles different batch sizes correctly."""
        model = MLP6Layer()

        for batch_size in [1, 16, 32, 64, 128]:
            x = torch.randn(batch_size, 784)
            output = model(x)
            assert output.shape == (batch_size, 1), (
                f"Failed for batch_size={batch_size}: expected ({batch_size}, 1), "
                f"got {output.shape}"
            )
