"""Multi-layer perceptron architectures for PU learning."""

import torch
import torch.nn as nn


class MLP6Layer(nn.Module):
    """6-layer MLP for binary classification.

    Architecture matches the nnPU paper specification:
    - Input: 784 dimensions (flattened 28x28 MNIST images)
    - Hidden layers: 5 layers of 300 units each with ReLU activation
    - Output: 1 unit with sigmoid activation for p(y=1|x) ∈ [0,1]

    Total architecture: [784, 300, 300, 300, 300, 300, 1]

    Attributes:
        layers: Sequential container of linear layers with activations.
    """

    def __init__(self) -> None:
        """Initialize the 6-layer MLP with Kaiming initialization."""
        super().__init__()

        # Define architecture: 6 layers total (5 hidden + 1 output)
        self.layers = nn.Sequential(
            # Layer 1: 784 -> 300
            nn.Linear(784, 300),
            nn.ReLU(),
            # Layer 2: 300 -> 300
            nn.Linear(300, 300),
            nn.ReLU(),
            # Layer 3: 300 -> 300
            nn.Linear(300, 300),
            nn.ReLU(),
            # Layer 4: 300 -> 300
            nn.Linear(300, 300),
            nn.ReLU(),
            # Layer 5: 300 -> 300
            nn.Linear(300, 300),
            nn.ReLU(),
            # Layer 6: 300 -> 1
            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

        # Initialize weights using Kaiming initialization for ReLU
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming (He) initialization for ReLU layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 784).

        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1].
        """
        return self.layers(x)
