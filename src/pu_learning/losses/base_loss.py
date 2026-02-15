"""Base class for PU learning loss functions."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PULoss(nn.Module, ABC):
    """Abstract base class for PU learning loss functions.

    All PU loss functions share common structure:
    - Accept class prior π (probability of positive class)
    - Compute loss on positive (labeled) and unlabeled data
    - Handle different mathematical formulations (PN, uPU, nnPU)

    Attributes:
        prior: Class prior π, P(y=1). For PU learning without prior estimation,
               this must be provided or estimated from data.
    """

    def __init__(self, prior: float | None = None) -> None:
        """Initialize PU loss function.

        Args:
            prior: Class prior π. Required for PU methods (uPU, nnPU).
                   Not used for PN (supervised baseline).
        """
        super().__init__()
        self.prior = prior

    @abstractmethod
    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_labeled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            outputs: Model predictions, shape (batch_size, 1), values in [0, 1].
            labels: Ground truth labels, shape (batch_size,).
                    1 for positive class, 0 for negative/unlabeled.
            is_labeled: Boolean mask, shape (batch_size,).
                       True for labeled samples, False for unlabeled.

        Returns:
            Scalar loss value.
        """
        pass
