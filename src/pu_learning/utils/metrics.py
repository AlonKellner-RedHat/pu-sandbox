"""Evaluation metrics for binary classification."""

import torch


def zero_one_loss(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute zero-one loss (classification error rate).

    The zero-one loss is the fraction of misclassified samples. It's the primary
    metric used in the nnPU paper for evaluating model performance.

    Args:
        predictions: Model predictions, either binary {0, 1} or probabilities [0, 1].
                    If probabilities, will be thresholded at 0.5 (> 0.5 → 1, <= 0.5 → 0).
                    Shape: (batch_size,) or (batch_size, 1).
        targets: Ground truth labels {0, 1}.
                Shape: (batch_size,) or (batch_size, 1).

    Returns:
        Classification error rate in [0, 1]. 0 = perfect, 1 = all wrong.
    """
    # Flatten if 2D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()

    # Threshold predictions at 0.5 if they're probabilities
    # Using > 0.5 so that exactly 0.5 is classified as 0
    binary_predictions = (predictions > 0.5).float()

    # Compute error rate
    errors = (binary_predictions != targets).float()
    error_rate = errors.mean().item()

    return error_rate


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Accuracy = 1 - zero_one_loss.

    Args:
        predictions: Model predictions, either binary {0, 1} or probabilities [0, 1].
                    If probabilities, will be thresholded at 0.5.
                    Shape: (batch_size,) or (batch_size, 1).
        targets: Ground truth labels {0, 1}.
                Shape: (batch_size,) or (batch_size, 1).

    Returns:
        Classification accuracy in [0, 1]. 1 = perfect, 0 = all wrong.
    """
    return 1.0 - zero_one_loss(predictions, targets)
