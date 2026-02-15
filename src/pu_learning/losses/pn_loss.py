"""PN (Positive-Negative) loss - supervised baseline."""

import torch
import torch.nn.functional as F  # noqa: N812

from pu_learning.losses.base_loss import PULoss


class PNLoss(PULoss):
    """PN (Positive-Negative) loss - supervised learning baseline.

    This is the standard supervised binary cross-entropy loss:
    L_PN = E_P[BCE(f(x), 1)] + E_N[BCE(f(x), 0)]

    Requires both positive AND negative labeled data.
    Used as baseline to compare against PU methods.
    """

    def __init__(self) -> None:
        """Initialize PN loss (no prior needed for supervised learning)."""
        super().__init__(prior=None)

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_labeled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PN (supervised) loss.

        L_PN = E_P[BCE(f(x), 1)] + E_N[BCE(f(x), 0)]

        Args:
            outputs: Model predictions, shape (batch_size, 1), values in [0, 1].
            labels: Ground truth labels, shape (batch_size,).
                    1 for positive, 0 for negative.
            is_labeled: Boolean mask, shape (batch_size,).
                       For PN loss, expects all samples to be labeled.

        Returns:
            Scalar loss value (always non-negative for BCE).
        """
        # Filter to labeled samples only
        labeled_mask = is_labeled
        labeled_outputs = outputs[labeled_mask]
        labeled_labels = labels[labeled_mask].float().unsqueeze(1)

        # Separate positive and negative samples
        positive_mask = labeled_labels == 1
        negative_mask = labeled_labels == 0

        # Compute positive loss: E_P[BCE(f(x), 1)]
        if positive_mask.any():
            positive_outputs = labeled_outputs[positive_mask.squeeze()]
            positive_targets = torch.ones_like(positive_outputs)
            positive_loss = F.binary_cross_entropy(
                positive_outputs, positive_targets, reduction="mean"
            )
        else:
            positive_loss = outputs.new_zeros(())

        # Compute negative loss: E_N[BCE(f(x), 0)]
        if negative_mask.any():
            negative_outputs = labeled_outputs[negative_mask.squeeze()]
            negative_targets = torch.zeros_like(negative_outputs)
            negative_loss = F.binary_cross_entropy(
                negative_outputs, negative_targets, reduction="mean"
            )
        else:
            negative_loss = outputs.new_zeros(())

        # Total loss
        return positive_loss + negative_loss
