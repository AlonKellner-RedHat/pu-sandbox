"""uPU (Unbiased PU) loss function."""

import torch

from pu_learning.losses.base_loss import PULoss


class uPULoss(PULoss):  # noqa: N801
    """Unbiased PU learning loss.

    Mathematical formulation:
    L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]

    where:
    - π is the class prior P(y=1)
    - l(z) is the sigmoid loss: l(z) = log(1 + exp(-z))
    - f(x) is the model output (logit before sigmoid, or log-odds after)
    - P denotes positive labeled samples
    - U denotes unlabeled samples

    Key property: This loss CAN become negative when the negative risk correction
    term π·E_P[l(-f(x))] is large. This can lead to overfitting.

    The nnPU method addresses this by using max(0, ...) to ensure non-negative risk.
    """

    def __init__(self, prior: float) -> None:
        """Initialize uPU loss.

        Args:
            prior: Class prior π = P(y=1). Must be in (0, 1).
        """
        super().__init__(prior=prior)
        if not 0 < prior < 1:
            raise ValueError(f"Prior must be in (0, 1), got {prior}")

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_labeled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uPU loss.

        L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]

        Args:
            outputs: Model predictions, shape (batch_size, 1), values in [0, 1].
            labels: Ground truth labels, shape (batch_size,).
                    1 for positive, 0 for unlabeled.
            is_labeled: Boolean mask, shape (batch_size,).
                       True for labeled positive, False for unlabeled.

        Returns:
            Scalar loss value (CAN be negative).
        """
        # Separate positive labeled and unlabeled samples
        positive_mask = is_labeled & (labels == 1)
        unlabeled_mask = ~is_labeled

        positive_outputs = outputs[positive_mask]
        unlabeled_outputs = outputs[unlabeled_mask]

        # Small epsilon to avoid log(0)
        eps = 1e-7

        # Sigmoid loss: l(z) = log(1 + exp(-z)) = -log(sigmoid(z))
        # For positive: l(f(x)) = -log(sigmoid(f(x))) = BCE with target 1
        # For negative: l(-f(x)) = -log(sigmoid(-f(x))) = BCE with target 0

        # Term 1: π·E_P[l(f(x))] - positive risk
        if positive_mask.any():
            # l(f(x)) = -log(sigmoid(f(x))) = -log(p(x))
            positive_risk = -torch.log(positive_outputs + eps).mean()
        else:
            positive_risk = torch.tensor(0.0, device=outputs.device)

        # Term 2: E_U[l(-f(x))] - unlabeled negative risk
        if unlabeled_mask.any():
            # l(-f(x)) = -log(sigmoid(-f(x))) = -log(1 - p(x))
            unlabeled_negative_risk = -torch.log(1 - unlabeled_outputs + eps).mean()
        else:
            unlabeled_negative_risk = torch.tensor(0.0, device=outputs.device)

        # Term 3: π·E_P[l(-f(x))] - correction term for negative risk
        if positive_mask.any():
            # l(-f(x)) for positive samples
            positive_negative_risk = -torch.log(1 - positive_outputs + eps).mean()
        else:
            positive_negative_risk = torch.tensor(0.0, device=outputs.device)

        # Combine terms: π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]
        loss = (
            self.prior * positive_risk
            + unlabeled_negative_risk
            - self.prior * positive_negative_risk
        )

        return loss
