"""nnPU (Non-Negative PU) loss function - the key research contribution."""

import torch

from pu_learning.losses.base_loss import PULoss


class nnPULoss(PULoss):  # noqa: N801
    """Non-Negative PU learning loss (nnPU) - the main contribution of the paper.

    Mathematical formulation:
    L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])

    where:
    - π is the class prior P(y=1)
    - l(z) is the sigmoid loss: l(z) = log(1 + exp(-z))
    - f(x) is the model output (logit before sigmoid, or log-odds after)
    - P denotes positive labeled samples
    - U denotes unlabeled samples

    Key property: This loss is ALWAYS non-negative due to the max(0, ...) operation.
    This prevents overfitting to negative risk that can occur with uPU.

    The max(0, ...) is applied to the unlabeled negative risk term:
        negative_risk = E_U[l(-f(x))] - π·E_P[l(-f(x))]

    When negative_risk < 0 (negative risk), it's clamped to 0, preventing the model
    from exploiting negative risk for overfitting.
    """

    def __init__(self, prior: float) -> None:
        """Initialize nnPU loss.

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
        """Compute nnPU loss.

        L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])

        Args:
            outputs: Model predictions, shape (batch_size, 1), values in [0, 1].
            labels: Ground truth labels, shape (batch_size,).
                    1 for positive, 0 for unlabeled.
            is_labeled: Boolean mask, shape (batch_size,).
                       True for labeled positive, False for unlabeled.

        Returns:
            Scalar loss value (ALWAYS >= 0).
        """
        # Separate positive labeled and unlabeled samples
        positive_mask = is_labeled & (labels == 1)
        unlabeled_mask = ~is_labeled

        positive_outputs = outputs[positive_mask]
        unlabeled_outputs = outputs[unlabeled_mask]

        # Small epsilon to avoid log(0)
        eps = 1e-7

        # Term 1: π·E_P[l(f(x))] - positive risk (always in final loss)
        if positive_mask.any():
            # l(f(x)) = -log(sigmoid(f(x))) = -log(p(x))
            positive_risk = -torch.log(positive_outputs + eps).mean()
        else:
            positive_risk = outputs.new_zeros(())

        # Term 2: E_U[l(-f(x))] - unlabeled negative risk
        if unlabeled_mask.any():
            # l(-f(x)) = -log(sigmoid(-f(x))) = -log(1 - p(x))
            unlabeled_negative_risk = -torch.log(1 - unlabeled_outputs + eps).mean()
        else:
            unlabeled_negative_risk = outputs.new_zeros(())

        # Term 3: π·E_P[l(-f(x))] - correction term for negative risk
        if positive_mask.any():
            # l(-f(x)) for positive samples
            positive_negative_risk = -torch.log(1 - positive_outputs + eps).mean()
        else:
            positive_negative_risk = outputs.new_zeros(())

        # Compute the unlabeled negative risk with correction
        # This term can be negative in uPU, but we clamp to 0 in nnPU
        negative_risk = unlabeled_negative_risk - self.prior * positive_negative_risk

        # nnPU: Apply max(0, ...) to negative risk term (KEY DIFFERENCE from uPU)
        negative_risk_clamped = torch.clamp(negative_risk, min=0.0)

        # Final loss: π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])
        loss = self.prior * positive_risk + negative_risk_clamped

        # Assertion: loss must be non-negative (critical property)
        assert loss >= 0, (
            f"nnPU loss MUST be non-negative! Got {loss}. "
            f"This violates the fundamental property of nnPU."
        )

        return loss
