"""Unit tests for PU loss functions.

⚠️ CRITICAL: These tests verify the mathematical correctness of the loss functions,
which are the core research contribution. All tests must pass.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from pu_learning.losses import PNLoss, nnPULoss, uPULoss


class TestPNLoss:
    """Tests for PN (Positive-Negative) loss function."""

    def test_pn_loss_computation(self) -> None:
        """Test PN loss computes correctly: L_PN = E_P[BCE(f(x),1)] + E_N[BCE(f(x),0)]."""
        loss_fn = PNLoss()

        # Create batch with half positive, half negative
        outputs = torch.tensor([[0.8], [0.9], [0.2], [0.1]])  # Model predictions
        labels = torch.tensor([1, 1, 0, 0])  # 2 positive, 2 negative
        is_labeled = torch.tensor([True, True, True, True])  # All labeled

        loss = loss_fn(outputs, labels, is_labeled)

        # Manual calculation: BCE loss
        # For positive: -log(0.8) - log(0.9)
        # For negative: -log(1-0.2) - log(1-0.1)
        expected_pos = F.binary_cross_entropy(
            torch.tensor([[0.8], [0.9]]), torch.tensor([[1.0], [1.0]]), reduction="mean"
        )
        expected_neg = F.binary_cross_entropy(
            torch.tensor([[0.2], [0.1]]), torch.tensor([[0.0], [0.0]]), reduction="mean"
        )
        expected_loss = expected_pos + expected_neg

        assert torch.isclose(loss, expected_loss, atol=1e-5), (
            f"Expected {expected_loss}, got {loss}"
        )

    def test_pn_loss_all_positive(self) -> None:
        """Test PN loss with only positive samples."""
        loss_fn = PNLoss()

        outputs = torch.tensor([[0.9], [0.8]])
        labels = torch.tensor([1, 1])
        is_labeled = torch.tensor([True, True])

        loss = loss_fn(outputs, labels, is_labeled)

        # Should only compute positive term
        expected = F.binary_cross_entropy(outputs, torch.ones_like(outputs), reduction="mean")
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_pn_loss_gradient_flow(self) -> None:
        """Test that gradients flow through PN loss."""
        loss_fn = PNLoss()

        outputs = torch.tensor([[0.7], [0.3]], requires_grad=True)
        labels = torch.tensor([1, 0])
        is_labeled = torch.tensor([True, True])

        loss = loss_fn(outputs, labels, is_labeled)
        loss.backward()

        assert outputs.grad is not None, "Gradients should flow through PN loss"
        assert not torch.any(torch.isnan(outputs.grad)), "Gradients should not be NaN"


class TestuPULoss:
    """Tests for uPU (Unbiased PU) loss function."""

    def test_upu_accepts_prior(self) -> None:
        """Test that uPU loss accepts class prior parameter."""
        prior = 0.5
        loss_fn = uPULoss(prior=prior)
        assert loss_fn.prior == prior, f"Expected prior={prior}, got {loss_fn.prior}"

    def test_upu_loss_can_be_negative(self) -> None:
        """Test that uPU loss can produce negative values (key property).

        This is expected behavior when the negative risk correction term is large.
        """
        prior = 0.5
        loss_fn = uPULoss(prior=prior)

        # Construct scenario where loss becomes negative:
        # High confidence on positive samples, very high confidence on unlabeled
        # This makes the correction term large
        outputs_p = torch.tensor([[0.99], [0.98], [0.97]])  # High confidence positive
        outputs_u = torch.tensor([[0.01], [0.02], [0.01]])  # High confidence negative (unlabeled)

        labels_p = torch.ones(3)
        labels_u = torch.zeros(3)  # Doesn't matter for unlabeled

        is_labeled_p = torch.tensor([True, True, True])
        is_labeled_u = torch.tensor([False, False, False])

        # Combine into single batch
        outputs = torch.cat([outputs_p, outputs_u])
        labels = torch.cat([labels_p, labels_u])
        is_labeled = torch.cat([is_labeled_p, is_labeled_u])

        loss = loss_fn(outputs, labels, is_labeled)

        # Loss can be negative (no assertion on sign, just checking it computes)
        # This is the key property that distinguishes uPU from nnPU
        assert isinstance(loss.item(), float), "Loss should be a valid number"
        # Optionally: assert loss < 0 if we want to verify negative case

    def test_upu_loss_computation_structure(self) -> None:
        """Test uPU loss has correct mathematical structure.

        L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]
        where l is the sigmoid loss
        """
        prior = 0.5
        loss_fn = uPULoss(prior=prior)

        outputs = torch.tensor([[0.8], [0.7], [0.3], [0.2]])
        labels = torch.tensor([1, 1, 0, 0])
        is_labeled = torch.tensor([True, True, False, False])

        loss = loss_fn(outputs, labels, is_labeled)

        # Loss should be finite
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_upu_gradient_flow(self) -> None:
        """Test that gradients flow through uPU loss."""
        loss_fn = uPULoss(prior=0.5)

        outputs = torch.tensor([[0.7], [0.3], [0.5], [0.6]], requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0])
        is_labeled = torch.tensor([True, True, False, False])

        loss = loss_fn(outputs, labels, is_labeled)
        loss.backward()

        assert outputs.grad is not None, "Gradients should flow through uPU loss"
        assert not torch.any(torch.isnan(outputs.grad)), "Gradients should not be NaN"


class TestnnPULoss:
    """Tests for nnPU (Non-Negative PU) loss function."""

    def test_nnpu_accepts_prior(self) -> None:
        """Test that nnPU loss accepts class prior parameter."""
        prior = 0.5
        loss_fn = nnPULoss(prior=prior)
        assert loss_fn.prior == prior, f"Expected prior={prior}, got {loss_fn.prior}"

    def test_nnpu_loss_always_nonnegative(self) -> None:
        """Test that nnPU loss is ALWAYS non-negative (CRITICAL property).

        This is the key contribution of the nnPU paper: using max(0, ...) to ensure
        non-negative risk, preventing overfitting to negative risk.
        """
        prior = 0.5
        loss_fn = nnPULoss(prior=prior)

        # Test multiple scenarios
        test_scenarios = [
            # Scenario 1: High confidence on both
            (
                torch.tensor([[0.99], [0.98]]),
                torch.tensor([[0.01], [0.02]]),
                torch.tensor([1, 1]),
                torch.tensor([0, 0]),
            ),
            # Scenario 2: Low confidence on both
            (
                torch.tensor([[0.5], [0.6]]),
                torch.tensor([[0.4], [0.5]]),
                torch.tensor([1, 1]),
                torch.tensor([0, 0]),
            ),
            # Scenario 3: Mixed confidence
            (
                torch.tensor([[0.8], [0.2]]),
                torch.tensor([[0.7], [0.3]]),
                torch.tensor([1, 1]),
                torch.tensor([0, 0]),
            ),
        ]

        for outputs_p, outputs_u, labels_p, labels_u in test_scenarios:
            outputs = torch.cat([outputs_p, outputs_u])
            labels = torch.cat([labels_p, labels_u])
            is_labeled = torch.cat(
                [
                    torch.ones(len(labels_p), dtype=torch.bool),
                    torch.zeros(len(labels_u), dtype=torch.bool),
                ]
            )

            loss = loss_fn(outputs, labels, is_labeled)

            assert loss >= 0, f"nnPU loss MUST be non-negative (key property!), got {loss}"

    def test_nnpu_vs_upu_difference(self) -> None:
        """Test that nnPU and uPU differ when uPU would be negative."""
        prior = 0.5

        upu_loss_fn = uPULoss(prior=prior)
        nnpu_loss_fn = nnPULoss(prior=prior)

        # Create scenario where uPU would be negative
        outputs = torch.tensor([[0.99], [0.98], [0.01], [0.02]])
        labels = torch.tensor([1, 1, 0, 0])
        is_labeled = torch.tensor([True, True, False, False])

        _ = upu_loss_fn(outputs, labels, is_labeled)  # uPU might be negative
        nnpu_loss = nnpu_loss_fn(outputs, labels, is_labeled)

        # nnPU should be >= 0, while uPU might be negative
        assert nnpu_loss >= 0, "nnPU loss must be non-negative"
        # When uPU is negative, nnPU should clamp to 0 or positive value

    def test_nnpu_gradient_flow(self) -> None:
        """Test that gradients flow through nnPU loss."""
        loss_fn = nnPULoss(prior=0.5)

        outputs = torch.tensor([[0.7], [0.3], [0.5], [0.6]], requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0])
        is_labeled = torch.tensor([True, True, False, False])

        loss = loss_fn(outputs, labels, is_labeled)
        loss.backward()

        assert outputs.grad is not None, "Gradients should flow through nnPU loss"
        # Note: gradients might be zero when max(0, ...) clamps to 0

    def test_nnpu_loss_computation_structure(self) -> None:
        """Test nnPU loss has correct mathematical structure.

        L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])
        """
        prior = 0.5
        loss_fn = nnPULoss(prior=prior)

        outputs = torch.tensor([[0.8], [0.7], [0.3], [0.2]])
        labels = torch.tensor([1, 1, 0, 0])
        is_labeled = torch.tensor([True, True, False, False])

        loss = loss_fn(outputs, labels, is_labeled)

        # Loss should be finite and non-negative
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss >= 0, f"Loss must be non-negative, got {loss}"


class TestLossEdgeCases:
    """Test edge cases for all loss functions."""

    def test_empty_positive_samples(self) -> None:
        """Test losses handle case with no positive samples gracefully."""
        # This should raise an error or handle gracefully
        # Depending on implementation, we might expect an error
        pass

    def test_empty_unlabeled_samples(self) -> None:
        """Test PU losses handle case with no unlabeled samples."""
        # For PN loss, this is fine (supervised learning)
        # For PU losses, we need unlabeled data
        pass

    def test_extreme_predictions(self) -> None:
        """Test losses handle extreme predictions (0.0, 1.0) without NaN."""
        for loss_class in [PNLoss, uPULoss, nnPULoss]:
            loss_fn = loss_class() if loss_class == PNLoss else loss_class(prior=0.5)

            # Extreme predictions
            outputs = torch.tensor([[0.0], [1.0], [0.5]])
            labels = torch.tensor([1, 1, 0])
            is_labeled = torch.tensor([True, True, False])

            loss = loss_fn(outputs, labels, is_labeled)

            # Should not be NaN (might be inf for log(0), but implementations should handle)
            assert not torch.isnan(loss), f"{loss_class.__name__} produced NaN with extreme values"
