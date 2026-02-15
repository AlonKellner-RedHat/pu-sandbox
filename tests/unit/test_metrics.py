"""Unit tests for evaluation metrics."""

import torch

from pu_learning.utils.metrics import accuracy, zero_one_loss


class TestZeroOneLoss:
    """Tests for zero-one loss (classification error rate)."""

    def test_zero_one_loss_perfect_predictions(self) -> None:
        """Test zero-one loss with perfect predictions (should be 0)."""
        predictions = torch.tensor([1, 1, 0, 0])
        targets = torch.tensor([1, 1, 0, 0])

        loss = zero_one_loss(predictions, targets)

        assert loss == 0.0, "Perfect predictions should have zero-one loss of 0"

    def test_zero_one_loss_all_wrong(self) -> None:
        """Test zero-one loss with all wrong predictions (should be 1)."""
        predictions = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([1, 1, 0, 0])

        loss = zero_one_loss(predictions, targets)

        assert loss == 1.0, "All wrong predictions should have zero-one loss of 1"

    def test_zero_one_loss_half_correct(self) -> None:
        """Test zero-one loss with 50% correct predictions."""
        predictions = torch.tensor([1, 0, 0, 1])
        targets = torch.tensor([1, 1, 0, 0])

        loss = zero_one_loss(predictions, targets)

        assert loss == 0.5, "50% correct should have zero-one loss of 0.5"

    def test_zero_one_loss_with_probabilities(self) -> None:
        """Test zero-one loss with probability inputs (should threshold at 0.5)."""
        # Probabilities > 0.5 become 1, <= 0.5 become 0
        predictions = torch.tensor([0.9, 0.8, 0.3, 0.1])  # [1, 1, 0, 0] after threshold
        targets = torch.tensor([1, 1, 0, 0])

        loss = zero_one_loss(predictions, targets)

        assert loss == 0.0, "Should threshold probabilities at 0.5"

    def test_zero_one_loss_boundary_case(self) -> None:
        """Test zero-one loss at threshold boundary (0.5)."""
        # Exactly 0.5 should be classified as 0 (using > 0.5)
        predictions = torch.tensor([0.5, 0.51, 0.49])
        targets = torch.tensor([0, 1, 0])

        loss = zero_one_loss(predictions, targets)

        assert loss == 0.0, "Threshold at 0.5: > 0.5 → 1, <= 0.5 → 0"

    def test_zero_one_loss_2d_tensors(self) -> None:
        """Test zero-one loss with 2D tensors (batch_size, 1)."""
        predictions = torch.tensor([[0.9], [0.2], [0.7], [0.1]])
        targets = torch.tensor([[1], [0], [1], [0]])

        loss = zero_one_loss(predictions, targets)

        assert loss == 0.0, "Should handle 2D tensors"

    def test_zero_one_loss_returns_float(self) -> None:
        """Test that zero-one loss returns a Python float."""
        predictions = torch.tensor([1, 0])
        targets = torch.tensor([1, 1])

        loss = zero_one_loss(predictions, targets)

        assert isinstance(loss, float), f"Expected float, got {type(loss)}"


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_accuracy_perfect_predictions(self) -> None:
        """Test accuracy with perfect predictions (should be 1.0)."""
        predictions = torch.tensor([1, 1, 0, 0])
        targets = torch.tensor([1, 1, 0, 0])

        acc = accuracy(predictions, targets)

        assert acc == 1.0, "Perfect predictions should have accuracy of 1.0"

    def test_accuracy_all_wrong(self) -> None:
        """Test accuracy with all wrong predictions (should be 0.0)."""
        predictions = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([1, 1, 0, 0])

        acc = accuracy(predictions, targets)

        assert acc == 0.0, "All wrong predictions should have accuracy of 0.0"

    def test_accuracy_half_correct(self) -> None:
        """Test accuracy with 50% correct predictions."""
        predictions = torch.tensor([1, 0, 0, 1])
        targets = torch.tensor([1, 1, 0, 0])

        acc = accuracy(predictions, targets)

        assert acc == 0.5, "50% correct should have accuracy of 0.5"

    def test_accuracy_equals_one_minus_zero_one_loss(self) -> None:
        """Test that accuracy = 1 - zero_one_loss."""
        predictions = torch.tensor([1, 0, 1, 0, 1])
        targets = torch.tensor([1, 1, 0, 0, 1])

        acc = accuracy(predictions, targets)
        loss = zero_one_loss(predictions, targets)

        assert abs(acc - (1 - loss)) < 1e-7, "Accuracy should equal 1 - zero_one_loss"

    def test_accuracy_returns_float(self) -> None:
        """Test that accuracy returns a Python float."""
        predictions = torch.tensor([1, 0])
        targets = torch.tensor([1, 1])

        acc = accuracy(predictions, targets)

        assert isinstance(acc, float), f"Expected float, got {type(acc)}"


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""

    def test_single_sample(self) -> None:
        """Test metrics with single sample."""
        predictions = torch.tensor([1])
        targets = torch.tensor([1])

        loss = zero_one_loss(predictions, targets)
        acc = accuracy(predictions, targets)

        assert loss == 0.0
        assert acc == 1.0

    def test_large_batch(self) -> None:
        """Test metrics with large batch."""
        # Create large batch: 1000 samples, 80% correct
        predictions = torch.cat([torch.ones(800), torch.zeros(200)])
        targets = torch.cat([torch.ones(800), torch.ones(200)])  # Last 200 are wrong

        loss = zero_one_loss(predictions, targets)
        acc = accuracy(predictions, targets)

        assert abs(loss - 0.2) < 1e-6, "Should have 20% error rate"
        assert abs(acc - 0.8) < 1e-6, "Should have 80% accuracy"
