"""Unit tests for MNIST PU dataset implementation."""

import torch
from torch.utils.data import DataLoader

from pu_learning.data.datasets import MNISTPUDataset


class TestMNISTPUDataset:
    """Test suite for MNIST PU dataset."""

    def test_dataset_initialization(self, device):
        """Test that dataset can be initialized with correct parameters."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )
        assert dataset is not None
        assert len(dataset) > 0

    def test_positive_set_size(self, device):
        """Test that positive set P contains exactly n_positive samples."""
        n_positive = 100
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=n_positive,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        # Count labeled positive samples
        labeled_count = sum(
            1 for i in range(len(dataset)) if dataset.is_labeled[i] and dataset.targets[i] == 1
        )
        assert labeled_count == n_positive, (
            f"Expected exactly {n_positive} labeled positive samples, got {labeled_count}"
        )

    def test_positive_class_even_digits(self, device):
        """Test that positive class contains only even digits (0,2,4,6,8)."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        even_digits = {0, 2, 4, 6, 8}

        # Check all samples with target=1 are even digits
        for i in range(len(dataset)):
            if dataset.targets[i] == 1:
                original_label = dataset.original_labels[i].item()
                assert original_label in even_digits, (
                    f"Positive sample has odd digit {original_label}, expected even"
                )

    def test_positive_class_odd_digits(self, device):
        """Test that positive class contains only odd digits (1,3,5,7,9)."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="odd",
            train=True,
            download=True,
            seed=42,
        )

        odd_digits = {1, 3, 5, 7, 9}

        # Check all samples with target=1 are odd digits
        for i in range(len(dataset)):
            if dataset.targets[i] == 1:
                original_label = dataset.original_labels[i].item()
                assert original_label in odd_digits, (
                    f"Positive sample has even digit {original_label}, expected odd"
                )

    def test_unlabeled_set_size(self, device):
        """Test that unlabeled set U contains remaining samples."""
        n_positive = 100
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=n_positive,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        # Count unlabeled samples
        unlabeled_count = sum(1 for i in range(len(dataset)) if not dataset.is_labeled[i])

        # MNIST train has 60,000 samples, minus 100 positive labeled
        expected_unlabeled = 60000 - n_positive
        assert unlabeled_count == expected_unlabeled, (
            f"Expected {expected_unlabeled} unlabeled samples, got {unlabeled_count}"
        )

    def test_class_prior_calculation(self, device):
        """Test that class prior π is calculated correctly."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        prior = dataset.class_prior

        # Count actual positives in unlabeled set
        unlabeled_positives = sum(
            1 for i in range(len(dataset)) if not dataset.is_labeled[i] and dataset.targets[i] == 1
        )
        unlabeled_total = sum(1 for i in range(len(dataset)) if not dataset.is_labeled[i])

        expected_prior = unlabeled_positives / unlabeled_total

        assert abs(prior - expected_prior) < 1e-6, (
            f"Class prior {prior} does not match expected {expected_prior}"
        )

    def test_deterministic_sampling(self, device):
        """Test that same seed produces same P/U split."""
        dataset1 = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        dataset2 = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        # Check that is_labeled arrays are identical
        assert torch.equal(dataset1.is_labeled, dataset2.is_labeled), (
            "Same seed should produce identical P/U splits"
        )

    def test_different_seeds_different_splits(self, device):
        """Test that different seeds produce different P/U splits."""
        dataset1 = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        dataset2 = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=43,
        )

        # Check that is_labeled arrays are different
        assert not torch.equal(dataset1.is_labeled, dataset2.is_labeled), (
            "Different seeds should produce different P/U splits"
        )

    def test_image_shape_flattened(self, device):
        """Test that images are flattened to 784 dimensions."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        image, _target, _is_labeled = dataset[0]

        assert image.shape == (784,), f"Expected shape (784,), got {image.shape}"

    def test_image_normalization(self, device):
        """Test that images are normalized with MNIST mean and std."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        # Get a batch of images
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        images, _, _ = next(iter(loader))

        # Check approximate mean and std (should be close to 0 and 1 after normalization)
        mean = images.mean().item()
        std = images.std().item()

        # After normalization with mean=0.1307, std=0.3081, distribution should be centered
        assert abs(mean) < 1.0, f"Mean {mean} too far from 0 after normalization"
        assert abs(std - 1.0) < 0.5, f"Std {std} too far from 1 after normalization"

    def test_dataloader_integration(self, device):
        """Test that dataset works with PyTorch DataLoader."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Get one batch
        images, targets, is_labeled = next(iter(loader))

        assert images.shape == (32, 784), f"Expected batch shape (32, 784), got {images.shape}"
        assert targets.shape == (32,), f"Expected targets shape (32,), got {targets.shape}"
        assert is_labeled.shape == (32,), f"Expected is_labeled shape (32,), got {is_labeled.shape}"

    def test_target_values_binary(self, device):
        """Test that targets are binary (0 or 1)."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        # Check all targets are 0 or 1
        for i in range(len(dataset)):
            _, target, _ = dataset[i]
            assert target in [0, 1], f"Target {target} is not binary (0 or 1)"

    def test_test_set_creation(self, device):
        """Test that test set can be created."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=False,  # Test set
            download=True,
            seed=42,
        )

        # MNIST test set has 10,000 samples
        assert len(dataset) == 10000, f"Expected 10,000 test samples, got {len(dataset)}"

    def test_class_prior_approximately_half_for_even(self, device):
        """Test that class prior for even digits is approximately 0.5."""
        dataset = MNISTPUDataset(
            root="./data",
            n_positive=100,
            positive_class="even",
            train=True,
            download=True,
            seed=42,
        )

        prior = dataset.class_prior

        # MNIST has roughly equal distribution, so even digits should be ~0.5
        assert 0.4 < prior < 0.6, f"Class prior {prior} not approximately 0.5 for even digits"
