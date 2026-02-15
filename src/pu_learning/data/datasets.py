"""MNIST Positive-Unlabeled (PU) dataset implementation."""

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MNISTPUDataset(Dataset):
    """
    MNIST dataset configured for Positive-Unlabeled (PU) learning.

    This dataset implements the PU learning setup from the nnPU paper:
    - Selects a subset of positive class as labeled positive set P
    - Treats remaining samples as unlabeled set U
    - U contains both positive and negative samples

    Args:
        root: Root directory for MNIST data
        n_positive: Number of labeled positive samples to select
        positive_class: Which digits to treat as positive class
            - "even": digits 0,2,4,6,8
            - "odd": digits 1,3,5,7,9
        train: If True, use training set; otherwise use test set
        download: If True, download MNIST if not already present
        seed: Random seed for reproducible P/U splits

    Attributes:
        data: Flattened and normalized image tensors (N, 784)
        targets: Binary labels (0=negative, 1=positive)
        original_labels: Original MNIST labels (0-9)
        is_labeled: Boolean tensor indicating which samples are in P
        class_prior: Proportion of positive class in unlabeled set U

    Examples:
        >>> dataset = MNISTPUDataset(root="./data", n_positive=100, positive_class="even")
        >>> image, target, is_labeled = dataset[0]
        >>> print(f"Image shape: {image.shape}, Target: {target}, Labeled: {is_labeled}")
        Image shape: torch.Size([784]), Target: 1, Labeled: False
    """

    # MNIST normalization constants
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    def __init__(
        self,
        root: str,
        n_positive: int,
        positive_class: Literal["even", "odd"] = "even",
        train: bool = True,
        download: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.root = root
        self.n_positive = n_positive
        self.positive_class = positive_class
        self.train = train
        self.seed = seed

        # Define positive class digits
        if positive_class == "even":
            self.positive_digits = {0, 2, 4, 6, 8}
        elif positive_class == "odd":
            self.positive_digits = {1, 3, 5, 7, 9}
        else:
            raise ValueError(f"positive_class must be 'even' or 'odd', got {positive_class}")

        # Load MNIST dataset
        # We'll apply normalization and flattening during __getitem__
        # to keep the original data intact for inspection
        mnist = datasets.MNIST(root=root, train=train, download=download)

        # Store original labels (mnist.targets is already a tensor)
        self.original_labels = mnist.targets.clone()

        # Convert to binary labels (1=positive, 0=negative) using tensor operations
        self.targets = torch.zeros_like(self.original_labels, dtype=torch.long)
        for digit in self.positive_digits:
            self.targets[self.original_labels == digit] = 1

        # Store images as tensors (mnist.data is already a tensor)
        self.data = mnist.data.float()

        # Initialize is_labeled tensor (all False initially)
        self.is_labeled = torch.zeros(len(self.data), dtype=torch.bool)

        # Select labeled positive samples
        self._create_pu_split()

        # Calculate class prior π (proportion of positive in unlabeled set)
        self._calculate_class_prior()

    def _create_pu_split(self) -> None:
        """
        Create P/U split by selecting n_positive labeled samples.

        Uses deterministic random sampling based on seed to ensure reproducibility.
        """
        # Get indices of all positive samples
        positive_indices = torch.where(self.targets == 1)[0].numpy()

        # Set seed for reproducible sampling
        rng = np.random.RandomState(self.seed)

        # Randomly select n_positive samples from positive class
        if len(positive_indices) < self.n_positive:
            raise ValueError(
                f"Not enough positive samples: need {self.n_positive}, "
                f"found {len(positive_indices)}"
            )

        labeled_indices = rng.choice(positive_indices, size=self.n_positive, replace=False)

        # Mark selected samples as labeled
        self.is_labeled[labeled_indices] = True

    def _calculate_class_prior(self) -> None:
        """Calculate class prior π = P(y=1) in unlabeled set U."""
        # Get unlabeled samples
        unlabeled_mask = ~self.is_labeled

        # Count positive samples in unlabeled set
        unlabeled_positives = (self.targets[unlabeled_mask] == 1).sum().item()
        unlabeled_total = unlabeled_mask.sum().item()

        # Calculate prior
        self.class_prior = unlabeled_positives / unlabeled_total if unlabeled_total > 0 else 0.0

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        """
        Get a single sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (image, target, is_labeled):
                - image: Flattened and normalized image tensor (784,)
                - target: Binary label (0 or 1)
                - is_labeled: Whether this sample is in labeled set P
        """
        # Get image and flatten (28x28 -> 784)
        image = self.data[index].reshape(-1)

        # Normalize with MNIST mean and std
        # First normalize to [0, 1] by dividing by 255
        image = image / 255.0

        # Then apply z-score normalization with MNIST statistics
        image = (image - self.MNIST_MEAN) / self.MNIST_STD

        # Get target and labeled flag
        target = self.targets[index].item()
        is_labeled = self.is_labeled[index].item()

        return image, target, is_labeled

    def get_positive_labeled_indices(self) -> torch.Tensor:
        """
        Get indices of labeled positive samples (set P).

        Returns:
            Tensor of indices for samples in P
        """
        return torch.where(self.is_labeled)[0]

    def get_unlabeled_indices(self) -> torch.Tensor:
        """
        Get indices of unlabeled samples (set U).

        Returns:
            Tensor of indices for samples in U
        """
        return torch.where(~self.is_labeled)[0]

    def get_statistics(self) -> dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        n_labeled = self.is_labeled.sum().item()
        n_unlabeled = (~self.is_labeled).sum().item()
        n_positive_total = (self.targets == 1).sum().item()
        n_negative_total = (self.targets == 0).sum().item()

        unlabeled_mask = ~self.is_labeled
        n_positive_unlabeled = (self.targets[unlabeled_mask] == 1).sum().item()
        n_negative_unlabeled = (self.targets[unlabeled_mask] == 0).sum().item()

        return {
            "total_samples": len(self),
            "labeled_samples": n_labeled,
            "unlabeled_samples": n_unlabeled,
            "total_positives": n_positive_total,
            "total_negatives": n_negative_total,
            "unlabeled_positives": n_positive_unlabeled,
            "unlabeled_negatives": n_negative_unlabeled,
            "class_prior": self.class_prior,
            "positive_class": self.positive_class,
            "positive_digits": sorted(self.positive_digits),
        }
