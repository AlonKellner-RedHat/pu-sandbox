"""PU learning trainer."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pu_learning.losses.base_loss import PULoss
from pu_learning.utils.metrics import accuracy, zero_one_loss


class PUTrainer:
    """Trainer for PU learning models.

    Handles training loops, evaluation, and checkpoint management for
    positive-unlabeled learning with any loss function (PN, uPU, nnPU).

    Attributes:
        model: Neural network model (e.g., MLP6Layer).
        loss_fn: Loss function (PNLoss, uPULoss, or nnPULoss).
        optimizer: PyTorch optimizer.
        device: Device for training (MPS, CUDA, or CPU).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: PULoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        """Initialize PU trainer.

        Args:
            model: Neural network model to train.
            loss_fn: Loss function for PU learning.
            optimizer: Optimizer for training.
            device: Device to use for training.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            dataloader: DataLoader providing (features, labels, is_labeled) batches.
            epoch: Current epoch number (for logging).

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch: (features, labels, is_labeled)
            if len(batch) == 3:
                features, labels, is_labeled = batch
            else:
                # Fallback for regular datasets
                features, labels = batch
                is_labeled = torch.ones(len(labels), dtype=torch.bool)

            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            is_labeled = is_labeled.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # Compute loss
            loss = self.loss_fn(outputs, labels, is_labeled)

            # Backward pass (only if loss requires gradients)
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on dataset.

        Args:
            dataloader: DataLoader providing (features, labels, is_labeled) batches.

        Returns:
            Dictionary with metrics: loss, accuracy, zero_one_loss.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                if len(batch) == 3:
                    features, labels, is_labeled = batch
                else:
                    features, labels = batch
                    is_labeled = torch.ones(len(labels), dtype=torch.bool)

                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                is_labeled = is_labeled.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Compute loss
                loss = self.loss_fn(outputs, labels, is_labeled)
                total_loss += loss.item()
                num_batches += 1

                # Collect predictions and labels for metrics
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Compute average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        acc = accuracy(all_predictions, all_labels)
        zol = zero_one_loss(all_predictions, all_labels)

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "zero_one_loss": zol,
        }

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            metrics: Optional metrics dictionary to save.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path | str) -> tuple[int, dict[str, float]]:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Tuple of (epoch, metrics) from checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint["epoch"]
        metrics = checkpoint.get("metrics", {})

        return epoch, metrics
