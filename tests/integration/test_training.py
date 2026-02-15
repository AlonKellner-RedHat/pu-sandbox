"""Integration tests for training pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pu_learning.losses import PNLoss, nnPULoss
from pu_learning.models import MLP6Layer
from pu_learning.training.trainer import PUTrainer
from pu_learning.utils.device import get_device
from pu_learning.utils.reproducibility import set_seed


@pytest.fixture
def simple_dataset():
    """Create a simple synthetic dataset for testing."""
    # Create simple linearly separable data
    torch.manual_seed(42)

    # Positive class (100 samples)
    x_pos = torch.randn(100, 784) + 1.0  # Mean shifted to 1
    y_pos = torch.ones(100)
    labeled_pos = torch.ones(100, dtype=torch.bool)

    # Negative class (100 samples, unlabeled in PU setting)
    x_neg = torch.randn(100, 784) - 1.0  # Mean shifted to -1
    y_neg = torch.zeros(100)
    labeled_neg = torch.zeros(100, dtype=torch.bool)

    # Combine
    x = torch.cat([x_pos, x_neg])
    y = torch.cat([y_pos, y_neg])
    labeled = torch.cat([labeled_pos, labeled_neg])

    # Create dataset with (features, labels, is_labeled)
    dataset = TensorDataset(x, y, labeled)
    return dataset


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return MLP6Layer()


class TestPUTrainerBasics:
    """Test basic trainer functionality."""

    def test_trainer_initialization(self, simple_model, simple_dataset):
        """Test that trainer initializes correctly."""
        device = get_device()
        loss_fn = PNLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        trainer = PUTrainer(
            model=simple_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        assert trainer.model is simple_model
        assert trainer.loss_fn is loss_fn
        assert trainer.optimizer is optimizer
        assert trainer.device == device

    def test_training_reduces_loss(self, simple_model, simple_dataset):
        """Test that training reduces loss over epochs."""
        set_seed(42)
        device = get_device()

        loss_fn = PNLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        dataloader = DataLoader(simple_dataset, batch_size=32, shuffle=True)

        trainer = PUTrainer(
            model=simple_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Train for a few epochs
        initial_loss = None
        final_loss = None

        for epoch in range(5):
            epoch_loss = trainer.train_epoch(dataloader, epoch)
            if epoch == 0:
                initial_loss = epoch_loss
            if epoch == 4:
                final_loss = epoch_loss

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Training should reduce loss: initial={initial_loss}, final={final_loss}"
        )

    def test_evaluation_computes_metrics(self, simple_model, simple_dataset):
        """Test that evaluation computes metrics correctly."""
        device = get_device()
        dataloader = DataLoader(simple_dataset, batch_size=32)

        trainer = PUTrainer(
            model=simple_model,
            loss_fn=PNLoss(),
            optimizer=torch.optim.SGD(simple_model.parameters(), lr=0.01),
            device=device,
        )

        # Evaluate (even on untrained model)
        metrics = trainer.evaluate(dataloader)

        # Check that metrics are returned
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "zero_one_loss" in metrics

        # Check that metrics are in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["zero_one_loss"] <= 1
        assert metrics["loss"] >= 0  # Loss should be non-negative

    def test_train_epoch_returns_loss(self, simple_model, simple_dataset):
        """Test that train_epoch returns average loss."""
        device = get_device()
        dataloader = DataLoader(simple_dataset, batch_size=32)

        trainer = PUTrainer(
            model=simple_model,
            loss_fn=PNLoss(),
            optimizer=torch.optim.SGD(simple_model.parameters(), lr=0.01),
            device=device,
        )

        loss = trainer.train_epoch(dataloader, epoch=0)

        assert isinstance(loss, float)
        assert loss >= 0


class TestPUTrainerCheckpoints:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint(self, simple_model, simple_dataset):
        """Test saving a checkpoint."""
        device = get_device()
        trainer = PUTrainer(
            model=simple_model,
            loss_fn=PNLoss(),
            optimizer=torch.optim.SGD(simple_model.parameters(), lr=0.01),
            device=device,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path, epoch=5, metrics={"loss": 0.5})

            # Check file exists
            assert checkpoint_path.exists()

    def test_load_checkpoint(self, simple_dataset):
        """Test loading a checkpoint restores state."""
        device = get_device()

        # Create and train first model
        model1 = MLP6Layer().to(device)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        trainer1 = PUTrainer(
            model=model1,
            loss_fn=PNLoss(),
            optimizer=optimizer1,
            device=device,
        )

        # Train for a few steps
        dataloader = DataLoader(simple_dataset, batch_size=32)
        trainer1.train_epoch(dataloader, epoch=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            trainer1.save_checkpoint(checkpoint_path, epoch=1, metrics={"loss": 0.5})

            # Create new model and trainer
            model2 = MLP6Layer().to(device)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            trainer2 = PUTrainer(
                model=model2,
                loss_fn=PNLoss(),
                optimizer=optimizer2,
                device=device,
            )

            # Load checkpoint
            loaded_epoch, loaded_metrics = trainer2.load_checkpoint(checkpoint_path)

            # Check epoch and metrics
            assert loaded_epoch == 1
            assert loaded_metrics == {"loss": 0.5}

            # Check that model weights match
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2), "Loaded model should match saved model"


class TestPUTrainerReproducibility:
    """Test training reproducibility."""

    def test_same_seed_identical_training(self, simple_dataset):
        """Test that same seed produces identical training results."""
        device = get_device()

        # First training run
        set_seed(42)
        model1 = MLP6Layer().to(device)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        trainer1 = PUTrainer(
            model=model1,
            loss_fn=PNLoss(),
            optimizer=optimizer1,
            device=device,
        )

        dataloader1 = DataLoader(simple_dataset, batch_size=32, shuffle=True)
        loss1_epoch0 = trainer1.train_epoch(dataloader1, epoch=0)
        loss1_epoch1 = trainer1.train_epoch(dataloader1, epoch=1)

        # Second training run with same seed
        set_seed(42)
        model2 = MLP6Layer().to(device)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        trainer2 = PUTrainer(
            model=model2,
            loss_fn=PNLoss(),
            optimizer=optimizer2,
            device=device,
        )

        dataloader2 = DataLoader(simple_dataset, batch_size=32, shuffle=True)
        loss2_epoch0 = trainer2.train_epoch(dataloader2, epoch=0)
        loss2_epoch1 = trainer2.train_epoch(dataloader2, epoch=1)

        # Losses should be identical
        assert abs(loss1_epoch0 - loss2_epoch0) < 1e-6, (
            f"Same seed should produce identical loss: {loss1_epoch0} vs {loss2_epoch0}"
        )
        assert abs(loss1_epoch1 - loss2_epoch1) < 1e-6

        # Model weights should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), (
                "Same seed should produce identical model weights"
            )


class TestPUTrainerWithPULosses:
    """Test trainer with PU-specific losses."""

    def test_trainer_with_nnpu_loss(self, simple_dataset):
        """Test that trainer works with nnPU loss."""
        set_seed(42)
        device = get_device()

        model = MLP6Layer().to(device)
        loss_fn = nnPULoss(prior=0.5)  # 50% class prior
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = PUTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        dataloader = DataLoader(simple_dataset, batch_size=32)

        # Should train without errors
        loss = trainer.train_epoch(dataloader, epoch=0)

        assert isinstance(loss, float)
        assert loss >= 0, "nnPU loss should be non-negative"

    def test_nnpu_loss_stays_nonnegative_during_training(self, simple_dataset):
        """Test that nnPU loss remains non-negative throughout training."""
        set_seed(42)
        device = get_device()

        model = MLP6Layer().to(device)
        loss_fn = nnPULoss(prior=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = PUTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        dataloader = DataLoader(simple_dataset, batch_size=32)

        # Train for multiple epochs
        for epoch in range(10):
            epoch_loss = trainer.train_epoch(dataloader, epoch)
            assert epoch_loss >= 0, (
                f"nnPU loss should always be non-negative, got {epoch_loss} at epoch {epoch}"
            )
