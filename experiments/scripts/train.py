"""Main training script for PU learning experiments.

Usage:
    python experiments/scripts/train.py --config configs/experiments/mnist_pn.yaml
    python experiments/scripts/train.py --config configs/experiments/mnist_upu.yaml --seed 43
    python experiments/scripts/train.py --config configs/experiments/mnist_nnpu.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pu_learning.data.datasets import MNISTPUDataset
from pu_learning.losses import PNLoss, nnPULoss, uPULoss
from pu_learning.models import MLP6Layer
from pu_learning.training.trainer import PUTrainer
from pu_learning.utils.device import get_device
from pu_learning.utils.reproducibility import set_seed


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)
    return config


def create_loss_function(config: dict):
    """Create loss function based on config.

    Args:
        config: Configuration dictionary.

    Returns:
        Loss function instance.
    """
    loss_type = config["loss"]["type"].lower()

    if loss_type == "pn":
        return PNLoss()
    elif loss_type == "upu":
        prior = config["loss"]["prior"]
        return uPULoss(prior=prior)
    elif loss_type == "nnpu":
        prior = config["loss"]["prior"]
        return nnPULoss(prior=prior)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer based on config.

    Args:
        model: Model to optimize.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    optimizer_type = config["training"]["optimizer"].lower()
    lr = config["training"]["learning_rate"]

    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PU learning model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = args.seed if args.seed is not None else config["seed"]
    set_seed(seed)
    print(f"\n{'=' * 80}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Seed: {seed}")
    print(f"{'=' * 80}\n")

    # Get device
    device = get_device()

    # Create dataset
    print("Creating dataset...")
    train_dataset = MNISTPUDataset(
        root="./data",
        train=True,
        n_positive=config["data"]["n_positive"],
        seed=seed,
    )

    test_dataset = MNISTPUDataset(
        root="./data",
        train=False,
        n_positive=config["data"]["n_positive"],
        seed=seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Class prior π: {train_dataset.class_prior:.4f}\n")

    # Create model
    print("Creating model...")
    model = MLP6Layer().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create loss function
    loss_fn = create_loss_function(config)
    print(f"Loss function: {loss_fn.__class__.__name__}")
    if hasattr(loss_fn, "prior"):
        print(f"Prior π: {loss_fn.prior}\n")
    else:
        print()

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {config['training']['learning_rate']}\n")

    # Create trainer
    trainer = PUTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    # Training loop
    print("Starting training...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Test Loss':>12} {'Test Acc':>10} {'Test Err':>10}")
    print("-" * 80)

    results = {
        "config": config,
        "seed": seed,
        "epochs": [],
    }

    log_interval = config["logging"]["log_interval"]

    for epoch in range(config["training"]["epochs"]):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Evaluate
        if epoch % log_interval == 0 or epoch == config["training"]["epochs"] - 1:
            test_metrics = trainer.evaluate(test_loader)

            print(
                f"{epoch:6d} {train_loss:12.6f} {test_metrics['loss']:12.6f} "
                f"{test_metrics['accuracy']:10.4f} {test_metrics['zero_one_loss']:10.4f}"
            )

            # Store results
            results["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_zero_one_loss": test_metrics["zero_one_loss"],
                }
            )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Results:")
    final_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {final_metrics['loss']:.6f}")
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Test Error (Zero-One Loss): {final_metrics['zero_one_loss']:.4f}")
    print("=" * 80 + "\n")

    # Save results
    if config["logging"]["save_model"]:
        checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        checkpoint_path = checkpoint_dir / f"model_seed{seed}.pt"
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=config["training"]["epochs"] - 1,
            metrics=final_metrics,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save results JSON
        results_path = checkpoint_dir / f"results_seed{seed}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results: {results_path}\n")


if __name__ == "__main__":
    main()
