"""Run all experiments with multiple seeds.

This script runs PN, uPU, and nnPU experiments with multiple random seeds
to verify reproducibility and compare performance.

Usage:
    python experiments/scripts/run_all.py --seeds 42 43 44
    python experiments/scripts/run_all.py --seeds 42 43 44 45 46 --quick
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(config_path: str, seed: int) -> int:
    """Run a single experiment with given config and seed.

    Args:
        config_path: Path to config file.
        seed: Random seed.

    Returns:
        Return code from subprocess (0 = success).
    """
    cmd = [
        sys.executable,
        "experiments/scripts/train.py",
        "--config",
        config_path,
        "--seed",
        str(seed),
    ]

    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 80}")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description="Run all PU learning experiments")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds to use (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use test configs with fewer epochs",
    )
    args = parser.parse_args()

    # Determine which configs to use
    if args.quick:
        configs = {
            "PN": "configs/experiments/mnist_pn_test.yaml",
            "uPU": "configs/experiments/mnist_upu_test.yaml",
            "nnPU": "configs/experiments/mnist_nnpu_test.yaml",
        }
        # Create quick test configs if they don't exist
        for name, path in configs.items():
            if not Path(path).exists():
                print(f"Creating quick test config: {path}")
                # Create test config with fewer epochs
                full_config = path.replace("_test.yaml", ".yaml")
                with Path(full_config).open() as f:
                    content = f.read()
                # Replace epochs: 100 with epochs: 10
                content = content.replace("epochs: 100", "epochs: 10")
                # Replace log_interval: 10 with log_interval: 2
                content = content.replace("log_interval: 10", "log_interval: 2")
                # Replace checkpoint_dir path
                content = content.replace(
                    f'checkpoint_dir: "experiments/results/mnist_{name.lower()}"',
                    f'checkpoint_dir: "experiments/results/mnist_{name.lower()}_test"',
                )
                with Path(path).open("w") as f:
                    f.write(content)
    else:
        configs = {
            "PN": "configs/experiments/mnist_pn.yaml",
            "uPU": "configs/experiments/mnist_upu.yaml",
            "nnPU": "configs/experiments/mnist_nnpu.yaml",
        }

    # Run experiments
    total_experiments = len(configs) * len(args.seeds)
    experiment_num = 0

    print(f"\n{'=' * 80}")
    print(f"Running {total_experiments} experiments")
    print(f"Methods: {list(configs.keys())}")
    print(f"Seeds: {args.seeds}")
    print(f"{'=' * 80}\n")

    failed_experiments = []

    for method_name, config_path in configs.items():
        for seed in args.seeds:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] {method_name} with seed {seed}")

            returncode = run_experiment(config_path, seed)

            if returncode != 0:
                failed_experiments.append(f"{method_name} (seed={seed})")
                print(f"❌ Failed: {method_name} (seed={seed})")
            else:
                print(f"✓ Success: {method_name} (seed={seed})")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {total_experiments - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
        sys.exit(1)
    else:
        print("\n✅ All experiments completed successfully!")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
