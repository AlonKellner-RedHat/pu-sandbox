"""Analyze experiment results and generate visualizations.

Usage:
    python experiments/scripts/analyze_results.py experiments/results
    python experiments/scripts/analyze_results.py experiments/results --export
"""

import argparse
from pathlib import Path

from pu_learning.utils.visualization import (
    export_plots,
    load_experiment_results,
    print_summary_statistics,
)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze PU learning experiment results")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export plots to HTML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/plots",
        help="Directory to save HTML plots (default: experiments/results/plots)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load and print summary statistics
    print_summary_statistics(results_dir)

    # Load data
    df = load_experiment_results(results_dir)

    if df.empty:
        print("No experiment results found.")
        return

    print(f"\nLoaded {len(df)} data points from {df['seed'].nunique()} seeds")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Epochs: {df['epoch'].min()} to {df['epoch'].max()}")

    # Export plots if requested
    if args.export:
        print(f"\nExporting plots to {args.output_dir}...")
        export_plots(results_dir, args.output_dir)
        print("\n✅ Export complete!")
    else:
        print("\n💡 Tip: Use --export to save interactive plots as HTML files")


if __name__ == "__main__":
    main()
