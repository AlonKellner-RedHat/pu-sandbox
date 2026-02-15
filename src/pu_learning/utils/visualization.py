"""Interactive visualization utilities using hvPlot + HoloViews + Bokeh.

This module provides interactive plotting functions for analyzing PU learning experiments.
All visualizations follow the project constitution: hvPlot + HoloViews + Bokeh stack only.
"""

import json
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa: F401 - enables pandas.hvplot
import pandas as pd

# Set HoloViews backend to Bokeh
hv.extension("bokeh")


def load_experiment_results(results_dir: Path | str) -> pd.DataFrame:
    """Load all experiment results from directory into a DataFrame.

    Args:
        results_dir: Directory containing results JSON files.

    Returns:
        DataFrame with columns: method, seed, epoch, train_loss, test_loss,
                                test_accuracy, test_zero_one_loss.
    """
    results_dir = Path(results_dir)
    all_data = []

    # Find all result JSON files
    for json_file in results_dir.rglob("results_seed*.json"):
        with json_file.open() as f:
            data = json.load(f)

        method = data["config"]["experiment"]["name"]
        # Extract method type (pn, upu, nnpu) from name
        if "pn" in method and "nnpu" not in method and "upu" not in method:
            method_type = "PN"
        elif "upu" in method and "nnpu" not in method:
            method_type = "uPU"
        elif "nnpu" in method:
            method_type = "nnPU"
        else:
            method_type = "Unknown"

        seed = data["seed"]

        # Extract epoch data
        for epoch_data in data["epochs"]:
            all_data.append(
                {
                    "method": method_type,
                    "seed": seed,
                    "epoch": epoch_data["epoch"],
                    "train_loss": epoch_data["train_loss"],
                    "test_loss": epoch_data["test_loss"],
                    "test_accuracy": epoch_data["test_accuracy"],
                    "test_zero_one_loss": epoch_data["test_zero_one_loss"],
                }
            )

    return pd.DataFrame(all_data)


def plot_training_curves(
    df: pd.DataFrame,
    metric: str = "test_zero_one_loss",
    title: str | None = None,
    width: int = 800,
    height: int = 400,
) -> hv.Overlay:
    """Plot training curves for all methods with interactive features.

    Args:
        df: DataFrame with experiment results.
        metric: Metric to plot (train_loss, test_loss, test_accuracy, test_zero_one_loss).
        title: Plot title. If None, auto-generated.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        HoloViews overlay plot with interactive features.
    """
    if title is None:
        metric_names = {
            "train_loss": "Training Loss",
            "test_loss": "Test Loss",
            "test_accuracy": "Test Accuracy",
            "test_zero_one_loss": "Test Error (Zero-One Loss)",
        }
        title = metric_names.get(metric, metric)

    # Group by method and compute mean + std across seeds
    grouped = df.groupby(["method", "epoch"])[metric].agg(["mean", "std"]).reset_index()

    # Create plot for each method
    plots = []
    colors = {"PN": "#1f77b4", "uPU": "#ff7f0e", "nnPU": "#2ca02c"}

    for method in grouped["method"].unique():
        method_data = grouped[grouped["method"] == method]

        # Main line plot
        line = method_data.hvplot.line(
            x="epoch",
            y="mean",
            label=method,
            color=colors.get(method, "gray"),
            line_width=2,
            width=width,
            height=height,
            title=title,
            xlabel="Epoch",
            ylabel=title,
            hover_cols=["method", "mean", "std"],
        )

        # Shaded error region (mean ± std)
        if not method_data["std"].isna().all():
            # Create upper and lower bounds
            method_data_copy = method_data.copy()
            method_data_copy["upper"] = method_data_copy["mean"] + method_data_copy["std"]
            method_data_copy["lower"] = method_data_copy["mean"] - method_data_copy["std"]

            # Area plot for error region
            area = method_data_copy.hvplot.area(
                x="epoch",
                y="lower",
                y2="upper",
                alpha=0.2,
                color=colors.get(method, "gray"),
                hover=False,
            )

            plots.append(area * line)
        else:
            plots.append(line)

    # Overlay all plots
    if plots:
        overlay = plots[0]
        for plot in plots[1:]:
            overlay = overlay * plot
        return overlay
    else:
        # Return empty plot if no data
        return hv.Curve([]).opts(width=width, height=height, title=title)


def plot_final_performance(
    df: pd.DataFrame,
    width: int = 600,
    height: int = 400,
) -> hv.Bars:
    """Plot final performance comparison across methods.

    Args:
        df: DataFrame with experiment results.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        HoloViews bar plot with error bars.
    """
    # Get final epoch for each method and seed
    final_results = df.loc[df.groupby(["method", "seed"])["epoch"].idxmax()]

    # Compute mean and std across seeds
    summary = (
        final_results.groupby("method")["test_zero_one_loss"].agg(["mean", "std"]).reset_index()
    )
    summary.columns = ["method", "error_rate", "std"]

    # Create bar plot
    bars = summary.hvplot.bar(
        x="method",
        y="error_rate",
        color="method",
        cmap={"PN": "#1f77b4", "uPU": "#ff7f0e", "nnPU": "#2ca02c"},
        width=width,
        height=height,
        title="Final Test Error Rate (Lower is Better)",
        xlabel="Method",
        ylabel="Test Error Rate",
        ylim=(0, None),
        legend=False,
    )

    return bars


def create_comparison_dashboard(results_dir: Path | str) -> hv.Layout:
    """Create comprehensive comparison dashboard.

    Args:
        results_dir: Directory containing experiment results.

    Returns:
        HoloViews layout with multiple plots.
    """
    # Load data
    df = load_experiment_results(results_dir)

    if df.empty:
        print(f"No results found in {results_dir}")
        return hv.Layout([])

    # Create plots
    train_loss_plot = plot_training_curves(df, "train_loss", title="Training Loss")
    test_loss_plot = plot_training_curves(df, "test_loss", title="Test Loss")
    test_error_plot = plot_training_curves(
        df, "test_zero_one_loss", title="Test Error (Zero-One Loss)"
    )
    test_acc_plot = plot_training_curves(df, "test_accuracy", title="Test Accuracy")
    final_perf_plot = plot_final_performance(df)

    # Create layout
    layout = (
        train_loss_plot + test_loss_plot + test_error_plot + test_acc_plot + final_perf_plot
    ).cols(2)

    return layout


def export_plots(
    results_dir: Path | str,
    output_dir: Path | str = "experiments/results/plots",
) -> None:
    """Export all plots to HTML files.

    Args:
        results_dir: Directory containing experiment results.
        output_dir: Directory to save HTML plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_experiment_results(results_dir)

    if df.empty:
        print(f"No results found in {results_dir}")
        return

    # Export individual plots
    plots = {
        "training_loss": plot_training_curves(df, "train_loss", title="Training Loss"),
        "test_loss": plot_training_curves(df, "test_loss", title="Test Loss"),
        "test_error": plot_training_curves(
            df, "test_zero_one_loss", title="Test Error (Zero-One Loss)"
        ),
        "test_accuracy": plot_training_curves(df, "test_accuracy", title="Test Accuracy"),
        "final_performance": plot_final_performance(df),
    }

    for name, plot in plots.items():
        output_path = output_dir / f"{name}.html"
        hv.save(plot, output_path, backend="bokeh")
        print(f"Saved: {output_path}")

    # Export dashboard
    dashboard = create_comparison_dashboard(results_dir)
    dashboard_path = output_dir / "comparison_dashboard.html"
    hv.save(dashboard, dashboard_path, backend="bokeh")
    print(f"Saved: {dashboard_path}")


def print_summary_statistics(results_dir: Path | str) -> None:
    """Print summary statistics for all experiments.

    Args:
        results_dir: Directory containing experiment results.
    """
    df = load_experiment_results(results_dir)

    if df.empty:
        print(f"No results found in {results_dir}")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Get final results
    final_results = df.loc[df.groupby(["method", "seed"])["epoch"].idxmax()]

    # Group by method
    for method in sorted(final_results["method"].unique()):
        method_data = final_results[final_results["method"] == method]

        print(f"\n{method}:")
        print(f"  Seeds: {sorted(method_data['seed'].unique())}")
        print(
            f"  Test Error: {method_data['test_zero_one_loss'].mean():.4f} "
            f"± {method_data['test_zero_one_loss'].std():.4f}"
        )
        print(
            f"  Test Accuracy: {method_data['test_accuracy'].mean():.4f} "
            f"± {method_data['test_accuracy'].std():.4f}"
        )
        print(
            f"  Test Loss: {method_data['test_loss'].mean():.4f} "
            f"± {method_data['test_loss'].std():.4f}"
        )

    print("\n" + "=" * 80 + "\n")
