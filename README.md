# PU Learning Research Project

PyTorch implementation of Positive-Unlabeled (PU) learning methods, focusing on reproducing the nnPU paper experiments.

## Overview

This project implements and compares three PU learning methods:
- **PN (Positive-Negative)**: Traditional supervised learning baseline
- **uPU (Unbiased PU)**: Unbiased risk estimator (can produce negative risk)
- **nnPU (Non-Negative PU)**: Proposed method using max(0, ...) to ensure non-negative risk

After successfully reproducing the baseline results, we will introduce and evaluate a custom PU loss function.

## Quick Start

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0.0 with MPS support (for M4 MacBook Pro)
- [just](https://github.com/casey/just) command runner

### Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install pre-commit hooks:
   ```bash
   just install-hooks
   ```

3. Verify MPS support:
   ```bash
   just verify-mps
   ```

## Development Workflow

### Common Commands

Using the `justfile` for streamlined development:

- **`just p`** - Stage all changes and run pre-commit hooks (use this often!)
- **`just test`** - Run tests with coverage
- **`just lint`** - Run ruff linter
- **`just format`** - Format code with ruff
- **`just all`** - Full workflow: format, lint, test, pre-commit
- **`just --list`** - Show all available commands

### TDD Workflow

This project follows strict Test-Driven Development (TDD):

1. **RED**: Write a failing test first
2. **GREEN**: Implement minimal code to make it pass
3. **REFACTOR**: Improve code quality while keeping tests green
4. **Verify**: Run `just p` to ensure pre-commit hooks pass
5. **Commit**: Commit with meaningful message

### Pre-Commit Hooks

Pre-commit hooks automatically:
- Fix trailing whitespace
- Fix end of files
- Run ruff linter with auto-fix
- Format code with ruff
- Run tests with coverage (pre-push only)
- Detect secrets

Use `just p` frequently during development to catch issues early!

## Project Structure

```
pu-sandbox/
├── src/pu_learning/          # Main package
│   ├── data/                 # MNIST PU dataset
│   ├── models/               # 6-layer MLP architecture
│   ├── losses/               # PN, uPU, nnPU loss functions
│   ├── training/             # Trainer and evaluator
│   └── utils/                # Device selection, reproducibility, metrics
├── tests/                    # Unit and integration tests
├── experiments/              # Experiment scripts and results
├── configs/                  # Experiment configurations
└── notebooks/                # Analysis notebooks
```

## Current Status

✅ **All Core Phases Complete (1-8)**:
- ✅ **Phase 1-3**: Foundation (spec-kit workflow, core utilities, MNIST PU dataset)
- ✅ **Phase 4**: Model Architecture (6-layer MLP with 597K parameters)
- ✅ **Phase 5**: Loss Functions (PN, uPU, nnPU with mathematical correctness)
- ✅ **Phase 6**: Training Pipeline (trainer, metrics, checkpoints)
- ✅ **Phase 7**: Experiment Configurations (YAML configs, training scripts)
- ✅ **Phase 8**: Interactive Visualization (hvPlot + HoloViews + Bokeh)

**Testing**: 76 tests passing (53 unit + 14 metrics + 9 integration), >80% coverage

🎯 **Ready**: Full nnPU paper reproduction pipeline is functional and tested!

## Technology Stack

### Core Dependencies
- **PyTorch 2.10.0**: Deep learning with MPS acceleration
- **torchvision**: MNIST dataset
- **NumPy**: Numerical operations
- **PyYAML**: Configuration management

### Visualization (HoloViz Stack)
- **hvPlot**: High-level interactive plotting
- **HoloViews**: Declarative visualizations
- **Bokeh**: Interactive visualization backend
- **Pandas**: Data manipulation for analysis

### Code Quality (Astral.sh Stack)
- **uv**: Fast package management
- **ruff**: Linting and formatting
- **pytest**: Testing framework (76 tests passing)
- **pre-commit**: Automated quality checks

## Implementation Details

### Model Architecture
- **MLP6Layer**: [784 → 300 → 300 → 300 → 300 → 300 → 1]
- **Activations**: ReLU (hidden), Sigmoid (output)
- **Parameters**: 597,001
- **Initialization**: Kaiming (He) for ReLU layers

### Loss Functions

All loss functions implement the exact formulations from the nnPU paper:

**PN (Supervised Baseline)**:
```
L_PN = E_P[BCE(f(x), 1)] + E_N[BCE(f(x), 0)]
```

**uPU (Unbiased PU)**:
```
L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]
```
- Can produce negative risk (may lead to overfitting)

**nnPU (Non-Negative PU)** - Key Contribution:
```
L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])
```
- Always non-negative (assertion enforced in code)
- Prevents overfitting to negative risk

Where:
- `π` = class prior P(y=1) ≈ 0.5 for even/odd split
- `l(z)` = sigmoid loss = log(1 + exp(-z))
- `P` = positive labeled samples (100)
- `U` = unlabeled samples (~59,900)

### Dataset Configuration
- **Positive class**: Even digits (0, 2, 4, 6, 8)
- **Negative class**: Odd digits (1, 3, 5, 7, 9)
- **Labeled positive**: 100 samples
- **Unlabeled**: ~59,900 samples (contains both positive and negative)
- **Class prior π**: ~0.49 (approximately balanced)

### Training Configuration
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Batch size**: 256
- **Epochs**: 100 (10 for quick tests)
- **Device**: MPS (Metal) > CUDA > CPU

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds for Python, NumPy, PyTorch (CPU/MPS/CUDA)
- Deterministic algorithms enabled
- Version-locked dependencies in `uv.lock`
- MPS acceleration for M4 MacBook Pro

## Testing

Run tests:
```bash
just test              # With coverage
just test-fast         # Without coverage (faster)
just test-file <path>  # Specific test file
```

Current coverage: **80%** (exceeds 80% threshold)

## Running Experiments

### Quick Test (10 epochs)

Run all three methods with 2 seeds (fast verification):
```bash
python experiments/scripts/run_all.py --seeds 42 43 --quick
```

### Full Experiments (100 epochs)

Run complete reproduction experiments with 5 seeds:
```bash
python experiments/scripts/run_all.py --seeds 42 43 44 45 46
```

### Single Method

Run a specific method:
```bash
# nnPU (recommended)
python experiments/scripts/train.py --config configs/experiments/mnist_nnpu.yaml

# uPU (unbiased)
python experiments/scripts/train.py --config configs/experiments/mnist_upu.yaml

# PN (baseline)
python experiments/scripts/train.py --config configs/experiments/mnist_pn.yaml
```

### Custom Seed

Override the config seed:
```bash
python experiments/scripts/train.py --config configs/experiments/mnist_nnpu.yaml --seed 99
```

## Analyzing Results

### Generate Summary Statistics

```bash
python experiments/scripts/analyze_results.py experiments/results
```

### Export Interactive Visualizations

Generate HTML plots with hvPlot + HoloViews + Bokeh:
```bash
python experiments/scripts/analyze_results.py experiments/results --export
```

This creates:
- `training_loss.html`: Training loss curves
- `test_loss.html`: Test loss curves
- `test_error.html`: Test error (zero-one loss) curves
- `test_accuracy.html`: Test accuracy curves
- `final_performance.html`: Bar chart comparing final performance
- `comparison_dashboard.html`: Complete dashboard with all metrics

### View Results

Open any HTML file in your browser for interactive exploration:
```bash
open experiments/results/plots/comparison_dashboard.html
```

Features:
- **Hover tooltips** with exact values
- **Pan and zoom** to explore specific regions
- **Toggleable legends** to show/hide methods
- **Mean ± std** visualization across multiple seeds

## Contributing

This project follows the constitution defined in `.specify/memory/constitution.md`:
- TDD methodology (RED-GREEN-REFACTOR)
- Commit early, commit often
- All commits co-authored with Claude Code
- >80% test coverage requirement
- Use `just p` before committing

## License

MIT

## Acknowledgments

Based on the nnPU paper:
> Kiryo, R., Niu, G., du Plessis, M. C., & Sugiyama, M. (2017).
> Positive-Unlabeled Learning with Non-Negative Risk Estimator.
> NIPS 2017. [arXiv:1703.00593](https://arxiv.org/abs/1703.00593)

Developed with [Claude Code](https://claude.com/claude-code) using spec-kit workflow.
