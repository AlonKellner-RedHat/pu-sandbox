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

✅ **Phase 1-3 Complete**:
- Spec-kit workflow setup (constitution, specification, plan, tasks)
- Core utilities (MPS device selection, reproducibility)
- MNIST PU dataset with comprehensive tests (30 passing tests, 80% coverage)

🚧 **Next Phases**:
- Phase 4: Model Architecture (6-layer MLP)
- Phase 5: Loss Functions (PN, uPU, nnPU)
- Phase 6: Training Pipeline

## Technology Stack

### Core Dependencies
- **PyTorch 2.10.0**: Deep learning with MPS acceleration
- **torchvision**: MNIST dataset
- **NumPy**: Numerical operations

### Visualization
- **hvPlot**: High-level interactive plotting
- **HoloViews**: Declarative visualizations
- **Bokeh**: Interactive visualization backend

### Code Quality (Astral.sh Stack)
- **uv**: Fast package management
- **ruff**: Linting and formatting
- **pytest**: Testing framework
- **pre-commit**: Automated quality checks

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
