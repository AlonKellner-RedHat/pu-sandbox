# Tasks: nnPU Paper Reproduction & PU Learning Research

**Input**: Design documents from `.specify/memory/` (constitution.md, specification.md, plan.md)
**Prerequisites**: ✓ constitution.md, ✓ specification.md, ✓ plan.md

**Organization**: Tasks are grouped by implementation phase and user story to enable systematic implementation.

## Format: `[ID] [P?] [Phase/Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Phase/Story]**: Implementation phase or user story (e.g., P1, US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Project Setup & Core Infrastructure

**Purpose**: Initialize project structure, dependencies, and foundational utilities

**⚠️ CRITICAL**: Must be complete before any ML implementation can begin

- [ ] T001 Create `pyproject.toml` with all dependencies (PyTorch, hvplot, holoviews, bokeh, pytest, ruff, ty)
- [ ] T002 Create package structure: `src/pu_learning/` with all module directories and `__init__.py` files
- [ ] T003 [P] Create `tests/` directory structure (unit/, integration/, conftest.py)
- [ ] T004 [P] Create `configs/experiments/` directory
- [ ] T005 [P] Create `experiments/scripts/` and `experiments/results/` directories
- [ ] T006 [P] Create `notebooks/` directory
- [ ] T007 [P] Configure ruff in pyproject.toml (line-length=100, formatting rules)
- [ ] T008 [P] Configure ty in pyproject.toml (strict mode, type checking)
- [ ] T009 [P] Configure pytest in pyproject.toml (coverage >80%, test paths)
- [ ] T010 Create `.gitignore` (Python, Jupyter, results, checkpoints, .env)
- [ ] T011 Install project in editable mode: `uv pip install -e ".[dev]"`

**Checkpoint**: Project structure ready for implementation

---

## Phase 2: Device Selection & Reproducibility (US2 - MPS Acceleration, US3 - Reproducibility)

**Purpose**: Core utilities for GPU acceleration and reproducible experiments

**⚠️ CRITICAL**: Required for all training and experiments

### Implementation

- [ ] T012 [P] Implement `src/pu_learning/utils/__init__.py` (empty or exports)
- [ ] T013 [P] Implement `src/pu_learning/utils/device.py`:
  - `get_device(prefer_mps=True)` function
  - Auto-detection: MPS > CUDA > CPU
  - Print device selection message
- [ ] T014 [P] Implement `src/pu_learning/utils/reproducibility.py`:
  - `set_seed(seed: int)` function
  - Set Python random, NumPy, PyTorch CPU/MPS/CUDA seeds
  - Set CUDNN deterministic mode
  - Set PYTHONHASHSEED environment variable

### Tests

- [ ] T015 [P] Test MPS availability in `tests/unit/test_device.py`
- [ ] T016 [P] Test device selection logic (MPS, CUDA, CPU fallback)
- [ ] T017 [P] Test seed setting in `tests/unit/test_reproducibility.py`
- [ ] T018 [P] Test reproducibility: same seed → identical random numbers

### Verification

- [ ] T019 Verify MPS available on M4 MacBook Pro: `torch.backends.mps.is_available() == True`
- [ ] T020 Verify MPS computation works: matrix multiplication on MPS device

**Checkpoint**: Device selection and reproducibility infrastructure ready

---

## Phase 3: Data Pipeline (US1 - nnPU Baseline, US3 - Reproducibility)

**Purpose**: Load MNIST and implement P/U splitting

### Implementation

- [ ] T021 [P] Implement `src/pu_learning/data/__init__.py`
- [ ] T022 Implement `src/pu_learning/data/datasets.py`:
  - `MNISTPUDataset` class
  - Load MNIST with torchvision.datasets.MNIST
  - Filter even digits (0,2,4,6,8) as positive class
  - Sample exactly 100 positive examples with fixed seed
  - Create unlabeled set from remaining ~59,900 samples
  - Calculate class prior π
  - Return (image, label, is_labeled) tuples
  - Normalize: mean=0.1307, std=0.3081
  - Flatten 28×28 to 784-dimensional vectors
- [ ] T023 [P] Implement `src/pu_learning/data/preprocessing.py`:
  - Normalization constants
  - Flattening transforms
  - Helper functions for data transforms

### Tests

- [ ] T024 Test MNIST loading in `tests/unit/test_datasets.py`
- [ ] T025 Test P set size == 100
- [ ] T026 Test U set size ~= 59,900
- [ ] T027 Test positive class contains only even digits
- [ ] T028 Test class prior π calculation
- [ ] T029 Test deterministic splitting: same seed → same split
- [ ] T030 Test data normalization and flattening

### Validation

- [ ] T031 Create `notebooks/01_data_exploration.ipynb`:
  - Load dataset
  - Visualize P/U split
  - Show class distribution
  - Verify data preprocessing

**Checkpoint**: Data pipeline ready and validated

---

## Phase 4: Model Architecture (US1 - nnPU Baseline)

**Purpose**: Implement 6-layer MLP matching nnPU paper specification

### Implementation

- [ ] T032 [P] Implement `src/pu_learning/models/__init__.py`
- [ ] T033 [P] Implement `src/pu_learning/models/base.py` (optional base class)
- [ ] T034 Implement `src/pu_learning/models/mlp.py`:
  - `MLP6Layer` class
  - Architecture: [784 → 300 → 300 → 300 → 300 → 300 → 1]
  - ReLU activations in hidden layers
  - Sigmoid activation in output layer
  - Xavier or Kaiming weight initialization
  - Forward pass returns p(x) ∈ [0,1]

### Tests

- [ ] T035 Test forward pass shape in `tests/unit/test_models.py`
- [ ] T036 Test output range [0, 1]
- [ ] T037 Test parameter count matches architecture
- [ ] T038 Test initialization (weights not NaN/Inf)
- [ ] T039 Test model can be moved to MPS device

**Checkpoint**: Model architecture implemented and tested

---

## Phase 5: Loss Functions (US1 - nnPU Baseline) 🎯 CRITICAL

**Purpose**: Implement PN, uPU, nnPU loss functions with correct mathematical formulations

**⚠️ CRITICAL**: This is the core research contribution - must be mathematically correct

### Implementation

- [ ] T040 [P] Implement `src/pu_learning/losses/__init__.py`
- [ ] T041 Implement `src/pu_learning/losses/base_loss.py`:
  - `PULoss` abstract base class
  - Accept `prior` (π) parameter
  - Abstract `forward()` method
- [ ] T042 Implement `src/pu_learning/losses/pn_loss.py`:
  - `PNLoss` class (supervised baseline)
  - L_PN = E_P[BCE(f(x),1)] + E_N[BCE(f(x),0)]
- [ ] T043 Implement `src/pu_learning/losses/upu_loss.py`:
  - `uPULoss` class (unbiased PU)
  - L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]
  - **Allow negative values** (no max constraint)
- [ ] T044 Implement `src/pu_learning/losses/nnpu_loss.py`:
  - `nnPULoss` class (non-negative PU)
  - L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])
  - **Assert non-negative** in forward pass

### Tests ⚠️ WRITE TESTS FIRST, ENSURE THEY FAIL

- [ ] T045 Test PN loss computation in `tests/unit/test_losses.py`
- [ ] T046 Test uPU loss can produce negative values
- [ ] T047 Test nnPU loss is always non-negative (critical assertion)
- [ ] T048 Test loss mathematical correctness with known inputs
- [ ] T049 Test gradient flow (no NaN/Inf gradients)
- [ ] T050 Test loss functions accept class prior π parameter
- [ ] T051 Test loss computation with edge cases (all P, all U batches)

### Validation

- [ ] T052 Create `notebooks/02_loss_visualization.ipynb`:
  - Plot loss functions
  - Demonstrate negative risk in uPU
  - Show max(0, ...) constraint in nnPU

**Checkpoint**: All three loss functions implemented and validated

---

## Phase 6: Training Pipeline (US1 - nnPU Baseline, US2 - MPS, US3 - Reproducibility)

**Purpose**: Implement trainer, evaluator, and metrics

### Implementation

- [ ] T053 [P] Implement `src/pu_learning/training/__init__.py`
- [ ] T054 Implement `src/pu_learning/training/trainer.py`:
  - `PUTrainer` class
  - Accept (model, loss_fn, optimizer, device, seed)
  - `train_epoch()`: training loop with P/U batch separation
  - `evaluate()`: evaluation on test set
  - `save_checkpoint()`: save model, optimizer, epoch
  - `load_checkpoint()`: restore training state
  - Log metrics per epoch
  - Support early stopping
- [ ] T055 [P] Implement `src/pu_learning/training/evaluator.py`:
  - Evaluation utilities
  - Batch evaluation logic
- [ ] T056 [P] Implement `src/pu_learning/utils/metrics.py`:
  - `zero_one_loss(predictions, targets)`: classification error rate
  - `accuracy(predictions, targets)`: 1 - zero_one_loss
  - Additional metrics (precision, recall, F1) optional
- [ ] T057 [P] Implement `src/pu_learning/utils/logging.py`:
  - Experiment logging with metadata
  - Log: seed, git commit hash, timestamp, hyperparameters, results

### Tests

- [ ] T058 Test training loop in `tests/integration/test_training.py`
- [ ] T059 Test training decreases loss
- [ ] T060 Test checkpoint save/load
- [ ] T061 Test evaluation computes metrics correctly
- [ ] T062 Test zero-one loss computation in `tests/unit/test_metrics.py`
- [ ] T063 Test reproducibility: same seed → identical training

**Checkpoint**: Training pipeline ready for experiments

---

## Phase 7: Experiment Configurations & Scripts (US1 - nnPU Baseline)

**Purpose**: Create configs and scripts to run PN, uPU, nnPU experiments

### Configuration Files

- [ ] T064 [P] Create `configs/experiments/mnist_pn.yaml`:
  - Experiment name: "mnist_pn"
  - Loss type: "pn"
  - 100 positive samples
  - Batch size: 256
  - Epochs: 100
  - Device: "mps"
  - Seed: 42
- [ ] T065 [P] Create `configs/experiments/mnist_upu.yaml`:
  - Experiment name: "mnist_upu"
  - Loss type: "upu"
  - Prior π: 0.5
  - Same data/training params as PN
- [ ] T066 [P] Create `configs/experiments/mnist_nnpu.yaml`:
  - Experiment name: "mnist_nnpu"
  - Loss type: "nnpu"
  - Prior π: 0.5
  - Same data/training params as uPU

### Experiment Scripts

- [ ] T067 [P] Implement `experiments/scripts/run_pn.py`:
  - Load config from YAML
  - Set seed
  - Select device (MPS)
  - Create dataset, model, PN loss
  - Create trainer
  - Train and evaluate
  - Save results
- [ ] T068 [P] Implement `experiments/scripts/run_upu.py`:
  - Same as T067 but with uPU loss
  - Track when loss becomes negative
- [ ] T069 [P] Implement `experiments/scripts/run_nnpu.py`:
  - Same as T067 but with nnPU loss
  - Assert loss stays non-negative
- [ ] T070 Implement `experiments/scripts/run_all.py`:
  - Run all three methods (PN, uPU, nnPU)
  - Support multiple seeds via CLI argument
  - Aggregate results
  - Generate summary statistics

### Run Experiments

- [ ] T071 Run PN experiment with seeds [42, 43, 44, 45, 46]
- [ ] T072 Run uPU experiment with seeds [42, 43, 44, 45, 46]
- [ ] T073 Run nnPU experiment with seeds [42, 43, 44, 45, 46]
- [ ] T074 Verify results reproducibility (same seed → same results)
- [ ] T075 Verify uPU loss becomes negative
- [ ] T076 Verify nnPU loss stays non-negative
- [ ] T077 Verify nnPU performance comparable to/better than uPU

**Checkpoint**: All baseline experiments complete and validated against paper findings

---

## Phase 8: Interactive Visualization (US4 - Interactive Result Analysis)

**Purpose**: Create interactive visualizations with hvPlot + HoloViews + Bokeh

### Implementation

- [ ] T078 Implement `src/pu_learning/utils/visualization.py`:
  - `plot_training_curves(results_df)`: interactive line plots
    - Loss vs epoch for all methods
    - Test error vs epoch
    - Hover tooltips with exact values
    - Pan/zoom enabled
    - Toggleable legends
  - `plot_loss_distribution(loss_history)`: interactive histogram
    - Loss value distribution by method
    - Highlight negative values for uPU
    - Vertical line at zero
  - `create_comparison_dashboard(results_df)`: Panel dashboard
    - Training curves
    - Final results bar chart
    - Statistical summary table

### Analysis Notebook

- [ ] T079 Create `notebooks/03_results_analysis.ipynb`:
  - Load experiment results from all methods
  - Create interactive training curves
  - Create loss distribution plots
  - Create comparison dashboard
  - Statistical analysis (mean ± std across seeds)
  - Export interactive plots to HTML
  - Compare PN vs uPU vs nnPU performance

### Export

- [ ] T080 Export `training_loss_curves.html`
- [ ] T081 Export `test_error_curves.html`
- [ ] T082 Export `loss_distribution.html`
- [ ] T083 Export `comparison_dashboard.html`

**Checkpoint**: Interactive visualization complete, results analyzed

---

## Phase 9: Documentation & README

**Purpose**: Document project for reproducibility and sharing

- [ ] T084 Create `README.md`:
  - Project overview and motivation
  - Installation instructions (uv, dependencies)
  - Quick start guide
  - Reproduction instructions
  - Results summary
  - Citation to nnPU paper
- [ ] T085 [P] Add docstrings to all public functions/classes (Google-style)
- [ ] T086 [P] Add type hints to all function signatures
- [ ] T087 [P] Run ruff format on all code
- [ ] T088 [P] Run ruff check and fix linting issues
- [ ] T089 [P] Run ty and fix type errors
- [ ] T090 Run pytest and verify >80% coverage
- [ ] T091 Verify all tests pass

**Checkpoint**: Project fully documented and ready for sharing

---

## Phase 10: Custom Loss Extension (US5 - Implement Custom PU Loss) - FUTURE

**Purpose**: Implement user's custom PU loss and compare against nnPU

**Note**: Only start after Phase 1-9 complete and baseline validated

### Implementation

- [ ] T092 Implement `src/pu_learning/losses/custom_loss.py`:
  - `CustomPULoss` class
  - L(1, p) = -log(p) + p
  - L(0, p) = p
  - Combined: L_custom = π·E_P[-log(p(x)) + p(x)] + E_U[p(x)]

### Tests

- [ ] T093 Test custom loss mathematical correctness
- [ ] T094 Test gradient flow

### Configuration & Experiments

- [ ] T095 Create `configs/experiments/mnist_custom.yaml`
- [ ] T096 Implement `experiments/scripts/run_custom.py`
- [ ] T097 Run custom loss experiments with seeds [42, 43, 44, 45, 46]
- [ ] T098 Compare custom loss vs nnPU baseline in analysis notebook
- [ ] T099 Document performance difference

**Checkpoint**: Custom loss implemented and compared against nnPU

---

## Summary

**Total Tasks**: 99
**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10

**Parallelization Opportunities**:
- Phase 1: T003-T009 can run in parallel (directory creation, config files)
- Phase 2: T012-T014 implementation in parallel, T015-T018 tests in parallel
- Phase 3: T023 preprocessing in parallel with T022 datasets
- Phase 4: T032-T033 can run in parallel
- Phase 5: T040-T042 can run in parallel (base, PN, uPU, nnPU are independent)
- Phase 6: T053-T057 implementations in parallel
- Phase 7: T064-T069 all in parallel (configs and scripts)

**Priority Order**:
1. **P1 - Must Have**: Phases 1-7 (baseline reproduction)
2. **P2 - Should Have**: Phase 8 (visualization)
3. **P3 - Nice to Have**: Phase 9 (documentation)
4. **P4 - Future**: Phase 10 (custom loss)
