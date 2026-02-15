# Feature Specification: nnPU Paper Reproduction & Custom PU Loss Implementation

**Feature Branch**: `main`
**Created**: 2026-02-15
**Status**: Draft
**Input**: Reproduce nnPU paper experiments (PN, uPU, nnPU on MNIST) with PyTorch, then implement custom PU loss

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reproduce nnPU Baseline Experiments (Priority: P1)

As a machine learning researcher, I want to reproduce the nnPU paper's baseline experiments comparing PN, uPU, and nnPU methods on MNIST, so that I can validate the methodology before extending it with my own loss function.

**Why this priority**: This is the foundation for all future work. Without a validated baseline, we cannot meaningfully compare custom methods or trust results.

**Independent Test**: Can be fully tested by running experiments with all three methods (PN, uPU, nnPU) on MNIST and verifying that:
- uPU loss becomes negative during training
- nnPU loss stays non-negative
- nnPU achieves comparable/better performance than uPU
- Results align with nnPU paper findings

**Acceptance Scenarios**:

1. **Given** MNIST dataset with even digits as positive class, **When** I sample 100 positive examples and create unlabeled set from remaining data, **Then** I get exactly 100 labeled positive samples and ~59,900 unlabeled samples with correct class distribution

2. **Given** a configured PN experiment, **When** I run training for 100 epochs with seed 42, **Then** the model trains successfully using both positive and negative labels and achieves >95% accuracy on test set

3. **Given** a configured uPU experiment, **When** I run training for 100 epochs, **Then** the training loss becomes negative at some point during training (demonstrating the negative risk phenomenon)

4. **Given** a configured nnPU experiment, **When** I run training for 100 epochs, **Then** the training loss remains non-negative throughout all epochs (verified by assertion in code)

5. **Given** completed experiments for all three methods with 5 different seeds, **When** I compare final test errors, **Then** nnPU achieves error comparable to or better than uPU, validating the paper's claims

6. **Given** experiment results, **When** I view interactive visualizations, **Then** I can explore training curves with hover tooltips, zoom into specific regions, and export plots as HTML files

---

### User Story 2 - GPU Acceleration on M4 MacBook Pro (Priority: P1)

As a researcher with an M4 MacBook Pro, I want training to use MPS (Metal Performance Shaders) GPU acceleration automatically, so that experiments run faster than CPU-only training.

**Why this priority**: Critical for practical use. CPU training on MNIST with deep networks would be prohibitively slow. MPS acceleration is essential for iterative experimentation.

**Independent Test**: Can be tested by running any single experiment and verifying:
- MPS device is detected and selected
- Training runs on MPS (verified by GPU activity in Activity Monitor)
- Training is significantly faster than CPU baseline

**Acceptance Scenarios**:

1. **Given** PyTorch with MPS support on M4 MacBook Pro, **When** I initialize the training pipeline, **Then** the system automatically detects and selects MPS as the device

2. **Given** a model and data on MPS device, **When** I run one training epoch, **Then** the GPU utilization shows in Activity Monitor → GPU History

3. **Given** the same experiment configuration, **When** I compare MPS vs CPU training time, **Then** MPS training is at least 5x faster

---

### User Story 3 - Reproducible Experiments (Priority: P1)

As a researcher, I want all experiments to be fully reproducible with fixed seeds, so that I can verify results and share reproducible findings with collaborators.

**Why this priority**: Reproducibility is fundamental to scientific validity. Without it, we cannot trust results or debug issues.

**Independent Test**: Can be tested by running the same experiment configuration twice with the same seed and verifying identical results (loss values, predictions, final metrics).

**Acceptance Scenarios**:

1. **Given** an experiment configuration with seed=42, **When** I run the experiment twice, **Then** I get identical loss values at every epoch

2. **Given** trained models from two runs with same seed, **When** I evaluate both on the test set, **Then** they produce identical predictions for every sample

3. **Given** experiment results logged with metadata, **When** I inspect the logs, **Then** I can see: seed, git commit hash, timestamp, all hyperparameters, and final metrics

---

### User Story 4 - Interactive Result Analysis (Priority: P2)

As a researcher, I want to analyze experiment results using interactive visualizations in Jupyter notebooks, so that I can explore training dynamics and compare methods effectively.

**Why this priority**: Important for understanding results and identifying issues, but experiments can run without this (results can be analyzed via raw data).

**Independent Test**: Can be tested by loading experiment results and creating interactive plots that:
- Show training curves with hover tooltips
- Allow pan/zoom exploration
- Support exporting to HTML

**Acceptance Scenarios**:

1. **Given** experiment results from all three methods, **When** I create training curve visualization, **Then** I see an interactive plot with hover tooltips showing exact epoch/loss/error values

2. **Given** interactive loss distribution plot, **When** I hover over the histogram, **Then** I can see the frequency of different loss values and identify negative uPU losses

3. **Given** comparison dashboard, **When** I toggle legend entries, **Then** specific methods show/hide in the visualization

4. **Given** any interactive plot, **When** I export to HTML, **Then** I get a standalone file that works in any browser with full interactivity preserved

---

### User Story 5 - Implement Custom PU Loss (Priority: P3)

As a researcher, I want to implement my custom PU loss function (L(1,p)=-log(p)+p, L(0,p)=p) and compare it against nnPU, so that I can evaluate whether my method improves upon the state-of-the-art.

**Why this priority**: This is the ultimate goal but depends on successful baseline reproduction. Can only be done after P1 stories are complete.

**Independent Test**: Can be tested by implementing the custom loss, running experiments with it, and comparing results against nnPU baseline.

**Acceptance Scenarios**:

1. **Given** the custom loss implementation, **When** I run unit tests, **Then** the loss computes the correct mathematical formulation: π·E_P[-log(p)+p] + E_U[p]

2. **Given** a configured custom loss experiment, **When** I run training for 100 epochs, **Then** the model trains successfully without errors

3. **Given** results from custom loss and nnPU, **When** I compare test errors, **Then** I can see the performance difference and determine if custom loss improves upon nnPU

---

### Edge Cases

- What happens when MPS is not available? → System falls back to CUDA if available, then CPU
- What happens when batch contains only positive or only unlabeled samples? → Loss computation handles this gracefully (no division by zero)
- What happens when loss becomes very negative in uPU? → Training continues but logs warning, demonstrates overfitting phenomenon
- What happens when user specifies invalid class prior π? → Validation raises error with helpful message
- What happens when MNIST download fails? → Clear error message with retry instructions
- What happens when running on different PyTorch versions? → pyproject.toml pins minimum version (2.0.0+)

## Requirements *(mandatory)*

### Functional Requirements

#### Data Processing
- **FR-001**: System MUST load MNIST dataset and split into train/test sets
- **FR-002**: System MUST filter even digits (0,2,4,6,8) as positive class and odd digits as negative class
- **FR-003**: System MUST sample exactly 100 positive examples for labeled set P
- **FR-004**: System MUST create unlabeled set U from remaining ~59,900 samples (without labels)
- **FR-005**: System MUST calculate class prior π as ratio of positive class in unlabeled data
- **FR-006**: System MUST normalize images with MNIST standard (mean=0.1307, std=0.3081)
- **FR-007**: System MUST flatten 28×28 images to 784-dimensional vectors
- **FR-008**: System MUST use deterministic sampling with fixed seed for reproducibility

#### Model Architecture
- **FR-009**: System MUST implement 6-layer MLP matching nnPU paper specification
- **FR-010**: MLP MUST have architecture: [784→300→300→300→300→300→1] with ReLU activations
- **FR-011**: Output layer MUST use sigmoid activation to produce probabilities p(x) ∈ [0,1]
- **FR-012**: Model MUST use Xavier or Kaiming weight initialization

#### Loss Functions
- **FR-013**: System MUST implement PN (Positive-Negative) loss: L_PN = E_P[BCE(f(x),1)] + E_N[BCE(f(x),0)]
- **FR-014**: System MUST implement uPU loss: L_uPU = π·E_P[l(f(x))] + E_U[l(-f(x))] - π·E_P[l(-f(x))]
- **FR-015**: System MUST implement nnPU loss: L_nnPU = π·E_P[l(f(x))] + max(0, E_U[l(-f(x))] - π·E_P[l(-f(x))])
- **FR-016**: uPU loss MUST be allowed to produce negative values (no constraint)
- **FR-017**: nnPU loss MUST always be non-negative (verified by assertion in code)
- **FR-018**: All loss functions MUST accept class prior π as parameter

#### Training & Evaluation
- **FR-019**: System MUST implement training loop that works with any (model, loss, data) combination
- **FR-020**: Trainer MUST evaluate on test set using zero-one loss (classification error rate)
- **FR-021**: Trainer MUST save checkpoints periodically during training
- **FR-022**: Trainer MUST log metrics: loss, accuracy, error rate per epoch
- **FR-023**: Trainer MUST support early stopping based on validation metric
- **FR-024**: System MUST compute zero-one loss as: (# misclassified) / (# total samples)

#### Reproducibility
- **FR-025**: System MUST set seeds for: Python random, NumPy, PyTorch CPU/MPS/CUDA
- **FR-026**: System MUST enable deterministic operations where available
- **FR-027**: System MUST log: seed, git commit hash, timestamp, hyperparameters, results for every experiment
- **FR-028**: Same seed MUST produce identical results across runs (within MPS numerical precision)

#### Hardware Acceleration
- **FR-029**: System MUST detect and use MPS (Metal Performance Shaders) on Apple Silicon
- **FR-030**: System MUST fall back to CUDA if MPS unavailable, then CPU
- **FR-031**: Device selection MUST be automatic with manual override option
- **FR-032**: System MUST clear MPS cache between experiments to prevent memory issues

#### Visualization
- **FR-033**: System MUST provide interactive plots using hvPlot + HoloViews + Bokeh
- **FR-034**: Training curves MUST show loss and test error vs epochs with hover tooltips
- **FR-035**: Loss distribution plots MUST highlight negative values for uPU method
- **FR-036**: All plots MUST support pan, zoom, and toggle legend features
- **FR-037**: All plots MUST be exportable as standalone HTML files

#### Testing
- **FR-038**: System MUST have >80% code coverage with pytest
- **FR-039**: Unit tests MUST verify: data loading, loss computations, model architecture, metrics
- **FR-040**: Integration tests MUST verify: full training loop, checkpointing, reproducibility
- **FR-041**: Scientific tests MUST verify: uPU negative risk, nnPU non-negative risk

#### Configuration
- **FR-042**: System MUST support YAML configuration files for experiments
- **FR-043**: Configs MUST specify: experiment name, seed, data params, model architecture, loss type, optimizer, training params
- **FR-044**: System MUST validate configuration before starting experiment

### Key Entities

- **MNISTPUDataset**: Represents the MNIST dataset with P/U split
  - Attributes: positive_data (100 samples), unlabeled_data (~59,900 samples), test_data, class_prior (π)
  - Methods: __getitem__ returns (image, label, is_labeled) tuples

- **MLP6Layer**: 6-layer multi-layer perceptron
  - Attributes: layers (6 Linear layers), activations (ReLU, Sigmoid)
  - Methods: forward(x) returns probability predictions p(x) ∈ [0,1]

- **PULoss (Base Class)**: Abstract base for all PU loss functions
  - Attributes: prior (π), loss_fn (sigmoid/logistic)
  - Methods: forward(outputs, targets, is_labeled) computes loss

- **uPULoss**: Unbiased PU loss (can be negative)
  - Inherits from PULoss
  - Implements: π·E_P[l(f)] + E_U[l(-f)] - π·E_P[l(-f)]

- **nnPULoss**: Non-negative PU loss
  - Inherits from PULoss
  - Implements: π·E_P[l(f)] + max(0, E_U[l(-f)] - π·E_P[l(-f)])

- **PUTrainer**: Generic trainer for PU learning
  - Attributes: model, loss_fn, optimizer, device, seed
  - Methods: train_epoch(), evaluate(), save_checkpoint(), load_checkpoint()

- **ExperimentConfig**: Configuration for a single experiment
  - Attributes: experiment (name, seed), data (n_positive, batch_size), model (architecture, hidden_dims), loss (type, prior), optimizer (type, lr), training (epochs, device)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All three methods (PN, uPU, nnPU) train successfully on MNIST for 100 epochs without errors

- **SC-002**: uPU loss becomes negative at some point during training in at least 80% of runs, demonstrating the negative risk phenomenon described in the paper

- **SC-003**: nnPU loss remains non-negative throughout all epochs in 100% of runs (verified by assertion)

- **SC-004**: nnPU achieves final test error within ±2% of uPU's final test error (may be better or comparable)

- **SC-005**: Running the same experiment with seed=42 produces identical loss values (±1e-5) at every epoch across multiple runs (accounting for MPS numerical precision)

- **SC-006**: MPS GPU acceleration detected and used on M4 MacBook Pro (verified by GPU activity monitor)

- **SC-007**: MPS training is at least 5x faster than CPU training for the same experiment

- **SC-008**: Code coverage is >80% as measured by pytest-cov

- **SC-009**: All pytest tests pass (100% pass rate)

- **SC-010**: Interactive visualizations display correctly in Jupyter notebooks with hover tooltips, pan/zoom, and export to HTML functionality working

- **SC-011**: Results align with nnPU paper findings: nnPU prevents overfitting that uPU exhibits, achieving comparable or better final test error

- **SC-012**: Custom PU loss (Phase 9) trains successfully and produces valid results for comparison against nnPU baseline
