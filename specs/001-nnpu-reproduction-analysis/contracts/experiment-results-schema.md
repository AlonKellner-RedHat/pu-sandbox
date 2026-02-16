# Experiment Results Schema

**Feature**: [spec.md](../spec.md) | **Data Model**: [data-model.md](../data-model.md)
**Created**: 2026-02-16
**Purpose**: Define the JSON schema for experiment result files

## Overview

This contract defines the standard JSON structure for storing experimental results from baseline and post-fix validation runs. All experiment result files must conform to this schema to ensure consistent analysis and comparison.

---

## File Naming Convention

**Format**: `{method}_seed{seed}.json`

**Examples**:
- `PN_seed42.json`
- `uPU_seed43.json`
- `nnPU_seed44.json`

**Locations**:
- Baseline: `experiments/results/baseline/{filename}`
- Post-fix: `experiments/results/fixed/{filename}`

---

## JSON Schema

### Root Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `experiment_id` | string | Yes | Unique identifier (format: `{method}_seed{seed}`) |
| `method` | string | Yes | One of: `"PN"`, `"uPU"`, `"nnPU"` |
| `seed` | integer | Yes | Random seed used (42, 43, 44, 45, or 46) |
| `config` | object | Yes | Experiment configuration (see below) |
| `results` | object | Yes | Per-epoch metrics (see below) |
| `final_metrics` | object | Yes | Summary of final epoch (see below) |
| `metadata` | object | Yes | Reproducibility metadata (see below) |

### `config` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epochs` | integer | Yes | Total training epochs (expected: 100) |
| `batch_size` | integer | Yes | Mini-batch size |
| `learning_rate` | number | Yes | Initial learning rate |
| `optimizer` | string | Yes | Optimizer name (e.g., `"SGD"`, `"Adam"`) |
| `optimizer_params` | object | No | Additional optimizer parameters (e.g., momentum, weight_decay) |
| `n_positive` | integer | Yes | Number of labeled positive samples |
| `prior` | number | Yes | Class prior π (proportion of positive class, range: [0.0, 1.0]) |
| `beta` | number | No | β parameter for nnPU (null for PN/uPU methods) |
| `beta_schedule` | string | No | β update schedule if applicable (e.g., `"fixed"`, `"annealing"`) |
| `device` | string | Yes | Compute device (e.g., `"mps"`, `"cuda:0"`, `"cpu"`) |

### `results` Object

Per-epoch arrays containing metrics for each epoch:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epochs` | array[integer] | Yes | Epoch numbers `[0, 1, 2, ..., 99]` |
| `train_loss` | array[number] | Yes | Training loss per epoch |
| `test_loss` | array[number] | Yes | Test loss per epoch |
| `test_accuracy` | array[number] | Yes | Test accuracy per epoch (range: [0.0, 1.0]) |
| `test_zero_one_loss` | array[number] | Yes | Test error rate per epoch (range: [0.0, 1.0]) |

**Array Length Constraint**: All arrays must have the same length, equal to `config.epochs`.

**Value Constraint**: For all `i`: `test_zero_one_loss[i] ≈ 1.0 - test_accuracy[i]` (within floating point precision).

### `final_metrics` Object

Summary statistics from the final epoch:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `final_epoch` | integer | Yes | Last epoch number (typically `config.epochs - 1`) |
| `final_train_loss` | number | Yes | Training loss at final epoch |
| `final_test_loss` | number | Yes | Test loss at final epoch |
| `final_test_accuracy` | number | Yes | Test accuracy at final epoch |
| `final_test_error` | number | Yes | Test error rate at final epoch (zero-one loss) |
| `best_test_error` | number | No | Minimum test error achieved across all epochs |
| `best_test_error_epoch` | integer | No | Epoch at which best test error was achieved |
| `convergence_epoch` | integer | No | Epoch where performance stabilized (optional, null if did not converge) |
| `loss_stayed_nonnegative` | boolean | No | For nnPU only: whether loss remained ≥ 0 throughout training |

**Consistency Constraint**: `final_*` values must match the last element of corresponding `results.*` arrays.

### `metadata` Object

Reproducibility and provenance information:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | string (ISO8601) | Yes | When experiment was run (e.g., `"2026-02-16T14:23:45Z"`) |
| `commit_hash` | string | Yes | Git commit SHA of code used (40-char hex string) |
| `branch` | string | Yes | Git branch name (e.g., `"001-nnpu-reproduction-analysis"`) |
| `python_version` | string | Yes | Python version (e.g., `"3.12.1"`) |
| `pytorch_version` | string | Yes | PyTorch version (e.g., `"2.2.0"`) |
| `device_name` | string | Yes | Device name (e.g., `"Apple M4 Pro"`) |
| `duration_seconds` | number | Yes | Total experiment runtime in seconds |
| `dataset` | string | Yes | Dataset name (e.g., `"MNIST-even-odd"`) |
| `dataset_config` | object | No | Dataset-specific configuration (e.g., train/test split sizes) |

---

## Complete Example

```json
{
  "experiment_id": "nnPU_seed42",
  "method": "nnPU",
  "seed": 42,
  "config": {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "optimizer_params": {
      "betas": [0.9, 0.999],
      "weight_decay": 0.0001
    },
    "n_positive": 100,
    "prior": 0.5,
    "beta": 0.0,
    "beta_schedule": "fixed",
    "device": "mps"
  },
  "results": {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "...", 99],
    "train_loss": [2.145, 1.987, 1.823, 1.654, 1.512, 1.387, 1.278, 1.189, 1.098, 0.987, "...", 0.234],
    "test_loss": [2.089, 1.934, 1.756, 1.612, 1.489, 1.356, 1.245, 1.134, 1.045, 0.934, "...", 0.198],
    "test_accuracy": [0.523, 0.612, 0.689, 0.734, 0.778, 0.812, 0.845, 0.876, 0.901, 0.918, "...", 0.956],
    "test_zero_one_loss": [0.477, 0.388, 0.311, 0.266, 0.222, 0.188, 0.155, 0.124, 0.099, 0.082, "...", 0.044]
  },
  "final_metrics": {
    "final_epoch": 99,
    "final_train_loss": 0.234,
    "final_test_loss": 0.198,
    "final_test_accuracy": 0.956,
    "final_test_error": 0.044,
    "best_test_error": 0.042,
    "best_test_error_epoch": 97,
    "convergence_epoch": 85,
    "loss_stayed_nonnegative": true
  },
  "metadata": {
    "timestamp": "2026-02-16T14:23:45Z",
    "commit_hash": "2dfacf4a8b9c3d1e5f6a7b8c9d0e1f2a3b4c5d6e",  // pragma: allowlist secret
    "branch": "001-nnpu-reproduction-analysis",
    "python_version": "3.12.1",
    "pytorch_version": "2.2.0",
    "device_name": "Apple M4 Pro",
    "duration_seconds": 342.7,
    "dataset": "MNIST-even-odd",
    "dataset_config": {
      "positive_class": "even",
      "negative_class": "odd",
      "train_size": 60000,
      "test_size": 10000,
      "n_positive_labeled": 100,
      "n_unlabeled": 59900
    }
  }
}
```

---

## Validation Rules

### Structural Validation

- All required fields must be present
- Field types must match specification
- Nested objects must conform to sub-schemas

### Value Validation

- `method` must be one of: `"PN"`, `"uPU"`, `"nnPU"`
- `seed` must be one of: 42, 43, 44, 45, 46
- `config.prior` must be in range [0.0, 1.0]
- `config.beta` must be null for PN/uPU methods
- All `results.*` arrays must have length equal to `config.epochs`
- `test_accuracy[i] + test_zero_one_loss[i]` ≈ 1.0 for all i (tolerance: 1e-6)
- `final_metrics.final_*` must match last element of `results.*` arrays
- `metadata.commit_hash` must be valid git SHA (40 hex chars)
- `metadata.timestamp` must be valid ISO8601 format

### Consistency Validation

- `experiment_id` must equal `"{method}_seed{seed}"`
- If baseline and post-fix results exist for same method/seed:
  - `seed` must match
  - `config` should match except where fixes changed parameters
  - Array lengths must match

### Reproducibility Validation

- `metadata.commit_hash` must exist in git history
- `metadata.branch` should exist (may be merged/deleted for post-fix runs)
- All experiments with same `seed` should produce identical results when run with same `commit_hash`

---

## Usage in Analysis Scripts

### Loading Results

```python
import json
from pathlib import Path

def load_experiment_results(results_dir: Path) -> dict:
    """Load all experiment results from a directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results[data["experiment_id"]] = data
    return results

# Load baseline results
baseline_results = load_experiment_results(Path("experiments/results/baseline"))

# Access specific experiment
nnpu_seed42 = baseline_results["nnPU_seed42"]
print(f"Final test error: {nnpu_seed42['final_metrics']['final_test_error']}")
```

### Comparing Baseline vs Post-Fix

```python
import numpy as np

def compare_results(baseline_file: Path, postfix_file: Path) -> dict:
    """Compare baseline and post-fix results."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(postfix_file) as f:
        postfix = json.load(f)

    return {
        "experiment_id": baseline["experiment_id"],
        "baseline_error": baseline["final_metrics"]["final_test_error"],
        "postfix_error": postfix["final_metrics"]["final_test_error"],
        "delta": postfix["final_metrics"]["final_test_error"] - baseline["final_metrics"]["final_test_error"],
        "percent_change": ((postfix["final_metrics"]["final_test_error"] - baseline["final_metrics"]["final_test_error"])
                          / baseline["final_metrics"]["final_test_error"]) * 100
    }
```

### Aggregating Across Seeds

```python
def aggregate_across_seeds(results_dir: Path, method: str) -> dict:
    """Compute mean and std across all seeds for a method."""
    errors = []
    for seed in [42, 43, 44, 45, 46]:
        file_path = results_dir / f"{method}_seed{seed}.json"
        with open(file_path) as f:
            data = json.load(f)
            errors.append(data["final_metrics"]["final_test_error"])

    return {
        "method": method,
        "mean": np.mean(errors),
        "std": np.std(errors),
        "min": np.min(errors),
        "max": np.max(errors),
        "values": errors
    }
```

---

## Integration with Existing Code

### Experiment Script Modifications

The existing `experiments/scripts/run_all.py` should be modified to:

1. Save results in this JSON format
2. Include all required metadata fields
3. Validate output against this schema before writing
4. Save to appropriate directory (`baseline/` or `fixed/`)

### Visualization Integration

The existing `notebooks/02_results_analysis.ipynb` already loads experiment results. Ensure it:

1. Reads JSON files conforming to this schema
2. Handles both `baseline/` and `fixed/` directories
3. Validates required fields are present
4. Uses standardized field names for analysis

---

## Schema Versioning

**Current Version**: 1.0
**Last Updated**: 2026-02-16

If schema changes are needed:

1. Increment version number
2. Add `"schema_version"` field to root object
3. Document migration path from previous versions
4. Maintain backward compatibility where possible

---

## Validation Checklist

Before committing experiment results, verify:

- [ ] Filename follows convention: `{method}_seed{seed}.json`
- [ ] File saved to correct directory (`baseline/` or `fixed/`)
- [ ] All required fields present
- [ ] All arrays have same length (equal to `config.epochs`)
- [ ] `test_accuracy + test_zero_one_loss ≈ 1.0` for all epochs
- [ ] `final_metrics` match last elements of `results` arrays
- [ ] `commit_hash` is valid and matches current git HEAD
- [ ] `timestamp` is in ISO8601 format
- [ ] JSON is valid and parseable

---

## Related Contracts

- [discrepancy-report-schema.md](discrepancy-report-schema.md): Report structure
- [analysis-scripts.md](analysis-scripts.md): Script interfaces for running experiments

See [data-model.md](../data-model.md#entity-baseline-results) for the full data model specification.
