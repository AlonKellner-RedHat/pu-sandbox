# Data Models: nnPU Paper Reproduction Analysis

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)
**Created**: 2026-02-16
**Status**: Complete

## Overview

This document defines the data entities and schemas used throughout the nnPU reproduction analysis and validation workflow. All entities are designed to support traceability, reproducibility, and quantified comparison between baseline and post-fix results.

---

## Entity: Discrepancy

Represents a single identified difference between the current implementation and the paper/reference implementations.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (format: `DISC-001`, `DISC-002`, etc.) |
| `category` | enum | Yes | One of: `Mathematical`, `Procedural`, `Hyperparameter` |
| `severity` | enum | Yes | One of: `Critical`, `High`, `Medium`, `Low` |
| `component` | string | Yes | Affected file/class/function (format: `src/pu_learning/losses/nnpu.py:nnPULoss.forward:42`) |
| `title` | string | Yes | Brief descriptive title (max 80 chars) |
| `description` | string | Yes | Summary of what discrepancy was found |
| `expected_behavior` | object | Yes | What paper/reference specifies (see sub-schema below) |
| `actual_behavior` | object | Yes | What current implementation does (see sub-schema below) |
| `impact` | object | Yes | How this affects reproduction (see sub-schema below) |
| `proposed_fix` | string | Yes | Specific code changes needed to resolve |
| `fix_status` | enum | Yes | One of: `NotStarted`, `InProgress`, `Implemented`, `Tested`, `Validated`, `WontFix` |
| `related_discrepancies` | array[string] | No | List of related DISC IDs (e.g., `["DISC-005", "DISC-012"]`) |
| `created_date` | ISO8601 | Yes | When discrepancy was identified |
| `fixed_date` | ISO8601 | No | When fix was validated (null if not yet fixed) |

### Sub-Schema: expected_behavior

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_type` | enum | Yes | One of: `Paper`, `ReferenceImpl`, `Both` |
| `paper_citation` | string | No | Paper section/equation (e.g., `"Section 3.2, Equation 6"`) |
| `kiryor_reference` | string | No | File and line in kiryor repo (e.g., `"pu_loss.py:42-45"`) |
| `cimeister_reference` | string | No | File and line in cimeister repo (e.g., `"losses.py:78"`) |
| `specification` | string | Yes | Exact quote, mathematical formula, or code snippet from source |

### Sub-Schema: actual_behavior

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes | Path to current implementation file |
| `line_numbers` | string | Yes | Line range (e.g., `"42-45"` or `"78"`) |
| `implementation` | string | Yes | Code snippet or description of current behavior |

### Sub-Schema: impact

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `expected_impact` | string | Yes | Theoretical/predicted effect on reproduction |
| `measured_impact` | object | No | Actual performance difference from baseline experiments (null before baseline run) |
| `measured_impact.metric` | string | No | Metric affected (e.g., `"test_zero_one_loss"`) |
| `measured_impact.baseline_value` | float | No | Baseline performance |
| `measured_impact.change` | float | No | Performance delta (negative = degradation, positive = improvement) |

### Example

```json
{
  "id": "DISC-003",
  "category": "Mathematical",
  "severity": "Critical",
  "component": "src/pu_learning/losses/nnpu.py:nnPULoss.forward:78",
  "title": "Missing gradient detachment in non-negative risk correction",
  "description": "The non-negative risk correction term does not detach gradients before comparison, potentially causing incorrect gradient flow",
  "expected_behavior": {
    "source_type": "Both",
    "paper_citation": "Section 3.2, Equation 6 - gradient should not flow through max(0, ...) comparison",
    "kiryor_reference": "pu_loss.py:125 - uses .detach() before comparison",
    "cimeister_reference": "losses.py:89 - implements detach() in nnPU risk",
    "specification": "The paper specifies that β in Eq. 6 is fixed per mini-batch and should not receive gradients. Both reference implementations use .detach() before the comparison."
  },
  "actual_behavior": {
    "file_path": "src/pu_learning/losses/nnpu.py",
    "line_numbers": "78-82",
    "implementation": "Directly compares risk without detach(): if risk < 0: risk = -beta * risk"
  },
  "impact": {
    "expected_impact": "Incorrect gradient flow may prevent nnPU from maintaining non-negative loss property, causing training instability",
    "measured_impact": {
      "metric": "test_zero_one_loss",
      "baseline_value": 0.089,
      "change": -0.023
    }
  },
  "proposed_fix": "Add .detach() to the negative risk component before comparison:\n\nif risk.detach() < 0:\n    risk = -self.beta * risk.detach()",
  "fix_status": "NotStarted",
  "related_discrepancies": ["DISC-004"],
  "created_date": "2026-02-16T10:30:00Z",
  "fixed_date": null
}
```

### Validation Rules

- `id` must be unique across all discrepancies
- `severity` assignment must follow rubric from [research.md](research.md#research-question-3-discrepancy-severity-criteria)
- `component` format: `<file_path>:<class/function_name>:<line_number>`
- `fix_status` lifecycle: `NotStarted` → `InProgress` → `Implemented` → `Tested` → `Validated` (or `WontFix`)
- `measured_impact` must be null until baseline experiments are run
- At least one of `paper_citation`, `kiryor_reference`, or `cimeister_reference` must be provided in `expected_behavior`

### Category Definitions

**Mathematical**: Discrepancies in loss function formulations, risk estimators, mathematical operations
- Examples: Wrong sign in equation, missing term, incorrect coefficient

**Procedural**: Discrepancies in training algorithm, optimization procedure, evaluation workflow
- Examples: Wrong optimizer, incorrect learning rate schedule, batch construction errors

**Hyperparameter**: Discrepancies in configuration values used in experiments
- Examples: Different number of epochs, batch size mismatch, wrong β value

---

## Entity: Baseline Results

Represents experimental outcomes from running the current implementation (before fixes) to establish baseline metrics for comparison.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `experiment_id` | string | Yes | Unique identifier (format: `{method}_seed{seed}`, e.g., `"nnPU_seed42"`) |
| `method` | enum | Yes | One of: `PN`, `uPU`, `nnPU` |
| `seed` | integer | Yes | Random seed used (one of: 42, 43, 44, 45, 46) |
| `config` | object | Yes | Full experiment configuration (see sub-schema below) |
| `results` | object | Yes | Per-epoch metrics (see sub-schema below) |
| `final_metrics` | object | Yes | Summary of final epoch performance (see sub-schema below) |
| `metadata` | object | Yes | Experiment metadata for reproducibility (see sub-schema below) |

### Sub-Schema: config

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epochs` | integer | Yes | Total training epochs (expected: 100) |
| `batch_size` | integer | Yes | Mini-batch size |
| `learning_rate` | float | Yes | Initial learning rate |
| `optimizer` | string | Yes | Optimizer name (e.g., `"SGD"`, `"Adam"`) |
| `n_positive` | integer | Yes | Number of labeled positive samples |
| `prior` | float | Yes | Class prior π (proportion of positive class) |
| `beta` | float | No | β parameter for nnPU (null for PN/uPU) |

### Sub-Schema: results (per-epoch arrays)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epochs` | array[int] | Yes | Epoch numbers `[0, 1, 2, ..., 99]` |
| `train_loss` | array[float] | Yes | Training loss per epoch |
| `test_loss` | array[float] | Yes | Test loss per epoch |
| `test_accuracy` | array[float] | Yes | Test accuracy per epoch (0.0-1.0) |
| `test_zero_one_loss` | array[float] | Yes | Test error rate per epoch (0.0-1.0) - **primary metric** |

### Sub-Schema: final_metrics (summary statistics from final epoch)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `final_epoch` | integer | Yes | Last epoch number (typically 99 for 100 epochs) |
| `final_train_loss` | float | Yes | Training loss at final epoch |
| `final_test_loss` | float | Yes | Test loss at final epoch |
| `final_test_accuracy` | float | Yes | Test accuracy at final epoch |
| `final_test_error` | float | Yes | Test error rate at final epoch (zero-one loss) |
| `convergence_epoch` | integer | No | Epoch where performance stabilized (optional) |

### Sub-Schema: metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | ISO8601 | Yes | When experiment was run |
| `commit_hash` | string | Yes | Git commit SHA of code used |
| `branch` | string | Yes | Git branch name (e.g., `"001-nnpu-reproduction-analysis"`) |
| `device` | string | Yes | Compute device (e.g., `"mps"`, `"cuda"`, `"cpu"`) |
| `duration_seconds` | float | Yes | Total experiment runtime |

### Example

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
    "n_positive": 100,
    "prior": 0.5,
    "beta": 0.0
  },
  "results": {
    "epochs": [0, 1, 2, "...", 99],
    "train_loss": [2.145, 1.987, 1.823, "...", 0.234],
    "test_loss": [2.089, 1.934, 1.756, "...", 0.198],
    "test_accuracy": [0.523, 0.612, 0.689, "...", 0.956],
    "test_zero_one_loss": [0.477, 0.388, 0.311, "...", 0.044]
  },
  "final_metrics": {
    "final_epoch": 99,
    "final_train_loss": 0.234,
    "final_test_loss": 0.198,
    "final_test_accuracy": 0.956,
    "final_test_error": 0.044,
    "convergence_epoch": 85
  },
  "metadata": {
    "timestamp": "2026-02-16T14:23:45Z",
    "commit_hash": "2dfacf4a8b9c3d1e5f6a7b8c9d0e1f2a3b4c5d6e",  // pragma: allowlist secret
    "branch": "001-nnpu-reproduction-analysis",
    "device": "mps",
    "duration_seconds": 342.7
  }
}
```

### Validation Rules

- `experiment_id` format: `{method}_seed{seed}` (e.g., `nnPU_seed42`)
- `method` must be one of the three implemented methods
- `seed` must be one of the specified seeds (42, 43, 44, 45, 46)
- All `results` arrays must have same length (equal to `config.epochs`)
- `final_metrics` values must match last element of corresponding `results` arrays
- `commit_hash` must be valid git SHA
- `test_zero_one_loss[i] = 1.0 - test_accuracy[i]` (within floating point precision)

### Storage Format

- Location: `experiments/results/baseline/{experiment_id}.json`
- Example files:
  - `experiments/results/baseline/PN_seed42.json`
  - `experiments/results/baseline/uPU_seed43.json`
  - `experiments/results/baseline/nnPU_seed44.json`
- Total files: 15 (3 methods × 5 seeds)

---

## Entity: Post-Fix Results

**Schema**: Identical to Baseline Results (see above)

**Purpose**: Capture experimental outcomes after applying fixes to compare against baseline

**Storage Format**:
- Location: `experiments/results/fixed/{experiment_id}.json`
- Same structure as baseline results
- Example files:
  - `experiments/results/fixed/PN_seed42.json`
  - `experiments/results/fixed/uPU_seed43.json`
  - `experiments/results/fixed/nnPU_seed44.json`

**Comparison Requirements**:
- Must use identical `seed` values as corresponding baseline experiments
- Must use identical `config` (except for fixes applied)
- `metadata.commit_hash` will differ (post-fix commits)

---

## Entity: Comparison

Represents before/after quantitative comparison for a specific discrepancy or overall reproduction quality.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `comparison_id` | string | Yes | Unique identifier (format: `COMP-{method}-{metric}` or `COMP-DISC-XXX`) |
| `comparison_type` | enum | Yes | One of: `OverallReproduction`, `DiscrepancyImpact` |
| `discrepancy_id` | string | No | Link to specific discrepancy (required if type=`DiscrepancyImpact`) |
| `method` | enum | Yes | One of: `PN`, `uPU`, `nnPU`, `All` |
| `metric` | string | Yes | Metric being compared (e.g., `"test_zero_one_loss"`, `"test_accuracy"`) |
| `baseline_stats` | object | Yes | Statistics from baseline experiments (see sub-schema below) |
| `postfix_stats` | object | Yes | Statistics from post-fix experiments (see sub-schema below) |
| `delta` | object | Yes | Quantified improvement (see sub-schema below) |
| `statistical_significance` | object | Yes | Significance testing results (see sub-schema below) |
| `paper_benchmark` | object | No | Comparison to paper-reported values (see sub-schema below) |

### Sub-Schema: baseline_stats / postfix_stats

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mean` | float | Yes | Mean across all seeds |
| `std` | float | Yes | Standard deviation across all seeds |
| `min` | float | Yes | Minimum value across all seeds |
| `max` | float | Yes | Maximum value across all seeds |
| `n_seeds` | integer | Yes | Number of seeds (expected: 5) |
| `values` | array[float] | Yes | Raw values from each seed |

### Sub-Schema: delta

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `absolute` | float | Yes | Post-fix mean - Baseline mean |
| `percent` | float | Yes | (Absolute / Baseline mean) × 100 |
| `interpretation` | enum | Yes | One of: `Improved`, `Degraded`, `NoChange` |

**Interpretation Rules** (for test error metric, lower is better):
- `Improved`: delta < -0.001 (error decreased by >0.1%)
- `Degraded`: delta > 0.001 (error increased by >0.1%)
- `NoChange`: abs(delta) ≤ 0.001 (within noise threshold)

### Sub-Schema: statistical_significance

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `test_type` | string | Yes | Statistical test used (e.g., `"paired t-test"`, `"Wilcoxon signed-rank"`) |
| `p_value` | float | Yes | P-value from statistical test |
| `is_significant` | boolean | Yes | `true` if p < 0.05, `false` otherwise |
| `effect_size` | float | No | Cohen's d or other effect size measure (optional) |

### Sub-Schema: paper_benchmark

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `paper_value` | float | No | Value reported in paper (if available) |
| `paper_citation` | string | No | Where in paper value is reported (e.g., `"Table 1"`) |
| `within_tolerance` | boolean | No | `true` if within ±2% of paper value |
| `delta_from_paper` | float | No | Post-fix mean - Paper value |

### Example

```json
{
  "comparison_id": "COMP-nnPU-test_error",
  "comparison_type": "OverallReproduction",
  "discrepancy_id": null,
  "method": "nnPU",
  "metric": "test_zero_one_loss",
  "baseline_stats": {
    "mean": 0.089,
    "std": 0.0034,
    "min": 0.084,
    "max": 0.093,
    "n_seeds": 5,
    "values": [0.089, 0.087, 0.093, 0.084, 0.091]
  },
  "postfix_stats": {
    "mean": 0.053,
    "std": 0.0021,
    "min": 0.050,
    "max": 0.056,
    "n_seeds": 5,
    "values": [0.053, 0.051, 0.056, 0.050, 0.055]
  },
  "delta": {
    "absolute": -0.036,
    "percent": -40.4,
    "interpretation": "Improved"
  },
  "statistical_significance": {
    "test_type": "paired t-test",
    "p_value": 0.0003,
    "is_significant": true,
    "effect_size": 2.87
  },
  "paper_benchmark": {
    "paper_value": 0.051,
    "paper_citation": "Table 1 - MNIST even/odd, nnPU with 100 labeled positive",
    "within_tolerance": true,
    "delta_from_paper": 0.002
  }
}
```

### Validation Rules

- If `comparison_type` is `DiscrepancyImpact`, `discrepancy_id` must be provided
- `baseline_stats.n_seeds` and `postfix_stats.n_seeds` must equal 5
- `delta.absolute` = `postfix_stats.mean` - `baseline_stats.mean`
- `delta.percent` = (`delta.absolute` / `baseline_stats.mean`) × 100
- `statistical_significance.is_significant` = (`p_value` < 0.05)
- If `paper_benchmark` provided: `within_tolerance` = (abs(`delta_from_paper`) ≤ 0.02)

---

## Data Flow Diagram

```
┌─────────────────┐
│ Paper TeX       │────┐
│ Source          │    │
└─────────────────┘    │
                       │
┌─────────────────┐    │    ┌──────────────────┐
│ Reference       │────┼───>│ Discrepancy      │
│ Implementations │    │    │ Analysis         │
└─────────────────┘    │    └──────────────────┘
                       │             │
┌─────────────────┐    │             │
│ Current         │────┘             │
│ Implementation  │                  │
└─────────────────┘                  │
                                     ▼
                            ┌─────────────────┐
                            │ Discrepancy     │
                            │ Report (MD)     │
                            │ [DISC-001,      │
                            │  DISC-002, ...] │
                            └─────────────────┘
                                     │
                                     │ Identifies fixes
                                     ▼
┌─────────────────┐         ┌─────────────────┐
│ Baseline        │         │ TDD Cycle       │
│ Experiments     │         │ (Write tests,   │
│ (100 epochs,    │         │  implement fix, │
│  5 seeds)       │         │  refactor)      │
└─────────────────┘         └─────────────────┘
         │                           │
         │                           ▼
         │                  ┌─────────────────┐
         │                  │ Post-Fix        │
         │                  │ Experiments     │
         │                  │ (100 epochs,    │
         │                  │  5 seeds)       │
         │                  └─────────────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
            ┌─────────────────┐
            │ Comparison      │
            │ Analysis        │
            │ (Baseline vs    │
            │  Post-Fix vs    │
            │  Paper)         │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Before/After    │
            │ Report          │
            │ (Quantified     │
            │  improvement)   │
            └─────────────────┘
```

---

## File Organization

```
pu-sandbox/
├── analysis/
│   ├── nnpu-discrepancy-report.md        # Discrepancy entities (markdown)
│   ├── paper-source/                      # Downloaded TeX source
│   │   └── 1703.00593/
│   └── reference-implementations/         # Cloned reference repos
│       ├── kiryor-nnPUlearning/
│       └── cimeister-pu-learning/
│
├── experiments/
│   └── results/
│       ├── baseline/                      # Baseline Results entities (JSON)
│       │   ├── PN_seed42.json
│       │   ├── PN_seed43.json
│       │   ├── ... (15 files total)
│       │   └── nnPU_seed46.json
│       │
│       └── fixed/                         # Post-Fix Results entities (JSON)
│           ├── PN_seed42.json
│           ├── ... (15 files total)
│           └── nnPU_seed46.json
│
└── scripts/
    ├── compare-results.py                 # Generates Comparison entities
    └── analysis/
        └── comparison-report.json         # Comparison entities (JSON)
```

---

## Next Steps

This data model is complete. Proceed to create:

1. **contracts/**: Standard schemas and interfaces
2. **quickstart.md**: Setup and usage guide

All entities are designed to support the full analysis workflow from discrepancy identification through fix validation and quantified improvement measurement.
