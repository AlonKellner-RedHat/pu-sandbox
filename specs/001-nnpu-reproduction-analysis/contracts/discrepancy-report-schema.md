# Discrepancy Report Schema

**Feature**: [spec.md](../spec.md) | **Data Model**: [data-model.md](../data-model.md)
**Created**: 2026-02-16
**Purpose**: Define the standard structure for `analysis/nnpu-discrepancy-report.md`

## Overview

This contract defines the required markdown structure for the discrepancy report. The report must be human-readable (for review and decision-making) while maintaining machine-parseable structure (for traceability and automation).

---

## File Location

**Path**: `analysis/nnpu-discrepancy-report.md`

---

## Document Structure

### Required Sections (in order)

1. **Title and Metadata**
2. **Executive Summary**
3. **Severity Matrix**
4. **Mathematical Discrepancies**
5. **Procedural Discrepancies**
6. **Hyperparameter Discrepancies**
7. **Fix Plan and Progress**
8. **Validation Results**

---

## Section Specifications

### 1. Title and Metadata

```markdown
# nnPU Paper Reproduction Discrepancy Report

**Feature**: [specs/001-nnpu-reproduction-analysis/spec.md](../specs/001-nnpu-reproduction-analysis/spec.md)
**Analysis Date**: YYYY-MM-DD
**Analyst**: [Name or "Claude Code + User"]
**Status**: Draft | In Progress | Complete

**Sources Analyzed**:
- Paper: arxiv.org/src/1703.00593 (TeX source)
- Reference 1: github.com/kiryor/nnPUlearning (commit SHA)
- Reference 2: github.com/cimeister/pu-learning (commit SHA)
- Current Implementation: (commit SHA)
```

### 2. Executive Summary

```markdown
## Executive Summary

**Total Discrepancies Found**: [N]

**Breakdown by Severity**:
- Critical: [N] - [All | Some | None] fixed
- High: [N] - [All | Some | None] fixed
- Medium: [N] - [All | Some | None] fixed
- Low: [N] - [All | Some | None] fixed

**Breakdown by Category**:
- Mathematical: [N] discrepancies
- Procedural: [N] discrepancies
- Hyperparameter: [N] discrepancies

**Key Findings**:
- [2-3 bullet points highlighting most critical discrepancies]
- [Impact on reproduction accuracy]
- [Overall assessment of implementation quality]

**Reproduction Status**:
- Baseline Test Error: [value] ± [std] (current implementation)
- Paper Benchmark: [value] (from paper Table X)
- Delta: [value] ([percentage]% difference)
- **Within Target**: [Yes/No] (±2% tolerance)
```

### 3. Severity Matrix

```markdown
## Severity Matrix

Summary table of all discrepancies sorted by severity:

| ID | Category | Severity | Component | Status | Impact |
|----|----------|----------|-----------|--------|--------|
| DISC-001 | Mathematical | Critical | losses/nnpu.py | Fixed | High |
| DISC-002 | Procedural | High | training/trainer.py | In Progress | Medium |
| ... | ... | ... | ... | ... | ... |

**Legend**:
- **Status**: NotStarted | InProgress | Implemented | Tested | Validated | WontFix
- **Impact**: High (>5% degradation) | Medium (1-5%) | Low (<1%) | Unknown (no baseline data)
```

### 4. Mathematical Discrepancies

```markdown
## Mathematical Discrepancies

Discrepancies in loss function formulations and mathematical operations.

---

### DISC-[XXX]: [Brief Title]

**Category**: Mathematical
**Severity**: Critical | High | Medium | Low
**Component**: `src/pu_learning/path/to/file.py:ClassName.method_name:line_number`
**Status**: NotStarted | InProgress | Implemented | Tested | Validated | WontFix

#### Description

[1-2 sentence summary of what discrepancy was found]

#### Expected Behavior

**Source**: Paper Section X.Y, Equation Z | Kiryor: `file.py:line` | Cimeister: `file.py:line`

**Specification**:
[Exact quote, mathematical formula, or code snippet from source]

**Mathematical Form** (if applicable):
```
[LaTeX or plain text equation from paper]
```

**Reference Implementation** (if applicable):
```python
# kiryor/nnPUlearning - pu_loss.py:125-130
def compute_risk(...):
    # Reference code snippet
    pass
```

#### Actual Behavior

**Current Code**: `src/pu_learning/losses/nnpu.py:42-45`

**Implementation**:
```python
# Current implementation snippet
def forward(self, ...):
    # Code showing the discrepancy
    pass
```

#### Impact

**Expected Impact**: [Theoretical/predicted effect on training, convergence, or final performance]

**Measured Impact** (from baseline experiments):
- **Metric**: test_zero_one_loss
- **Baseline Value**: 0.089 ± 0.0034
- **Estimated Degradation**: ~2.3% (extrapolated from similar issues in references)

#### Proposed Fix

**Changes Required**:
```python
# Proposed fixed implementation
def forward(self, ...):
    # Fixed code
    pass
```

**Files to Modify**:
- `src/pu_learning/losses/nnpu.py` (lines 42-45)

**Tests to Add**:
- Unit test: `tests/unit/test_losses.py::test_nnpu_gradient_detachment`
- Regression test: Verify nnPU loss stays non-negative

#### Related Discrepancies

- [DISC-XXX](#disc-xxx): [Brief description of relationship]

---

[Repeat structure for each mathematical discrepancy]
```

### 5. Procedural Discrepancies

```markdown
## Procedural Discrepancies

Discrepancies in training algorithm, optimization procedure, and evaluation workflow.

---

### DISC-[XXX]: [Brief Title]

[Same structure as Mathematical Discrepancies section]

---

[Repeat for each procedural discrepancy]
```

### 6. Hyperparameter Discrepancies

```markdown
## Hyperparameter Discrepancies

Discrepancies in configuration values and experimental setup.

---

### DISC-[XXX]: [Brief Title]

[Same structure as Mathematical Discrepancies section, adjusted for hyperparameter context]

---

[Repeat for each hyperparameter discrepancy]
```

### 7. Fix Plan and Progress

```markdown
## Fix Plan and Progress

### Phase 1: Critical Fixes (Blocking)

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| DISC-001 | ... | ... | ... | Validated |
| DISC-003 | ... | ... | ... | In Progress |

**Blockers**: [Any dependencies or conflicts preventing progress]

### Phase 2: High Priority Fixes (Blocking)

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| DISC-002 | ... | ... | ... | Tested |

### Phase 3: Medium Priority Fixes (Time-Permitting)

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| DISC-010 | ... | ... | ... | NotStarted |

### Phase 4: Low Priority (Document Only)

| DISC ID | Title | Rationale for WontFix |
|---------|-------|----------------------|
| DISC-020 | ... | No measurable impact on reproduction |

### Conflicts and Resolutions

**Conflict 1**: [Description of conflicting fixes]
- **Affected**: DISC-XXX, DISC-YYY
- **Resolution**: [How conflict was resolved following severity prioritization]
```

### 8. Validation Results

```markdown
## Validation Results

### Baseline Experiments (Before Fixes)

**Configuration**:
- Epochs: 100
- Seeds: 42, 43, 44, 45, 46
- Commit: [SHA]
- Run Date: YYYY-MM-DD

**Results**:

| Method | Test Error (mean ± std) | Test Accuracy | Convergence Epoch |
|--------|-------------------------|---------------|-------------------|
| PN | 0.045 ± 0.002 | 95.5% | 78 |
| uPU | 0.067 ± 0.004 | 93.3% | 82 |
| nnPU | 0.089 ± 0.003 | 91.1% | Did not converge |

**Issues Identified**:
- nnPU does not converge within 100 epochs
- nnPU underperforms uPU (expected: nnPU ≥ uPU)
- Loss curve behavior: [description of observed vs expected]

### Post-Fix Experiments (After Fixes)

**Configuration**:
- Epochs: 100
- Seeds: 42, 43, 44, 45, 46 (same seeds as baseline)
- Commit: [SHA]
- Run Date: YYYY-MM-DD

**Results**:

| Method | Test Error (mean ± std) | Test Accuracy | Convergence Epoch | Δ from Baseline |
|--------|-------------------------|---------------|-------------------|-----------------|
| PN | 0.044 ± 0.002 | 95.6% | 75 | +0.1% |
| uPU | 0.055 ± 0.003 | 94.5% | 68 | +1.2% |
| nnPU | 0.053 ± 0.002 | 94.7% | 71 | **+3.6%** |

**Improvements Observed**:
- ✅ nnPU now converges within 100 epochs
- ✅ nnPU achieves equal performance to uPU
- ✅ Loss curve behavior matches paper expectations

### Comparison to Paper Benchmarks

**Paper Values** (from Table 1, MNIST even/odd, 100 labeled positive):
- PN: ~4.5% error
- uPU: ~5.2% error
- nnPU: ~5.1% error

**Our Post-Fix Values**:
- PN: 4.4% error (Δ: -0.1%, **within ±2%** ✅)
- uPU: 5.5% error (Δ: +0.3%, **within ±2%** ✅)
- nnPU: 5.3% error (Δ: +0.2%, **within ±2%** ✅)

**Conclusion**: All methods reproduce paper results within acceptable tolerance.

### Statistical Significance

| Method | Baseline | Post-Fix | p-value | Significant? |
|--------|----------|----------|---------|--------------|
| PN | 0.045 | 0.044 | 0.234 | No (marginal change) |
| uPU | 0.067 | 0.055 | 0.012 | ✅ Yes |
| nnPU | 0.089 | 0.053 | 0.0003 | ✅ Yes |

**Test Used**: Paired t-test across 5 seeds

### Test Coverage

**Before Fixes**: 82.3%
**After Fixes**: 84.1%
**Target**: ≥80% ✅

**New Tests Added**:
- `tests/unit/test_losses.py`: 12 new tests for loss function fixes
- `tests/integration/test_experiments.py`: 3 new integration tests
```

---

## Formatting Guidelines

### Code Snippets

Use Python syntax highlighting:

```python
# code here
```

### Mathematical Equations

Use code blocks for equations (or LaTeX if rendering supported):

```
R_nnPU(f) = max(0, R_uPU(f)) + β * min(0, R_uPU(f))
```

### Links and Cross-References

- Use markdown heading anchors for cross-references: `[DISC-003](#disc-003)`
- Link to source files: `[losses/nnpu.py](../../src/pu_learning/losses/nnpu.py)`
- Link to paper sections: `Paper Section 3.2, Equation 6`

### Tables

Use markdown tables with alignment:

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value    | Value    | Value    |
```

---

## Validation Checklist

Before marking report complete, verify:

- [ ] All sections present and filled in
- [ ] Each discrepancy has unique DISC-XXX ID
- [ ] All discrepancies categorized (Mathematical | Procedural | Hyperparameter)
- [ ] All discrepancies assigned severity (Critical | High | Medium | Low)
- [ ] Each discrepancy cites specific source (paper section/equation OR reference file:line)
- [ ] Each discrepancy references current code location (file:class:line)
- [ ] Impact assessment provided for each discrepancy (expected + measured if available)
- [ ] Fix plan prioritized by severity
- [ ] Baseline experiment results captured (before any fixes)
- [ ] Post-fix experiment results captured (after all fixes)
- [ ] Statistical significance testing performed
- [ ] Comparison to paper benchmarks included
- [ ] Test coverage verified ≥80%
- [ ] All conflicts documented and resolved

---

## Example Discrepancy Entry

See [data-model.md](../data-model.md#example) for a complete example of the Discrepancy entity structure in JSON format. The markdown report should contain equivalent information in human-readable form.
