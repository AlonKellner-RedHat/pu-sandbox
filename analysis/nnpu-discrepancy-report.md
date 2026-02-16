# nnPU Paper Reproduction Discrepancy Report

**Feature**: [specs/001-nnpu-reproduction-analysis/spec.md](../specs/001-nnpu-reproduction-analysis/spec.md)
**Analysis Date**: 2026-02-16
**Analyst**: Claude Code + User
**Status**: Draft

**Sources Analyzed**:
- Paper: arxiv.org/src/1703.00593 (TeX source)
- Reference 1: github.com/kiryor/nnPUlearning (commit 3d9769e, last updated 2022-12-14)
- Reference 2: github.com/cimeister/pu-learning (commit 39d1d24, last updated 2024-08-21)
- Current Implementation: (commit ac95e3c)

---

## Executive Summary

**Total Discrepancies Found**: [TBD - will be filled during analysis]

**Breakdown by Severity**:
- Critical: [TBD] - None fixed yet
- High: [TBD] - None fixed yet
- Medium: [TBD] - None fixed yet
- Low: [TBD] - None fixed yet

**Breakdown by Category**:
- Mathematical: [TBD] discrepancies
- Procedural: [TBD] discrepancies
- Hyperparameter: [TBD] discrepancies

**Key Findings**:
- [Will be filled after analysis phase - US1, US2, US3]
- [Impact on reproduction accuracy will be assessed after baseline experiments]
- [Overall assessment pending]

**Reproduction Status**:
- Baseline Test Error: [TBD - will be filled after baseline experiments in Phase 2]
- Paper Benchmark: [TBD - to be extracted from paper]
- Delta: [TBD]
- **Within Target**: [TBD] (±2% tolerance)

---

## Severity Matrix

Summary table of all discrepancies sorted by severity:

| ID | Category | Severity | Component | Status | Impact |
|----|----------|----------|-----------|--------|--------|
| [TBD] | [TBD] | [TBD] | [TBD] | NotStarted | Unknown |

**Legend**:
- **Status**: NotStarted | InProgress | Implemented | Tested | Validated | WontFix
- **Impact**: High (>5% degradation) | Medium (1-5%) | Low (<1%) | Unknown (no baseline data)

*This table will be populated during the analysis phase (User Stories 1-3).*

---

## Mathematical Discrepancies

Discrepancies in loss function formulations and mathematical operations.

---

### DISC-001: PN Loss Missing Prior Weighting

**Category**: Mathematical
**Severity**: High
**Component**: `src/pu_learning/losses/pn_loss.py:PNLoss.forward:73`
**Status**: NotStarted

#### Description

The PN loss implementation does not weight the positive and negative risks by their class priors (π_p and π_n). It simply sums the two risk components without weighting.

#### Expected Behavior

**Source**: Paper pu_nnre.tex, Line 141, Equation `eq:risk-pn-hat`

**Specification**:
```
\hRpn(g) = π_p * \hRp^+(g) + π_n * \hRn^-(g)
```

Where:
- π_p = Prior probability of positive class
- π_n = Prior probability of negative class (π_n = 1 - π_p)
- \hRp^+(g) = (1/N_p) * Σ ℓ(g(x_p_i), +1)
- \hRn^-(g) = (1/N_n) * Σ ℓ(g(x_n_i), -1)

**Reference Implementations**:
- kiryor: [TBD - needs review]
- cimeister: [TBD - needs review]

#### Actual Behavior

**Current Code**: `src/pu_learning/losses/pn_loss.py:73`

**Implementation**:
```python
# Total loss
return positive_loss + negative_loss
```

The current implementation computes:
```
L_PN = E_P[BCE(f(x), 1)] + E_N[BCE(f(x), 0)]
```

But should compute:
```
L_PN = π_p * E_P[BCE(f(x), 1)] + π_n * E_N[BCE(f(x), 0)]
```

#### Impact

**Expected Impact**:
- For balanced datasets (π_p ≈ 0.5): Minimal impact (~1-2% test error change)
- For imbalanced datasets: High impact (>5% degradation possible)
- MNIST even/odd has π_p ≈ 0.5, so impact should be Low-Medium

**Measured Impact**: [Will be filled after baseline experiments]

#### Proposed Fix

**Changes Required**:
```python
# File: src/pu_learning/losses/pn_loss.py

# In __init__, accept prior parameter
def __init__(self, prior: float = 0.5) -> None:
    """Initialize PN loss.

    Args:
        prior: Class prior π_p = P(y=1). Defaults to 0.5 (balanced).
    """
    super().__init__(prior=prior)

# In forward, apply prior weighting
def forward(self, outputs, labels, is_labeled) -> torch.Tensor:
    # ... existing code to compute positive_loss and negative_loss ...

    # Apply prior weighting
    pi_p = self.prior
    pi_n = 1.0 - self.prior

    return pi_p * positive_loss + pi_n * negative_loss
```

#### Related Discrepancies

None currently identified.

---

### DISC-002: nnPU Loss May Have Gradient Flow Issue in Max Operation

**Category**: Mathematical
**Severity**: Critical
**Component**: `src/pu_learning/losses/nnpu_loss.py:nnPULoss.forward:94-97`
**Status**: NotStarted

#### Description

The nnPU loss uses `torch.clamp(negative_risk, min=0.0)` to implement the max(0, ...) operation. This may allow gradients to flow through the comparison, which could violate the paper's intent for the non-negative correction. The paper discussion suggests that when risk goes negative, gradient should not flow through the max operation.

#### Expected Behavior

**Source**: Paper pu_nnre.tex, Line 223, Equation `eq:risk-pu-tilde`

**Specification**:
```
\tRpu(g) = π_p * \hRp^+(g) + max{0, \hRu^-(g) - π_p * \hRp^-(g)}
```

**Gradient Behavior** (inferred from paper discussion and standard practice):
When the negative risk term goes below 0, gradients should NOT flow through the max operation. This typically requires:
1. Detaching the comparison term before checking if < 0
2. Using a conditional with `.detach()` rather than `torch.clamp`

**Reference Implementations**:
- kiryor: [TBD - needs review of gradient handling]
- cimeister: [TBD - needs review of gradient handling]

#### Actual Behavior

**Current Code**: `src/pu_learning/losses/nnpu_loss.py:94-97`

**Implementation**:
```python
# Compute the unlabeled negative risk with correction
# This term can be negative in uPU, but we clamp to 0 in nnPU
negative_risk = unlabeled_negative_risk - self.prior * positive_negative_risk

# nnPU: Apply max(0, ...) to negative risk term (KEY DIFFERENCE from uPU)
negative_risk_clamped = torch.clamp(negative_risk, min=0.0)
```

Using `torch.clamp` allows gradient flow in both directions:
- When `negative_risk >= 0`: gradient flows normally
- When `negative_risk < 0`: gradient is **zeroed** (derivative of clamp at boundary = 0)

This might be correct, but paper implementation often uses explicit detach:
```python
if negative_risk.detach() < 0:
    negative_risk_clamped = -self.beta * negative_risk.detach()  # or just 0
else:
    negative_risk_clamped = negative_risk
```

#### Impact

**Expected Impact**:
- **Critical** if gradient flow causes instability or prevents convergence
- Could explain why nnPU might not converge within 100 epochs
- Could cause loss to become negative despite assertion (though assertion would catch this)

**Measured Impact**: [Will be filled after baseline experiments - check if nnPU converges]

#### Proposed Fix

**Option 1: Use detached comparison (conservative approach)**:
```python
# File: src/pu_learning/losses/nnpu_loss.py:94-100

# Compute the unlabeled negative risk with correction
negative_risk = unlabeled_negative_risk - self.prior * positive_negative_risk

# nnPU: Apply max(0, ...) with detached comparison
if negative_risk.detach().item() < 0:
    # When negative, use detached value multiplied by -beta (or just 0 if beta=0)
    # Gradients do NOT flow through this branch
    negative_risk_clamped = torch.zeros_like(negative_risk)
else:
    # When positive, gradients flow normally
    negative_risk_clamped = negative_risk

# Final loss
loss = self.prior * positive_risk + negative_risk_clamped
```

**Option 2: Current clamp approach may be correct**:
- Need to verify with reference implementations
- `torch.clamp` gradient behavior might be intentional
- Defer decision until reference review (T017-T018)

#### Related Discrepancies

May relate to convergence issues if they exist (to be identified in T012 baseline analysis).

---

### Structure for Each Discrepancy

Each discrepancy will follow this format:

```
### DISC-XXX: [Brief Title]

**Category**: Mathematical
**Severity**: Critical | High | Medium | Low
**Component**: `src/pu_learning/path/to/file.py:ClassName.method:line`
**Status**: NotStarted

#### Description
[What discrepancy was found]

#### Expected Behavior
**Source**: Paper Section X, Equation Y
**Specification**: [Quote from paper or reference code]

#### Actual Behavior
**Current Code**: `src/pu_learning/file.py:line`
**Implementation**: [Current code snippet]

#### Impact
**Expected Impact**: [Theoretical/predicted effect]
**Measured Impact**: [Will be filled after baseline experiments]

#### Proposed Fix
[Specific code changes needed]

#### Related Discrepancies
[Links to related DISC-XXX]
```

---

## Procedural Discrepancies

Discrepancies in training algorithm, optimization procedure, and evaluation workflow.

*This section will be populated during User Story 2 (Phase 4).*

---

## Hyperparameter Discrepancies

Discrepancies in configuration values and experimental setup.

*This section will be populated during User Story 3 (Phase 5).*

---

## Fix Plan and Progress

### Phase 1: Critical Fixes (Blocking)

*Will be populated after discrepancies are identified and prioritized.*

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| [TBD] | [TBD] | [TBD] | [TBD] | NotStarted |

**Blockers**: [None currently]

### Phase 2: High Priority Fixes (Blocking)

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| [TBD] | [TBD] | [TBD] | [TBD] | NotStarted |

### Phase 3: Medium Priority Fixes (Time-Permitting)

| DISC ID | Title | Assigned | ETA | Status |
|---------|-------|----------|-----|--------|
| [TBD] | [TBD] | [TBD] | [TBD] | NotStarted |

### Phase 4: Low Priority (Document Only)

| DISC ID | Title | Rationale for WontFix |
|---------|-------|----------------------|
| [TBD] | [TBD] | [TBD] |

### Conflicts and Resolutions

*Will be documented if any conflicting fixes are discovered.*

---

## Validation Results

### Baseline Experiments (Before Fixes)

**Configuration**:
- Epochs: [TBD - will be captured in Phase 2]
- Seeds: [TBD]
- Commit: [TBD]
- Run Date: [TBD]

**Results**:

| Method | Test Error (mean ± std) | Test Accuracy | Convergence Epoch |
|--------|-------------------------|---------------|-------------------|
| PN | [TBD] | [TBD] | [TBD] |
| uPU | [TBD] | [TBD] | [TBD] |
| nnPU | [TBD] | [TBD] | [TBD] |

**Issues Identified**:
- [Will be filled after baseline experiments]

### Post-Fix Experiments (After Fixes)

**Configuration**:
- Epochs: [TBD - will run in Phase 6 (User Story 4)]
- Seeds: [TBD - same seeds as baseline]
- Commit: [TBD]
- Run Date: [TBD]

**Results**:

| Method | Test Error (mean ± std) | Test Accuracy | Convergence Epoch | Δ from Baseline |
|--------|-------------------------|---------------|-------------------|-----------------|
| PN | [TBD] | [TBD] | [TBD] | [TBD] |
| uPU | [TBD] | [TBD] | [TBD] | [TBD] |
| nnPU | [TBD] | [TBD] | [TBD] | [TBD] |

**Improvements Observed**:
- [Will be documented after post-fix experiments]

### Comparison to Paper Benchmarks

**Paper Values** (from Table/Figure [TBD]):
- PN: [TBD]
- uPU: [TBD]
- nnPU: [TBD]

**Our Post-Fix Values**:
- PN: [TBD]
- uPU: [TBD]
- nnPU: [TBD]

**Conclusion**: [Will be documented after validation]

### Statistical Significance

| Method | Baseline | Post-Fix | p-value | Significant? |
|--------|----------|----------|---------|--------------|
| PN | [TBD] | [TBD] | [TBD] | [TBD] |
| uPU | [TBD] | [TBD] | [TBD] | [TBD] |
| nnPU | [TBD] | [TBD] | [TBD] | [TBD] |

**Test Used**: [TBD - likely paired t-test]

### Test Coverage

**Before Fixes**: [TBD - will check current coverage]
**After Fixes**: [TBD - must maintain ≥80%]
**Target**: ≥80% ✓

**New Tests Added**:
- [Will be documented during fix implementation]

---

## Notes

- This report will be progressively filled during the implementation of User Stories 1-4
- Phase 2 (Foundational) will provide baseline metrics needed to assess discrepancy severity
- Phases 3-5 (US1-3) will identify and document all discrepancies
- Phase 6 (US4) will implement fixes and validate improvements
- All findings must be traceable to specific paper sections, equations, or reference implementation code
