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

*This section will be populated during User Story 1 (Phase 3).*

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
