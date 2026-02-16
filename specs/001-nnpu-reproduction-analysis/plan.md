# Implementation Plan: nnPU Paper Reproduction Analysis and Fixes

**Branch**: `001-nnpu-reproduction-analysis` | **Date**: 2026-02-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-nnpu-reproduction-analysis/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Perform comprehensive analysis to identify and fix discrepancies between the current nnPU implementation and the original paper (arxiv.org/src/1703.00593). Compare mathematical formulations (loss functions), training procedures (optimizer, learning rate, batch construction), and hyperparameters (epochs, batch size, β, n_positive) against paper specifications and reference implementations (kiryor/nnPUlearning, cimeister/pu-learning). Capture baseline metrics before fixes, implement Critical/High severity fixes with test coverage, validate improvements through full experiments (100 epochs, 5 seeds), and document quantified before/after improvements.

## Technical Context

**Language/Version**: Python 3.12 (already in use per pyproject.toml)
**Primary Dependencies**:
- PyTorch >= 2.0.0 (with MPS support for M4 MacBook Pro)
- hvplot, holoviews, bokeh (interactive visualization - already configured)
- pytest >= 7.4.0, pytest-cov (testing framework - already configured)
- uv (package management - already in use)
- ruff (linting/formatting - already configured)

**Storage**: Local filesystem for:
- Paper TeX source (download from arxiv.org/src/1703.00593)
- Reference implementations (clone from GitHub: kiryor/nnPUlearning, cimeister/pu-learning)
- Discrepancy report markdown (analysis/nnpu-discrepancy-report.md)
- Baseline experiment results (experiments/results/baseline/)
- Post-fix experiment results (experiments/results/fixed/)

**Testing**: pytest with >80% coverage requirement (constitution mandate)
- New tests for Critical/High severity fixes (mandatory per clarifications)
- Regression tests to prevent re-introduction of bugs
- Integration tests validating full experiment workflows

**Target Platform**: macOS (M4 MacBook Pro with MPS acceleration)
**Project Type**: Single project (research codebase) - existing structure in `src/pu_learning/`

**Performance Goals**:
- Training convergence: nnPU must converge within 100 epochs
- Reproduction accuracy: Test error within ±2% of paper values
- Baseline vs post-fix: Measurable improvement in test error

**Constraints**:
- Reproducibility: Same seed must produce identical results (constitution mandate)
- Non-negative nnPU loss: Must stay ≥0 throughout training
- Test coverage: Maintain ≥80% after all fixes (constitution mandate)
- No implementation details in analysis report (focus on mathematical/procedural discrepancies)

**Scale/Scope**:
- 3 loss functions to analyze: PN, uPU, nnPU
- 2 reference implementations to compare against
- 5 random seeds for statistical validation
- 100 epochs per experiment (baseline + post-fix = 200 epochs per method per seed)
- Expected: 10-30 discrepancies across all categories (math, procedure, hyperparameters)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Principle I: Reproducibility First (NON-NEGOTIABLE)
- **Status**: PASS
- **Rationale**: Feature explicitly requires baseline experiments with 5 seeds, capturing reproducible metrics before/after fixes. Analysis will verify seed management and deterministic operations match paper specifications.
- **Actions**:
  - Verify seed setting for Python random, NumPy, PyTorch (CPU/MPS)
  - Confirm deterministic data loading (already implemented in MNISTPUDataset)
  - Validate experiment logging captures all required metadata

### ✅ Principle II: Modular Architecture
- **Status**: PASS
- **Rationale**: Feature focuses on analyzing and fixing existing modular architecture (data, models, losses, training modules). Fixes will preserve clean separation of concerns.
- **Actions**:
  - Ensure loss function fixes remain standalone and composable
  - Validate no circular dependencies introduced by fixes
  - Confirm extensibility maintained

### ✅ Principle III: Test Coverage >80% (NON-NEGOTIABLE)
- **Status**: PASS
- **Rationale**: FR-009 requires maintaining ≥80% coverage; clarifications mandate test coverage for all Critical/High severity fixes. This aligns with constitution.
- **Actions**:
  - Add tests for all Critical/High severity fixes (mandatory)
  - Add tests for Medium severity fixes (should)
  - Run pytest with --cov after all fixes to verify ≥80% maintained

### ✅ Principle IV: Scientific Integrity
- **Status**: PASS - **PRIMARY FOCUS OF THIS FEATURE**
- **Rationale**: This feature IS about ensuring scientific integrity - validating paper alignment, fixing discrepancies, achieving reproduction accuracy.
- **Actions**:
  - Compare loss formulations against paper equations (Eq. 3, 5, 6)
  - Verify MNIST configuration (even/odd split, n_positive=100)
  - Validate zero-one loss as primary metric
  - Report all results (baseline + post-fix) with mean ± std across seeds

### ✅ Principle V: Interactive Visualization
- **Status**: PASS
- **Rationale**: Existing visualization infrastructure (hvPlot/HoloViews/Bokeh) will be used to analyze baseline vs post-fix results. No matplotlib allowed.
- **Actions**:
  - Use existing notebooks/02_results_analysis.ipynb for before/after comparison
  - Generate interactive plots for baseline vs fixed training curves
  - Export HTML dashboards showing improvements

### ✅ Principle VI: Test-Driven Development (TDD) - NON-NEGOTIABLE
- **Status**: PASS
- **Rationale**: All fixes for Critical/High severity discrepancies must follow RED-GREEN-REFACTOR cycle. Write failing test for expected behavior, implement fix to pass test, refactor.
- **Actions**:
  - For each Critical/High fix: Write failing test first (RED)
  - Implement minimal fix to pass test (GREEN)
  - Refactor while keeping tests green (REFACTOR)
  - Commit only when all tests pass

### ✅ Technology Stack Requirements
- **Status**: PASS
- **Rationale**: All required dependencies already in place (PyTorch, hvplot, pytest). Using uv + ruff as mandated. No prohibited tools (matplotlib, mypy, black) will be used.
- **Actions**:
  - Verify PyTorch MPS acceleration works for baseline/post-fix experiments
  - Use ruff for all linting/formatting
  - Use uv for any new dependencies (if needed)

### ✅ Git Workflow - NON-NEGOTIABLE
- **Status**: PASS
- **Rationale**: Feature will follow commit early/commit often workflow. Pre-commit hooks will run before all commits. Co-authoring with Claude Code in all commit messages.
- **Actions**:
  - Commit after each TDD cycle (test + fix + refactor)
  - Push frequently (after each logical unit of work)
  - NEVER use --no-verify (constitution mandate)
  - Use `just p` to verify pre-commit hooks before committing

### Gate Decision: **PROCEED** ✅

All constitution principles align with this feature. No violations. This feature actively enforces constitution Principle IV (Scientific Integrity) by validating paper alignment and fixing discrepancies.

## Project Structure

### Documentation (this feature)

```text
specs/001-nnpu-reproduction-analysis/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0: Research findings on paper analysis approach
├── data-model.md        # Phase 1: Discrepancy report schema
├── quickstart.md        # Phase 1: Quick guide to running analysis and validation
├── contracts/           # Phase 1: Analysis report structure contracts
│   └── discrepancy-report-schema.md
└── tasks.md             # Phase 2: Task breakdown (created by /speckit.tasks)
```

### Source Code (repository root)

```text
# Existing structure (analysis adds to this)
src/pu_learning/
├── data/
│   ├── __init__.py
│   └── datasets.py          # MNISTPUDataset - may need fixes
├── models/
│   ├── __init__.py
│   └── mlp.py               # MLP model - may need fixes
├── losses/
│   ├── __init__.py
│   ├── base.py
│   ├── pn.py                # PNLoss - may need fixes
│   ├── upu.py               # uPULoss - may need fixes
│   └── nnpu.py              # nnPULoss - may need fixes
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Trainer - may need fixes
│   └── metrics.py           # Metrics - may need fixes
└── utils/
    ├── __init__.py
    ├── device.py
    ├── reproducibility.py   # Seeding - may need fixes
    └── visualization.py

# New directories for analysis
analysis/
├── nnpu-discrepancy-report.md   # Main analysis output (FR-007)
├── paper-source/                # Downloaded TeX source
│   └── 1703.00593/              # arxiv.org/src/1703.00593
├── reference-implementations/   # Cloned reference repos
│   ├── kiryor-nnPUlearning/     # github.com/kiryor/nnPUlearning
│   └── cimeister-pu-learning/   # github.com/cimeister/pu-learning
└── baseline-results/            # Baseline experiment outputs
    └── [timestamp]-baseline/    # Results before fixes

experiments/results/
├── baseline/                    # Baseline experiments (FR-010)
│   ├── PN_seed42.json
│   ├── PN_seed43.json
│   ├── ... (5 seeds × 3 methods = 15 files)
└── fixed/                       # Post-fix experiments (FR-010)
    ├── PN_seed42.json
    ├── PN_seed43.json
    └── ... (15 files)

tests/
├── unit/
│   ├── test_losses.py           # May add tests for loss function fixes
│   ├── test_datasets.py         # May add tests for dataset fixes
│   └── test_trainer.py          # May add tests for trainer fixes
├── integration/
│   └── test_experiments.py      # May add baseline/validation tests
└── analysis/                    # New: Tests for analysis workflow
    └── test_discrepancy_detection.py

# Analysis scripts (new)
scripts/
├── fetch-paper-source.sh        # Download arxiv.org/src/1703.00593
├── clone-references.sh          # Clone reference implementations
├── run-baseline-experiments.sh  # FR-010: Baseline experiments
├── run-fixed-experiments.sh     # FR-010: Post-fix validation
└── compare-results.py           # FR-011: Before/after comparison
```

**Structure Decision**: This is an **analysis and fixing feature** for an existing single-project research codebase. The structure follows the existing `src/pu_learning/` modular architecture. Analysis artifacts (paper source, reference implementations, discrepancy report) are organized in a new `analysis/` directory. Baseline and post-fix experiment results are stored in `experiments/results/` with clear separation. Analysis scripts are added to `scripts/` directory for reproducible analysis workflow.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations** - this section is not applicable. All constitution principles are satisfied.

## Phase 0: Research & Technical Decisions

### Research Questions

1. **How to systematically compare mathematical formulations?**
   - **Question**: What's the best approach to compare loss function implementations against paper equations?
   - **Why needed**: Must ensure term-by-term accuracy without missing subtle differences

2. **How to handle ambiguous paper notation?**
   - **Question**: When paper notation is unclear or has typos, how to determine correct interpretation?
   - **Why needed**: Paper may have notational inconsistencies that need resolution

3. **How to prioritize discrepancies by severity?**
   - **Question**: What criteria determine Critical vs High vs Medium vs Low severity?
   - **Why needed**: Need objective rubric for severity ratings to ensure consistent prioritization

4. **How to structure discrepancy report for traceability?**
   - **Question**: What fields/format ensure each discrepancy is traceable to source and verifiable?
   - **Why needed**: FR-007 requires standard detail level with comparisons, expected vs actual, source references

5. **How to validate fixes don't introduce regressions?**
   - **Question**: What testing strategy ensures fixes improve reproduction without breaking existing functionality?
   - **Why needed**: Must maintain test coverage and existing test passage while fixing bugs

### Research Outputs

**All research findings will be documented in `research.md`** (created in Phase 0) with:
- Decision: [what was chosen]
- Rationale: [why chosen]
- Alternatives considered: [what else evaluated]

## Phase 1: Design Artifacts

### Data Models

**Entity: Discrepancy** (documented in `data-model.md`)
- ID: Unique identifier (e.g., DISC-001)
- Category: "Mathematical" | "Procedural" | "Hyperparameter"
- Severity: "Critical" | "High" | "Medium" | "Low"
- Component: File/class/function affected (e.g., "src/pu_learning/losses/nnpu.py:nnPULoss.forward")
- Description: What discrepancy was found
- Expected: What paper/reference implementation specifies (with source citation)
- Actual: What current implementation does (with code reference)
- Impact: How this affects reproduction accuracy
- FixStatus: "NotFixed" | "InProgress" | "Fixed" | "Validated" | "WontFix"

**Entity: Baseline Results** (documented in `data-model.md`)
- Method: "PN" | "uPU" | "nnPU"
- Seed: Integer (42, 43, 44, 45, 46)
- Epochs: Array of epoch numbers
- TrainLoss: Array of training loss values per epoch
- TestLoss: Array of test loss values per epoch
- TestAccuracy: Array of test accuracy values per epoch
- TestZeroOneLoss: Array of test error rates per epoch
- FinalMetrics: Summary of final epoch metrics
- Timestamp: When experiment was run
- CommitHash: Git commit SHA of code used

**Entity: Comparison** (documented in `data-model.md`)
- DiscrepancyID: Link to discrepancy
- BaselineMetric: Pre-fix metric value (from baseline results)
- PostFixMetric: Post-fix metric value (from fixed results)
- Delta: Quantified improvement (post-fix - baseline)
- DeltaPercent: Percentage improvement
- StatisticalSignificance: Whether improvement is significant (t-test)

### API Contracts

**No REST/GraphQL APIs** - this is a research analysis feature. Instead, contracts define:

1. **Discrepancy Report Schema** (`contracts/discrepancy-report-schema.md`):
   - Markdown structure for `analysis/nnpu-discrepancy-report.md`
   - Sections: Executive Summary, Mathematical Discrepancies, Procedural Discrepancies, Hyperparameter Discrepancies, Severity Matrix, Fix Plan
   - Each discrepancy entry format (standardized)

2. **Analysis Script Interface** (`contracts/analysis-scripts.md`):
   - `fetch-paper-source.sh` - Downloads and extracts paper TeX source
   - `clone-references.sh` - Clones reference implementations to specific directory
   - `run-baseline-experiments.sh` - Runs 15 baseline experiments (3 methods × 5 seeds)
   - `run-fixed-experiments.sh` - Runs 15 post-fix validation experiments
   - `compare-results.py` - Generates before/after comparison report

3. **Experiment Results Format** (`contracts/experiment-results.json`):
   - JSON schema for experiment output files
   - Fields: method, seed, config, results (per-epoch metrics), final_metrics, timestamp, commit_hash

### Quickstart Guide

**`quickstart.md`** will provide:
1. **Setup**: Clone repo, install dependencies with `uv sync`
2. **Run Analysis**: Step-by-step commands to fetch paper, run baseline, identify discrepancies
3. **Apply Fixes**: How to implement and test fixes following TDD
4. **Validate**: How to run post-fix experiments and compare results
5. **View Results**: How to use notebooks to visualize before/after improvements

## Next Steps

After this plan is complete (`/speckit.plan` command finishes):

1. **Phase 0 Execution**: Research questions will be investigated and `research.md` will be created with findings
2. **Phase 1 Execution**: Design artifacts will be created:
   - `data-model.md` with entity schemas
   - `contracts/` directory with all contract definitions
   - `quickstart.md` with setup and usage guide
3. **Agent Context Update**: Run `.specify/scripts/bash/update-agent-context.sh claude` to update agent context with new analysis workflow
4. **Task Breakdown**: After planning complete, user runs `/speckit.tasks` to generate actionable task list in `tasks.md`
5. **Implementation**: Follow tasks in TDD methodology (RED-GREEN-REFACTOR) to execute analysis and fixes

## Success Criteria Validation

This plan satisfies all success criteria from spec.md:

- **SC-001**: Mathematical formulation comparison in Phase 0/1 research
- **SC-002**: ±2% reproduction accuracy validated in post-fix experiments
- **SC-003**: nnPU ≥ uPU performance validated in comparison analysis
- **SC-004**: Loss curve behavior validated in experimental analysis
- **SC-005**: Hyperparameter matching validated in discrepancy analysis
- **SC-006**: 90% discrepancy detection through systematic comparison
- **SC-007**: Critical/High fixes with priority-based implementation
- **SC-008**: Test coverage maintained through TDD for all fixes
- **SC-009**: Quantified improvement documented in before/after comparison
