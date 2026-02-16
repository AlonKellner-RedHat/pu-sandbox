# Research Findings: nnPU Paper Reproduction Analysis

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)
**Created**: 2026-02-16
**Status**: Complete

## Overview

This document captures research findings and technical decisions for analyzing discrepancies between the current nnPU implementation and the original paper. All decisions are based on best practices for scientific reproduction analysis, comparison with reference implementations, and systematic validation approaches.

---

## Research Question 1: Systematic Mathematical Comparison

**Question**: What's the best approach to compare loss function implementations against paper equations?

### Decision
Use a **three-layer term-by-term verification approach**:

1. **TeX Source Analysis**: Extract equations from paper TeX source (Eq. 3, 5, 6) and decompose into individual terms
2. **Reference Implementation Cross-Check**: Compare against both reference implementations (kiryor, cimeister) to identify consensus patterns
3. **Code-to-Math Mapping**: Map each line of current loss function code to specific mathematical terms in paper equations

### Rationale
- **TeX source is authoritative**: Published PDFs may have rendering issues; TeX source shows exact mathematical notation
- **Reference implementations provide validation**: When paper notation is ambiguous, consensus across multiple independent implementations indicates correct interpretation
- **Term-by-term granularity catches subtle errors**: Comparing entire equations can miss sign errors, missing coefficients, or incorrect operator precedence
- **Precedent from scientific software validation**: This approach is standard in computational science for validating implementations against theory

### Alternatives Considered

**Alternative 1: PDF visual comparison**
- Rejected because: PDFs may have rendering artifacts, difficult to parse programmatically, copy-paste introduces encoding errors

**Alternative 2: Symbolic math verification (SymPy)**
- Considered but not primary approach because: Requires manual translation of both paper and code to symbolic form, which introduces its own error risks; better used as supplementary validation after manual comparison

**Alternative 3: Unit test with known inputs/outputs**
- Considered but insufficient alone because: Requires knowing correct outputs for test cases; doesn't validate general correctness of formula; useful as secondary validation after formula is verified correct

### Implementation Notes
- Download TeX source from `arxiv.org/src/1703.00593`
- Clone reference implementations to `analysis/reference-implementations/`
- Create comparison matrix: `[Paper Eq Term] | [Current Code] | [Kiryor] | [Cimeister] | [Match?]`
- Document each term with: mathematical notation, Python code equivalent, file/line references

---

## Research Question 2: Ambiguous Paper Notation Resolution

**Question**: When paper notation is unclear or has typos, how to determine correct interpretation?

### Decision
Use **consensus voting across multiple sources** with this priority hierarchy:

1. **Reference implementation consensus** (both kiryor and cimeister agree) → Highest confidence
2. **Single reference + physical/mathematical validity** (e.g., loss must be non-negative) → High confidence
3. **Related paper citations** (follow citation trail for clarification) → Medium confidence
4. **Direct author contact** (if critical ambiguity blocks reproduction) → Last resort

### Rationale
- **Community validation**: If multiple independent implementations interpret notation identically, likely correct
- **Physical constraints guide interpretation**: In machine learning, certain mathematical properties must hold (e.g., losses ≥ 0, probabilities in [0,1])
- **Academic precedent**: Following citation trail is standard practice in scientific research for resolving notation conflicts
- **Efficiency**: Consensus voting avoids analysis paralysis; can proceed with high-confidence interpretation and document uncertainty

### Alternatives Considered

**Alternative 1: Always defer to TeX source literally**
- Rejected because: Papers can have typos; literal interpretation of incorrect notation leads to incorrect implementation

**Alternative 2: Contact authors first**
- Rejected as primary approach because: Slow response time blocks progress; authors may not recall details years later; community implementations may have already resolved ambiguity

**Alternative 3: Prefer one reference implementation as "canonical"**
- Rejected because: No objective basis for choosing one over another; single implementation may have its own bugs

### Implementation Notes
- Document all ambiguities in discrepancy report with: paper notation, reference implementations' interpretations, chosen interpretation + rationale
- Mark high-uncertainty interpretations for potential revisiting if reproduction fails
- Create mapping table: `[Paper Notation] | [Kiryor Interpretation] | [Cimeister Interpretation] | [Our Choice] | [Confidence]`

---

## Research Question 3: Discrepancy Severity Criteria

**Question**: What criteria determine Critical vs High vs Medium vs Low severity?

### Decision
Use **impact-based severity rubric**:

| Severity | Impact Criteria | Examples | Fix Priority |
|----------|----------------|----------|--------------|
| **Critical** | Prevents convergence or makes reproduction impossible; mathematical correctness violated | Wrong sign in loss gradient, missing non-negative constraint, incorrect risk estimator | MUST fix (blocking) |
| **High** | Significantly degrades performance (>5% error rate difference); prevents meeting ±2% reproduction target | Wrong learning rate by order of magnitude, incorrect batch construction, missing key hyperparameter | MUST fix (blocking) |
| **Medium** | Minor performance impact (<5% difference); reproduction possible but not optimal | Suboptimal initialization, minor hyperparameter mismatch, logging/output differences | SHOULD fix (time-permitting) |
| **Low** | No measurable performance impact; cosmetic or documentation issues | Variable naming differences, code style, non-functional improvements | DOCUMENT only |

### Rationale
- **Quantifiable thresholds**: 5% and 2% thresholds align with success criteria (SC-002: within ±2% of paper)
- **Blocking vs non-blocking**: Critical/High must be fixed to achieve reproduction; Medium/Low are optimizations
- **Mathematical correctness is critical**: Any violation of mathematical properties (e.g., nnPU loss going negative) is Critical regardless of empirical performance
- **Evidence-based prioritization**: Severity determined by measured impact on baseline experiments, not subjective assessment

### Alternatives Considered

**Alternative 1: All discrepancies same priority**
- Rejected because: Inefficient use of time; cosmetic issues don't prevent reproduction; may never complete if perfectionism applied to everything

**Alternative 2: Developer judgment-based severity**
- Rejected because: Subjective, inconsistent, hard to justify; quantitative rubric provides objective standard

**Alternative 3: Paper section-based severity (e.g., "core algorithm" = Critical)**
- Rejected because: Doesn't account for actual measured impact; assumptions about importance may be wrong

### Implementation Notes
- **Initial severity assignment**: Based on discrepancy category (mathematical > procedural > hyperparameter) and theoretical impact
- **Severity validation**: After baseline experiments, reassess severity based on actual performance impact
- **Severity escalation**: If Medium discrepancy is found to cause >5% degradation in experiments, escalate to High
- **Document all severities**: Include rationale in discrepancy report (not just labels)

---

## Research Question 4: Discrepancy Report Structure

**Question**: What fields/format ensure each discrepancy is traceable to source and verifiable?

### Decision
Use **structured markdown report** with standardized entry format:

```markdown
### DISC-XXX: [Brief Title]

**Category**: Mathematical | Procedural | Hyperparameter
**Severity**: Critical | High | Medium | Low
**Component**: `src/pu_learning/path/to/file.py:ClassName.method_name:line_number`

#### Description
[1-2 sentence summary of what discrepancy was found]

#### Expected Behavior
[What paper/reference specifies, with specific citation]
- **Source**: Paper Section X.Y, Equation Z | Kiryor: `file.py:line` | Cimeister: `file.py:line`
- **Specification**: [Exact quote or mathematical formula from source]

#### Actual Behavior
[What current implementation does]
- **Current Code**: `src/pu_learning/losses/nnpu.py:42-45`
- **Implementation**: [Code snippet or description]

#### Impact
[How this affects reproduction accuracy, training behavior, or final metrics]
- **Expected Impact**: [Theoretical/predicted effect]
- **Measured Impact** (if baseline run): [Actual performance difference]

#### Proposed Fix
[Specific code changes needed to resolve discrepancy]

#### Fix Status
- [ ] Not Started
- [ ] In Progress
- [ ] Implemented
- [ ] Tested
- [ ] Validated (experiments confirm improvement)
- [ ] Won't Fix (with rationale)

#### Related Discrepancies
[Links to other DISCs that may be related or conflicting]
```

### Rationale
- **Unique IDs**: DISC-XXX format allows cross-referencing across documents, commits, tests
- **Structured fields**: Ensures no critical information omitted; supports automation (e.g., filtering by severity)
- **Dual sourcing**: Both "Expected" (paper/reference) and "Actual" (current code) with line numbers enables verification
- **Impact tracking**: Separates theoretical vs measured impact; baseline experiments fill in "Measured Impact"
- **Fix lifecycle**: Status checklist tracks progress from identification through validation
- **Traceability**: File paths and line numbers allow direct navigation to code
- **Standard format**: Consistent across all discrepancies enables systematic review

### Alternatives Considered

**Alternative 1: Flat bullet list**
- Rejected because: Difficult to parse, no structure for verification, can't track fix status, hard to reference specific discrepancies

**Alternative 2: Spreadsheet/CSV**
- Rejected because: Poor for long descriptions, code snippets, mathematical notation; not git-friendly; harder to review in PRs

**Alternative 3: GitHub Issues**
- Rejected because: External to repo, harder to archive with code, requires network access, not suitable for comprehensive analysis document

### Implementation Notes
- **ID numbering**: Sequential starting from DISC-001, grouped by category if helpful
- **Markdown file location**: `analysis/nnpu-discrepancy-report.md`
- **Section organization**: Group by category (Mathematical, Procedural, Hyperparameter) then severity within each
- **Cross-references**: Use markdown links `[DISC-XXX](#disc-xxx)` for internal references
- **Code snippets**: Use syntax highlighting (```python) for readability
- **Tables**: Use markdown tables for side-by-side comparisons where helpful

---

## Research Question 5: Fix Validation Strategy

**Question**: What testing strategy ensures fixes improve reproduction without breaking existing functionality?

### Decision
Use **TDD with regression testing and experimental validation**:

**Phase 1: Test-Driven Fix Development**
1. For each Critical/High discrepancy → Write failing unit test (RED)
2. Implement minimal fix to pass test (GREEN)
3. Refactor while keeping tests green (REFACTOR)
4. Commit only when all tests pass (including existing tests)

**Phase 2: Regression Prevention**
1. Run full existing test suite before any fix → Baseline (all must pass)
2. After each fix → Re-run full test suite (all must still pass)
3. Add regression test for the specific bug fixed

**Phase 3: Experimental Validation**
1. After all Critical/High fixes applied → Run full experiments (100 epochs, 5 seeds)
2. Compare post-fix results vs baseline results (quantify improvement)
3. Compare post-fix results vs paper benchmarks (verify within ±2%)
4. If any metric regresses, investigate and iterate

### Rationale
- **TDD ensures correctness**: Writing test first forces clear specification of expected behavior
- **Regression tests prevent re-introduction**: Captures the specific bug in a test that will fail if bug returns
- **Experimental validation is mandatory**: Unit tests validate correctness of components; experiments validate end-to-end reproduction
- **Constitutional compliance**: Maintains ≥80% coverage (Principle III); follows TDD methodology (Principle VI)
- **Incremental verification**: Each fix validated independently before moving to next; prevents cascading failures

### Alternatives Considered

**Alternative 1: Fix all discrepancies then test**
- Rejected because: If tests fail, hard to identify which fix caused problem; violates TDD principle; high risk of breaking existing functionality

**Alternative 2: Only experimental validation (no unit tests)**
- Rejected because: Slow feedback loop (experiments take hours); doesn't meet coverage requirements; hard to debug failures without unit-level validation

**Alternative 3: Manual code review only**
- Rejected because: Subjective, error-prone, doesn't prevent regressions, not reproducible

### Implementation Notes

**Test Coverage Requirements** (from clarifications):
- **Critical/High severity fixes**: MUST have test coverage (constitutional requirement)
- **Medium severity fixes**: SHOULD have test coverage (recommended but not blocking)
- **Low severity fixes**: MAY have test coverage (at discretion)

**Test Organization**:
- Unit tests: `tests/unit/test_<component>.py` (e.g., `test_losses.py` for loss function fixes)
- Integration tests: `tests/integration/test_experiments.py` (validate full experiment workflows)
- Regression tests: Add to existing test files with comment `# Regression test for DISC-XXX`

**Testing Commands**:
```bash
# Run full test suite with coverage
pytest --cov=src/pu_learning --cov-report=term-missing

# Run specific test category
pytest tests/unit/test_losses.py -v

# Verify ≥80% coverage after fixes
pytest --cov=src/pu_learning --cov-fail-under=80
```

**Experimental Validation**:
- Baseline experiments: `experiments/results/baseline/` (captured before any fixes)
- Post-fix experiments: `experiments/results/fixed/` (captured after all fixes)
- Comparison script: `scripts/compare-results.py` (generates before/after metrics)

**Acceptance Criteria for Fix Completion**:
1. ✅ Unit test exists and passes
2. ✅ All existing tests still pass (no regressions)
3. ✅ Test coverage maintained ≥80%
4. ✅ Fix resolves discrepancy as verified by test
5. ✅ (After all fixes) Experiments show improvement over baseline
6. ✅ (After all fixes) Results within ±2% of paper benchmarks

---

## Technology Stack Validation

### Existing Dependencies (No Changes Required)

**Language**: Python 3.12 ✅
- Already in use per `pyproject.toml`
- Compatible with PyTorch MPS backend on M4 MacBook Pro

**Core Framework**: PyTorch >= 2.0.0 ✅
- Already configured with MPS acceleration
- Existing loss functions in `src/pu_learning/losses/`
- No migration needed

**Visualization**: hvplot + holoviews + bokeh ✅
- Already configured in project
- Existing notebook: `notebooks/02_results_analysis.ipynb`
- Constitution-compliant (no matplotlib)

**Testing**: pytest >= 7.4.0, pytest-cov ✅
- Already in use with >80% coverage
- No new testing framework needed

**Package Management**: uv ✅
- Already in use for dependency management
- No migration needed

**Linting/Formatting**: ruff ✅
- Already configured
- Constitution-compliant (no black/mypy)

### New Requirements (Analysis-Specific)

**None** - All analysis can be performed with existing tools:
- Paper TeX source: Download via `curl` or `wget` (standard CLI tools)
- Reference implementations: Clone via `git clone` (already available)
- Comparison scripts: Python scripts using existing dependencies
- Report generation: Markdown files (no special tools needed)

---

## Integration Patterns

### Paper Source Acquisition

**Pattern**: Download and extract TeX source directly from arXiv

```bash
# Fetch TeX source tarball
mkdir -p analysis/paper-source
cd analysis/paper-source
wget https://arxiv.org/src/1703.00593
tar -xzf 1703.00593
```

**Files of Interest**:
- `main.tex` or `paper.tex`: Likely contains equations
- `*.bib`: References for citation trail
- Figures/tables: May contain experimental setup details

### Reference Implementation Integration

**Pattern**: Clone repositories for local comparison (no dependency integration)

```bash
mkdir -p analysis/reference-implementations
cd analysis/reference-implementations

# Clone kiryor's implementation
git clone https://github.com/kiryor/nnPUlearning kiryor-nnPUlearning

# Clone cimeister's implementation
git clone https://github.com/cimeister/pu-learning cimeister-pu-learning
```

**Analysis Approach**:
- No code import/dependency on reference implementations
- Manual code review and comparison
- Extract key implementation patterns (optimizer, learning rate, batch construction)
- Document differences in discrepancy report

### Baseline Experiment Workflow

**Pattern**: Capture reproducible baseline before any fixes

```bash
# Run baseline experiments (100 epochs, 5 seeds)
python experiments/scripts/run_all.py \
  --seeds 42 43 44 45 46 \
  --epochs 100 \
  --output experiments/results/baseline/

# Results stored as JSON files:
# experiments/results/baseline/PN_seed42.json
# experiments/results/baseline/uPU_seed42.json
# experiments/results/baseline/nnPU_seed42.json
# (15 files total: 3 methods × 5 seeds)
```

**Baseline Metrics to Capture** (from clarifications):
- Final test error rate (zero-one loss) - primary metric
- Training/test loss curves - convergence behavior
- Per-epoch test accuracy - learning dynamics
- Mean ± std across 5 seeds - statistical validity
- Git commit SHA - reproducibility

---

## Success

 Criteria Mapping

| Success Criterion | Research Support |
|------------------|------------------|
| SC-001: Zero mathematical discrepancies | RQ1: Term-by-term comparison approach |
| SC-002: ±2% reproduction accuracy | RQ3: High severity threshold, RQ5: Experimental validation |
| SC-003: nnPU ≥ uPU performance | RQ5: Post-fix experiment comparison |
| SC-004: Expected loss curve behavior | RQ3: Critical severity for mathematical violations |
| SC-005: Hyperparameter matching | RQ1: Systematic comparison against paper |
| SC-006: ≥90% discrepancy detection | RQ4: Structured report format ensures comprehensive capture |
| SC-007: Critical/High fixes completed | RQ3: Severity rubric defines Critical/High clearly |
| SC-008: ≥80% test coverage maintained | RQ5: TDD with coverage requirements |
| SC-009: Quantified improvement over baseline | RQ5: Before/after experimental comparison |

---

## Next Steps

This research phase is complete. Proceed to Phase 1 (Design) to create:

1. **data-model.md**: Entity schemas for Discrepancy, Baseline Results, Comparison
2. **contracts/**: Standard formats for discrepancy report, experiment results, analysis scripts
3. **quickstart.md**: Setup and usage guide for running analysis and validation

All research questions have been answered with clear decisions, rationales, and implementation notes.
