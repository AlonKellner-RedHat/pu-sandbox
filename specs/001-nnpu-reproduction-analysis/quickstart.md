# Quickstart Guide: nnPU Paper Reproduction Analysis

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)
**Created**: 2026-02-16
**Audience**: Developers and researchers working on nnPU paper reproduction fixes

## Overview

This guide provides step-by-step instructions for running the complete nnPU reproduction analysis workflow, from initial setup through fix validation. Follow these steps to systematically identify and fix discrepancies between the current implementation and the original paper.

**Time Estimate**: ~6-8 hours total (analysis: 2h, baseline experiments: 2h, fixing: 2-3h, validation: 2h)

---

## Prerequisites

### System Requirements

- **Hardware**: M4 MacBook Pro (or compatible system with MPS/CUDA/CPU)
- **OS**: macOS (tested), Linux (should work), Windows WSL (should work)
- **RAM**: ≥16GB recommended for experiments
- **Disk**: ≥5GB free space (for datasets, results, paper source)

### Software Requirements

- **Python**: 3.12 (already configured in project)
- **Git**: For version control
- **uv**: Package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Internet**: For downloading paper source and cloning reference repos

### Knowledge Requirements

- Familiarity with PU (Positive-Unlabeled) learning concepts
- Basic understanding of PyTorch and neural network training
- Ability to read research papers and compare mathematical formulations

---

## Workflow Phases

```
Phase 1: Setup & Analysis
├─ 1.1: Clone repo and install dependencies
├─ 1.2: Fetch paper source and reference implementations
├─ 1.3: Analyze discrepancies
└─ 1.4: Document findings in report

Phase 2: Baseline Experiments
├─ 2.1: Run baseline experiments (before fixes)
├─ 2.2: Analyze baseline results
└─ 2.3: Confirm reproduction gap

Phase 3: Fix Implementation
├─ 3.1: Prioritize fixes by severity
├─ 3.2: Implement fixes using TDD
├─ 3.3: Verify all tests pass
└─ 3.4: Commit fixes

Phase 4: Validation
├─ 4.1: Run post-fix experiments
├─ 4.2: Compare baseline vs post-fix
├─ 4.3: Verify paper reproduction
└─ 4.4: Document improvements
```

---

## Phase 1: Setup & Analysis

### 1.1: Clone Repository and Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd pu-sandbox

# Checkout feature branch
git checkout 001-nnpu-reproduction-analysis

# Install dependencies using uv
uv sync

# Verify installation
uv run pytest --version  # Should show pytest version
uv run python -c "import torch; print(torch.__version__)"  # Should show PyTorch version
```

**Expected Output**:
```
✓ Python 3.12.x
✓ PyTorch 2.x.x
✓ pytest 7.x.x
```

### 1.2: Fetch Paper Source and Reference Implementations

```bash
# Fetch paper TeX source from arXiv
./scripts/fetch-paper-source.sh

# Clone reference implementations
./scripts/clone-references.sh
```

**Expected Output**:
```
Fetching paper source for 1703.00593...
✓ Downloaded source tarball
✓ Extracted to analysis/paper-source/1703.00593/
✓ Found TeX files: nnPU.tex

Cloning reference implementations...
✓ Cloned kiryor/nnPUlearning (commit: a1b2c3d...)
✓ Cloned cimeister/pu-learning (commit: e5f6g7h...)
```

**Verify Files**:
```bash
# Paper source should exist
ls -la analysis/paper-source/1703.00593/

# Reference implementations should exist
ls -la analysis/reference-implementations/
```

### 1.3: Analyze Discrepancies

Now perform systematic comparison across three dimensions:

#### Mathematical Formulations

1. Open paper TeX source: `analysis/paper-source/1703.00593/nnPU.tex`
2. Locate loss function equations (typically Eq. 3, 5, 6)
3. Open current implementations:
   - `src/pu_learning/losses/pn.py`
   - `src/pu_learning/losses/upu.py`
   - `src/pu_learning/losses/nnpu.py`
4. Compare term-by-term:
   - Signs (+ vs -)
   - Coefficients (π, β, etc.)
   - Expectations (averages over positive/unlabeled data)
   - Gradient flow (detach operations)

5. Cross-check with reference implementations:
   ```bash
   # View kiryor's nnPU loss
   cat analysis/reference-implementations/kiryor-nnPUlearning/pu_loss.py

   # View cimeister's nnPU loss
   cat analysis/reference-implementations/cimeister-pu-learning/losses.py
   ```

6. Document each discrepancy found (see template in next section)

#### Training Procedures

1. Open paper Section 4 (Experiments)
2. Extract training details:
   - Optimizer (SGD? Adam? Momentum?)
   - Learning rate and schedule
   - Batch size
   - Number of epochs
3. Compare against current trainer: `src/pu_learning/training/trainer.py`
4. Check batch construction in dataset: `src/pu_learning/data/datasets.py`
5. Document discrepancies

#### Hyperparameters

1. Open paper experimental setup (typically Section 4 or Table 1)
2. Extract MNIST configuration:
   - Number of labeled positive samples (n_positive)
   - Class prior (π)
   - Beta parameter (β) for nnPU
   - Dataset split (even vs odd)
3. Compare against experiment configs: `experiments/configs/`
4. Document discrepancies

### 1.4: Document Findings in Discrepancy Report

Create the analysis report following the schema:

```bash
# Create report file
touch analysis/nnpu-discrepancy-report.md
```

Open `analysis/nnpu-discrepancy-report.md` and follow the structure from [contracts/discrepancy-report-schema.md](contracts/discrepancy-report-schema.md).

**Template for Each Discrepancy**:

```markdown
### DISC-001: [Brief Title]

**Category**: Mathematical | Procedural | Hyperparameter
**Severity**: Critical | High | Medium | Low
**Component**: `src/pu_learning/path/to/file.py:ClassName.method:line`
**Status**: NotStarted

#### Description
[What discrepancy was found]

#### Expected Behavior
**Source**: Paper Section X, Equation Y
**Specification**: [Quote from paper or reference code]

#### Actual Behavior
**Current Code**: `src/pu_learning/file.py:42`
**Implementation**: [Current code snippet]

#### Impact
**Expected Impact**: [Predicted effect]
**Measured Impact**: [Will be filled after baseline experiments]

#### Proposed Fix
[Specific code changes needed]

#### Related Discrepancies
[Links to related DISC-XXX]
```

**Severity Assignment Guidelines** (from [research.md](research.md#research-question-3-discrepancy-severity-criteria)):

- **Critical**: Prevents convergence, violates mathematical properties
- **High**: >5% performance degradation, blocks ±2% reproduction target
- **Medium**: <5% performance impact, suboptimal but functional
- **Low**: No measurable impact, cosmetic/documentation only

---

## Phase 2: Baseline Experiments

### 2.1: Run Baseline Experiments

Run full experiments with current (unfixed) implementation:

```bash
# Run baseline experiments (15 runs: 3 methods × 5 seeds)
# This will take ~2 hours depending on hardware
./scripts/run-baseline-experiments.sh

# For quick testing (10 epochs, 2 seeds):
./scripts/run-baseline-experiments.sh --epochs 10 --seeds 42,43
```

**Expected Output**:
```
Running baseline experiments...
Git commit: 2dfacf4a...
Output: experiments/results/baseline/

[1/15] PN_seed42... ✓ (234.5s, test error: 0.045)
[2/15] PN_seed43... ✓ (232.1s, test error: 0.047)
...
[15/15] nnPU_seed46... ✓ (289.3s, test error: 0.091)

Summary:
┌────────┬────────────────────┬───────────┬───────────┐
│ Method │ Test Error (mean)  │ Std Dev   │ Converged │
├────────┼────────────────────┼───────────┼───────────┤
│ PN     │ 0.045 ± 0.002      │ 0.002     │ 5/5       │
│ uPU    │ 0.067 ± 0.004      │ 0.004     │ 5/5       │
│ nnPU   │ 0.089 ± 0.003      │ 0.003     │ 3/5       │
└────────┴────────────────────┴───────────┴───────────┘
```

**Verify Results**:
```bash
# Check result files exist
ls -la experiments/results/baseline/*.json

# Should see 15 files:
# PN_seed42.json, PN_seed43.json, ..., nnPU_seed46.json
```

### 2.2: Analyze Baseline Results

Use the existing analysis notebook to visualize baseline performance:

```bash
# Open Jupyter notebook
uv run jupyter notebook notebooks/02_results_analysis.ipynb
```

In the notebook, update the results directory path to point to baseline:
```python
results_dir = Path("../experiments/results/baseline")
```

Run all cells to generate:
- Training loss curves
- Test loss curves
- Test error (zero-one loss) curves
- Final performance comparison

**Key Questions to Answer**:
1. Do all methods converge within 100 epochs?
2. Does nnPU achieve better or equal performance to uPU?
3. Does nnPU loss stay non-negative throughout training?
4. How far are current results from paper benchmarks?

### 2.3: Update Discrepancy Report with Measured Impact

For each discrepancy in `analysis/nnpu-discrepancy-report.md`:

1. Review baseline experiment results
2. Estimate measured impact of the discrepancy
3. Fill in **Measured Impact** section:

```markdown
#### Impact
**Expected Impact**: Missing gradient detachment causes incorrect gradient flow
**Measured Impact**:
- **Metric**: test_zero_one_loss
- **Baseline Value**: 0.089 ± 0.003
- **Estimated Degradation**: ~2-3% (nnPU should perform at ~0.05-0.06 per paper)
```

4. Reassess severity if needed (e.g., if "Medium" discrepancy shows >5% impact, escalate to "High")

---

## Phase 3: Fix Implementation

### 3.1: Prioritize Fixes by Severity

Review the discrepancy report and create fix plan:

1. **Critical Fixes** (must fix, blocking):
   - List all DISC-XXX with Severity=Critical
   - These prevent convergence or violate mathematical properties

2. **High Priority Fixes** (must fix, blocking):
   - List all DISC-XXX with Severity=High
   - These cause >5% degradation or block ±2% reproduction target

3. **Medium Priority Fixes** (time-permitting):
   - List all DISC-XXX with Severity=Medium
   - Fix if time allows after Critical/High

4. **Low Priority** (document only):
   - List all DISC-XXX with Severity=Low
   - No implementation needed, just document for completeness

### 3.2: Implement Fixes Using TDD

For each Critical/High severity discrepancy, follow the **RED-GREEN-REFACTOR** cycle:

#### Step 1: RED - Write Failing Test

Example for DISC-003 (missing gradient detachment):

```python
# File: tests/unit/test_losses.py

def test_nnpu_gradient_detachment():
    """Test that nnPU loss detaches gradients in non-negative correction."""
    import torch
    from pu_learning.losses.nnpu import nnPULoss

    # Create dummy tensors with requires_grad=True
    positive_output = torch.tensor([0.8, 0.9], requires_grad=True)
    unlabeled_output = torch.tensor([0.3, 0.7, 0.5, 0.6], requires_grad=True)
    positive_labels = torch.ones(2)
    unlabeled_labels = torch.zeros(4)

    # Create loss
    loss_fn = nnPULoss(prior=0.5, beta=0.0)

    # Compute loss
    loss = loss_fn(
        positive=(positive_output, positive_labels),
        unlabeled=(unlabeled_output, unlabeled_labels)
    )

    # Loss should be a scalar that can backpropagate
    assert loss.requires_grad

    # Backward pass should work without error
    loss.backward()

    # Gradients should exist for both inputs
    assert positive_output.grad is not None
    assert unlabeled_output.grad is not None
```

Run test (should FAIL):
```bash
uv run pytest tests/unit/test_losses.py::test_nnpu_gradient_detachment -v
```

#### Step 2: GREEN - Implement Minimal Fix

```python
# File: src/pu_learning/losses/nnpu.py

class nnPULoss(nn.Module):
    def forward(self, positive, unlabeled):
        # ... existing code ...

        # Compute uPU risk
        risk = positive_risk - self.prior * unlabeled_risk

        # Non-negative correction with gradient detachment
        if risk.detach().item() < 0:  # ← FIX: Add .detach()
            risk = -self.beta * risk.detach()  # ← FIX: Add .detach()

        return risk
```

Run test (should PASS):
```bash
uv run pytest tests/unit/test_losses.py::test_nnpu_gradient_detachment -v
```

#### Step 3: REFACTOR - Clean Up While Keeping Tests Green

```python
# Improved implementation with clearer logic
class nnPULoss(nn.Module):
    def forward(self, positive, unlabeled):
        # ... existing code ...

        # Compute uPU risk
        risk = positive_risk - self.prior * unlabeled_risk

        # Apply non-negative correction (Eq. 6 in paper)
        # Gradient should not flow through the comparison
        if risk.detach().item() < 0:
            # Use negative risk estimator with detached risk
            risk = -self.beta * risk.detach()

        return risk
```

Verify refactoring didn't break anything:
```bash
uv run pytest tests/unit/test_losses.py -v
```

#### Step 4: Update Discrepancy Report

```markdown
### DISC-003: Missing gradient detachment in nnPU loss

**Status**: Validated ✓

[... existing sections ...]

#### Fix Implementation

**Commit**: abc123def456...
**PR**: #42
**Tests Added**: `tests/unit/test_losses.py::test_nnpu_gradient_detachment`
**Coverage**: Increased from 82% to 84%
```

#### Step 5: Commit

```bash
# Stage changes
git add src/pu_learning/losses/nnpu.py tests/unit/test_losses.py

# Commit with descriptive message referencing DISC ID
git commit -m "$(cat <<'EOF'
Fix DISC-003: Add gradient detachment to nnPU loss correction

The non-negative risk correction in nnPU loss was missing gradient
detachment before the comparison, causing incorrect gradient flow.
This fix adds .detach() to match paper specification (Eq. 6) and
reference implementations (kiryor, cimeister).

Changes:
- Add .detach() before risk comparison
- Add .detach() in negative risk correction
- Add unit test for gradient flow

Severity: Critical
Test Coverage: +2% (82% → 84%)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"

# Push to remote
git push origin 001-nnpu-reproduction-analysis
```

### 3.3: Verify All Tests Pass

After implementing all Critical/High fixes:

```bash
# Run full test suite
uv run pytest --cov=src/pu_learning --cov-report=term-missing

# Verify coverage ≥80%
uv run pytest --cov=src/pu_learning --cov-fail-under=80

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

**Expected Output**:
```
========================= test session starts =========================
collected 87 items

tests/unit/test_losses.py::test_pn_loss_basic ✓
tests/unit/test_losses.py::test_upu_loss_basic ✓
tests/unit/test_losses.py::test_nnpu_gradient_detachment ✓
[... more tests ...]

---------- coverage: platform darwin, python 3.12.1 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/pu_learning/__init__.py                 5      0   100%
src/pu_learning/losses/base.py             23      2    91%   45-46
src/pu_learning/losses/nnpu.py             42      3    93%   78, 92
[... more files ...]
---------------------------------------------------------------------
TOTAL                                     512     42    92%

========================= 87 passed in 12.34s =========================
```

---

## Phase 4: Validation

### 4.1: Run Post-Fix Experiments

After all Critical/High fixes are implemented and tested:

```bash
# Run post-fix validation experiments (same seeds as baseline)
./scripts/run-fixed-experiments.sh --seeds 42,43,44,45,46

# For quick testing:
./scripts/run-fixed-experiments.sh --epochs 10 --seeds 42,43
```

**Expected Output**:
```
Running post-fix experiments...
Git commit: xyz789abc...
Output: experiments/results/fixed/

[1/15] PN_seed42... ✓ (228.3s, test error: 0.044)
[2/15] PN_seed43... ✓ (230.7s, test error: 0.046)
...
[15/15] nnPU_seed46... ✓ (275.1s, test error: 0.055)

Summary:
┌────────┬────────────────────┬───────────┬───────────┐
│ Method │ Test Error (mean)  │ Std Dev   │ Converged │
├────────┼────────────────────┼───────────┼───────────┤
│ PN     │ 0.044 ± 0.002      │ 0.002     │ 5/5       │
│ uPU    │ 0.055 ± 0.003      │ 0.003     │ 5/5       │
│ nnPU   │ 0.053 ± 0.002      │ 0.002     │ 5/5       │
└────────┴────────────────────┴───────────┴───────────┘

✅ All methods converged successfully
```

### 4.2: Compare Baseline vs Post-Fix

Generate quantified comparison:

```bash
# Generate comparison report (JSON + Markdown)
python scripts/compare-results.py --format both
```

**Expected Output**:
```
Comparing baseline vs post-fix results...

┌────────┬─────────────────┬──────────────────┬──────────┬───────────┬─────────────┐
│ Method │ Baseline Error  │ Post-Fix Error   │ Delta    │ % Change  │ Significant │
├────────┼─────────────────┼──────────────────┼──────────┼───────────┼─────────────┤
│ PN     │ 0.045 ± 0.002   │ 0.044 ± 0.002    │ -0.001   │ -2.2%     │ No (p=0.23) │
│ uPU    │ 0.067 ± 0.004   │ 0.055 ± 0.003    │ -0.012   │ -17.9%    │ Yes (p=0.01)│
│ nnPU   │ 0.089 ± 0.003   │ 0.053 ± 0.002    │ -0.036   │ -40.4%    │ Yes (p<0.001│
└────────┴─────────────────┴──────────────────┴──────────┴───────────┴─────────────┘

✅ nnPU shows statistically significant improvement
✅ nnPU now achieves equal performance to uPU (as expected from paper)
```

Review generated reports:
- `analysis/comparison-report.json`: Structured data
- `analysis/comparison-report.md`: Human-readable summary

### 4.3: Verify Paper Reproduction

Compare post-fix results to paper benchmarks (typically from paper Table 1):

```bash
# View comparison with paper benchmarks
cat analysis/comparison-report.md
```

**Look for**:
```markdown
## Paper Benchmarks

| Method | Paper Value | Our Post-Fix | Delta | Within ±2%? |
|--------|-------------|--------------|-------|-------------|
| PN     | 0.045       | 0.044        | -0.001| ✓ Yes       |
| uPU    | 0.052       | 0.055        | +0.003| ✓ Yes       |
| nnPU   | 0.051       | 0.053        | +0.002| ✓ Yes       |

✅ All methods reproduce paper results within acceptable tolerance
```

Success criteria from [spec.md](spec.md):
- ✅ SC-002: Within ±2% of paper values
- ✅ SC-003: nnPU ≥ uPU performance
- ✅ SC-004: Expected loss curve behavior
- ✅ SC-008: ≥80% test coverage maintained
- ✅ SC-009: Quantified improvement over baseline

### 4.4: Document Improvements

Update the discrepancy report with validation results:

1. Open `analysis/nnpu-discrepancy-report.md`
2. Fill in "Validation Results" section (Section 8)
3. Update "Fix Plan and Progress" (Section 7) - mark all as "Validated"
4. Update "Executive Summary" (Section 2) with final metrics

**Example Update**:
```markdown
## Executive Summary

**Total Discrepancies Found**: 12
**Breakdown by Severity**:
- Critical: 3 - ✅ All fixed and validated
- High: 5 - ✅ All fixed and validated
- Medium: 3 - ✅ 2 fixed, 1 documented
- Low: 1 - Documented only

**Reproduction Status**:
- Baseline Test Error (nnPU): 0.089 ± 0.003
- Post-Fix Test Error (nnPU): 0.053 ± 0.002
- Paper Benchmark (nnPU): 0.051
- **Improvement**: -40.4% from baseline
- **Within Target**: ✅ Yes (±2% tolerance)

**Key Improvements**:
- nnPU now converges reliably within 100 epochs (was 3/5, now 5/5 seeds)
- nnPU achieves equal performance to uPU (paper expectation met)
- Loss stays non-negative throughout training (mathematical property restored)
```

---

## Troubleshooting

### Experiments Fail to Run

**Error**: `CUDA out of memory` or `MPS allocation failed`

**Solution**:
```bash
# Reduce batch size in config
# Edit experiments/configs/mnist_config.py
batch_size = 128  # Instead of 256

# Or force CPU (slower but more stable)
python experiments/scripts/run_all.py --device cpu
```

### Reference Repo Clone Fails

**Error**: `fatal: could not read from remote repository`

**Solution**:
```bash
# Clone manually
cd analysis/reference-implementations
git clone https://github.com/kiryor/nnPUlearning.git kiryor-nnPUlearning
git clone https://github.com/cimeister/pu-learning.git cimeister-pu-learning
```

### Tests Fail After Fix

**Error**: Some existing tests fail after implementing fix

**Solution**:
1. Review test failure - is the test checking for old (incorrect) behavior?
2. If test was validating the bug, update test to expect correct behavior
3. If test failure is legitimate, fix introduced regression - revert and debug

### Paper TeX Source Not Found

**Error**: arXiv source unavailable

**Solution**:
```bash
# Download PDF instead and manually extract equations
wget https://arxiv.org/pdf/1703.00593.pdf -O analysis/nnPU_paper.pdf

# Read paper directly from PDF for equation extraction
```

---

## Next Steps

After completing this analysis and validation:

1. **Merge to Main**:
   ```bash
   # Create pull request
   gh pr create --title "Fix nnPU paper reproduction discrepancies" \
                --body "Resolves reproduction accuracy issues by fixing mathematical formulations and training procedures. See analysis/nnpu-discrepancy-report.md for details."
   ```

2. **Update Documentation**:
   - Add reproduction results to main README
   - Document known limitations (if any Medium/Low discrepancies remain unfixed)

3. **Future Work**:
   - Test reproduction on other datasets (CIFAR, FashionMNIST)
   - Experiment with different network architectures
   - Try different values of n_positive

---

## Summary

This quickstart guide covered:

✅ **Phase 1**: Setup and systematic discrepancy analysis
✅ **Phase 2**: Baseline experiments to establish current performance
✅ **Phase 3**: TDD-based fix implementation for Critical/High discrepancies
✅ **Phase 4**: Post-fix validation and paper reproduction verification

**Key Outputs**:
- `analysis/nnpu-discrepancy-report.md`: Comprehensive analysis report
- `experiments/results/baseline/`: Baseline experiment results (15 files)
- `experiments/results/fixed/`: Post-fix experiment results (15 files)
- `analysis/comparison-report.{json,md}`: Before/after comparison
- Fixed implementation with ≥80% test coverage

**Success Indicators**:
- ✅ All Critical/High discrepancies fixed and validated
- ✅ Test coverage maintained ≥80%
- ✅ Paper reproduction within ±2% tolerance
- ✅ nnPU converges and achieves expected performance
- ✅ Quantified improvement documented

For questions or issues, refer to:
- [spec.md](spec.md): Feature requirements
- [research.md](research.md): Research decisions
- [data-model.md](data-model.md): Entity schemas
- [contracts/](contracts/): Interface contracts
