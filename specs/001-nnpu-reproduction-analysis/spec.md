# Feature Specification: nnPU Paper Reproduction Analysis and Fixes

**Feature Branch**: `001-nnpu-reproduction-analysis`
**Created**: 2026-02-15
**Status**: Draft
**Input**: User description: "Analyze and fix discrepancies between current implementation and nnPU paper. Compare against paper TeX sources (arxiv.org/src/1703.00593) and reference implementations (kiryor/nnPUlearning, cimeister/pu-learning) to identify mathematical, implementation, and hyperparameter differences affecting result reproduction."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Identify Mathematical Formulation Discrepancies (Priority: P1)

A researcher needs to verify that the loss function implementations (PN, uPU, nnPU) exactly match the mathematical formulations in the paper, as even small errors in the math will completely prevent accurate reproduction of results.

**Why this priority**: Mathematical correctness is foundational - incorrect loss functions make reproduction impossible regardless of other settings. This is the highest-impact area for discrepancies.

**Independent Test**: Can be fully tested by extracting equations from paper TeX source, comparing against current implementation code, and verifying each mathematical operation matches (including signs, coefficients, expectations, and risk estimators). Delivers immediate value by identifying formula bugs.

**Acceptance Scenarios**:

1. **Given** the paper TeX source and current loss function implementations, **When** researcher compares PN loss formula (Eq. 3 in paper), **Then** all terms, signs, and coefficients match exactly
2. **Given** the paper TeX source and current loss function implementations, **When** researcher compares uPU loss formula (Eq. 5 in paper), **Then** all terms including the negative risk component match exactly
3. **Given** the paper TeX source and current loss function implementations, **When** researcher compares nnPU loss formula (Eq. 6 in paper), **Then** the non-negative risk correction matches exactly including the β parameter and gradient detachment
4. **Given** reference implementations (kiryor, cimeister), **When** researcher compares loss function code, **Then** any differences from current implementation are documented with rationale

---

### User Story 2 - Identify Training Procedure Discrepancies (Priority: P2)

A researcher needs to ensure the training algorithm, optimizer settings, and evaluation procedure match the paper's specifications, as these significantly impact convergence and final performance.

**Why this priority**: After mathematical correctness, the training procedure is the next most critical factor. Incorrect optimizer, learning rate schedule, or batch processing can prevent convergence to paper-reported results.

**Independent Test**: Can be tested independently by comparing training algorithm pseudocode from paper against current trainer implementation, verifying optimizer type/settings, learning rate schedule, and batch construction. Delivers value by identifying procedural errors.

**Acceptance Scenarios**:

1. **Given** paper Section 4 (Experiments) and current Trainer class, **When** researcher compares optimizer choice, **Then** optimizer type matches paper specification (likely SGD or Adam with momentum)
2. **Given** paper experimental setup and current training code, **When** researcher compares learning rate and schedule, **Then** initial learning rate and any decay schedule match
3. **Given** paper and current implementation, **When** researcher compares batch construction for PU learning, **Then** labeled positive samples and unlabeled samples are combined correctly per paper's approach
4. **Given** reference implementations, **When** researcher compares training loops, **Then** any differences in gradient computation or parameter updates are documented

---

### User Story 3 - Identify Hyperparameter Discrepancies (Priority: P3)

A researcher needs to verify that all hyperparameters (epochs, batch size, beta for nnPU, prior π) match the paper's experimental setup for MNIST even/odd classification.

**Why this priority**: While less critical than math/procedure, wrong hyperparameters can still prevent matching paper results. These are easier to fix but important for exact reproduction.

**Independent Test**: Can be tested by extracting hyperparameters from paper Table 1 and experimental setup sections, comparing against config files and dataset initialization. Delivers value by ensuring experiment configuration matches paper.

**Acceptance Scenarios**:

1. **Given** paper experimental setup for MNIST, **When** researcher compares number of epochs, **Then** training epochs match paper specification
2. **Given** paper setup and current config, **When** researcher compares batch size, **Then** batch size matches paper (likely 256 or similar)
3. **Given** paper nnPU algorithm and current config, **When** researcher compares β parameter, **Then** β value and update schedule match paper
4. **Given** paper dataset setup, **When** researcher compares class prior π estimation, **Then** prior calculation method matches paper approach
5. **Given** paper and current dataset, **When** researcher compares number of labeled positive samples, **Then** n_positive matches paper experiments (typically 100-1000 for MNIST)

---

### User Story 4 - Implement and Validate Fixes (Priority: P4)

A researcher needs to apply fixes for identified discrepancies, re-run experiments, and verify that results now match paper-reported metrics within acceptable tolerance.

**Why this priority**: This is the final verification step that depends on all previous analysis. It validates that fixes actually improve reproduction accuracy.

**Independent Test**: Can be tested by applying each category of fixes (math, procedure, hyperparameters) independently, running experiments, and comparing results against paper Table 1/Figure 2. Delivers value by confirming successful reproduction.

**Acceptance Scenarios**:

1. **Given** fixes for mathematical discrepancies, **When** researcher re-runs experiments, **Then** loss curves show expected behavior (no negative values for nnPU, uPU may go negative)
2. **Given** all identified fixes applied, **When** researcher runs full MNIST experiments with multiple seeds, **Then** final test error rates are within ±2% of paper-reported values
3. **Given** fixed implementation, **When** researcher compares training curves, **Then** nnPU achieves better or equal final performance compared to uPU (as expected from paper)
4. **Given** reproduction validation results, **When** researcher documents remaining discrepancies, **Then** any differences from paper are explained with potential causes

---

### Edge Cases

- What happens when paper equations are ambiguous or have typos (compare against reference implementations for consensus)?
- How does the system handle situations where reference implementations disagree with each other or with the paper?
- What if paper uses non-standard notation that requires careful interpretation?
- How to handle hyperparameters not explicitly stated in paper (use reference implementations as guide)?
- What if fixes for one discrepancy negatively impact another aspect of reproduction?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Analysis MUST fetch and extract the nnPU paper TeX source from arxiv.org/src/1703.00593
- **FR-002**: Analysis MUST clone/fetch both reference implementations (kiryor/nnPUlearning, cimeister/pu-learning)
- **FR-003**: Analysis MUST compare current loss function implementations (PN, uPU, nnPU) against paper equations (Eq. 3, 5, 6) term-by-term
- **FR-004**: Analysis MUST compare training procedure (optimizer, learning rate, batch construction) against paper Section 4
- **FR-005**: Analysis MUST compare hyperparameters (epochs, batch size, β, n_positive) against paper experimental setup
- **FR-006**: Analysis MUST create a prioritized list of discrepancies with severity ratings (Critical/High/Medium/Low)
- **FR-007**: Analysis MUST document findings in a structured markdown report at `analysis/nnpu-discrepancy-report.md`
- **FR-008**: Fixes MUST be implemented for all Critical and High severity discrepancies
- **FR-009**: Implementation MUST preserve existing test coverage (maintain ≥80% coverage after fixes)
- **FR-010**: Validation MUST include running full experiments (100 epochs, 5 seeds) comparing results to paper benchmarks
- **FR-011**: Documentation MUST include before/after comparison of key metrics (test error, training curves)
- **FR-012**: All fixes MUST be verified by re-running existing unit tests and integration tests

### Key Entities

- **Discrepancy Report**: Structured document containing categorized list of differences between current implementation and paper/references, with severity ratings and fix priorities
- **Mathematical Formulation**: Equations from paper (PN risk, uPU risk, nnPU risk) that must be implemented exactly
- **Training Configuration**: Set of hyperparameters and procedural settings (optimizer, learning rate, epochs, batch size, β) that define experiment setup
- **Reproduction Metrics**: Quantitative measurements (test error rate, accuracy, loss curves) used to evaluate reproduction quality against paper benchmarks
- **Reference Implementation**: Existing codebases (kiryor, cimeister) that serve as comparison points and potential sources of correct implementations

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All loss function formulations (PN, uPU, nnPU) match paper equations exactly with zero mathematical discrepancies
- **SC-002**: MNIST even/odd experiment achieves test error within ±2 percentage points of paper-reported values (paper reports ~5-10% error for nnPU)
- **SC-003**: nnPU method achieves equal or better final test error compared to uPU method across all random seeds (as expected from paper theory)
- **SC-004**: Training loss curves show expected behavior: nnPU stays non-negative, uPU may become negative, convergence occurs within 100 epochs
- **SC-005**: Hyperparameter configuration exactly matches paper's MNIST setup (epochs, batch size, β schedule, n_positive count)
- **SC-006**: Analysis report documents at least 90% of discrepancies that could impact reproduction accuracy
- **SC-007**: All Critical severity discrepancies are fixed and verified through experiments
- **SC-008**: Fixed implementation maintains ≥80% test coverage with all existing tests passing

## Assumptions *(optional)*

- Paper TeX source is available and complete at arxiv.org/src/1703.00593
- Reference implementations (kiryor, cimeister) are accessible and representative of correct reproduction
- Current test suite adequately covers loss functions and training components
- MNIST dataset preparation (even/odd split) is already correct in current implementation
- Paper's reported metrics are reliable and reproducible with their specified setup
- Standard deviation across random seeds is expected (paper likely reports mean ± std)
- Where paper is ambiguous, reference implementations represent community consensus on correct interpretation

## Dependencies *(optional)*

- **External**: Access to arxiv.org for paper TeX source download
- **External**: Access to GitHub for reference implementation repositories
- **Internal**: Existing unit tests and integration tests must pass before fixes
- **Internal**: Current MNISTPUDataset implementation provides correct even/odd split
- **Internal**: Experiment infrastructure (run_all.py, config files) supports running validation experiments

## Non-Goals *(optional)*

- Reproducing results for datasets other than MNIST (focus is on fixing MNIST reproduction first)
- Implementing additional loss functions or methods beyond PN, uPU, nnPU
- Optimizing training speed or efficiency (focus is on correctness)
- Adding new features or experiments not in original paper
- Reproducing results with different neural network architectures (stick to paper's architecture)
- Creating a general-purpose PU learning library (focus is on accurate paper reproduction)
