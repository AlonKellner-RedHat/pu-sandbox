# Tasks: nnPU Paper Reproduction Analysis and Fixes

**Input**: Design documents from `/specs/001-nnpu-reproduction-analysis/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: This feature follows TDD methodology (constitution Principle VI). Tests MUST be written first for all Critical/High severity fixes (RED-GREEN-REFACTOR cycle).

**Organization**: Tasks are grouped by user story to enable independent analysis, fixing, and validation. Each story delivers value: US1-3 identify discrepancies, US4 validates fixes.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Analysis artifacts**: `analysis/` directory
- **Source code**: `src/pu_learning/` (existing structure)
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/analysis/`
- **Scripts**: `scripts/` directory
- **Experiment results**: `experiments/results/baseline/`, `experiments/results/fixed/`

---

## Phase 1: Setup (Analysis Infrastructure)

**Purpose**: Initialize analysis structure and fetch reference materials

- [X] T001 Create analysis directory structure per plan.md (analysis/, analysis/paper-source/, analysis/reference-implementations/, analysis/baseline-results/)
- [X] T002 Create scripts directory for analysis workflow (scripts/)
- [X] T003 [P] Implement fetch-paper-source.sh script to download arXiv TeX source per contracts/analysis-scripts.md
- [X] T004 [P] Implement clone-references.sh script to clone reference implementations per contracts/analysis-scripts.md
- [X] T005 Execute fetch-paper-source.sh to download paper 1703.00593 TeX source to analysis/paper-source/
- [X] T006 Execute clone-references.sh to clone kiryor/nnPUlearning and cimeister/pu-learning to analysis/reference-implementations/
- [X] T007 Create analysis/nnpu-discrepancy-report.md skeleton following contracts/discrepancy-report-schema.md

**Checkpoint**: ✅ Analysis infrastructure ready - reference materials available for comparison

---

## Phase 2: Foundational (Baseline Experiments - BLOCKING)

**Purpose**: Capture baseline metrics with current (unfixed) implementation before any analysis or fixes

**⚠️ CRITICAL**: Baseline experiments MUST complete before any discrepancy analysis begins. These metrics are essential for measuring improvement and assessing discrepancy severity.

- [ ] T008 Implement run-baseline-experiments.sh script per contracts/analysis-scripts.md to execute all baseline experiments
- [ ] T009 Update experiments/scripts/run_all.py to output results in JSON format conforming to contracts/experiment-results-schema.md (if not already compliant)
- [ ] T010 Execute run-baseline-experiments.sh to run baseline experiments (3 methods × 5 seeds × 100 epochs = 15 experiment files) and save to experiments/results/baseline/
- [ ] T011 Verify all 15 baseline result files exist and conform to contracts/experiment-results-schema.md JSON schema
- [ ] T012 Analyze baseline results using notebooks/02_results_analysis.ipynb to identify performance issues (convergence, nnPU vs uPU comparison, loss behavior)
- [ ] T013 Document baseline performance summary in analysis/nnpu-discrepancy-report.md Section 8 (Validation Results - Baseline Experiments)

**Checkpoint**: Baseline metrics captured - analysis can now proceed with measured impact data

---

## Phase 3: User Story 1 - Identify Mathematical Formulation Discrepancies (Priority: P1) 🎯 MVP

**Goal**: Systematically compare loss function implementations (PN, uPU, nnPU) against paper equations to identify mathematical errors

**Independent Test**: Can be fully tested by comparing extracted equations from paper TeX source against current implementation code, verifying each mathematical operation matches

### Analysis for User Story 1

- [ ] T014 [P] [US1] Extract PN loss equation (Eq. 3) from paper TeX source in analysis/paper-source/1703.00593/ and decompose into terms
- [ ] T015 [P] [US1] Extract uPU loss equation (Eq. 5) from paper TeX source and decompose into terms
- [ ] T016 [P] [US1] Extract nnPU loss equation (Eq. 6) from paper TeX source and decompose into terms including non-negative correction
- [ ] T017 [P] [US1] Review kiryor PN/uPU/nnPU loss implementations in analysis/reference-implementations/kiryor-nnPUlearning/ for consensus patterns
- [ ] T018 [P] [US1] Review cimeister PN/uPU/nnPU loss implementations in analysis/reference-implementations/cimeister-pu-learning/ for consensus patterns
- [ ] T019 [US1] Compare current PN loss implementation in src/pu_learning/losses/pn.py term-by-term against paper Eq. 3 and reference implementations
- [ ] T020 [US1] Compare current uPU loss implementation in src/pu_learning/losses/upu.py term-by-term against paper Eq. 5 and reference implementations
- [ ] T021 [US1] Compare current nnPU loss implementation in src/pu_learning/losses/nnpu.py term-by-term against paper Eq. 6 and reference implementations
- [ ] T022 [US1] Document all mathematical discrepancies found in analysis/nnpu-discrepancy-report.md Section 4 (Mathematical Discrepancies) with DISC-XXX IDs
- [ ] T023 [US1] Assign severity ratings to each mathematical discrepancy using rubric from research.md (Critical: prevents convergence/violates math properties, High: >5% degradation)
- [ ] T024 [US1] Update baseline experiment analysis to estimate measured impact for each mathematical discrepancy and fill "Measured Impact" sections in report

**Checkpoint**: All mathematical formulation discrepancies identified, documented with severity ratings, and cross-referenced with baseline metrics

---

## Phase 4: User Story 2 - Identify Training Procedure Discrepancies (Priority: P2)

**Goal**: Ensure training algorithm, optimizer settings, and evaluation procedure match paper specifications

**Independent Test**: Can be tested independently by comparing training algorithm pseudocode from paper against current trainer implementation, verifying optimizer type/settings and batch construction

### Analysis for User Story 2

- [ ] T025 [P] [US2] Extract optimizer specification from paper Section 4 (Experiments) - type, learning rate, momentum, weight decay
- [ ] T026 [P] [US2] Extract learning rate schedule from paper (initial LR, decay schedule if any)
- [ ] T027 [P] [US2] Extract batch construction approach from paper (how positive and unlabeled samples are combined)
- [ ] T028 [P] [US2] Review kiryor training procedure in analysis/reference-implementations/kiryor-nnPUlearning/ for optimizer and batch handling
- [ ] T029 [P] [US2] Review cimeister training procedure in analysis/reference-implementations/cimeister-pu-learning/ for optimizer and batch handling
- [ ] T030 [US2] Compare current Trainer optimizer configuration in src/pu_learning/training/trainer.py against paper specification
- [ ] T031 [US2] Compare current learning rate and schedule in experiments/configs/ against paper specification
- [ ] T032 [US2] Compare current batch construction in src/pu_learning/data/datasets.py against paper approach (verify positive/unlabeled combination)
- [ ] T033 [US2] Compare gradient computation and parameter update logic in src/pu_learning/training/trainer.py against reference implementations
- [ ] T034 [US2] Document all procedural discrepancies found in analysis/nnpu-discrepancy-report.md Section 5 (Procedural Discrepancies) with DISC-XXX IDs
- [ ] T035 [US2] Assign severity ratings to each procedural discrepancy using rubric from research.md
- [ ] T036 [US2] Estimate measured impact for each procedural discrepancy based on baseline experiment results

**Checkpoint**: All training procedure discrepancies identified, documented with severity ratings, and impact estimated

---

## Phase 5: User Story 3 - Identify Hyperparameter Discrepancies (Priority: P3)

**Goal**: Verify all hyperparameters (epochs, batch size, β, prior π) match paper's MNIST even/odd experimental setup

**Independent Test**: Can be tested by extracting hyperparameters from paper Table 1 and experimental setup, comparing against config files and dataset initialization

### Analysis for User Story 3

- [ ] T037 [P] [US3] Extract number of epochs from paper MNIST experimental setup
- [ ] T038 [P] [US3] Extract batch size from paper experimental setup
- [ ] T039 [P] [US3] Extract β parameter value and update schedule for nnPU from paper
- [ ] T040 [P] [US3] Extract class prior π calculation method from paper
- [ ] T041 [P] [US3] Extract n_positive (number of labeled positive samples) from paper MNIST experiments
- [ ] T042 [P] [US3] Extract dataset split specification (even vs odd classes) from paper
- [ ] T043 [US3] Compare current epoch configuration in experiments/configs/ against paper specification
- [ ] T044 [US3] Compare current batch size in experiments/configs/ against paper specification
- [ ] T045 [US3] Compare current β parameter configuration for nnPU method against paper specification
- [ ] T046 [US3] Compare current prior π calculation in src/pu_learning/data/datasets.py against paper approach
- [ ] T047 [US3] Compare current n_positive configuration against paper experiments (verify 100 labeled positive samples for MNIST)
- [ ] T048 [US3] Verify dataset split (even vs odd) matches paper setup in src/pu_learning/data/datasets.py
- [ ] T049 [US3] Document all hyperparameter discrepancies found in analysis/nnpu-discrepancy-report.md Section 6 (Hyperparameter Discrepancies) with DISC-XXX IDs
- [ ] T050 [US3] Assign severity ratings to each hyperparameter discrepancy using rubric from research.md
- [ ] T051 [US3] Estimate measured impact for each hyperparameter discrepancy based on baseline experiment results
- [ ] T052 [US3] Complete analysis/nnpu-discrepancy-report.md Sections 2-3 (Executive Summary and Severity Matrix) with total discrepancy counts and breakdown

**Checkpoint**: All hyperparameter discrepancies identified, complete discrepancy report with all findings categorized and prioritized

---

## Phase 6: User Story 4 - Implement and Validate Fixes (Priority: P4)

**Goal**: Apply fixes for identified discrepancies, re-run experiments, and verify results match paper-reported metrics within tolerance

**Independent Test**: Can be tested by applying fixes, running experiments, comparing results against paper Table 1/Figure 2, and verifying improvement from baseline

### Fix Planning for User Story 4

- [ ] T053 [US4] Review complete discrepancy report and create prioritized fix plan in Section 7 (Fix Plan and Progress) organized by severity (Critical → High → Medium → Low)
- [ ] T054 [US4] Identify conflicts between proposed fixes and document resolution strategy in Section 7

### TDD Cycle for Critical/High Fixes (User Story 4)

**Pattern**: For EACH Critical/High discrepancy, follow RED-GREEN-REFACTOR cycle:
1. Write failing unit test (RED)
2. Implement minimal fix (GREEN)
3. Refactor while keeping tests green (REFACTOR)
4. Commit when all tests pass

Example tasks below - actual tasks depend on discrepancies found. Adjust count and details based on actual analysis results.

#### Example: Mathematical Discrepancy Fixes (adapt based on findings)

- [ ] T055 [P] [US4] RED: Write failing unit test for DISC-001 (example: gradient detachment in nnPU) in tests/unit/test_losses.py
- [ ] T056 [US4] GREEN: Implement fix for DISC-001 in src/pu_learning/losses/nnpu.py to pass test
- [ ] T057 [US4] REFACTOR: Clean up DISC-001 fix while keeping all tests passing
- [ ] T058 [US4] Commit DISC-001 fix with message referencing discrepancy ID and severity
- [ ] T059 [P] [US4] RED: Write failing unit test for DISC-002 (example: sign error in uPU) in tests/unit/test_losses.py
- [ ] T060 [US4] GREEN: Implement fix for DISC-002 in src/pu_learning/losses/upu.py to pass test
- [ ] T061 [US4] REFACTOR: Clean up DISC-002 fix while keeping all tests passing
- [ ] T062 [US4] Commit DISC-002 fix with message referencing discrepancy ID
- [ ] T063 [P] [US4] RED: Write failing unit test for DISC-003 (example: missing term in PN) in tests/unit/test_losses.py
- [ ] T064 [US4] GREEN: Implement fix for DISC-003 in src/pu_learning/losses/pn.py to pass test
- [ ] T065 [US4] REFACTOR: Clean up DISC-003 fix while keeping all tests passing
- [ ] T066 [US4] Commit DISC-003 fix with message referencing discrepancy ID

#### Example: Procedural Discrepancy Fixes (adapt based on findings)

- [ ] T067 [P] [US4] RED: Write failing integration test for DISC-004 (example: wrong optimizer) in tests/integration/test_trainer.py
- [ ] T068 [US4] GREEN: Implement fix for DISC-004 in src/pu_learning/training/trainer.py to pass test
- [ ] T069 [US4] REFACTOR: Clean up DISC-004 fix while keeping all tests passing
- [ ] T070 [US4] Commit DISC-004 fix with message referencing discrepancy ID
- [ ] T071 [P] [US4] RED: Write failing test for DISC-005 (example: incorrect batch construction) in tests/unit/test_datasets.py
- [ ] T072 [US4] GREEN: Implement fix for DISC-005 in src/pu_learning/data/datasets.py to pass test
- [ ] T073 [US4] REFACTOR: Clean up DISC-005 fix while keeping all tests passing
- [ ] T074 [US4] Commit DISC-005 fix with message referencing discrepancy ID

#### Example: Hyperparameter Discrepancy Fixes (adapt based on findings)

- [ ] T075 [P] [US4] Fix DISC-006 (example: wrong batch size) in experiments/configs/ - no test needed for config changes
- [ ] T076 [P] [US4] Fix DISC-007 (example: wrong n_positive) in experiments/configs/
- [ ] T077 [P] [US4] Fix DISC-008 (example: wrong β schedule) in experiments/configs/
- [ ] T078 [US4] Commit hyperparameter fixes with message listing all DISC IDs addressed

### Validation and Comparison (User Story 4)

- [ ] T079 [US4] Verify all existing tests still pass after all Critical/High fixes: run pytest with --cov and confirm ≥80% coverage maintained
- [ ] T080 [US4] Implement run-fixed-experiments.sh script per contracts/analysis-scripts.md to execute post-fix validation experiments
- [ ] T081 [US4] Execute run-fixed-experiments.sh to run post-fix experiments (same 15 runs: 3 methods × 5 seeds × 100 epochs) and save to experiments/results/fixed/
- [ ] T082 [US4] Verify all 15 post-fix result files exist and conform to contracts/experiment-results-schema.md JSON schema
- [ ] T083 [US4] Implement compare-results.py script per contracts/analysis-scripts.md to generate before/after comparison
- [ ] T084 [US4] Execute compare-results.py to generate analysis/comparison-report.json and analysis/comparison-report.md comparing baseline vs post-fix results
- [ ] T085 [US4] Verify post-fix training curves using notebooks/02_results_analysis.ipynb show expected behavior (nnPU converges, stays non-negative, nnPU ≥ uPU performance)
- [ ] T086 [US4] Compare post-fix final test error rates against paper benchmarks from Table 1 and verify within ±2% tolerance for all methods
- [ ] T087 [US4] Perform statistical significance testing (paired t-test) to confirm improvements are statistically significant (p < 0.05)
- [ ] T088 [US4] Update analysis/nnpu-discrepancy-report.md Section 8 (Validation Results - Post-Fix Experiments) with post-fix metrics and comparison to baseline
- [ ] T089 [US4] Update analysis/nnpu-discrepancy-report.md Section 7 (Fix Plan and Progress) to mark all Critical/High fixes as "Validated" status
- [ ] T090 [US4] Update analysis/nnpu-discrepancy-report.md Section 2 (Executive Summary) with final reproduction status showing quantified improvement

**Checkpoint**: All Critical/High fixes implemented, validated, and verified to reproduce paper results within ±2% tolerance

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and ensuring full compliance with success criteria

- [ ] T091 [P] Review all Medium severity discrepancies and implement fixes if time permits (following same TDD cycle)
- [ ] T092 [P] Document all Low severity discrepancies in report with "WontFix" rationale explaining no measurable impact
- [ ] T093 [P] Add regression tests to tests/unit/ for all fixed Critical/High discrepancies to prevent re-introduction
- [ ] T094 [P] Update main README.md with reproduction results summary and link to analysis/nnpu-discrepancy-report.md
- [ ] T095 [P] Run ruff linter and formatter on all modified Python files (src/, tests/, scripts/)
- [ ] T096 [P] Verify all scripts (fetch-paper-source.sh, clone-references.sh, run-baseline-experiments.sh, run-fixed-experiments.sh) have executable permissions and --help text
- [ ] T097 Run full test suite one final time and confirm ≥80% coverage: pytest --cov=src/pu_learning --cov-fail-under=80
- [ ] T098 Validate quickstart.md workflow by following all steps from setup through validation to ensure reproducibility
- [ ] T099 Export interactive visualization plots from notebooks/02_results_analysis.ipynb to experiments/results/plots/ for sharing
- [ ] T100 Review all commits follow constitution git workflow (no --no-verify, all have Co-Authored-By line)

**Checkpoint**: Feature complete, all success criteria met, ready for PR

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - **BLOCKS all user stories**
  - ⚠️ Baseline experiments MUST complete before any analysis can begin
  - Baseline metrics are required to assess discrepancy severity and measure improvement
- **User Stories 1-3 (Phases 3-5)**: All depend on Foundational phase completion
  - US1 (Mathematical): Can start after baseline experiments complete
  - US2 (Procedural): Can start after baseline experiments complete - **INDEPENDENT from US1**
  - US3 (Hyperparameter): Can start after baseline experiments complete - **INDEPENDENT from US1, US2**
  - All three analysis stories can proceed **in parallel** if staffed
- **User Story 4 (Phase 6)**: Depends on US1, US2, US3 completion
  - Requires complete discrepancy report from all three analysis stories
  - Implements fixes for all identified discrepancies
  - Validates improvements
- **Polish (Phase 7)**: Depends on US4 completion (all Critical/High fixes validated)

### User Story Dependencies

- **User Story 1 (P1 - Mathematical)**: Depends only on Foundational (Phase 2) - **INDEPENDENT**
- **User Story 2 (P2 - Procedural)**: Depends only on Foundational (Phase 2) - **INDEPENDENT**
- **User Story 3 (P3 - Hyperparameter)**: Depends only on Foundational (Phase 2) - **INDEPENDENT**
- **User Story 4 (P4 - Fixes)**: Depends on US1, US2, US3 completion - **INTEGRATES all findings**

### Within Each User Story

**Analysis Stories (US1-3)**:
- Paper extraction tasks [P] can run in parallel
- Reference implementation review tasks [P] can run in parallel
- Current implementation comparison tasks run sequentially (after extraction/review)
- Documentation tasks run after all comparisons complete

**Fix Story (US4)**:
- Fix planning depends on complete discrepancy report
- TDD cycles for different discrepancies marked [P] can run in parallel (different files)
- Within each TDD cycle: RED → GREEN → REFACTOR → COMMIT (sequential)
- Post-fix experiments depend on all Critical/High fixes being implemented
- Comparison and validation depend on post-fix experiments completing

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003 (fetch-paper-source.sh) and T004 (clone-references.sh) can run in parallel

**Phase 2 (Foundational)**:
- Baseline experiments for different methods/seeds run sequentially (share GPU resources)
- Analysis of results can happen while later experiments run

**Phases 3-5 (Analysis Stories US1-3)**:
- All three user stories can be worked on in parallel by different team members
- Within US1: T014, T015, T016, T017, T018 can all run in parallel
- Within US2: T025, T026, T027, T028, T029 can all run in parallel
- Within US3: T037-T042 can all run in parallel

**Phase 6 (Fix Story US4)**:
- Test writing for different discrepancies (T055, T059, T063, T067, T071) can run in parallel
- Hyperparameter fixes (T075, T076, T077) can run in parallel
- Post-fix experiments run sequentially

**Phase 7 (Polish)**:
- T091, T092, T093, T094, T095, T096 can all run in parallel

---

## Parallel Example: User Story 1 (Mathematical Analysis)

```bash
# Launch all paper equation extraction tasks together:
Task: "Extract PN loss equation (Eq. 3) from paper TeX source"
Task: "Extract uPU loss equation (Eq. 5) from paper TeX source"
Task: "Extract nnPU loss equation (Eq. 6) from paper TeX source"
Task: "Review kiryor PN/uPU/nnPU loss implementations"
Task: "Review cimeister PN/uPU/nnPU loss implementations"

# After extraction/review, run comparisons sequentially:
# (sequential because they build on each other for report consistency)
```

## Parallel Example: User Story 4 (Fix Implementation)

```bash
# Launch all test-writing tasks for different discrepancies together:
Task: "RED: Write failing unit test for DISC-001 in tests/unit/test_losses.py"
Task: "RED: Write failing unit test for DISC-002 in tests/unit/test_losses.py"
Task: "RED: Write failing unit test for DISC-003 in tests/unit/test_losses.py"
Task: "RED: Write failing integration test for DISC-004 in tests/integration/test_trainer.py"
Task: "RED: Write failing test for DISC-005 in tests/unit/test_datasets.py"

# Launch all hyperparameter config fixes together:
Task: "Fix DISC-006 (batch size) in experiments/configs/"
Task: "Fix DISC-007 (n_positive) in experiments/configs/"
Task: "Fix DISC-008 (β schedule) in experiments/configs/"
```

---

## Implementation Strategy

### MVP First (User Stories 1-3 Analysis Only)

This is an **analysis-focused feature**, so the MVP is the complete discrepancy analysis:

1. Complete Phase 1: Setup (fetch paper & references)
2. Complete Phase 2: Foundational (baseline experiments - **CRITICAL**)
3. Complete Phases 3-5: Analysis Stories (US1 + US2 + US3 in parallel if possible)
4. **STOP and VALIDATE**: Review complete discrepancy report
5. Present findings to stakeholders before implementing fixes

**Rationale**: Understanding the problem completely before attempting fixes reduces rework risk.

### Full Implementation (Analysis + Fixes)

1. Complete Setup + Foundational → Baseline metrics captured
2. Complete US1 + US2 + US3 → Complete discrepancy analysis
3. **CHECKPOINT**: Review and prioritize fixes based on severity and measured impact
4. Complete US4 → All Critical/High fixes implemented and validated
5. **CHECKPOINT**: Verify reproduction within ±2% of paper
6. Complete Polish → Documentation and final validation

### Incremental Delivery

Each checkpoint represents a deliverable:

1. **Baseline Metrics** (Phase 2) → Understand current performance gap
2. **Discrepancy Analysis** (US1-3) → Comprehensive problem identification
3. **Critical Fixes** (US4 subset) → Address most severe issues first
4. **High Priority Fixes** (US4 subset) → Achieve reproduction target
5. **Medium Fixes** (Optional, US4 subset) → Further improvements if time permits
6. **Final Report** (Polish) → Complete documentation of improvements

### Parallel Team Strategy

With multiple team members:

1. **Team completes Setup + Foundational together** (critical for consistency)
2. **Once baseline experiments done**:
   - **Analyst A**: User Story 1 (Mathematical discrepancies)
   - **Analyst B**: User Story 2 (Procedural discrepancies)
   - **Analyst C**: User Story 3 (Hyperparameter discrepancies)
3. **Team reviews complete discrepancy report together** (ensure consistency)
4. **Fix implementation (US4)** can be parallelized:
   - **Developer A**: Mathematical fixes (loss functions)
   - **Developer B**: Procedural fixes (trainer, datasets)
   - **Developer C**: Hyperparameter fixes (configs)
5. **Team validates together**: Post-fix experiments and comparison

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story (US1-3) is independently completable - can deliver just analysis without fixes
- US4 (fixes) depends on complete analysis from US1-3
- **TDD is mandatory** for all Critical/High severity fixes (constitution requirement)
- Verify tests FAIL before implementing fixes (RED step)
- Commit after each TDD cycle completes (all tests passing)
- **Baseline experiments MUST complete** before any analysis - they provide measured impact data
- Stop at checkpoints to validate progress
- All scripts must follow contracts/ interface specifications
- All experiment results must conform to JSON schema
- ≥80% test coverage must be maintained (constitution requirement)
- Never use --no-verify when committing (constitution requirement)

---

## Task Count Summary

- **Phase 1 (Setup)**: 7 tasks
- **Phase 2 (Foundational)**: 6 tasks (BLOCKING)
- **Phase 3 (US1 - Mathematical)**: 11 tasks
- **Phase 4 (US2 - Procedural)**: 12 tasks
- **Phase 5 (US3 - Hyperparameter)**: 16 tasks
- **Phase 6 (US4 - Fixes)**: 38 tasks (includes TDD cycles for example discrepancies - adjust based on actual findings)
- **Phase 7 (Polish)**: 10 tasks

**Total**: 100 tasks

**Parallelizable**: 35 tasks marked [P] (35% can run in parallel within constraints)

**Independent Stories**: US1, US2, US3 can all run in parallel after Foundational phase

**MVP Scope**: Phases 1-5 (52 tasks) delivers complete discrepancy analysis without fixes

**Full Feature**: All 100 tasks delivers analysis + fixes + validation + documentation
