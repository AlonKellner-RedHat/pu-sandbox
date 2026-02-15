# PU Learning Research Project Constitution

## Core Principles

### I. Reproducibility First (NON-NEGOTIABLE)
All experiments must be fully reproducible:
- **Seed Everything**: Set seeds for Python random, NumPy, PyTorch (CPU/MPS/CUDA)
- **Deterministic Operations**: Use deterministic algorithms where available
- **Version Locking**: Pin all dependencies to specific versions in pyproject.toml
- **Data Integrity**: Deterministic data loading and splitting (fixed seeds)
- **Results Logging**: Every experiment run logs: seed, commit hash, timestamp, hyperparameters, and results
- **Multiple Runs**: Report mean ± std across minimum 5 random seeds
- **MPS Caveat**: Document that MPS may have minor numerical differences vs CUDA/CPU

### II. Modular Architecture
Clean separation of concerns for research flexibility:
- **Data Module**: Self-contained dataset loading, splitting, and preprocessing
- **Models Module**: Network architectures independent of loss functions
- **Losses Module**: PU loss functions as standalone, composable components
- **Training Module**: Generic trainer that works with any (model, loss, data) combination
- **Utils Module**: Shared utilities (device selection, seeding, metrics, visualization)
- **No Circular Dependencies**: Each module has clear interfaces and minimal coupling
- **Extensibility**: Adding a new loss function requires only implementing one class

### III. Test Coverage >80% (NON-NEGOTIABLE)
Comprehensive testing ensures scientific validity:
- **Unit Tests**: Every module has tests for core functionality
  - Data loading produces correct P/U split sizes and class distributions
  - Loss functions implement correct mathematical formulations
  - Models produce outputs in valid range [0,1]
  - Metrics compute accurately
- **Integration Tests**: End-to-end training loops work correctly
  - Training decreases loss
  - Checkpointing and resuming work
  - Reproducibility: same seed → identical results
- **Scientific Tests**: Critical research properties verified
  - uPU loss can become negative
  - nnPU loss stays non-negative (assertion in code)
  - PN baseline trains successfully
- **Pytest Framework**: All tests runnable with `pytest tests/ -v --cov=pu_learning`

### IV. Scientific Integrity
Research code must be scientifically sound:
- **Paper Alignment**: Implementation matches nnPU paper specifications exactly
  - MNIST: Even digits=positive, odd=negative
  - 100 positive samples, ~59,900 unlabeled samples
  - 6-layer MLP: [784, 300, 300, 300, 300, 300, 1] with ReLU + Sigmoid
  - Loss formulations: PN, uPU, nnPU as defined in paper
  - Primary metric: Zero-one loss (classification error)
- **No Cherry-Picking**: Report all results, including failed experiments
- **Statistical Rigor**: Mean ± std across multiple seeds, not single best run
- **Validation**: Results must align with nnPU paper findings before extension

### V. Interactive Visualization
Use modern interactive visualization for exploratory analysis:
- **HoloViz Stack**: hvPlot + HoloViews + Bokeh (no matplotlib)
- **Interactive Features**: Hover tooltips, pan/zoom, toggleable legends
- **Export HTML**: All plots exportable as standalone HTML files
- **Dashboard Views**: Combined visualizations for comparing methods
- **Notebook Integration**: All visualizations work seamlessly in Jupyter notebooks

### VI. Test-Driven Development (TDD) - NON-NEGOTIABLE
Follow strict TDD methodology for all implementation:

**Red-Green-Refactor Cycle**:
1. **RED**: Write a failing test first
   - Define expected behavior before implementation
   - Test must fail initially (proves test is valid)
   - Write minimal test code to express requirement
2. **GREEN**: Write minimal code to make test pass
   - Implement only what's needed to pass the test
   - No premature optimization or over-engineering
   - Focus on making the test green, nothing more
3. **REFACTOR**: Improve code quality while keeping tests green
   - Clean up implementation
   - Remove duplication
   - Improve readability
   - All tests must still pass

**TDD Requirements**:
- **Tests Before Code**: Never write production code without a failing test
- **One Test at a Time**: Focus on single functionality per test
- **Fast Feedback**: Run tests frequently (pre-commit hooks enforce this)
- **Scientific Validity**: Tests encode paper specifications
  - Loss function formulas as test assertions
  - Expected behaviors as test cases
  - Edge cases documented in tests

**TDD Benefits for Research**:
- **Correctness**: Mathematical formulations verified by tests
- **Documentation**: Tests document expected behavior
- **Regression Prevention**: Changes don't break existing functionality
- **Confidence**: Refactor fearlessly with test safety net
- **Reproducibility**: Tests ensure consistent behavior across environments

**Enforcement**:
- Pre-commit hooks run pytest with >80% coverage requirement
- Pull requests require all tests passing
- New features require new tests (no exceptions)
- Bug fixes require regression tests

## Technology Stack Requirements

### Required Dependencies
- **PyTorch >= 2.0.0**: Deep learning framework with MPS support
- **torchvision >= 0.15.0**: MNIST dataset
- **hvplot >= 0.9.0**: Interactive plotting
- **holoviews >= 1.18.0**: Declarative visualizations
- **bokeh >= 3.3.0**: Visualization backend
- **pytest >= 7.4.0**: Testing framework
- **Python >= 3.9**: Modern Python features

### Hardware Acceleration
- **MPS Priority**: Use MPS (Metal Performance Shaders) for M4 MacBook Pro
- **Fallback**: CUDA if available, then CPU
- **Device Selection**: Automatic device detection with manual override option
- **Memory Management**: Clear MPS cache between experiments

### Code Quality Tools (Astral.sh Stack) - NON-NEGOTIABLE
- **uv**: Package and environment management (fast Rust-based tooling)
- **ruff**: Code formatting and linting (all-in-one tool)
  - Line length: 100
  - Auto-fix enabled for formatting issues
  - Replaces: black, isort, flake8, pylint, pygrep-hooks
  - Enabled rulesets: E, W, F, I, N, UP, B, C4, DTZ, SIM, PTH, PGH, RUF
- **ty**: Static type checking (Astral.sh's fast type checker)
  - **MANDATORY**: Use ONLY `ty`, never mypy, pyright, or other type checkers
  - Currently disabled as ty is not yet released by Astral.sh
  - Will enable when ty becomes available
- **pytest-cov**: Code coverage reporting
- **pre-commit**: Automated code quality checks (ruff, ty, pytest, secrets detection)
- **Type hints**: All function signatures have type annotations (ready for ty)
- **Docstrings**: Google-style docstrings for all public functions/classes

**Explicitly Prohibited Tools**:
- ❌ **mypy** (use ty instead when available)
- ❌ **pyright** (use ty instead when available)
- ❌ **black** (use ruff format instead)
- ❌ **isort** (use ruff check --select I instead)
- ❌ **flake8** (use ruff check instead)
- ❌ **pylint** (use ruff check instead)
- ❌ **pygrep-hooks** (use ruff check --select PGH instead)
- ❌ **matplotlib** (use hvplot + holoviews + bokeh instead)

## Development Workflow

### Implementation Sequence
1. **Spec-Kit Workflow**: Follow constitution → specify → plan → tasks → implement
2. **Test-Driven Development (TDD)**: RED → GREEN → REFACTOR cycle for every feature
   - Write failing test first (RED)
   - Implement minimal code to pass (GREEN)
   - Refactor while keeping tests green (REFACTOR)
   - Commit only when all tests pass
3. **Incremental Development**: Build in phases (data → models → losses → training → experiments)
4. **Validation Gates**: Each phase must pass tests before moving to next phase

**Example TDD Workflow for Loss Functions**:
```python
# Step 1 (RED): Write failing test
def test_nnpu_loss_is_nonnegative():
    loss_fn = nnPULoss(prior=0.5)
    outputs = torch.randn(32, 1)
    targets = torch.randint(0, 2, (32,))
    loss = loss_fn(outputs, targets)
    assert loss >= 0, "nnPU loss must be non-negative"
    # Test FAILS - nnPULoss not implemented yet

# Step 2 (GREEN): Implement minimal code to pass
class nnPULoss(PULoss):
    def forward(self, outputs, targets):
        # Minimal implementation using max(0, ...)
        ...
    # Test PASSES

# Step 3 (REFACTOR): Improve code quality
class nnPULoss(PULoss):
    """Improved with docstrings, type hints, etc."""
    def forward(self, outputs, targets):
        # Clean, well-documented implementation
        ...
    # Tests still PASS
```

### Experiment Workflow
1. **Configuration**: Define experiment in YAML config file
2. **Single Method**: Test single method (PN, uPU, or nnPU) first
3. **Comparison**: Run all three methods with same seeds
4. **Analysis**: Analyze results in interactive notebooks
5. **Documentation**: Document findings and update README

### Git Workflow - NON-NEGOTIABLE
**Commit Early, Commit Often:**
- **Frequent Commits**: Commit after completing each logical unit of work (feature, bug fix, refactor)
- **After TDD Cycle**: Commit after completing RED-GREEN-REFACTOR cycle with passing tests
- **After Pre-Commit**: Only commit when all pre-commit hooks pass
- **Atomic Commits**: Each commit should be self-contained and buildable
- **Commit Size**: Prefer smaller, focused commits over large monolithic ones

**Commit Messages:**
- **Meaningful**: Clearly describe what changed and why
- **Format**: Imperative mood, present tense (e.g., "Add", "Fix", "Update", not "Added", "Fixed")
- **Detail**: First line: brief summary (<70 chars); body: detailed explanation if needed
- **Co-Authoring**: All commits must include co-authoring with Claude Code:
  ```
  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
  ```

**Push Frequently:**
- **After Each Commit**: Push to remote after committing to avoid losing work
- **Multiple Times Per Session**: Push at least after each major feature/phase completion
- **Before Context Switch**: Always push before taking a break or switching tasks

**Git Best Practices:**
- **Clean History**: No merge commits, rebase if needed
- **No Force Push**: Never force push to main/master branch
- **No Secrets**: Never commit .env, credentials, API keys, or sensitive data
- **Pre-Commit Hooks**: Always run and pass pre-commit hooks before committing
- **Branch Protection**: Work on feature branches, merge to main after review (when team grows)

**Hook Installation:**
- Install pre-commit hooks with: `uvx pre-commit install --install-hooks`
- Hooks will automatically fix issues where possible (ruff, trailing whitespace, etc.)
- If hooks fail, fix issues and re-commit (hooks already applied auto-fixes)

## Governance

### Constitution Supremacy
- This constitution supersedes all other development practices
- All code changes must comply with these principles
- Non-compliance requires explicit justification and documentation

### Amendment Process
- Amendments require clear rationale and impact analysis
- Major changes require updating all affected code
- Version bumps follow semantic versioning

### Quality Gates
- All tests must pass before any git commit
- Code coverage must be >80% for all new code
- Scientific validation required before publishing results
- MPS acceleration must work on M4 MacBook Pro

### Complexity Budget
- YAGNI: Implement only what's needed for nnPU reproduction initially
- No premature optimization
- Custom loss implementation comes AFTER successful baseline reproduction
- Simplicity preferred over clever code

---

**Version**: 1.2.0 | **Ratified**: 2026-02-15 | **Last Amended**: 2026-02-15

**Amendment History**:
- **v1.2.0** (2026-02-15): Expanded Git Workflow with "commit early, commit often" guidance, push frequency requirements, detailed commit message format, and hook installation instructions
- **v1.1.0** (2026-02-15): Added Test-Driven Development (TDD) as Principle VI, enforced Astral.sh stack (uv, ruff, ty only), added pre-commit hooks
- **v1.0.0** (2026-02-15): Initial constitution with core principles
