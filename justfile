# Justfile for PU Learning Research Project
# Install just: https://github.com/casey/just

# Default recipe (show available commands)
default:
    @just --list

# Run pre-commit hooks on all staged files (alias: p)
pre-commit:
    git add -A
    uvx pre-commit run

# Alias for pre-commit
alias p := pre-commit

# Run pre-commit on all files
pre-commit-all:
    uvx pre-commit run --all-files

# Run tests with coverage
test:
    uv run pytest --cov=pu_learning --cov-report=term-missing --cov-fail-under=80 -v

# Run tests without coverage (faster)
test-fast:
    uv run pytest -v --no-cov

# Run specific test file
test-file FILE:
    uv run pytest {{FILE}} -v

# Run ruff linter
lint:
    uvx ruff check src/ tests/

# Run ruff formatter
format:
    uvx ruff format src/ tests/

# Fix linting issues automatically
fix:
    uvx ruff check --fix src/ tests/

# Install/update pre-commit hooks
install-hooks:
    uvx pre-commit install --install-hooks

# Update pre-commit hooks to latest versions
update-hooks:
    uvx pre-commit autoupdate

# Sync dependencies
sync:
    uv sync

# Clean up Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name ".coverage" -delete 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run MPS verification (check PyTorch MPS support)
verify-mps:
    uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

# Run all checks (lint, format, test)
check: lint test

# Full workflow: format, lint, test, pre-commit
all: format fix test pre-commit

# Show project statistics
stats:
    @echo "Project Statistics:"
    @echo "==================="
    @echo "Python files:"
    @find src/ tests/ -name "*.py" | wc -l
    @echo "Lines of code (src/):"
    @find src/ -name "*.py" -exec wc -l {} + | tail -1
    @echo "Lines of test code (tests/):"
    @find tests/ -name "*.py" -exec wc -l {} + | tail -1
    @echo "Git commits:"
    @git log --oneline | wc -l
