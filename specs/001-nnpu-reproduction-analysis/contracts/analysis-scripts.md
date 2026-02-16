# Analysis Scripts Interface Contract

**Feature**: [spec.md](../spec.md) | **Data Model**: [data-model.md](../data-model.md)
**Created**: 2026-02-16
**Purpose**: Define the interface and behavior of analysis workflow scripts

## Overview

This contract specifies the command-line interface, inputs, outputs, and behavior of all scripts used in the nnPU reproduction analysis workflow. All scripts must follow these specifications to ensure consistent, reproducible analysis.

---

## Script: `fetch-paper-source.sh`

### Purpose
Download and extract the nnPU paper TeX source from arXiv.

### Location
`scripts/fetch-paper-source.sh`

### Interface

```bash
./scripts/fetch-paper-source.sh [OPTIONS]
```

**Options**:
- `--output-dir <path>`: Output directory (default: `analysis/paper-source`)
- `--paper-id <id>`: arXiv paper ID (default: `1703.00593`)
- `--force`: Re-download even if already exists
- `--help`: Show usage information

### Inputs
- arXiv paper ID (default: 1703.00593 for nnPU paper)
- Internet connection to arxiv.org

### Outputs
- **Directory**: `analysis/paper-source/1703.00593/` containing:
  - TeX source files (`*.tex`)
  - Bibliography files (`*.bib`)
  - Any included figures/tables
- **Exit Code**:
  - `0`: Success
  - `1`: Download failed (network error, invalid paper ID)
  - `2`: Extraction failed (corrupted archive)

### Behavior

1. Create output directory if it doesn't exist
2. Check if source already exists (skip if exists unless `--force`)
3. Download source tarball from `https://arxiv.org/src/<paper-id>`
4. Extract tarball to output directory
5. Verify extraction (check for `.tex` files)
6. Print summary of downloaded files

### Example Usage

```bash
# Download with defaults
./scripts/fetch-paper-source.sh

# Custom output directory
./scripts/fetch-paper-source.sh --output-dir /tmp/papers

# Force re-download
./scripts/fetch-paper-source.sh --force
```

### Example Output

```
Fetching paper source for 1703.00593...
✓ Downloaded source tarball (2.3 MB)
✓ Extracted to analysis/paper-source/1703.00593/
✓ Found 1 TeX file: nnPU.tex
✓ Found 1 BibTeX file: references.bib
✓ Found 3 figure files

Paper source ready for analysis.
```

---

## Script: `clone-references.sh`

### Purpose
Clone reference implementations of nnPU learning for comparison.

### Location
`scripts/clone-references.sh`

### Interface

```bash
./scripts/clone-references.sh [OPTIONS]
```

**Options**:
- `--output-dir <path>`: Output directory (default: `analysis/reference-implementations`)
- `--repos <list>`: Comma-separated list of repos to clone (default: `kiryor,cimeister`)
- `--shallow`: Shallow clone (faster, less history)
- `--help`: Show usage information

### Inputs
- GitHub repository URLs (hardcoded defaults for kiryor and cimeister repos)
- Internet connection to github.com

### Outputs
- **Directories**:
  - `analysis/reference-implementations/kiryor-nnPUlearning/`
  - `analysis/reference-implementations/cimeister-pu-learning/`
- **Exit Code**:
  - `0`: Success
  - `1`: Clone failed (network error, invalid repo)
  - `2`: Directory already exists (unless `--force` used)

### Behavior

1. Create output directory if it doesn't exist
2. Check if repos already cloned (skip if exist unless `--force`)
3. Clone kiryor/nnPUlearning to `kiryor-nnPUlearning/`
4. Clone cimeister/pu-learning to `cimeister-pu-learning/`
5. Record commit SHAs for reference in discrepancy report
6. Print summary of cloned repos

### Example Usage

```bash
# Clone with defaults
./scripts/clone-references.sh

# Shallow clone for faster download
./scripts/clone-references.sh --shallow

# Custom output directory
./scripts/clone-references.sh --output-dir /tmp/references
```

### Example Output

```
Cloning reference implementations...
✓ Cloned kiryor/nnPUlearning (commit: a1b2c3d4...)
  - 234 commits, last updated 2019-03-15
✓ Cloned cimeister/pu-learning (commit: e5f6g7h8...)
  - 87 commits, last updated 2020-11-22

Reference implementations ready for comparison.
Record these commit SHAs in your discrepancy report.
```

---

## Script: `run-baseline-experiments.sh`

### Purpose
Run full baseline experiments (before fixes) to establish performance metrics.

### Location
`scripts/run-baseline-experiments.sh`

### Interface

```bash
./scripts/run-baseline-experiments.sh [OPTIONS]
```

**Options**:
- `--seeds <list>`: Comma-separated random seeds (default: `42,43,44,45,46`)
- `--epochs <n>`: Number of epochs (default: `100`)
- `--methods <list>`: Comma-separated methods (default: `PN,uPU,nnPU`)
- `--output-dir <path>`: Output directory (default: `experiments/results/baseline`)
- `--parallel`: Run experiments in parallel (experimental)
- `--dry-run`: Print commands without executing
- `--help`: Show usage information

### Inputs
- Current implementation code (as of current git commit)
- MNIST dataset (will be downloaded if not present)
- Configuration from existing experiment configs

### Outputs
- **Files**: 15 JSON files (3 methods × 5 seeds) in output directory:
  - `PN_seed42.json`, `PN_seed43.json`, ..., `PN_seed46.json`
  - `uPU_seed42.json`, `uPU_seed43.json`, ..., `uPU_seed46.json`
  - `nnPU_seed42.json`, `nnPU_seed43.json`, ..., `nnPU_seed46.json`
- **Log Files**: `experiments/results/baseline/logs/` containing per-experiment logs
- **Exit Code**:
  - `0`: All experiments completed successfully
  - `1`: Some experiments failed (check logs)

### Behavior

1. Validate current git state (warn if uncommitted changes)
2. Create output directory and logs subdirectory
3. For each method × seed combination:
   - Run experiment using `experiments/scripts/run_all.py`
   - Save results to JSON file conforming to [experiment-results-schema.md](experiment-results-schema.md)
   - Log stdout/stderr to log file
4. Print summary table of results
5. Generate baseline summary report

### Example Usage

```bash
# Run full baseline (15 experiments)
./scripts/run-baseline-experiments.sh

# Quick test with fewer seeds
./scripts/run-baseline-experiments.sh --seeds 42,43 --epochs 10

# Only specific methods
./scripts/run-baseline-experiments.sh --methods nnPU

# Dry run to see what would execute
./scripts/run-baseline-experiments.sh --dry-run
```

### Example Output

```
Running baseline experiments...
Git commit: 2dfacf4a8b9c3d1e5f6a7b8c9d0e1f2a3b4c5d6e
Branch: 001-nnpu-reproduction-analysis
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

Total runtime: 1h 3m 47s
Results saved to: experiments/results/baseline/

⚠ Warning: nnPU did not converge for 2/5 seeds (see logs)
```

---

## Script: `run-fixed-experiments.sh`

### Purpose
Run post-fix validation experiments to measure improvement.

### Location
`scripts/run-fixed-experiments.sh`

### Interface

**Same as `run-baseline-experiments.sh`** but with `--output-dir` default: `experiments/results/fixed`

```bash
./scripts/run-fixed-experiments.sh [OPTIONS]
```

### Behavior

Identical to `run-baseline-experiments.sh` except:
- Default output directory is `experiments/results/fixed/`
- Should be run AFTER all Critical/High fixes are applied
- Must use same seeds as baseline for valid comparison

### Example Usage

```bash
# Run post-fix experiments with same seeds as baseline
./scripts/run-fixed-experiments.sh --seeds 42,43,44,45,46
```

---

## Script: `compare-results.py`

### Purpose
Generate quantified before/after comparison of baseline vs post-fix results.

### Location
`scripts/compare-results.py`

### Interface

```bash
python scripts/compare-results.py [OPTIONS]
```

**Options**:
- `--baseline-dir <path>`: Baseline results directory (default: `experiments/results/baseline`)
- `--fixed-dir <path>`: Post-fix results directory (default: `experiments/results/fixed`)
- `--output <path>`: Output comparison report (default: `analysis/comparison-report.json`)
- `--format <type>`: Output format: `json`, `markdown`, `both` (default: `both`)
- `--methods <list>`: Methods to compare (default: `PN,uPU,nnPU`)
- `--metric <name>`: Primary metric (default: `test_zero_one_loss`)
- `--significance-level <α>`: Significance level for statistical tests (default: `0.05`)
- `--help`: Show usage information

### Inputs
- Baseline experiment results (JSON files from baseline-dir)
- Post-fix experiment results (JSON files from fixed-dir)

### Outputs
- **JSON Report**: Structured comparison data (see [data-model.md](../data-model.md#entity-comparison))
- **Markdown Report** (optional): Human-readable comparison summary
- **Exit Code**:
  - `0`: Comparison completed successfully
  - `1`: Missing baseline or post-fix results
  - `2`: Results do not match (different seeds, mismatched configs)

### Behavior

1. Load all baseline results
2. Load all post-fix results
3. Validate seed matching (same seeds in both)
4. For each method:
   - Compute aggregate statistics (mean, std, min, max)
   - Calculate delta (post-fix - baseline)
   - Perform paired t-test for significance
   - Compare to paper benchmarks (if available)
5. Generate comparison objects (see [data-model.md](../data-model.md#entity-comparison))
6. Save JSON report
7. Generate markdown summary (if requested)
8. Print summary to stdout

### Example Usage

```bash
# Generate both JSON and markdown reports
python scripts/compare-results.py

# JSON only
python scripts/compare-results.py --format json

# Custom directories
python scripts/compare-results.py \
  --baseline-dir /path/to/baseline \
  --fixed-dir /path/to/fixed
```

### Example Output

```
Comparing baseline vs post-fix results...

Loading baseline: experiments/results/baseline/ (15 files)
Loading post-fix: experiments/results/fixed/ (15 files)
✓ All seeds match between baseline and post-fix

┌────────┬─────────────────┬──────────────────┬──────────┬───────────┬─────────────┐
│ Method │ Baseline Error  │ Post-Fix Error   │ Delta    │ % Change  │ Significant │
├────────┼─────────────────┼──────────────────┼──────────┼───────────┼─────────────┤
│ PN     │ 0.045 ± 0.002   │ 0.044 ± 0.002    │ -0.001   │ -2.2%     │ No (p=0.23) │
│ uPU    │ 0.067 ± 0.004   │ 0.055 ± 0.003    │ -0.012   │ -17.9%    │ Yes (p=0.01)│
│ nnPU   │ 0.089 ± 0.003   │ 0.053 ± 0.002    │ -0.036   │ -40.4%    │ Yes (p<0.001│
└────────┴─────────────────┴──────────────────┴──────────┴───────────┴─────────────┘

Paper Benchmarks (MNIST even/odd, 100 labeled positive):
┌────────┬──────────────┬──────────────┬────────────┬───────────────┐
│ Method │ Paper Value  │ Our Post-Fix │ Delta      │ Within ±2%?   │
├────────┼──────────────┼──────────────┼────────────┼───────────────┤
│ PN     │ 0.045        │ 0.044        │ -0.001     │ ✓ Yes         │
│ uPU    │ 0.052        │ 0.055        │ +0.003     │ ✓ Yes         │
│ nnPU   │ 0.051        │ 0.053        │ +0.002     │ ✓ Yes         │
└────────┴──────────────┴──────────────┴────────────┴───────────────┘

✅ All methods reproduce paper results within acceptable tolerance

Reports saved:
- JSON: analysis/comparison-report.json
- Markdown: analysis/comparison-report.md
```

---

## Common Script Requirements

All scripts must:

1. **Error Handling**: Exit with non-zero code on failure, print clear error messages
2. **Logging**: Support `--verbose` flag for detailed output
3. **Idempotency**: Safe to run multiple times (check for existing outputs)
4. **Help Text**: Provide `--help` with usage examples
5. **Validation**: Validate inputs before execution (file paths exist, valid parameter ranges)
6. **Progress Indication**: Show progress for long-running operations
7. **Exit Codes**: Use standard exit codes (0=success, 1=error, 2=usage error)

### Standard Exit Codes

- `0`: Success
- `1`: General error (network failure, file I/O error, unexpected exception)
- `2`: Invalid arguments (missing required param, invalid value)
- `3`: Validation failure (results don't match schema, missing expected files)

### Standard Output Format

- **Informational**: Plain text to stdout
- **Errors**: Plain text to stderr with `ERROR:` prefix
- **Warnings**: Plain text to stderr with `WARNING:` prefix
- **Progress**: Use consistent format (e.g., `[N/Total] Task... ✓ (duration)`)
- **Summary Tables**: Use aligned columns with box-drawing characters

---

## Integration with Existing Infrastructure

### Compatibility with `experiments/scripts/run_all.py`

The analysis scripts should:
- Invoke existing `run_all.py` for running experiments
- NOT modify `run_all.py` unless necessary for schema compliance
- Use `run_all.py --output <dir>` to control result location
- Pass through relevant parameters (seeds, epochs, methods)

### Dataset Handling

Scripts assume:
- MNIST dataset is available or will be downloaded automatically
- Dataset is cached in `notebooks/data/` (per `.gitignore`)
- No manual dataset preparation required

### Compute Device

Scripts should:
- Auto-detect available device (MPS on macOS, CUDA on Linux/Windows, CPU fallback)
- Allow override via `--device` flag if needed
- Log device being used in experiment metadata

---

## Testing Scripts

Each script should have a corresponding test:

- `tests/scripts/test_fetch_paper_source.sh`: Test paper source download
- `tests/scripts/test_clone_references.sh`: Test reference repo cloning
- `tests/scripts/test_run_experiments.sh`: Test experiment execution (with `--dry-run`)
- `tests/scripts/test_compare_results.py`: Test comparison logic with fixtures

---

## Related Contracts

- [experiment-results-schema.md](experiment-results-schema.md): JSON schema for experiment outputs
- [discrepancy-report-schema.md](discrepancy-report-schema.md): Markdown schema for analysis report

See [data-model.md](../data-model.md) for complete entity definitions.
