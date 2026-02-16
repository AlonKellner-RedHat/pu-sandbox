#!/usr/bin/env bash

# run-baseline-experiments.sh
# Run full baseline experiments (before fixes) to establish performance metrics
# Specification: specs/001-nnpu-reproduction-analysis/contracts/analysis-scripts.md

set -euo pipefail

# Default values
SEEDS="42,43,44,45,46"
EPOCHS=100
METHODS="PN,uPU,nnPU"
OUTPUT_DIR="experiments/results/baseline"
PARALLEL=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage information
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run full baseline experiments (before fixes) to establish performance metrics.

OPTIONS:
    --seeds <list>         Comma-separated random seeds (default: 42,43,44,45,46)
    --epochs <n>           Number of epochs (default: 100)
    --methods <list>       Comma-separated methods (default: PN,uPU,nnPU)
    --output-dir <path>    Output directory (default: experiments/results/baseline)
    --parallel             Run experiments in parallel (experimental)
    --dry-run              Print commands without executing
    --help                 Show this help message

EXAMPLES:
    # Run full baseline (15 experiments, ~2-4 hours)
    $0

    # Quick test with fewer seeds and epochs
    $0 --seeds 42,43 --epochs 10

    # Only specific methods
    $0 --methods nnPU

    # Dry run to see what would execute
    $0 --dry-run
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}ERROR: Unknown option: $1${NC}" >&2
            echo "Run '$0 --help' for usage information" >&2
            exit 2
            ;;
    esac
done

# Validate git state
if [[ "$DRY_RUN" == "false" ]]; then
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "${GREEN}Git working directory clean${NC}"
    else
        echo -e "${YELLOW}WARNING: Uncommitted changes in working directory${NC}" >&2
        echo "Baseline results may not be fully reproducible" >&2
    fi
fi

# Get current git commit
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Running baseline experiments..."
echo "Git commit: $COMMIT_SHORT"
echo "Branch: $BRANCH"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory and logs subdirectory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Convert comma-separated lists to arrays
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#METHOD_ARRAY[@]} * ${#SEED_ARRAY[@]}))
CURRENT_EXPERIMENT=0
FAILED_EXPERIMENTS=()

echo "Configuration:"
echo "  Methods: ${METHOD_ARRAY[*]}"
echo "  Seeds: ${SEED_ARRAY[*]}"
echo "  Epochs: $EPOCHS"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo ""

# Start timer
START_TIME=$(date +%s)

# Run experiments
for method in "${METHOD_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        EXPERIMENT_ID="${method}_seed${seed}"
        OUTPUT_FILE="$OUTPUT_DIR/${EXPERIMENT_ID}.json"
        LOG_FILE="$OUTPUT_DIR/logs/${EXPERIMENT_ID}.log"

        echo -ne "${BLUE}[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}]${NC} ${EXPERIMENT_ID}... "

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "(dry run)"
            echo "  Command: python experiments/scripts/run_all.py --method $method --seed $seed --epochs $EPOCHS --output $OUTPUT_FILE"
            continue
        fi

        # Start experiment timer
        EXP_START=$(date +%s)

        # Run experiment
        if python experiments/scripts/run_all.py \
            --method "$method" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --output "$OUTPUT_FILE" \
            > "$LOG_FILE" 2>&1; then

            # Calculate duration
            EXP_END=$(date +%s)
            DURATION=$((EXP_END - EXP_START))

            # Extract test error from output (if available)
            if [[ -f "$OUTPUT_FILE" ]]; then
                # Try to get final test error from JSON
                TEST_ERROR=$(python -c "import json; data=json.load(open('$OUTPUT_FILE')); print(f\"{data['final_metrics']['final_test_error']:.3f}\")" 2>/dev/null || echo "N/A")
                echo -e "${GREEN}✓${NC} (${DURATION}s, test error: ${TEST_ERROR})"
            else
                echo -e "${GREEN}✓${NC} (${DURATION}s)"
            fi
        else
            echo -e "${RED}✗ FAILED${NC}"
            FAILED_EXPERIMENTS+=("$EXPERIMENT_ID")
            echo "  See log: $LOG_FILE"
        fi
    done
done

# Calculate total duration
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================"

# Print summary
if [[ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]]; then
    echo -e "${GREEN}All experiments completed successfully${NC}"
else
    echo -e "${RED}${#FAILED_EXPERIMENTS[@]} experiments failed:${NC}"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

# Print summary table (if not dry run and at least one success)
if [[ "$DRY_RUN" == "false" ]] && [[ $CURRENT_EXPERIMENT -gt ${#FAILED_EXPERIMENTS[@]} ]]; then
    echo ""
    echo "Summary:"
    printf "┌────────┬────────────────────┬───────────┬───────────┐\n"
    printf "│ %-6s │ %-18s │ %-9s │ %-9s │\n" "Method" "Test Error (mean)" "Std Dev" "Converged"
    printf "├────────┼────────────────────┼───────────┼───────────┤\n"

    for method in "${METHOD_ARRAY[@]}"; do
        # Collect test errors for this method
        ERRORS=()
        CONVERGED_COUNT=0

        for seed in "${SEED_ARRAY[@]}"; do
            OUTPUT_FILE="$OUTPUT_DIR/${method}_seed${seed}.json"
            if [[ -f "$OUTPUT_FILE" ]]; then
                ERROR=$(python -c "import json; data=json.load(open('$OUTPUT_FILE')); print(data['final_metrics']['final_test_error'])" 2>/dev/null || echo "")
                if [[ -n "$ERROR" ]]; then
                    ERRORS+=("$ERROR")
                    # Check if converged (simple heuristic: has convergence_epoch field)
                    CONV=$(python -c "import json; data=json.load(open('$OUTPUT_FILE')); print(1 if data['final_metrics'].get('convergence_epoch') else 0)" 2>/dev/null || echo "0")
                    CONVERGED_COUNT=$((CONVERGED_COUNT + CONV))
                fi
            fi
        done

        if [[ ${#ERRORS[@]} -gt 0 ]]; then
            # Calculate mean and std
            MEAN=$(python -c "import statistics; errors=[${ERRORS[*]}]; print(f'{statistics.mean(errors):.3f}')" 2>/dev/null || echo "N/A")
            STD=$(python -c "import statistics; errors=[${ERRORS[*]}]; print(f'{statistics.stdev(errors):.3f}' if len(errors) > 1 else '0.000')" 2>/dev/null || echo "N/A")
            CONV_RATIO="${CONVERGED_COUNT}/${#ERRORS[@]}"

            printf "│ %-6s │ %-18s │ %-9s │ %-9s │\n" "$method" "$MEAN ± $STD" "$STD" "$CONV_RATIO"
        fi
    done

    printf "└────────┴────────────────────┴───────────┴───────────┘\n"
fi

echo ""
printf "Total runtime: %dh %dm %ds\n" "$HOURS" "$MINUTES" "$SECONDS"
echo "Results saved to: $OUTPUT_DIR"

# Check for warnings
if [[ "$DRY_RUN" == "false" ]]; then
    # Check if nnPU converged for all seeds
    NNPU_FILES=($(ls "$OUTPUT_DIR"/nnPU_seed*.json 2>/dev/null || true))
    if [[ ${#NNPU_FILES[@]} -gt 0 ]]; then
        NNPU_CONVERGED=0
        for file in "${NNPU_FILES[@]}"; do
            CONV=$(python -c "import json; data=json.load(open('$file')); print(1 if data['final_metrics'].get('convergence_epoch') else 0)" 2>/dev/null || echo "0")
            NNPU_CONVERGED=$((NNPU_CONVERGED + CONV))
        done

        if [[ $NNPU_CONVERGED -lt ${#NNPU_FILES[@]} ]]; then
            NOT_CONVERGED=$((${#NNPU_FILES[@]} - NNPU_CONVERGED))
            echo ""
            echo -e "${YELLOW}⚠ Warning: nnPU did not converge for ${NOT_CONVERGED}/${#NNPU_FILES[@]} seeds (see logs)${NC}"
        fi
    fi
fi

# Exit code
if [[ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]]; then
    exit 0
else
    exit 1
fi
