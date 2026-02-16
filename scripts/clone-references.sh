#!/usr/bin/env bash

# clone-references.sh
# Clone reference implementations of nnPU learning for comparison
# Specification: specs/001-nnpu-reproduction-analysis/contracts/analysis-scripts.md

set -euo pipefail

# Default values
OUTPUT_DIR="analysis/reference-implementations"
REPOS="kiryor,cimeister"
SHALLOW=false
FORCE=false

# Repository URLs
KIRYOR_REPO="https://github.com/kiryor/nnPUlearning.git"
CIMEISTER_REPO="https://github.com/cimeister/pu-learning.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Usage information
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Clone reference implementations of nnPU learning for comparison.

OPTIONS:
    --output-dir <path>     Output directory (default: analysis/reference-implementations)
    --repos <list>          Comma-separated list of repos to clone (default: kiryor,cimeister)
                            Available: kiryor, cimeister
    --shallow               Shallow clone (faster, less history)
    --force                 Re-clone even if already exists
    --help                  Show this help message

EXAMPLES:
    # Clone with defaults
    $0

    # Shallow clone for faster download
    $0 --shallow

    # Custom output directory
    $0 --output-dir /tmp/references

    # Clone only one repository
    $0 --repos kiryor
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repos)
            REPOS="$2"
            shift 2
            ;;
        --shallow)
            SHALLOW=true
            shift
            ;;
        --force)
            FORCE=true
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Cloning reference implementations..."

# Clone options
CLONE_OPTS=""
if [[ "$SHALLOW" == "true" ]]; then
    CLONE_OPTS="--depth 1"
fi

# Track success/failure
ALL_SUCCESS=true

# Clone kiryor if requested
if [[ "$REPOS" == *"kiryor"* ]]; then
    KIRYOR_DIR="$OUTPUT_DIR/kiryor-nnPUlearning"

    if [[ -d "$KIRYOR_DIR" ]] && [[ "$FORCE" == "false" ]]; then
        echo -e "${YELLOW}kiryor/nnPUlearning already exists at $KIRYOR_DIR${NC}"
        echo "Use --force to re-clone"
    else
        # Remove existing if force
        if [[ "$FORCE" == "true" ]] && [[ -d "$KIRYOR_DIR" ]]; then
            rm -rf "$KIRYOR_DIR"
        fi

        # Clone
        echo "Cloning kiryor/nnPUlearning..."
        if git clone $CLONE_OPTS "$KIRYOR_REPO" "$KIRYOR_DIR" 2>&1 | grep -v "^Cloning"; then
            # Get commit info
            COMMIT_SHA=$(cd "$KIRYOR_DIR" && git rev-parse --short HEAD)
            COMMIT_COUNT=$(cd "$KIRYOR_DIR" && git rev-list --count HEAD)
            LAST_UPDATE=$(cd "$KIRYOR_DIR" && git log -1 --format=%cd --date=short)

            echo -e "${GREEN}✓${NC} Cloned kiryor/nnPUlearning (commit: $COMMIT_SHA)"
            echo "  - $COMMIT_COUNT commits, last updated $LAST_UPDATE"
        else
            echo -e "${RED}ERROR: Failed to clone kiryor/nnPUlearning${NC}" >&2
            ALL_SUCCESS=false
        fi
    fi
fi

# Clone cimeister if requested
if [[ "$REPOS" == *"cimeister"* ]]; then
    CIMEISTER_DIR="$OUTPUT_DIR/cimeister-pu-learning"

    if [[ -d "$CIMEISTER_DIR" ]] && [[ "$FORCE" == "false" ]]; then
        echo -e "${YELLOW}cimeister/pu-learning already exists at $CIMEISTER_DIR${NC}"
        echo "Use --force to re-clone"
    else
        # Remove existing if force
        if [[ "$FORCE" == "true" ]] && [[ -d "$CIMEISTER_DIR" ]]; then
            rm -rf "$CIMEISTER_DIR"
        fi

        # Clone
        echo "Cloning cimeister/pu-learning..."
        if git clone $CLONE_OPTS "$CIMEISTER_REPO" "$CIMEISTER_DIR" 2>&1 | grep -v "^Cloning"; then
            # Get commit info
            COMMIT_SHA=$(cd "$CIMEISTER_DIR" && git rev-parse --short HEAD)
            COMMIT_COUNT=$(cd "$CIMEISTER_DIR" && git rev-list --count HEAD)
            LAST_UPDATE=$(cd "$CIMEISTER_DIR" && git log -1 --format=%cd --date=short)

            echo -e "${GREEN}✓${NC} Cloned cimeister/pu-learning (commit: $COMMIT_SHA)"
            echo "  - $COMMIT_COUNT commits, last updated $LAST_UPDATE"
        else
            echo -e "${RED}ERROR: Failed to clone cimeister/pu-learning${NC}" >&2
            ALL_SUCCESS=false
        fi
    fi
fi

if [[ "$ALL_SUCCESS" == "true" ]]; then
    echo ""
    echo "Reference implementations ready for comparison."
    echo "Record these commit SHAs in your discrepancy report."
    exit 0
else
    echo -e "${RED}Some repositories failed to clone${NC}" >&2
    exit 1
fi
