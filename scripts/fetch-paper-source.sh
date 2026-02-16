#!/usr/bin/env bash

# fetch-paper-source.sh
# Download and extract nnPU paper TeX source from arXiv
# Specification: specs/001-nnpu-reproduction-analysis/contracts/analysis-scripts.md

set -euo pipefail

# Default values
OUTPUT_DIR="analysis/paper-source"
PAPER_ID="1703.00593"
FORCE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Usage information
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download and extract the nnPU paper TeX source from arXiv.

OPTIONS:
    --output-dir <path>    Output directory (default: analysis/paper-source)
    --paper-id <id>        arXiv paper ID (default: 1703.00593)
    --force                Re-download even if already exists
    --help                 Show this help message

EXAMPLES:
    # Download with defaults
    $0

    # Custom output directory
    $0 --output-dir /tmp/papers

    # Force re-download
    $0 --force
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
        --paper-id)
            PAPER_ID="$2"
            shift 2
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

# Full path to paper directory
PAPER_DIR="$OUTPUT_DIR/$PAPER_ID"

# Check if source already exists
if [[ -d "$PAPER_DIR" ]] && [[ "$FORCE" == "false" ]]; then
    echo -e "${YELLOW}Paper source already exists at $PAPER_DIR${NC}"
    echo "Use --force to re-download"
    exit 0
fi

echo "Fetching paper source for $PAPER_ID..."

# Download source tarball
TARBALL_URL="https://arxiv.org/src/$PAPER_ID"
TARBALL_FILE="$OUTPUT_DIR/${PAPER_ID}.tar.gz"

echo "Downloading from $TARBALL_URL..."
if curl -L -o "$TARBALL_FILE" "$TARBALL_URL" 2>&1; then
    FILESIZE=$(du -h "$TARBALL_FILE" | cut -f1)
    echo -e "${GREEN}✓${NC} Downloaded source tarball ($FILESIZE)"
else
    echo -e "${RED}ERROR: Download failed (network error or invalid paper ID)${NC}" >&2
    rm -f "$TARBALL_FILE"
    exit 1
fi

# Create paper directory
mkdir -p "$PAPER_DIR"

# Extract tarball
echo "Extracting to $PAPER_DIR/..."
if tar -xzf "$TARBALL_FILE" -C "$PAPER_DIR" 2>&1; then
    echo -e "${GREEN}✓${NC} Extracted to $PAPER_DIR/"
else
    echo -e "${RED}ERROR: Extraction failed (corrupted archive)${NC}" >&2
    rm -rf "$PAPER_DIR"
    rm -f "$TARBALL_FILE"
    exit 2
fi

# Clean up tarball
rm -f "$TARBALL_FILE"

# Verify extraction (check for .tex files)
TEX_COUNT=$(find "$PAPER_DIR" -name "*.tex" | wc -l | tr -d ' ')
BIB_COUNT=$(find "$PAPER_DIR" -name "*.bib" | wc -l | tr -d ' ')
FIG_COUNT=$(find "$PAPER_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.pdf" -o -name "*.eps" \) | wc -l | tr -d ' ')

if [[ "$TEX_COUNT" -eq 0 ]]; then
    echo -e "${RED}ERROR: No .tex files found in extracted archive${NC}" >&2
    exit 2
fi

echo -e "${GREEN}✓${NC} Found $TEX_COUNT TeX file(s): $(find "$PAPER_DIR" -name "*.tex" -exec basename {} \; | tr '\n' ' ')"
if [[ "$BIB_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}✓${NC} Found $BIB_COUNT BibTeX file(s): $(find "$PAPER_DIR" -name "*.bib" -exec basename {} \; | tr '\n' ' ')"
fi
if [[ "$FIG_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}✓${NC} Found $FIG_COUNT figure file(s)"
fi

echo ""
echo "Paper source ready for analysis."
exit 0
