#!/bin/bash
# =============================================================================
# Regenerate All Publication Figures
# Style: Anthropic Research Palette
# =============================================================================
#
# Usage:
#   ./scripts/regenerate_all_figures.sh
#   ./scripts/regenerate_all_figures.sh --quick   # Skip slow figures
#
# This script activates the robodiff environment and runs all figure generation.
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "======================================================================"
echo "  PUBLICATION FIGURE GENERATOR"
echo "  Style: Anthropic Research Palette"
echo "  BFN: #2D8B8B (Teal) | Diffusion: #D97757 (Coral)"
echo "======================================================================"
echo ""
echo "  Project: $PROJECT_ROOT"
echo "  Time: $(date)"
echo ""

# =============================================================================
# Activate conda environment
# =============================================================================
echo ">>> Activating conda environment..."

# Try different activation methods
if command -v conda &> /dev/null; then
    # Initialize conda for bash
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate robodiff 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not activate robodiff, trying base...${NC}"
        conda activate base 2>/dev/null || true
    }
fi

# Verify Python
echo "  Python: $(which python)"
echo "  Version: $(python --version 2>&1)"
echo ""

# =============================================================================
# Create output directories
# =============================================================================
mkdir -p figures/publication
mkdir -p figures/tables

# =============================================================================
# Quick mode check
# =============================================================================
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}>>> Quick mode: skipping slow figure generation${NC}"
    echo ""
fi

# =============================================================================
# Helper function
# =============================================================================
run_script() {
    local script=$1
    local args=$2
    local name=$(basename "$script" .py)
    
    echo "----------------------------------------------------------------------"
    echo ">>> Running: $name"
    
    if python "$script" $args 2>&1; then
        echo -e "${GREEN}  ✓ $name completed${NC}"
        return 0
    else
        echo -e "${RED}  ✗ $name failed${NC}"
        return 1
    fi
}

# =============================================================================
# FIGURE GENERATION
# =============================================================================

echo "======================================================================"
echo "  GENERATING FIGURES"
echo "======================================================================"
echo ""

# 1. Ablation figures (from JSON results)
if [[ -f "results/ablation/ablation_results.json" ]]; then
    run_script "scripts/plot_ablation_results.py" \
        "--results-file results/ablation/ablation_results.json --output-dir figures/publication" || true
else
    echo -e "${YELLOW}  ○ Skipping ablation: results file not found${NC}"
fi

# 2. Prediction visualizations
run_script "scripts/generate_prediction_visualizations.py" "" || true

# 3. Setup figures (horizon config, etc.)
run_script "scripts/generate_setup_figures.py" "" || true

# 4. Training stability
run_script "scripts/generate_training_stability_figure.py" "" || true

# 5. Multimodal figures
run_script "scripts/generate_multimodal_figures.py" "" || true

# 6. Main publication figures (if training data exists)
if [[ -d "outputs/2025.12.25" ]]; then
    if [[ "$QUICK_MODE" == false ]]; then
        run_script "scripts/generate_publication_figures.py" \
            "--checkpoint-dir outputs/2025.12.25 --output-dir figures/publication" || true
        
        # Training loss curves
        run_script "scripts/plot_training_loss.py" \
            "--base-dir outputs/2025.12.25 --output-dir figures/publication --all-plots" || true
    else
        echo -e "${YELLOW}  ○ Skipping slow figures (--quick mode)${NC}"
    fi
else
    echo -e "${YELLOW}  ○ Skipping training figures: outputs/2025.12.25 not found${NC}"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "======================================================================"
echo "  SUMMARY"
echo "======================================================================"
echo ""

# Count output files
PDF_COUNT=$(find figures/publication -name "*.pdf" 2>/dev/null | wc -l)
PNG_COUNT=$(find figures/publication -name "*.png" 2>/dev/null | wc -l)
TEX_COUNT=$(find figures/tables -name "*.tex" 2>/dev/null | wc -l)

echo "  Output directory: figures/publication/"
echo "  PDF files: $PDF_COUNT"
echo "  PNG files: $PNG_COUNT"
echo "  LaTeX tables: $TEX_COUNT"
echo ""

# List recent files
echo "  Recently modified:"
ls -lt figures/publication/*.pdf 2>/dev/null | head -10 | awk '{print "    • " $NF}' | sed 's|figures/publication/||'
echo ""

echo "======================================================================"
echo -e "  ${GREEN}DONE!${NC}"
echo "======================================================================"
echo ""

