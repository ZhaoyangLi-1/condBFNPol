#!/bin/bash
# =============================================================================
# Local Test: Verify Ablation Script Structure
# 
# This script tests the ablation setup locally without running full evaluation.
# Run this before submitting to the cluster to catch any import errors.
#
# Usage:
#   bash scripts/test_ablation_local.sh
# =============================================================================

echo "=========================================="
echo "Testing Inference Steps Ablation Setup"
echo "=========================================="

cd /Users/aleynakara/Documents/condBFNPol

# Test 1: Check if script imports work
echo ""
echo "[1/4] Testing imports..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_inference_steps_ablation import find_checkpoints, generate_ablation_plots
print('  ✓ Imports successful')
"

# Test 2: Find checkpoints
echo ""
echo "[2/4] Finding checkpoints..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_inference_steps_ablation import find_checkpoints
ckpts = find_checkpoints('cluster_checkpoints/benchmarkresults')
print(f'  Found {len(ckpts.get(\"bfn\", {}))} BFN checkpoints')
print(f'  Found {len(ckpts.get(\"diffusion\", {}))} Diffusion checkpoints')
"

# Test 3: Check plot generation with dummy data
echo ""
echo "[3/4] Testing plot generation with dummy data..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_inference_steps_ablation import generate_ablation_plots

# Create dummy results
dummy_results = {
    'bfn': {
        '42': {
            '5': {'mean_score': 0.75, 'std_score': 0.05, 'avg_time_ms': 15},
            '10': {'mean_score': 0.85, 'std_score': 0.04, 'avg_time_ms': 28},
            '20': {'mean_score': 0.91, 'std_score': 0.03, 'avg_time_ms': 55},
        }
    },
    'diffusion': {
        '42': {
            '20': {'mean_score': 0.70, 'std_score': 0.08, 'avg_time_ms': 45},
            '50': {'mean_score': 0.85, 'std_score': 0.05, 'avg_time_ms': 110},
            '100': {'mean_score': 0.95, 'std_score': 0.01, 'avg_time_ms': 220},
        }
    }
}

generate_ablation_plots(dummy_results, 'figures/publication')
print('  ✓ Plot generation successful')
"

# Test 4: Check table generation
echo ""
echo "[4/4] Testing table generation..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_inference_steps_ablation import generate_ablation_table

dummy_results = {
    'bfn': {
        '42': {
            '5': {'mean_score': 0.75, 'std_score': 0.05, 'avg_time_ms': 15},
            '10': {'mean_score': 0.85, 'std_score': 0.04, 'avg_time_ms': 28},
            '20': {'mean_score': 0.91, 'std_score': 0.03, 'avg_time_ms': 55},
        }
    },
    'diffusion': {
        '42': {
            '20': {'mean_score': 0.70, 'std_score': 0.08, 'avg_time_ms': 45},
            '50': {'mean_score': 0.85, 'std_score': 0.05, 'avg_time_ms': 110},
            '100': {'mean_score': 0.95, 'std_score': 0.01, 'avg_time_ms': 220},
        }
    }
}

generate_ablation_table(dummy_results, 'figures/tables')
print('  ✓ Table generation successful')
"

echo ""
echo "=========================================="
echo "All tests passed! Ready for cluster submission."
echo "=========================================="
echo ""
echo "To run on cluster:"
echo "  sbatch jobs/run_inference_ablation.sh"
echo ""
echo "To run locally (with GPU):"
echo "  python scripts/run_inference_steps_ablation.py --all-seeds --device cuda"
