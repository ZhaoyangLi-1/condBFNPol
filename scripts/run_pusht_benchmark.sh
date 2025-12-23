#!/bin/bash
# PushT Benchmark: BFN vs Diffusion Policy
#
# This script runs the full benchmark comparing BFN and Diffusion policies
# on the PushT task with multiple seeds.
#
# Usage:
#   ./scripts/run_pusht_benchmark.sh [--seeds "42 43 44"] [--epochs 300] [--device cuda:0]

set -e

# Default values
SEEDS="42 43 44"
EPOCHS=300
DEVICE="cuda:0"
PROJECT="pusht_bfn_vs_diffusion"
DATA_PATH=""  # Set to absolute path on cluster

# Parse arguments
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
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "PushT Benchmark: BFN vs Diffusion"
echo "======================================"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "WandB Project: $PROJECT"
echo "Data Path: ${DATA_PATH:-data/pusht/pusht_cchi_v7_replay.zarr (default)}"
echo "======================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Run benchmark for each seed
for SEED in $SEEDS; do
    echo ""
    echo "======================================"
    echo "Running seed $SEED"
    echo "======================================"
    
    # Build data path override if specified
    DATA_OVERRIDE=""
    if [ -n "$DATA_PATH" ]; then
        DATA_OVERRIDE="task.dataset.zarr_path=$DATA_PATH"
    fi
    
    # Train BFN
    echo ""
    echo "--- Training BFN (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_bfn_pusht \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        logging.project=$PROJECT \
        logging.mode=online \
        $DATA_OVERRIDE
    
    # Train Diffusion
    echo ""
    echo "--- Training Diffusion (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_diffusion_pusht \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        logging.project=$PROJECT \
        logging.mode=online \
        $DATA_OVERRIDE
done

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "======================================"
echo "Results logged to WandB project: $PROJECT"
echo "Check the dashboard for comparison plots."
