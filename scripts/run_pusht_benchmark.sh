#!/bin/bash
#SBATCH --job-name=pusht_benchmark
#SBATCH --partition=lrz-hgx-h100-94x4    # H100 partition
#SBATCH --gres=gpu:1                     # Request 1 H100 GPU
#SBATCH --cpus-per-task=16               # 16 CPU cores for fast data loading
#SBATCH --mem=64G                        # 64GB RAM
#SBATCH --time=48:00:00                  # 48 hour limit for full benchmark
#SBATCH --output=logs/benchmark-%j.out   # Save logs here

# PushT Benchmark: Flow Matching vs Diffusion Policy
#
# This script runs the full benchmark comparing Flow Matching and Diffusion policies
# on the PushT task with multiple seeds.
#
# Flow Matching is preferred over BFN because training and inference are perfectly aligned:
# - Training: predict velocity v = data - noise
# - Inference: integrate ODE from noise to data
#
# Usage (interactive):
#   ./scripts/run_pusht_benchmark.sh [--seeds "42 43 44"] [--epochs 300]
#
# Usage (SLURM):
#   sbatch scripts/run_pusht_benchmark.sh

set -e

# ================== Environment Setup ==================
# Activate conda environment
source activate robodiff

# Debug info
echo "Running on host: $(hostname)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# Set environment variables
export WANDB_API_KEY='d47c0bfd84b46e8364f863f142e4fa03c425500e'
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1

# ================== Configuration ==================
# Default values
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda:0}"
PROJECT="${PROJECT:-pusht-benchmark}"
ENTITY="${ENTITY:-aleyna-research}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Data path - auto-detect based on hostname or set explicitly
if [ -z "$DATA_PATH" ]; then
    if [[ "$(hostname)" == *"lrz"* ]] || [[ "$(hostname)" == *"gpu"* ]]; then
        # Cluster path
        DATA_PATH="/dss/dsshome1/0D/ge87gob2/condBFNPol/data/pusht/pusht_cchi_v7_replay.zarr"
    else
        # Local path
        DATA_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
    fi
fi

# Parse command line arguments (for interactive use)
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
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "PushT Benchmark: Flow Matching vs Diffusion"
echo "======================================"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "WandB Project: $PROJECT"
echo "WandB Entity: $ENTITY"
echo "Data Path: $DATA_PATH"
echo "Num Workers: $NUM_WORKERS"
echo "======================================"

# Get script directory and cd to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs

# ================== Run Benchmark ==================
for SEED in $SEEDS; do
    echo ""
    echo "======================================"
    echo "Running seed $SEED"
    echo "======================================"
    
    # Train Flow Matching (recommended over BFN - training/inference aligned)
    echo ""
    echo "--- Training Flow Matching (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_flow_pusht \
        hydra.run.dir="data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_flow_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=online \
        +logging.entity=$ENTITY
    
    # Train Diffusion
    echo ""
    echo "--- Training Diffusion (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_diffusion_pusht \
        hydra.run.dir="/benchmark/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_diffusion_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=online \
        +logging.entity=$ENTITY
done

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "======================================"
echo "Results logged to WandB project: $PROJECT"
echo "View at: https://wandb.ai/$ENTITY/$PROJECT"
