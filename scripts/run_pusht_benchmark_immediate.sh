#!/bin/bash
#SBATCH --job-name=pusht_benchmark
#SBATCH --partition=test-hgx-h100-94x4,test-hgx-a100-80x4,test-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --chdir=/dss/dsshome1/0D/ge87gob2/condBFNPol

# Immediate Start Version - Uses Test Partitions
# 
# This version uses test partitions which have idle nodes available NOW:
# - test-hgx-h100-94x4: 1 idle node (H100 - fastest!)
# - test-hgx-a100-80x4: 1 idle node (A100 - fast)
# - test-v100x2: 1 idle node (V100 - slower but available)
#
# NOTE: Test partitions may have shorter time limits. Check with:
#   sinfo -p test-hgx-h100-94x4 -o '%P %l'

set -e

# ================== Environment Setup ==================
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$(dirname "$0")/../scripts/train_workspace.py" ]; then
    cd "$(dirname "$0")/.."
fi
echo "Working directory: $(pwd)"

source activate robodiff

echo "Running on host: $(hostname)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

export WANDB_API_KEY='d47c0bfd84b46e8364f863f142e4fa03c425500e'
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1

# ================== Disk Space Management ==================
# Use scratch directory (usually has more space)
if [ -d "/dss/dssfs04/scratch/ge87gob2" ]; then
    SCRATCH_DIR="/dss/dssfs04/scratch/ge87gob2"
    echo "Using SCRATCH directory: $SCRATCH_DIR"
elif [ -d "$HOME/scratch" ]; then
    SCRATCH_DIR="$HOME/scratch"
    echo "Using local scratch: $SCRATCH_DIR"
else
    SCRATCH_DIR="$HOME"
    echo "Using HOME directory: $SCRATCH_DIR"
fi

OUTPUT_DIR="${SCRATCH_DIR}/condBFNPol/outputs"
mkdir -p "$OUTPUT_DIR" || { echo "ERROR: Cannot create $OUTPUT_DIR"; exit 1; }

# Aggressive cleanup before starting
echo "======================================"
echo "CLEANING UP OLD FILES"
echo "======================================"
echo "Removing outputs older than 1 day..."
find "$OUTPUT_DIR" -type d -name "20*" -mtime +1 -exec rm -rf {} + 2>/dev/null || true

# Use WandB offline mode
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH_DIR}/condBFNPol/wandb"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/condBFNPol/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" 2>/dev/null || true

echo "Removing WandB runs older than 3 days..."
find "$WANDB_DIR" -type d -name "run-*" -mtime +3 -exec rm -rf {} + 2>/dev/null || true

# Check initial disk space
echo ""
echo "Initial disk space:"
df -h "$SCRATCH_DIR" | tail -1
AVAILABLE_GB=$(df -BG "$SCRATCH_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -lt 30 ]; then
    echo "WARNING: Less than 30GB available. Training may fail."
    echo "Please free up more space or use a different directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ================== Configuration ==================
# ALL SEEDS as in original implementation
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda:0}"
PROJECT="${PROJECT:-ablation-benchmark}"
ENTITY="${ENTITY:-aleyna-research}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Data path
if [ -z "$DATA_PATH" ]; then
    if [[ "$(hostname)" == *"lrz"* ]] || [[ "$(hostname)" == *"gpu"* ]]; then
        DATA_PATH="/dss/dsshome1/0D/ge87gob2/condBFNPol/data/pusht/pusht_cchi_v7_replay.zarr"
    else
        DATA_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
    fi
fi

echo "======================================"
echo "PushT Benchmark: BFN vs Diffusion"
echo "IMMEDIATE START - TEST PARTITIONS"
echo "======================================"
echo "Seeds: $SEEDS (ALL SEEDS)"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "WandB: $PROJECT (OFFLINE MODE)"
echo "Output: $OUTPUT_DIR"
echo "Partition: TEST (for immediate start)"
echo "Checkpoint strategy: Only at epochs 100, 200, 300"
echo "Top-K: 1 (only best checkpoint kept)"
echo "======================================"

if [ ! -f "scripts/train_workspace.py" ]; then
    echo "ERROR: Cannot find scripts/train_workspace.py"
    exit 1
fi

# ================== Run Benchmark for ALL SEEDS ==================
for SEED in $SEEDS; do
    echo ""
    echo "======================================"
    echo "SEED $SEED (of 42, 43, 44)"
    echo "======================================"
    
    # Train BFN
    echo ""
    echo "--- Training BFN (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_bfn_pusht \
        hydra.run.dir="${OUTPUT_DIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_bfn_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        training.checkpoint_every=100 \
        checkpoint.save_last_ckpt=false \
        checkpoint.topk.k=1 \
        checkpoint.save_last_snapshot=false \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=offline \
        +logging.entity=$ENTITY
    
    # Train Diffusion
    echo ""
    echo "--- Training Diffusion (seed=$SEED) ---"
    # NOTE: Diffusion config defaults are k=5, save_last_ckpt=true, checkpoint_every=50
    # We override all of these to minimize disk usage
    python scripts/train_workspace.py \
        --config-name=benchmark_diffusion_pusht \
        hydra.run.dir="${OUTPUT_DIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_diffusion_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        training.checkpoint_every=100 \
        checkpoint.save_last_ckpt=false \
        checkpoint.topk.k=1 \
        checkpoint.save_last_snapshot=false \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=offline \
        +logging.entity=$ENTITY
    
    # Clean up after each seed to free space for next seed
    echo ""
    echo "Cleaning up after seed $SEED..."
    # Remove any temporary files
    find "${OUTPUT_DIR}" -name "*.tmp" -delete 2>/dev/null || true
    find "${OUTPUT_DIR}" -name "*.log" -size +100M -delete 2>/dev/null || true
    
    # Check disk space
    echo "Disk space after seed $SEED:"
    df -h "$SCRATCH_DIR" | tail -1
    echo ""
done

echo ""
echo "======================================"
echo "BENCHMARK COMPLETE - ALL SEEDS DONE!"
echo "======================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Checkpoints saved:"
echo "  - Each run: Top-1 checkpoint (best test score)"
echo "  - Checkpoints at epochs: 100, 200, 300"
echo "  - Total: 6 runs × ~3 checkpoints = ~18 checkpoints"
echo ""
echo "WandB logs (offline): $WANDB_DIR"
echo "To sync later: wandb sync $WANDB_DIR"

