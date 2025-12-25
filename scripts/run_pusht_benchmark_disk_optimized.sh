#!/bin/bash
#SBATCH --job-name=pusht_benchmark
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --chdir=/dss/dsshome1/0D/ge87gob2/condBFNPol

# Disk-Space-Optimized PushT Benchmark
# 
# This version reduces disk usage by:
# 1. Using scratch directory (if available)
# 2. Reducing checkpoint frequency
# 3. Keeping only top-1 checkpoint
# 4. Using WandB offline mode
# 5. Cleaning up old outputs before starting

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
# Try to use scratch directory first (usually has more space)
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

# Clean up old outputs to free space (keep only last 2 days)
echo "Cleaning old outputs (keeping last 2 days)..."
find "$OUTPUT_DIR" -type d -name "20*" -mtime +2 -exec rm -rf {} + 2>/dev/null || true

# Use WandB offline mode to avoid syncing large files
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH_DIR}/condBFNPol/wandb"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/condBFNPol/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" 2>/dev/null || true

# Clean up old wandb runs (keep only last 7 days)
echo "Cleaning old WandB runs..."
find "$WANDB_DIR" -type d -name "run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true

# Check disk space
echo ""
echo "Disk space check:"
df -h "$SCRATCH_DIR" | tail -1
echo ""

# ================== Configuration ==================
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda:0}"
PROJECT="${PROJECT:-pusht-benchmark}"
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --entity) ENTITY="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================"
echo "PushT Benchmark: BFN vs Diffusion"
echo "DISK-SPACE OPTIMIZED VERSION"
echo "======================================"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "WandB Project: $PROJECT (OFFLINE MODE)"
echo "WandB Entity: $ENTITY"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Workers: $NUM_WORKERS"
echo "======================================"

if [ ! -f "scripts/train_workspace.py" ]; then
    echo "ERROR: Cannot find scripts/train_workspace.py in $(pwd)"
    exit 1
fi

# ================== Run Benchmark ==================
for SEED in $SEEDS; do
    echo ""
    echo "======================================"
    echo "Running seed $SEED"
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
        training.checkpoint_every=150 \
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
    python scripts/train_workspace.py \
        --config-name=benchmark_diffusion_pusht \
        hydra.run.dir="${OUTPUT_DIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_diffusion_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        training.checkpoint_every=150 \
        checkpoint.save_last_ckpt=false \
        checkpoint.topk.k=1 \
        checkpoint.save_last_snapshot=false \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=offline \
        +logging.entity=$ENTITY
    
    # Clean up after each seed (remove non-topk checkpoints to save space)
    echo "Cleaning up intermediate files after seed $SEED..."
    # Keep only topk checkpoints, remove others
    find "${OUTPUT_DIR}" -path "*/bfn_seed${SEED}/*" -name "*.ckpt" -type f ! -name "*epoch=*" -delete 2>/dev/null || true
    find "${OUTPUT_DIR}" -path "*/diffusion_seed${SEED}/*" -name "*.ckpt" -type f ! -name "*epoch=*" -delete 2>/dev/null || true
    
    # Check disk space after each seed
    echo "Disk space after seed $SEED:"
    df -h "$SCRATCH_DIR" | tail -1
done

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "======================================"
echo "Results saved to: $OUTPUT_DIR"
echo "WandB logs (offline): $WANDB_DIR"
echo ""
echo "To sync WandB logs later, run:"
echo "  wandb sync $WANDB_DIR"

