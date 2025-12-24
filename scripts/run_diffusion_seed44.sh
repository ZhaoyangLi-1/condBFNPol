#!/bin/bash
#SBATCH --job-name=diffusion_seed44
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/diffusion_seed44-%j.out
#SBATCH --chdir=/dss/dsshome1/0D/ge87gob2/condBFNPol

# Run Diffusion Policy with seed 44 (failed run)

set -e

cd /dss/dsshome1/0D/ge87gob2/condBFNPol
echo "Working directory: $(pwd)"

# Activate conda
source activate robodiff

# Debug info
echo "Running on host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Environment variables
export WANDB_API_KEY='d47c0bfd84b46e8364f863f142e4fa03c425500e'
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1

# WandB directories
export WANDB_DIR="$HOME/condBFNPol/wandb"
export WANDB_CACHE_DIR="$HOME/condBFNPol/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "======================================"
echo "Running Diffusion Policy - Seed 44"
echo "======================================"

python scripts/train_workspace.py \
    --config-name=benchmark_diffusion_pusht \
    hydra.run.dir="outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_diffusion_seed44" \
    task.dataset.zarr_path="/dss/dsshome1/0D/ge87gob2/condBFNPol/data/pusht/pusht_cchi_v7_replay.zarr" \
    training.seed=44 \
    training.device=cuda:0 \
    training.num_epochs=300 \
    training.checkpoint_every=100 \
    checkpoint.save_last_ckpt=false \
    checkpoint.topk.k=2 \
    dataloader.num_workers=8 \
    val_dataloader.num_workers=8 \
    logging.project=pusht-benchmark \
    logging.mode=online \
    +logging.entity=aleyna-research

echo "======================================"
echo "Diffusion seed 44 complete!"
echo "======================================"
