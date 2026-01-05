#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=logs/ablation-%j.out
#SBATCH --error=logs/ablation-%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================================================
# Ablation Study Job Script
# ==============================================================================

echo "========================================"
echo "Ablation Study - $(date)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Change to project directory
cd /dss/dsshome1/0D/ge87gob2/condBFNPol

# Activate conda environment
source ~/.bashrc
source activate robodiff

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run ablation study
echo "Starting ablation study..."
python scripts/comprehensive_ablation_study.py \
    --checkpoint-dir /dss/dsshome1/0D/ge87gob2/condBFNPol/outputs/2025.12.25 \
    --n-envs 50 \
    --device cuda \
    --output-dir results/ablation

echo ""
echo "========================================"
echo "Ablation study complete - $(date)"
echo "========================================"

