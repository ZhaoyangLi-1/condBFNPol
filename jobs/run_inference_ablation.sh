#!/bin/bash
#SBATCH --job-name=inference_ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# Inference Steps Ablation Study - SLURM Job Script
# 
# This script runs the inference steps ablation on the LRZ cluster.
# It evaluates both BFN and Diffusion policies at various inference steps.
#
# Usage:
#   sbatch jobs/run_inference_ablation.sh
# =============================================================================

echo "=========================================="
echo "Inference Steps Ablation Study"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# Load modules
module load cuda/12.1
module load python/3.10

# Activate environment
source ~/.bashrc
conda activate bfn

# Set working directory
cd /dss/dsshome1/01/ge46zey3/condBFNPol

# Create logs directory
mkdir -p logs
mkdir -p results/ablation

# Run ablation with all available seeds
python scripts/run_inference_steps_ablation.py \
    --all-seeds \
    --checkpoint-dir cluster_checkpoints/benchmarkresults \
    --bfn-steps "5,10,15,20,25,30,50" \
    --diffusion-steps "10,20,30,40,50,75,100" \
    --n-envs 50 \
    --device cuda \
    --output-dir results/ablation

echo "=========================================="
echo "Ablation study complete!"
echo "Results saved to: results/ablation/"
echo "=========================================="
