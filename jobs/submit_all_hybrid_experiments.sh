#!/bin/bash
# =============================================================================
# Master Script: Submit All Hybrid Discrete Experiments
# 
# This script submits all experiments for the thesis comparing:
# - BFN-Policy with native discrete handling (Dirichlet-Categorical)
# - Diffusion Policy with continuous relaxation (baseline)
#
# Tasks: Lift, Can, Square (RoboMimic - all have gripper actions)
# Seeds: 42, 43, 44 (for statistical significance)
# Total jobs: 3 tasks × 2 methods × 3 seeds = 18 jobs
#
# Usage:
#   bash jobs/submit_all_hybrid_experiments.sh
#   bash jobs/submit_all_hybrid_experiments.sh --dry-run  # Preview only
# =============================================================================

# Don't exit on error - we want to submit all jobs even if some fail
# set -e

# Configuration
PROJECT_DIR="/dss/dsshome1/0D/ge87gob2/condBFNPol"
JOB_SCRIPT="jobs/run_hybrid_discrete_experiments.sh"

# Experiment parameters
TASKS=("lift")  # Start with lift only; add "can" "square" later if needed
METHODS=("bfn_hybrid" "diffusion")
SEEDS=(42 43 44)

# Parse arguments
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE (no jobs will be submitted) ==="
    echo ""
fi

cd $PROJECT_DIR

echo "=============================================="
echo "Hybrid Discrete Experiments - Thesis Submission"
echo "=============================================="
echo "Tasks:   ${TASKS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Seeds:   ${SEEDS[*]}"
echo "Total jobs: $((${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))"
echo "=============================================="
echo ""

# Track submitted jobs
SUBMITTED=0
JOB_IDS=()

# Submit all experiments
for TASK in "${TASKS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            JOB_NAME="${METHOD}_${TASK}_s${SEED}"
            
            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY RUN] Would submit: $JOB_NAME"
                echo "  Command: sbatch --export=TASK=$TASK,METHOD=$METHOD,SEED=$SEED $JOB_SCRIPT"
            else
                echo "Submitting: $JOB_NAME"
                
                # Submit job and capture job ID
                JOB_OUTPUT=$(sbatch --export=TASK=$TASK,METHOD=$METHOD,SEED=$SEED \
                    --job-name=$JOB_NAME \
                    $JOB_SCRIPT 2>&1)
                
                # Extract job ID
                JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+' | head -1)
                
                if [ -n "$JOB_ID" ]; then
                    echo "  -> Job ID: $JOB_ID"
                    JOB_IDS+=($JOB_ID)
                    ((SUBMITTED++))
                else
                    echo "  -> ERROR: $JOB_OUTPUT"
                fi
            fi
            
            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo ""
echo "=============================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "DRY RUN COMPLETE"
    echo "Would submit $((${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]})) jobs"
else
    echo "SUBMISSION COMPLETE"
    echo "Submitted: $SUBMITTED jobs"
    echo "Job IDs: ${JOB_IDS[*]}"
fi
echo "=============================================="

# Show queue status
echo ""
echo "Current queue status:"
squeue -u $USER -o "%.10i %.9P %.25j %.8T %.10M" 2>/dev/null | head -20

echo ""
echo "=============================================="
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "View logs:"
echo "  tail -f logs/hybrid_*.out"
echo ""
echo "After completion, analyze results:"
echo "  python scripts/analyze_hybrid_experiments.py --task lift"
echo "=============================================="

