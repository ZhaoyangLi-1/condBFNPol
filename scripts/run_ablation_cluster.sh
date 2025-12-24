#!/bin/bash
# =============================================================================
# Cluster Ablation Study Script
# Runs comprehensive ablation study with disk space efficiency
# =============================================================================

set -e  # Exit on error
set -u  # Exit on unset variable

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default values
CHECKPOINT_DIR="${CHECKPOINT_DIR:-cluster_checkpoints/benchmarkresults}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ablation}"
DEVICE="${DEVICE:-cuda}"
N_ENVS="${N_ENVS:-50}"
BFN_STEPS="${BFN_STEPS:-5,10,15,20,30,50}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10,20,30,50,75,100}"

# Disk space management
USE_TMPFS="${USE_TMPFS:-false}"  # Use tmpfs for intermediate files if available
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-false}"  # Delete checkpoints after loading
COMPRESS_OUTPUT="${COMPRESS_OUTPUT:-true}"  # Compress output files
KEEP_LOGS="${KEEP_LOGS:-true}"  # Keep log files
MAX_DISK_USAGE_GB="${MAX_DISK_USAGE_GB:-10}"  # Maximum disk usage in GB

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE:-/dev/stdout}"
}

check_disk_space() {
    local dir="$1"
    local usage_gb=$(df -BG "$dir" | tail -1 | awk '{print $4}' | sed 's/G//')
    log "Disk space available: ${usage_gb}GB"
    if [ "$usage_gb" -lt "$MAX_DISK_USAGE_GB" ]; then
        log "WARNING: Low disk space (${usage_gb}GB < ${MAX_DISK_USAGE_GB}GB)"
        return 1
    fi
    return 0
}

cleanup_temp_files() {
    log "Cleaning up temporary files..."
    find "$OUTPUT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$OUTPUT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$OUTPUT_DIR" -name "*.pyc" -delete 2>/dev/null || true
}

compress_output() {
    if [ "$COMPRESS_OUTPUT" = "true" ]; then
        log "Compressing output files..."
        if [ -f "$OUTPUT_DIR/ablation_results.json" ]; then
            gzip -f "$OUTPUT_DIR/ablation_results.json" 2>/dev/null || true
            log "Compressed: ablation_results.json.gz"
        fi
    fi
}

setup_tmpfs() {
    if [ "$USE_TMPFS" = "true" ] && [ -w "/dev/shm" ]; then
        TMPFS_DIR="/dev/shm/ablation_$$"
        mkdir -p "$TMPFS_DIR"
        export TMPDIR="$TMPFS_DIR"
        export TEMP="$TMPFS_DIR"
        export TMP="$TMPFS_DIR"
        log "Using tmpfs for temporary files: $TMPFS_DIR"
        trap "rm -rf $TMPFS_DIR" EXIT
    else
        log "Not using tmpfs (USE_TMPFS=false or /dev/shm not available)"
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

main() {
    log "=============================================================================="
    log "CLUSTER ABLATION STUDY"
    log "=============================================================================="
    log "Checkpoint dir: $CHECKPOINT_DIR"
    log "Output dir: $OUTPUT_DIR"
    log "Device: $DEVICE"
    log "N environments: $N_ENVS"
    log "BFN steps: $BFN_STEPS"
    log "Diffusion steps: $DIFFUSION_STEPS"
    log "=============================================================================="
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Setup logging
    LOG_FILE="$OUTPUT_DIR/ablation_run.log"
    log "Log file: $LOG_FILE"
    
    # Check disk space
    check_disk_space "$OUTPUT_DIR" || {
        log "ERROR: Insufficient disk space. Aborting."
        exit 1
    }
    
    # Setup tmpfs if requested
    setup_tmpfs
    
    # Change to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_ROOT"
    log "Working directory: $PROJECT_ROOT"
    
    # Activate conda environment if needed
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        log "Using conda environment: $CONDA_DEFAULT_ENV"
    elif [ -f "environment.yml" ] || [ -f "environment.yaml" ]; then
        log "NOTE: Consider activating your conda environment before running"
    fi
    
    # Check if Python script exists
    ABLATION_SCRIPT="scripts/comprehensive_ablation_study.py"
    if [ ! -f "$ABLATION_SCRIPT" ]; then
        log "ERROR: Ablation script not found: $ABLATION_SCRIPT"
        exit 1
    fi
    
    # Check if checkpoints exist, search if not found
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log "WARNING: Checkpoint directory not found: $CHECKPOINT_DIR"
        log "Searching for checkpoints from benchmark runs..."
        
        # Try to find checkpoints in outputs (from run_pusht_benchmark.sh)
        if [ -d "outputs" ]; then
            local ckpt_count=$(find outputs -name "*.ckpt" 2>/dev/null | wc -l)
            if [ "$ckpt_count" -gt 0 ]; then
                log "Found $ckpt_count checkpoints in outputs/"
                log "Organizing them for ablation study..."
                
                # Use the new preparation script
                if bash scripts/prepare_ablation_from_outputs.sh --symlink 2>&1 | tee -a "$LOG_FILE"; then
                    log "✓ Checkpoints organized! Continuing with ablation..."
                else
                    log "Could not automatically organize. Please run manually:"
                    log "  bash scripts/prepare_ablation_from_outputs.sh"
                    log "  # Then re-run this script"
                    exit 1
                fi
            else
                log "No checkpoints found in outputs/"
                log ""
                log "You need to run the benchmark first:"
                log "  sbatch scripts/run_pusht_benchmark.sh"
                log ""
                log "Or if you have checkpoints elsewhere:"
                log "  export CHECKPOINT_DIR=/path/to/checkpoints"
                log "  bash scripts/run_ablation_cluster.sh"
                log ""
                exit 1
            fi
        else
            log "No outputs directory found."
            log ""
            log "You need to run the benchmark first:"
            log "  sbatch scripts/run_pusht_benchmark.sh"
            log ""
            log "After benchmark completes, this script will automatically find and use the checkpoints."
            log ""
            exit 1
        fi
    fi
    
    # Pre-cleanup
    cleanup_temp_files
    
    # Run ablation study
    log "=============================================================================="
    log "STARTING ABLATION STUDY"
    log "=============================================================================="
    
    # Build command
    CMD="python $ABLATION_SCRIPT"
    CMD="$CMD --checkpoint-dir '$CHECKPOINT_DIR'"
    CMD="$CMD --output-dir '$OUTPUT_DIR'"
    CMD="$CMD --device '$DEVICE'"
    CMD="$CMD --n-envs $N_ENVS"
    CMD="$CMD --bfn-steps '$BFN_STEPS'"
    CMD="$CMD --diffusion-steps '$DIFFUSION_STEPS'"
    
    log "Running: $CMD"
    
    # Run with timeout (optional, in seconds)
    if [ -n "${TIMEOUT:-}" ]; then
        timeout "$TIMEOUT" bash -c "$CMD" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    fi
    
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: Ablation study failed with exit code $EXIT_CODE"
        
        # Cleanup on failure
        if [ "$CLEANUP_CHECKPOINTS" = "true" ]; then
            cleanup_temp_files
        fi
        
        exit $EXIT_CODE
    fi
    
    log "=============================================================================="
    log "ABLATION STUDY COMPLETE"
    log "=============================================================================="
    
    # Post-processing
    log "Post-processing results..."
    
    # Generate plots and tables (if not already done by script)
    if [ -f "$OUTPUT_DIR/ablation_results.json" ]; then
        log "Generating plots and tables..."
        python "$ABLATION_SCRIPT" \
            --plot-only \
            --results-file "$OUTPUT_DIR/ablation_results.json" \
            2>&1 | tee -a "$LOG_FILE" || log "WARNING: Plot generation had errors"
    fi
    
    # Compress output
    compress_output
    
    # Cleanup
    if [ "$CLEANUP_CHECKPOINTS" = "true" ]; then
        log "Cleaning up checkpoints (CLEANUP_CHECKPOINTS=true)..."
        # Don't actually delete checkpoints, just clean temp files
        cleanup_temp_files
    else
        cleanup_temp_files
    fi
    
    # Final disk space check
    check_disk_space "$OUTPUT_DIR" || log "WARNING: Disk space low after completion"
    
    # Summary
    log "=============================================================================="
    log "SUMMARY"
    log "=============================================================================="
    log "Output directory: $OUTPUT_DIR"
    
    if [ "$COMPRESS_OUTPUT" = "true" ]; then
        if [ -f "$OUTPUT_DIR/ablation_results.json.gz" ]; then
            SIZE=$(du -h "$OUTPUT_DIR/ablation_results.json.gz" | cut -f1)
            log "Results file: ablation_results.json.gz ($SIZE)"
        fi
    else
        if [ -f "$OUTPUT_DIR/ablation_results.json" ]; then
            SIZE=$(du -h "$OUTPUT_DIR/ablation_results.json" | cut -f1)
            log "Results file: ablation_results.json ($SIZE)"
        fi
    fi
    
    if [ -d "$OUTPUT_DIR/../figures/tables" ]; then
        log "Tables: figures/tables/table_*.tex"
    fi
    
    if [ -d "$OUTPUT_DIR/../figures/publication" ]; then
        log "Figures: figures/publication/fig_*.pdf"
    fi
    
    log "Log file: $LOG_FILE"
    log "=============================================================================="
    log "SUCCESS: Ablation study complete!"
    log "=============================================================================="
    
    return 0
}

# =============================================================================
# SLURM JOB SCRIPT GENERATION (if running as SLURM job)
# =============================================================================

generate_slurm_script() {
    cat > "${1:-run_ablation_slurm.sh}" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=ablation_bfn
#SBATCH --output=ablation_%j.out
#SBATCH --error=ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate environment
# conda activate your_env

# Set variables
export CHECKPOINT_DIR="cluster_checkpoints/benchmarkresults"
export OUTPUT_DIR="results/ablation"
export DEVICE="cuda"
export N_ENVS=50
export BFN_STEPS="5,10,15,20,30,50"
export DIFFUSION_STEPS="10,20,30,50,75,100"
export COMPRESS_OUTPUT="true"
export CLEANUP_CHECKPOINTS="false"

# Run ablation script
bash scripts/run_ablation_cluster.sh
SLURM_EOF
    chmod +x "${1:-run_ablation_slurm.sh}"
    echo "Generated SLURM script: ${1:-run_ablation_slurm.sh}"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# If script is called with --generate-slurm, create SLURM script
if [ "${1:-}" = "--generate-slurm" ]; then
    generate_slurm_script "${2:-run_ablation_slurm.sh}"
    exit 0
fi

# Run main function
main "$@"
