#!/bin/bash
# =============================================================================
# Prepare Ablation Study from Benchmark Outputs
# 
# This script organizes checkpoints from outputs/ (created by run_pusht_benchmark.sh)
# into the format expected by the ablation study.
#
# Usage:
#   bash scripts/prepare_ablation_from_outputs.sh
#   bash scripts/prepare_ablation_from_outputs.sh --outputs-dir /custom/path
# =============================================================================

set -e

OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
TARGET_DIR="${TARGET_DIR:-cluster_checkpoints/benchmarkresults}"
LINK_MODE="${LINK_MODE:-symlink}"  # symlink, copy, or move
DATE_FILTER="${DATE_FILTER:-}"  # Optional: filter by date (e.g., "2025.12.24")

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

find_latest_benchmark() {
    # Find the most recent benchmark run in outputs/
    local latest_date=""
    local latest_time=""
    
    for date_dir in "$OUTPUTS_DIR"/*; do
        if [ ! -d "$date_dir" ]; then
            continue
        fi
        
        local date_name=$(basename "$date_dir")
        # Check if it looks like a date (YYYY.MM.DD)
        if [[ "$date_name" =~ ^[0-9]{4}\.[0-9]{2}\.[0-9]{2}$ ]]; then
            # Check for benchmark runs
            for run_dir in "$date_dir"/*_seed*; do
                if [ -d "$run_dir" ] && [ -d "$run_dir/checkpoints" ]; then
                    if [ -z "$latest_date" ] || [ "$date_name" \> "$latest_date" ]; then
                        latest_date="$date_name"
                    fi
                fi
            done
        fi
    done
    
    echo "$latest_date"
}

organize_checkpoints() {
    log "=============================================================================="
    log "PREPARING ABLATION FROM BENCHMARK OUTPUTS"
    log "=============================================================================="
    log "Source: $OUTPUTS_DIR"
    log "Target: $TARGET_DIR"
    log "Mode: $LINK_MODE"
    
    if [ ! -d "$OUTPUTS_DIR" ]; then
        log "ERROR: Outputs directory not found: $OUTPUTS_DIR"
        log "Run your benchmark first: sbatch scripts/run_pusht_benchmark.sh"
        exit 1
    fi
    
    # Determine which date to use
    local use_date="$DATE_FILTER"
    if [ -z "$use_date" ]; then
        use_date=$(find_latest_benchmark)
        if [ -z "$use_date" ]; then
            log "ERROR: No benchmark runs found in $OUTPUTS_DIR"
            log "Expected structure: $OUTPUTS_DIR/YYYY.MM.DD/HH.MM.SS_method_seed*/checkpoints/*.ckpt"
            exit 1
        fi
        log "Using latest benchmark date: $use_date"
    else
        log "Using specified date: $use_date"
    fi
    
    local source_date_dir="$OUTPUTS_DIR/$use_date"
    if [ ! -d "$source_date_dir" ]; then
        log "ERROR: Date directory not found: $source_date_dir"
        exit 1
    fi
    
    # Create target structure
    local target_date_dir="$TARGET_DIR/$use_date"
    mkdir -p "$target_date_dir"
    
    local organized=0
    
    # Process each run directory
    for run_dir in "$source_date_dir"/*_seed*; do
        if [ ! -d "$run_dir" ]; then
            continue
        fi
        
        local run_name=$(basename "$run_dir")
        
        # Check if it's a benchmark run (bfn_seed* or diffusion_seed*)
        if [[ ! "$run_name" =~ (bfn|diffusion)_seed[0-9]+ ]]; then
            continue
        fi
        
        local target_run_dir="$target_date_dir/$run_name"
        local target_ckpt_dir="$target_run_dir/checkpoints"
        
        # Check if checkpoints exist
        local source_ckpt_dir="$run_dir/checkpoints"
        if [ ! -d "$source_ckpt_dir" ]; then
            log "Skipping $run_name (no checkpoints directory)"
            continue
        fi
        
        local ckpt_count=$(find "$source_ckpt_dir" -name "*.ckpt" 2>/dev/null | wc -l)
        if [ "$ckpt_count" -eq 0 ]; then
            log "Skipping $run_name (no checkpoint files)"
            continue
        fi
        
        log "Processing $run_name ($ckpt_count checkpoints)..."
        
        # Create target directory
        mkdir -p "$target_ckpt_dir"
        
        # Link/copy/move checkpoints
        for ckpt in "$source_ckpt_dir"/*.ckpt; do
            if [ ! -f "$ckpt" ]; then
                continue
            fi
            
            local ckpt_name=$(basename "$ckpt")
            local target_ckpt="$target_ckpt_dir/$ckpt_name"
            
            if [ "$LINK_MODE" = "symlink" ]; then
                if [ ! -e "$target_ckpt" ]; then
                    ln -s "$(realpath "$ckpt")" "$target_ckpt" 2>/dev/null || \
                    ln -s "$ckpt" "$target_ckpt"
                    organized=$((organized + 1))
                fi
            elif [ "$LINK_MODE" = "copy" ]; then
                if [ ! -e "$target_ckpt" ]; then
                    cp "$ckpt" "$target_ckpt"
                    organized=$((organized + 1))
                fi
            elif [ "$LINK_MODE" = "move" ]; then
                if [ ! -e "$target_ckpt" ]; then
                    mv "$ckpt" "$target_ckpt"
                    organized=$((organized + 1))
                fi
            fi
        done
        
        # Also link/copy logs.json.txt if it exists
        if [ -f "$run_dir/logs.json.txt" ]; then
            if [ "$LINK_MODE" = "symlink" ]; then
                ln -sf "$(realpath "$run_dir/logs.json.txt")" "$target_run_dir/logs.json.txt" 2>/dev/null || \
                ln -sf "$run_dir/logs.json.txt" "$target_run_dir/logs.json.txt"
            elif [ "$LINK_MODE" = "copy" ]; then
                cp "$run_dir/logs.json.txt" "$target_run_dir/logs.json.txt" 2>/dev/null || true
            elif [ "$LINK_MODE" = "move" ]; then
                mv "$run_dir/logs.json.txt" "$target_run_dir/logs.json.txt" 2>/dev/null || true
            fi
        fi
        
        log "  ✓ Organized $run_name"
    done
    
    if [ "$organized" -gt 0 ]; then
        log ""
        log "=============================================================================="
        log "SUCCESS: Organized $organized checkpoints"
        log "=============================================================================="
        log "Checkpoints ready at: $TARGET_DIR"
        log ""
        log "Now you can run:"
        log "  bash scripts/run_ablation_cluster.sh"
        log ""
        
        # Show summary
        local bfn_count=$(find "$target_date_dir" -path "*/bfn_seed*/checkpoints/*.ckpt" 2>/dev/null | wc -l)
        local diff_count=$(find "$target_date_dir" -path "*/diffusion_seed*/checkpoints/*.ckpt" 2>/dev/null | wc -l)
        log "Summary:"
        log "  BFN checkpoints: $bfn_count"
        log "  Diffusion checkpoints: $diff_count"
    else
        log "ERROR: No checkpoints were organized"
        log "Check that your benchmark has completed and checkpoints exist"
        exit 1
    fi
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --outputs-dir)
                OUTPUTS_DIR="$2"
                shift 2
                ;;
            --target-dir)
                TARGET_DIR="$2"
                shift 2
                ;;
            --copy)
                LINK_MODE="copy"
                shift
                ;;
            --move)
                LINK_MODE="move"
                shift
                ;;
            --symlink)
                LINK_MODE="symlink"
                shift
                ;;
            --date)
                DATE_FILTER="$2"
                shift 2
                ;;
            --help|-h)
                cat << EOF
Usage: $0 [OPTIONS]

Organize checkpoints from benchmark outputs/ into format for ablation study.

Options:
    --outputs-dir DIR    Source directory (default: outputs)
    --target-dir DIR     Target directory (default: cluster_checkpoints/benchmarkresults)
    --symlink            Create symlinks (default, saves space)
    --copy               Copy files (uses more space)
    --move               Move files (saves space but removes from source)
    --date YYYY.MM.DD    Use specific date (default: latest)

Examples:
    # Use latest benchmark run
    bash scripts/prepare_ablation_from_outputs.sh
    
    # Use specific date
    bash scripts/prepare_ablation_from_outputs.sh --date 2025.12.24
    
    # Copy instead of symlink (if you need to delete outputs/)
    bash scripts/prepare_ablation_from_outputs.sh --copy
EOF
                exit 0
                ;;
            *)
                log "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    organize_checkpoints
}

main "$@"
