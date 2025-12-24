#!/bin/bash
# Find checkpoints in outputs directory and link to cluster_checkpoints structure
# Useful when checkpoints are in outputs/ from training but ablation expects cluster_checkpoints/

set -e

OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
TARGET_DIR="${TARGET_DIR:-cluster_checkpoints/benchmarkresults}"
LINK_MODE="${LINK_MODE:-symlink}"  # symlink, copy, or move

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

find_and_organize() {
    log "Searching for checkpoints in: $OUTPUTS_DIR"
    
    # Create target directory structure
    mkdir -p "$TARGET_DIR"
    
    local found=0
    
    # Search for checkpoints in outputs
    for ckpt_file in $(find "$OUTPUTS_DIR" -name "*.ckpt" 2>/dev/null); do
        # Extract run info from path
        # Pattern: outputs/2025.12.24/00.23.32_bfn_seed42/checkpoints/epoch=0200-test_mean_score=0.877.ckpt
        local rel_path=$(echo "$ckpt_file" | sed "s|^$OUTPUTS_DIR/||")
        local date_dir=$(echo "$rel_path" | cut -d'/' -f1)
        local run_dir=$(echo "$rel_path" | cut -d'/' -f2)
        local ckpt_name=$(basename "$ckpt_file")
        
        if [ -z "$date_dir" ] || [ -z "$run_dir" ]; then
            continue
        fi
        
        # Create target structure
        local target_date_dir="$TARGET_DIR/$date_dir"
        local target_run_dir="$target_date_dir/$run_dir"
        local target_ckpt_dir="$target_run_dir/checkpoints"
        
        mkdir -p "$target_ckpt_dir"
        
        local target_ckpt="$target_ckpt_dir/$ckpt_name"
        
        if [ "$LINK_MODE" = "symlink" ]; then
            if [ ! -e "$target_ckpt" ]; then
                ln -s "$(realpath "$ckpt_file")" "$target_ckpt"
                log "Linked: $run_dir/$ckpt_name"
                found=1
            fi
        elif [ "$LINK_MODE" = "copy" ]; then
            if [ ! -e "$target_ckpt" ]; then
                cp "$ckpt_file" "$target_ckpt"
                log "Copied: $run_dir/$ckpt_name"
                found=1
            fi
        elif [ "$LINK_MODE" = "move" ]; then
            if [ ! -e "$target_ckpt" ]; then
                mv "$ckpt_file" "$target_ckpt"
                log "Moved: $run_dir/$ckpt_name"
                found=1
            fi
        fi
        
        # Also copy/link logs if they exist
        local log_file=$(dirname "$ckpt_file")/../logs.json.txt
        if [ -f "$log_file" ]; then
            local target_log="$target_run_dir/logs.json.txt"
            if [ ! -e "$target_log" ]; then
                if [ "$LINK_MODE" = "symlink" ]; then
                    ln -s "$(realpath "$log_file")" "$target_log"
                elif [ "$LINK_MODE" = "copy" ]; then
                    cp "$log_file" "$target_log"
                elif [ "$LINK_MODE" = "move" ]; then
                    mv "$log_file" "$target_log"
                fi
            fi
        fi
    done
    
    if [ "$found" -eq 1 ]; then
        log ""
        log "✓ Checkpoints organized in: $TARGET_DIR"
        log "You can now run: bash scripts/run_ablation_cluster.sh"
    else
        log "No checkpoints found in $OUTPUTS_DIR"
        log ""
        log "Searching in other locations..."
        find_checkpoints_anywhere
    fi
}

find_checkpoints_anywhere() {
    local count=0
    
    # Search more broadly
    for pattern in "**/*.ckpt" "*/*.ckpt" "*/checkpoints/*.ckpt"; do
        for ckpt in $(find . -name "*.ckpt" -type f 2>/dev/null | head -20); do
            if [ -f "$ckpt" ]; then
                log "Found: $ckpt"
                count=$((count + 1))
            fi
        done
        [ $count -gt 0 ] && break
    done
    
    if [ $count -eq 0 ]; then
        log "No checkpoints found anywhere."
        log ""
        log "You may need to:"
        log "  1. Train models first"
        log "  2. Transfer checkpoints from another location"
        log "  3. Use --setup-training to prepare training scripts"
    fi
}

main() {
    log "=============================================================================="
    log "FIND AND ORGANIZE CHECKPOINTS"
    log "=============================================================================="
    
    if [ -d "$OUTPUTS_DIR" ]; then
        find_and_organize
    else
        log "Outputs directory not found: $OUTPUTS_DIR"
        find_checkpoints_anywhere
    fi
}

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
        *)
            log "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
