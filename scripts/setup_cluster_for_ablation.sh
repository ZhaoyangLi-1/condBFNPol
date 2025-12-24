#!/bin/bash
# Quick setup script for running ablation study on cluster
# Handles the case when checkpoints don't exist on cluster

set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

print_menu() {
    cat << 'EOF'
==============================================================================
CLUSTER ABLATION STUDY SETUP
==============================================================================

You don't have checkpoints on the cluster. Choose an option:

1. TRANSFER CHECKPOINTS FROM LOCAL MACHINE (if you have them locally)
   - Requires: Local machine with checkpoints, SSH access to cluster
   - Command: bash scripts/transfer_checkpoints_to_cluster.sh --cluster-host <host>
   - Time: 10-30 minutes (depends on network speed)

2. TRAIN ON CLUSTER (if you need to train new models)
   - Requires: Training data, GPU access
   - Command: bash scripts/transfer_checkpoints_to_cluster.sh --setup-training
   - Then: sbatch scripts/train_ablation_slurm.sh
   - Time: 2-3 days (depending on cluster resources)

3. USE EXISTING CHECKPOINTS IN OUTPUTS (if training already done)
   - If you've already trained but checkpoints are in outputs/
   - Command: bash scripts/find_and_link_checkpoints.sh
   - Then: bash scripts/run_ablation_cluster.sh

4. SPECIFY CUSTOM CHECKPOINT LOCATION
   - If checkpoints are in a different location
   - Command: export CHECKPOINT_DIR=/your/path && bash scripts/run_ablation_cluster.sh

==============================================================================
EOF
}

check_for_checkpoints() {
    log "Searching for checkpoints..."
    
    local found_locations=()
    
    # Check common locations
    for dir in \
        "cluster_checkpoints/benchmarkresults" \
        "outputs" \
        "checkpoints" \
        "results"
    do
        if [ -d "$dir" ]; then
            local count=$(find "$dir" -name "*.ckpt" 2>/dev/null | wc -l)
            if [ "$count" -gt 0 ]; then
                found_locations+=("$dir ($count checkpoints)")
            fi
        fi
    done
    
    # Check with glob patterns
    for pattern in outputs/*/checkpoints cluster_checkpoints/**/checkpoints; do
        if ls $pattern/*.ckpt 1>/dev/null 2>&1; then
            found_locations+=("$pattern")
        fi
    done
    
    if [ ${#found_locations[@]} -gt 0 ]; then
        log "Found checkpoints in:"
        for loc in "${found_locations[@]}"; do
            log "  - $loc"
        done
        return 0
    else
        log "No checkpoints found."
        return 1
    fi
}

main() {
    print_menu
    
    if check_for_checkpoints; then
        log ""
        log "Checkpoints found! You can:"
        log "  1. Organize them: bash scripts/find_and_link_checkpoints.sh"
        log "  2. Or specify location: export CHECKPOINT_DIR=<found_location>"
    else
        log ""
        log "No checkpoints found. Please choose one of the options above."
    fi
}

main "$@"
