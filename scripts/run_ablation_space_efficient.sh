#!/bin/bash
# =============================================================================
# Space-Efficient Ablation Study - Processes one checkpoint at a time
# Minimizes disk usage by loading, evaluating, and immediately saving results
# =============================================================================

set -e
set -u

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-cluster_checkpoints/benchmarkresults}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ablation}"
DEVICE="${DEVICE:-cuda}"
N_ENVS="${N_ENVS:-50}"

# Step configurations
BFN_STEPS=(5 10 15 20 30 50)
DIFFUSION_STEPS=(10 20 30 50 75 100)

# Disk management
MAX_SIZE_MB="${MAX_SIZE_MB:-500}"  # Max size for temp files in MB
CLEANUP_AFTER_EACH="${CLEANUP_AFTER_EACH:-true}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

check_space() {
    local dir="$1"
    local available_kb=$(df "$dir" | tail -1 | awk '{print $4}')
    local available_mb=$((available_kb / 1024))
    log "Available space: ${available_mb}MB"
    if [ "$available_mb" -lt "$MAX_SIZE_MB" ]; then
        log "WARNING: Low disk space"
        return 1
    fi
    return 0
}

cleanup_cuda_cache() {
    if [ "$DEVICE" = "cuda" ]; then
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
}

# Python script to run single evaluation
cat > /tmp/ablation_single.py << 'PYEOF'
import sys
import json
import torch
from pathlib import Path

# Import your evaluation functions
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.comprehensive_ablation_study import (
    load_bfn_policy_from_checkpoint,
    load_diffusion_policy_from_checkpoint,
    evaluate_policy_at_steps,
    measure_inference_time
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["bfn", "diffusion"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    try:
        # Load policy
        if args.method == "bfn":
            policy, cfg = load_bfn_policy_from_checkpoint(args.checkpoint, args.device)
        else:
            policy, cfg = load_diffusion_policy_from_checkpoint(args.checkpoint, args.device)
        
        # Evaluate
        eval_result = evaluate_policy_at_steps(
            policy, args.method, args.steps, args.n_envs, device=args.device
        )
        
        # Measure time
        time_result = measure_inference_time(
            policy, args.method, args.steps, device=args.device
        )
        
        # Combine results
        result = {
            **eval_result,
            **time_result,
            "steps": args.steps,
            "method": args.method,
            "checkpoint": args.checkpoint,
        }
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Cleanup
        del policy
        cleanup_cuda_cache()
        
        print(f"SUCCESS: {args.method} {args.steps} steps")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
PYEOF

run_single_eval() {
    local checkpoint="$1"
    local method="$2"
    local steps="$3"
    local output_file="$4"
    
    log "  Evaluating: $method at $steps steps"
    
    python /tmp/ablation_single.py \
        --checkpoint "$checkpoint" \
        --method "$method" \
        --steps "$steps" \
        --n-envs "$N_ENVS" \
        --device "$DEVICE" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Cleanup CUDA cache
    cleanup_cuda_cache
    
    # Cleanup temp files
    if [ "$CLEANUP_AFTER_EACH" = "true" ]; then
        find "$OUTPUT_DIR" -name "*.tmp" -delete 2>/dev/null || true
        python -c "import gc; gc.collect()" 2>/dev/null || true
    fi
}

aggregate_results() {
    log "Aggregating results..."
    python - << 'PYEOF'
import json
import glob
from pathlib import Path
from collections import defaultdict
import sys

output_dir = Path(sys.argv[1])
results = {"bfn": defaultdict(dict), "diffusion": defaultdict(dict)}

# Collect all result files
for result_file in glob.glob(str(output_dir / "*.json")):
    if "aggregated" in result_file:
        continue
    try:
        with open(result_file) as f:
            data = json.load(f)
            method = data.get("method")
            steps = data.get("steps")
            checkpoint = Path(data.get("checkpoint", "")).parent.parent.name
            
            # Extract seed from checkpoint path
            seed = None
            if "seed" in checkpoint.lower():
                for part in checkpoint.split("_"):
                    if part.startswith("seed"):
                        seed = int(part.replace("seed", ""))
                        break
            
            if seed and method and steps is not None:
                results[method][seed][steps] = data
    except:
        pass

# Save aggregated
with open(output_dir / "ablation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("Aggregated results saved")
PYEOF
}

main() {
    log "=============================================================================="
    log "SPACE-EFFICIENT ABLATION STUDY"
    log "=============================================================================="
    
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/ablation_run.log"
    log "Starting space-efficient ablation study"
    
    check_space "$OUTPUT_DIR" || {
        log "ERROR: Insufficient disk space"
        exit 1
    }
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_ROOT"
    
    # Find all checkpoints
    log "Finding checkpoints..."
    
    # BFN checkpoints
    for ckpt_dir in "$CHECKPOINT_DIR"/*/bfn_seed*/checkpoints/*.ckpt; do
        if [ ! -f "$ckpt_dir" ]; then
            continue
        fi
        
        # Extract seed
        seed=$(echo "$ckpt_dir" | grep -oP 'seed\d+' | grep -oP '\d+')
        if [ -z "$seed" ]; then
            continue
        fi
        
        log "Processing BFN seed $seed: $(basename $ckpt_dir)"
        
        for steps in "${BFN_STEPS[@]}"; do
            output_file="$OUTPUT_DIR/bfn_seed${seed}_steps${steps}.json"
            
            if [ -f "$output_file" ]; then
                log "  Skipping (already exists): $output_file"
                continue
            fi
            
            run_single_eval "$ckpt_dir" "bfn" "$steps" "$output_file"
            check_space "$OUTPUT_DIR" || log "WARNING: Disk space low"
        done
    done
    
    # Diffusion checkpoints
    for ckpt_dir in "$CHECKPOINT_DIR"/*/diffusion_seed*/checkpoints/*.ckpt; do
        if [ ! -f "$ckpt_dir" ]; then
            continue
        fi
        
        seed=$(echo "$ckpt_dir" | grep -oP 'seed\d+' | grep -oP '\d+')
        if [ -z "$seed" ]; then
            continue
        fi
        
        log "Processing Diffusion seed $seed: $(basename $ckpt_dir)"
        
        for steps in "${DIFFUSION_STEPS[@]}"; do
            output_file="$OUTPUT_DIR/diffusion_seed${seed}_steps${steps}.json"
            
            if [ -f "$output_file" ]; then
                log "  Skipping (already exists): $output_file"
                continue
            fi
            
            run_single_eval "$ckpt_dir" "diffusion" "$steps" "$output_file"
            check_space "$OUTPUT_DIR" || log "WARNING: Disk space low"
        done
    done
    
    # Aggregate results
    aggregate_results "$OUTPUT_DIR"
    
    # Generate plots
    log "Generating plots and tables..."
    python scripts/comprehensive_ablation_study.py \
        --plot-only \
        --results-file "$OUTPUT_DIR/ablation_results.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Cleanup individual files (keep aggregated)
    if [ "$CLEANUP_AFTER_EACH" = "true" ]; then
        log "Cleaning up individual result files..."
        rm -f "$OUTPUT_DIR"/*_seed*_steps*.json
    fi
    
    log "=============================================================================="
    log "COMPLETE"
    log "=============================================================================="
}

main "$@"
