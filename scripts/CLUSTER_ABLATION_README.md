# Cluster Ablation Study - Disk Space Efficient

This document explains how to run the ablation study on your cluster while managing disk space efficiently.

## Quick Start

### Option 1: Standard Script (Recommended)
```bash
bash scripts/run_ablation_cluster.sh
```

### Option 2: Space-Efficient Script (Minimal Disk Usage)
```bash
bash scripts/run_ablation_space_efficient.sh
```

### Option 3: SLURM Job
```bash
# Generate SLURM script
bash scripts/run_ablation_cluster.sh --generate-slurm

# Submit job
sbatch run_ablation_slurm.sh
```

## Environment Variables

Configure the ablation study using environment variables:

```bash
# Required
export CHECKPOINT_DIR="cluster_checkpoints/benchmarkresults"
export OUTPUT_DIR="results/ablation"
export DEVICE="cuda"  # or "cpu", "mps"

# Optional - Disk management
export COMPRESS_OUTPUT="true"        # Compress JSON results (saves ~70% space)
export CLEANUP_CHECKPOINTS="false"   # Don't delete checkpoints after loading
export USE_TMPFS="true"              # Use /dev/shm for temp files (if available)
export MAX_DISK_USAGE_GB="10"        # Warning threshold

# Optional - Evaluation settings
export N_ENVS="50"                   # Number of environments per evaluation
export BFN_STEPS="5,10,15,20,30,50"
export DIFFUSION_STEPS="10,20,30,50,75,100"
```

## Disk Space Management

### Automatic Features

1. **Compression**: Results JSON is automatically compressed (`.gz`)
2. **Cleanup**: Temporary files are cleaned after each run
3. **Space Checking**: Script warns if disk space is low
4. **Tmpfs Support**: Uses RAM for temporary files if available

### Manual Space Saving Tips

1. **Use Space-Efficient Script**:
   ```bash
   # Processes one evaluation at a time, cleans up immediately
   bash scripts/run_ablation_space_efficient.sh
   ```

2. **Reduce Number of Steps**:
   ```bash
   # Only test key configurations
   export BFN_STEPS="10,20"
   export DIFFUSION_STEPS="20,50,100"
   ```

3. **Reduce Environments**:
   ```bash
   # Fewer evaluations per configuration
   export N_ENVS="25"  # Instead of 50
   ```

4. **Use Tmpfs (if available)**:
   ```bash
   export USE_TMPFS="true"
   # Uses /dev/shm (RAM disk) for temporary files
   ```

5. **Compress Output**:
   ```bash
   export COMPRESS_OUTPUT="true"
   # Automatically gzips results JSON
   ```

## SLURM Job Configuration

### Generate SLURM Script
```bash
bash scripts/run_ablation_cluster.sh --generate-slurm my_job.sh
```

### Edit SLURM Script
Edit `run_ablation_slurm.sh` (or your generated script) to match your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=ablation_bfn
#SBATCH --output=ablation_%j.out
#SBATCH --error=ablation_%j.err
#SBATCH --time=24:00:00      # Adjust based on your needs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G            # Adjust based on your model size
#SBATCH --gres=gpu:1         # Request GPU

# Your cluster-specific module loads
# module load python/3.9
# module load cuda/11.8

# Activate environment
conda activate your_env

# Run script
bash scripts/run_ablation_cluster.sh
```

### Submit Job
```bash
sbatch run_ablation_slurm.sh
```

## Monitoring Progress

### Check Logs
```bash
# Real-time log monitoring
tail -f results/ablation/ablation_run.log

# Or if running as SLURM job
tail -f ablation_<job_id>.out
```

### Check Disk Space
```bash
# In another terminal
watch -n 60 'df -h results/ablation'
```

### Check Progress
```bash
# Count completed evaluations
ls results/ablation/*.json 2>/dev/null | wc -l
```

## Expected Disk Usage

### Space-Efficient Mode
- **Per evaluation**: ~1-5 MB (JSON result)
- **Total for all configs**: ~50-200 MB
- **With compression**: ~15-60 MB

### Standard Mode
- **Intermediate files**: ~100-500 MB
- **Results**: ~50-200 MB
- **Plots/tables**: ~10-50 MB
- **Total**: ~200-750 MB

### With Compression
- **Compressed results**: ~15-60 MB
- **Total**: ~100-200 MB

## Troubleshooting

### Out of Disk Space

1. **Clean up existing results**:
   ```bash
   rm -rf results/ablation/*.json
   # Keep only aggregated results
   ```

2. **Use space-efficient script**:
   ```bash
   bash scripts/run_ablation_space_efficient.sh
   ```

3. **Process fewer configurations**:
   ```bash
   export BFN_STEPS="20"  # Only default
   export DIFFUSION_STEPS="100"  # Only default
   ```

4. **Use tmpfs**:
   ```bash
   export USE_TMPFS="true"
   ```

### Checkpoint Corruption

If you see checkpoint loading errors:

1. **Verify checkpoints exist**:
   ```bash
   ls -lh cluster_checkpoints/benchmarkresults/*/bfn_seed*/checkpoints/*.ckpt
   ```

2. **Check checkpoint sizes** (should be ~200MB):
   ```bash
   du -h cluster_checkpoints/benchmarkresults/*/bfn_seed*/checkpoints/*.ckpt
   ```

3. **Try loading one manually**:
   ```python
   import torch
   ckpt = torch.load("path/to/checkpoint.ckpt", map_location='cpu')
   print(ckpt.keys())
   ```

### Job Timeout

If your job times out:

1. **Increase time limit** in SLURM script:
   ```bash
   #SBATCH --time=48:00:00  # Increase from 24 hours
   ```

2. **Reduce number of evaluations**:
   ```bash
   export N_ENVS="25"  # Fewer envs per config
   ```

3. **Run in batches**:
   ```bash
   # Run BFN first
   export BFN_STEPS="5,10,15,20,30,50"
   export DIFFUSION_STEPS=""  # Skip for now
   
   # Then run Diffusion separately
   export BFN_STEPS=""
   export DIFFUSION_STEPS="10,20,30,50,75,100"
   ```

## Resuming Failed Runs

Both scripts automatically skip already-completed evaluations:

```bash
# If script fails, just re-run it
bash scripts/run_ablation_cluster.sh

# It will skip existing results and continue from where it stopped
```

For space-efficient script:
```bash
# Individual JSON files are preserved
# Re-run will skip existing files
bash scripts/run_ablation_space_efficient.sh
```

## Output Files

After completion, you'll have:

```
results/ablation/
├── ablation_run.log              # Full execution log
├── ablation_results.json         # Aggregated results (or .gz if compressed)
└── [individual result files]     # If space-efficient mode

figures/
├── tables/
│   ├── table_comprehensive_ablation.tex
│   └── table_ablation_summary.tex
└── publication/
    ├── fig_ablation_score_vs_steps.pdf
    ├── fig_ablation_pareto.pdf
    └── fig_ablation_combined.pdf
```

## Best Practices

1. **Start with small test**: Run with fewer steps first
2. **Monitor disk space**: Check periodically during long runs
3. **Use compression**: Always set `COMPRESS_OUTPUT=true`
4. **Clean up as you go**: Use space-efficient script for large runs
5. **Check logs**: Monitor for errors early
6. **Resume capability**: Scripts automatically resume from failures

## Example: Minimal Disk Usage Run

```bash
# Minimal configuration for testing
export N_ENVS="10"                 # Fewer evaluations
export BFN_STEPS="10,20"           # Only 2 configs
export DIFFUSION_STEPS="20,100"    # Only 2 configs
export COMPRESS_OUTPUT="true"      # Compress results
export USE_TMPFS="true"            # Use RAM for temp files

bash scripts/run_ablation_space_efficient.sh
```

This will use ~50-100 MB total disk space.
