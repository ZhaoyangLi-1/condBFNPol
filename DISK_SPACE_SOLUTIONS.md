# Disk Space Solutions for Benchmark Training

You're getting disk space errors when running the benchmark. Here are several solutions:

---

## Solution 1: Use Minimal Disk Script (Recommended - ALL SEEDS)

**Use the minimal disk script that runs ALL seeds (42, 43, 44) with maximum disk savings:**

```bash
bash scripts/run_pusht_benchmark_minimal_disk.sh
```

**What it does:**
- ✅ **Runs ALL seeds: 42, 43, 44** (as in original implementation)
- ✅ Uses scratch directory (if available) instead of home
- ✅ Checkpoints at epochs 100, 200, 300 (standard frequency)
- ✅ Keeps only top-1 checkpoint per run (best test score)
- ✅ Uses WandB offline mode (no syncing during training)
- ✅ Aggressive cleanup before starting
- ✅ Cleans intermediate files after each seed
- ✅ ✅ Monitors disk space after each seed

**Disk usage:**
- 6 runs total (3 seeds × 2 methods)
- ~3 checkpoints per run (epochs 100, 200, 300)
- Only top-1 kept = ~6-18 checkpoints total (much less than original)

---

## Solution 2: Manual Configuration Override

**Run the original script with disk-saving overrides:**

```bash
bash scripts/run_pusht_benchmark.sh \
    --epochs 300
```

Then manually override checkpoint settings:

```bash
# For each seed, run with reduced checkpoints:
python scripts/train_workspace.py \
    --config-name=benchmark_bfn_pusht \
    hydra.run.dir="outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_bfn_seed42" \
    training.checkpoint_every=200 \
    checkpoint.topk.k=1 \
    checkpoint.save_last_ckpt=false \
    logging.mode=offline \
    # ... other args
```

---

## Solution 3: Clean Up Before Running

**Free up space before training:**

```bash
# 1. Clean old outputs (keep only last 2 days)
find outputs/ -type d -name "20*" -mtime +2 -exec rm -rf {} +

# 2. Clean old WandB runs
find ~/condBFNPol/wandb -type d -name "run-*" -mtime +7 -exec rm -rf {} +

# 3. Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 4. Check disk space
df -h ~
```

---

## Solution 4: Use Scratch Directory

**If you have access to a scratch directory with more space:**

```bash
# Set output to scratch directory
export SCRATCH_DIR="/dss/dssfs04/scratch/ge87gob2"
mkdir -p "$SCRATCH_DIR/condBFNPol/outputs"

# Run with custom output directory
bash scripts/run_pusht_benchmark.sh
```

The disk-optimized script automatically detects and uses scratch directories.

---

## Solution 5: Reduce Checkpoint Frequency

**Edit the config files to save checkpoints less frequently:**

**File: `config/benchmark_bfn_pusht.yaml`**
```yaml
training:
  checkpoint_every: 200  # Instead of 100

checkpoint:
  topk:
    k: 1  # Instead of 3 - only keep best checkpoint
  save_last_ckpt: false
```

**File: `config/benchmark_diffusion_pusht.yaml`**
```yaml
training:
  checkpoint_every: 200

checkpoint:
  topk:
    k: 1
  save_last_ckpt: false
```

---

## Solution 6: Train One Seed at a Time

**Run seeds sequentially and clean up between:**

```bash
# Train seed 42
python scripts/train_workspace.py \
    --config-name=benchmark_bfn_pusht \
    training.seed=42 \
    training.checkpoint_every=200 \
    checkpoint.topk.k=1 \
    logging.mode=offline

# Clean up before next seed
find outputs/ -name "*.ckpt" ! -name "*epoch=*" -delete

# Train seed 43
python scripts/train_workspace.py \
    --config-name=benchmark_bfn_pusht \
    training.seed=43 \
    # ... same args
```

---

## Solution 7: Use WandB Offline Mode

**Prevent WandB from syncing large files during training:**

```bash
export WANDB_MODE=offline
bash scripts/run_pusht_benchmark.sh
```

**Sync later when you have space:**
```bash
wandb sync ~/condBFNPol/wandb/run-*
```

---

## Solution 8: Compress Checkpoints (Advanced)

**If you need to keep more checkpoints, compress them:**

Add this to your training script or workspace:
```python
import gzip
import shutil

# After saving checkpoint
with open(ckpt_path, 'rb') as f_in:
    with gzip.open(f"{ckpt_path}.gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(ckpt_path)
```

---

## Solution 9: Check Disk Space Before Starting

**Add this check to your script:**

```bash
# Check available space (need at least 50GB)
AVAILABLE=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE" -lt 50 ]; then
    echo "ERROR: Less than 50GB available. Please free up space."
    exit 1
fi
```

---

## Recommended Approach

**For your situation (need checkpoints for ablation study, ALL seeds 42, 43, 44):**

1. **Use the minimal disk script (runs ALL seeds):**
   ```bash
   bash scripts/run_pusht_benchmark_minimal_disk.sh
   ```

2. **This will:**
   - ✅ Run ALL seeds: 42, 43, 44 (as in original)
   - ✅ Save checkpoints at epochs 100, 200, 300
   - ✅ Keep only top-1 checkpoint per run (best test score)
   - ✅ Use offline WandB mode
   - ✅ Clean up old files automatically
   - ✅ Monitor disk space after each seed

3. **After training completes:**
   - You'll have checkpoints at epochs 100, 200, 300 for each seed
   - Total: ~6-18 checkpoints (much less than original ~36)
   - Enough for ablation study (you can evaluate at different inference steps)

4. **If you still run out of space:**
   - The script will warn you if <30GB available
   - Increase `checkpoint_every` to 150 or 200 in the script
   - Use a different scratch directory with more space
   - Train one method at a time (BFN first, then Diffusion) - but this keeps all seeds

---

## Quick Fix: Minimal Checkpoints

**If you only need checkpoints for ablation, save only at the end:**

```bash
python scripts/train_workspace.py \
    --config-name=benchmark_bfn_pusht \
    training.checkpoint_every=300 \
    checkpoint.topk.k=1 \
    checkpoint.save_last_ckpt=true \
    # ... other args
```

This saves only the final checkpoint, minimizing disk usage.

---

## Check Disk Usage

**Monitor disk usage during training:**

```bash
# In another terminal, watch disk usage:
watch -n 60 'df -h ~ | tail -1'
```

---

## Summary

**Best solution for your case:**
1. Use `run_pusht_benchmark_disk_optimized.sh`
2. It automatically handles disk space management
3. Keeps enough checkpoints for ablation study
4. Uses offline WandB to avoid syncing issues

**If still having issues:**
- Increase `checkpoint_every` to 200
- Reduce `topk.k` to 1
- Train one seed at a time

