# Ablation Study Requirements for Thesis

## What Your Ablation Study Needs

### Minimum Requirements ✅

**For a complete thesis ablation study, you need:**

1. **One checkpoint per seed per method:**
   - BFN: seeds 42, 43, 44 = **3 checkpoints**
   - Diffusion: seeds 42, 43, 44 = **3 checkpoints**
   - **Total: 6 checkpoints** (minimum)

2. **The best checkpoint per seed:**
   - The ablation study automatically finds the checkpoint with highest `test_mean_score`
   - It uses: `epoch=XXXX-test_mean_score=0.XXX.ckpt`
   - **The minimal disk script saves top-1 checkpoint = the best one** ✅

### What the Ablation Study Does

The `comprehensive_ablation_study.py` script:

1. **Finds checkpoints automatically:**
   ```python
   # Looks for: epoch=XXXX-test_mean_score=0.XXX.ckpt
   # Picks the one with highest score per seed
   ```

2. **Evaluates at different inference steps:**
   - BFN: [5, 10, 15, 20, 30, 50] steps
   - Diffusion: [10, 20, 30, 50, 75, 100] steps

3. **Generates thesis figures:**
   - Score vs inference steps
   - Time vs inference steps  
   - Pareto frontier
   - Seed robustness analysis

### Is the Minimal Disk Script Enough? ✅ YES

**The minimal disk script provides:**

- ✅ **All 3 seeds (42, 43, 44)** - Required for seed robustness
- ✅ **Both methods (BFN + Diffusion)** - Required for comparison
- ✅ **Top-1 checkpoint per run** - This is the **best checkpoint** (highest test score)
- ✅ **Checkpoints at epochs 100, 200, 300** - The best one among these is kept

**Result:**
- 6 runs total (3 seeds × 2 methods)
- Each run keeps only the **best checkpoint** (top-1)
- Total: **6 checkpoints** (exactly what you need!)

### What You'll Get

After running the minimal disk script:

```
outputs/
├── 2025.01.XX/
│   ├── HH.MM.SS_bfn_seed42/
│   │   └── checkpoints/
│   │       └── epoch=0200-test_mean_score=0.912.ckpt  ← Best checkpoint
│   ├── HH.MM.SS_bfn_seed43/
│   │   └── checkpoints/
│   │       └── epoch=0300-test_mean_score=0.901.ckpt  ← Best checkpoint
│   ├── HH.MM.SS_bfn_seed44/
│   │   └── checkpoints/
│   │       └── epoch=0200-test_mean_score=0.895.ckpt  ← Best checkpoint
│   ├── HH.MM.SS_diffusion_seed42/
│   │   └── checkpoints/
│   │       └── epoch=0200-test_mean_score=0.950.ckpt  ← Best checkpoint
│   ├── HH.MM.SS_diffusion_seed43/
│   │   └── checkpoints/
│   │       └── epoch=0300-test_mean_score=0.945.ckpt  ← Best checkpoint
│   └── HH.MM.SS_diffusion_seed44/
│       └── checkpoints/
│           └── epoch=0200-test_mean_score=0.948.ckpt  ← Best checkpoint
```

**Total: 6 checkpoints** (one best per seed per method)

### Running the Ablation Study

Once you have the checkpoints:

```bash
# Run full ablation study
python scripts/comprehensive_ablation_study.py \
    --checkpoint-dir outputs \
    --seeds 42,43,44 \
    --n-envs 50 \
    --device cuda
```

This will:
1. ✅ Find all 6 checkpoints automatically
2. ✅ Evaluate each at different inference steps
3. ✅ Generate thesis figures and tables
4. ✅ Show seed robustness (all 3 seeds)

### Thesis Requirements Checklist

For a complete thesis ablation study, you need:

- [x] **Multiple seeds (42, 43, 44)** - ✅ Minimal disk script provides this
- [x] **Both methods (BFN + Diffusion)** - ✅ Minimal disk script provides this  
- [x] **Best checkpoint per seed** - ✅ Top-1 = best checkpoint
- [x] **Different inference steps** - ✅ Ablation script evaluates this
- [x] **Statistical robustness** - ✅ 3 seeds = enough for statistics

### Optional: Multiple Epochs (Not Required)

**You DON'T need multiple epochs for the ablation study.**

The ablation study:
- Only uses the **best checkpoint** per seed
- Evaluates it at different inference steps
- Doesn't compare different training epochs

**However**, if you want to show training progress in your thesis:
- You could keep top-2 checkpoints instead of top-1
- But this is **optional**, not required for ablation

### Recommendation

**✅ YES, the minimal disk script is enough for your thesis ablation study.**

**Why:**
1. Provides all 3 seeds (required for robustness)
2. Provides both methods (required for comparison)
3. Keeps the best checkpoint (what ablation study needs)
4. Saves significant disk space (~70% reduction)

**What you'll be able to show in your thesis:**
- ✅ Inference steps ablation (performance vs steps)
- ✅ Efficiency comparison (BFN vs Diffusion)
- ✅ Seed robustness (variability across seeds)
- ✅ Statistical significance (3 seeds = enough)
- ✅ Pareto frontier (speed vs performance)

### If You Want Extra Safety

If you want to be extra safe, you can modify the script to keep top-2 checkpoints:

```bash
# In the script, change:
checkpoint.topk.k=2  # Instead of 1
```

This gives you:
- Best checkpoint (for ablation)
- Second-best checkpoint (backup/optional analysis)

But **top-1 is sufficient** for the ablation study.

---

## Summary

**Question: Is the minimal disk script enough for thesis ablation?**

**Answer: ✅ YES, absolutely!**

- Provides all required checkpoints (6 total)
- All 3 seeds for robustness
- Best checkpoint per seed (what ablation needs)
- Saves 70% disk space
- Complete ablation study possible

**You're good to go!** 🎓

