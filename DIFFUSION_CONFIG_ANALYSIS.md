# Diffusion Config Analysis: Is Minimal Disk Script Enough?

## Issue Found ⚠️

The **diffusion config has more aggressive checkpoint settings** than BFN:

### Diffusion Config Defaults:
```yaml
training:
  checkpoint_every: 50  # Saves every 50 epochs (more frequent!)

checkpoint:
  topk:
    k: 5  # Keeps top 5 checkpoints (not 1!)
  save_last_ckpt: true  # Also saves last checkpoint!
```

### BFN Config Defaults:
```yaml
training:
  checkpoint_every: 100  # Saves every 100 epochs

checkpoint:
  topk:
    k: 3  # Keeps top 3 checkpoints
  save_last_ckpt: false  # Doesn't save last
```

## Good News ✅

**The minimal disk script DOES override these settings!**

Looking at the script:
```bash
# Diffusion training command
training.checkpoint_every=100 \        # Overrides 50 → 100
checkpoint.save_last_ckpt=false \     # Overrides true → false
checkpoint.topk.k=1 \                  # Overrides 5 → 1
```

**So the script correctly overrides the diffusion config defaults.**

## Verification

The script overrides:
- ✅ `checkpoint_every: 50` → `100` (less frequent)
- ✅ `topk.k: 5` → `1` (only best checkpoint)
- ✅ `save_last_ckpt: true` → `false` (no last checkpoint)

## Result

**Both BFN and Diffusion will have:**
- Checkpoints at epochs 100, 200, 300
- Only top-1 checkpoint kept per run
- No last checkpoint saved
- **Same disk usage for both methods**

## Is It Still Enough? ✅ YES

**For your ablation study:**

1. **You get the best checkpoint per seed per method** ✅
   - Script overrides ensure only top-1 is kept
   - This is exactly what ablation study needs

2. **All 3 seeds (42, 43, 44)** ✅
   - Script runs all seeds for both methods

3. **Total checkpoints: 6** ✅
   - 3 seeds × 2 methods = 6 checkpoints
   - Each is the best checkpoint for that seed

## Potential Issue (Minor)

**If the Hydra override doesn't work for some reason**, the diffusion config would:
- Save checkpoints every 50 epochs (instead of 100)
- Keep top 5 checkpoints (instead of 1)
- Save last checkpoint (extra file)

**But this is unlikely** - Hydra overrides should work.

## Recommendation

**The script is still enough**, but to be extra safe:

1. **Verify the overrides work** - Check after first run that only 1 checkpoint is saved
2. **Or update the diffusion config directly** - Change defaults in the config file

### Option: Update Config File (More Reliable)

You could update `config/benchmark_diffusion_pusht.yaml`:

```yaml
training:
  checkpoint_every: 100  # Change from 50 to 100

checkpoint:
  topk:
    k: 1  # Change from 5 to 1
  save_last_ckpt: false  # Change from true to false
```

This way, even if Hydra overrides fail, you're safe.

## Summary

**Is it still enough? ✅ YES**

- Script correctly overrides diffusion config defaults
- Both methods will have same checkpoint behavior
- You'll get exactly what you need for ablation study
- 6 checkpoints total (one best per seed per method)

**Optional safety measure:** Update the diffusion config file directly to match BFN config defaults, but the script overrides should work fine.

