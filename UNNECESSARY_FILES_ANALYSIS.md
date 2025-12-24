# Unnecessary Files and Implementations Analysis

## Summary

Your benchmark only uses **BFN** and **Diffusion** policies with U-Net hybrid image architecture. Many other implementations are unused and can be removed or archived.

---

## 🔴 UNNECESSARY FILES (Can be deleted)

### 1. **Unused Policy Implementations**

These policies are NOT used in your benchmark:

- ❌ `policies/conditional_bfn_policy.py` - Conditional BFN (not in benchmark)
- ❌ `policies/conditional_bfn_unet_hybrid_image_policy.py` - Conditional BFN with images (not in benchmark)
- ❌ `policies/guided_bfn_policy.py` - Guided BFN / Flow Matching (not in benchmark)
- ❌ `policies/flow_matching_unet_hybrid_image_policy.py` - Flow Matching (not in benchmark)
- ❌ `policies/bfn_policy.py` - Low-dim BFN (you only use `bfn_unet_hybrid_image_policy.py`)
- ❌ `policies/diffusion_policy.py` - Low-dim Diffusion (you only use `diffusion_unet_hybrid_image_policy.py`)

**Keep only:**
- ✅ `policies/bfn_unet_hybrid_image_policy.py` - Used in benchmark
- ✅ `policies/diffusion_unet_hybrid_image_policy.py` - Used in benchmark
- ✅ `policies/base.py` - Base class (required)

### 2. **Unused Workspaces**

- ❌ `workspaces/train_conditional_bfn_workspace.py` - For conditional BFN (not used)
- ❌ `workspaces/train_guided_bfn_workspace.py` - For guided BFN (not used)

**Keep only:**
- ✅ `workspaces/train_bfn_workspace.py` - Used in benchmark
- ✅ `workspaces/train_diffusion_unet_hybrid_workspace.py` - Used in benchmark
- ✅ `workspaces/base_workspace.py` - Base class (required)

### 3. **Unused Training Scripts**

These are redundant - you use `train_workspace.py` instead:

- ❌ `scripts/train_policy.py` - PyTorch Lightning-based (not used in benchmark)
- ❌ `scripts/train_diffusion_workspace.py` - Duplicate of train_workspace.py
- ❌ `scripts/train_diffusion_pusht.py` - Standalone script (not used)
- ❌ `scripts/train_diffusion_transformer_pusht.py` - Transformer variant (not used)
- ❌ `scripts/train_conditional_bfn.py` - For conditional BFN (not used)
- ❌ `scripts/train_guided_bfn.py` - For guided BFN (not used)
- ❌ `scripts/train_guided_bfn_pusht.py` - Standalone guided BFN (not used)
- ❌ `scripts/standalone.py` - Old standalone script (not used)

**Keep only:**
- ✅ `scripts/train_workspace.py` - Main entry point (used in benchmark)
- ✅ `scripts/train_for_ablation.py` - For ablation studies (if needed)

### 4. **Redundant Shell Scripts**

- ❌ `scripts/run_diffusion_seed44.sh` - **Redundant single-seed script**
  - Does the same thing as `run_pusht_benchmark.sh` but only for seed 44
  - Comment says "# Run Diffusion Policy with seed 44 (failed run)" - was used to re-run a failed training
  - **Recommendation**: Delete or archive - `run_pusht_benchmark.sh` handles all seeds (42, 43, 44) for both methods
  - **Alternative**: Keep as convenience script if you frequently need to re-run individual failed seeds

### 5. **Unused Config Files**

- ❌ `config/train_bfn.yaml` - Low-dim BFN config (not used)
- ❌ `config/train_conditional_bfn.yaml` - Conditional BFN config (not used)
- ❌ `config/train_conditional_bfn_unet_hybrid.yaml` - Conditional BFN image (not used)
- ❌ `config/train_guided_bfn.yaml` - Guided BFN config (not used)
- ❌ `config/train_guided_bfn_unet_hybrid.yaml` - Guided BFN image (not used)
- ❌ `config/train_diffusion_pusht.yaml` - Old diffusion config (not used)
- ❌ `config/train_diffusion_transformer_pusht.yaml` - Transformer config (not used)
- ❌ `config/config_cond_bfn.yaml` - Old conditional BFN config (not used)
- ❌ `config/config_guided.yaml` - Old guided config (not used)
- ❌ `config/config_diffusion.yaml` - Old diffusion config (not used)
- ❌ `config/policy_train.yaml` - PyTorch Lightning config (not used)
- ❌ `config/benchmark_flow_pusht.yaml` - Flow matching benchmark (not used)

**Keep only:**
- ✅ `config/benchmark_bfn_pusht.yaml` - Used in benchmark
- ✅ `config/benchmark_diffusion_pusht.yaml` - Used in benchmark
- ✅ `config/train_bfn_unet_hybrid.yaml` - Alternative BFN config (if needed)
- ✅ `config/train_diffusion_unet_hybrid.yaml` - Alternative diffusion config (if needed)

### 6. **Unused Evaluation Scripts**

- ❌ `scripts/eval_checkpoint.py` - Old evaluation script (not used)
- ❌ `scripts/eval_diffusion.py` - Specific diffusion eval (not used)
- ❌ `scripts/eval_pusht_checkpoint.py` - Old pusht eval (not used)

**Keep:**
- ✅ `scripts/comprehensive_ablation_study.py` - Main evaluation script
- ✅ `scripts/benchmark_pusht.py` - Benchmark runner
- ✅ `scripts/extract_scores_from_logs.py` - Score extraction utility

---

## 🟡 POTENTIALLY UNNECESSARY (Review before deleting)

### 1. **Analysis Scripts (Keep if generating paper figures)**

- `scripts/analyze_benchmark_results.py` - WandB analysis
- `scripts/analyze_local_results.py` - Local checkpoint analysis
- `scripts/analyze_bfn_advantages.py` - BFN advantage analysis
- `scripts/statistical_analysis.py` - Statistical tests
- `scripts/visualize_benchmark.py` - Visualization
- `scripts/analyze_pusht_benchmark.py` - PushT analysis

**Decision:** Keep if you're actively using them for paper figures/tables. Otherwise archive.

### 2. **Utility Scripts**

- `scripts/pusht_video_metrics.py` - Video metrics (keep if needed)
- `scripts/generate_*.py` - Figure generation scripts (keep if generating paper)
- `scripts/debug_loss.py` - Debugging utility (can delete if not debugging)

### 3. **Shell Scripts**

- `scripts/run_diffusion_seed44.sh` - Single seed script (redundant with benchmark script)
- `scripts/run_ablation_*.sh` - Ablation scripts (keep if using)
- `scripts/setup_cluster_for_ablation.sh` - Cluster setup (keep if needed)
- `scripts/test_ablation_local.sh` - Test script (can delete)

---

## 🟢 KEEP (Essential for benchmark)

### Core Files (Required)
- ✅ `scripts/train_workspace.py` - Main training entry point
- ✅ `scripts/run_pusht_benchmark.sh` - Benchmark runner
- ✅ `scripts/comprehensive_ablation_study.py` - Ablation study
- ✅ `scripts/benchmark_pusht.py` - Benchmark utilities
- ✅ `scripts/extract_scores_from_logs.py` - Score extraction

### Policies (Used in benchmark)
- ✅ `policies/base.py` - Base policy class
- ✅ `policies/bfn_unet_hybrid_image_policy.py` - BFN policy (used)
- ✅ `policies/diffusion_unet_hybrid_image_policy.py` - Diffusion policy (used)

### Workspaces (Used in benchmark)
- ✅ `workspaces/base_workspace.py` - Base workspace
- ✅ `workspaces/train_bfn_workspace.py` - BFN training (used)
- ✅ `workspaces/train_diffusion_unet_hybrid_workspace.py` - Diffusion training (used)

### Configs (Used in benchmark)
- ✅ `config/benchmark_bfn_pusht.yaml` - BFN benchmark config
- ✅ `config/benchmark_diffusion_pusht.yaml` - Diffusion benchmark config

### Networks (Required by policies)
- ✅ `networks/conditional_bfn.py` - BFN implementation
- ✅ `networks/base.py` - Base network
- ✅ `networks/unet.py` - U-Net (if used)
- ✅ `networks/multi_image_obs_encoder.py` - Image encoder (if used)

---

## 📊 Cleanup Summary

### Files to Delete: ~32 files

**Policies:** 6 files
**Workspaces:** 2 files  
**Scripts:** 8-10 files
**Configs:** 12-15 files
**Total:** ~30-40 files

### Space Saved: ~500KB - 2MB (code files)

---

## 🛠️ Recommended Action Plan

### Phase 1: Archive Unused Implementations
```bash
# Create archive directory
mkdir -p archive/unused_implementations

# Move unused policies
mv policies/conditional_bfn_policy.py archive/
mv policies/conditional_bfn_unet_hybrid_image_policy.py archive/
mv policies/guided_bfn_policy.py archive/
mv policies/flow_matching_unet_hybrid_image_policy.py archive/
mv policies/bfn_policy.py archive/
mv policies/diffusion_policy.py archive/

# Move unused workspaces
mv workspaces/train_conditional_bfn_workspace.py archive/
mv workspaces/train_guided_bfn_workspace.py archive/

# Move unused scripts
mv scripts/train_policy.py archive/
mv scripts/train_diffusion_workspace.py archive/
mv scripts/train_*.py archive/  # (except train_workspace.py)
mv scripts/run_diffusion_seed44.sh archive/  # Redundant - benchmark script handles all seeds
```

### Phase 2: Clean Configs
```bash
# Move unused configs
mv config/train_conditional_bfn*.yaml archive/
mv config/train_guided_bfn*.yaml archive/
mv config/config_*.yaml archive/
mv config/policy_train.yaml archive/
mv config/benchmark_flow_pusht.yaml archive/
```

### Phase 3: Update Imports
After archiving, update:
- `policies/__init__.py` - Remove unused imports
- Any remaining scripts that import archived files

---

## ⚠️ Before Deleting

1. **Verify benchmark still works:**
   ```bash
   python scripts/train_workspace.py --config-name=benchmark_bfn_pusht --help
   python scripts/train_workspace.py --config-name=benchmark_diffusion_pusht --help
   ```

2. **Check for any hidden dependencies:**
   ```bash
   grep -r "conditional_bfn\|guided_bfn\|flow_matching" scripts/ config/ workspaces/
   ```

3. **Backup first:**
   ```bash
   git add -A
   git commit -m "Archive unused implementations before cleanup"
   ```

---

## 🔄 CODE DUPLICATION ISSUES

### **Major Duplication: Workspace Training Loops**

All workspace classes (`TrainBFNWorkspace`, `TrainDiffusionUnetHybridWorkspace`, `TrainConditionalBFNWorkspace`, `TrainGuidedBFNWorkspace`) have **95% identical code**:

- Same training loop structure
- Same validation logic
- Same checkpointing
- Same logging
- Only difference: minor variations in sampling evaluation

**Recommendation:** Refactor to a single `TrainWorkspace` class that works for all policies:

```python
# Instead of 4 separate workspace classes, use one:
class TrainWorkspace(BaseWorkspace):
    def run(self):
        # Generic training loop that works for BFN, Diffusion, etc.
        # Policy-specific logic handled by policy.compute_loss()
```

**Impact:** Reduces ~400 lines of duplicate code to ~100 lines shared code.

### **Duplicate Training Scripts**

- `scripts/train_workspace.py` - Main entry point ✅
- `scripts/train_diffusion_workspace.py` - **Exact duplicate** ❌
- `scripts/train_policy.py` - PyTorch Lightning version (different framework) ❌

**Recommendation:** Delete `train_diffusion_workspace.py` - it's identical to `train_workspace.py`.

---

## 📝 Notes

- The benchmark **only** uses BFN and Diffusion with U-Net hybrid image policies
- All other variants (conditional, guided, flow matching) are experimental and not in the benchmark
- The `train_workspace.py` script is the unified entry point - other training scripts are redundant
- Analysis scripts can be kept if actively generating paper figures/tables
- **Workspace classes are 95% identical** - major refactoring opportunity

