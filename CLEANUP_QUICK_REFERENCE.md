# Quick Reference: Files to Delete/Archive

## 🗑️ DELETE (Not Used in Benchmark)

### Policies (6 files)
```
policies/conditional_bfn_policy.py
policies/conditional_bfn_unet_hybrid_image_policy.py
policies/guided_bfn_policy.py
policies/flow_matching_unet_hybrid_image_policy.py
policies/bfn_policy.py  # Low-dim version, not used
policies/diffusion_policy.py  # Low-dim version, not used
```

### Workspaces (2 files)
```
workspaces/train_conditional_bfn_workspace.py
workspaces/train_guided_bfn_workspace.py
```

### Scripts (9 files)
```
scripts/train_policy.py  # PyTorch Lightning (not used)
scripts/train_diffusion_workspace.py  # Duplicate of train_workspace.py
scripts/train_diffusion_pusht.py
scripts/train_diffusion_transformer_pusht.py
scripts/train_conditional_bfn.py
scripts/train_guided_bfn.py
scripts/train_guided_bfn_pusht.py
scripts/standalone.py
scripts/run_diffusion_seed44.sh  # Redundant - run_pusht_benchmark.sh handles all seeds
```

### Configs (12 files)
```
config/train_bfn.yaml
config/train_conditional_bfn.yaml
config/train_conditional_bfn_unet_hybrid.yaml
config/train_guided_bfn.yaml
config/train_guided_bfn_unet_hybrid.yaml
config/train_diffusion_pusht.yaml
config/train_diffusion_transformer_pusht.yaml
config/config_cond_bfn.yaml
config/config_guided.yaml
config/config_diffusion.yaml
config/policy_train.yaml
config/benchmark_flow_pusht.yaml
```

### Evaluation Scripts (3 files)
```
scripts/eval_checkpoint.py
scripts/eval_diffusion.py
scripts/eval_pusht_checkpoint.py
```

**Total: ~32 files to delete**

---

## ✅ KEEP (Used in Benchmark)

### Core Training
- `scripts/train_workspace.py` - Main entry point
- `scripts/run_pusht_benchmark.sh` - Benchmark runner

### Policies
- `policies/base.py`
- `policies/bfn_unet_hybrid_image_policy.py` ✅
- `policies/diffusion_unet_hybrid_image_policy.py` ✅

### Workspaces
- `workspaces/base_workspace.py`
- `workspaces/train_bfn_workspace.py` ✅
- `workspaces/train_diffusion_unet_hybrid_workspace.py` ✅

### Configs
- `config/benchmark_bfn_pusht.yaml` ✅
- `config/benchmark_diffusion_pusht.yaml` ✅

### Evaluation/Analysis
- `scripts/comprehensive_ablation_study.py`
- `scripts/benchmark_pusht.py`
- `scripts/extract_scores_from_logs.py`

---

## 🔄 REFACTOR (Code Duplication)

**Workspace classes are 95% identical** - consider refactoring to single `TrainWorkspace` class.

---

## 📋 Quick Cleanup Command

```bash
# Create archive
mkdir -p archive/unused

# Move unused files
mv policies/{conditional_bfn,guided_bfn,flow_matching}* archive/unused/ 2>/dev/null
mv policies/bfn_policy.py policies/diffusion_policy.py archive/unused/ 2>/dev/null
mv workspaces/train_{conditional,guided}_bfn_workspace.py archive/unused/ 2>/dev/null
mv scripts/train_{policy,diffusion_workspace,diffusion_pusht,diffusion_transformer,conditional,guided,standalone}.py archive/unused/ 2>/dev/null
mv scripts/run_diffusion_seed44.sh archive/unused/ 2>/dev/null
mv config/train_{conditional,guided}*.yaml config/config_*.yaml config/policy_train.yaml config/benchmark_flow*.yaml archive/unused/ 2>/dev/null
mv scripts/eval_*.py archive/unused/ 2>/dev/null

# Update imports
# Edit policies/__init__.py to remove unused imports
```

