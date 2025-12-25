# Commands Run: From Benchmark to Figures

This document lists the commands you ran one by one to get your benchmark results and generate all the figures.

---

## Step 1: Run Benchmark Training

**Command:**
```bash
bash scripts/run_pusht_benchmark.sh
```

**What it does:**
- Trains BFN and Diffusion policies for seeds 42, 43, 44
- Saves checkpoints to `outputs/` or `cluster_checkpoints/benchmarkresults/`
- Generates training logs (`logs.json.txt`) in each run directory

**Output locations:**
- Training runs: `outputs/YYYY.MM.DD/HH.MM.SS_{bfn|diffusion}_seed{42|43|44}/`
- Or: `cluster_checkpoints/benchmarkresults/YYYY.MM.DD/...`

---

## Step 2: Generate Main Publication Figures

**Command:**
```bash
python scripts/generate_publication_figures.py
```

**What it does:**
- Loads training data from `cluster_checkpoints/benchmarkresults/`
- Generates main publication figures:
  - `fig1_main_results.pdf/png` - Bar chart comparing final performance
  - `fig2_per_seed.pdf/png` - Per-seed comparison
  - `fig3_efficiency.pdf/png` - Computational efficiency
  - `fig4_training_curves.pdf/png` - Training loss curves
  - `fig5_pareto.pdf/png` - Performance vs inference cost
  - `fig_combined.pdf/png` - Combined 4-panel figure

**Output:** `figures/publication/fig*.pdf` and `.png`

---

## Step 3: Generate Multimodal Behavior Figures

**Command:**
```bash
python scripts/generate_multimodal_figures.py
```

**What it does:**
- Generates visualizations of multimodal behavior:
  - `fig_multimodal_comparison.pdf/png` - 5-panel comparison
  - `fig_multimodal_grid.pdf/png` - 2x3 grid layout
  - `fig_bfn_diffusion_multimodal.pdf/png` - BFN vs Diffusion focused
  - `fig_mode_distribution.pdf/png` - Mode distribution histogram
  - `fig_temporal_consistency.pdf/png` - Temporal consistency

**Output:** `figures/publication/fig_*modal*.pdf` and `.png`

---

## Step 4: Generate Prediction Visualizations

**Command:**
```bash
python scripts/generate_prediction_visualizations.py
```

**What it does:**
- Generates prediction and trajectory visualizations:
  - `fig_trajectory_comparison.pdf/png` - BFN vs Diffusion vs Ground Truth
  - `fig_denoising_process.pdf/png` - Denoising steps visualization
  - `fig_prediction_variance.pdf/png` - Prediction uncertainty
  - `fig_action_dimensions.pdf/png` - Action dimensions over time
  - `fig_pusht_visualization.pdf/png` - Push-T task visualization
  - `fig_qualitative_combined.pdf/png` - Combined qualitative figure
  - `fig_method_diagram.pdf/png` - Method comparison diagram

**Output:** `figures/publication/fig_*.pdf` and `.png`

---

## Step 5: Generate Training Stability Figures

**Command:**
```bash
python scripts/generate_training_stability_figure.py
```

**What it does:**
- Generates training stability visualizations:
  - `fig_training_stability.pdf/png` - BFN vs Diffusion stability
  - `fig_training_stability_with_ibc.pdf/png` - With IBC comparison
  - `fig_checkpoint_selection.pdf/png` - Checkpoint selection difficulty
  - `fig_loss_performance_correlation.pdf/png` - Loss vs performance correlation

**Output:** `figures/publication/fig_training*.pdf` and `.png`

---

## Step 6: Run Comprehensive Ablation Study (Optional)

**Command:**
```bash
python scripts/comprehensive_ablation_study.py \
    --checkpoint-dir cluster_checkpoints/benchmarkresults \
    --n-envs 50 \
    --device cuda
```

**What it does:**
- Evaluates policies at different inference steps
- Generates ablation study figures:
  - `fig_ablation_score_vs_steps.pdf/png`
  - `fig_ablation_time_vs_steps.pdf/png`
  - `fig_ablation_pareto.pdf/png`
  - `fig_ablation_combined.pdf/png`

**Output:** `figures/publication/fig_ablation*.pdf` and `.png`

---

## Step 7: Generate Analysis Tables (Optional)

**Command:**
```bash
python scripts/analyze_local_results.py
```

**What it does:**
- Analyzes benchmark results
- Generates:
  - `figures/results_table.tex` - LaTeX table
  - `figures/benchmark_comparison.pdf` - Training curves
  - `figures/per_seed_comparison.pdf` - Per-seed bars

**Output:** `figures/*.tex` and `figures/*.pdf`

---

## Step 8: Extract Scores from Logs (If needed)

**Command:**
```bash
python scripts/extract_scores_from_logs.py
```

**What it does:**
- Extracts test scores from `logs.json.txt` files
- Generates summary table: `figures/tables/table_scores_from_logs.tex`

**Output:** `figures/tables/table_scores_from_logs.tex`

---

## Complete Command Sequence

Here's the complete sequence you likely ran:

```bash
# 1. Train models
bash scripts/run_pusht_benchmark.sh

# 2. Generate main figures
python scripts/generate_publication_figures.py

# 3. Generate multimodal figures
python scripts/generate_multimodal_figures.py

# 4. Generate prediction visualizations
python scripts/generate_prediction_visualizations.py

# 5. Generate training stability figures
python scripts/generate_training_stability_figure.py

# 6. (Optional) Run ablation study
python scripts/comprehensive_ablation_study.py \
    --checkpoint-dir cluster_checkpoints/benchmarkresults \
    --n-envs 50 \
    --device cuda

# 7. (Optional) Generate analysis tables
python scripts/analyze_local_results.py

# 8. (Optional) Extract scores
python scripts/extract_scores_from_logs.py
```

---

## Output Summary

All figures are saved to:
- **Main figures:** `figures/publication/fig*.pdf` and `.png`
- **Tables:** `figures/tables/*.tex`
- **Analysis:** `figures/*.pdf` and `figures/*.tex`

---

## Notes

1. **Training must complete first** - All figure generation scripts require training results
2. **Checkpoint location** - Scripts look for checkpoints in `cluster_checkpoints/benchmarkresults/` by default
3. **Device** - Ablation study requires GPU (`--device cuda`)
4. **Environment** - Make sure you're in the project root directory when running commands

