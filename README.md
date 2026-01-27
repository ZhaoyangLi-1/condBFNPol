<h1 align="center">BFN-Policy</h1>

<p align="center">
  <strong>Bayesian Flow Networks for Visuomotor Policy Learning</strong>
</p>

<p align="center">
  <a href="#results">Results</a> · 
  <a href="#setup">Setup</a> · 
  <a href="#usage">Usage</a> · 
  <a href="#method">Method</a>
</p>

---

BFN-Policy reformulates action generation as continuous-time Bayesian inference, enabling **5× faster inference** than diffusion policies while maintaining competitive performance.

## Results

| Method | Coverage | Latency | Steps |
|:-------|:--------:|:-------:|:-----:|
| **BFN-Policy** | **92.5%** | **18ms** | 20 |
| Diffusion Policy | 96.0% | 91ms | 100 |

> At equal compute (20 steps): BFN 92.5% vs Diffusion 16.8%

## Setup

```bash
conda create -n bfn python=3.10 && conda activate bfn
pip install -r requirements.txt
conda env config vars set PYTHONPATH="/scr/zhaoyang/condBFNPol/src/diffusion-policy:$PYTHONPATH"
conda deactivate
conda activate bfn
```

## Usage

```bash
# Train
python scripts/train_workspace.py --config-name=benchmark_bfn_pusht

# Evaluate
python scripts/analysis/comprehensive_ablation_study.py --checkpoint-dir outputs/

# Figures
bash scripts/regenerate_all_figures.sh
```

## Structure

```
networks/     Core BFN algorithm
policies/     Policy implementations
config/       Experiment configs
scripts/      Training & analysis
jobs/         SLURM scripts
```

## Method

BFN-Policy treats action prediction as Bayesian belief updating:

1. Initialize uninformative prior
2. Iteratively refine via learned updates  
3. Return posterior mean

Key insight: inference steps are decoupled from training—train once, deploy at any speed.
