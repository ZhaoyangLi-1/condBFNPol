<p align="center">
  <img src="assets/logo.svg" width="280" alt="BFN-Policy">
</p>

<p align="center">
  Bayesian Flow Networks for Visuomotor Policy Learning
</p>

<br>

## Overview

BFN-Policy reformulates action generation as continuous-time Bayesian inference, enabling **5× faster inference** than diffusion policies while maintaining competitive performance.

<br>

## Results

|  | Coverage | Inference | Steps |
|:--|--:|--:|--:|
| **BFN-Policy** | 92.5% | 18 ms | 20 |
| Diffusion | 96.0% | 91 ms | 100 |

At equal compute budget, BFN outperforms Diffusion by **76 percentage points**.

<br>

## Setup

```bash
conda create -n bfn python=3.10
conda activate bfn
pip install -r requirements.txt
```

<br>

## Usage

**Train**
```bash
python scripts/train_workspace.py --config-name=benchmark_bfn_pusht
```

**Evaluate**
```bash
python scripts/analysis/comprehensive_ablation_study.py \
    --checkpoint-dir outputs/2025.12.25
```

**Generate figures**
```bash
bash scripts/regenerate_all_figures.sh
```

<br>

## Structure

```
├── networks/        BFN algorithm
├── policies/        Policy implementations
├── config/          Experiment configs
├── scripts/         Training & analysis
└── jobs/            Cluster scripts
```

<br>

## Method

BFN-Policy treats action prediction as Bayesian belief updating:

1. Initialize with uninformative prior
2. Iteratively refine via learned updates
3. Return posterior mean

The key insight: **inference steps are decoupled from training**, allowing flexible speed-quality tradeoffs at deployment.

<br>

## Citation

```bibtex
@mastersthesis{bfn_policy_2025,
  title   = {Bayesian Flow Networks for Visuomotor Policy Learning},
  author  = {Your Name},
  school  = {Technical University of Munich},
  year    = {2025}
}
```

<br>

## License

MIT
