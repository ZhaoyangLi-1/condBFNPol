#!/usr/bin/env python
"""
Quick Training for Ablation Study

Train BFN and Diffusion policies for the ablation study.
Uses fewer epochs for faster iteration but enough for meaningful comparison.

Usage:
    # Train both methods with seed 42
    python scripts/train_for_ablation.py --seed 42
    
    # Train all seeds
    python scripts/train_for_ablation.py --all-seeds
    
    # Train only BFN
    python scripts/train_for_ablation.py --seed 42 --method bfn
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def train_bfn(seed: int, epochs: int = 100, device: str = "mps"):
    """Train BFN policy."""
    output_dir = PROJECT_ROOT / f"results/ablation_checkpoints/bfn_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/train_workspace.py",
        "--config-name=benchmark_bfn_pusht",
        f"training.seed={seed}",
        f"training.device={device}",
        f"training.num_epochs={epochs}",
        f"training.checkpoint_every=50",
        f"hydra.run.dir={output_dir}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Training BFN (seed={seed}, epochs={epochs})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def train_diffusion(seed: int, epochs: int = 100, device: str = "mps"):
    """Train Diffusion policy."""
    output_dir = PROJECT_ROOT / f"results/ablation_checkpoints/diffusion_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/train_workspace.py",
        "--config-name=benchmark_diffusion_pusht",
        f"training.seed={seed}",
        f"training.device={device}",
        f"training.num_epochs={epochs}",
        f"training.checkpoint_every=50",
        f"hydra.run.dir={output_dir}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Training Diffusion (seed={seed}, epochs={epochs})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--method", choices=["bfn", "diffusion", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    seeds = [42, 43, 44] if args.all_seeds else [args.seed]
    
    for seed in seeds:
        if args.method in ["bfn", "both"]:
            train_bfn(seed, args.epochs, args.device)
        if args.method in ["diffusion", "both"]:
            train_diffusion(seed, args.epochs, args.device)
    
    print("\n" + "="*60)
    print("Training complete! Checkpoints saved to:")
    print("  results/ablation_checkpoints/")
    print("="*60)


if __name__ == "__main__":
    main()
