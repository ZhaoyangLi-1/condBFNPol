#!/usr/bin/env python
"""
PushT Benchmark: BFN vs Diffusion Policy

This script runs comprehensive benchmarks comparing BFN and Diffusion policies
on the PushT task. It measures:
1. Task performance (success rate, mean score)
2. Inference speed at different sampling steps  
3. Training efficiency (convergence speed)

Usage:
    # Run full benchmark with 3 seeds
    python scripts/benchmark_pusht.py --mode full --seeds 42,43,44
    
    # Run only evaluation (requires trained checkpoints)
    python scripts/benchmark_pusht.py --mode eval --bfn_ckpt path/to/bfn.ckpt --diff_ckpt path/to/diff.ckpt
    
    # Run inference speed test
    python scripts/benchmark_pusht.py --mode speed --bfn_ckpt path/to/bfn.ckpt --diff_ckpt path/to/diff.ckpt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
from diffusion_policy.model.common.normalizer import LinearNormalizer


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a policy from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    
    cfg = payload.get('cfg', payload.get('config'))
    state_dict = payload.get('state_dicts', payload.get('state_dict', {}))
    
    # Instantiate policy from config
    from hydra.utils import instantiate
    policy = instantiate(cfg.policy)
    
    # Load weights
    if 'model' in state_dict:
        policy.model.load_state_dict(state_dict['model'])
    if 'obs_encoder' in state_dict:
        policy.obs_encoder.load_state_dict(state_dict['obs_encoder'])
    if 'normalizer' in state_dict:
        policy.normalizer.load_state_dict(state_dict['normalizer'])
    elif 'ema_model' in state_dict:
        # Try loading from ema
        ema_state = state_dict['ema_model']
        if hasattr(policy, 'load_state_dict'):
            policy.load_state_dict(ema_state)
    
    policy.to(device)
    policy.eval()
    return policy, cfg


def create_env_runner(cfg, n_test: int = 50) -> PushTImageRunner:
    """Create PushT environment runner for evaluation."""
    runner = PushTImageRunner(
        n_train=0,
        n_train_vis=0,
        train_start_seed=0,
        n_test=n_test,
        n_test_vis=4,
        test_start_seed=100000,
        max_steps=300,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        fps=10,
        past_action=False,
        n_envs=None,
        legacy_test=True,
    )
    return runner


def evaluate_policy(policy, runner: PushTImageRunner, device: str = "cuda") -> Dict:
    """Evaluate a policy on PushT and return metrics."""
    policy.eval()
    policy.to(device)
    
    with torch.no_grad():
        result = runner.run(policy)
    
    return {
        'mean_score': result.get('test/mean_score', result.get('test_mean_score', 0)),
        'success_rate': result.get('test/success_rate', 0),
        'max_score': result.get('test/max_score', 0),
        'min_score': result.get('test/min_score', 0),
    }


def measure_inference_speed(
    policy,
    n_samples: int = 100,
    batch_size: int = 1,
    device: str = "cuda",
    n_timesteps_list: Optional[List[int]] = None,
) -> Dict:
    """Measure inference speed at different sampling steps."""
    policy.eval()
    policy.to(device)
    
    # Create dummy observation
    dummy_obs = {
        'image': torch.randn(batch_size, 2, 3, 96, 96, device=device),
        'agent_pos': torch.randn(batch_size, 2, 2, device=device),
    }
    
    results = {}
    
    # Determine what timesteps to test
    if n_timesteps_list is None:
        if hasattr(policy, 'n_timesteps'):
            # BFN policy
            n_timesteps_list = [5, 10, 20, 50]
        elif hasattr(policy, 'noise_scheduler'):
            # Diffusion policy
            n_timesteps_list = [10, 20, 50, 100]
        else:
            n_timesteps_list = [20]
    
    for n_steps in n_timesteps_list:
        # Set inference steps
        if hasattr(policy, 'n_timesteps'):
            original_steps = policy.n_timesteps
            policy.n_timesteps = n_steps
        elif hasattr(policy, 'num_inference_steps'):
            original_steps = policy.num_inference_steps
            policy.num_inference_steps = n_steps
        elif hasattr(policy, 'noise_scheduler'):
            original_steps = policy.noise_scheduler.config.num_train_timesteps
            policy.noise_scheduler.set_timesteps(n_steps)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = policy.predict_action(dummy_obs)
        
        # Measure
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(n_samples):
            with torch.no_grad():
                _ = policy.predict_action(dummy_obs)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        results[n_steps] = {
            'total_time': elapsed,
            'avg_time_ms': (elapsed / n_samples) * 1000,
            'fps': n_samples / elapsed,
        }
        
        # Restore original steps
        if hasattr(policy, 'n_timesteps'):
            policy.n_timesteps = original_steps
        elif hasattr(policy, 'num_inference_steps'):
            policy.num_inference_steps = original_steps
    
    return results


def run_training(
    config_name: str,
    seed: int,
    output_dir: str,
    num_epochs: int = 100,
    device: str = "cuda",
):
    """Run training with given config and seed."""
    import subprocess
    
    cmd = [
        "python", "scripts/train_workspace.py",
        f"--config-name={config_name}",
        f"training.seed={seed}",
        f"training.device={device}",
        f"training.num_epochs={num_epochs}",
        f"hydra.run.dir={output_dir}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_full_benchmark(
    seeds: List[int],
    num_epochs: int = 100,
    device: str = "cuda",
    output_base: str = "benchmark_results",
):
    """Run full benchmark: train both methods and evaluate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"pusht_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'bfn': {'seeds': {}, 'aggregate': {}},
        'diffusion': {'seeds': {}, 'aggregate': {}},
        'speed_comparison': {},
    }
    
    # Train and evaluate both methods for each seed
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
        
        # BFN
        bfn_dir = output_dir / f"bfn_seed{seed}"
        print(f"\n--- Training BFN (seed={seed}) ---")
        success = run_training(
            config_name="train_bfn_unet_hybrid",
            seed=seed,
            output_dir=str(bfn_dir),
            num_epochs=num_epochs,
            device=device,
        )
        
        if success:
            # Find best checkpoint
            ckpt_dir = bfn_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if ckpts:
                    best_ckpt = max(ckpts, key=lambda x: x.stat().st_mtime)
                    results['bfn']['seeds'][seed] = {'checkpoint': str(best_ckpt)}
        
        # Diffusion
        diff_dir = output_dir / f"diffusion_seed{seed}"
        print(f"\n--- Training Diffusion (seed={seed}) ---")
        success = run_training(
            config_name="train_diffusion_unet_hybrid",
            seed=seed,
            output_dir=str(diff_dir),
            num_epochs=num_epochs,
            device=device,
        )
        
        if success:
            ckpt_dir = diff_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if ckpts:
                    best_ckpt = max(ckpts, key=lambda x: x.stat().st_mtime)
                    results['diffusion']['seeds'][seed] = {'checkpoint': str(best_ckpt)}
    
    # Save results
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    return results


def run_evaluation_benchmark(
    bfn_ckpt: str,
    diff_ckpt: str,
    n_test: int = 50,
    device: str = "cuda",
) -> Dict:
    """Run evaluation benchmark on pre-trained checkpoints."""
    results = {}
    
    # Load and evaluate BFN
    print("\n--- Evaluating BFN Policy ---")
    bfn_policy, bfn_cfg = load_policy(bfn_ckpt, device)
    bfn_runner = create_env_runner(bfn_cfg, n_test)
    results['bfn'] = evaluate_policy(bfn_policy, bfn_runner, device)
    print(f"BFN Results: {results['bfn']}")
    
    # Load and evaluate Diffusion
    print("\n--- Evaluating Diffusion Policy ---")
    diff_policy, diff_cfg = load_policy(diff_ckpt, device)
    diff_runner = create_env_runner(diff_cfg, n_test)
    results['diffusion'] = evaluate_policy(diff_policy, diff_runner, device)
    print(f"Diffusion Results: {results['diffusion']}")
    
    return results


def run_speed_benchmark(
    bfn_ckpt: str,
    diff_ckpt: str,
    n_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """Run inference speed benchmark."""
    results = {}
    
    # BFN speed test
    print("\n--- BFN Inference Speed ---")
    bfn_policy, _ = load_policy(bfn_ckpt, device)
    results['bfn'] = measure_inference_speed(
        bfn_policy,
        n_samples=n_samples,
        device=device,
        n_timesteps_list=[5, 10, 20, 50],
    )
    
    # Diffusion speed test
    print("\n--- Diffusion Inference Speed ---")
    diff_policy, _ = load_policy(diff_ckpt, device)
    results['diffusion'] = measure_inference_speed(
        diff_policy,
        n_samples=n_samples,
        device=device,
        n_timesteps_list=[10, 20, 50, 100],
    )
    
    # Print comparison table
    print("\n" + "="*70)
    print("Inference Speed Comparison (ms per action prediction)")
    print("="*70)
    print(f"{'Steps':<10} {'BFN (ms)':<15} {'Diffusion (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for steps in [10, 20, 50]:
        bfn_time = results['bfn'].get(steps, {}).get('avg_time_ms', float('nan'))
        diff_time = results['diffusion'].get(steps, {}).get('avg_time_ms', float('nan'))
        speedup = diff_time / bfn_time if bfn_time > 0 else float('nan')
        print(f"{steps:<10} {bfn_time:<15.2f} {diff_time:<15.2f} {speedup:<10.2f}x")
    
    return results


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table from results."""
    data = []
    
    for method, method_results in results.items():
        if isinstance(method_results, dict) and 'mean_score' in method_results:
            data.append({
                'Method': method.upper(),
                'Mean Score': method_results['mean_score'],
                'Success Rate': method_results.get('success_rate', 'N/A'),
                'Max Score': method_results.get('max_score', 'N/A'),
            })
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="PushT Benchmark: BFN vs Diffusion")
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['full', 'eval', 'speed'],
                        help='Benchmark mode: full (train+eval), eval (eval only), speed (inference speed)')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                        help='Comma-separated list of seeds for full benchmark')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--bfn_ckpt', type=str, default=None,
                        help='Path to BFN checkpoint (for eval/speed modes)')
    parser.add_argument('--diff_ckpt', type=str, default=None,
                        help='Path to Diffusion checkpoint (for eval/speed modes)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--n_test', type=int, default=50,
                        help='Number of test episodes')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        seeds = [int(s) for s in args.seeds.split(',')]
        results = run_full_benchmark(
            seeds=seeds,
            num_epochs=args.num_epochs,
            device=args.device,
            output_base=args.output_dir,
        )
        
    elif args.mode == 'eval':
        if not args.bfn_ckpt or not args.diff_ckpt:
            print("Error: --bfn_ckpt and --diff_ckpt are required for eval mode")
            sys.exit(1)
        
        results = run_evaluation_benchmark(
            bfn_ckpt=args.bfn_ckpt,
            diff_ckpt=args.diff_ckpt,
            n_test=args.n_test,
            device=args.device,
        )
        
        # Print table
        df = create_comparison_table(results)
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(df.to_string(index=False))
        
    elif args.mode == 'speed':
        if not args.bfn_ckpt or not args.diff_ckpt:
            print("Error: --bfn_ckpt and --diff_ckpt are required for speed mode")
            sys.exit(1)
        
        results = run_speed_benchmark(
            bfn_ckpt=args.bfn_ckpt,
            diff_ckpt=args.diff_ckpt,
            device=args.device,
        )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"pusht_{args.mode}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
