#!/usr/bin/env python
"""
LOCAL Inference Steps Ablation Study (Mac/MPS version)

Run the ablation study locally on your Mac using MPS or CPU.
Uses fewer evaluation episodes for faster iteration.

Usage:
    # Quick test (3 envs per config)
    python scripts/run_ablation_local.py --quick
    
    # Full local run (20 envs per config)  
    python scripts/run_ablation_local.py --n-envs 20
    
    # Single seed only
    python scripts/run_ablation_local.py --seed 42 --n-envs 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def find_local_checkpoints(base_dir: str = "cluster_checkpoints/benchmarkresults") -> Dict:
    """Find all benchmark checkpoints."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: {base_dir} not found")
        return {'bfn': {}, 'diffusion': {}}
    
    checkpoints = {'bfn': {}, 'diffusion': {}}
    
    for run_dir in base_path.rglob("*_seed*"):
        if not run_dir.is_dir():
            continue
        
        dir_name = run_dir.name
        if "bfn" in dir_name.lower():
            method = "bfn"
        elif "diffusion" in dir_name.lower():
            method = "diffusion"
        else:
            continue
        
        # Extract seed
        seed = None
        for part in dir_name.split("_"):
            if part.startswith("seed"):
                try:
                    seed = int(part.replace("seed", ""))
                except ValueError:
                    pass
        
        if seed is None:
            continue
        
        # Find best checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        
        best_score = -1
        best_ckpt = None
        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            name = ckpt_file.stem
            if "test_mean_score=" in name:
                try:
                    score_str = name.split("test_mean_score=")[1]
                    score = float(score_str)
                    if score > best_score:
                        best_score = score
                        best_ckpt = str(ckpt_file)
                except:
                    pass
        
        if best_ckpt:
            checkpoints[method][seed] = best_ckpt
            print(f"  Found {method} seed {seed}: score={best_score:.3f}")
    
    return checkpoints


def load_policy(checkpoint_path: str, method: str, device: str):
    """Load a policy from checkpoint."""
    print(f"Loading {method} from: {Path(checkpoint_path).name}")
    
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = payload.get('cfg', payload.get('config'))
    state_dicts = payload.get('state_dicts', {})
    
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    # Override device in config
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'device'):
        with OmegaConf.read_write(cfg):
            cfg.policy.device = device
    
    policy = instantiate(cfg.policy)
    
    # Load weights
    if 'model' in state_dicts:
        policy.model.load_state_dict(state_dicts['model'])
    if 'obs_encoder' in state_dicts:
        policy.obs_encoder.load_state_dict(state_dicts['obs_encoder'])
    
    # Load normalizer
    if 'normalizer' in payload:
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(payload['normalizer'])
        policy.set_normalizer(normalizer)
    
    policy.to(device)
    policy.eval()
    return policy, cfg


def evaluate_policy_at_steps(
    policy,
    method: str,
    n_steps: int,
    n_envs: int = 10,
    max_steps: int = 300,
    device: str = "mps",
) -> Dict:
    """Evaluate policy with specific inference steps."""
    from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
    
    # Set inference steps
    original_steps = None
    if method == 'bfn':
        original_steps = getattr(policy, 'n_timesteps', 20)
        policy.n_timesteps = n_steps
    else:
        if hasattr(policy, 'num_inference_steps'):
            original_steps = policy.num_inference_steps
            policy.num_inference_steps = n_steps
        if hasattr(policy, 'noise_scheduler'):
            policy.noise_scheduler.set_timesteps(n_steps)
    
    # Create runner
    runner = PushTImageRunner(
        n_train=0,
        n_train_vis=0,
        train_start_seed=0,
        n_test=n_envs,
        n_test_vis=0,
        test_start_seed=100000,
        max_steps=max_steps,
        n_obs_steps=policy.n_obs_steps,
        n_action_steps=policy.n_action_steps,
        fps=10,
        past_action=False,
        n_envs=None,
        legacy_test=True,
    )
    
    # Time the evaluation
    start_time = time.perf_counter()
    
    policy.eval()
    with torch.no_grad():
        result = runner.run(policy)
    
    elapsed = time.perf_counter() - start_time
    
    # Restore steps
    if original_steps is not None:
        if method == 'bfn':
            policy.n_timesteps = original_steps
        elif hasattr(policy, 'num_inference_steps'):
            policy.num_inference_steps = original_steps
    
    return {
        'mean_score': result.get('test/mean_score', result.get('test_mean_score', 0)),
        'max_score': result.get('test/max_score', 0),
        'min_score': result.get('test/min_score', 0),
        'eval_time_sec': elapsed,
    }


def measure_inference_time(policy, method: str, n_steps: int, device: str, n_samples: int = 20):
    """Measure inference time per action prediction."""
    # Dummy observation
    dummy_obs = {
        'image': torch.randn(1, 2, 3, 96, 96, device=device),
        'agent_pos': torch.randn(1, 2, 2, device=device),
    }
    
    # Set steps
    if method == 'bfn':
        original = policy.n_timesteps
        policy.n_timesteps = n_steps
    else:
        if hasattr(policy, 'num_inference_steps'):
            original = policy.num_inference_steps
            policy.num_inference_steps = n_steps
        if hasattr(policy, 'noise_scheduler'):
            policy.noise_scheduler.set_timesteps(n_steps)
    
    # Warmup
    policy.eval()
    for _ in range(3):
        with torch.no_grad():
            _ = policy.predict_action(dummy_obs)
    
    # Measure
    start = time.perf_counter()
    for _ in range(n_samples):
        with torch.no_grad():
            _ = policy.predict_action(dummy_obs)
    elapsed = time.perf_counter() - start
    
    # Restore
    if method == 'bfn':
        policy.n_timesteps = original
    elif hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = original
    
    return {
        'avg_time_ms': (elapsed / n_samples) * 1000,
        'hz': n_samples / elapsed,
    }


def run_local_ablation(
    checkpoints: Dict,
    bfn_steps: List[int],
    diffusion_steps: List[int],
    n_envs: int,
    device: str,
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/ablation",
):
    """Run ablation study locally."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'bfn': defaultdict(dict),
        'diffusion': defaultdict(dict),
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'n_envs': n_envs,
            'bfn_steps': bfn_steps,
            'diffusion_steps': diffusion_steps,
        }
    }
    
    # Filter seeds if specified
    bfn_ckpts = checkpoints.get('bfn', {})
    diff_ckpts = checkpoints.get('diffusion', {})
    
    if seeds:
        bfn_ckpts = {s: p for s, p in bfn_ckpts.items() if s in seeds}
        diff_ckpts = {s: p for s, p in diff_ckpts.items() if s in seeds}
    
    # BFN Ablation
    print("\n" + "=" * 60)
    print("BFN INFERENCE STEPS ABLATION")
    print("=" * 60)
    
    for seed, ckpt_path in bfn_ckpts.items():
        print(f"\n--- BFN Seed {seed} ---")
        try:
            policy, cfg = load_policy(ckpt_path, 'bfn', device)
        except Exception as e:
            print(f"Error loading: {e}")
            continue
        
        for n_steps in bfn_steps:
            print(f"  Steps={n_steps}: ", end="", flush=True)
            try:
                # Measure inference time first (quick)
                time_result = measure_inference_time(policy, 'bfn', n_steps, device)
                
                # Run evaluation
                eval_result = evaluate_policy_at_steps(
                    policy, 'bfn', n_steps, n_envs, device=device
                )
                
                results['bfn'][seed][n_steps] = {
                    'mean_score': eval_result['mean_score'],
                    'max_score': eval_result['max_score'],
                    'min_score': eval_result['min_score'],
                    'avg_time_ms': time_result['avg_time_ms'],
                    'std_score': 0,  # Single seed, no std
                }
                
                print(f"Score={eval_result['mean_score']:.3f}, Time={time_result['avg_time_ms']:.1f}ms")
            except Exception as e:
                print(f"Error: {e}")
    
    # Diffusion Ablation
    print("\n" + "=" * 60)
    print("DIFFUSION INFERENCE STEPS ABLATION")
    print("=" * 60)
    
    for seed, ckpt_path in diff_ckpts.items():
        print(f"\n--- Diffusion Seed {seed} ---")
        try:
            policy, cfg = load_policy(ckpt_path, 'diffusion', device)
        except Exception as e:
            print(f"Error loading: {e}")
            continue
        
        for n_steps in diffusion_steps:
            print(f"  Steps={n_steps}: ", end="", flush=True)
            try:
                time_result = measure_inference_time(policy, 'diffusion', n_steps, device)
                eval_result = evaluate_policy_at_steps(
                    policy, 'diffusion', n_steps, n_envs, device=device
                )
                
                results['diffusion'][seed][n_steps] = {
                    'mean_score': eval_result['mean_score'],
                    'max_score': eval_result['max_score'],
                    'min_score': eval_result['min_score'],
                    'avg_time_ms': time_result['avg_time_ms'],
                    'std_score': 0,
                }
                
                print(f"Score={eval_result['mean_score']:.3f}, Time={time_result['avg_time_ms']:.1f}ms")
            except Exception as e:
                print(f"Error: {e}")
    
    # Save results
    results_to_save = {
        'bfn': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in results['bfn'].items()},
        'diffusion': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in results['diffusion'].items()},
        'metadata': results['metadata'],
    }
    
    results_file = output_path / "ablation_results_local.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results_to_save


def generate_plots(results: Dict, output_dir: str = "figures/publication"):
    """Generate publication plots from results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    COLORS = {'bfn': '#4285F4', 'diffusion': '#EA4335'}
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Aggregate across seeds
    def aggregate(method_results):
        all_steps = set()
        for seed_data in method_results.values():
            all_steps.update(seed_data.keys())
        
        agg = {}
        for steps in sorted(all_steps, key=lambda x: int(x)):
            scores = [d[steps]['mean_score'] for d in method_results.values() if steps in d]
            times = [d[steps]['avg_time_ms'] for d in method_results.values() if steps in d]
            if scores:
                agg[int(steps)] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores) if len(scores) > 1 else 0,
                    'mean_time': np.mean(times),
                }
        return agg
    
    bfn_agg = aggregate(results.get('bfn', {}))
    diff_agg = aggregate(results.get('diffusion', {}))
    
    if not bfn_agg and not diff_agg:
        print("No data to plot")
        return
    
    # Combined 2-panel figure
    fig = plt.figure(figsize=(7.16, 3.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # Panel A: Score vs Steps
    ax1 = fig.add_subplot(gs[0])
    
    if bfn_agg:
        steps = sorted(bfn_agg.keys())
        scores = [bfn_agg[s]['mean_score'] for s in steps]
        stds = [bfn_agg[s]['std_score'] for s in steps]
        ax1.errorbar(steps, scores, yerr=stds, color=COLORS['bfn'], marker='o',
                     markersize=6, capsize=3, linewidth=1.5, label='BFN Policy')
    
    if diff_agg:
        steps = sorted(diff_agg.keys())
        scores = [diff_agg[s]['mean_score'] for s in steps]
        stds = [diff_agg[s]['std_score'] for s in steps]
        ax1.errorbar(steps, scores, yerr=stds, color=COLORS['diffusion'], marker='s',
                     markersize=6, capsize=3, linewidth=1.5, label='Diffusion Policy')
    
    ax1.set_xlabel('Inference Steps')
    ax1.set_ylabel('Success Score')
    ax1.set_title('(a) Performance vs Steps', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Vertical lines for defaults
    ax1.axvline(x=20, color=COLORS['bfn'], linestyle='--', alpha=0.3)
    ax1.axvline(x=100, color=COLORS['diffusion'], linestyle='--', alpha=0.3)
    
    # Panel B: Pareto (Score vs Time)
    ax2 = fig.add_subplot(gs[1])
    
    if bfn_agg:
        steps = sorted(bfn_agg.keys())
        times = [bfn_agg[s]['mean_time'] for s in steps]
        scores = [bfn_agg[s]['mean_score'] for s in steps]
        ax2.plot(times, scores, color=COLORS['bfn'], marker='o', markersize=6,
                 linewidth=1.5, label='BFN Policy')
        for s, t, sc in zip(steps, times, scores):
            ax2.annotate(f'{s}', (t, sc), xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=COLORS['bfn'])
    
    if diff_agg:
        steps = sorted(diff_agg.keys())
        times = [diff_agg[s]['mean_time'] for s in steps]
        scores = [diff_agg[s]['mean_score'] for s in steps]
        ax2.plot(times, scores, color=COLORS['diffusion'], marker='s', markersize=6,
                 linewidth=1.5, label='Diffusion Policy')
        for s, t, sc in zip(steps, times, scores):
            ax2.annotate(f'{s}', (t, sc), xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=COLORS['diffusion'])
    
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Success Score')
    ax2.set_title('(b) Efficiency Trade-off', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig_ablation_combined.pdf')
    fig.savefig(output_path / 'fig_ablation_combined.png', dpi=300)
    plt.close(fig)
    
    print(f"Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Local inference steps ablation")
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 envs")
    parser.add_argument("--n-envs", type=int, default=10, help="Environments per eval")
    parser.add_argument("--seed", type=int, help="Run only this seed")
    parser.add_argument("--bfn-steps", type=str, default="5,10,20,30",
                        help="BFN steps to test")
    parser.add_argument("--diffusion-steps", type=str, default="20,50,100",
                        help="Diffusion steps to test")
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    if args.quick:
        args.n_envs = 3
        args.bfn_steps = "10,20"
        args.diffusion_steps = "50,100"
    
    bfn_steps = [int(x) for x in args.bfn_steps.split(",")]
    diffusion_steps = [int(x) for x in args.diffusion_steps.split(",")]
    seeds = [args.seed] if args.seed else None
    
    if args.plot_only:
        results_file = Path(args.output_dir) / "ablation_results_local.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            generate_plots(results)
        else:
            print(f"No results file found: {results_file}")
        return
    
    # Find checkpoints
    print("\nFinding checkpoints...")
    checkpoints = find_local_checkpoints()
    
    if not checkpoints['bfn'] and not checkpoints['diffusion']:
        print("No checkpoints found!")
        return
    
    # Run ablation
    results = run_local_ablation(
        checkpoints=checkpoints,
        bfn_steps=bfn_steps,
        diffusion_steps=diffusion_steps,
        n_envs=args.n_envs,
        device=device,
        seeds=seeds,
        output_dir=args.output_dir,
    )
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    generate_plots(results)
    
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
