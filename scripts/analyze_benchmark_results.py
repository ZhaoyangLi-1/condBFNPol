#!/usr/bin/env python3
"""
Analyze BFN vs Diffusion Benchmark Results

This script helps analyze and compare results from the benchmark runs.
It can pull data from WandB or analyze local checkpoints.

Usage:
    python scripts/analyze_benchmark_results.py --wandb-project pusht-benchmark
    python scripts/analyze_benchmark_results.py --local-dir results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Only local analysis available.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_wandb_runs(project: str, entity: Optional[str] = None) -> List[Dict]:
    """Fetch runs from WandB project."""
    if not HAS_WANDB:
        raise ImportError("wandb is required for this function")
    
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)
    
    results = []
    for run in runs:
        # Extract key info
        config = run.config
        summary = run.summary._json_dict
        
        # Determine policy type
        if 'bfn' in run.name.lower() or 'bfn' in config.get('name', '').lower():
            policy_type = 'BFN'
        elif 'diffusion' in run.name.lower() or 'diffusion' in config.get('name', '').lower():
            policy_type = 'Diffusion'
        else:
            policy_type = 'Unknown'
        
        results.append({
            'name': run.name,
            'id': run.id,
            'policy_type': policy_type,
            'seed': config.get('training', {}).get('seed', None),
            'epochs': config.get('training', {}).get('num_epochs', None),
            'state': run.state,
            # Key metrics
            'final_loss': summary.get('train/loss', None),
            'final_action_mse': summary.get('train/action_mse_error', None),
            'best_test_score': summary.get('test_mean_score', None),
            'best_test_score_max': summary.get('test_mean_score_max', None),
            # Training info
            'total_epochs': summary.get('epoch', None),
            'runtime_hours': run.summary.get('_runtime', 0) / 3600,
        })
    
    return results


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of BFN vs Diffusion results."""
    
    bfn_runs = [r for r in results if r['policy_type'] == 'BFN' and r['state'] == 'finished']
    diff_runs = [r for r in results if r['policy_type'] == 'Diffusion' and r['state'] == 'finished']
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS: BFN vs Diffusion on PushT")
    print("="*80)
    
    # BFN Results
    print("\n📊 BFN Policy Results:")
    print("-" * 60)
    if bfn_runs:
        scores = [r['best_test_score'] for r in bfn_runs if r['best_test_score'] is not None]
        mses = [r['final_action_mse'] for r in bfn_runs if r['final_action_mse'] is not None]
        
        for run in bfn_runs:
            print(f"  Seed {run['seed']}: test_score={run['best_test_score']:.3f}, "
                  f"action_mse={run['final_action_mse']:.4f}, "
                  f"epochs={run['total_epochs']}")
        
        if scores:
            print(f"\n  Mean Test Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(f"  Mean Action MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    else:
        print("  No completed BFN runs found.")
    
    # Diffusion Results
    print("\n📊 Diffusion Policy Results:")
    print("-" * 60)
    if diff_runs:
        scores = [r['best_test_score'] for r in diff_runs if r['best_test_score'] is not None]
        mses = [r['final_action_mse'] for r in diff_runs if r['final_action_mse'] is not None]
        
        for run in diff_runs:
            print(f"  Seed {run['seed']}: test_score={run['best_test_score']:.3f}, "
                  f"action_mse={run['final_action_mse']:.4f}, "
                  f"epochs={run['total_epochs']}")
        
        if scores:
            print(f"\n  Mean Test Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(f"  Mean Action MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    else:
        print("  No completed Diffusion runs found.")
    
    # Comparison
    print("\n" + "="*80)
    print("📈 COMPARISON SUMMARY")
    print("="*80)
    
    if bfn_runs and diff_runs:
        bfn_scores = [r['best_test_score'] for r in bfn_runs if r['best_test_score'] is not None]
        diff_scores = [r['best_test_score'] for r in diff_runs if r['best_test_score'] is not None]
        
        if bfn_scores and diff_scores:
            bfn_mean = np.mean(bfn_scores)
            diff_mean = np.mean(diff_scores)
            
            print(f"\n  BFN Mean Score:       {bfn_mean:.3f}")
            print(f"  Diffusion Mean Score: {diff_mean:.3f}")
            
            diff_pct = ((bfn_mean - diff_mean) / diff_mean) * 100
            if bfn_mean > diff_mean:
                print(f"\n  ✅ BFN outperforms Diffusion by {abs(diff_pct):.1f}%")
            elif diff_mean > bfn_mean:
                print(f"\n  ⚠️  Diffusion outperforms BFN by {abs(diff_pct):.1f}%")
            else:
                print(f"\n  ≈ Results are approximately equal")
    
    print("\n" + "="*80)


def export_to_latex(results: List[Dict], output_path: str = "benchmark_table.tex"):
    """Export results to LaTeX table for thesis."""
    
    bfn_runs = [r for r in results if r['policy_type'] == 'BFN' and r['state'] == 'finished']
    diff_runs = [r for r in results if r['policy_type'] == 'Diffusion' and r['state'] == 'finished']
    
    # Calculate statistics
    bfn_scores = [r['best_test_score'] for r in bfn_runs if r['best_test_score'] is not None]
    diff_scores = [r['best_test_score'] for r in diff_runs if r['best_test_score'] is not None]
    
    bfn_mse = [r['final_action_mse'] for r in bfn_runs if r['final_action_mse'] is not None]
    diff_mse = [r['final_action_mse'] for r in diff_runs if r['final_action_mse'] is not None]
    
    latex = r"""
\begin{table}[h]
\centering
\caption{BFN vs Diffusion Policy Comparison on PushT Task}
\label{tab:bfn_vs_diffusion}
\begin{tabular}{lcccc}
\toprule
\textbf{Policy} & \textbf{Test Score} $\uparrow$ & \textbf{Action MSE} $\downarrow$ & \textbf{Training Time} \\
\midrule
"""
    
    if bfn_scores:
        latex += f"BFN & ${np.mean(bfn_scores):.3f} \\pm {np.std(bfn_scores):.3f}$ "
        latex += f"& ${np.mean(bfn_mse):.4f} \\pm {np.std(bfn_mse):.4f}$ "
        latex += f"& {np.mean([r['runtime_hours'] for r in bfn_runs]):.1f}h \\\\\n"
    
    if diff_scores:
        latex += f"Diffusion & ${np.mean(diff_scores):.3f} \\pm {np.std(diff_scores):.3f}$ "
        latex += f"& ${np.mean(diff_mse):.4f} \\pm {np.std(diff_mse):.4f}$ "
        latex += f"& {np.mean([r['runtime_hours'] for r in diff_runs]):.1f}h \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table exported to: {output_path}")


def analyze_local_checkpoints(checkpoint_dir: str):
    """Analyze results from local checkpoint files."""
    import torch
    
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\nAnalyzing checkpoints in: {checkpoint_dir}")
    print("-" * 60)
    
    for ckpt_path in sorted(checkpoint_dir.glob("**/*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt.get('epoch', 'N/A')
            loss = ckpt.get('loss', ckpt.get('train_loss', 'N/A'))
            print(f"  {ckpt_path.name}: epoch={epoch}, loss={loss}")
        except Exception as e:
            print(f"  {ckpt_path.name}: Error loading - {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze BFN vs Diffusion benchmark results')
    parser.add_argument('--wandb-project', type=str, help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, help='WandB entity/username')
    parser.add_argument('--local-dir', type=str, help='Local checkpoint directory')
    parser.add_argument('--export-latex', action='store_true', help='Export results to LaTeX')
    parser.add_argument('--output', type=str, default='benchmark_results.json', 
                        help='Output file for results')
    
    args = parser.parse_args()
    
    results = []
    
    if args.wandb_project:
        if not HAS_WANDB:
            print("Error: wandb not installed. Install with: pip install wandb")
            return
        
        print(f"Fetching results from WandB project: {args.wandb_project}")
        results = get_wandb_runs(args.wandb_project, args.wandb_entity)
        
        # Print comparison
        print_comparison_table(results)
        
        # Export to LaTeX if requested
        if args.export_latex:
            export_to_latex(results)
        
        # Save raw results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nRaw results saved to: {args.output}")
    
    if args.local_dir:
        analyze_local_checkpoints(args.local_dir)
    
    if not args.wandb_project and not args.local_dir:
        print("Please specify --wandb-project or --local-dir")
        parser.print_help()


if __name__ == '__main__':
    main()
