#!/usr/bin/env python3
"""
Analyze BFN vs Diffusion benchmark results from local checkpoints.

Usage:
    python scripts/analyze_local_results.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

# Publication-quality settings
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

COLORS = {
    'BFN': '#2E86AB',        # Blue
    'Diffusion': '#E94F37',  # Red
}


def parse_logs(log_file: Path) -> Dict:
    """Parse logs.json.txt file to extract training metrics."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'test_mean_score': [],
        'train_action_mse_error': [],
    }
    
    if not log_file.exists():
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'epoch' in data:
                    metrics['epochs'].append(data['epoch'])
                if 'train_loss' in data:
                    metrics['train_loss'].append(data['train_loss'])
                if 'test_mean_score' in data:
                    metrics['test_mean_score'].append(data['test_mean_score'])
                if 'train_action_mse_error' in data:
                    metrics['train_action_mse_error'].append(data['train_action_mse_error'])
            except json.JSONDecodeError:
                continue
    
    return metrics


def get_best_score_from_checkpoints(checkpoint_dir: Path) -> Tuple[float, int]:
    """Extract best score and epoch from checkpoint filenames."""
    best_score = 0.0
    best_epoch = 0
    
    for ckpt in checkpoint_dir.glob("*.ckpt"):
        # Parse: epoch=0200-test_mean_score=0.877.ckpt
        match = re.search(r'epoch=(\d+)-test_mean_score=([0-9.]+)\.ckpt', ckpt.name)
        if match:
            epoch = int(match.group(1))
            score_str = match.group(2).rstrip('.')  # Remove trailing dot if any
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_epoch = epoch
    
    return best_score, best_epoch


def analyze_results(base_dir: Path) -> Dict:
    """Analyze all results in the benchmark directory."""
    results = {
        'BFN': {'scores': [], 'epochs': [], 'histories': []},
        'Diffusion': {'scores': [], 'epochs': [], 'histories': []},
    }
    
    # Find all run directories
    for run_dir in sorted(base_dir.glob("**/checkpoints")):
        run_name = run_dir.parent.name
        
        # Determine policy type and seed
        if 'bfn' in run_name.lower():
            policy_type = 'BFN'
        elif 'diffusion' in run_name.lower():
            policy_type = 'Diffusion'
        else:
            continue
        
        # Extract seed
        seed_match = re.search(r'seed(\d+)', run_name)
        seed = int(seed_match.group(1)) if seed_match else 0
        
        # Get best score from checkpoints
        best_score, best_epoch = get_best_score_from_checkpoints(run_dir)
        
        # Parse training logs
        log_file = run_dir.parent / 'logs.json.txt'
        history = parse_logs(log_file)
        
        results[policy_type]['scores'].append(best_score)
        results[policy_type]['epochs'].append(best_epoch)
        results[policy_type]['histories'].append({
            'seed': seed,
            'history': history,
            'best_score': best_score,
            'best_epoch': best_epoch,
        })
        
        print(f"{policy_type} seed {seed}: best_score={best_score:.3f} at epoch {best_epoch}")
    
    return results


def print_comparison_table(results: Dict):
    """Print a formatted comparison table."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS: BFN vs Diffusion on PushT")
    print("="*70)
    
    for policy_type in ['BFN', 'Diffusion']:
        scores = results[policy_type]['scores']
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"\n{policy_type}:")
            print(f"  Scores: {scores}")
            print(f"  Mean ± Std: {mean:.3f} ± {std:.3f}")
    
    # Statistical comparison
    bfn_scores = results['BFN']['scores']
    diff_scores = results['Diffusion']['scores']
    
    if bfn_scores and diff_scores:
        bfn_mean = np.mean(bfn_scores)
        diff_mean = np.mean(diff_scores)
        
        print("\n" + "-"*70)
        print("COMPARISON:")
        print(f"  BFN Mean:       {bfn_mean:.3f}")
        print(f"  Diffusion Mean: {diff_mean:.3f}")
        print(f"  Difference:     {diff_mean - bfn_mean:.3f} ({(diff_mean - bfn_mean)/bfn_mean*100:.1f}%)")
        
        if diff_mean > bfn_mean:
            print(f"\n  → Diffusion outperforms BFN by {(diff_mean - bfn_mean)/bfn_mean*100:.1f}%")
        else:
            print(f"\n  → BFN outperforms Diffusion by {(bfn_mean - diff_mean)/diff_mean*100:.1f}%")
    
    print("="*70)


def generate_latex_table(results: Dict, output_path: Path):
    """Generate LaTeX table for thesis."""
    bfn_scores = results['BFN']['scores']
    diff_scores = results['Diffusion']['scores']
    
    latex = r"""\begin{table}[h]
\centering
\caption{Comparison of BFN and Diffusion Policy on PushT Task. 
Results show test success rate averaged over 3 random seeds.}
\label{tab:bfn_vs_diffusion}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Seed 42} & \textbf{Seed 43} & \textbf{Seed 44} & \textbf{Mean} & \textbf{Std} \\
\midrule
"""
    
    # BFN row
    if len(bfn_scores) >= 3:
        latex += f"BFN & {bfn_scores[0]:.3f} & {bfn_scores[1]:.3f} & {bfn_scores[2]:.3f} "
        latex += f"& {np.mean(bfn_scores):.3f} & {np.std(bfn_scores):.3f} \\\\\n"
    
    # Diffusion row
    if len(diff_scores) >= 3:
        latex += f"Diffusion & {diff_scores[0]:.3f} & {diff_scores[1]:.3f} & {diff_scores[2]:.3f} "
        latex += f"& {np.mean(diff_scores):.3f} & {np.std(diff_scores):.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_path}")
    print("\n" + latex)


def plot_training_curves(results: Dict, output_dir: Path):
    """Plot training curves comparison."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots - matplotlib not installed")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot test scores over epochs
    for policy_type in ['BFN', 'Diffusion']:
        color = COLORS[policy_type]
        histories = results[policy_type]['histories']
        
        all_scores = []
        for h in histories:
            if h['history']['test_mean_score']:
                all_scores.append(h['history']['test_mean_score'])
        
        if all_scores:
            # Find minimum length and truncate
            min_len = min(len(s) for s in all_scores)
            scores_arr = np.array([s[:min_len] for s in all_scores])
            mean_scores = np.mean(scores_arr, axis=0)
            std_scores = np.std(scores_arr, axis=0)
            
            epochs = np.arange(len(mean_scores)) * 50  # Assuming rollout_every=50
            
            axes[0].plot(epochs, mean_scores, color=color, label=policy_type, linewidth=2)
            axes[0].fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                                 color=color, alpha=0.2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Success Rate')
    axes[0].set_title('(a) Test Performance During Training')
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)
    
    # Bar chart of final scores
    policies = ['BFN', 'Diffusion']
    means = [np.mean(results[p]['scores']) for p in policies]
    stds = [np.std(results[p]['scores']) for p in policies]
    colors = [COLORS[p] for p in policies]
    
    x = np.arange(len(policies))
    bars = axes[1].bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    axes[1].set_ylabel('Test Success Rate')
    axes[1].set_title('(b) Final Performance Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies)
    axes[1].set_ylim(0, 1.0)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        axes[1].annotate(f'{mean:.3f}±{std:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, mean + std + 0.02),
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'benchmark_comparison.pdf')
    fig.savefig(output_dir / 'benchmark_comparison.png')
    plt.close(fig)
    
    print(f"\nPlots saved to: {output_dir}/benchmark_comparison.pdf")


def plot_per_seed_comparison(results: Dict, output_dir: Path):
    """Plot per-seed comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    seeds = [42, 43, 44]
    x = np.arange(len(seeds))
    width = 0.35
    
    bfn_scores = results['BFN']['scores']
    diff_scores = results['Diffusion']['scores']
    
    bars1 = ax.bar(x - width/2, bfn_scores, width, label='BFN', color=COLORS['BFN'], edgecolor='black')
    bars2 = ax.bar(x + width/2, diff_scores, width, label='Diffusion', color=COLORS['Diffusion'], edgecolor='black')
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Test Success Rate')
    ax.set_title('Per-Seed Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'per_seed_comparison.pdf')
    fig.savefig(output_dir / 'per_seed_comparison.png')
    plt.close(fig)
    
    print(f"Per-seed plot saved to: {output_dir}/per_seed_comparison.pdf")


def main():
    # Find the benchmark results directory
    base_paths = [
        Path("cluster_checkpoints/benchmarkresults"),
        Path("cluster_checkpoints"),
        Path("outputs"),
    ]
    
    base_dir = None
    for path in base_paths:
        if path.exists():
            base_dir = path
            break
    
    if base_dir is None:
        print("ERROR: Could not find benchmark results directory")
        print("Expected one of:", base_paths)
        return
    
    print(f"Analyzing results in: {base_dir}")
    print("-" * 70)
    
    # Analyze results
    results = analyze_results(base_dir)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    # Generate LaTeX table
    generate_latex_table(results, output_dir / "results_table.tex")
    
    # Generate plots
    plot_training_curves(results, output_dir)
    plot_per_seed_comparison(results, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - figures/results_table.tex (LaTeX table for thesis)")
    print(f"  - figures/benchmark_comparison.pdf (Training curves)")
    print(f"  - figures/per_seed_comparison.pdf (Per-seed bars)")


if __name__ == '__main__':
    main()
