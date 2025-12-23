#!/usr/bin/env python3
"""
Visualization Script for BFN vs Diffusion Benchmark

Generates publication-quality figures for thesis/paper comparing
Bayesian Flow Networks and Diffusion Policies on PushT task.

Usage:
    # From WandB data
    python scripts/visualize_benchmark.py --wandb-project pusht-benchmark --wandb-entity aleyna-research
    
    # From local logs
    python scripts/visualize_benchmark.py --local-dir outputs/
    
    # Generate all figures
    python scripts/visualize_benchmark.py --wandb-project pusht-benchmark --all-figures

Output:
    figures/
    ├── training_curves.pdf
    ├── performance_comparison.pdf
    ├── action_trajectories.pdf
    ├── inference_time.pdf
    └── ablation_timesteps.pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Color scheme for policies
COLORS = {
    'BFN': '#2E86AB',        # Blue
    'Diffusion': '#E94F37',  # Red/Orange
    'Flow': '#8ACB88',       # Green (if you add Flow Matching)
}

MARKERS = {
    'BFN': 'o',
    'Diffusion': 's',
    'Flow': '^',
}


def setup_output_dir(output_dir: str = 'figures') -> Path:
    """Create output directory for figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ==================== WandB Data Loading ====================

def load_wandb_data(project: str, entity: Optional[str] = None) -> Dict:
    """Load training history from WandB."""
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb required. Install with: pip install wandb")
    
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)
    
    data = {'BFN': [], 'Diffusion': []}
    
    for run in runs:
        if run.state != 'finished':
            continue
            
        # Determine policy type
        name_lower = run.name.lower()
        config_name = run.config.get('name', '').lower()
        
        if 'bfn' in name_lower or 'bfn' in config_name:
            policy_type = 'BFN'
        elif 'diffusion' in name_lower or 'diffusion' in config_name:
            policy_type = 'Diffusion'
        else:
            continue
        
        # Get training history
        history = run.history(samples=1000)
        
        run_data = {
            'name': run.name,
            'id': run.id,
            'seed': run.config.get('training', {}).get('seed', 42),
            'history': {
                'epoch': history['epoch'].tolist() if 'epoch' in history else [],
                'loss': history['train/loss'].tolist() if 'train/loss' in history else [],
                'action_mse': history['train/action_mse_error'].tolist() if 'train/action_mse_error' in history else [],
                'test_score': history['test_mean_score'].tolist() if 'test_mean_score' in history else [],
            },
            'summary': {
                'final_loss': run.summary.get('train/loss'),
                'final_action_mse': run.summary.get('train/action_mse_error'),
                'best_test_score': run.summary.get('test_mean_score'),
                'runtime_hours': run.summary.get('_runtime', 0) / 3600,
            }
        }
        
        data[policy_type].append(run_data)
    
    return data


def load_local_data(log_dir: str) -> Dict:
    """Load training data from local log files."""
    import torch
    
    log_path = Path(log_dir)
    data = {'BFN': [], 'Diffusion': []}
    
    # Find all output directories
    for run_dir in log_path.glob("**/"):
        # Check for training logs
        log_file = run_dir / 'train.log'
        metrics_file = run_dir / 'metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Determine policy type from directory name
            dir_name = run_dir.name.lower()
            if 'bfn' in dir_name:
                policy_type = 'BFN'
            elif 'diffusion' in dir_name:
                policy_type = 'Diffusion'
            else:
                continue
            
            data[policy_type].append({
                'name': run_dir.name,
                'history': metrics.get('history', {}),
                'summary': metrics.get('summary', {}),
            })
    
    return data


# ==================== Plotting Functions ====================

def plot_training_curves(data: Dict, output_dir: Path, show_std: bool = True):
    """
    Plot training loss and action MSE curves.
    
    Creates a 2-panel figure showing:
    - Left: Training loss over epochs
    - Right: Action MSE error over epochs
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if not runs:
            continue
        
        color = COLORS[policy_type]
        
        # Collect data from all runs
        all_losses = []
        all_mses = []
        all_epochs = []
        
        for run in runs:
            history = run.get('history', {})
            if history.get('loss') and history.get('epoch'):
                epochs = history['epoch']
                losses = history['loss']
                all_epochs.append(epochs)
                all_losses.append(losses)
            if history.get('action_mse'):
                all_mses.append(history['action_mse'])
        
        # Plot loss (left panel)
        if all_losses:
            # Find common epoch range
            min_len = min(len(l) for l in all_losses)
            losses_arr = np.array([l[:min_len] for l in all_losses])
            epochs_arr = np.array(all_epochs[0][:min_len])
            
            mean_loss = np.mean(losses_arr, axis=0)
            std_loss = np.std(losses_arr, axis=0)
            
            axes[0].plot(epochs_arr, mean_loss, color=color, label=policy_type, linewidth=2)
            if show_std and len(all_losses) > 1:
                axes[0].fill_between(epochs_arr, mean_loss - std_loss, mean_loss + std_loss,
                                     color=color, alpha=0.2)
        
        # Plot action MSE (right panel)
        if all_mses:
            min_len = min(len(m) for m in all_mses)
            mses_arr = np.array([m[:min_len] for m in all_mses])
            epochs_arr = np.array(all_epochs[0][:min_len]) if all_epochs else np.arange(min_len)
            
            mean_mse = np.mean(mses_arr, axis=0)
            std_mse = np.std(mses_arr, axis=0)
            
            axes[1].plot(epochs_arr, mean_mse, color=color, label=policy_type, linewidth=2)
            if show_std and len(all_mses) > 1:
                axes[1].fill_between(epochs_arr, mean_mse - std_mse, mean_mse + std_mse,
                                     color=color, alpha=0.2)
    
    # Format left panel (Loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('(a) Training Loss')
    axes[0].legend(loc='upper right')
    axes[0].set_yscale('log')
    
    # Format right panel (Action MSE)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Action MSE Error')
    axes[1].set_title('(b) Action Prediction Error')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save in multiple formats
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'training_curves.pdf')
    fig.savefig(output_dir / 'training_curves.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'training_curves.pdf'}")


def plot_performance_comparison(data: Dict, output_dir: Path):
    """
    Plot bar chart comparing final performance metrics.
    
    Creates grouped bar chart showing:
    - Test success rate
    - Final action MSE
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Collect summary statistics
    metrics = {}
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if not runs:
            continue
        
        test_scores = [r['summary'].get('best_test_score', 0) for r in runs 
                       if r['summary'].get('best_test_score') is not None]
        action_mses = [r['summary'].get('final_action_mse', 0) for r in runs 
                       if r['summary'].get('final_action_mse') is not None]
        
        metrics[policy_type] = {
            'test_score_mean': np.mean(test_scores) if test_scores else 0,
            'test_score_std': np.std(test_scores) if test_scores else 0,
            'action_mse_mean': np.mean(action_mses) if action_mses else 0,
            'action_mse_std': np.std(action_mses) if action_mses else 0,
        }
    
    # Bar positions
    policies = list(metrics.keys())
    x = np.arange(len(policies))
    width = 0.6
    
    # Left panel: Test Score
    test_means = [metrics[p]['test_score_mean'] for p in policies]
    test_stds = [metrics[p]['test_score_std'] for p in policies]
    colors = [COLORS[p] for p in policies]
    
    bars1 = axes[0].bar(x, test_means, width, yerr=test_stds, capsize=5,
                        color=colors, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Test Success Rate')
    axes[0].set_title('(a) Task Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies)
    axes[0].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, test_means, test_stds):
        height = bar.get_height()
        axes[0].annotate(f'{mean:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    # Right panel: Action MSE
    mse_means = [metrics[p]['action_mse_mean'] for p in policies]
    mse_stds = [metrics[p]['action_mse_std'] for p in policies]
    
    bars2 = axes[1].bar(x, mse_means, width, yerr=mse_stds, capsize=5,
                        color=colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Action MSE Error')
    axes[1].set_title('(b) Prediction Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies)
    
    # Add value labels
    for bar, mean in zip(bars2, mse_means):
        height = bar.get_height()
        axes[1].annotate(f'{mean:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'performance_comparison.pdf')
    fig.savefig(output_dir / 'performance_comparison.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'performance_comparison.pdf'}")


def plot_inference_time_comparison(output_dir: Path, 
                                   bfn_steps: int = 20, 
                                   diffusion_steps: int = 100):
    """
    Plot inference time comparison between BFN and Diffusion.
    
    This is a theoretical comparison based on number of network forward passes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    policies = ['BFN', 'Diffusion']
    steps = [bfn_steps, diffusion_steps]
    colors = [COLORS[p] for p in policies]
    
    x = np.arange(len(policies))
    bars = ax.bar(x, steps, width=0.6, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Denoising Steps (Network Forward Passes)')
    ax.set_title('Inference Complexity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    
    # Add value labels
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        ax.annotate(f'{step}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add speedup annotation
    speedup = diffusion_steps / bfn_steps
    ax.annotate(f'{speedup:.1f}× fewer steps',
               xy=(0.5, max(steps) * 0.5),
               ha='center', fontsize=11, color='green',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'inference_time.pdf')
    fig.savefig(output_dir / 'inference_time.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'inference_time.pdf'}")


def plot_test_score_over_training(data: Dict, output_dir: Path):
    """
    Plot test score evolution during training.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if not runs:
            continue
        
        color = COLORS[policy_type]
        
        # Collect test scores from all runs
        all_scores = []
        all_epochs = []
        
        for run in runs:
            history = run.get('history', {})
            if history.get('test_score') and history.get('epoch'):
                # Filter out None values
                epochs = history['epoch']
                scores = history['test_score']
                valid_idx = [i for i, s in enumerate(scores) if s is not None and not np.isnan(s)]
                
                if valid_idx:
                    all_epochs.append([epochs[i] for i in valid_idx])
                    all_scores.append([scores[i] for i in valid_idx])
        
        if all_scores:
            # Plot individual runs with low alpha
            for epochs, scores in zip(all_epochs, all_scores):
                ax.plot(epochs, scores, color=color, alpha=0.3, linewidth=1)
            
            # Plot mean with markers
            # Interpolate to common epochs for mean calculation
            common_epochs = sorted(set(e for eps in all_epochs for e in eps))
            interpolated = []
            for epochs, scores in zip(all_epochs, all_scores):
                interp_scores = np.interp(common_epochs, epochs, scores)
                interpolated.append(interp_scores)
            
            if interpolated:
                mean_scores = np.mean(interpolated, axis=0)
                ax.plot(common_epochs, mean_scores, color=color, linewidth=2.5,
                       label=policy_type, marker=MARKERS[policy_type], 
                       markevery=max(1, len(common_epochs)//10), markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Success Rate')
    ax.set_title('Test Performance During Training')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'test_score_training.pdf')
    fig.savefig(output_dir / 'test_score_training.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'test_score_training.pdf'}")


def plot_seed_comparison(data: Dict, output_dir: Path):
    """
    Plot performance comparison across different seeds.
    Shows reproducibility and variance.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect data by seed
    results = []
    
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        for run in runs:
            seed = run.get('seed', 0)
            score = run['summary'].get('best_test_score')
            if score is not None:
                results.append({
                    'policy': policy_type,
                    'seed': seed,
                    'score': score
                })
    
    if not results:
        print("No seed data available for plotting")
        return
    
    # Create grouped bar chart
    seeds = sorted(set(r['seed'] for r in results))
    x = np.arange(len(seeds))
    width = 0.35
    
    for i, policy_type in enumerate(['BFN', 'Diffusion']):
        scores = []
        for seed in seeds:
            score = next((r['score'] for r in results 
                         if r['policy'] == policy_type and r['seed'] == seed), 0)
            scores.append(score)
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=policy_type, 
                     color=COLORS[policy_type], edgecolor='black')
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Test Success Rate')
    ax.set_title('Performance Across Random Seeds')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'seed_comparison.pdf')
    fig.savefig(output_dir / 'seed_comparison.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'seed_comparison.pdf'}")


def plot_combined_summary(data: Dict, output_dir: Path):
    """
    Create a combined 2x2 summary figure for the paper.
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel (a): Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if not runs:
            continue
        
        all_losses = []
        for run in runs:
            history = run.get('history', {})
            if history.get('loss'):
                all_losses.append(history['loss'])
        
        if all_losses:
            min_len = min(len(l) for l in all_losses)
            losses_arr = np.array([l[:min_len] for l in all_losses])
            mean_loss = np.mean(losses_arr, axis=0)
            
            ax1.plot(mean_loss, color=COLORS[policy_type], label=policy_type, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Panel (b): Test Score
    ax2 = fig.add_subplot(gs[0, 1])
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if not runs:
            continue
        
        all_scores = []
        for run in runs:
            history = run.get('history', {})
            if history.get('test_score'):
                scores = [s for s in history['test_score'] if s is not None]
                if scores:
                    all_scores.append(scores)
        
        if all_scores:
            min_len = min(len(s) for s in all_scores)
            scores_arr = np.array([s[:min_len] for s in all_scores])
            mean_scores = np.mean(scores_arr, axis=0)
            
            ax2.plot(mean_scores, color=COLORS[policy_type], label=policy_type, linewidth=2)
    
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('(b) Test Performance')
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    
    # Panel (c): Final Performance Bars
    ax3 = fig.add_subplot(gs[1, 0])
    policies = []
    means = []
    stds = []
    colors = []
    
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if runs:
            scores = [r['summary'].get('best_test_score', 0) for r in runs 
                     if r['summary'].get('best_test_score') is not None]
            if scores:
                policies.append(policy_type)
                means.append(np.mean(scores))
                stds.append(np.std(scores))
                colors.append(COLORS[policy_type])
    
    if policies:
        x = np.arange(len(policies))
        ax3.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(policies)
        ax3.set_ylabel('Success Rate')
        ax3.set_title('(c) Final Performance')
        ax3.set_ylim(0, 1.0)
    
    # Panel (d): Inference Steps
    ax4 = fig.add_subplot(gs[1, 1])
    policies = ['BFN', 'Diffusion']
    steps = [20, 100]  # Default values
    colors = [COLORS[p] for p in policies]
    
    x = np.arange(len(policies))
    bars = ax4.bar(x, steps, color=colors, edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(policies)
    ax4.set_ylabel('Denoising Steps')
    ax4.set_title('(d) Inference Complexity')
    
    # Add speedup annotation
    ax4.annotate(f'{steps[1]/steps[0]:.0f}× fewer',
                xy=(0, steps[0] + 5), ha='center', fontsize=10, color='green')
    
    plt.suptitle('BFN vs Diffusion Policy Comparison on PushT', fontsize=14, fontweight='bold')
    
    fig.savefig(output_dir / 'combined_summary.pdf')
    fig.savefig(output_dir / 'combined_summary.png')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'combined_summary.pdf'}")


def create_latex_table(data: Dict, output_dir: Path):
    """Generate a LaTeX table for the paper."""
    
    results = {}
    for policy_type in ['BFN', 'Diffusion']:
        runs = data.get(policy_type, [])
        if runs:
            scores = [r['summary'].get('best_test_score', 0) for r in runs 
                     if r['summary'].get('best_test_score') is not None]
            mses = [r['summary'].get('final_action_mse', 0) for r in runs 
                   if r['summary'].get('final_action_mse') is not None]
            times = [r['summary'].get('runtime_hours', 0) for r in runs]
            
            results[policy_type] = {
                'score_mean': np.mean(scores) if scores else 0,
                'score_std': np.std(scores) if scores else 0,
                'mse_mean': np.mean(mses) if mses else 0,
                'mse_std': np.std(mses) if mses else 0,
                'time_mean': np.mean(times) if times else 0,
            }
    
    latex = r"""\begin{table}[t]
\centering
\caption{Comparison of BFN and Diffusion policies on the PushT manipulation task. 
Results are averaged over 3 random seeds. $\uparrow$ indicates higher is better, 
$\downarrow$ indicates lower is better.}
\label{tab:pusht_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Success Rate} $\uparrow$ & \textbf{Action MSE} $\downarrow$ & \textbf{Steps} \\
\midrule
"""
    
    if 'Diffusion' in results:
        r = results['Diffusion']
        latex += f"Diffusion & ${r['score_mean']:.3f} \\pm {r['score_std']:.3f}$ "
        latex += f"& ${r['mse_mean']:.4f} \\pm {r['mse_std']:.4f}$ & 100 \\\\\n"
    
    if 'BFN' in results:
        r = results['BFN']
        latex += f"BFN (Ours) & ${r['score_mean']:.3f} \\pm {r['score_std']:.3f}$ "
        latex += f"& ${r['mse_mean']:.4f} \\pm {r['mse_std']:.4f}$ & 20 \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_file = output_dir / 'results_table.tex'
    with open(output_file, 'w') as f:
        f.write(latex)
    
    print(f"Saved: {output_file}")
    print("\nLaTeX table content:")
    print(latex)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for BFN vs Diffusion benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all figures from WandB
    python scripts/visualize_benchmark.py --wandb-project pusht-benchmark --all-figures
    
    # Generate specific figure
    python scripts/visualize_benchmark.py --wandb-project pusht-benchmark --training-curves
    
    # Generate LaTeX table only
    python scripts/visualize_benchmark.py --wandb-project pusht-benchmark --latex-table
        """
    )
    
    # Data source
    parser.add_argument('--wandb-project', type=str, help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, help='WandB entity/username')
    parser.add_argument('--local-dir', type=str, help='Local log directory')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    
    # Figure selection
    parser.add_argument('--all-figures', action='store_true', help='Generate all figures')
    parser.add_argument('--training-curves', action='store_true', help='Training loss/MSE curves')
    parser.add_argument('--performance', action='store_true', help='Performance comparison bars')
    parser.add_argument('--inference-time', action='store_true', help='Inference time comparison')
    parser.add_argument('--test-score', action='store_true', help='Test score over training')
    parser.add_argument('--seed-comparison', action='store_true', help='Seed comparison')
    parser.add_argument('--combined', action='store_true', help='Combined 2x2 summary')
    parser.add_argument('--latex-table', action='store_true', help='Generate LaTeX table')
    
    args = parser.parse_args()
    
    # Load data
    data = None
    if args.wandb_project:
        print(f"Loading data from WandB: {args.wandb_project}")
        data = load_wandb_data(args.wandb_project, args.wandb_entity)
    elif args.local_dir:
        print(f"Loading data from local: {args.local_dir}")
        data = load_local_data(args.local_dir)
    else:
        print("Please specify --wandb-project or --local-dir")
        parser.print_help()
        return
    
    # Print summary
    print(f"\nLoaded data:")
    print(f"  BFN runs: {len(data.get('BFN', []))}")
    print(f"  Diffusion runs: {len(data.get('Diffusion', []))}")
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate requested figures
    generate_all = args.all_figures or not any([
        args.training_curves, args.performance, args.inference_time,
        args.test_score, args.seed_comparison, args.combined, args.latex_table
    ])
    
    if generate_all or args.training_curves:
        print("\nGenerating training curves...")
        plot_training_curves(data, output_dir)
    
    if generate_all or args.performance:
        print("\nGenerating performance comparison...")
        plot_performance_comparison(data, output_dir)
    
    if generate_all or args.inference_time:
        print("\nGenerating inference time comparison...")
        plot_inference_time_comparison(output_dir)
    
    if generate_all or args.test_score:
        print("\nGenerating test score evolution...")
        plot_test_score_over_training(data, output_dir)
    
    if generate_all or args.seed_comparison:
        print("\nGenerating seed comparison...")
        plot_seed_comparison(data, output_dir)
    
    if generate_all or args.combined:
        print("\nGenerating combined summary figure...")
        plot_combined_summary(data, output_dir)
    
    if generate_all or args.latex_table:
        print("\nGenerating LaTeX table...")
        create_latex_table(data, output_dir)
    
    print(f"\n✅ Done! Figures saved to: {output_dir}/")


if __name__ == '__main__':
    main()
