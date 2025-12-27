#!/usr/bin/env python3
"""
Plot Training Loss Curves from Actual Training Logs

Extracts training loss from logs.json.txt files and creates publication-quality
visualizations comparing BFN and Diffusion Policy training curves.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import argparse

# =============================================================================
# STYLING (Google Research / Diffusion Policy Paper Style)
# =============================================================================

COLORS = {
    'bfn': '#4285F4',           # Google Blue
    'diffusion': '#EA4335',     # Google Red
    'gray': '#5F6368',
    'light_gray': '#E8EAED',
    'black': '#202124',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SINGLE_COL = 3.5
DOUBLE_COL = 7.16


# =============================================================================
# LOG PARSING
# =============================================================================

def parse_training_logs(log_file: Path) -> Dict:
    """
    Parse logs.json.txt file to extract training metrics.
    
    Returns:
        Dictionary with keys: 'epochs', 'train_loss', 'val_loss', 'test_mean_score'
    """
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'test_mean_score': [],
        'global_step': [],
    }
    
    if not log_file.exists():
        print(f"Warning: Log file not found: {log_file}")
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'epoch' in data:
                    metrics['epochs'].append(data['epoch'])
                if 'train_loss' in data:
                    metrics['train_loss'].append(data['train_loss'])
                if 'val_loss' in data:
                    metrics['val_loss'].append(data['val_loss'])
                if 'test_mean_score' in data:
                    metrics['test_mean_score'].append(data['test_mean_score'])
                if 'global_step' in data:
                    metrics['global_step'].append(data['global_step'])
            except json.JSONDecodeError:
                continue
    
    return metrics


def find_training_runs(base_dir: Path, method: str, seeds: List[int] = None) -> Dict[int, Path]:
    """
    Find all training run directories for a given method.
    
    Args:
        base_dir: Base directory containing training outputs
        method: 'bfn' or 'diffusion'
        seeds: Optional list of seeds to filter (e.g., [42, 43, 44])
    
    Returns:
        Dictionary mapping seed -> log file path
    """
    runs = {}
    
    # Search patterns
    if method.lower() == 'bfn':
        patterns = ['*bfn*seed*', '*bfn_seed*']
    elif method.lower() == 'diffusion':
        patterns = ['*diffusion*seed*', '*diffusion_seed*']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Search in base_dir and subdirectories
    for pattern in patterns:
        for run_dir in base_dir.rglob(pattern):
            if run_dir.is_dir():
                log_file = run_dir / 'logs.json.txt'
                if log_file.exists():
                    # Extract seed from directory name
                    dir_name = run_dir.name
                    seed = None
                    for s in (seeds or [42, 43, 44]):
                        if f'seed{s}' in dir_name or f'_seed_{s}' in dir_name:
                            seed = s
                            break
                    
                    if seed is not None:
                        if seed not in runs or len(parse_training_logs(log_file)['epochs']) > len(parse_training_logs(runs[seed])['epochs']):
                            runs[seed] = log_file
    
    return runs


def aggregate_losses(runs: Dict[int, Path], metric: str = 'train_loss') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate losses across multiple seeds.
    
    Returns:
        epochs: Array of epoch numbers
        mean_loss: Mean loss across seeds
        std_loss: Standard deviation across seeds
    """
    all_epochs = []
    all_losses = {}
    
    # Collect data from all seeds
    for seed, log_file in runs.items():
        metrics = parse_training_logs(log_file)
        epochs = np.array(metrics['epochs'])
        losses = np.array(metrics[metric])
        
        if len(epochs) > 0:
            all_epochs.append(epochs)
            all_losses[seed] = losses
    
    if len(all_epochs) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Find common epoch range
    min_epoch = max([e.min() for e in all_epochs])
    max_epoch = min([e.max() for e in all_epochs])
    
    # Interpolate to common epochs
    common_epochs = np.arange(min_epoch, max_epoch + 1)
    interpolated_losses = []
    
    for seed in sorted(all_losses.keys()):
        epochs = all_epochs[list(all_losses.keys()).index(seed)]
        losses = all_losses[seed]
        
        # Interpolate to common epochs
        interp_losses = np.interp(common_epochs, epochs, losses)
        interpolated_losses.append(interp_losses)
    
    # Compute mean and std
    interpolated_losses = np.array(interpolated_losses)
    mean_loss = np.mean(interpolated_losses, axis=0)
    std_loss = np.std(interpolated_losses, axis=0)
    
    return common_epochs, mean_loss, std_loss


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_loss_comparison(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
    smooth: bool = True,
    show_std: bool = True,
):
    """
    Plot training loss comparison between BFN and Diffusion Policy.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, SINGLE_COL * 0.7))
    
    # Get aggregated data
    bfn_epochs, bfn_mean, bfn_std = aggregate_losses(bfn_runs, 'train_loss')
    diff_epochs, diff_mean, diff_std = aggregate_losses(diffusion_runs, 'train_loss')
    
    if len(bfn_epochs) == 0 and len(diff_epochs) == 0:
        print("Error: No training data found!")
        return
    
    # Smooth if requested
    if smooth:
        if len(bfn_mean) > 0:
            bfn_mean = gaussian_filter1d(bfn_mean, sigma=2)
        if len(diff_mean) > 0:
            diff_mean = gaussian_filter1d(diff_mean, sigma=2)
    
    # Plot BFN
    if len(bfn_epochs) > 0:
        ax.plot(bfn_epochs, bfn_mean, color=COLORS['bfn'], linewidth=2.0,
                label='BFN Policy (Ours)', alpha=0.9)
        if show_std and len(bfn_std) > 0:
            ax.fill_between(bfn_epochs, bfn_mean - bfn_std, bfn_mean + bfn_std,
                           color=COLORS['bfn'], alpha=0.2)
    
    # Plot Diffusion
    if len(diff_epochs) > 0:
        ax.plot(diff_epochs, diff_mean, color=COLORS['diffusion'], linewidth=2.0,
                label='Diffusion Policy', alpha=0.9)
        if show_std and len(diff_std) > 0:
            ax.fill_between(diff_epochs, diff_mean - diff_std, diff_mean + diff_std,
                           color=COLORS['diffusion'], alpha=0.2)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Training Loss', fontsize=10)
    ax.set_title('Training Loss Comparison', fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor=COLORS['light_gray'], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis to start from 0 or slightly below minimum
    if len(bfn_mean) > 0 and len(diff_mean) > 0:
        y_min = min(bfn_mean.min(), diff_mean.min())
        ax.set_ylim(bottom=max(0, y_min * 0.8))
    elif len(bfn_mean) > 0:
        ax.set_ylim(bottom=max(0, bfn_mean.min() * 0.8))
    elif len(diff_mean) > 0:
        ax.set_ylim(bottom=max(0, diff_mean.min() * 0.8))
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / 'training_loss_comparison.pdf')
    fig.savefig(str(output_dir / 'training_loss_comparison.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'training_loss_comparison.pdf'}")


def plot_individual_seeds(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
    smooth: bool = True,
):
    """
    Plot training loss for each seed separately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.7))
    
    # BFN subplot
    ax1 = axes[0]
    for seed in sorted(bfn_runs.keys()):
        metrics = parse_training_logs(bfn_runs[seed])
        epochs = np.array(metrics['epochs'])
        losses = np.array(metrics['train_loss'])
        
        if len(epochs) > 0:
            if smooth and len(losses) > 1:
                losses = gaussian_filter1d(losses, sigma=2)
            ax1.plot(epochs, losses, color=COLORS['bfn'], linewidth=1.5,
                    alpha=0.6, label=f'Seed {seed}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('BFN Policy', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Diffusion subplot
    ax2 = axes[1]
    for seed in sorted(diffusion_runs.keys()):
        metrics = parse_training_logs(diffusion_runs[seed])
        epochs = np.array(metrics['epochs'])
        losses = np.array(metrics['train_loss'])
        
        if len(epochs) > 0:
            if smooth and len(losses) > 1:
                losses = gaussian_filter1d(losses, sigma=2)
            ax2.plot(epochs, losses, color=COLORS['diffusion'], linewidth=1.5,
                    alpha=0.6, label=f'Seed {seed}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Diffusion Policy', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / 'training_loss_per_seed.pdf')
    fig.savefig(str(output_dir / 'training_loss_per_seed.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'training_loss_per_seed.pdf'}")


def plot_loss_vs_performance(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
):
    """
    Scatter plot showing correlation between training loss and test performance.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, SINGLE_COL * 0.8))
    
    # Collect data
    for seed, log_file in bfn_runs.items():
        metrics = parse_training_logs(log_file)
        losses = np.array(metrics['train_loss'])
        scores = np.array(metrics.get('test_mean_score', []))
        
        if len(losses) > 0 and len(scores) > 0:
            # Match epochs
            epochs_loss = np.array(metrics['epochs'])
            epochs_score = np.array(metrics['epochs'])[:len(scores)]
            
            # Plot points where we have both
            common_epochs = np.intersect1d(epochs_loss, epochs_score)
            if len(common_epochs) > 0:
                loss_vals = [losses[epochs_loss == e][0] for e in common_epochs]
                score_vals = [scores[epochs_score == e][0] for e in common_epochs]
                ax.scatter(loss_vals, score_vals, color=COLORS['bfn'], s=30,
                          alpha=0.5, label=f'BFN (seed {seed})' if seed == list(bfn_runs.keys())[0] else '')
    
    for seed, log_file in diffusion_runs.items():
        metrics = parse_training_logs(log_file)
        losses = np.array(metrics['train_loss'])
        scores = np.array(metrics.get('test_mean_score', []))
        
        if len(losses) > 0 and len(scores) > 0:
            epochs_loss = np.array(metrics['epochs'])
            epochs_score = np.array(metrics['epochs'])[:len(scores)]
            
            common_epochs = np.intersect1d(epochs_loss, epochs_score)
            if len(common_epochs) > 0:
                loss_vals = [losses[epochs_loss == e][0] for e in common_epochs]
                score_vals = [scores[epochs_score == e][0] for e in common_epochs]
                ax.scatter(loss_vals, score_vals, color=COLORS['diffusion'], s=30,
                          alpha=0.5, label=f'Diffusion (seed {seed})' if seed == list(diffusion_runs.keys())[0] else '')
    
    ax.set_xlabel('Training Loss')
    ax.set_ylabel('Test Success Rate')
    ax.set_title('Loss vs Performance', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / 'loss_vs_performance.pdf')
    fig.savefig(str(output_dir / 'loss_vs_performance.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'loss_vs_performance.pdf'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves from logs')
    parser.add_argument('--base-dir', type=str, default='cluster_checkpoints/benchmarkresults',
                       help='Base directory containing training outputs')
    parser.add_argument('--output-dir', type=str, default='figures/training_loss',
                       help='Output directory for figures')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Seeds to include')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable smoothing')
    parser.add_argument('--no-std', action='store_true',
                       help='Hide standard deviation bands')
    parser.add_argument('--all-plots', action='store_true',
                       help='Generate all plot types')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Plotting Training Loss Curves")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seeds: {args.seeds}")
    print()
    
    # Find training runs
    print("Searching for training runs...")
    bfn_runs = find_training_runs(base_dir, 'bfn', args.seeds)
    diffusion_runs = find_training_runs(base_dir, 'diffusion', args.seeds)
    
    print(f"Found BFN runs: {len(bfn_runs)} seeds")
    for seed, log_file in bfn_runs.items():
        metrics = parse_training_logs(log_file)
        print(f"  Seed {seed}: {len(metrics['epochs'])} epochs")
    
    print(f"Found Diffusion runs: {len(diffusion_runs)} seeds")
    for seed, log_file in diffusion_runs.items():
        metrics = parse_training_logs(log_file)
        print(f"  Seed {seed}: {len(metrics['epochs'])} epochs")
    print()
    
    if len(bfn_runs) == 0 and len(diffusion_runs) == 0:
        print("Error: No training runs found!")
        print(f"  Searched in: {base_dir}")
        print("  Looking for directories containing 'bfn*seed*' or 'diffusion*seed*'")
        return
    
    # Generate plots
    smooth = not args.no_smooth
    show_std = not args.no_std
    
    print("Generating plots...")
    
    # Main comparison plot
    plot_training_loss_comparison(
        bfn_runs, diffusion_runs, output_dir,
        smooth=smooth, show_std=show_std
    )
    
    # Additional plots if requested
    if args.all_plots:
        plot_individual_seeds(bfn_runs, diffusion_runs, output_dir, smooth=smooth)
        plot_loss_vs_performance(bfn_runs, diffusion_runs, output_dir)
    
    print()
    print("=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

