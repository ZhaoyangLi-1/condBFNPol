#!/usr/bin/env python3
"""
Google Research Level Plot: Action Prediction MSE Error

Publication-quality visualization of action prediction error (MSE) comparing
BFN Policy and Diffusion Policy during training.

Style: Google Research / Diffusion Policy Paper
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import argparse

# =============================================================================
# GOOGLE RESEARCH STYLE
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
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
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
    """Parse logs.json.txt to extract metrics."""
    metrics = {
        'epochs': [],
        'train_action_mse_error': [],
        'train_action_mse_error_epochs': [],  # Track which epochs have MSE
        'train_loss': [],
        'test_mean_score': [],
    }
    
    if not log_file.exists():
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'epoch' in data:
                    metrics['epochs'].append(data['epoch'])
                if 'train_action_mse_error' in data:
                    metrics['train_action_mse_error'].append(data['train_action_mse_error'])
                    metrics['train_action_mse_error_epochs'].append(data['epoch'])  # Track epoch for MSE
                if 'train_loss' in data:
                    metrics['train_loss'].append(data['train_loss'])
                if 'test_mean_score' in data:
                    metrics['test_mean_score'].append(data['test_mean_score'])
            except json.JSONDecodeError:
                continue
    
    return metrics


def find_training_runs(base_dir: Path, method: str, seeds: List[int] = None) -> Dict[int, Path]:
    """Find all training run directories for a given method."""
    runs = {}
    
    if seeds is None:
        seeds = [42, 43, 44]
    
    if method.lower() == 'bfn':
        patterns = ['*bfn*seed*', '*bfn_seed*']
    elif method.lower() == 'diffusion':
        patterns = ['*diffusion*seed*', '*diffusion_seed*']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for pattern in patterns:
        for run_dir in base_dir.rglob(pattern):
            if run_dir.is_dir():
                log_file = run_dir / 'logs.json.txt'
                if log_file.exists():
                    dir_name = run_dir.name
                    seed = None
                    for s in seeds:
                        if f'seed{s}' in dir_name or f'_seed_{s}' in dir_name:
                            seed = s
                            break
                    
                    if seed is not None:
                        if seed not in runs or len(parse_training_logs(log_file)['epochs']) > len(parse_training_logs(runs[seed])['epochs']):
                            runs[seed] = log_file
    
    return runs


def aggregate_metrics(runs: Dict[int, Path], metric: str = 'train_action_mse_error') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate metrics across multiple seeds.
    
    Returns:
        epochs: Array of epoch numbers
        mean_metric: Mean across seeds
        std_metric: Standard deviation across seeds
    """
    all_epoch_value_pairs = []
    
    # Collect data from all seeds - match epochs with their metric values
    for seed, log_file in runs.items():
        data = parse_training_logs(log_file)
        values = np.array(data[metric])
        
        if len(values) == 0:
            continue
        
        # For train_action_mse_error, use the epochs where it was logged
        if metric == 'train_action_mse_error' and len(data['train_action_mse_error_epochs']) > 0:
            epochs = np.array(data['train_action_mse_error_epochs'])
        else:
            # For other metrics, use all epochs
            epochs = np.array(data['epochs'])
            # If lengths don't match, use the epochs that have values
            if len(epochs) != len(values):
                if len(epochs) > len(values):
                    # Values are subset - use last len(values) epochs
                    epochs = epochs[-len(values):]
                else:
                    # More values than epochs - use first len(epochs) values
                    values = values[:len(epochs)]
        
        if len(epochs) > 0 and len(values) > 0 and len(epochs) == len(values):
            all_epoch_value_pairs.append((epochs, values))
    
    if len(all_epoch_value_pairs) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Find common epoch range (intersection of all epoch sets)
    all_epoch_sets = [set(ep) for ep, _ in all_epoch_value_pairs]
    common_epoch_set = set.intersection(*all_epoch_sets) if len(all_epoch_sets) > 1 else all_epoch_sets[0]
    
    if len(common_epoch_set) == 0:
        # No common epochs - use union instead
        common_epoch_set = set.union(*all_epoch_sets) if len(all_epoch_sets) > 1 else all_epoch_sets[0]
    
    common_epochs = np.array(sorted(common_epoch_set))
    
    # Interpolate/extract values for common epochs
    interpolated_values = []
    
    for epochs, values in all_epoch_value_pairs:
        epochs = np.array(epochs)
        values = np.array(values)
        
        # Interpolate to common epochs
        # Only interpolate within the range of available epochs
        valid_mask = (common_epochs >= epochs.min()) & (common_epochs <= epochs.max())
        if valid_mask.sum() > 0:
            interp_values = np.interp(common_epochs[valid_mask], epochs, values)
            # Create full array with NaN for out-of-range epochs
            full_values = np.full(len(common_epochs), np.nan)
            full_values[valid_mask] = interp_values
            interpolated_values.append(full_values)
        else:
            # No overlap - skip this seed
            continue
    
    if len(interpolated_values) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Compute mean and std, handling NaN values
    interpolated_values = np.array(interpolated_values)
    
    # Remove epochs where all seeds have NaN
    valid_epoch_mask = ~np.isnan(interpolated_values).all(axis=0)
    common_epochs = common_epochs[valid_epoch_mask]
    interpolated_values = interpolated_values[:, valid_epoch_mask]
    
    # Compute mean and std, ignoring NaN
    mean_metric = np.nanmean(interpolated_values, axis=0)
    std_metric = np.nanstd(interpolated_values, axis=0)
    
    return common_epochs, mean_metric, std_metric


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_action_mse_error_comparison(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
    smooth: bool = True,
    show_std: bool = True,
):
    """
    Main plot: Action Prediction MSE Error comparison.
    Google Research style, single column figure.
    """
    fig = plt.figure(figsize=(SINGLE_COL, SINGLE_COL * 0.75))
    
    ax = fig.add_subplot(111)
    
    # Get aggregated data
    bfn_epochs, bfn_mean, bfn_std = aggregate_metrics(bfn_runs, 'train_action_mse_error')
    diff_epochs, diff_mean, diff_std = aggregate_metrics(diffusion_runs, 'train_action_mse_error')
    
    if len(bfn_epochs) == 0 and len(diff_epochs) == 0:
        print("Error: No action MSE error data found!")
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
                label='BFN Policy (Ours)', alpha=0.9, zorder=3)
        if show_std and len(bfn_std) > 0:
            ax.fill_between(bfn_epochs, bfn_mean - bfn_std, bfn_mean + bfn_std,
                           color=COLORS['bfn'], alpha=0.15, zorder=1)
    
    # Plot Diffusion
    if len(diff_epochs) > 0:
        ax.plot(diff_epochs, diff_mean, color=COLORS['diffusion'], linewidth=2.0,
                label='Diffusion Policy', alpha=0.9, zorder=3)
        if show_std and len(diff_std) > 0:
            ax.fill_between(diff_epochs, diff_mean - diff_std, diff_mean + diff_std,
                           color=COLORS['diffusion'], alpha=0.15, zorder=1)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Action Prediction MSE Error', fontsize=9)
    ax.set_title('Action Prediction Error During Training', fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor=COLORS['light_gray'], fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis to start from 0 or slightly below minimum
    if len(bfn_mean) > 0 and len(diff_mean) > 0:
        y_min = min(bfn_mean.min(), diff_mean.min())
        ax.set_ylim(bottom=max(0, y_min * 0.7))
    elif len(bfn_mean) > 0:
        ax.set_ylim(bottom=max(0, bfn_mean.min() * 0.7))
    elif len(diff_mean) > 0:
        ax.set_ylim(bottom=max(0, diff_mean.min() * 0.7))
    
    # Add final value annotations
    if len(bfn_mean) > 0:
        final_bfn = bfn_mean[-1]
        ax.text(0.98, 0.15, f'BFN: {final_bfn:.4f}', 
               transform=ax.transAxes, fontsize=7, color=COLORS['bfn'],
               ha='right', va='bottom', fontweight='bold')
    
    if len(diff_mean) > 0:
        final_diff = diff_mean[-1]
        ax.text(0.98, 0.08, f'Diffusion: {final_diff:.4f}', 
               transform=ax.transAxes, fontsize=7, color=COLORS['diffusion'],
               ha='right', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / 'action_mse_error.pdf')
    fig.savefig(str(output_dir / 'action_mse_error.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'action_mse_error.pdf'}")


def plot_action_mse_with_training_loss(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
    smooth: bool = True,
):
    """
    Two-panel figure: Action MSE Error and Training Loss.
    Google Research style, double column figure.
    """
    fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_COL * 0.75))
    
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.08, right=0.98, top=0.90, bottom=0.18)
    
    # ===================
    # LEFT PANEL: Action MSE Error
    # ===================
    ax1 = fig.add_subplot(gs[0])
    
    bfn_epochs, bfn_mean, bfn_std = aggregate_metrics(bfn_runs, 'train_action_mse_error')
    diff_epochs, diff_mean, diff_std = aggregate_metrics(diffusion_runs, 'train_action_mse_error')
    
    if smooth:
        if len(bfn_mean) > 0:
            bfn_mean = gaussian_filter1d(bfn_mean, sigma=2)
        if len(diff_mean) > 0:
            diff_mean = gaussian_filter1d(diff_mean, sigma=2)
    
    if len(bfn_epochs) > 0:
        ax1.plot(bfn_epochs, bfn_mean, color=COLORS['bfn'], linewidth=2.0,
                label='BFN Policy (Ours)', alpha=0.9)
        if len(bfn_std) > 0:
            ax1.fill_between(bfn_epochs, bfn_mean - bfn_std, bfn_mean + bfn_std,
                            color=COLORS['bfn'], alpha=0.15)
    
    if len(diff_epochs) > 0:
        ax1.plot(diff_epochs, diff_mean, color=COLORS['diffusion'], linewidth=2.0,
                label='Diffusion Policy', alpha=0.9)
        if len(diff_std) > 0:
            ax1.fill_between(diff_epochs, diff_mean - diff_std, diff_mean + diff_std,
                            color=COLORS['diffusion'], alpha=0.15)
    
    ax1.set_xlabel('Epoch', fontsize=9)
    ax1.set_ylabel('Action Prediction MSE Error', fontsize=9)
    ax1.set_title('(a) Action Prediction Error', fontweight='bold', pad=8)
    ax1.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor=COLORS['light_gray'], fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # ===================
    # RIGHT PANEL: Training Loss
    # ===================
    ax2 = fig.add_subplot(gs[1])
    
    bfn_epochs_loss, bfn_mean_loss, bfn_std_loss = aggregate_metrics(bfn_runs, 'train_loss')
    diff_epochs_loss, diff_mean_loss, diff_std_loss = aggregate_metrics(diffusion_runs, 'train_loss')
    
    if smooth:
        if len(bfn_mean_loss) > 0:
            bfn_mean_loss = gaussian_filter1d(bfn_mean_loss, sigma=2)
        if len(diff_mean_loss) > 0:
            diff_mean_loss = gaussian_filter1d(diff_mean_loss, sigma=2)
    
    if len(bfn_epochs_loss) > 0:
        ax2.plot(bfn_epochs_loss, bfn_mean_loss, color=COLORS['bfn'], linewidth=2.0,
                label='BFN Policy (Ours)', alpha=0.9)
        if len(bfn_std_loss) > 0:
            ax2.fill_between(bfn_epochs_loss, bfn_mean_loss - bfn_std_loss, bfn_mean_loss + bfn_std_loss,
                            color=COLORS['bfn'], alpha=0.15)
    
    if len(diff_epochs_loss) > 0:
        ax2.plot(diff_epochs_loss, diff_mean_loss, color=COLORS['diffusion'], linewidth=2.0,
                label='Diffusion Policy', alpha=0.9)
        if len(diff_std_loss) > 0:
            ax2.fill_between(diff_epochs_loss, diff_mean_loss - diff_std_loss, diff_mean_loss + diff_std_loss,
                            color=COLORS['diffusion'], alpha=0.15)
    
    ax2.set_xlabel('Epoch', fontsize=9)
    ax2.set_ylabel('Training Loss', fontsize=9)
    ax2.set_title('(b) Training Loss', fontweight='bold', pad=8)
    ax2.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor=COLORS['light_gray'], fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    fig.savefig(str(output_dir / 'action_mse_and_training_loss.pdf')
    fig.savefig(str(output_dir / 'action_mse_and_training_loss.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'action_mse_and_training_loss.pdf'}")


def plot_per_seed_comparison(
    bfn_runs: Dict[int, Path],
    diffusion_runs: Dict[int, Path],
    output_dir: Path,
    smooth: bool = True,
):
    """
    Plot action MSE error for each seed separately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.7))
    
    # BFN subplot
    ax1 = axes[0]
    for seed in sorted(bfn_runs.keys()):
        metrics = parse_training_logs(bfn_runs[seed])
        epochs = np.array(metrics['epochs'])
        mse_errors = np.array(metrics['train_action_mse_error'])
        
        if len(epochs) > 0 and len(mse_errors) > 0:
            if smooth and len(mse_errors) > 1:
                mse_errors = gaussian_filter1d(mse_errors, sigma=2)
            ax1.plot(epochs, mse_errors, color=COLORS['bfn'], linewidth=1.5,
                    alpha=0.6, label=f'Seed {seed}')
    
    ax1.set_xlabel('Epoch', fontsize=9)
    ax1.set_ylabel('Action Prediction MSE Error', fontsize=9)
    ax1.set_title('BFN Policy', fontweight='bold', pad=8)
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Diffusion subplot
    ax2 = axes[1]
    for seed in sorted(diffusion_runs.keys()):
        metrics = parse_training_logs(diffusion_runs[seed])
        epochs = np.array(metrics['epochs'])
        mse_errors = np.array(metrics['train_action_mse_error'])
        
        if len(epochs) > 0 and len(mse_errors) > 0:
            if smooth and len(mse_errors) > 1:
                mse_errors = gaussian_filter1d(mse_errors, sigma=2)
            ax2.plot(epochs, mse_errors, color=COLORS['diffusion'], linewidth=1.5,
                    alpha=0.6, label=f'Seed {seed}')
    
    ax2.set_xlabel('Epoch', fontsize=9)
    ax2.set_ylabel('Action Prediction MSE Error', fontsize=9)
    ax2.set_title('Diffusion Policy', fontweight='bold', pad=8)
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / 'action_mse_per_seed.pdf')
    fig.savefig(str(output_dir / 'action_mse_per_seed.png'), dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'action_mse_per_seed.pdf'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot action prediction MSE error (Google Research style)')
    parser.add_argument('--base-dir', type=str, default='cluster_checkpoints/benchmarkresults',
                       help='Base directory containing training outputs')
    parser.add_argument('--output-dir', type=str, default='figures/publication',
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
    print("Plotting Action Prediction MSE Error")
    print("Style: Google Research / Diffusion Policy Paper")
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
        n_mse = len(metrics['train_action_mse_error'])
        print(f"  Seed {seed}: {len(metrics['epochs'])} epochs, {n_mse} MSE measurements")
    
    print(f"Found Diffusion runs: {len(diffusion_runs)} seeds")
    for seed, log_file in diffusion_runs.items():
        metrics = parse_training_logs(log_file)
        n_mse = len(metrics['train_action_mse_error'])
        print(f"  Seed {seed}: {len(metrics['epochs'])} epochs, {n_mse} MSE measurements")
    print()
    
    if len(bfn_runs) == 0 and len(diffusion_runs) == 0:
        print("Error: No training runs found!")
        return
    
    # Check if we have action MSE data
    has_bfn_mse = any(len(parse_training_logs(log_file)['train_action_mse_error']) > 0 
                      for log_file in bfn_runs.values())
    has_diff_mse = any(len(parse_training_logs(log_file)['train_action_mse_error']) > 0 
                       for log_file in diffusion_runs.values())
    
    if not has_bfn_mse and not has_diff_mse:
        print("Warning: No action MSE error data found in logs!")
        print("  This metric is logged when sample_every > 0 in training config.")
        return
    
    # Generate plots
    smooth = not args.no_smooth
    show_std = not args.no_std
    
    print("Generating plots...")
    
    # Main plot
    plot_action_mse_error_comparison(
        bfn_runs, diffusion_runs, output_dir,
        smooth=smooth, show_std=show_std
    )
    
    # Additional plots if requested
    if args.all_plots:
        plot_action_mse_with_training_loss(bfn_runs, diffusion_runs, output_dir, smooth=smooth)
        plot_per_seed_comparison(bfn_runs, diffusion_runs, output_dir, smooth=smooth)
    
    print()
    print("=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

