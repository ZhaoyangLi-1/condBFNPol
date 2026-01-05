#!/usr/bin/env python3
"""
Publication-Quality Figures for BFN vs Diffusion Policy Comparison
Style: Google Research / NeurIPS / ICML format
"""

import argparse
import numpy as np
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# =============================================================================
# ANTHROPIC RESEARCH STYLE CONFIGURATION
# =============================================================================

from colors import COLORS, setup_matplotlib_style, SINGLE_COL, DOUBLE_COL

# Setup matplotlib style
setup_matplotlib_style()

# Height/Width ratio for figures
ASPECT = 0.75


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_checkpoint_scores(run_dir: Path) -> dict:
    """Extract evaluation scores from checkpoint filenames."""
    scores_by_epoch = {}
    checkpoint_dir = run_dir / 'checkpoints'
    if checkpoint_dir.exists():
        for ckpt in checkpoint_dir.glob("*.ckpt"):
            match = re.search(r'epoch=(\d+)-test_mean_score=([0-9]+\.[0-9]+)', ckpt.name)
            if match:
                epoch = int(match.group(1))
                score = float(match.group(2))
                scores_by_epoch[epoch] = score
    return scores_by_epoch


def parse_training_logs(log_file: Path) -> dict:
    """Parse training logs."""
    data = {'train_loss': [], 'global_step': [], 'epoch': []}
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if 'train_loss' in entry:
                    data['train_loss'].append(entry['train_loss'])
                    data['global_step'].append(entry.get('global_step', 0))
                    data['epoch'].append(entry.get('epoch', 0))
            except:
                continue
    return data


def load_all_data(base_dir: Path) -> dict:
    """Load all experimental data."""
    runs = {'BFN': [], 'Diffusion': []}
    
    for run_dir in sorted(base_dir.glob("**/logs.json.txt")):
        parent = run_dir.parent
        name = parent.name
        
        if 'bfn' in name.lower():
            policy = 'BFN'
        elif 'diffusion' in name.lower():
            policy = 'Diffusion'
        else:
            continue
        
        seed_match = re.search(r'seed(\d+)', name)
        seed = int(seed_match.group(1)) if seed_match else 0
        
        data = parse_training_logs(run_dir)
        data['ckpt_scores'] = parse_checkpoint_scores(parent)
        
        runs[policy].append({'seed': seed, 'data': data})
    
    return runs


# =============================================================================
# FIGURE 1: MAIN RESULTS BAR CHART
# =============================================================================

def plot_main_results(runs: dict, output_dir: Path):
    """Bar chart comparing final performance."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
    
    # Get scores
    bfn_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['BFN']]
    diff_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']]
    
    methods = ['BFN Policy\n(Ours)', 'Diffusion\nPolicy']
    means = [np.mean(bfn_scores), np.mean(diff_scores)]
    stds = [np.std(bfn_scores), np.std(diff_scores)]
    colors = [COLORS['bfn'], COLORS['diffusion']]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor=[COLORS['black']]*2, linewidth=0.8,
                  error_kw={'linewidth': 1, 'capthick': 1})
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    # Add horizontal line at best performance
    ax.axhline(y=max(means), color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig1_main_results.pdf'))
    fig.savefig(str(output_dir / 'fig1_main_results.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure 1: Main results bar chart")


# =============================================================================
# FIGURE 2: PER-SEED COMPARISON
# =============================================================================

def plot_per_seed(runs: dict, output_dir: Path):
    """Grouped bar chart showing per-seed results."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))
    
    seeds = [42, 43, 44]
    bfn_by_seed = {r['seed']: max(r['data']['ckpt_scores'].values()) for r in runs['BFN']}
    diff_by_seed = {r['seed']: max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']}
    
    x = np.arange(len(seeds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [bfn_by_seed[s] for s in seeds], width,
                   label='BFN (Ours)', color=COLORS['bfn'],
                   edgecolor=COLORS['black'], linewidth=0.5)
    bars2 = ax.bar(x + width/2, [diff_by_seed[s] for s in seeds], width,
                   label='Diffusion', color=COLORS['diffusion'],
                   edgecolor=COLORS['black'], linewidth=0.5)
    
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Random Seed')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0.8, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig2_per_seed.pdf'))
    fig.savefig(str(output_dir / 'fig2_per_seed.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure 2: Per-seed comparison")


# =============================================================================
# FIGURE 3: COMPUTATIONAL EFFICIENCY
# =============================================================================

def plot_efficiency(runs: dict, output_dir: Path):
    """Efficiency comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.7))
    
    bfn_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['BFN']]
    diff_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']]
    
    bfn_mean, diff_mean = np.mean(bfn_scores), np.mean(diff_scores)
    bfn_steps, diff_steps = 20, 100
    
    # Left: Inference steps comparison
    ax = axes[0]
    methods = ['BFN\n(Ours)', 'Diffusion']
    steps = [bfn_steps, diff_steps]
    colors = [COLORS['bfn'], COLORS['diffusion']]
    
    bars = ax.bar(methods, steps, color=colors, edgecolor=COLORS['black'], linewidth=0.8)
    ax.set_ylabel('Inference Steps')
    ax.set_ylim(0, 120)
    
    # Add speedup annotation
    ax.annotate('', xy=(0, bfn_steps + 5), xytext=(1, diff_steps + 5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.5))
    ax.text(0.5, (bfn_steps + diff_steps) / 2 + 15, '5× faster',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['bfn'])
    
    for bar, val in zip(bars, steps):
        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Right: Efficiency (score per 100 steps)
    ax = axes[1]
    efficiency = [bfn_mean / bfn_steps * 100, diff_mean / diff_steps * 100]
    
    bars = ax.bar(methods, efficiency, color=colors, edgecolor=COLORS['black'], linewidth=0.8)
    ax.set_ylabel('Efficiency\n(Score per 100 steps)')
    ax.set_ylim(0, 5.5)
    
    # Add efficiency ratio
    ax.annotate('', xy=(0, efficiency[0] + 0.2), xytext=(1, efficiency[1] + 0.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1.5))
    ax.text(0.5, (efficiency[0] + efficiency[1]) / 2 + 0.8, f'{efficiency[0]/efficiency[1]:.1f}× better',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['bfn'])
    
    for bar, val in zip(bars, efficiency):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Add panel labels
    axes[0].text(-0.15, 1.05, '(a)', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[1].text(-0.15, 1.05, '(b)', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig3_efficiency.pdf'))
    fig.savefig(str(output_dir / 'fig3_efficiency.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure 3: Computational efficiency")


# =============================================================================
# FIGURE 4: TRAINING LOSS CURVES
# =============================================================================

def plot_training_curves(runs: dict, output_dir: Path):
    """Training loss curves with confidence bands."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.8))
    
    for policy, color, light in [('BFN', COLORS['bfn'], COLORS['bfn_light']),
                                  ('Diffusion', COLORS['diffusion'], COLORS['diffusion_light'])]:
        all_losses = []
        all_steps = []
        
        for run in runs[policy]:
            losses = run['data']['train_loss']
            steps = run['data']['global_step']
            if losses:
                all_losses.append(losses)
                all_steps.append(steps)
        
        if not all_losses:
            continue
        
        # Subsample for cleaner plots (every 100 steps)
        min_len = min(len(l) for l in all_losses)
        subsample = 100
        indices = list(range(0, min_len, subsample))
        
        loss_arr = np.array([[l[i] for i in indices] for l in all_losses])
        step_arr = np.array([all_steps[0][i] for i in indices])
        
        mean_loss = np.mean(loss_arr, axis=0)
        std_loss = np.std(loss_arr, axis=0)
        
        # Plot with confidence band
        ax.fill_between(step_arr, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.2, color=color, linewidth=0)
        ax.plot(step_arr, mean_loss, color=color, linewidth=1.5,
                label=f'{policy}' if policy == 'Diffusion' else f'{policy} (Ours)')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Training Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    
    # Format x-axis with K notation
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else str(int(x))))
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig4_training_curves.pdf'))
    fig.savefig(str(output_dir / 'fig4_training_curves.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure 4: Training curves")


# =============================================================================
# FIGURE 5: COMBINED 4-PANEL FIGURE (MAIN PAPER FIGURE)
# =============================================================================

def plot_combined_figure(runs: dict, output_dir: Path):
    """Publication-ready 4-panel combined figure."""
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    
    # Create 2x2 grid with custom spacing (compatible with matplotlib 2.x)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.98, top=0.95, bottom=0.1)
    
    bfn_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['BFN']]
    diff_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']]
    bfn_mean, diff_mean = np.mean(bfn_scores), np.mean(diff_scores)
    bfn_std, diff_std = np.std(bfn_scores), np.std(diff_scores)
    
    # -------------------------------------------------------------------------
    # (a) Main results
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    
    methods = ['BFN (Ours)', 'Diffusion']
    means = [bfn_mean, diff_mean]
    stds = [bfn_std, diff_std]
    colors = [COLORS['bfn'], COLORS['diffusion']]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor=COLORS['black'], linewidth=0.5,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})
    
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.2f}', xy=(bar.get_x() + bar.get_width()/2, mean + 0.04),
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.text(-0.2, 1.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (b) Per-seed comparison
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    
    seeds = [42, 43, 44]
    bfn_by_seed = {r['seed']: max(r['data']['ckpt_scores'].values()) for r in runs['BFN']}
    diff_by_seed = {r['seed']: max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']}
    
    x = np.arange(len(seeds))
    width = 0.35
    
    ax.bar(x - width/2, [bfn_by_seed[s] for s in seeds], width,
           label='BFN (Ours)', color=COLORS['bfn'], edgecolor=COLORS['black'], linewidth=0.5)
    ax.bar(x + width/2, [diff_by_seed[s] for s in seeds], width,
           label='Diffusion', color=COLORS['diffusion'], edgecolor=COLORS['black'], linewidth=0.5)
    
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Seed')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc='lower right', fontsize=7)
    ax.text(-0.2, 1.08, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (c) Inference efficiency
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    
    bfn_steps, diff_steps = 20, 100
    efficiency = [bfn_mean / bfn_steps * 100, diff_mean / diff_steps * 100]
    
    bars = ax.bar(['BFN (Ours)', 'Diffusion'], efficiency,
                  color=colors, edgecolor=COLORS['black'], linewidth=0.5)
    
    ax.set_ylabel('Efficiency\n(Score / 100 steps)')
    
    # Highlight the advantage
    ax.annotate(f'{efficiency[0]/efficiency[1]:.1f}×', xy=(0, efficiency[0] + 0.15),
                ha='center', fontsize=9, fontweight='bold', color=COLORS['bfn'])
    
    for bar, val in zip(bars, efficiency):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords='offset points', ha='center', fontsize=7)
    
    ax.text(-0.2, 1.08, '(c)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (d) Training curves
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    
    for policy, color in [('BFN', COLORS['bfn']), ('Diffusion', COLORS['diffusion'])]:
        all_losses = [run['data']['train_loss'] for run in runs[policy] if run['data']['train_loss']]
        all_steps = [run['data']['global_step'] for run in runs[policy] if run['data']['global_step']]
        
        if not all_losses:
            continue
        
        min_len = min(len(l) for l in all_losses)
        subsample = 200
        indices = list(range(0, min_len, subsample))
        
        loss_arr = np.array([[l[i] for i in indices] for l in all_losses])
        step_arr = np.array([all_steps[0][i] for i in indices])
        
        mean_loss = np.mean(loss_arr, axis=0)
        std_loss = np.std(loss_arr, axis=0)
        
        ax.fill_between(step_arr, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.15, color=color)
        label = f'{policy} (Ours)' if policy == 'BFN' else policy
        ax.plot(step_arr, mean_loss, color=color, linewidth=1.2, label=label)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
    ax.text(-0.2, 1.08, '(d)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    fig.savefig(str(output_dir / 'fig_combined.pdf'))
    fig.savefig(str(output_dir / 'fig_combined.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure (combined): Publication-ready 4-panel figure")


# =============================================================================
# FIGURE 6: PARETO FRONTIER
# =============================================================================

def plot_pareto(runs: dict, output_dir: Path):
    """Performance vs Inference Cost scatter plot."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85))
    
    bfn_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['BFN']]
    diff_scores = [max(r['data']['ckpt_scores'].values()) for r in runs['Diffusion']]
    
    # Plot individual runs
    for score in bfn_scores:
        ax.scatter(20, score, c=COLORS['bfn'], s=60, alpha=0.7, edgecolor=COLORS['black'], linewidth=0.5)
    for score in diff_scores:
        ax.scatter(100, score, c=COLORS['diffusion'], s=60, alpha=0.7, edgecolor=COLORS['black'], linewidth=0.5)
    
    # Plot means with larger markers
    bfn_mean = np.mean(bfn_scores)
    diff_mean = np.mean(diff_scores)
    ax.scatter(20, bfn_mean, c=COLORS['bfn'], s=150, marker='*', edgecolor=COLORS['black'],
               linewidth=0.8, zorder=5, label=f'BFN (Ours): {bfn_mean:.2f}')
    ax.scatter(100, diff_mean, c=COLORS['diffusion'], s=150, marker='*', edgecolor=COLORS['black'],
               linewidth=0.8, zorder=5, label=f'Diffusion: {diff_mean:.2f}')
    
    # Add arrow showing trade-off
    ax.annotate('', xy=(25, bfn_mean), xytext=(95, diff_mean),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5, ls='--'))
    ax.text(55, (bfn_mean + diff_mean)/2 + 0.015, '5× faster\n4% accuracy loss',
            ha='center', va='bottom', fontsize=7, color=COLORS['gray'])
    
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Success Rate')
    ax.set_xlim(0, 120)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right', fontsize=7)
    
    # Shade Pareto-optimal region
    ax.fill_between([0, 25], [1.0, 1.0], [bfn_mean, bfn_mean], alpha=0.1, color=COLORS['bfn'])
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig5_pareto.pdf'))
    fig.savefig(str(output_dir / 'fig5_pareto.png'), dpi=300)
    plt.close(fig)
    print(f"  ✓ Figure 5: Pareto frontier")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality figures")
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='cluster_checkpoints/benchmarkresults',
        help='Directory containing benchmark results (default: cluster_checkpoints/benchmarkresults)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/publication',
        help='Output directory for figures (default: figures/publication)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication-Quality Figures")
    print("Style: Google Research / NeurIPS")
    print("=" * 60)
    
    if not base_dir.exists():
        print(f"ERROR: Data directory not found: {base_dir}")
        return
    
    runs = load_all_data(base_dir)
    print(f"\nLoaded {len(runs['BFN'])} BFN runs, {len(runs['Diffusion'])} Diffusion runs\n")
    
    # Generate all figures
    plot_main_results(runs, output_dir)
    plot_per_seed(runs, output_dir)
    plot_efficiency(runs, output_dir)
    plot_training_curves(runs, output_dir)
    plot_pareto(runs, output_dir)
    plot_combined_figure(runs, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  • {f.name}")


if __name__ == '__main__':
    main()
