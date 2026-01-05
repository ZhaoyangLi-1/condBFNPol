#!/usr/bin/env python3
"""
Plot: Inference Time vs. Task Performance

Generates a publication-quality figure comparing BFN-Policy and Diffusion Policy
at equal computational budgets. This directly addresses the question:
"What happens when we run diffusion with fewer steps to match BFN's speed?"

Usage:
    python scripts/plot_time_vs_performance.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Anthropic color palette
COLORS = {
    'teal': '#2D8B8B',
    'coral': '#D97757', 
    'slate': '#3D4F5F',
    'sand': '#F5F0E8',
    'tan': '#C4B7A6',
    'cream': '#FAF7F2',
}

# Output paths
OUTPUT_DIR = Path('figures/publication')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data (from Table 4.2: Ablation over inference steps)
# =============================================================================

BFN = {
    'steps': np.array([5, 10, 15, 20, 30, 50]),
    'score': np.array([92.1, 91.6, 90.4, 89.9, 89.5, 89.2]),
    'score_std': np.array([0.9, 3.7, 2.0, 2.1, 1.7, 2.8]),
    'time_ms': np.array([35, 62, 86, 115, 168, 279]),
}

DIFFUSION = {
    'steps': np.array([10, 20, 30, 50, 100]),
    'score': np.array([12.8, 13.6, 35.2, 80.2, 95.0]),
    'score_std': np.array([1.4, 1.6, 7.9, 7.0, 2.3]),
    'time_ms': np.array([60, 119, 196, 291, 587]),
}

# =============================================================================
# Plot Setup
# =============================================================================

def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# =============================================================================
# Main Plot
# =============================================================================

def plot_time_vs_performance():
    """Generate the inference time vs. performance comparison."""
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # --- Plot BFN-Policy ---
    ax.errorbar(
        BFN['time_ms'], BFN['score'], yerr=BFN['score_std'],
        fmt='o-', color=COLORS['teal'], 
        linewidth=2.5, markersize=9, capsize=3, capthick=1.5,
        label='BFN-Policy (Ours)', zorder=10
    )
    
    # --- Plot Diffusion Policy ---
    ax.errorbar(
        DIFFUSION['time_ms'], DIFFUSION['score'], yerr=DIFFUSION['score_std'],
        fmt='s--', color=COLORS['coral'],
        linewidth=2.5, markersize=8, capsize=3, capthick=1.5,
        label='Diffusion Policy', zorder=9
    )
    
    # --- Step count labels ---
    # BFN labels (above points) - with custom offsets to avoid overlap
    bfn_offsets = [(0, 14), (0, 14), (0, 14), (0, 14), (0, 14), (0, 14)]
    for (t, s, n), offset in zip(
        zip(BFN['time_ms'], BFN['score'], BFN['steps']),
        bfn_offsets
    ):
        ax.annotate(
            '{}'.format(n), (t, s), textcoords="offset points",
            xytext=offset, ha='center', fontsize=8, 
            color=COLORS['teal'], fontweight='bold'
        )
    
    # Diffusion labels - position carefully to avoid overlaps
    # steps: [10, 20, 30, 50, 100] at scores [12.8, 13.6, 35.2, 80.2, 95.0]
    diff_offsets = [(-12, 0), (12, 0), (0, -16), (0, -16), (0, 12)]
    for (t, s, n), offset in zip(
        zip(DIFFUSION['time_ms'], DIFFUSION['score'], DIFFUSION['steps']), 
        diff_offsets
    ):
        ax.annotate(
            '{}'.format(n), (t, s), textcoords="offset points",
            xytext=offset, ha='center', fontsize=8,
            color=COLORS['coral'], fontweight='bold'
        )
    
    # --- Key result annotation ---
    # Arrow pointing to the gap at equal budget (between BFN@20 and Diff@20)
    ax.annotate(
        '',
        xy=(117, 18), xytext=(117, 85),
        arrowprops=dict(
            arrowstyle='<->', color=COLORS['slate'],
            lw=1.5, shrinkA=3, shrinkB=3
        )
    )
    
    # Gap label - positioned to the right of the arrow
    ax.annotate(
        '76 pp\ngap',
        xy=(130, 52), fontsize=9, color=COLORS['slate'],
        fontweight='bold', ha='left', va='center'
    )
    
    # Equal budget label - at the bottom
    ax.axvline(x=117, color=COLORS['slate'], linestyle=':', alpha=0.5, linewidth=1.5, ymax=0.85)
    ax.annotate(
        '~115 ms\n(equal budget)',
        xy=(117, 5), fontsize=8, color=COLORS['slate'],
        ha='center', va='bottom', style='italic'
    )
    
    # --- Axes ---
    ax.set_xlabel('Inference Time (ms)', fontweight='medium')
    ax.set_ylabel('Task Coverage (%)', fontweight='medium')
    ax.set_xlim(0, 620)
    ax.set_ylim(0, 105)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['slate'])
    ax.spines['bottom'].set_color(COLORS['slate'])
    
    # Legend
    ax.legend(
        loc='lower right', framealpha=0.95,
        edgecolor=COLORS['tan'], fancybox=False
    )
    
    # --- Title ---
    ax.set_title(
        'Performance vs. Inference Latency',
        fontweight='bold', color=COLORS['slate']
    )
    
    plt.tight_layout()
    
    # --- Save ---
    pdf_path = str(OUTPUT_DIR / 'fig_time_vs_performance.pdf')
    png_path = str(OUTPUT_DIR / 'fig_time_vs_performance.png')
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    print("Saved to {}".format(pdf_path))
    plt.close()

# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary():
    """Print key statistics for the paper."""
    
    print("\n" + "="*60)
    print("EQUAL-BUDGET COMPARISON (Key Result)")
    print("="*60)
    
    # Find closest time match
    bfn_20 = {'time': 115, 'score': 89.9, 'steps': 20}
    diff_20 = {'time': 119, 'score': 13.6, 'steps': 20}
    
    print(f"\nAt ~115ms inference budget:")
    print(f"  BFN-Policy  @ {bfn_20['steps']} steps: {bfn_20['score']:.1f}% ({bfn_20['time']}ms)")
    print(f"  Diffusion   @ {diff_20['steps']} steps: {diff_20['score']:.1f}% ({diff_20['time']}ms)")
    print(f"  Gap: {bfn_20['score'] - diff_20['score']:.1f} percentage points")
    
    print("\n" + "-"*60)
    print("OPERATING POINTS (Default Configuration)")
    print("-"*60)
    
    bfn_default = {'time': 115, 'score': 89.9, 'steps': 20}
    diff_default = {'time': 587, 'score': 95.0, 'steps': 100}
    
    print(f"\nBFN-Policy  @ {bfn_default['steps']} steps: {bfn_default['score']:.1f}% ({bfn_default['time']}ms)")
    print(f"Diffusion   @ {diff_default['steps']} steps: {diff_default['score']:.1f}% ({diff_default['time']}ms)")
    print(f"Speedup: {diff_default['time'] / bfn_default['time']:.1f}x")
    print(f"Score gap: {diff_default['score'] - bfn_default['score']:.1f} percentage points")
    
    print("\n" + "="*60 + "\n")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    print_summary()
    plot_time_vs_performance()


