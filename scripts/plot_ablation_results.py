#!/usr/bin/env python3
"""Generate ablation study figures from existing results.

This script reads the ablation_results.json and generates publication-quality figures.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import centralized Anthropic color palette
from colors import COLORS, setup_matplotlib_style, SINGLE_COL as _SINGLE_COL, DOUBLE_COL as _DOUBLE_COL

# Setup matplotlib style
setup_matplotlib_style()

DOUBLE_COL = 7.0   # inches (NeurIPS double column width)
SINGLE_COL = 3.5   # inches (NeurIPS single column width)

# 90% threshold color (use neutral tan instead of green)
THRESHOLD_COLOR = '#C4A77D'  # Anthropic Tan


def aggregate_method(method_results):
    """Aggregate results across seeds."""
    all_steps = set()
    for seed_data in method_results.values():
        all_steps.update(int(k) for k in seed_data.keys())
    
    aggregated = {}
    for steps in sorted(all_steps):
        steps_str = str(steps)
        scores = []
        times = []
        for seed_data in method_results.values():
            if steps_str in seed_data:
                scores.append(seed_data[steps_str]['mean_score'])
                times.append(seed_data[steps_str]['avg_time_ms'])
        
        if scores:
            aggregated[steps] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
            }
    return aggregated


def generate_ablation_figures(results, output_dir="figures/publication"):
    """Generate all ablation figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bfn_agg = aggregate_method(results.get('bfn', {}))
    diff_agg = aggregate_method(results.get('diffusion', {}))
    
    print(f"BFN aggregated: {len(bfn_agg)} step configurations")
    print(f"Diffusion aggregated: {len(diff_agg)} step configurations")
    
    if not bfn_agg or not diff_agg:
        print("ERROR: No aggregated data!")
        return
    
    # =========================================================================
    # Figure 1: Score vs Steps
    # =========================================================================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    bfn_steps = sorted(bfn_agg.keys())
    bfn_scores = [bfn_agg[s]['mean_score'] for s in bfn_steps]
    bfn_stds = [bfn_agg[s]['std_score'] for s in bfn_steps]
    
    ax.errorbar(bfn_steps, bfn_scores, yerr=bfn_stds, 
                color=COLORS['bfn'], marker='o', markersize=7,
                capsize=4, linewidth=2, label='BFN Policy', linestyle='-')
    
    diff_steps = sorted(diff_agg.keys())
    diff_scores = [diff_agg[s]['mean_score'] for s in diff_steps]
    diff_stds = [diff_agg[s]['std_score'] for s in diff_steps]
    
    ax.errorbar(diff_steps, diff_scores, yerr=diff_stds,
                color=COLORS['diffusion'], marker='s', markersize=7,
                capsize=4, linewidth=2, label='Diffusion Policy', linestyle='-')
    
    ax.set_xlabel('Inference Steps', fontsize=11)
    ax.set_ylabel('Success Score', fontsize=11)
    ax.set_title('Performance vs Inference Steps', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 105)
    
    # Add annotations for key points
    # BFN at 5 steps
    ax.annotate(f'{bfn_scores[0]:.2f}', xy=(bfn_steps[0], bfn_scores[0]), 
                xytext=(bfn_steps[0]+5, bfn_scores[0]-0.08),
                fontsize=8, color=COLORS['bfn'], fontweight='bold')
    
    # Diffusion at 100 steps
    ax.annotate(f'{diff_scores[-1]:.2f}', xy=(diff_steps[-1], diff_scores[-1]), 
                xytext=(diff_steps[-1]-15, diff_scores[-1]+0.03),
                fontsize=8, color=COLORS['diffusion'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(str(output_path / 'fig_ablation_score_vs_steps.pdf'), format='pdf')
    fig.savefig(str(output_path / 'fig_ablation_score_vs_steps.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"✓ Score vs Steps figure saved")
    
    # =========================================================================
    # Figure 2: Time vs Steps
    # =========================================================================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    bfn_times = [bfn_agg[s]['mean_time'] for s in bfn_steps]
    bfn_time_stds = [bfn_agg[s]['std_time'] for s in bfn_steps]
    
    diff_times = [diff_agg[s]['mean_time'] for s in diff_steps]
    diff_time_stds = [diff_agg[s]['std_time'] for s in diff_steps]
    
    ax.errorbar(bfn_steps, bfn_times, yerr=bfn_time_stds, 
                color=COLORS['bfn'], marker='o', markersize=7,
                capsize=4, linewidth=2, label='BFN Policy', linestyle='-')
    
    ax.errorbar(diff_steps, diff_times, yerr=diff_time_stds,
                color=COLORS['diffusion'], marker='s', markersize=7,
                capsize=4, linewidth=2, label='Diffusion Policy', linestyle='-')
    
    ax.set_xlabel('Inference Steps', fontsize=11)
    ax.set_ylabel('Inference Time (ms)', fontsize=11)
    ax.set_title('Inference Time vs Steps', fontweight='bold', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    fig.savefig(str(output_path / 'fig_ablation_time_vs_steps.pdf'), format='pdf')
    fig.savefig(str(output_path / 'fig_ablation_time_vs_steps.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"✓ Time vs Steps figure saved")
    
    # =========================================================================
    # Figure 3: Pareto Frontier (Score vs Time)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    # Plot with step labels
    for i, s in enumerate(bfn_steps):
        ax.scatter(bfn_agg[s]['mean_time'], bfn_agg[s]['mean_score'],
                   color=COLORS['bfn'], s=80, zorder=3, marker='o',
                   edgecolors='white', linewidth=1)
        ax.annotate(f'{s}', xy=(bfn_agg[s]['mean_time'], bfn_agg[s]['mean_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7, color=COLORS['bfn'])
    
    for i, s in enumerate(diff_steps):
        ax.scatter(diff_agg[s]['mean_time'], diff_agg[s]['mean_score'],
                   color=COLORS['diffusion'], s=80, zorder=3, marker='s',
                   edgecolors='white', linewidth=1)
        ax.annotate(f'{s}', xy=(diff_agg[s]['mean_time'], diff_agg[s]['mean_score']),
                    xytext=(5, -10), textcoords='offset points', fontsize=7, color=COLORS['diffusion'])
    
    # Connect points with lines
    ax.plot(bfn_times, bfn_scores, color=COLORS['bfn'], linewidth=1.5, alpha=0.7, label='BFN Policy')
    ax.plot(diff_times, diff_scores, color=COLORS['diffusion'], linewidth=1.5, alpha=0.7, label='Diffusion Policy')
    
    ax.set_xlabel('Inference Time (ms)', fontsize=11)
    ax.set_ylabel('Success Score', fontsize=11)
    ax.set_title('Efficiency-Performance Trade-off', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    
    # Add efficiency frontier zone
    ax.axhline(y=0.9, color=THRESHOLD_COLOR, linestyle=':', alpha=0.7, linewidth=1)
    ax.text(20, 0.91, '90% Success', fontsize=8, color=THRESHOLD_COLOR)
    
    plt.tight_layout()
    fig.savefig(str(output_path / 'fig_ablation_pareto.pdf'), format='pdf')
    fig.savefig(str(output_path / 'fig_ablation_pareto.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"✓ Pareto frontier figure saved")
    
    # =========================================================================
    # Figure 4: Combined 2x2 Figure for Paper
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    
    # Panel A: Score vs Steps
    ax = axes[0, 0]
    ax.errorbar(bfn_steps, bfn_scores, yerr=bfn_stds, 
                color=COLORS['bfn'], marker='o', markersize=5,
                capsize=3, linewidth=1.5, label='BFN Policy', linestyle='-')
    ax.errorbar(diff_steps, diff_scores, yerr=diff_stds,
                color=COLORS['diffusion'], marker='s', markersize=5,
                capsize=3, linewidth=1.5, label='Diffusion Policy', linestyle='-')
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Success Score')
    ax.set_title('(a) Performance vs Inference Steps', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 105)
    
    # Panel B: Time vs Steps
    ax = axes[0, 1]
    ax.errorbar(bfn_steps, bfn_times, yerr=bfn_time_stds, 
                color=COLORS['bfn'], marker='o', markersize=5,
                capsize=3, linewidth=1.5, label='BFN Policy', linestyle='-')
    ax.errorbar(diff_steps, diff_times, yerr=diff_time_stds,
                color=COLORS['diffusion'], marker='s', markersize=5,
                capsize=3, linewidth=1.5, label='Diffusion Policy', linestyle='-')
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('(b) Inference Time vs Steps', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 105)
    
    # Panel C: Pareto Frontier
    ax = axes[1, 0]
    ax.plot(bfn_times, bfn_scores, color=COLORS['bfn'], linewidth=1.5, 
            marker='o', markersize=5, label='BFN Policy')
    ax.plot(diff_times, diff_scores, color=COLORS['diffusion'], linewidth=1.5,
            marker='s', markersize=5, label='Diffusion Policy')
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Success Score')
    ax.set_title('(c) Efficiency-Performance Trade-off', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color=THRESHOLD_COLOR, linestyle=':', alpha=0.7, linewidth=1)
    
    # Panel D: Efficiency (Score per 100ms)
    ax = axes[1, 1]
    bfn_efficiency = [bfn_agg[s]['mean_score'] / bfn_agg[s]['mean_time'] * 100 for s in bfn_steps]
    diff_efficiency = [diff_agg[s]['mean_score'] / diff_agg[s]['mean_time'] * 100 for s in diff_steps]
    
    ax.bar(np.arange(len(bfn_steps)) - 0.2, bfn_efficiency, 0.35, 
           label='BFN Policy', color=COLORS['bfn'], alpha=0.8)
    ax.bar(np.arange(len(diff_steps)) + 0.2, diff_efficiency, 0.35,
           label='Diffusion Policy', color=COLORS['diffusion'], alpha=0.8)
    
    # Set x-tick labels
    all_steps = sorted(set(bfn_steps) | set(diff_steps))
    ax.set_xticks(np.arange(max(len(bfn_steps), len(diff_steps))))
    ax.set_xticklabels([str(s) for s in bfn_steps] if len(bfn_steps) >= len(diff_steps) 
                        else [str(s) for s in diff_steps])
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Efficiency (Score / 100ms)')
    ax.set_title('(d) Computational Efficiency', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    fig.savefig(str(output_path / 'fig_ablation_combined.pdf'), format='pdf')
    fig.savefig(str(output_path / 'fig_ablation_combined.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"✓ Combined 2x2 figure saved")
    
    # =========================================================================
    # Figure 5: Key Finding - Speedup Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.8))
    
    # Find steps needed to reach 90% performance
    bfn_90_steps = None
    for s in bfn_steps:
        if bfn_agg[s]['mean_score'] >= 0.90:
            bfn_90_steps = s
            break
    
    diff_90_steps = None
    for s in diff_steps:
        if diff_agg[s]['mean_score'] >= 0.90:
            diff_90_steps = s
            break
    
    if bfn_90_steps and diff_90_steps:
        bfn_time_90 = bfn_agg[bfn_90_steps]['mean_time']
        diff_time_90 = diff_agg[diff_90_steps]['mean_time']
        speedup = diff_time_90 / bfn_time_90
        
        methods = ['BFN Policy', 'Diffusion Policy']
        times = [bfn_time_90, diff_time_90]
        steps_needed = [bfn_90_steps, diff_90_steps]
        colors = [COLORS['bfn'], COLORS['diffusion']]
        
        bars = ax.barh(methods, times, color=colors, height=0.5, alpha=0.85)
        
        # Add labels
        for bar, t, s in zip(bars, times, steps_needed):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'{t:.0f} ms ({s} steps)', va='center', fontsize=10)
        
        ax.set_xlabel('Time to Reach 90% Success Score (ms)', fontsize=11)
        ax.set_title(f'Time to 90% Performance: BFN is {speedup:.1f}× Faster', 
                     fontweight='bold', fontsize=12)
        ax.set_xlim(0, max(times) * 1.5)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        plt.tight_layout()
        fig.savefig(str(output_path / 'fig_ablation_speedup.pdf'), format='pdf')
        fig.savefig(str(output_path / 'fig_ablation_speedup.png'), format='png', dpi=300)
        plt.close(fig)
        print(f"✓ Speedup comparison figure saved")
        print(f"  → BFN reaches 90% with {bfn_90_steps} steps ({bfn_time_90:.0f} ms)")
        print(f"  → Diffusion reaches 90% with {diff_90_steps} steps ({diff_time_90:.0f} ms)")
        print(f"  → Speedup: {speedup:.1f}×")


def generate_ablation_tables(results, output_dir="figures/tables"):
    """Generate LaTeX tables for ablation study."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bfn_agg = aggregate_method(results.get('bfn', {}))
    diff_agg = aggregate_method(results.get('diffusion', {}))
    
    if not bfn_agg or not diff_agg:
        print("ERROR: No aggregated data for tables!")
        return
    
    # Table: Comprehensive Ablation
    table_content = r"""\begin{table*}[t]
\centering
\caption{\textbf{Inference Steps Ablation Study.} Performance and inference time at different step counts.
BFN achieves 90\%+ success with just 5 steps, while Diffusion requires 50+ steps.}
\label{tab:ablation}
\vspace{0.5em}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Steps} & \textbf{Score} & \textbf{Time (ms)} & \textbf{Efficiency} \\
\midrule
"""
    
    # BFN rows
    for steps in sorted(bfn_agg.keys()):
        data = bfn_agg[steps]
        efficiency = data['mean_score'] / data['mean_time'] * 100
        table_content += f"BFN & {steps} & ${data['mean_score']:.3f} \\pm {data['std_score']:.3f}$ & ${data['mean_time']:.1f}$ & ${efficiency:.2f}$ \\\\\n"
    
    table_content += r"\midrule" + "\n"
    
    # Diffusion rows
    for steps in sorted(diff_agg.keys()):
        data = diff_agg[steps]
        efficiency = data['mean_score'] / data['mean_time'] * 100
        table_content += f"Diffusion & {steps} & ${data['mean_score']:.3f} \\pm {data['std_score']:.3f}$ & ${data['mean_time']:.1f}$ & ${efficiency:.2f}$ \\\\\n"
    
    table_content += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    table_file = output_path / "table_ablation.tex"
    with open(table_file, 'w') as f:
        f.write(table_content)
    print(f"✓ Ablation table saved to: {table_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate ablation figures from results')
    parser.add_argument('--results-file', type=str, 
                        default='results/ablation/ablation_results.json',
                        help='Path to ablation results JSON')
    parser.add_argument('--output-dir', type=str, default='figures/publication',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ABLATION STUDY FIGURE GENERATION")
    print("=" * 70)
    
    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return
    
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"  BFN seeds: {list(results.get('bfn', {}).keys())}")
    print(f"  Diffusion seeds: {list(results.get('diffusion', {}).keys())}")
    
    # Generate figures
    print(f"\nGenerating figures to: {args.output_dir}")
    generate_ablation_figures(results, args.output_dir)
    
    # Generate tables
    print(f"\nGenerating tables...")
    generate_ablation_tables(results)
    
    print("\n" + "=" * 70)
    print("DONE! All ablation figures and tables generated.")
    print("=" * 70)


if __name__ == '__main__':
    main()

