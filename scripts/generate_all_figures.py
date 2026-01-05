#!/usr/bin/env python3
"""
Master Script: Generate All Publication Figures
Style: Anthropic Research Palette

This script regenerates ALL publication figures with consistent Anthropic colors:
- BFN Policy: Teal (#2D8B8B)
- Diffusion Policy: Coral (#D97757)

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --skip-slow  # Skip time-consuming figures
    python scripts/generate_all_figures.py --only ablation  # Only specific category
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "figures" / "publication"
TABLES_DIR = PROJECT_ROOT / "figures" / "tables"
DATA_DIR = PROJECT_ROOT / "outputs" / "2025.12.25"
ABLATION_RESULTS = PROJECT_ROOT / "results" / "ablation" / "ablation_results.json"

# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_status(name: str, status: str, color: str = None):
    """Print status with optional color."""
    symbols = {"success": "✓", "error": "✗", "skip": "○", "running": "→"}
    symbol = symbols.get(status, "•")
    print(f"  {symbol} {name}")


def run_script(script_name: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print_status(script_name, "error")
        print(f"      Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        # Use stdout/stderr=PIPE for Python 3.6 compatibility
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print_status(script_name, "success")
            return True
        else:
            print_status(script_name, "error")
            if result.stderr:
                # Print first few lines of error
                stderr_text = result.stderr.decode('utf-8', errors='ignore')
                error_lines = stderr_text.strip().split('\n')[-3:]
                for line in error_lines:
                    print(f"      {line[:80]}")
            return False
    except Exception as e:
        print_status(script_name, "error")
        print(f"      Exception: {e}")
        return False


def generate_ablation_figures():
    """Generate ablation study figures."""
    print_header("ABLATION FIGURES")
    
    if not ABLATION_RESULTS.exists():
        print_status("ablation_results.json", "error")
        print(f"      Results file not found: {ABLATION_RESULTS}")
        return False
    
    return run_script("plot_ablation_results.py", [
        "--results-file", str(ABLATION_RESULTS),
        "--output-dir", str(OUTPUT_DIR)
    ])


def generate_prediction_visualizations():
    """Generate trajectory and prediction visualizations."""
    print_header("PREDICTION VISUALIZATIONS")
    return run_script("generate_prediction_visualizations.py")


def generate_publication_figures():
    """Generate main publication figures."""
    print_header("MAIN PUBLICATION FIGURES")
    
    if not DATA_DIR.exists():
        print_status("training data", "error")
        print(f"      Data directory not found: {DATA_DIR}")
        return False
    
    return run_script("generate_publication_figures.py", [
        "--checkpoint-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR)
    ])


def generate_setup_figures():
    """Generate experimental setup figures."""
    print_header("SETUP FIGURES")
    return run_script("generate_setup_figures.py")


def generate_training_stability():
    """Generate training stability comparison."""
    print_header("TRAINING STABILITY")
    return run_script("generate_training_stability_figure.py")


def generate_multimodal_figures():
    """Generate multimodal behavior visualizations."""
    print_header("MULTIMODAL FIGURES")
    return run_script("generate_multimodal_figures.py")


def generate_enhanced_figures():
    """Generate enhanced paper figures."""
    print_header("ENHANCED FIGURES")
    
    if not DATA_DIR.exists():
        print_status("training data", "skip")
        return True
    
    return run_script("generate_enhanced_paper_figures.py", [
        "--checkpoint-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR)
    ])


def generate_training_loss():
    """Generate training loss curves."""
    print_header("TRAINING LOSS CURVES")
    
    if not DATA_DIR.exists():
        print_status("training data", "skip")
        return True
    
    return run_script("plot_training_loss.py", [
        "--base-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--all-plots"
    ])


def generate_action_mse():
    """Generate action MSE error plots."""
    print_header("ACTION MSE ERROR")
    
    if not DATA_DIR.exists():
        print_status("training data", "skip")
        return True
    
    return run_script("plot_action_mse_error.py", [
        "--base-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--all-plots"
    ])


# =============================================================================
# INLINE FIGURE GENERATION (for figures without separate scripts)
# =============================================================================

def generate_color_demo():
    """Generate a color palette demonstration figure."""
    print_header("COLOR PALETTE DEMO")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from colors import COLORS, setup_matplotlib_style
        
        setup_matplotlib_style()
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        
        # Left: Method comparison bars
        ax = axes[0]
        methods = ['BFN Policy\n(Ours)', 'Diffusion\nPolicy']
        scores = [0.925, 0.960]
        colors = [COLORS['bfn'], COLORS['diffusion']]
        
        bars = ax.bar(methods, scores, color=colors, edgecolor='white', linewidth=2)
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.1)
        ax.set_title('Method Comparison', fontweight='bold')
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, score + 0.03, 
                   f'{score:.1%}', ha='center', fontweight='bold')
        
        # Right: Inference efficiency
        ax = axes[1]
        steps = [20, 100]
        
        bars = ax.bar(methods, steps, color=colors, edgecolor='white', linewidth=2)
        ax.set_ylabel('Inference Steps')
        ax.set_title('Inference Efficiency', fontweight='bold')
        
        # Add speedup annotation
        ax.annotate('5× faster', xy=(0, 30), fontsize=10, 
                   fontweight='bold', color=COLORS['bfn'], ha='center')
        
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / 'fig_color_demo.pdf'
        fig.savefig(str(output_path))
        fig.savefig(str(output_path).replace('.pdf', '.png'), dpi=300)
        plt.close(fig)
        
        print_status("Color demo figure", "success")
        return True
        
    except Exception as e:
        print_status("Color demo figure", "error")
        print(f"      {e}")
        return False


def generate_main_results_enhanced():
    """Generate enhanced main results figure with Anthropic style."""
    print_header("MAIN RESULTS (ENHANCED)")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from colors import COLORS, setup_matplotlib_style
        
        setup_matplotlib_style()
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        
        # Data from experiments
        bfn_scores = [0.943, 0.942, 0.890]
        diff_scores = [0.966, 0.945, 0.970]
        seeds = [42, 43, 44]
        
        bfn_mean, bfn_std = np.mean(bfn_scores), np.std(bfn_scores)
        diff_mean, diff_std = np.mean(diff_scores), np.std(diff_scores)
        
        # Panel (a): Main comparison
        ax = axes[0]
        x = np.arange(2)
        means = [bfn_mean, diff_mean]
        stds = [bfn_std, diff_std]
        colors = [COLORS['bfn'], COLORS['diffusion']]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                     edgecolor='white', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['BFN\n(Ours)', 'Diffusion'])
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.1)
        ax.set_title('(a) Performance Comparison', fontweight='bold')
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.03,
                   f'{mean:.1%}', ha='center', fontsize=9, fontweight='bold')
        
        # Panel (b): Per-seed breakdown
        ax = axes[1]
        x = np.arange(len(seeds))
        width = 0.35
        
        ax.bar(x - width/2, bfn_scores, width, label='BFN (Ours)',
              color=COLORS['bfn'], edgecolor='white')
        ax.bar(x + width/2, diff_scores, width, label='Diffusion',
              color=COLORS['diffusion'], edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Seed {s}' for s in seeds])
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0.8, 1.02)
        ax.set_title('(b) Per-Seed Results', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        
        # Panel (c): Efficiency
        ax = axes[2]
        bfn_steps, diff_steps = 20, 100
        efficiency = [bfn_mean / bfn_steps * 100, diff_mean / diff_steps * 100]
        
        bars = ax.bar(['BFN\n(Ours)', 'Diffusion'], efficiency, 
                     color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_ylabel('Efficiency\n(Score / 100 steps)')
        ax.set_title('(c) Computational Efficiency', fontweight='bold')
        
        # Highlight advantage
        ratio = efficiency[0] / efficiency[1]
        ax.text(0, efficiency[0] + 0.3, f'{ratio:.1f}×', ha='center',
               fontsize=11, fontweight='bold', color=COLORS['bfn'])
        
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / 'fig1_main_results_enhanced.pdf'
        fig.savefig(str(output_path))
        fig.savefig(str(output_path).replace('.pdf', '.png'), dpi=300)
        plt.close(fig)
        
        print_status("Main results enhanced", "success")
        return True
        
    except Exception as e:
        print_status("Main results enhanced", "error")
        print(f"      {e}")
        return False


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(results: dict):
    """Print summary of figure generation."""
    print_header("SUMMARY")
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n  Generated: {success}/{total} figure sets")
    print()
    
    # List output files
    if OUTPUT_DIR.exists():
        pdf_files = list(OUTPUT_DIR.glob("*.pdf"))
        png_files = list(OUTPUT_DIR.glob("*.png"))
        
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  PDF files: {len(pdf_files)}")
        print(f"  PNG files: {len(png_files)}")
        
        # List recent files
        print("\n  Recently generated:")
        recent = sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
        for f in recent:
            print(f"    • {f.name}")
    
    # Tables
    if TABLES_DIR.exists():
        tex_files = list(TABLES_DIR.glob("*.tex"))
        print(f"\n  LaTeX tables: {len(tex_files)}")
        for f in tex_files:
            print(f"    • {f.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all publication figures with Anthropic colors"
    )
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip time-consuming figure generation')
    parser.add_argument('--only', type=str, choices=[
        'ablation', 'prediction', 'publication', 'setup', 
        'stability', 'multimodal', 'training', 'demo'
    ], help='Generate only specific category')
    parser.add_argument('--list', action='store_true',
                       help='List available figure categories')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable figure categories:")
        print("  ablation     - Inference steps ablation study")
        print("  prediction   - Trajectory and denoising visualizations")
        print("  publication  - Main paper figures")
        print("  setup        - Experimental setup diagrams")
        print("  stability    - Training stability comparison")
        print("  multimodal   - Multimodal behavior figures")
        print("  training     - Training loss curves")
        print("  demo         - Color palette demonstration")
        return
    
    # Print header
    print("\n" + "=" * 70)
    print("  PUBLICATION FIGURE GENERATOR")
    print("  Style: Anthropic Research Palette")
    print("  BFN: #2D8B8B (Teal) | Diffusion: #D97757 (Coral)")
    print("=" * 70)
    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {OUTPUT_DIR}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {}
    
    # Define generation functions
    generators = {
        'demo': generate_color_demo,
        'main_enhanced': generate_main_results_enhanced,
        'ablation': generate_ablation_figures,
        'prediction': generate_prediction_visualizations,
        'publication': generate_publication_figures,
        'setup': generate_setup_figures,
        'stability': generate_training_stability,
        'multimodal': generate_multimodal_figures,
        'training': generate_training_loss,
    }
    
    # Slow generators (skip with --skip-slow)
    slow_generators = {'training', 'publication'}
    
    # Filter based on arguments
    if args.only:
        if args.only == 'demo':
            generators = {'demo': generate_color_demo, 'main_enhanced': generate_main_results_enhanced}
        else:
            generators = {args.only: generators.get(args.only)}
    
    # Run generators
    for name, func in generators.items():
        if func is None:
            continue
        if args.skip_slow and name in slow_generators:
            print_header(f"SKIPPING: {name.upper()}")
            results[name] = None
            continue
        
        try:
            results[name] = func()
        except Exception as e:
            print(f"  Error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

