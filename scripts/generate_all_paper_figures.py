#!/usr/bin/env python3
"""
Master script to generate all paper figures and tables.

This script runs all figure generation scripts in the correct order to produce
publication-ready figures and LaTeX tables for the paper.

Usage:
    python scripts/generate_all_paper_figures.py
    
    # Or with custom paths:
    python scripts/generate_all_paper_figures.py \
        --checkpoint-dir cluster_checkpoints/benchmarkresults \
        --output-dir figures
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import time


def run_script(
    script_path: Path,
    description: str,
    check_exists: bool = True,
    extra_args: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """Run a Python script and return success status and output."""
    if check_exists and not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    try:
        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)
        
        # Use stdout/stderr instead of capture_output for Python < 3.7 compatibility
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            # Return stderr if available, otherwise stdout
            error_output = result.stderr if result.stderr else result.stdout
            return False, error_output
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all figures with default paths
    python scripts/generate_all_paper_figures.py
    
    # Generate with custom checkpoint directory
    python scripts/generate_all_paper_figures.py --checkpoint-dir /path/to/checkpoints
        """
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='cluster_checkpoints/benchmarkresults',
        help='Directory containing benchmark results (default: cluster_checkpoints/benchmarkresults)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures)'
    )
    
    parser.add_argument(
        '--skip-ablation',
        action='store_true',
        help='Skip ablation study (requires GPU and takes time)'
    )
    
    parser.add_argument(
        '--skip-multimodal',
        action='store_true',
        help='Skip multimodal figures (may require additional data)'
    )
    
    parser.add_argument(
        '--skip-predictions',
        action='store_true',
        help='Skip prediction visualizations (may require additional data)'
    )
    
    args = parser.parse_args()
    
    # Scripts directory
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    
    print("=" * 80)
    print("PAPER FIGURE AND TABLE GENERATION")
    print("=" * 80)
    print(f"\nCheckpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Project root: {project_root}")
    print("\n" + "=" * 80)
    
    # Define all scripts to run in order
    scripts_to_run = [
        # Core publication figures (required)
        {
            'script': scripts_dir / 'generate_publication_figures.py',
            'description': 'Main publication figures (results, per-seed, efficiency, training curves, pareto)',
            'required': True,
        },
        
        # Training metrics plots
        {
            'script': scripts_dir / 'plot_training_loss.py',
            'description': 'Training loss curves',
            'required': True,
        },
        
        {
            'script': scripts_dir / 'plot_action_mse_error.py',
            'description': 'Action prediction MSE error',
            'required': True,
        },
        
        # Analysis and tables
        {
            'script': scripts_dir / 'analyze_local_results.py',
            'description': 'Analysis tables and comparison figures',
            'required': True,
        },
        
        {
            'script': scripts_dir / 'extract_scores_from_logs.py',
            'description': 'Extract scores from logs and generate tables',
            'required': True,
            'args': ['--checkpoint-dir', args.checkpoint_dir],
        },
        
        # Training stability
        {
            'script': scripts_dir / 'generate_training_stability_figure.py',
            'description': 'Training stability figures',
            'required': False,
        },
        
        {
            'script': scripts_dir / 'analyze_loss_performance_relationship.py',
            'description': 'Loss vs performance relationship analysis',
            'required': False,
        },
        
        # Multimodal figures (optional - may require additional data)
        {
            'script': scripts_dir / 'generate_multimodal_figures.py',
            'description': 'Multimodal behavior figures',
            'required': False,
            'skip_flag': args.skip_multimodal,
        },
        
        # Prediction visualizations (optional - may require additional data)
        {
            'script': scripts_dir / 'generate_prediction_visualizations.py',
            'description': 'Prediction and trajectory visualizations',
            'required': False,
            'skip_flag': args.skip_predictions,
        },
        
        # Ablation study (optional - requires GPU and takes time)
        {
            'script': scripts_dir / 'comprehensive_ablation_study.py',
            'description': 'Comprehensive ablation study (requires GPU)',
            'required': False,
            'skip_flag': args.skip_ablation,
            'args': ['--checkpoint-dir', args.checkpoint_dir, '--n-envs', '50', '--device', 'cuda'],
        },
    ]
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
    }
    
    start_time = time.time()
    
    # Run each script
    for i, script_info in enumerate(scripts_to_run, 1):
        script_path = script_info['script']
        description = script_info['description']
        required = script_info.get('required', False)
        skip_flag = script_info.get('skip_flag', False)
        extra_args = script_info.get('args', [])
        
        print(f"\n[{i}/{len(scripts_to_run)}] {description}")
        print(f"  Script: {script_path.name}")
        print("-" * 80)
        
        # Check if should skip
        if skip_flag:
            print(f"  ⏭️  SKIPPED (--skip-* flag set)")
            results['skipped'].append((script_path.name, description))
            continue
        
        # Check if script exists
        if not script_path.exists():
            if required:
                print(f"  ❌ ERROR: Required script not found!")
                results['failed'].append((script_path.name, description, "Script not found"))
                continue
            else:
                print(f"  ⏭️  SKIPPED (script not found)")
                results['skipped'].append((script_path.name, description))
                continue
        
        # Run the script
        success, output = run_script(script_path, description, extra_args=extra_args)
        
        if success:
            print(f"  ✅ SUCCESS")
            if output.strip():
                # Show last few lines of output
                lines = output.strip().split('\n')
                if len(lines) > 5:
                    print("  Output (last 5 lines):")
                    for line in lines[-5:]:
                        print(f"    {line}")
                else:
                    print("  Output:")
                    for line in lines:
                        print(f"    {line}")
            results['success'].append((script_path.name, description))
        else:
            error_msg = output.split('\n')[-1] if output else "Unknown error"
            print(f"  ❌ FAILED: {error_msg}")
            if required:
                print(f"  ⚠️  WARNING: This is a required script!")
            results['failed'].append((script_path.name, description, error_msg))
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
    print(f"\n✅ Successful: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    
    if results['success']:
        print("\n✅ Successfully generated:")
        for script_name, desc in results['success']:
            print(f"   • {script_name}: {desc}")
    
    if results['failed']:
        print("\n❌ Failed scripts:")
        for script_name, desc, error in results['failed']:
            print(f"   • {script_name}: {desc}")
            print(f"     Error: {error}")
    
    if results['skipped']:
        print("\n⏭️  Skipped scripts:")
        for script_name, desc in results['skipped']:
            print(f"   • {script_name}: {desc}")
    
    # Output locations
    output_path = project_root / args.output_dir
    publication_path = output_path / 'publication'
    tables_path = output_path / 'tables'
    
    print("\n" + "=" * 80)
    print("OUTPUT LOCATIONS")
    print("=" * 80)
    print(f"\n📁 Main figures: {publication_path}")
    if publication_path.exists():
        pdf_files = list(publication_path.glob("*.pdf"))
        png_files = list(publication_path.glob("*.png"))
        print(f"   • {len(pdf_files)} PDF files")
        print(f"   • {len(png_files)} PNG files")
    
    print(f"\n📁 Tables: {tables_path}")
    if tables_path.exists():
        tex_files = list(tables_path.glob("*.tex"))
        print(f"   • {len(tex_files)} LaTeX table files")
    
    print(f"\n📁 Root figures: {output_path}")
    if output_path.exists():
        pdf_files = list(output_path.glob("*.pdf"))
        tex_files = list(output_path.glob("*.tex"))
        print(f"   • {len(pdf_files)} PDF files")
        print(f"   • {len(tex_files)} LaTeX files")
    
    print("\n" + "=" * 80)
    
    # Exit code
    if results['failed']:
        required_failed = [
            f for f in results['failed']
            if any(
                s['script'].name == f[0] and s.get('required', False)
                for s in scripts_to_run
            )
        ]
        if required_failed:
            print("\n⚠️  Some required scripts failed. Please check the errors above.")
            sys.exit(1)
        else:
            print("\n⚠️  Some optional scripts failed, but all required ones succeeded.")
            sys.exit(0)
    else:
        print("\n✅ All scripts completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

