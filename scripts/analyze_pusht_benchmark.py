#!/usr/bin/env python
"""
Analyze PushT Benchmark Results

This script analyzes the benchmark results from WandB and generates:
1. Performance comparison tables
2. Training curves comparison plots
3. Inference speed comparison
4. Statistical significance tests

Usage:
    python scripts/analyze_pusht_benchmark.py --project pusht_bfn_vs_diffusion --output results/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Cannot fetch results from WandB.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not installed. Plotting disabled.")


def fetch_wandb_runs(project: str, entity: Optional[str] = None) -> List[Dict]:
    """Fetch runs from WandB project."""
    if not HAS_WANDB:
        return []
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)
    
    results = []
    for run in runs:
        # Extract key metrics
        summary = run.summary._json_dict
        config = run.config
        history = run.history()
        
        results.append({
            'name': run.name,
            'id': run.id,
            'state': run.state,
            'tags': run.tags,
            'config': config,
            'summary': summary,
            'history': history,
        })
    
    return results


def extract_method_and_seed(run: Dict) -> tuple:
    """Extract method (bfn/diffusion) and seed from run."""
    tags = run.get('tags', [])
    name = run.get('name', '')
    
    method = 'unknown'
    if 'bfn' in tags or 'bfn' in name.lower():
        method = 'BFN'
    elif 'diffusion' in tags or 'diffusion' in name.lower():
        method = 'Diffusion'
    
    seed = run.get('config', {}).get('training', {}).get('seed', 'unknown')
    
    return method, seed


def create_performance_table(runs: List[Dict]) -> pd.DataFrame:
    """Create performance comparison table."""
    data = []
    
    for run in runs:
        method, seed = extract_method_and_seed(run)
        summary = run.get('summary', {})
        
        data.append({
            'Method': method,
            'Seed': seed,
            'Mean Score': summary.get('test_mean_score', summary.get('test/mean_score', np.nan)),
            'Max Score': summary.get('test_max_score', summary.get('test/max_score', np.nan)),
            'Final Train Loss': summary.get('train_loss', summary.get('train/loss', np.nan)),
            'Epochs': summary.get('epoch', np.nan),
        })
    
    df = pd.DataFrame(data)
    return df


def create_aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated statistics table."""
    agg = df.groupby('Method').agg({
        'Mean Score': ['mean', 'std', 'max'],
        'Max Score': ['mean', 'max'],
        'Final Train Loss': ['mean', 'std'],
    }).round(4)
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def plot_training_curves(runs: List[Dict], output_dir: Path):
    """Plot training curves comparison."""
    if not HAS_PLOT:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'BFN': 'tab:blue', 'Diffusion': 'tab:orange'}
    
    for run in runs:
        method, seed = extract_method_and_seed(run)
        history = run.get('history')
        
        if history is None or history.empty:
            continue
        
        color = colors.get(method, 'gray')
        alpha = 0.7
        
        # Training loss
        if 'train_loss' in history.columns:
            axes[0].plot(history['_step'], history['train_loss'], 
                        color=color, alpha=alpha, label=f'{method} (seed={seed})')
        
        # Test score
        test_col = 'test_mean_score' if 'test_mean_score' in history.columns else 'test/mean_score'
        if test_col in history.columns:
            valid_data = history[history[test_col].notna()]
            axes[1].plot(valid_data['_step'], valid_data[test_col],
                        color=color, alpha=alpha, label=f'{method} (seed={seed})')
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Test Mean Score')
    axes[1].set_title('Test Performance Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")


def plot_performance_boxplot(df: pd.DataFrame, output_dir: Path):
    """Create box plot comparing performance."""
    if not HAS_PLOT:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(data=df, x='Method', y='Mean Score', ax=ax)
    sns.stripplot(data=df, x='Method', y='Mean Score', color='black', alpha=0.5, ax=ax)
    
    ax.set_title('PushT Performance: BFN vs Diffusion Policy')
    ax.set_ylabel('Mean Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance boxplot to {output_dir / 'performance_boxplot.png'}")


def compute_statistical_tests(df: pd.DataFrame) -> Dict:
    """Compute statistical significance tests."""
    from scipy import stats
    
    bfn_scores = df[df['Method'] == 'BFN']['Mean Score'].dropna()
    diff_scores = df[df['Method'] == 'Diffusion']['Mean Score'].dropna()
    
    results = {}
    
    if len(bfn_scores) >= 2 and len(diff_scores) >= 2:
        # Welch's t-test (doesn't assume equal variance)
        t_stat, p_value = stats.ttest_ind(bfn_scores, diff_scores, equal_var=False)
        results['welch_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_0.05': p_value < 0.05,
        }
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = stats.mannwhitneyu(bfn_scores, diff_scores, alternative='two-sided')
        results['mann_whitney'] = {
            'u_statistic': float(u_stat),
            'p_value': float(p_value_mw),
            'significant_0.05': p_value_mw < 0.05,
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((bfn_scores.std()**2 + diff_scores.std()**2) / 2)
        cohens_d = (bfn_scores.mean() - diff_scores.mean()) / pooled_std if pooled_std > 0 else 0
        results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
        }
    
    return results


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for thesis."""
    agg = df.groupby('Method').agg({
        'Mean Score': ['mean', 'std'],
    })
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{PushT Performance: BFN vs Diffusion Policy}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Mean Score & Std \\",
        r"\midrule",
    ]
    
    for method in ['BFN', 'Diffusion']:
        if method in agg.index:
            mean = agg.loc[method, ('Mean Score', 'mean')]
            std = agg.loc[method, ('Mean Score', 'std')]
            lines.append(f"{method} & {mean:.3f} & {std:.3f} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:pusht_results}",
        r"\end{table}",
    ])
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze PushT Benchmark Results")
    parser.add_argument('--project', type=str, default='pusht_bfn_vs_diffusion',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username or team)')
    parser.add_argument('--output', type=str, default='results/pusht_benchmark',
                        help='Output directory for results')
    parser.add_argument('--local_results', type=str, default=None,
                        help='Path to local results JSON (if not using WandB)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch or load results
    if args.local_results:
        with open(args.local_results) as f:
            runs = json.load(f)
    else:
        print(f"Fetching runs from WandB project: {args.project}")
        runs = fetch_wandb_runs(args.project, args.entity)
    
    if not runs:
        print("No runs found. Please check your WandB project or provide local results.")
        return
    
    print(f"Found {len(runs)} runs")
    
    # Create performance table
    df = create_performance_table(runs)
    print("\n" + "="*60)
    print("Performance Table")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save raw results
    df.to_csv(output_dir / 'performance_raw.csv', index=False)
    
    # Create aggregate table
    agg_df = create_aggregate_table(df)
    print("\n" + "="*60)
    print("Aggregated Results")
    print("="*60)
    print(agg_df.to_string(index=False))
    
    agg_df.to_csv(output_dir / 'performance_aggregate.csv', index=False)
    
    # Statistical tests
    print("\n" + "="*60)
    print("Statistical Tests")
    print("="*60)
    try:
        stats_results = compute_statistical_tests(df)
        print(json.dumps(stats_results, indent=2))
        
        with open(output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
    except ImportError:
        print("scipy not installed, skipping statistical tests")
    
    # Generate plots
    if HAS_PLOT:
        plot_training_curves(runs, output_dir)
        plot_performance_boxplot(df, output_dir)
    
    # Generate LaTeX table
    latex = generate_latex_table(df)
    print("\n" + "="*60)
    print("LaTeX Table")
    print("="*60)
    print(latex)
    
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
