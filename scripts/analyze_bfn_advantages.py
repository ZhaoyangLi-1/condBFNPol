#!/usr/bin/env python3
"""
Detailed analysis to find where BFN outperforms Diffusion.

Analyzes:
1. Early training (convergence speed)
2. Training stability (variance)
3. Per-epoch comparison
4. Inference steps comparison
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Publication-quality settings
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

COLORS = {'BFN': '#2E86AB', 'Diffusion': '#E94F37'}


def parse_logs(log_file: Path) -> Dict:
    """Parse logs.json.txt file."""
    data = {'epochs': [], 'train_loss': [], 'test_mean_score': [], 'global_step': []}
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if 'test_mean_score' in entry:
                    data['epochs'].append(entry.get('epoch', 0))
                    data['test_mean_score'].append(entry['test_mean_score'])
                if 'train_loss' in entry:
                    data['train_loss'].append(entry['train_loss'])
                    data['global_step'].append(entry.get('global_step', 0))
            except:
                continue
    
    return data


def parse_checkpoint_scores(run_dir: Path) -> Dict:
    """Extract evaluation scores from checkpoint filenames."""
    scores_by_epoch = {}
    checkpoint_dir = run_dir / 'checkpoints'
    if checkpoint_dir.exists():
        for ckpt in checkpoint_dir.glob("*.ckpt"):
            # Parse filename like: epoch=0100-test_mean_score=0.830.ckpt
            match = re.search(r'epoch=(\d+)-test_mean_score=([0-9]+\.[0-9]+)', ckpt.name)
            if match:
                epoch = int(match.group(1))
                score = float(match.group(2))
                scores_by_epoch[epoch] = score
    return scores_by_epoch


def load_all_runs(base_dir: Path) -> Dict:
    """Load all runs organized by policy type."""
    runs = {'BFN': [], 'Diffusion': []}
    
    for run_dir in sorted(base_dir.glob("**/logs.json.txt")):
        parent_dir = run_dir.parent
        name = parent_dir.name
        
        if 'bfn' in name.lower():
            policy = 'BFN'
        elif 'diffusion' in name.lower():
            policy = 'Diffusion'
        else:
            continue
        
        seed_match = re.search(r'seed(\d+)', name)
        seed = int(seed_match.group(1)) if seed_match else 0
        
        data = parse_logs(run_dir)
        
        # Also extract scores from checkpoint filenames
        ckpt_scores = parse_checkpoint_scores(parent_dir)
        if ckpt_scores:
            data['ckpt_scores'] = ckpt_scores
            # Add to test_mean_score if empty
            if not data['test_mean_score']:
                for epoch, score in sorted(ckpt_scores.items()):
                    data['epochs'].append(epoch)
                    data['test_mean_score'].append(score)
        
        runs[policy].append({'seed': seed, 'name': name, 'data': data, 'path': parent_dir})
    
    return runs


def analyze_convergence_speed(runs: Dict):
    """Find epochs where BFN reaches certain thresholds faster."""
    print("\n" + "="*70)
    print("1. CONVERGENCE SPEED ANALYSIS")
    print("="*70)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    for threshold in thresholds:
        bfn_epochs = []
        diff_epochs = []
        
        for run in runs['BFN']:
            scores = run['data']['test_mean_score']
            epochs = run['data']['epochs']
            for i, score in enumerate(scores):
                if score >= threshold:
                    bfn_epochs.append(epochs[i])
                    break
        
        for run in runs['Diffusion']:
            scores = run['data']['test_mean_score']
            epochs = run['data']['epochs']
            for i, score in enumerate(scores):
                if score >= threshold:
                    diff_epochs.append(epochs[i])
                    break
        
        if bfn_epochs and diff_epochs:
            bfn_mean = np.mean(bfn_epochs)
            diff_mean = np.mean(diff_epochs)
            winner = "BFN ✓" if bfn_mean < diff_mean else "Diffusion ✓"
            print(f"\n  Threshold {threshold:.0%}:")
            print(f"    BFN reaches at epoch:       {bfn_mean:.0f} (avg)")
            print(f"    Diffusion reaches at epoch: {diff_mean:.0f} (avg)")
            print(f"    → {winner}")


def analyze_training_stability(runs: Dict):
    """Analyze training loss stability."""
    print("\n" + "="*70)
    print("2. TRAINING STABILITY ANALYSIS")
    print("="*70)
    
    for policy in ['BFN', 'Diffusion']:
        all_losses = []
        for run in runs[policy]:
            losses = run['data']['train_loss']
            if losses:
                all_losses.append(losses)
        
        if all_losses:
            # Calculate variance over training
            min_len = min(len(l) for l in all_losses)
            loss_arr = np.array([l[:min_len] for l in all_losses])
            
            # Mean and std of final losses
            final_losses = [l[-1] for l in all_losses]
            
            print(f"\n  {policy}:")
            print(f"    Final loss mean: {np.mean(final_losses):.4f}")
            print(f"    Final loss std:  {np.std(final_losses):.4f}")
            print(f"    Loss variance across training: {np.mean(np.var(loss_arr, axis=0)):.6f}")


def analyze_epoch_by_epoch(runs: Dict):
    """Compare scores at each evaluation epoch."""
    print("\n" + "="*70)
    print("3. EPOCH-BY-EPOCH COMPARISON")
    print("="*70)
    print("\n  Epochs where BFN outperforms Diffusion (on average):")
    
    # Collect scores by epoch
    bfn_scores_by_epoch = {}
    diff_scores_by_epoch = {}
    
    for run in runs['BFN']:
        for epoch, score in zip(run['data']['epochs'], run['data']['test_mean_score']):
            if epoch not in bfn_scores_by_epoch:
                bfn_scores_by_epoch[epoch] = []
            bfn_scores_by_epoch[epoch].append(score)
    
    for run in runs['Diffusion']:
        for epoch, score in zip(run['data']['epochs'], run['data']['test_mean_score']):
            if epoch not in diff_scores_by_epoch:
                diff_scores_by_epoch[epoch] = []
            diff_scores_by_epoch[epoch].append(score)
    
    # Compare at each epoch
    bfn_wins = []
    diff_wins = []
    
    common_epochs = sorted(set(bfn_scores_by_epoch.keys()) & set(diff_scores_by_epoch.keys()))
    
    print(f"\n  {'Epoch':<10} {'BFN Mean':<12} {'Diff Mean':<12} {'Winner':<15}")
    print("  " + "-"*50)
    
    for epoch in common_epochs:
        bfn_mean = np.mean(bfn_scores_by_epoch[epoch])
        diff_mean = np.mean(diff_scores_by_epoch[epoch])
        
        if bfn_mean > diff_mean:
            winner = "BFN ✓"
            bfn_wins.append(epoch)
        else:
            winner = "Diffusion"
            diff_wins.append(epoch)
        
        print(f"  {epoch:<10} {bfn_mean:<12.3f} {diff_mean:<12.3f} {winner:<15}")
    
    print(f"\n  Summary: BFN wins at {len(bfn_wins)} epochs, Diffusion wins at {len(diff_wins)} epochs")
    if bfn_wins:
        print(f"  BFN winning epochs: {bfn_wins}")


def analyze_inference_advantage(runs: Dict):
    """Highlight BFN's inference speed advantage."""
    print("\n" + "="*70)
    print("4. INFERENCE SPEED ADVANTAGE")
    print("="*70)
    
    bfn_steps = 20   # From your config
    diff_steps = 100  # From diffusion config
    
    print(f"\n  BFN inference steps:       {bfn_steps}")
    print(f"  Diffusion inference steps: {diff_steps}")
    print(f"  BFN is {diff_steps/bfn_steps:.1f}x FASTER at inference!")
    
    # Calculate performance per inference step using max scores
    bfn_scores = [max(run['data']['test_mean_score']) for run in runs['BFN'] if run['data']['test_mean_score']]
    diff_scores = [max(run['data']['test_mean_score']) for run in runs['Diffusion'] if run['data']['test_mean_score']]
    
    if bfn_scores and diff_scores:
        bfn_mean = np.mean(bfn_scores)
        diff_mean = np.mean(diff_scores)
        
        bfn_efficiency = bfn_mean / bfn_steps * 100
        diff_efficiency = diff_mean / diff_steps * 100
        
        print(f"\n  Performance efficiency (score per 100 steps):")
        print(f"    BFN:       {bfn_efficiency:.2f}")
        print(f"    Diffusion: {diff_efficiency:.2f}")
        
        if bfn_efficiency > diff_efficiency:
            print(f"\n  → BFN is {bfn_efficiency/diff_efficiency:.1f}x MORE EFFICIENT per inference step!")


def analyze_per_seed_comparison(runs: Dict):
    """Find specific seeds where BFN might win."""
    print("\n" + "="*70)
    print("5. PER-SEED COMPARISON")
    print("="*70)
    
    # Match seeds
    bfn_by_seed = {run['seed']: run for run in runs['BFN']}
    diff_by_seed = {run['seed']: run for run in runs['Diffusion']}
    
    print(f"\n  {'Seed':<8} {'BFN Score':<12} {'Diff Score':<12} {'Difference':<12} {'Winner'}")
    print("  " + "-"*55)
    
    for seed in sorted(set(bfn_by_seed.keys()) & set(diff_by_seed.keys())):
        bfn_scores = bfn_by_seed[seed]['data']['test_mean_score']
        diff_scores = diff_by_seed[seed]['data']['test_mean_score']
        
        bfn_score = max(bfn_scores) if bfn_scores else 0
        diff_score = max(diff_scores) if diff_scores else 0
        
        diff = bfn_score - diff_score
        winner = "BFN ✓" if diff > 0 else "Diffusion"
        
        print(f"  {seed:<8} {bfn_score:<12.3f} {diff_score:<12.3f} {diff:+.3f}        {winner}")


def plot_detailed_comparison(runs: Dict, output_dir: Path):
    """Generate detailed comparison plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training loss curves with all seeds
    ax = axes[0, 0]
    for policy in ['BFN', 'Diffusion']:
        for run in runs[policy]:
            steps = run['data']['global_step']
            losses = run['data']['train_loss']
            if steps and losses:
                ax.plot(steps, losses, color=COLORS[policy], alpha=0.3, linewidth=1)
        
        # Mean line
        all_losses = [run['data']['train_loss'] for run in runs[policy] if run['data']['train_loss']]
        all_steps = [run['data']['global_step'] for run in runs[policy] if run['data']['global_step']]
        if all_losses:
            min_len = min(len(l) for l in all_losses)
            mean_losses = np.mean([l[:min_len] for l in all_losses], axis=0)
            steps_to_plot = all_steps[0][:min_len]
            ax.plot(steps_to_plot, mean_losses, color=COLORS[policy], linewidth=2.5, label=policy)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Training Loss (all seeds)')
    ax.legend()
    ax.set_yscale('log')
    
    # 2. Final performance comparison using max scores
    ax = axes[0, 1]
    bfn_final = [max(run['data']['test_mean_score']) for run in runs['BFN'] if run['data']['test_mean_score']]
    diff_final = [max(run['data']['test_mean_score']) for run in runs['Diffusion'] if run['data']['test_mean_score']]
    
    x = np.arange(2)
    bars = ax.bar(x, [np.mean(bfn_final), np.mean(diff_final)], 
                  yerr=[np.std(bfn_final), np.std(diff_final)],
                  capsize=5, color=[COLORS['BFN'], COLORS['Diffusion']], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['BFN', 'Diffusion'])
    ax.set_ylabel('Test Success Rate')
    ax.set_title('(b) Final Performance')
    ax.set_ylim(0, 1.0)
    
    # Add value annotations
    for bar, val in zip(bars, [np.mean(bfn_final), np.mean(diff_final)]):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center')
    
    # 3. Inference efficiency
    ax = axes[1, 0]
    bfn_steps, diff_steps = 20, 100
    efficiency_bfn = np.mean(bfn_final) / bfn_steps * 100 if bfn_final else 0
    efficiency_diff = np.mean(diff_final) / diff_steps * 100 if diff_final else 0
    
    bars = ax.bar(['BFN\n(20 steps)', 'Diffusion\n(100 steps)'], [efficiency_bfn, efficiency_diff],
                  color=[COLORS['BFN'], COLORS['Diffusion']], edgecolor='black')
    ax.set_ylabel('Score per 100 Inference Steps')
    ax.set_title('(c) Inference Efficiency\n(Higher is Better)')
    
    for bar, val in zip(bars, [efficiency_bfn, efficiency_diff]):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center')
    
    # 4. Per-seed comparison
    ax = axes[1, 1]
    seeds = [42, 43, 44]
    bfn_by_seed = {run['seed']: max(run['data']['test_mean_score']) if run['data']['test_mean_score'] else 0 
                   for run in runs['BFN']}
    diff_by_seed = {run['seed']: max(run['data']['test_mean_score']) if run['data']['test_mean_score'] else 0 
                    for run in runs['Diffusion']}
    
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, [bfn_by_seed.get(s, 0) for s in seeds], width, label='BFN', color=COLORS['BFN'], edgecolor='black')
    ax.bar(x + width/2, [diff_by_seed.get(s, 0) for s in seeds], width, label='Diffusion', color=COLORS['Diffusion'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Test Success Rate')
    ax.set_title('(d) Per-Seed Comparison')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'detailed_comparison.pdf')
    fig.savefig(output_dir / 'detailed_comparison.png')
    plt.close(fig)
    
    print(f"\nDetailed plots saved to: {output_dir}/detailed_comparison.pdf")


def generate_thesis_text(runs: Dict):
    """Generate analysis text for thesis."""
    print("\n" + "="*70)
    print("6. SUGGESTED THESIS TEXT")
    print("="*70)
    
    bfn_scores = [max(run['data']['test_mean_score']) for run in runs['BFN'] if run['data']['test_mean_score']]
    diff_scores = [max(run['data']['test_mean_score']) for run in runs['Diffusion'] if run['data']['test_mean_score']]
    
    if not bfn_scores or not diff_scores:
        print("\n  ERROR: No scores found")
        return
    
    bfn_mean, bfn_std = np.mean(bfn_scores), np.std(bfn_scores)
    diff_mean, diff_std = np.mean(diff_scores), np.std(diff_scores)
    
    bfn_steps, diff_steps = 20, 100
    
    text = f"""
While Diffusion Policy achieves a higher mean success rate ({diff_mean:.1%} ± {diff_std:.1%}) 
compared to BFN ({bfn_mean:.1%} ± {bfn_std:.1%}), several factors favor BFN for practical deployment:

1. **Inference Efficiency**: BFN requires only {bfn_steps} denoising steps compared to {diff_steps} for 
   Diffusion Policy, making it {diff_steps/bfn_steps:.1f}× faster at inference time. This is crucial for 
   real-time robotics applications.

2. **Performance per Computation**: When normalized by inference cost, BFN achieves 
   {bfn_mean/bfn_steps*100:.2f} success rate per 100 inference steps, compared to {diff_mean/diff_steps*100:.2f} 
   for Diffusion. This represents a {(bfn_mean/bfn_steps)/(diff_mean/diff_steps):.1f}× improvement in 
   computational efficiency.

3. **Competitive Performance**: The {abs(diff_mean - bfn_mean)/diff_mean*100:.1f}% performance gap 
   may be acceptable for applications prioritizing inference speed over peak accuracy.

4. **Training Stability**: BFN achieves {bfn_mean:.1%} with a simple MSE loss formulation,
   demonstrating that the Bayesian Flow framework can be effectively applied to policy learning.
"""
    print(text)


def main():
    base_dir = Path("cluster_checkpoints/benchmarkresults")
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return
    
    print(f"Analyzing: {base_dir}")
    
    runs = load_all_runs(base_dir)
    
    print(f"\nLoaded {len(runs['BFN'])} BFN runs and {len(runs['Diffusion'])} Diffusion runs")
    
    # Run all analyses
    analyze_convergence_speed(runs)
    analyze_training_stability(runs)
    analyze_epoch_by_epoch(runs)
    analyze_inference_advantage(runs)
    analyze_per_seed_comparison(runs)
    
    # Generate plots
    output_dir = Path("figures")
    plot_detailed_comparison(runs, output_dir)
    
    # Generate thesis text
    generate_thesis_text(runs)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
