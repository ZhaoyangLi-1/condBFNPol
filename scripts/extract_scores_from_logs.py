#!/usr/bin/env python
"""
Extract evaluation scores from training logs when checkpoints are unavailable.

This script extracts the test_mean_score values from logs.json.txt files
to create a summary table when checkpoints are corrupted or unavailable.

Usage:
    python scripts/extract_scores_from_logs.py --checkpoint-dir cluster_checkpoints/benchmarkresults
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

def extract_scores_from_logs(base_dir: str = "cluster_checkpoints/benchmarkresults") -> Dict:
    """Extract test scores from logs.json.txt files."""
    base_path = Path(base_dir)
    results = {'bfn': {}, 'diffusion': {}}
    
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir():
            continue
        
        for run_dir in date_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            dir_name = run_dir.name
            method = None
            if "bfn" in dir_name.lower():
                method = "bfn"
            elif "diffusion" in dir_name.lower():
                method = "diffusion"
            else:
                continue
            
            seed = None
            for part in dir_name.split("_"):
                if part.startswith("seed"):
                    try:
                        seed = int(part.replace("seed", ""))
                    except ValueError:
                        pass
            
            if seed is None:
                continue
            
            log_file = run_dir / "logs.json.txt"
            if not log_file.exists():
                continue
            
            # Parse log file for test_mean_score entries
            scores = []
            epochs = []
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if 'test/mean_score' in entry:
                            scores.append(entry['test/mean_score'])
                            epochs.append(entry.get('epoch', 0))
                    except:
                        pass
            
            if scores:
                results[method][seed] = {
                    'scores': scores,
                    'epochs': epochs,
                    'max_score': max(scores),
                    'max_epoch': epochs[scores.index(max(scores))],
                    'final_score': scores[-1],
                    'final_epoch': epochs[-1],
                }
                print(f"{method} seed {seed}: max={max(scores):.3f} at epoch {epochs[scores.index(max(scores))]}, final={scores[-1]:.3f} at epoch {epochs[-1]}")
    
    return results

def generate_summary_table(results: Dict, output_file: str = "figures/tables/table_scores_from_logs.tex"):
    """Generate LaTeX table from extracted scores."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate across seeds
    bfn_scores = [data['max_score'] for data in results['bfn'].values()]
    diff_scores = [data['max_score'] for data in results['diffusion'].values()]
    
    table = r"""% =============================================================================
% TABLE: Performance Scores Extracted from Training Logs
% Note: These are evaluation scores from training runs, not full ablation study
% =============================================================================

\begin{table}[t]
\centering
\caption{\textbf{Training Evaluation Scores (from Logs).} 
Maximum test scores achieved during training for BFN and Diffusion policies.
These scores represent the best performance observed at default inference steps 
(20 for BFN, 100 for Diffusion) during training evaluation.}
\label{tab:scores_from_logs}
\vspace{0.5em}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{Max Score} & \textbf{Mean $\pm$ Std} & \textbf{Seeds} \\
\midrule
"""
    
    if bfn_scores:
        table += f"BFN Policy & {max(bfn_scores):.3f} & ${np.mean(bfn_scores):.3f} \\pm {np.std(bfn_scores):.3f}$ & {len(bfn_scores)} \\\\\n"
    
    if diff_scores:
        table += f"Diffusion Policy & {max(diff_scores):.3f} & ${np.mean(diff_scores):.3f} \\pm {np.std(diff_scores):.3f}$ & {len(diff_scores)} \\\\\n"
    
    table += r"""\bottomrule
\end{tabular}
\vspace{0.3em}
\raggedright
\footnotesize
\textit{Note:} These scores are from training-time evaluations at default inference steps.
For ablation study with different inference steps, working checkpoints are required.
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"\n✓ Summary table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract scores from training logs")
    parser.add_argument("--checkpoint-dir", type=str, 
                        default="cluster_checkpoints/benchmarkresults",
                        help="Base directory containing checkpoints and logs")
    parser.add_argument("--output", type=str,
                        default="figures/tables/table_scores_from_logs.tex",
                        help="Output LaTeX table file")
    
    args = parser.parse_args()
    
    print("Extracting scores from training logs...")
    results = extract_scores_from_logs(args.checkpoint_dir)
    
    if results['bfn'] or results['diffusion']:
        generate_summary_table(results, args.output)
        print("\n✓ Score extraction complete")
    else:
        print("No scores found in logs")

if __name__ == "__main__":
    main()
