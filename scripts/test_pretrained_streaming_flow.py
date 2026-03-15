#!/usr/bin/env python3
"""
Test pretrained Streaming Flow Policy checkpoint on training dataset.
Load checkpoint, evaluate on 32 batches, and visualize results.
"""

import os
import sys
import pathlib
import random
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
dp_root = os.path.join(project_root, "src", "diffusion-policy")
if os.path.isdir(dp_root) and dp_root not in sys.path:
    sys.path.insert(0, dp_root)

from omegaconf import OmegaConf
from datetime import datetime

# Register resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)
OmegaConf.register_new_resolver("oc.env", lambda var, default=None: os.environ.get(var, default), replace=True)

from workspaces.train_streaming_flow_workspace import TrainStreamingFlowWorkspace


class StreamingFlowTester:
    """Test pretrained streaming flow policy."""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        
        # Results storage
        self.test_results = {
            'action_errors': [],
            'batch_losses': [],
            'predictions': [],
            'ground_truths': [],
            'batch_indices': []
        }
        
    def load_model(self):
        """Load the pretrained model."""
        print("🔧 LOADING PRETRAINED MODEL")
        print("=" * 50)
        
        # Load config
        print(f"Loading config from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            cfg = OmegaConf.load(f)
            
        # Handle environment variables and time interpolations
        if hasattr(cfg, 'output_root') and '${oc.env:BFN_OUTPUT_ROOT' in str(cfg.output_root):
            cfg.output_root = "/scr2/zhaoyang/BFN_outputs"
        if hasattr(cfg, 'dataset_root') and '${oc.env:BFN_DATA_ROOT' in str(cfg.dataset_root):
            cfg.dataset_root = "/scr2/zhaoyang/BFN_data"
            
        print("✓ Config loaded successfully")
        
        # Create workspace
        output_dir = pathlib.Path("/tmp/streaming_flow_test")
        output_dir.mkdir(exist_ok=True)
        workspace = TrainStreamingFlowWorkspace(cfg, output_dir=output_dir)
        print("✓ Workspace created")
        
        # Load checkpoint
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Load model weights
        if 'state_dicts' in checkpoint and 'policy' in checkpoint['state_dicts']:
            model_state_dict = checkpoint['state_dicts']['policy']
            missing_keys, unexpected_keys = workspace.policy.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning - Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Warning - Unexpected keys: {len(unexpected_keys)}")
                
            print("✓ Model weights loaded successfully")
        else:
            raise ValueError("No policy state dict found in checkpoint")
            
        # Set to evaluation mode and move to device
        workspace.policy.eval()
        workspace.policy = workspace.policy.to(self.device)
        
        self.workspace = workspace
        self.model = workspace.policy
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model loaded: {total_params:,} parameters")
        print(f"✓ Device: {self.device}")
        
    def test_on_batches(self, num_batches: int = 32):
        """Test model on specified number of batches."""
        print(f"\n🧪 TESTING ON {num_batches} BATCHES")
        print("=" * 50)
        
        dataloader = self.workspace.dataloader
        
        # Reset results
        self.test_results = {
            'action_errors': [],
            'batch_losses': [],
            'predictions': [],
            'ground_truths': [],
            'batch_indices': []
        }
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Testing batches", total=min(num_batches, len(dataloader)))
            
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= num_batches:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                batch_size = len(batch['obs']['camera_0'])
                
                try:
                    # Get model prediction
                    result = self.model.predict_action(batch['obs'])
                    
                    # Handle tensor/numpy conversion
                    if isinstance(result['action'], np.ndarray):
                        pred_actions = torch.from_numpy(result['action']).float().to(self.device)
                    else:
                        pred_actions = result['action']
                    
                    # Get ground truth actions
                    gt_actions = batch['action']
                    
                    # Align dimensions (model predicts n_action_steps, GT has full horizon)
                    n_action_steps = pred_actions.shape[1]
                    gt_actions_aligned = gt_actions[:, :n_action_steps, :]
                    
                    # Compute metrics
                    action_error = F.mse_loss(pred_actions, gt_actions_aligned, reduction='none')
                    action_error_per_sample = action_error.mean(dim=(1, 2))  # [B]
                    
                    # Store results
                    self.test_results['action_errors'].extend(action_error_per_sample.cpu().numpy())
                    self.test_results['batch_indices'].extend([batch_idx] * batch_size)
                    
                    # Store some predictions for visualization (first few samples)
                    if batch_idx < 5:  # Store first 5 batches for visualization
                        self.test_results['predictions'].append(pred_actions[:3].cpu().numpy())  # First 3 samples
                        self.test_results['ground_truths'].append(gt_actions_aligned[:3].cpu().numpy())
                    
                    # Try to compute loss if possible
                    try:
                        loss_dict = self.model.compute_loss(batch)
                        if isinstance(loss_dict['loss'], torch.Tensor):
                            loss_value = loss_dict['loss'].item()
                        else:
                            loss_value = float(loss_dict['loss'])
                        self.test_results['batch_losses'].append(loss_value)
                    except:
                        self.test_results['batch_losses'].append(float('nan'))
                    
                    # Update progress bar
                    avg_error = np.mean(action_error_per_sample.cpu().numpy())
                    pbar.set_postfix({'Avg Error': f'{avg_error:.4f}'})
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        print(f"✓ Tested on {len(self.test_results['action_errors'])} samples")
        
    def compute_statistics(self):
        """Compute and display test statistics."""
        print(f"\n📊 TEST STATISTICS")
        print("=" * 50)
        
        action_errors = np.array(self.test_results['action_errors'])
        batch_losses = [l for l in self.test_results['batch_losses'] if not np.isnan(l)]
        
        # Action error statistics
        print("Action Prediction Errors:")
        print(f"  Mean: {np.mean(action_errors):.6f}")
        print(f"  Std:  {np.std(action_errors):.6f}")
        print(f"  Min:  {np.min(action_errors):.6f}")
        print(f"  Max:  {np.max(action_errors):.6f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(action_errors, p):.6f}")
        
        # Loss statistics
        if batch_losses:
            print(f"\nBatch Losses ({len(batch_losses)} valid):")
            print(f"  Mean: {np.mean(batch_losses):.6f}")
            print(f"  Std:  {np.std(batch_losses):.6f}")
        else:
            print(f"\nBatch Losses: All invalid (likely computation issues)")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        good_samples = np.sum(action_errors < 0.5)
        moderate_samples = np.sum((action_errors >= 0.5) & (action_errors < 1.5))
        poor_samples = np.sum(action_errors >= 1.5)
        total_samples = len(action_errors)
        
        print(f"  Good (error < 0.5):     {good_samples:4d} ({good_samples/total_samples*100:.1f}%)")
        print(f"  Moderate (0.5-1.5):     {moderate_samples:4d} ({moderate_samples/total_samples*100:.1f}%)")
        print(f"  Poor (error > 1.5):     {poor_samples:4d} ({poor_samples/total_samples*100:.1f}%)")
        
        return {
            'mean_error': np.mean(action_errors),
            'std_error': np.std(action_errors),
            'good_ratio': good_samples/total_samples,
            'moderate_ratio': moderate_samples/total_samples,
            'poor_ratio': poor_samples/total_samples
        }
    
    def visualize_results(self, save_path: str = "streaming_flow_test_results.png"):
        """Create comprehensive visualization of test results."""
        print(f"\n📈 CREATING VISUALIZATIONS")
        print("=" * 50)
        
        action_errors = np.array(self.test_results['action_errors'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Streaming Flow Policy Test Results', fontsize=16, fontweight='bold')
        
        # 1. Error distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(action_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(action_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(action_errors):.3f}')
        ax1.axvline(np.median(action_errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(action_errors):.3f}')
        ax1.set_xlabel('Action Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error over batches
        ax2 = axes[0, 1]
        batch_indices = self.test_results['batch_indices']
        ax2.scatter(batch_indices, action_errors, alpha=0.6, s=10, c='blue')
        
        # Compute rolling average
        unique_batches = sorted(set(batch_indices))
        batch_means = []
        for batch_idx in unique_batches:
            batch_errors = [action_errors[i] for i, b in enumerate(batch_indices) if b == batch_idx]
            batch_means.append(np.mean(batch_errors))
        
        ax2.plot(unique_batches, batch_means, color='red', linewidth=2, label='Batch Average')
        ax2.set_xlabel('Batch Index')
        ax2.set_ylabel('Action Prediction Error')
        ax2.set_title('Error Over Batches')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot of errors
        ax3 = axes[0, 2]
        ax3.boxplot(action_errors, vert=True)
        ax3.set_ylabel('Action Prediction Error')
        ax3.set_title('Error Distribution (Box Plot)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction vs Ground Truth (sample)
        ax4 = axes[1, 0]
        if self.test_results['predictions'] and self.test_results['ground_truths']:
            # Show first sample from first batch
            pred_sample = self.test_results['predictions'][0][0]  # [T, D]
            gt_sample = self.test_results['ground_truths'][0][0]  # [T, D]
            
            time_steps = range(pred_sample.shape[0])
            
            # Plot first 3 action dimensions
            for dim in range(min(3, pred_sample.shape[1])):
                ax4.plot(time_steps, pred_sample[:, dim], '--', label=f'Pred dim {dim}', alpha=0.8)
                ax4.plot(time_steps, gt_sample[:, dim], '-', label=f'GT dim {dim}', alpha=0.8)
            
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Action Value')
            ax4.set_title('Sample Prediction vs Ground Truth')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sample Predictions (No Data)')
        
        # 5. Performance categories pie chart
        ax5 = axes[1, 1]
        good_count = np.sum(action_errors < 0.5)
        moderate_count = np.sum((action_errors >= 0.5) & (action_errors < 1.5))
        poor_count = np.sum(action_errors >= 1.5)
        
        sizes = [good_count, moderate_count, poor_count]
        labels = ['Good (<0.5)', 'Moderate (0.5-1.5)', 'Poor (>1.5)']
        colors = ['green', 'orange', 'red']
        
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Performance Categories')
        
        # 6. Cumulative error distribution
        ax6 = axes[1, 2]
        sorted_errors = np.sort(action_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax6.plot(sorted_errors, cumulative, linewidth=2, color='purple')
        ax6.set_xlabel('Action Prediction Error')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('Cumulative Error Distribution')
        ax6.grid(True, alpha=0.3)
        
        # Add reference lines
        for threshold in [0.5, 1.0, 1.5]:
            idx = np.searchsorted(sorted_errors, threshold)
            if idx < len(cumulative):
                ax6.axvline(threshold, color='red', linestyle=':', alpha=0.7)
                ax6.text(threshold, cumulative[idx], f'{cumulative[idx]:.1%}', rotation=90, va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
        
        # Also create action dimension comparison if possible
        if self.test_results['predictions'] and self.test_results['ground_truths']:
            self._visualize_action_dimensions(save_path.replace('.png', '_action_dims.png'))
    
    def _visualize_action_dimensions(self, save_path: str):
        """Visualize predictions for each action dimension separately."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 7 dims + 1 extra
        fig.suptitle('Prediction vs Ground Truth by Action Dimension', fontsize=16)
        
        # Collect all predictions and ground truths
        all_preds = []
        all_gts = []
        
        for batch_preds, batch_gts in zip(self.test_results['predictions'], self.test_results['ground_truths']):
            all_preds.extend(batch_preds)
            all_gts.extend(batch_gts)
        
        all_preds = np.array(all_preds)  # [N, T, D]
        all_gts = np.array(all_gts)      # [N, T, D]
        
        action_dim = all_preds.shape[2]
        
        for dim in range(min(action_dim, 7)):  # Show up to 7 dimensions
            row = dim // 4
            col = dim % 4
            ax = axes[row, col]
            
            # Show multiple samples
            for i in range(min(5, len(all_preds))):
                time_steps = range(all_preds.shape[1])
                ax.plot(time_steps, all_preds[i, :, dim], 'b--', alpha=0.6, linewidth=1)
                ax.plot(time_steps, all_gts[i, :, dim], 'r-', alpha=0.6, linewidth=1)
            
            # Add legend only on first subplot
            if dim == 0:
                ax.plot([], [], 'b--', label='Predictions', alpha=0.8)
                ax.plot([], [], 'r-', label='Ground Truth', alpha=0.8)
                ax.legend()
            
            ax.set_title(f'Action Dimension {dim}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action Value')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for dim in range(action_dim, 8):
            row = dim // 4
            col = dim % 4
            if row < 2 and col < 4:
                axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Action dimension visualization saved to: {save_path}")
    
    def run_full_test(self, num_batches: int = 32):
        """Run complete test pipeline."""
        print("🚀 STREAMING FLOW POLICY TESTING")
        print("=" * 80)
        
        # Load model
        self.load_model()
        
        # Test on batches
        self.test_on_batches(num_batches)
        
        # Compute statistics
        stats = self.compute_statistics()
        
        # Create visualizations
        self.visualize_results()
        
        # Final summary
        print(f"\n🏁 TESTING COMPLETE")
        print("=" * 50)
        print(f"✓ Tested {len(self.test_results['action_errors'])} samples")
        print(f"✓ Mean action error: {stats['mean_error']:.4f}")
        print(f"✓ Good performance: {stats['good_ratio']*100:.1f}% of samples")
        print(f"✓ Visualizations saved")
        
        # Performance verdict
        if stats['mean_error'] < 0.5:
            print("🟢 EXCELLENT: Very low prediction errors!")
        elif stats['mean_error'] < 1.0:
            print("🟡 GOOD: Moderate prediction errors, acceptable for most tasks.")
        elif stats['mean_error'] < 2.0:
            print("🟠 FAIR: Higher prediction errors, may need improvement.")
        else:
            print("🔴 POOR: High prediction errors, significant issues.")


def main():
    """Main testing function."""
    # Configuration
    checkpoint_path = "/scr2/zhaoyang/BFN_outputs/2026.03.02/08.30.31_streaming_flow_trainable_real_image_seed42/checkpoints/epoch=0093-val_loss=0.000.ckpt"
    config_path = "/scr2/zhaoyang/condBFNPol/config/train_streaming_flow_pusht_real.yaml"
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create tester and run
    tester = StreamingFlowTester(checkpoint_path, config_path, device)
    tester.run_full_test(num_batches=32)


if __name__ == "__main__":
    main()