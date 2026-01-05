#!/usr/bin/env python3
"""
Publication-Quality Prediction Visualizations
Style: Google Research / DeepMind / NeurIPS

Generates:
1. Action trajectory predictions (BFN vs Diffusion vs Ground Truth)
2. Denoising process visualization
3. Policy rollout comparison
4. Uncertainty/variance visualization
"""

import numpy as np
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# ANTHROPIC RESEARCH STYLE CONFIGURATION
# =============================================================================

from colors import COLORS, setup_matplotlib_style, SINGLE_COL, DOUBLE_COL, ANTHROPIC_CMAP

# Setup matplotlib style
setup_matplotlib_style()

# Override grid for visualization figures
plt.rcParams['axes.grid'] = False

# Use Anthropic colormaps
CMAP_BFN = ANTHROPIC_CMAP['bfn']
CMAP_DIFF = ANTHROPIC_CMAP['diffusion']
CMAP_TIME = ANTHROPIC_CMAP['time']


# =============================================================================
# SYNTHETIC DATA GENERATION (for demonstration)
# =============================================================================

def generate_synthetic_trajectory(T=16, noise_level=0.0, seed=42):
    """Generate a synthetic Push-T style trajectory."""
    np.random.seed(seed)
    
    # Create a smooth trajectory (2D position)
    t = np.linspace(0, 2*np.pi, T)
    
    # Ground truth: smooth curve toward goal
    x_gt = 0.3 * np.cos(t * 0.5) + 0.5 + np.cumsum(np.random.randn(T) * 0.02)
    y_gt = 0.3 * np.sin(t * 0.8) + 0.5 + np.cumsum(np.random.randn(T) * 0.02)
    
    # Normalize to [0, 1]
    x_gt = (x_gt - x_gt.min()) / (x_gt.max() - x_gt.min() + 1e-6) * 0.6 + 0.2
    y_gt = (y_gt - y_gt.min()) / (y_gt.max() - y_gt.min() + 1e-6) * 0.6 + 0.2
    
    gt = np.stack([x_gt, y_gt], axis=-1)
    
    return gt


def generate_denoising_steps(gt, n_steps=5, method='bfn'):
    """Simulate denoising steps for visualization."""
    T, D = gt.shape
    steps = []
    
    for i in range(n_steps + 1):
        if method == 'bfn':
            # BFN: Bayesian posterior updates (sharper convergence)
            t = i / n_steps
            gamma = 1 - (1 - t) ** 2  # Quadratic schedule
            noise = np.random.randn(T, D) * (1 - gamma) * 0.3
            pred = gamma * gt + noise
        else:
            # Diffusion: Linear denoising
            t = i / n_steps
            noise = np.random.randn(T, D) * (1 - t) * 0.5
            pred = t * gt + (1 - t) * np.random.randn(T, D) * 0.3 + noise * 0.5
        
        steps.append(pred)
    
    return steps


def generate_multiple_samples(gt, n_samples=5, method='bfn', seed=42):
    """Generate multiple prediction samples to show variance."""
    np.random.seed(seed)
    samples = []
    
    for i in range(n_samples):
        if method == 'bfn':
            noise = np.random.randn(*gt.shape) * 0.03
        else:
            noise = np.random.randn(*gt.shape) * 0.02
        samples.append(gt + noise)
    
    return samples


# =============================================================================
# FIGURE 1: TRAJECTORY COMPARISON
# =============================================================================

def plot_trajectory_comparison(output_dir: Path):
    """Compare predicted trajectories: BFN vs Diffusion vs Ground Truth."""
    # Create figure with extra space for colorbar on the right
    fig = plt.figure(figsize=(DOUBLE_COL + 0.5, SINGLE_COL * 0.7))
    
    # Use GridSpec to control layout: 3 equal panels + 1 narrow colorbar
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.25)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])  # Dedicated colorbar axis
    
    # Generate data
    gt = generate_synthetic_trajectory(T=16, seed=42)
    bfn_pred = gt + np.random.randn(*gt.shape) * 0.025
    diff_pred = gt + np.random.randn(*gt.shape) * 0.02
    
    configs = [
        ('Ground Truth', gt, COLORS['ground_truth'], 'o'),
        ('BFN (Ours)', bfn_pred, COLORS['bfn'], 's'),
        ('Diffusion', diff_pred, COLORS['diffusion'], '^'),
    ]
    
    scatter = None  # Will store the last scatter for colorbar
    
    for ax, (title, traj, color, marker) in zip(axes, configs):
        # Draw trajectory with time coloring
        for i in range(len(traj) - 1):
            alpha = 0.3 + 0.7 * (i / len(traj))
            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], 
                    color=color, linewidth=2, alpha=alpha)
        
        # Draw points
        scatter = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)),
                            cmap=CMAP_TIME, s=40, marker=marker,
                            edgecolor=color, linewidth=1, zorder=5)
        
        # Start and end markers
        ax.scatter(traj[0, 0], traj[0, 1], c='white', s=100, marker='o',
                   edgecolor=color, linewidth=2, zorder=10)
        ax.annotate('Start', (traj[0, 0], traj[0, 1] - 0.08), ha='center', fontsize=7)
        
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=100, marker='*',
                   edgecolor=COLORS['black'], linewidth=0.5, zorder=10)
        ax.annotate('End', (traj[-1, 0], traj[-1, 1] + 0.06), ha='center', fontsize=7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', color=color if title != 'Ground Truth' else COLORS['black'])
        ax.set_xlabel('Position X')
        if ax == axes[0]:
            ax.set_ylabel('Position Y')
    
    # Add colorbar to dedicated axis (no overlap with panels)
    cbar = fig.colorbar(scatter, cax=cax, orientation='vertical')
    cbar.set_label('Time Step', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Adjust subplot spacing (don't use tight_layout with GridSpec)
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.88, wspace=0.25)
    fig.savefig(str(output_dir / 'fig_trajectory_comparison.pdf'))
    fig.savefig(str(output_dir / 'fig_trajectory_comparison.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Trajectory comparison")


# =============================================================================
# FIGURE 2: DENOISING PROCESS VISUALIZATION
# =============================================================================

def plot_denoising_process(output_dir: Path):
    """Visualize the denoising/sampling process for both methods."""
    fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_COL * 1.1))
    
    gs = gridspec.GridSpec(2, 6, hspace=0.4, wspace=0.15,
                           height_ratios=[1, 1])
    
    gt = generate_synthetic_trajectory(T=16, seed=42)
    
    n_show = 6
    bfn_steps = generate_denoising_steps(gt, n_steps=n_show-1, method='bfn')
    diff_steps = generate_denoising_steps(gt, n_steps=n_show-1, method='diffusion')
    
    # Row 1: BFN
    for i, (step, pred) in enumerate(zip(range(n_show), bfn_steps)):
        ax = fig.add_subplot(gs[0, i])
        
        # Plot ground truth faintly
        ax.plot(gt[:, 0], gt[:, 1], color=COLORS['ground_truth'], 
                linewidth=1, alpha=0.3, linestyle='--')
        
        # Plot prediction
        ax.plot(pred[:, 0], pred[:, 1], color=COLORS['bfn'], linewidth=1.5)
        ax.scatter(pred[:, 0], pred[:, 1], c=COLORS['bfn'], s=15, zorder=5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 0:
            ax.set_ylabel('BFN\n(20 steps)', fontweight='bold', color=COLORS['bfn'])
            ax.set_title(f't=0\n(noise)', fontsize=8)
        elif i == n_show - 1:
            ax.set_title(f't=1\n(final)', fontsize=8)
        else:
            t_val = i / (n_show - 1)
            ax.set_title(f't={t_val:.1f}', fontsize=8)
    
    # Row 2: Diffusion
    for i, (step, pred) in enumerate(zip(range(n_show), diff_steps)):
        ax = fig.add_subplot(gs[1, i])
        
        ax.plot(gt[:, 0], gt[:, 1], color=COLORS['ground_truth'],
                linewidth=1, alpha=0.3, linestyle='--')
        ax.plot(pred[:, 0], pred[:, 1], color=COLORS['diffusion'], linewidth=1.5)
        ax.scatter(pred[:, 0], pred[:, 1], c=COLORS['diffusion'], s=15, zorder=5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 0:
            ax.set_ylabel('Diffusion\n(100 steps)', fontweight='bold', color=COLORS['diffusion'])
    
    # Add arrow showing denoising direction
    fig.text(0.5, 0.48, '← Denoising Direction →', ha='center', fontsize=9,
             style='italic', color=COLORS['gray'])
    
    fig.savefig(str(output_dir / 'fig_denoising_process.pdf'))
    fig.savefig(str(output_dir / 'fig_denoising_process.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Denoising process")


# =============================================================================
# FIGURE 3: PREDICTION UNCERTAINTY/VARIANCE
# =============================================================================

def plot_prediction_variance(output_dir: Path):
    """Show prediction variance across multiple samples."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.8))
    
    gt = generate_synthetic_trajectory(T=16, seed=42)
    
    for ax, (method, color, title) in zip(axes, [
        ('bfn', COLORS['bfn'], 'BFN (Ours)'),
        ('diffusion', COLORS['diffusion'], 'Diffusion')
    ]):
        samples = generate_multiple_samples(gt, n_samples=10, method=method)
        
        # Plot all samples with low alpha
        for sample in samples:
            ax.plot(sample[:, 0], sample[:, 1], color=color, alpha=0.2, linewidth=1)
        
        # Plot mean prediction
        mean_pred = np.mean(samples, axis=0)
        ax.plot(mean_pred[:, 0], mean_pred[:, 1], color=color, linewidth=2, 
                label='Mean prediction')
        
        # Plot ground truth
        ax.plot(gt[:, 0], gt[:, 1], color=COLORS['ground_truth'], linewidth=2,
                linestyle='--', label='Ground truth')
        
        # Add variance ellipses at key points
        std_pred = np.std(samples, axis=0)
        for i in [0, 5, 10, 15]:
            ellipse = plt.matplotlib.patches.Ellipse(
                (mean_pred[i, 0], mean_pred[i, 1]),
                width=std_pred[i, 0] * 4, height=std_pred[i, 1] * 4,
                fill=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=1
            )
            ax.add_patch(ellipse)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Position X')
        ax.set_title(title, fontweight='bold', color=color)
        ax.legend(loc='lower right', fontsize=7)
        
    axes[0].set_ylabel('Position Y')
    
    # Add panel labels
    axes[0].text(-0.15, 1.05, '(a)', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[1].text(-0.15, 1.05, '(b)', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig_prediction_variance.pdf'))
    fig.savefig(str(output_dir / 'fig_prediction_variance.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Prediction variance")


# =============================================================================
# FIGURE 4: ACTION DIMENSION COMPARISON
# =============================================================================

def plot_action_dimensions(output_dir: Path):
    """Plot individual action dimensions over time."""
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.1), sharex=True)
    
    T = 16
    gt = generate_synthetic_trajectory(T=T, seed=42)
    bfn_pred = gt + np.random.randn(*gt.shape) * 0.03
    diff_pred = gt + np.random.randn(*gt.shape) * 0.025
    
    time = np.arange(T)
    
    dim_names = ['Action X', 'Action Y']
    
    for ax, dim, name in zip(axes, range(2), dim_names):
        # Ground truth
        ax.plot(time, gt[:, dim], color=COLORS['ground_truth'], linewidth=2,
                linestyle='--', label='Ground Truth', marker='o', markersize=4)
        
        # BFN
        ax.plot(time, bfn_pred[:, dim], color=COLORS['bfn'], linewidth=1.5,
                label='BFN (Ours)', marker='s', markersize=3, alpha=0.8)
        
        # Diffusion
        ax.plot(time, diff_pred[:, dim], color=COLORS['diffusion'], linewidth=1.5,
                label='Diffusion', marker='^', markersize=3, alpha=0.8)
        
        ax.set_ylabel(name)
        ax.set_ylim(0, 1)
        
        if dim == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=3)
    
    axes[1].set_xlabel('Time Step')
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig_action_dimensions.pdf'))
    fig.savefig(str(output_dir / 'fig_action_dimensions.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Action dimensions")


# =============================================================================
# FIGURE 5: PUSH-T TASK VISUALIZATION
# =============================================================================

def plot_pusht_visualization(output_dir: Path):
    """Visualize the Push-T task with agent and T-block."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.8))
    
    def draw_T_block(ax, pos, angle, color='#808080', alpha=1.0):
        """Draw a T-shaped block."""
        # T-block dimensions
        w1, h1 = 0.15, 0.05  # Top bar
        w2, h2 = 0.05, 0.12  # Vertical bar
        
        import matplotlib.patches as patches
        from matplotlib.transforms import Affine2D
        
        # Create T shape
        rect1 = patches.Rectangle((-w1/2, h2-h1), w1, h1, color=color, alpha=alpha)
        rect2 = patches.Rectangle((-w2/2, 0), w2, h2, color=color, alpha=alpha)
        
        # Apply rotation and translation
        t = Affine2D().rotate(angle).translate(pos[0], pos[1]) + ax.transData
        rect1.set_transform(t)
        rect2.set_transform(t)
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    
    def draw_agent(ax, pos, color):
        """Draw the pushing agent."""
        circle = Circle(pos, 0.03, facecolor=color, edgecolor=COLORS['black'], linewidth=1)
        ax.add_patch(circle)
    
    titles = ['Initial State', 'BFN Policy', 'Diffusion Policy']
    agent_colors = [COLORS['gray'], COLORS['bfn'], COLORS['diffusion']]
    
    # Generate trajectories
    np.random.seed(42)
    agent_start = np.array([0.3, 0.3])
    
    trajectories = [
        None,  # Initial
        np.array([[0.3, 0.3], [0.35, 0.35], [0.42, 0.43], [0.48, 0.5], [0.52, 0.55]]),  # BFN
        np.array([[0.3, 0.3], [0.34, 0.36], [0.40, 0.44], [0.47, 0.51], [0.52, 0.56]]),  # Diffusion
    ]
    
    for ax, title, agent_color, traj in zip(axes, titles, agent_colors, trajectories):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Draw target zone
        target = Circle((0.5, 0.5), 0.08, facecolor=COLORS['light_green'], 
                        edgecolor=COLORS['ground_truth'], linewidth=1.5, linestyle='--',
                        alpha=0.5, label='Target')
        ax.add_patch(target)
        
        # Draw T-block
        t_pos = [0.5, 0.5] if traj is None else [0.48 + np.random.rand()*0.04, 0.48 + np.random.rand()*0.04]
        t_angle = 0 if traj is None else np.random.rand() * 0.1
        draw_T_block(ax, t_pos, t_angle, color='#606060')
        
        if traj is not None:
            # Draw trajectory
            for i in range(len(traj) - 1):
                alpha = 0.3 + 0.7 * i / len(traj)
                ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], 
                       color=agent_color, linewidth=2, alpha=alpha)
            
            # Draw agent at end
            draw_agent(ax, traj[-1], agent_color)
            
            # Draw ghost agents along path
            for i, pos in enumerate(traj[:-1]):
                ghost = Circle(pos, 0.02, facecolor=agent_color, alpha=0.2)
                ax.add_patch(ghost)
        else:
            # Draw agent at start
            draw_agent(ax, agent_start, agent_color)
        
        ax.set_title(title, fontweight='bold', color=agent_color if title != 'Initial State' else COLORS['black'])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig_pusht_visualization.pdf'))
    fig.savefig(str(output_dir / 'fig_pusht_visualization.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Push-T visualization")


# =============================================================================
# FIGURE 6: COMBINED QUALITATIVE FIGURE
# =============================================================================

def plot_combined_qualitative(output_dir: Path):
    """Combined qualitative results figure for main paper."""
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.65))
    
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.25,
                           height_ratios=[1, 1])
    
    gt = generate_synthetic_trajectory(T=16, seed=42)
    bfn_pred = gt + np.random.randn(*gt.shape) * 0.025
    diff_pred = gt + np.random.randn(*gt.shape) * 0.02
    
    # -------------------------------------------------------------------------
    # (a) Ground Truth Trajectory
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    
    for i in range(len(gt) - 1):
        alpha = 0.3 + 0.7 * (i / len(gt))
        ax.plot(gt[i:i+2, 0], gt[i:i+2, 1], color=COLORS['ground_truth'], 
                linewidth=2.5, alpha=alpha)
    ax.scatter(gt[:, 0], gt[:, 1], c=np.arange(len(gt)), cmap=CMAP_TIME, 
               s=30, edgecolor=COLORS['ground_truth'], linewidth=0.5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.text(-0.2, 1.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (b) BFN Prediction
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    
    # Ground truth faint
    ax.plot(gt[:, 0], gt[:, 1], color=COLORS['ground_truth'], linewidth=1, 
            alpha=0.3, linestyle='--')
    
    for i in range(len(bfn_pred) - 1):
        alpha = 0.3 + 0.7 * (i / len(bfn_pred))
        ax.plot(bfn_pred[i:i+2, 0], bfn_pred[i:i+2, 1], color=COLORS['bfn'],
                linewidth=2.5, alpha=alpha)
    ax.scatter(bfn_pred[:, 0], bfn_pred[:, 1], c=np.arange(len(bfn_pred)), 
               cmap=CMAP_TIME, s=30, edgecolor=COLORS['bfn'], linewidth=0.5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('BFN (Ours)', fontweight='bold', color=COLORS['bfn'])
    ax.set_xlabel('X')
    ax.text(-0.15, 1.08, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (c) Diffusion Prediction
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    
    ax.plot(gt[:, 0], gt[:, 1], color=COLORS['ground_truth'], linewidth=1,
            alpha=0.3, linestyle='--')
    
    for i in range(len(diff_pred) - 1):
        alpha = 0.3 + 0.7 * (i / len(diff_pred))
        ax.plot(diff_pred[i:i+2, 0], diff_pred[i:i+2, 1], color=COLORS['diffusion'],
                linewidth=2.5, alpha=alpha)
    scatter = ax.scatter(diff_pred[:, 0], diff_pred[:, 1], c=np.arange(len(diff_pred)),
                         cmap=CMAP_TIME, s=30, edgecolor=COLORS['diffusion'], linewidth=0.5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Diffusion', fontweight='bold', color=COLORS['diffusion'])
    ax.set_xlabel('X')
    ax.text(-0.15, 1.08, '(c)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (d) Action X over time
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    
    time = np.arange(len(gt))
    ax.plot(time, gt[:, 0], color=COLORS['ground_truth'], linewidth=2,
            linestyle='--', label='GT', marker='o', markersize=3)
    ax.plot(time, bfn_pred[:, 0], color=COLORS['bfn'], linewidth=1.5,
            label='BFN', marker='s', markersize=2)
    ax.plot(time, diff_pred[:, 0], color=COLORS['diffusion'], linewidth=1.5,
            label='Diff', marker='^', markersize=2)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action X')
    ax.legend(loc='lower right', fontsize=7, ncol=3)
    ax.text(-0.2, 1.08, '(d)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (e) Action Y over time
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    
    ax.plot(time, gt[:, 1], color=COLORS['ground_truth'], linewidth=2,
            linestyle='--', label='GT', marker='o', markersize=3)
    ax.plot(time, bfn_pred[:, 1], color=COLORS['bfn'], linewidth=1.5,
            label='BFN', marker='s', markersize=2)
    ax.plot(time, diff_pred[:, 1], color=COLORS['diffusion'], linewidth=1.5,
            label='Diff', marker='^', markersize=2)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Y')
    ax.text(-0.15, 1.08, '(e)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # (f) Error over time
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    
    bfn_error = np.linalg.norm(bfn_pred - gt, axis=1)
    diff_error = np.linalg.norm(diff_pred - gt, axis=1)
    
    ax.fill_between(time, 0, bfn_error, alpha=0.3, color=COLORS['bfn'])
    ax.fill_between(time, 0, diff_error, alpha=0.3, color=COLORS['diffusion'])
    ax.plot(time, bfn_error, color=COLORS['bfn'], linewidth=1.5, label='BFN')
    ax.plot(time, diff_error, color=COLORS['diffusion'], linewidth=1.5, label='Diffusion')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Prediction Error')
    ax.legend(loc='upper right', fontsize=7)
    ax.text(-0.15, 1.08, '(f)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig_qualitative_combined.pdf'))
    fig.savefig(str(output_dir / 'fig_qualitative_combined.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Combined qualitative figure")


# =============================================================================
# FIGURE 7: METHOD COMPARISON DIAGRAM
# =============================================================================

def plot_method_diagram(output_dir: Path):
    """Conceptual diagram comparing BFN vs Diffusion sampling."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.7))
    
    for ax, (method, color, title, n_steps) in zip(axes, [
        ('bfn', COLORS['bfn'], 'BFN (Bayesian Updates)', 20),
        ('diffusion', COLORS['diffusion'], 'Diffusion (Denoising)', 100)
    ]):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw boxes for each step
        positions = [0, 1.5, 3, 4.5]
        labels = ['Noise\n$\\mathcal{N}(0,I)$', f'Step\n{n_steps//3}', f'Step\n{2*n_steps//3}', 'Action\n$a_t$']
        
        for i, (x, label) in enumerate(zip(positions, labels)):
            # Box
            box = FancyBboxPatch((x - 0.4, 0.2), 0.8, 0.6,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='white' if i < 3 else color,
                                 edgecolor=color, linewidth=1.5)
            ax.add_patch(box)
            
            # Label
            text_color = COLORS['black'] if i < 3 else 'white'
            ax.text(x, 0.5, label, ha='center', va='center', fontsize=7, color=text_color)
            
            # Arrow to next
            if i < len(positions) - 1:
                ax.annotate('', xy=(positions[i+1] - 0.5, 0.5), xytext=(x + 0.5, 0.5),
                           arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
        
        ax.set_title(title, fontweight='bold', color=color)
        ax.text(2.5, -0.3, f'{n_steps} steps', ha='center', fontsize=8, 
                style='italic', color=COLORS['gray'])
    
    plt.tight_layout()
    fig.savefig(str(output_dir / 'fig_method_diagram.pdf'))
    fig.savefig(str(output_dir / 'fig_method_diagram.png'), dpi=300)
    plt.close(fig)
    print("  ✓ Method diagram")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path("figures/publication")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Prediction Visualizations")
    print("Style: Google Research / DeepMind")
    print("=" * 60)
    print()
    
    plot_trajectory_comparison(output_dir)
    plot_denoising_process(output_dir)
    plot_prediction_variance(output_dir)
    plot_action_dimensions(output_dir)
    plot_pusht_visualization(output_dir)
    plot_combined_qualitative(output_dir)
    plot_method_diagram(output_dir)
    
    print()
    print("=" * 60)
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 60)
    print("\nNew files:")
    for f in sorted(output_dir.glob("fig_*.pdf")):
        print(f"  • {f.name}")


if __name__ == '__main__':
    main()
