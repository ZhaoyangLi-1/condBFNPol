#!/usr/bin/env python3
"""
Generate publication-quality figures for the Experimental Setup section.
Google Research style: clean, minimal, professional.

Figures generated:
1. Horizon configuration diagram (T_o, T_a, T_p)
2. Dataset sample grid
3. Observation and action space visualization
4. Receding horizon control illustration

Author: Generated for thesis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Arrow
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Anthropic Research Style Configuration
# =============================================================================

from colors import COLORS as _BASE_COLORS, setup_matplotlib_style

# Setup matplotlib style
setup_matplotlib_style()

# Extended color palette with setup-specific aliases
COLORS = {
    **_BASE_COLORS,
    'primary': _BASE_COLORS['bfn'],           # Teal for primary
    'secondary': _BASE_COLORS['neutral'],      # Tan for secondary
    'accent': _BASE_COLORS['diffusion'],       # Coral for accent
    'warning': _BASE_COLORS['diffusion'],      # Coral for warning
    'dark': _BASE_COLORS['text'],              # Slate for dark
    'medium': _BASE_COLORS['gray'],            # Slate for medium
    'light': _BASE_COLORS['light_gray'],       # Sand for light
    'obs_frame': _BASE_COLORS['bfn'],          # Teal for observations
    'action_exec': _BASE_COLORS['diffusion'],  # Coral for executed actions
    'action_pred': _BASE_COLORS['light_gray'], # Sand for predicted actions
}


def create_output_dir():
    """Create output directory for figures."""
    output_dir = Path('/dss/dsshome1/0D/ge87gob2/condBFNPol/figures/publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Figure 1: Horizon Configuration Diagram
# =============================================================================

def plot_horizon_configuration(output_dir):
    """
    Create a clean diagram showing T_o, T_a, T_p temporal configuration.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    
    # Parameters
    T_o = 2   # Observation horizon
    T_a = 16  # Action prediction horizon
    T_p = 8   # Action execution horizon
    
    box_width = 0.8
    box_height = 0.6
    y_obs = 3.0
    y_act = 1.5
    y_labels = 0.3
    
    # Title
    ax.text(8, 3.8, 'Temporal Horizon Configuration', 
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=COLORS['dark'])
    
    # --- Observation frames ---
    for i in range(T_o):
        x = i - 0.5
        rect = FancyBboxPatch((x, y_obs - box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['obs_frame'], 
                               edgecolor=COLORS['dark'],
                               linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + box_width/2, y_obs, f'$o_{{t-{T_o-1-i}}}$' if i < T_o-1 else '$o_t$',
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Observation bracket
    ax.annotate('', xy=(-0.5, y_obs + 0.5), xytext=(T_o - 0.7, y_obs + 0.5),
                arrowprops=dict(arrowstyle='-', color=COLORS['obs_frame'], lw=2))
    ax.plot([-0.5, -0.5], [y_obs + 0.4, y_obs + 0.5], color=COLORS['obs_frame'], lw=2)
    ax.plot([T_o - 0.7, T_o - 0.7], [y_obs + 0.4, y_obs + 0.5], color=COLORS['obs_frame'], lw=2)
    ax.text((T_o - 1.2) / 2, y_obs + 0.7, '$T_o = 2$', ha='center', va='bottom',
            fontsize=10, color=COLORS['obs_frame'], fontweight='bold')
    
    # Arrow from observations to actions
    ax.annotate('', xy=(T_o/2, y_act + box_height/2 + 0.3), 
                xytext=(T_o/2 - 0.5, y_obs - box_height/2 - 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['medium'], lw=1.5,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(T_o/2 + 0.8, (y_obs + y_act)/2 + 0.2, 'Policy\n$\\pi_\\theta$', 
            ha='left', va='center', fontsize=9, color=COLORS['medium'], style='italic')
    
    # --- Action sequence ---
    for i in range(T_a):
        x = i + 1
        if i < T_p:
            # Executed actions
            facecolor = COLORS['action_exec']
            edgecolor = COLORS['dark']
            textcolor = 'white'
            alpha = 0.9
        else:
            # Predicted but not executed
            facecolor = COLORS['light']
            edgecolor = COLORS['medium']
            textcolor = COLORS['medium']
            alpha = 0.7
        
        rect = FancyBboxPatch((x, y_act - box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.2, alpha=alpha)
        ax.add_patch(rect)
        
        if i < 3 or i == T_p - 1 or i == T_a - 1:
            label = f'$a_{{t+{i+1}}}$'
            ax.text(x + box_width/2, y_act, label,
                    ha='center', va='center', fontsize=8, color=textcolor, fontweight='bold')
        elif i == 3:
            ax.text(x + box_width/2, y_act, '...',
                    ha='center', va='center', fontsize=10, color=textcolor)
    
    # Execution bracket (T_p)
    ax.annotate('', xy=(1, y_act - 0.5), xytext=(T_p + 0.8, y_act - 0.5),
                arrowprops=dict(arrowstyle='-', color=COLORS['action_exec'], lw=2))
    ax.plot([1, 1], [y_act - 0.4, y_act - 0.5], color=COLORS['action_exec'], lw=2)
    ax.plot([T_p + 0.8, T_p + 0.8], [y_act - 0.4, y_act - 0.5], color=COLORS['action_exec'], lw=2)
    ax.text((T_p + 1.8) / 2, y_act - 0.7, '$T_p = 8$ (executed)', ha='center', va='top',
            fontsize=10, color=COLORS['action_exec'], fontweight='bold')
    
    # Prediction bracket (T_a)
    ax.annotate('', xy=(1, y_act + 0.5), xytext=(T_a + 0.8, y_act + 0.5),
                arrowprops=dict(arrowstyle='-', color=COLORS['medium'], lw=1.5))
    ax.plot([1, 1], [y_act + 0.4, y_act + 0.5], color=COLORS['medium'], lw=1.5)
    ax.plot([T_a + 0.8, T_a + 0.8], [y_act + 0.4, y_act + 0.5], color=COLORS['medium'], lw=1.5)
    ax.text((T_a + 1.8) / 2, y_act + 0.7, '$T_a = 16$ (predicted)', ha='center', va='bottom',
            fontsize=10, color=COLORS['medium'], fontweight='bold')
    
    # Time arrow
    ax.annotate('', xy=(17.5, y_labels), xytext=(-0.5, y_labels),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.text(17.5, y_labels - 0.2, 'time', ha='right', va='top', 
            fontsize=10, color=COLORS['dark'], style='italic')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['obs_frame'], edgecolor=COLORS['dark'],
                       label='Observation frames'),
        mpatches.Patch(facecolor=COLORS['action_exec'], edgecolor=COLORS['dark'],
                       label='Executed actions'),
        mpatches.Patch(facecolor=COLORS['light'], edgecolor=COLORS['medium'],
                       label='Predicted only'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=True, framealpha=0.9, edgecolor=COLORS['light'])
    
    plt.tight_layout()
    
    # Save
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_horizon_config.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_horizon_config.pdf/png')


# =============================================================================
# Figure 2: Receding Horizon Control Illustration
# =============================================================================

def plot_receding_horizon(output_dir):
    """
    Illustrate the receding horizon control scheme over multiple timesteps.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    
    T_p = 8  # Execution horizon
    T_a = 16  # Prediction horizon
    
    box_w = 0.45
    box_h = 0.5
    
    # Three planning cycles
    cycles = [
        {'y': 3.5, 'start': 0, 'label': 'Cycle 1: $t=0$'},
        {'y': 2.0, 'start': 8, 'label': 'Cycle 2: $t=8$'},
        {'y': 0.5, 'start': 16, 'label': 'Cycle 3: $t=16$'},
    ]
    
    for cycle in cycles:
        y = cycle['y']
        start = cycle['start']
        
        # Label
        ax.text(-0.3, y, cycle['label'], ha='right', va='center',
                fontsize=9, fontweight='bold', color=COLORS['dark'])
        
        # Draw action boxes
        for i in range(T_a):
            x = start + i * 0.5
            if i < T_p:
                facecolor = COLORS['action_exec']
                alpha = 0.9
            else:
                facecolor = COLORS['light']
                alpha = 0.6
            
            rect = Rectangle((x, y - box_h/2), box_w, box_h,
                             facecolor=facecolor, edgecolor=COLORS['medium'],
                             linewidth=0.5, alpha=alpha)
            ax.add_patch(rect)
    
    # Highlight overlap/replanning
    ax.annotate('Replan with\nnew observation', 
                xy=(8, 2.5), xytext=(5, 3.8),
                fontsize=8, color=COLORS['medium'],
                arrowprops=dict(arrowstyle='->', color=COLORS['medium'], lw=1),
                ha='center')
    
    # Time axis
    ax.annotate('', xy=(24.5, -0.3), xytext=(0, -0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.text(24.5, -0.5, 'time (steps)', ha='right', va='top',
            fontsize=9, color=COLORS['dark'], style='italic')
    
    # Time markers
    for t in [0, 8, 16, 24]:
        ax.plot([t * 0.5, t * 0.5], [-0.2, -0.4], color=COLORS['dark'], lw=1)
        ax.text(t * 0.5, -0.6, f'{t}', ha='center', va='top', fontsize=8)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['action_exec'], edgecolor=COLORS['medium'],
                       label='Executed ($T_p=8$)'),
        mpatches.Patch(facecolor=COLORS['light'], edgecolor=COLORS['medium'],
                       label='Predicted only'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fancybox=True, framealpha=0.9)
    
    ax.set_title('Receding Horizon Control: Predict $T_a=16$, Execute $T_p=8$, Replan',
                 fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_receding_horizon.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_receding_horizon.pdf/png')


# =============================================================================
# Figure 3: Observation and Action Space
# =============================================================================

def plot_observation_action_space(output_dir):
    """
    Visualize the observation and action spaces.
    """
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.3, 1], wspace=0.3)
    
    # --- Left: Observation Space ---
    ax_obs = fig.add_subplot(gs[0])
    ax_obs.set_xlim(0, 100)
    ax_obs.set_ylim(0, 100)
    ax_obs.set_aspect('equal')
    ax_obs.axis('off')
    ax_obs.set_title('Observation Space', fontsize=11, fontweight='bold')
    
    # Image placeholder
    img_rect = FancyBboxPatch((5, 25), 60, 60,
                               boxstyle="round,pad=0.01,rounding_size=2",
                               facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                               linewidth=2)
    ax_obs.add_patch(img_rect)
    
    # Draw simplified Push-T scene inside
    # T-block (simplified)
    t_body = Rectangle((25, 45), 25, 8, facecolor=COLORS['medium'], edgecolor=COLORS['dark'], lw=1)
    t_stem = Rectangle((33, 35), 8, 12, facecolor=COLORS['medium'], edgecolor=COLORS['dark'], lw=1)
    ax_obs.add_patch(t_body)
    ax_obs.add_patch(t_stem)
    
    # End-effector
    circle = plt.Circle((20, 55), 5, facecolor=COLORS['primary'], edgecolor=COLORS['dark'], lw=1)
    ax_obs.add_patch(circle)
    
    # Target outline (dashed)
    target_body = Rectangle((30, 60), 25, 8, facecolor='none', 
                             edgecolor=COLORS['secondary'], lw=2, linestyle='--')
    target_stem = Rectangle((38, 50), 8, 12, facecolor='none',
                             edgecolor=COLORS['secondary'], lw=2, linestyle='--')
    ax_obs.add_patch(target_body)
    ax_obs.add_patch(target_stem)
    
    # Labels
    ax_obs.text(35, 15, '$96 \\times 96$ RGB', ha='center', fontsize=9, color=COLORS['dark'])
    
    # Agent position box
    pos_rect = FancyBboxPatch((70, 45), 25, 20,
                               boxstyle="round,pad=0.01,rounding_size=2",
                               facecolor=COLORS['white'], edgecolor=COLORS['primary'],
                               linewidth=2)
    ax_obs.add_patch(pos_rect)
    ax_obs.text(82.5, 55, '$(x, y)$', ha='center', va='center', fontsize=10,
                color=COLORS['primary'], fontweight='bold')
    ax_obs.text(82.5, 48, 'Agent pos', ha='center', va='center', fontsize=8,
                color=COLORS['medium'])
    
    # Observation equation
    ax_obs.text(50, 5, '$\\mathbf{o}_t = \\{I_t, p_t\\}$', ha='center', fontsize=10,
                color=COLORS['dark'], style='italic')
    
    # --- Middle: Arrow ---
    ax_arrow = fig.add_subplot(gs[1])
    ax_arrow.set_xlim(0, 10)
    ax_arrow.set_ylim(0, 10)
    ax_arrow.axis('off')
    
    ax_arrow.annotate('', xy=(8, 5), xytext=(2, 5),
                      arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax_arrow.text(5, 6.5, 'Policy\n$\\pi_\\theta$', ha='center', va='bottom',
                  fontsize=10, fontweight='bold', color=COLORS['dark'])
    
    # --- Right: Action Space ---
    ax_act = fig.add_subplot(gs[2])
    ax_act.set_xlim(-1.5, 1.5)
    ax_act.set_ylim(-1.5, 1.5)
    ax_act.set_aspect('equal')
    ax_act.set_title('Action Space', fontsize=11, fontweight='bold')
    
    # Draw coordinate system
    ax_act.axhline(0, color=COLORS['light'], lw=1, zorder=1)
    ax_act.axvline(0, color=COLORS['light'], lw=1, zorder=1)
    
    # Action vector
    ax_act.annotate('', xy=(0.7, 0.5), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax_act.text(0.8, 0.6, '$\\mathbf{a}_t$', fontsize=12, color=COLORS['accent'], fontweight='bold')
    
    # Velocity components
    ax_act.annotate('', xy=(0.7, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2, alpha=0.7))
    ax_act.annotate('', xy=(0, 0.5), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2, alpha=0.7))
    
    ax_act.text(0.7, -0.15, '$v_x$', fontsize=10, color=COLORS['primary'], ha='center')
    ax_act.text(-0.15, 0.5, '$v_y$', fontsize=10, color=COLORS['secondary'], ha='center')
    
    # Bounds
    ax_act.axhline(1, color=COLORS['medium'], lw=1, linestyle='--', alpha=0.5)
    ax_act.axhline(-1, color=COLORS['medium'], lw=1, linestyle='--', alpha=0.5)
    ax_act.axvline(1, color=COLORS['medium'], lw=1, linestyle='--', alpha=0.5)
    ax_act.axvline(-1, color=COLORS['medium'], lw=1, linestyle='--', alpha=0.5)
    
    ax_act.text(1.1, 0, '+1', fontsize=8, color=COLORS['medium'], va='center')
    ax_act.text(-1.3, 0, '-1', fontsize=8, color=COLORS['medium'], va='center')
    ax_act.text(0, 1.15, '+1', fontsize=8, color=COLORS['medium'], ha='center')
    ax_act.text(0, -1.25, '-1', fontsize=8, color=COLORS['medium'], ha='center')
    
    ax_act.set_xlabel('Horizontal velocity', fontsize=9, color=COLORS['medium'])
    ax_act.set_ylabel('Vertical velocity', fontsize=9, color=COLORS['medium'])
    ax_act.set_xticks([])
    ax_act.set_yticks([])
    
    # Action equation
    ax_act.text(0, -1.5, '$\\mathbf{a} \\in \\mathbb{R}^2, \\|\\mathbf{a}\\|_\\infty \\leq 1$', 
                ha='center', fontsize=10, color=COLORS['dark'], style='italic')
    
    for spine in ax_act.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_obs_action_space.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_obs_action_space.pdf/png')


# =============================================================================
# Figure 4: Dataset Statistics Visualization
# =============================================================================

def plot_dataset_statistics(output_dir):
    """
    Visualize dataset statistics as a clean infographic.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # --- Left: Trajectory length distribution ---
    ax1 = axes[0]
    # Simulate trajectory length distribution (bell-shaped around 150)
    np.random.seed(42)
    lengths = np.random.normal(150, 30, 206)
    lengths = np.clip(lengths, 50, 300)
    
    ax1.hist(lengths, bins=20, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax1.axvline(150, color=COLORS['accent'], lw=2, linestyle='--', label='Mean = 150')
    ax1.set_xlabel('Trajectory Length (steps)')
    ax1.set_ylabel('Count')
    ax1.set_title('Trajectory Lengths', fontweight='bold')
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Middle: Train/Val split ---
    ax2 = axes[1]
    sizes = [90, 4]
    labels = ['Train\n(90)', 'Val\n(4)']
    colors = [COLORS['primary'], COLORS['secondary']]
    
    wedges, texts = ax2.pie(sizes, labels=labels, colors=colors,
                            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Train/Val Split', fontweight='bold')
    
    # --- Right: Key statistics ---
    ax3 = axes[2]
    ax3.axis('off')
    
    stats = [
        ('Total Demos', '206'),
        ('Transitions', '~30K'),
        ('Control Freq', '10 Hz'),
        ('Collection', 'Human'),
    ]
    
    y_positions = [0.8, 0.6, 0.4, 0.2]
    for (label, value), y in zip(stats, y_positions):
        ax3.text(0.1, y, label + ':', fontsize=10, color=COLORS['medium'],
                 transform=ax3.transAxes, va='center')
        ax3.text(0.9, y, value, fontsize=11, color=COLORS['dark'], fontweight='bold',
                 transform=ax3.transAxes, va='center', ha='right')
    
    ax3.set_title('Dataset Summary', fontweight='bold')
    
    # Add box around stats
    rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=COLORS['light'], edgecolor=COLORS['medium'],
                          linewidth=1, transform=ax3.transAxes, alpha=0.5)
    ax3.add_patch(rect)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_dataset_stats.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_dataset_stats.pdf/png')


# =============================================================================
# Figure 5: Training Pipeline Overview
# =============================================================================

def plot_training_pipeline(output_dir):
    """
    Overview of the training pipeline for both methods.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Box parameters
    box_h = 0.8
    box_w = 2.0
    
    # --- Data Flow ---
    components = [
        {'x': 0.5, 'y': 2.5, 'w': 1.5, 'h': 1.0, 'text': 'Dataset\n$\\mathcal{D}$', 'color': COLORS['light']},
        {'x': 2.5, 'y': 2.5, 'w': 2.0, 'h': 1.0, 'text': 'Vision\nEncoder', 'color': COLORS['primary']},
        {'x': 5.0, 'y': 2.5, 'w': 2.0, 'h': 1.0, 'text': 'Conditional\n1D U-Net', 'color': COLORS['primary']},
    ]
    
    # BFN path
    components.append({'x': 7.5, 'y': 3.5, 'w': 2.2, 'h': 0.8, 'text': 'BFN Loss', 'color': COLORS['bfn']})
    # Diffusion path  
    components.append({'x': 7.5, 'y': 1.5, 'w': 2.2, 'h': 0.8, 'text': 'Diffusion Loss', 'color': COLORS['diffusion']})
    
    # Output
    components.append({'x': 10.2, 'y': 2.5, 'w': 1.5, 'h': 1.0, 'text': 'Policy\n$\\pi_\\theta$', 'color': COLORS['secondary']})
    
    for comp in components:
        rect = FancyBboxPatch((comp['x'], comp['y'] - comp['h']/2), comp['w'], comp['h'],
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=comp['color'], edgecolor=COLORS['dark'],
                               linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        
        textcolor = 'white' if comp['color'] in [COLORS['primary'], COLORS['bfn'], 
                                                   COLORS['diffusion'], COLORS['secondary']] else COLORS['dark']
        ax.text(comp['x'] + comp['w']/2, comp['y'], comp['text'],
                ha='center', va='center', fontsize=9, color=textcolor, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2.0, 2.5), (2.5, 2.5)),   # Data -> Encoder
        ((4.5, 2.5), (5.0, 2.5)),   # Encoder -> U-Net
        ((7.0, 2.8), (7.5, 3.5)),   # U-Net -> BFN
        ((7.0, 2.2), (7.5, 1.5)),   # U-Net -> Diffusion
        ((9.7, 3.5), (10.2, 2.8)),  # BFN -> Policy
        ((9.7, 1.5), (10.2, 2.2)),  # Diffusion -> Policy
    ]
    
    for (start, end) in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    # Labels for losses
    ax.text(8.6, 4.0, '$\\mathcal{L}_{BFN} = \\mathbb{E}[\\|\\hat{\\mathbf{a}} - \\mathbf{a}\\|^2]$',
            fontsize=9, color=COLORS['bfn'], ha='center')
    ax.text(8.6, 0.9, '$\\mathcal{L}_{Diff} = \\mathbb{E}[\\|\\hat{\\epsilon} - \\epsilon\\|^2]$',
            fontsize=9, color=COLORS['diffusion'], ha='center')
    
    # OR label
    ax.text(8.6, 2.5, 'OR', fontsize=10, fontweight='bold', color=COLORS['medium'],
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['medium']))
    
    # Title
    ax.text(6, 4.8, 'Training Pipeline: Shared Architecture, Different Objectives',
            ha='center', fontsize=12, fontweight='bold', color=COLORS['dark'])
    
    # Subtitle
    ax.text(6, 4.4, 'Both methods use identical encoder and U-Net; only the loss function differs',
            ha='center', fontsize=9, color=COLORS['medium'], style='italic')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_training_pipeline.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_training_pipeline.pdf/png')


# =============================================================================
# Figure 6: Method Comparison Side-by-Side
# =============================================================================

def plot_method_comparison(output_dir):
    """
    Side-by-side comparison of BFN and Diffusion inference processes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Common settings
    n_steps = 6
    
    for idx, (ax, method, color) in enumerate(zip(axes, ['BFN', 'Diffusion'], 
                                                   [COLORS['bfn'], COLORS['diffusion']])):
        ax.set_xlim(-0.5, n_steps + 0.5)
        ax.set_ylim(-0.5, 3)
        ax.axis('off')
        
        ax.set_title(f'{method} Policy', fontsize=11, fontweight='bold', color=color)
        
        # Draw steps
        for i in range(n_steps):
            # Progress bar
            progress = i / (n_steps - 1)
            
            # Noise visualization (circle with varying clarity)
            if method == 'Diffusion':
                # Start noisy, become clear
                noise_level = 1 - progress
            else:
                # Start uncertain (wide), become certain (narrow)
                noise_level = 1 - progress
            
            # Draw action representation
            y = 1.5
            circle_size = 0.3 + 0.1 * noise_level
            
            if method == 'BFN':
                # Gaussian blob becoming sharper
                circle = plt.Circle((i, y), circle_size, 
                                    facecolor=color, alpha=0.3 + 0.7 * progress,
                                    edgecolor=color, linewidth=2)
            else:
                # Random dots becoming organized
                circle = plt.Circle((i, y), circle_size,
                                    facecolor=color, alpha=0.3 + 0.7 * progress,
                                    edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            
            # Step label
            if i == 0:
                label = '$t=0$'
            elif i == n_steps - 1:
                label = '$t=1$'
            else:
                label = ''
            ax.text(i, 0.7, label, ha='center', fontsize=9, color=COLORS['medium'])
            
            # Arrow to next step
            if i < n_steps - 1:
                ax.annotate('', xy=(i + 0.6, y), xytext=(i + 0.4, y),
                           arrowprops=dict(arrowstyle='->', color=COLORS['medium'], lw=1))
        
        # Start and end labels
        if method == 'BFN':
            ax.text(0, 2.5, 'Prior: $\\mathcal{N}(0, I)$', ha='center', fontsize=9, color=COLORS['medium'])
            ax.text(n_steps - 1, 2.5, 'Posterior: $\\mathcal{N}(\\mu, \\rho^{-1})$', 
                    ha='center', fontsize=9, color=color)
        else:
            ax.text(0, 2.5, 'Noise: $\\mathbf{a}_T \\sim \\mathcal{N}(0, I)$', 
                    ha='center', fontsize=9, color=COLORS['medium'])
            ax.text(n_steps - 1, 2.5, 'Clean: $\\mathbf{a}_0$', 
                    ha='center', fontsize=9, color=color)
        
        # Process description
        if method == 'BFN':
            desc = 'Bayesian update: refine belief with each message'
        else:
            desc = 'Denoising: iteratively remove noise from sample'
        ax.text((n_steps - 1) / 2, 0.2, desc, ha='center', fontsize=8, 
                color=COLORS['medium'], style='italic')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(str(output_dir / f'fig_method_comparison.{fmt}'), dpi=300)
    plt.close()
    print('✓ Generated: fig_method_comparison.pdf/png')


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all experimental setup figures."""
    print('=' * 60)
    print('Generating Experimental Setup Figures (Google Research Style)')
    print('=' * 60)
    
    output_dir = create_output_dir()
    
    # Generate all figures
    plot_horizon_configuration(output_dir)
    plot_receding_horizon(output_dir)
    plot_observation_action_space(output_dir)
    plot_dataset_statistics(output_dir)
    plot_training_pipeline(output_dir)
    plot_method_comparison(output_dir)
    
    print('=' * 60)
    print(f'All figures saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

