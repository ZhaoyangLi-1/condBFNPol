#!/usr/bin/env python3
"""Generate architecture diagram for BFN Policy paper.

Creates a publication-quality architecture diagram comparing BFN and Diffusion policies.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.8,
})

# Colors (Google-style)
COLORS = {
    'input': '#E8F5E9',        # Light green
    'encoder': '#E3F2FD',      # Light blue
    'unet': '#FFF3E0',         # Light orange
    'bfn': '#4285F4',          # Google blue
    'diffusion': '#EA4335',    # Google red
    'output': '#F3E5F5',       # Light purple
    'arrow': '#424242',        # Dark gray
    'text': '#212121',         # Near black
    'box_edge': '#757575',     # Gray
}


def draw_rounded_box(ax, xy, width, height, label, color, fontsize=8, 
                     text_color='black', edge_color=None, alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    if edge_color is None:
        edge_color = COLORS['box_edge']
    
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=1.2,
        alpha=alpha,
        transform=ax.transData,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='medium', zorder=3)
    
    return box


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=1.5,
        connectionstyle=connectionstyle,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow


def create_architecture_diagram():
    """Create the main architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 6.2, 'BFN Policy Architecture', fontsize=14, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])
    
    # =========================================================================
    # Input Section
    # =========================================================================
    # Image observation
    draw_rounded_box(ax, (0, 4), 1.5, 1.2, 'Image\nObservation\n$o_t^{img}$', 
                     COLORS['input'], fontsize=8)
    
    # Agent state
    draw_rounded_box(ax, (0, 2.5), 1.5, 1.0, 'Agent State\n$o_t^{state}$', 
                     COLORS['input'], fontsize=8)
    
    # =========================================================================
    # Encoder Section  
    # =========================================================================
    # Vision encoder
    draw_rounded_box(ax, (2.2, 4), 1.8, 1.2, 'Vision Encoder\n(ResNet-18)', 
                     COLORS['encoder'], fontsize=8)
    
    # State MLP
    draw_rounded_box(ax, (2.2, 2.5), 1.8, 1.0, 'State MLP', 
                     COLORS['encoder'], fontsize=8)
    
    # Feature concatenation
    draw_rounded_box(ax, (4.5, 3.2), 1.2, 0.8, 'Concat', 
                     '#ECEFF1', fontsize=8)
    
    # =========================================================================
    # Core Model
    # =========================================================================
    # Conditional U-Net
    draw_rounded_box(ax, (6.2, 2.8), 2.0, 1.6, 'Conditional\n1D U-Net\n$\\epsilon_\\theta(x, t, c)$', 
                     COLORS['unet'], fontsize=9)
    
    # =========================================================================
    # Generative Process (side by side comparison)
    # =========================================================================
    # BFN Process
    bfn_y = 0.8
    draw_rounded_box(ax, (5.0, bfn_y), 2.8, 1.2, 
                     'BFN Bayesian Update\n$\\mu \\leftarrow \\mu + \\alpha \\cdot (\\hat{x} - \\mu)$\n5-20 steps', 
                     COLORS['bfn'], fontsize=8, text_color='white', alpha=0.95)
    
    # Add "OR" text
    ax.text(7.2, 2.3, 'Sampling', fontsize=8, ha='center', va='center',
            style='italic', color=COLORS['text'])
    
    # =========================================================================
    # Output Section
    # =========================================================================
    draw_rounded_box(ax, (8.5, 2.8), 1.8, 1.6, 'Action\nSequence\n$a_{t:t+H}$', 
                     COLORS['output'], fontsize=9)
    
    # =========================================================================
    # Arrows
    # =========================================================================
    # Input to encoders
    draw_arrow(ax, (1.5, 4.6), (2.2, 4.6))
    draw_arrow(ax, (1.5, 3.0), (2.2, 3.0))
    
    # Encoders to concat
    draw_arrow(ax, (4.0, 4.6), (4.5, 3.8), connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (4.0, 3.0), (4.5, 3.4), connectionstyle='arc3,rad=0.2')
    
    # Concat to U-Net
    draw_arrow(ax, (5.7, 3.6), (6.2, 3.6))
    
    # U-Net to BFN
    draw_arrow(ax, (7.2, 2.8), (7.2, 2.0), color=COLORS['bfn'])
    
    # BFN loop arrow
    ax.annotate('', xy=(5.0, 1.4), xytext=(5.0, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['bfn'], 
                               connectionstyle='arc3,rad=-1.5', lw=1.5))
    ax.text(4.3, 1.1, 'iterate', fontsize=7, color=COLORS['bfn'], style='italic')
    
    # BFN to output
    draw_arrow(ax, (7.8, 1.4), (8.8, 2.8), color=COLORS['bfn'], 
               connectionstyle='arc3,rad=-0.3')
    
    # =========================================================================
    # Labels and annotations
    # =========================================================================
    # Section labels
    ax.text(0.75, 5.5, 'Observation', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(3.1, 5.5, 'Encoding', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(7.2, 5.0, 'Denoising Network', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(9.4, 5.0, 'Output', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['text'])
    
    # Add time conditioning arrow to U-Net
    ax.annotate('$t$', xy=(7.2, 4.4), xytext=(7.2, 5.3),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    ax.text(7.5, 5.1, '(timestep)', fontsize=7, color=COLORS['text'])
    
    plt.tight_layout()
    return fig


def create_comparison_diagram():
    """Create a side-by-side comparison of BFN vs Diffusion inference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax in axes:
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 4)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # BFN Panel (Left)
    # =========================================================================
    ax = axes[0]
    ax.set_title('BFN Policy Inference', fontsize=12, fontweight='bold', 
                 color=COLORS['bfn'])
    
    # Steps
    steps_bfn = ['$\\mu_0 = 0$', '$\\mu_1$', '$\\mu_2$', '...', '$\\mu_T = \\hat{a}$']
    for i, step in enumerate(steps_bfn):
        x = 0.5 + i * 1.1
        color = COLORS['bfn'] if i == len(steps_bfn)-1 else '#BBDEFB'
        text_color = 'white' if i == len(steps_bfn)-1 else 'black'
        draw_rounded_box(ax, (x-0.4, 1.5), 0.8, 0.8, step, color, 
                        fontsize=9, text_color=text_color)
        if i < len(steps_bfn) - 1:
            draw_arrow(ax, (x+0.4, 1.9), (x+0.7, 1.9))
    
    # Description
    ax.text(2.75, 0.5, 'Bayesian posterior update\n$\\mu \\leftarrow \\mu + \\alpha(\\hat{x}_\\theta - \\mu)$',
            ha='center', fontsize=10, style='italic', color=COLORS['text'])
    ax.text(2.75, 3.2, '5-20 steps typically sufficient',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['bfn'])
    
    # =========================================================================
    # Diffusion Panel (Right)
    # =========================================================================
    ax = axes[1]
    ax.set_title('Diffusion Policy Inference', fontsize=12, fontweight='bold',
                 color=COLORS['diffusion'])
    
    # Steps
    steps_diff = ['$x_T \\sim \\mathcal{N}$', '$x_{T-1}$', '$x_{T-2}$', '...', '$x_0 = \\hat{a}$']
    for i, step in enumerate(steps_diff):
        x = 0.5 + i * 1.1
        color = COLORS['diffusion'] if i == len(steps_diff)-1 else '#FFCDD2'
        text_color = 'white' if i == len(steps_diff)-1 else 'black'
        draw_rounded_box(ax, (x-0.4, 1.5), 0.8, 0.8, step, color,
                        fontsize=9, text_color=text_color)
        if i < len(steps_diff) - 1:
            draw_arrow(ax, (x+0.4, 1.9), (x+0.7, 1.9))
    
    # Description
    ax.text(2.75, 0.5, 'DDPM denoising\n$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\epsilon_\\theta)$',
            ha='center', fontsize=10, style='italic', color=COLORS['text'])
    ax.text(2.75, 3.2, '75-100 steps typically required',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['diffusion'])
    
    plt.tight_layout()
    return fig


def create_simple_block_diagram():
    """Create a simple block diagram suitable for paper."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Blocks
    blocks = [
        (0.5, 1, 1.5, 1, 'Observation\n$o_t$', COLORS['input']),
        (2.5, 1, 1.5, 1, 'Encoder\n$\\phi(o_t)$', COLORS['encoder']),
        (4.5, 1, 1.8, 1, 'Cond. U-Net\n$\\epsilon_\\theta$', COLORS['unet']),
        (7.0, 1, 1.8, 1, 'BFN Sample\n(5-20 steps)', COLORS['bfn']),
        (9.2, 1, 0.6, 1, '$a_t$', COLORS['output']),
    ]
    
    for x, y, w, h, label, color in blocks:
        text_color = 'white' if color == COLORS['bfn'] else 'black'
        draw_rounded_box(ax, (x, y), w, h, label, color, 
                        fontsize=9, text_color=text_color)
    
    # Arrows
    arrows = [(2.0, 1.5), (4.0, 1.5), (6.3, 1.5), (8.8, 1.5)]
    for i in range(len(arrows) - 1):
        draw_arrow(ax, arrows[i], (arrows[i][0] + 0.5, arrows[i][1]))
    draw_arrow(ax, (8.8, 1.5), (9.2, 1.5))
    
    # Title
    ax.text(5, 2.7, 'BFN Policy: Observation → Action Pipeline', 
            fontsize=11, fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig


def main():
    import os
    output_dir = 'figures/publication'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Architecture Diagrams")
    print("=" * 60)
    
    # Main architecture
    print("\n1. Creating main architecture diagram...")
    fig = create_architecture_diagram()
    fig.savefig(f'{output_dir}/fig_architecture.pdf', format='pdf', 
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_dir}/fig_architecture.png', format='png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"   ✓ Saved: {output_dir}/fig_architecture.pdf")
    
    # Comparison diagram
    print("\n2. Creating BFN vs Diffusion comparison...")
    fig = create_comparison_diagram()
    fig.savefig(f'{output_dir}/fig_inference_comparison.pdf', format='pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_dir}/fig_inference_comparison.png', format='png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"   ✓ Saved: {output_dir}/fig_inference_comparison.pdf")
    
    # Simple block diagram
    print("\n3. Creating simple block diagram...")
    fig = create_simple_block_diagram()
    fig.savefig(f'{output_dir}/fig_pipeline.pdf', format='pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_dir}/fig_pipeline.png', format='png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"   ✓ Saved: {output_dir}/fig_pipeline.pdf")
    
    print("\n" + "=" * 60)
    print("All architecture diagrams generated!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  📊 {output_dir}/fig_architecture.pdf     - Full architecture")
    print(f"  📊 {output_dir}/fig_inference_comparison.pdf - BFN vs Diffusion")
    print(f"  📊 {output_dir}/fig_pipeline.pdf         - Simple pipeline")


if __name__ == '__main__':
    main()

