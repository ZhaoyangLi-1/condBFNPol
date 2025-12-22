#!/usr/bin/env python
"""Train Diffusion U-Net Hybrid Image Policy using Workspace.

This script provides a simple entry point for training the diffusion policy
using the workspace architecture. It can be used with or without Hydra.

Usage:
    # With Hydra (recommended):
    python scripts/train_diffusion_workspace.py
    
    # With custom config:
    python scripts/train_diffusion_workspace.py --config-path /path/to/config.yaml
    
    # Override specific values:
    python scripts/train_diffusion_workspace.py training.num_epochs=200 dataloader.batch_size=128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Train Diffusion U-Net Hybrid Image Policy"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file. If not provided, uses Hydra defaults."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs."
    )
    parser.add_argument(
        "--no-hydra",
        action="store_true",
        help="Run without Hydra (requires --config-path)."
    )
    
    args, remaining = parser.parse_known_args()
    
    if args.no_hydra:
        # Run without Hydra - requires config file
        if args.config_path is None:
            parser.error("--config-path is required when using --no-hydra")
        
        import yaml
        from omegaconf import OmegaConf
        from workspaces import TrainDiffusionUnetHybridWorkspace
        
        with open(args.config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        cfg = OmegaConf.create(cfg_dict)
        
        # Apply any overrides from remaining args
        for override in remaining:
            if '=' in override:
                key, value = override.split('=', 1)
                OmegaConf.update(cfg, key, yaml.safe_load(value))
        
        workspace = TrainDiffusionUnetHybridWorkspace(
            cfg,
            output_dir=args.output_dir
        )
        workspace.run()
    
    else:
        # Run with Hydra
        try:
            import hydra
            from omegaconf import DictConfig
        except ImportError:
            print("Hydra not installed. Use --no-hydra with --config-path.")
            sys.exit(1)
        
        # Build Hydra overrides
        overrides = remaining
        if args.config_path is not None:
            # Use custom config
            config_dir = Path(args.config_path).parent
            config_name = Path(args.config_path).stem
        else:
            config_dir = ROOT_DIR / "config"
            config_name = "train_diffusion_unet_hybrid"
        
        @hydra.main(
            version_base=None,
            config_path=str(config_dir),
            config_name=config_name
        )
        def run_hydra(cfg: DictConfig):
            from workspaces import TrainDiffusionUnetHybridWorkspace
            workspace = TrainDiffusionUnetHybridWorkspace(cfg)
            workspace.run()
        
        # Pass overrides to Hydra
        sys.argv = [sys.argv[0]] + overrides
        run_hydra()


if __name__ == "__main__":
    main()
