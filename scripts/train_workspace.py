"""
Unified training script for all policy workspaces.

Usage:
    # Train diffusion policy
    python scripts/train_workspace.py --config-name=train_diffusion_unet_hybrid
    
    # Train BFN policy
    python scripts/train_workspace.py --config-name=train_bfn
    
    # Train Conditional BFN policy
    python scripts/train_workspace.py --config-name=train_conditional_bfn
    
    # Train Guided BFN policy
    python scripts/train_workspace.py --config-name=train_guided_bfn
    
    # Override parameters
    python scripts/train_workspace.py --config-name=train_bfn training.num_epochs=50 dataloader.batch_size=128
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import OmegaConf, DictConfig

# Register custom resolver for eval expressions (used in diffusion_policy configs)
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path='../config',
    config_name=None,
)
def main(cfg: DictConfig):
    """
    Main training function. Instantiates the workspace based on config
    and runs training.
    
    The workspace class is determined by the `_target_` field in the config.
    """
    # Resolve all interpolations
    OmegaConf.resolve(cfg)
    
    # Print config for debugging
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Get workspace class from config target
    workspace_cls = hydra.utils.get_class(cfg._target_)
    
    # Create and initialize workspace
    workspace = workspace_cls(cfg)
    
    # Run training
    workspace.run()


if __name__ == "__main__":
    main()
