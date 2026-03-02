"""Training Workspace for Streaming Flow Policy.

This workspace handles the setup and inference for Streaming Flow Policy
on image-based tasks like PushT. Since Streaming Flow Policy is demonstration-based
rather than learning-based, this workspace focuses on trajectory loading and inference.

Features:
- Demonstration trajectory loading
- Policy inference setup
- Evaluation rollouts
- Performance logging
"""

from __future__ import annotations

import copy
import os
import pathlib
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import hydra
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

from workspaces.base_workspace import BaseWorkspace, copy_to_cpu

# Import for data handling
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger

__all__ = ["TrainStreamingFlowWorkspace"]


# Register OmegaConf resolver for eval
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainStreamingFlowWorkspace(BaseWorkspace):
    """Workspace for Streaming Flow Policy.
    
    This workspace manages the setup and inference pipeline including:
    - Model instantiation from config
    - Dataset loading for demonstration trajectories
    - Policy setup with demonstrations
    - Environment rollout evaluation
    - Logging and result tracking
    """
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """Initialize the workspace.
        
        Args:
            cfg: Hydra configuration object.
            output_dir: Output directory for logs and results.
        """
        super().__init__(cfg, output_dir=output_dir)
        
        # Set random seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Configure device
        device = torch.device(cfg.training.device)
        self.device = device
        
        # Initialize policy
        policy = hydra.utils.instantiate(cfg.policy)
        
        # Move policy to device
        policy = policy.to(device)
        self.policy = policy
        
        # Initialize dataset and dataloader for trajectory loading
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        self.dataset = dataset
        self.dataloader = dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader
        
        # Setup normalizers
        normalizer = LinearNormalizer()
        self.normalizer = normalizer
        
        # Load demonstrations and initialize policy trajectories
        self._load_demonstrations()
        
        # Initialize environment runner for evaluation
        if 'env_runner' in cfg.task and cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner)
            assert env_runner.legacy_test == cfg.task.dataset.legacy_test
            self.env_runner = env_runner
        else:
            self.env_runner = None
        
        # Initialize logging
        self.global_step = 0
        self.epoch = 0
        
        # WandB logging
        if cfg.logging.mode != 'disabled':
            wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            self.wandb_run_dir = pathlib.Path(wandb.run.dir)
        else:
            self.wandb_run_dir = None
        
        # JSON logger
        self.json_logger = JsonLogger(self.output_dir.joinpath("log.json"))
        
        print("============= Configuration =============")
        print(OmegaConf.to_yaml(cfg))
        print("=========================================")
    
    def _load_demonstrations(self):
        """Load demonstration trajectories for the streaming flow policy."""
        # This is a placeholder implementation
        # In a real scenario, you would:
        # 1. Extract trajectories from the dataset
        # 2. Convert them to the format expected by streaming flow policy
        # 3. Initialize the policy with these trajectories
        
        print("Loading demonstration trajectories...")
        
        # For now, we'll use the policy's default initialization
        # In a full implementation, you would extract trajectories from self.dataset
        # and call self.policy.update_trajectories(trajectories, priors)
        
        print(f"Initialized with {len(getattr(self.policy, 'trajectories', []))} demonstration trajectories")
    
    def run(self):
        """Main execution method.
        
        Since Streaming Flow Policy doesn't require training in the traditional sense,
        this method focuses on evaluation and logging.
        """
        cfg = copy.deepcopy(self.cfg)
        
        # Run initial evaluation
        if self.env_runner is not None:
            print("Running initial evaluation...")
            runner_log = self._run_validation()
            
            # Log results
            if self.wandb_run_dir is not None:
                wandb.log(runner_log, step=self.global_step)
            self.json_logger.log(runner_log)
            
        # Save final model state
        self._save_checkpoint()
        
        print("Streaming Flow Policy setup and evaluation completed!")
    
    def _run_validation(self) -> Dict[str, Any]:
        """Run validation rollouts."""
        cfg = self.cfg
        self.policy.eval()
        
        runner_log = None
        if self.env_runner is not None:
            policy_copy = copy.deepcopy(self.policy)
            policy_copy.eval()
            runner_log = self.env_runner.run(policy_copy)
            
            # Cleanup
            del policy_copy
        
        return runner_log if runner_log is not None else {}
    
    def _save_checkpoint(self):
        """Save current policy state."""
        ckpt_path = self.output_dir.joinpath("streaming_flow_policy.ckpt")
        
        payload = {
            'cfg': self.cfg,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'state_dicts': {
                'model': self.policy.state_dict(),
            }
        }
        
        torch.save(payload, ckpt_path.open('wb'), pickle_module=self.pickle_module)
        print(f"Saved checkpoint to {ckpt_path}")
    
    def save_checkpoint(self):
        """Public interface for saving checkpoints."""
        self._save_checkpoint()