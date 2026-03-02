"""Training Workspace for Unified Streaming Flow Policy.

This workspace handles complete training of the Streaming Flow Policy
with support for both image and low-dimensional observations.

Features:
- Network training with gradient descent
- Support for mixed image + low-dim observations  
- EMA model tracking
- LR scheduling with warmup
- Validation and rollout evaluation
- WandB logging
- Top-K checkpoint management
"""

from __future__ import annotations

import copy
import os
import pathlib
import random
from typing import Any, Dict, Optional

import dill
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

# Import from diffusion_policy for utilities
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger

try:
    from diffusion_policy.model.diffusion.ema_model import EMAModel
except ImportError:
    from utils.ema import EMAModel

try:
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
except ImportError:
    from torch.optim.lr_scheduler import LambdaLR
    
    def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """Simple scheduler factory."""
        if name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=num_training_steps, last_epoch=last_epoch)
        elif name == 'linear':
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
            return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
        else:
            raise ValueError(f"Unknown scheduler: {name}")

__all__ = ["TrainStreamingFlowWorkspace"]


# Register OmegaConf resolver for eval
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainStreamingFlowWorkspace(BaseWorkspace):
    """Workspace for training Unified Streaming Flow Policy.
    
    This workspace manages the complete training pipeline including:
    - Model instantiation from config
    - Dataset loading and normalization
    - EMA model tracking
    - Training loop with validation
    - Environment rollout evaluation
    - Logging and checkpointing
    """
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """Initialize the workspace.
        
        Args:
            cfg: Hydra configuration object.
            output_dir: Output directory for logs and checkpoints.
        """
        super().__init__(cfg, output_dir=output_dir)
        
        # Add pickle module for compatibility
        self.pickle_module = dill
        
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
        
        # Initialize EMA
        if cfg.training.use_ema:
            # Directly instantiate EMA to avoid parameter conflicts
            from utils.ema import EMAModel
            self.ema = EMAModel(
                model=policy,
                update_after_step=cfg.ema.update_after_step,
                inv_gamma=cfg.ema.inv_gamma,
                power=cfg.ema.power,
                min_value=cfg.ema.min_value,
                max_value=cfg.ema.max_value,
            )
        else:
            self.ema = None
        
        # Initialize optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, 
            params=policy.parameters())
        self.optimizer = optimizer
        
        # Initialize dataset and dataloader
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
        
        # Initialize LR scheduler
        num_epochs = cfg.training.num_epochs
        num_training_steps = len(dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = lr_scheduler
        
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
        
        # Initialize checkpoint manager
        if 'topk' in cfg.checkpoint:
            self.checkpoint_manager = TopKCheckpointManager(
                save_dir=self.output_dir.joinpath('checkpoints'),
                **cfg.checkpoint.topk
            )
        else:
            self.checkpoint_manager = None
            
        print("============= Configuration =============")
        print(OmegaConf.to_yaml(cfg))
        print("=========================================")
    
    def run(self):
        """Main training loop."""
        cfg = copy.deepcopy(self.cfg)
        
        # Training loop
        num_epochs = cfg.training.num_epochs
        
        with tqdm.tqdm(range(num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                self.epoch = epoch_idx
                epoch_loss = []
                
                # Training
                self.policy.train()
                with tqdm.tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Move batch to device
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        
                        # Compute loss
                        loss_dict = self.policy.compute_loss(batch)
                        loss = loss_dict['loss']
                        
                        # Backward pass
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # LR scheduler step
                        self.lr_scheduler.step()
                        
                        # EMA update
                        if self.ema is not None:
                            self.ema.step(self.policy)
                        
                        # Logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                        
                        self.global_step += 1
                
                # Epoch logging
                epoch_loss_mean = np.mean(epoch_loss)
                tglobal.set_postfix(loss=epoch_loss_mean)
                
                log_data = {
                    'epoch': self.epoch,
                    'train_loss': epoch_loss_mean,
                    'lr': self.lr_scheduler.get_last_lr()[0]
                }
                
                # Validation
                if (epoch_idx + 1) % cfg.training.val_every == 0:
                    val_loss = self._run_validation()
                    log_data['val_loss'] = val_loss
                
                # Environment evaluation
                if (epoch_idx + 1) % cfg.training.rollout_every == 0 and self.env_runner is not None:
                    runner_log = self._run_env_evaluation()
                    log_data.update(runner_log)
                
                # Log to wandb and JSON
                if self.wandb_run_dir is not None:
                    wandb.log(log_data, step=self.global_step)
                self.json_logger.log(log_data)
                
                # Checkpointing
                if (epoch_idx + 1) % cfg.training.checkpoint_every == 0:
                    if self.checkpoint_manager is not None:
                        # Save with topk manager
                        self.checkpoint_manager.save_checkpoint(
                            {
                                'cfg': cfg,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'state_dicts': {
                                    'model': self.policy.state_dict(),
                                    'ema': self.ema.state_dict() if self.ema is not None else None,
                                    'optimizer': self.optimizer.state_dict(),
                                    'lr_scheduler': self.lr_scheduler.state_dict(),
                                }
                            },
                            metrics={'val_loss': log_data.get('val_loss', epoch_loss_mean)},
                            epoch=self.epoch
                        )
                    else:
                        # Save regular checkpoint
                        self._save_checkpoint()
        
        # Save final checkpoint
        if cfg.checkpoint.save_last_ckpt:
            self._save_checkpoint(tag='last')
        
        print("Streaming Flow Policy training completed!")
    
    def _run_validation(self) -> float:
        """Run validation."""
        self.policy.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                loss_dict = self.policy.compute_loss(batch)
                val_losses.append(loss_dict['loss'].item())
        
        self.policy.train()
        return np.mean(val_losses)
    
    def _run_env_evaluation(self) -> Dict[str, Any]:
        """Run environment evaluation."""
        if self.env_runner is None:
            return {}
            
        self.policy.eval()
        
        # Use EMA model for evaluation if available
        if self.ema is not None:
            # Use local EMA interface - use averaged_model directly
            policy_copy = self.ema.averaged_model
        else:
            policy_copy = copy.deepcopy(self.policy)
        
        policy_copy.eval()
        runner_log = self.env_runner.run(policy_copy)
        
        # Cleanup
        del policy_copy
        
        self.policy.train()
        return runner_log
    
    def _save_checkpoint(self, tag: str = 'latest'):
        """Save current training state."""
        ckpt_path = self.output_dir.joinpath(f"streaming_flow_{tag}.ckpt")
        
        payload = {
            'cfg': self.cfg,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'state_dicts': {
                'model': self.policy.state_dict(),
                'ema': self.ema.state_dict() if self.ema is not None else None,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
        }
        
        torch.save(payload, ckpt_path.open('wb'), pickle_module=dill)
        print(f"Saved checkpoint to {ckpt_path}")
    
    def save_checkpoint(self):
        """Public interface for saving checkpoints."""
        self._save_checkpoint()