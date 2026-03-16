"""Training workspace for Streaming Flow Policy.

This workspace handles the training loop for streaming flow policies,
following the same pattern as other workspaces in the condBFNPol framework.
"""

from __future__ import annotations

import os
import pathlib
import logging
from typing import Any, Dict, Optional

import torch
import tqdm
import wandb
import numpy as np
from omegaconf import DictConfig
import hydra

from workspaces.base_workspace import BaseWorkspace

# Try importing from local utils first, fallback to diffusion_policy
try:
    from utils.normalizer import LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

__all__ = ["TrainStreamingFlowWorkspace"]


class TrainStreamingFlowWorkspace(BaseWorkspace):
    """Workspace for training streaming flow policies."""

    def __init__(self, cfg: DictConfig, output_dir: Optional[pathlib.Path] = None):
        """Initialize training workspace.
        
        Args:
            cfg: Configuration object
            output_dir: Output directory for checkpoints and logs
        """
        super().__init__(cfg, output_dir=output_dir)
        
        # Initialize device
        device_str = self.cfg.training.device
        if device_str == 'cuda' and not torch.cuda.is_available():
            device_str = 'cpu'
            log.warning("CUDA not available, falling back to CPU")
        
        self.device = torch.device(device_str)
        
        # Set random seeds
        self._set_seed(self.cfg.training.seed)
        
        # Initialize policy
        policy_cfg = self.cfg.policy
        self.policy = hydra.utils.instantiate(policy_cfg, device=self.device)
        self.policy.to(self.device)
        
        # Initialize dataset and dataloader
        task_cfg = self.cfg.task
        dataset = hydra.utils.instantiate(task_cfg.dataset)
        
        # Create normalizers
        self.action_normalizer = LinearNormalizer()
        self.obs_normalizer = LinearNormalizer()
        
        # Fit normalizers on dataset
        log.info("Fitting normalizers...")
        self._fit_normalizers(dataset)
        
        # Set normalizers in policy
        self.policy.set_normalizers(
            action_normalizer=self.action_normalizer,
            obs_normalizer=self.obs_normalizer
        )
        
        # Create EMA model (recreate instead of deepcopy to avoid NeuralODE issues)
        if self.cfg.training.use_ema:
            self.ema_model = hydra.utils.instantiate(policy_cfg, device=self.device)
            self.ema_model.set_normalizers(
                action_normalizer=self.action_normalizer,
                obs_normalizer=self.obs_normalizer
            )
            # Copy weights from main policy to EMA model
            self.ema_model.load_state_dict(self.policy.state_dict())
        else:
            self.ema_model = None
        
        # Create dataloaders
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            **self.cfg.dataloader,
        )
        
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset,
            **self.cfg.val_dataloader,
        )
        
        # Initialize optimizer
        optimizer_cfg = self.cfg.optimizer
        self.optimizer = hydra.utils.instantiate(
            optimizer_cfg,
            params=self.policy.get_params()
        )
        
        # Initialize EMA
        if self.cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(
                self.cfg.ema,
                model=self.ema_model
            )
        else:
            self.ema = None
        
        # Initialize scheduler
        if self.cfg.training.lr_scheduler == 'cosine':
            from diffusers.optimization import get_scheduler
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=self.cfg.training.lr_warmup_steps,
                num_training_steps=len(self.dataloader) * self.cfg.training.num_epochs
            )
        else:
            self.lr_scheduler = None
        
        # Initialize environment runner for evaluation (if available)
        try:
            env_runner_cfg = task_cfg.env_runner
            self.env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=self.output_dir)
        except Exception as e:
            log.warning(f"Failed to initialize environment runner: {e}")
            self.env_runner = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Move models to device
        self.policy.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
        
        log.info(f"TrainStreamingFlowWorkspace initialized on device {self.device}")
        log.info(f"Dataset size: {len(dataset)}")
        log.info(f"Batch size: {self.cfg.dataloader.batch_size}")
        log.info(f"Number of epochs: {self.cfg.training.num_epochs}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _fit_normalizers(self, dataset):
        """Fit normalizers on the dataset."""
        # Sample a batch to get data shapes
        sample_batch = dataset[0]
        
        # Collect all data for normalization
        all_actions = []
        all_obs = {}
        
        # Initialize observation containers
        for key in sample_batch:
            if 'obs' in key or key == 'obs':
                all_obs[key] = []
        
        # Collect data from dataset
        log.info("Collecting data for normalization...")
        for i in range(0, len(dataset), max(1, len(dataset) // 1000)):  # Sample ~1000 data points
            batch = dataset[i]
            
            # Collect actions
            if 'action' in batch:
                all_actions.append(batch['action'])
            
            # Collect observations
            for key in all_obs.keys():
                if key in batch:
                    all_obs[key].append(batch[key])
        
        # Fit action normalizer
        if all_actions:
            actions_tensor = torch.stack(all_actions)
            self.action_normalizer.fit(actions_tensor)
            log.info(f"Action normalizer fitted on {len(all_actions)} samples")
        
        # Fit observation normalizer (if we have observations)
        if all_obs:
            # For now, we'll skip observation normalization for hybrid policies
            # as they handle normalization internally through the encoder
            log.info("Skipping observation normalization for hybrid policy")
    
    def run(self):
        """Run training loop."""
        log.info("Starting training...")
        
        # Initialize logging
        if self.cfg.logging.mode == 'online' and self.cfg.logging.project:
            wandb.init(
                project=self.cfg.logging.project,
                name=self.cfg.logging.name,
                config=dict(self.cfg),
                tags=self.cfg.logging.tags,
                group=self.cfg.logging.group,
                resume=self.cfg.logging.resume,
            )
        
        try:
            for epoch in range(self.cfg.training.num_epochs):
                self.epoch = epoch
                
                # Training step
                train_losses = self._train_epoch()
                
                # Validation step
                if epoch % self.cfg.training.val_every == 0:
                    val_losses = self._val_epoch()
                else:
                    val_losses = {}
                
                # Environment evaluation step (like BFN workspace)
                if (epoch % self.cfg.training.rollout_every == 0 and 
                    self.env_runner is not None):
                    try:
                        # Use EMA model for evaluation if available, otherwise use main policy
                        eval_policy = self.ema_model if self.ema_model is not None else self.policy
                        eval_policy.eval()
                        runner_log = self.env_runner.run(eval_policy)
                        eval_results = dict(runner_log) if runner_log else {}
                    except Exception as e:
                        log.warning(f"Environment evaluation failed: {e}")
                        eval_results = {}
                else:
                    eval_results = {}
                
                # Sampling evaluation (like BFN workspace)
                if (epoch % self.cfg.training.sample_every == 0):
                    try:
                        # Get a training batch for sampling evaluation
                        sample_batch = next(iter(self.dataloader))
                        sample_batch = self._to_device(sample_batch, self.device)
                        
                        # Use EMA model for sampling if available, otherwise use main policy
                        eval_policy = self.ema_model if self.ema_model is not None else self.policy
                        eval_policy.eval()
                        
                        with torch.no_grad():
                            # Get ground truth actions
                            gt_action = sample_batch['action']  # [B, horizon, action_dim]
                            
                            # Sample actions from policy
                            obs = sample_batch['obs']
                            pred_action = eval_policy.predict_action(obs)['action']  # [B, horizon, action_dim]
                            
                            # Compute MSE error for action prediction
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            train_losses['train_action_mse_error'] = mse.item()
                            
                    except Exception as e:
                        log.warning(f"Sampling evaluation failed: {e}")
                
                # Combine all log data (like BFN workspace)
                log_data = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    **train_losses,
                    **val_losses,
                    **eval_results,
                }
                
                # Log epoch-level metrics to wandb (like BFN workspace)
                if self.cfg.logging.mode == 'online':
                    wandb.log(log_data, step=self.global_step)
                
                # Print progress
                loss_str = f"train_loss={train_losses.get('train_loss', 0.0):.6f}"
                if val_losses:
                    loss_str += f", val_loss={val_losses.get('val_loss', 0.0):.6f}"
                if eval_results:
                    loss_str += f", test_score={eval_results.get('test_mean_score', 0.0):.3f}"
                    
                log.info(f"Epoch {epoch:04d}: {loss_str}")
                
                # Checkpointing
                if epoch % self.cfg.training.checkpoint_every == 0:
                    self._save_checkpoint(epoch, log_data)
            
            log.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            log.info("Training interrupted by user")
        except Exception as e:
            log.error(f"Training failed with error: {e}")
            raise
        finally:
            if wandb.run is not None:
                wandb.finish()

    def _train_epoch(self):
        """Run one training epoch."""
        self.policy.train()
        
        epoch_losses = []
        
        with tqdm.tqdm(
            self.dataloader,
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=self.cfg.training.tqdm_interval_sec
        ) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # Move batch to device
                batch = self._to_device(batch, self.device)
                
                # Compute loss
                loss = self.policy.compute_loss(batch)
                raw_loss_cpu = loss.item()
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.ema is not None:
                        self.ema.step(self.ema_model)
                    
                    # Update scheduler
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Log step-level metrics to wandb (like BFN workspace)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    
                    is_last_batch = (batch_idx == (len(self.dataloader) - 1))
                    if not is_last_batch and self.cfg.logging.mode == 'online':
                        wandb.log(step_log, step=self.global_step)
                    
                    self.global_step += 1
                
                epoch_losses.append(raw_loss_cpu)
                
                # Update progress bar
                tepoch.set_postfix({
                    'loss': f'{raw_loss_cpu:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return {
            'train_loss': np.mean(epoch_losses),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    def _val_epoch(self):
        """Run one validation epoch."""
        self.policy.eval()
        
        epoch_losses = []
        
        with torch.no_grad():
            with tqdm.tqdm(
                self.val_dataloader,
                desc=f"Validation epoch {self.epoch}",
                leave=False,
                mininterval=self.cfg.training.tqdm_interval_sec
            ) as tepoch:
                for batch in tepoch:
                    # Move batch to device
                    batch = self._to_device(batch, self.device)
                    
                    # Compute loss
                    loss = self.policy.compute_loss(batch)
                    val_loss_cpu = loss.item()
                    epoch_losses.append(val_loss_cpu)
                    
                    # Update progress bar
                    tepoch.set_postfix({
                        'val_loss': f'{val_loss_cpu:.6f}'
                    })
        
        return {
            'val_loss': np.mean(epoch_losses),
        }

    def _eval_epoch(self):
        """Run environment evaluation."""
        if self.env_runner is None:
            return {}
        
        self.policy.eval()
        
        # Use EMA model for evaluation if available, otherwise use main policy
        eval_policy = self.ema_model if self.ema_model is not None else self.policy
        eval_policy.eval()
        
        try:
            # Run environment evaluation
            results = self.env_runner.run(eval_policy)
            
            # Process results
            eval_results = {}
            if hasattr(results, 'test_mean_score'):
                eval_results['test_mean_score'] = results.test_mean_score
            if hasattr(results, 'test_std_score'):
                eval_results['test_std_score'] = results.test_std_score
            
            return eval_results
            
        except Exception as e:
            log.warning(f"Environment evaluation failed: {e}")
            return {}

    def _save_checkpoint(self, epoch: int, log_data: Dict[str, Any]):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': dict(self.cfg),
            'action_normalizer': self.action_normalizer.state_dict() if self.action_normalizer else None,
            'obs_normalizer': self.obs_normalizer.state_dict() if self.obs_normalizer else None,
        }
        
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.ckpt"
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
        torch.save(checkpoint, epoch_path)
        
        log.info(f"Checkpoint saved: {epoch_path}")

    def _to_device(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}