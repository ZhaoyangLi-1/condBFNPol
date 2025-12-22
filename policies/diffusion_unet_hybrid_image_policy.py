"""Diffusion U-Net Hybrid Image Policy.

This module implements a diffusion-based policy for visual robot control tasks.
It closely follows the original diffusion_policy implementation from real-stanford,
adapted to work with the BasePolicy interface.

The policy uses:
- A vision encoder (e.g., ResNet) to extract features from RGB images
- A conditional 1D U-Net to denoise action trajectories
- DDPM scheduler for the diffusion process
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from policies.base import BasePolicy

# Try to import robomimic components for encoder
try:
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.models.base_nets as rmbn
    import diffusion_policy.model.vision.crop_randomizer as dmvc
    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False

__all__ = ["DiffusionUnetHybridImagePolicy"]


class DiffusionUnetHybridImagePolicy(BasePolicy):
    """Diffusion-based policy for image observations.
    
    This policy processes multi-modal observations (images + low-dim state)
    through a vision encoder, then uses a conditional U-Net to generate
    action trajectories via iterative denoising.
    
    Key features:
    - Supports both global conditioning (obs as context) and inpainting modes
    - Uses FiLM conditioning in the U-Net for observation features
    - Compatible with the standard diffusion_policy training pipeline
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        # Vision encoder config
        crop_shape: tuple = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        # U-Net config
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # Base policy args
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        # Additional kwargs for scheduler.step
        **kwargs,
    ):
        # Parse shape_meta to get dimensions
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1, "Action must be 1D"
        action_dim = action_shape[0]
        
        # Initialize base with None action_space (will be set later if needed)
        super().__init__(
            action_space=None,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )
        
        # Parse observation shape meta
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        
        # Build observation encoder using robomimic if available
        if HAS_ROBOMIMIC:
            obs_encoder = self._build_robomimic_encoder(
                obs_config=obs_config,
                obs_key_shapes=obs_key_shapes,
                action_dim=action_dim,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        else:
            # Fallback to simple encoder
            obs_encoder = self._build_simple_encoder(obs_shape_meta)
        
        # Create the conditional U-Net
        obs_feature_dim = obs_encoder.output_shape()[0] if HAS_ROBOMIMIC else 512
        
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        # Mask generator for inpainting mode
        mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        # Store components
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = mask_generator
        self.normalizer = LinearNormalizer()
        
        # Store dimensions and config
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        # Resolve inference steps
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        print(f"Diffusion params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")
    
    def reset(self):
        """Reset action buffer for new episode."""
        return None
    
    def _build_robomimic_encoder(
        self,
        obs_config: dict,
        obs_key_shapes: dict,
        action_dim: int,
        crop_shape: tuple,
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> nn.Module:
        """Build observation encoder using robomimic."""
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
        
        ObsUtils.initialize_obs_utils_with_config(config)
        
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        
        return obs_encoder
    
    def _build_simple_encoder(self, obs_shape_meta: dict) -> nn.Module:
        """Fallback simple encoder when robomimic is not available."""
        import torchvision
        
        class SimpleImageEncoder(nn.Module):
            def __init__(self, obs_shape_meta):
                super().__init__()
                self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
                self.backbone.fc = nn.Identity()
                self._output_dim = 512
                
                # Check for low-dim inputs
                self.low_dim_keys = []
                self.low_dim_dim = 0
                for key, attr in obs_shape_meta.items():
                    if attr.get('type', 'low_dim') == 'low_dim':
                        self.low_dim_keys.append(key)
                        self.low_dim_dim += attr['shape'][0]
                
                if self.low_dim_dim > 0:
                    self._output_dim += self.low_dim_dim
            
            def output_shape(self):
                return (self._output_dim,)
            
            def forward(self, obs_dict):
                # Process images
                features = []
                for key, value in obs_dict.items():
                    if value.ndim == 4 and value.shape[1] == 3:  # RGB image
                        feat = self.backbone(value)
                        features.append(feat)
                    elif key in self.low_dim_keys:
                        features.append(value.flatten(start_dim=1))
                
                return torch.cat(features, dim=-1)
        
        return SimpleImageEncoder(obs_shape_meta)
    
    # ========= Inference ============
    
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample a trajectory using iterative denoising.
        
        Args:
            condition_data: Data to condition on (inpainting mode)
            condition_mask: Boolean mask for conditioned positions
            local_cond: Local conditioning (per-timestep)
            global_cond: Global conditioning (context)
            generator: Random generator for reproducibility
            
        Returns:
            Denoised trajectory tensor
        """
        model = self.model
        scheduler = self.noise_scheduler
        
        # Start from pure noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        
        # Set timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        
        for t in scheduler.timesteps:
            # Apply conditioning (inpainting)
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # Predict noise
            model_output = model(
                trajectory, t,
                local_cond=local_cond,
                global_cond=global_cond
            )
            
            # Denoise step
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample
        
        # Final conditioning enforcement
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory
    
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict actions from observations.
        
        Args:
            obs_dict: Dictionary of observations with keys matching shape_meta
            
        Returns:
            Dictionary with 'action' and 'action_pred' tensors
        """
        assert 'past_action' not in obs_dict, "past_action not implemented"
        
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        device = self.device
        dtype = self.dtype
        
        # Build conditioning
        local_cond = None
        global_cond = None
        
        if self.obs_as_global_cond:
            # Condition through global feature
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to [B, Do*To]
            global_cond = nobs_features.reshape(B, -1)
            # Empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Condition through inpainting
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to [B, To, Do]
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
        
        # Run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )
        
        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # Extract action steps
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }
    
    # ========= Training ============
    
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set the normalizer for observations and actions."""
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            batch: Dictionary with 'obs' and 'action' keys
            
        Returns:
            Scalar loss tensor
        """
        assert 'valid_mask' not in batch, "valid_mask not implemented"
        
        # Normalize inputs
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # Build conditioning
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # Reshape [B, T, ...] to [B*T, ...]
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to [B, Do*To]
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # Reshape [B, T, ...] to [B*T, ...]
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to [B, T, Do]
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        
        # Generate inpainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        # Sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        
        # Add noise (forward diffusion)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # Compute loss mask
        loss_mask = ~condition_mask
        
        # Apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict noise
        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=local_cond,
            global_cond=global_cond
        )
        
        # Compute target based on prediction type
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # MSE loss with masking
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    # ========= BasePolicy Interface ============
    
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass for inference.
        
        Args:
            obs: Observations (dict or tensor)
            deterministic: Not used (diffusion is stochastic)
            
        Returns:
            Action tensor
        """
        if isinstance(obs, torch.Tensor):
            # Assume it's already encoded observations
            obs_dict = {'obs': obs}
        else:
            obs_dict = obs
        
        result = self.predict_action(obs_dict)
        return result['action']
