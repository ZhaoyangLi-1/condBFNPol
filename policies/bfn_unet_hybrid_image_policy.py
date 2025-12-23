"""BFN U-Net Hybrid Image Policy.

This module implements a Bayesian Flow Network policy for visual robot control tasks.
It uses the same architecture as DiffusionUnetHybridImagePolicy but with BFN sampling.

The policy uses:
- A vision encoder (e.g., ResNet) to extract features from RGB images
- A conditional 1D U-Net as the backbone network
- Continuous BFN for generative modeling
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from policies.base import BasePolicy
from networks.conditional_bfn import ContinuousBFN
from networks.base import BFNetwork

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

__all__ = ["BFNUnetHybridImagePolicy"]


class UnetBFNWrapper(BFNetwork):
    """Wrapper to make ConditionalUnet1D compatible with ContinuousBFN.
    
    The BFN expects a network that takes (x, t, cond) and outputs the same shape as x.
    ConditionalUnet1D expects (sample, timestep, global_cond) and outputs the same shape.
    """
    
    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        action_dim: int,
        cond_dim: int,
    ):
        super().__init__(is_conditional_model=True)
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim
        # Required by ContinuousBFN for conditional models
        self.cond_dim = cond_dim
        self.cond_is_discrete = False  # We use continuous conditioning
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            x: Noisy actions [B, horizon * action_dim]
            t: Timesteps [B] or scalar, values in [0, 1]
            cond: Global conditioning [B, cond_dim]
            
        Returns:
            Output [B, horizon * action_dim]
        """
        B = x.shape[0]
        
        # Reshape from [B, horizon * action_dim] to [B, horizon, action_dim]
        # ConditionalUnet1D expects [B, T, input_dim] and internally rearranges to [B, input_dim, T]
        x = x.view(B, self.horizon, self.action_dim)  # [B, horizon, action_dim]
        
        # Convert BFN time to integer timesteps for U-Net
        if t.dim() == 0:
            t = t.expand(B)
        
        # Scale t from [0, 1] to integer timesteps
        timesteps = (t * 999).long().clamp(0, 999)
        
        # Forward through U-Net
        out = self.model(
            sample=x,
            timestep=timesteps,
            global_cond=cond,
        )
        
        # U-Net outputs [B, horizon, action_dim], flatten to [B, horizon * action_dim]
        out = out.reshape(B, -1)  # [B, horizon * action_dim]
        
        return out


class BFNUnetHybridImagePolicy(BasePolicy):
    """BFN-based policy for image observations using U-Net backbone.
    
    This policy processes multi-modal observations (images + low-dim state)
    through a vision encoder, then uses a conditional U-Net with BFN
    to generate action trajectories.
    
    Key features:
    - Same architecture as DiffusionUnetHybridImagePolicy
    - Uses Continuous BFN instead of DDPM for sampling
    - Compatible with the standard training pipeline
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # BFN config
        sigma_1: float = 0.001,
        n_timesteps: int = 20,
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
        **kwargs,
    ):
        # Parse shape_meta to get dimensions
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1, "Action must be 1D"
        action_dim = action_shape[0]
        
        # Initialize base with None action_space
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
            obs_encoder = self._build_simple_encoder(obs_shape_meta)
        
        # Create the conditional U-Net
        obs_feature_dim = obs_encoder.output_shape()[0] if HAS_ROBOMIMIC else 512
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        unet_model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        # Wrap U-Net for BFN compatibility
        unet_wrapper = UnetBFNWrapper(
            model=unet_model,
            horizon=horizon,
            action_dim=action_dim,
            cond_dim=global_cond_dim,
        )
        
        # Create BFN
        bfn = ContinuousBFN(
            dim=horizon * action_dim,
            net=unet_wrapper,
            device_str=device,
            dtype_str=dtype,
        )
        
        # Store components
        self.obs_encoder = obs_encoder
        self.model = unet_model  # Keep reference to underlying model
        self.unet_wrapper = unet_wrapper
        self.bfn = bfn
        self.normalizer = LinearNormalizer()
        
        # Store dimensions and config
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.sigma_1 = sigma_1
        self.n_timesteps = n_timesteps
        self.kwargs = kwargs
        
        print(f"BFN U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")
    
    def _build_robomimic_encoder(
        self,
        obs_config: dict,
        obs_key_shapes: dict,
        action_dim: int,
        crop_shape: tuple,
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> nn.Module:
        """Build observation encoder using robomimic via algo_factory."""
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
        """Build a simple MLP encoder as fallback."""
        total_dim = 0
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            if attr.get('type', 'low_dim') == 'low_dim':
                total_dim += shape[0]
            else:
                total_dim += 512
        
        return nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
    
    # ==================== Normalizer Methods ====================
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for inputs/outputs."""
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    # ==================== Forward (required by BasePolicy) ====================
    
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for inference.
        
        Args:
            obs: Observation tensor or dict with shape [B, ...]
            deterministic: Not used for BFN sampling
            
        Returns:
            Action tensor of shape [B, n_action_steps, action_dim]
        """
        # Wrap single observation in dict format expected by predict_action
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': obs}
        else:
            obs_dict = obs
        
        result = self.predict_action(obs_dict)
        return result['action']
    
    # ==================== Inference ====================
    
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict action from observation.
        
        Args:
            obs_dict: Dictionary of observations with shape [B, T, ...]
            
        Returns:
            Dictionary with 'action' key containing predicted actions
        """
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        
        # Flatten observations for encoder
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        
        # Reshape to [B, T*Do]
        cond = nobs_features.reshape(B, -1)
        
        # Sample from BFN
        naction = self.bfn.sample(
            batch_size=B,
            sigma_1=self.sigma_1,
            n_timesteps=self.n_timesteps,
            cond=cond,
        )
        
        # Reshape actions
        naction = naction.reshape(B, T, Da)
        
        # Extract action steps (starting from n_obs_steps - 1)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]
        
        # Unnormalize
        action = self.normalizer['action'].unnormalize(action)
        
        result = {
            'action': action,
            'action_pred': naction
        }
        return result
    
    # ==================== Training ====================
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute BFN loss for training.
        
        Args:
            batch: Dictionary containing 'obs' and 'action' tensors
            
        Returns:
            Scalar loss tensor
        """
        # Normalize inputs
        # obs is a dict with keys like 'image', 'agent_pos'
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        B = naction.shape[0]
        T = self.horizon
        Da = self.action_dim
        
        # Encode observations
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        
        # Reshape to [B, T*Do]
        cond = nobs_features.reshape(B, -1)
        
        # Flatten actions for BFN
        naction_flat = naction.reshape(B, -1)  # [B, T * Da]
        
        # Compute BFN loss
        loss = self.bfn.loss(
            naction_flat,
            cond=cond,
            sigma_1=self.sigma_1,
        )
        
        return loss.mean()
    
    def set_actions(self, action: torch.Tensor):
        """Set actions for inpainting (not used with BFN)."""
        pass
    
    # ==================== State Dict ====================
    
    def state_dict(self):
        """Get state dict including all components."""
        return {
            'obs_encoder': self.obs_encoder.state_dict(),
            'model': self.model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for all components."""
        self.obs_encoder.load_state_dict(state_dict['obs_encoder'])
        self.model.load_state_dict(state_dict['model'])
        if 'normalizer' in state_dict:
            self.normalizer.load_state_dict(state_dict['normalizer'])
