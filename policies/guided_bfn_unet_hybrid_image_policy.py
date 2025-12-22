"""Guided BFN U-Net Hybrid Image Policy.

This module implements a Guided Bayesian Flow Network (Flow Matching) policy 
for visual robot control tasks using the same architecture as DiffusionUnetHybridImagePolicy.

The policy uses:
- A vision encoder (e.g., ResNet) to extract features from RGB images
- A conditional 1D U-Net as the backbone network
- Flow Matching with ODE integration for sampling
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional, Union, NamedTuple, Deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from policies.base import BasePolicy
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

__all__ = ["GuidedBFNUnetHybridImagePolicy", "GuidanceConfig", "HorizonConfig"]


class GuidanceConfig(NamedTuple):
    """Configuration for flow guidance mechanisms."""
    steps: int = 20
    cfg_scale: float = 1.0
    grad_scale: float = 0.0
    method: str = "midpoint"  # euler, midpoint, rk4
    use_tqdm: bool = False


class HorizonConfig(NamedTuple):
    """Configuration for planning and execution horizons."""
    obs_history: int = 2
    prediction: int = 16
    execution: int = 8


class UnetFlowWrapper(BFNetwork):
    """Wrapper to make ConditionalUnet1D compatible with Flow Matching.
    
    The Flow Matching expects a network that predicts velocity v(x_t, t, cond).
    """
    
    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        action_dim: int,
    ):
        super().__init__(is_conditional_model=True)
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the U-Net to predict velocity.
        
        Args:
            x: Current state [B, horizon * action_dim] or [B, horizon, action_dim]
            t: Time values [B] or scalar, values in [0, 1]
            cond: Global conditioning [B, cond_dim]
            
        Returns:
            Velocity prediction [B, horizon * action_dim] or same shape as input
        """
        # Handle input shape
        original_shape = x.shape
        if x.dim() == 2:
            # [B, horizon * action_dim] -> [B, action_dim, horizon]
            B = x.shape[0]
            x = x.view(B, self.horizon, self.action_dim)
            x = x.permute(0, 2, 1)
        elif x.dim() == 3:
            # [B, horizon, action_dim] -> [B, action_dim, horizon]
            x = x.permute(0, 2, 1)
        
        # Convert flow time to integer timesteps for U-Net
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        
        # Scale t from [0, 1] to integer timesteps
        timesteps = (t * 999).long().clamp(0, 999)
        
        # Forward through U-Net
        out = self.model(
            sample=x,
            timestep=timesteps,
            global_cond=cond,
        )
        
        # Convert back to original shape
        out = out.permute(0, 2, 1)  # [B, horizon, action_dim]
        if len(original_shape) == 2:
            out = out.reshape(original_shape[0], -1)
        
        return out


class GuidedBFNUnetHybridImagePolicy(BasePolicy):
    """Guided BFN (Flow Matching) policy for image observations using U-Net backbone.
    
    This policy processes multi-modal observations (images + low-dim state)
    through a vision encoder, then uses a conditional U-Net with Flow Matching
    to generate action trajectories via ODE integration.
    
    Key features:
    - Same architecture as DiffusionUnetHybridImagePolicy
    - Uses Flow Matching objective for training
    - ODE-based sampling (euler, midpoint, rk4)
    - Supports classifier-free guidance
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon_config: HorizonConfig,
        guidance_config: Optional[GuidanceConfig] = None,
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
        cond_drop_prob: float = 0.1,
        # Base policy args
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        action_bounds: Optional[tuple] = None,
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
        
        # Store configs
        self.horizon_config = horizon_config
        self.guidance_config = guidance_config or GuidanceConfig()
        self.cond_drop_prob = cond_drop_prob
        self.action_bounds = action_bounds
        
        # Derived dimensions
        horizon = horizon_config.prediction
        n_obs_steps = horizon_config.obs_history
        n_action_steps = horizon_config.execution
        
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
        
        # Wrap U-Net for Flow Matching compatibility
        flow_wrapper = UnetFlowWrapper(
            model=unet_model,
            horizon=horizon,
            action_dim=action_dim,
        )
        
        # Store components
        self.obs_encoder = obs_encoder
        self.model = unet_model
        self.flow_wrapper = flow_wrapper
        self.normalizer = LinearNormalizer()
        
        # Store dimensions
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.flat_dim = horizon * action_dim
        self.kwargs = kwargs
        
        # Action buffer for temporal consistency
        self._action_buffer: Deque[torch.Tensor] = collections.deque(maxlen=horizon)
        
        print(f"Guided BFN U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
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
        """Build observation encoder using robomimic."""
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        
        with config.unlocked():
            config.observation.modalities.obs.low_dim = obs_config['low_dim']
            config.observation.modalities.obs.rgb = obs_config['rgb']
            config.observation.encoder.rgb.core_class = "VisualCore"
            config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
            config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
            config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
            config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = crop_shape[0]
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = crop_shape[1]
            config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
            config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
        
        ObsUtils.initialize_obs_utils_with_config(config)
        
        obs_encoder = ObsUtils.obs_encoder_factory(
            obs_shapes=obs_key_shapes,
        )
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )
        
        obs_encoder = replace_submodules(
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
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, dmvc.CropRandomizer),
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
            deterministic: Not used for flow matching sampling
            
        Returns:
            Action tensor of shape [B, n_action_steps, action_dim]
        """
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': obs}
        else:
            obs_dict = obs
        
        result = self.predict_action(obs_dict)
        return result['action']
    
    # ==================== ODE Integration ====================
    
    def _euler_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler step."""
        v = self.flow_wrapper(x, t, cond=cond)
        return x + dt * v
    
    def _midpoint_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single midpoint step."""
        v1 = self.flow_wrapper(x, t, cond=cond)
        x_mid = x + 0.5 * dt * v1
        t_mid = t + 0.5 * dt
        v2 = self.flow_wrapper(x_mid, t_mid, cond=cond)
        return x + dt * v2
    
    def _rk4_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single RK4 step."""
        k1 = self.flow_wrapper(x, t, cond=cond)
        k2 = self.flow_wrapper(x + 0.5 * dt * k1, t + 0.5 * dt, cond=cond)
        k3 = self.flow_wrapper(x + 0.5 * dt * k2, t + 0.5 * dt, cond=cond)
        k4 = self.flow_wrapper(x + dt * k3, t + dt, cond=cond)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def _integrate_ode(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        method: str = "midpoint",
    ) -> torch.Tensor:
        """Integrate ODE from noise to data."""
        dt = 1.0 / steps
        x = x0
        
        step_fn = {
            "euler": self._euler_step,
            "midpoint": self._midpoint_step,
            "rk4": self._rk4_step,
        }.get(method, self._midpoint_step)
        
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = step_fn(x, t, dt, cond)
        
        return x
    
    # ==================== Inference ====================
    
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict action from observation using Flow Matching.
        
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
        
        # Flatten observations for encoder
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        
        # Reshape to [B, n_obs_steps * Do]
        cond = nobs_features.reshape(B, -1)
        
        # Sample from standard normal
        device = cond.device
        dtype = cond.dtype
        x0 = torch.randn(B, self.flat_dim, device=device, dtype=dtype)
        
        # Integrate ODE with optional classifier-free guidance
        cfg = self.guidance_config
        if cfg.cfg_scale != 1.0:
            naction = self._sample_with_cfg(x0, cond, cfg.cfg_scale, cfg.steps, cfg.method)
        else:
            naction = self._integrate_ode(x0, cond, cfg.steps, cfg.method)
        
        # Reshape actions
        naction = naction.reshape(B, T, Da)
        
        # Apply action bounds if specified
        if self.action_bounds is not None:
            lo, hi = self.action_bounds
            naction = naction.clamp(lo, hi)
        
        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]
        
        # Unnormalize
        result = {'action': action}
        result = self.normalizer['action'].unnormalize(result)
        
        return result
    
    def _sample_with_cfg(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        cfg_scale: float,
        steps: int,
        method: str,
    ) -> torch.Tensor:
        """Sample with classifier-free guidance."""
        dt = 1.0 / steps
        x = x0
        B = x.shape[0]
        
        step_fn = {
            "euler": self._euler_step,
            "midpoint": self._midpoint_step,
            "rk4": self._rk4_step,
        }.get(method, self._midpoint_step)
        
        for i in range(steps):
            t = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
            
            # Conditioned velocity
            v_cond = self.flow_wrapper(x, t, cond=cond)
            
            # Unconditioned velocity
            v_uncond = self.flow_wrapper(x, t, cond=torch.zeros_like(cond))
            
            # Guided velocity
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Step
            x = x + dt * v
        
        return x
    
    # ==================== Training ====================
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Flow Matching loss for training.
        
        The Flow Matching objective trains the network to predict
        the velocity field that transforms noise to data.
        
        Args:
            batch: Dictionary containing 'obs' and 'action' tensors
            
        Returns:
            Scalar loss tensor
        """
        # Normalize batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']
        
        B = naction.shape[0]
        T = self.horizon
        Da = self.action_dim
        device = naction.device
        dtype = naction.dtype
        
        # Encode observations
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        
        # Reshape to [B, n_obs_steps * Do]
        cond = nobs_features.reshape(B, -1)
        
        # Apply conditioning dropout
        if self.training and self.cond_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cond_drop_prob
            cond = torch.where(drop_mask.unsqueeze(-1), torch.zeros_like(cond), cond)
        
        # Flatten actions: [B, T, Da] -> [B, T * Da]
        x1 = naction.reshape(B, -1)
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample time uniformly
        t = torch.rand(B, device=device, dtype=dtype)
        
        # Interpolate: x_t = (1 - t) * x0 + t * x1
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # Target velocity: v = x1 - x0
        target_v = x1 - x0
        
        # Predict velocity
        pred_v = self.flow_wrapper(x_t, t, cond=cond)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss
    
    def reset(self):
        """Reset action buffer for new episode."""
        self._action_buffer.clear()
    
    def set_actions(self, action: torch.Tensor):
        """Set actions for inpainting (not used with Flow Matching)."""
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
