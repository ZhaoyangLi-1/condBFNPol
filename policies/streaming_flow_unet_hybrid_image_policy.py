"""Streaming Flow Policy with U-Net for hybrid image tasks.

This module implements a streaming flow policy specifically for tasks with both
low-dimensional and high-dimensional (image) observations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union, List

import torch
import torch.nn as nn
import numpy as np

from policies.streaming_flow_policy import StreamingFlowPolicy, StreamingFlowConfig
from networks.obs_encoders import get_obs_encoder

# Try importing from local utils first, fallback to diffusion_policy
try:
    from utils.normalizer import LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

__all__ = ["StreamingFlowUnetHybridImagePolicy"]


class ConditionalUnet1D(nn.Module):
    """Simple U-Net for action sequence generation conditioned on observations."""
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: List[int] = [512, 1024, 2048],
        kernel_size: int = 5,
        n_groups: int = 8,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        
        # Time embedding
        self.time_embed_dim = diffusion_step_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, diffusion_step_embed_dim),
            nn.ReLU(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )
        
        # Global condition embedding
        self.global_cond_embed = nn.Linear(global_cond_dim, diffusion_step_embed_dim)
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], 1)
        
        # Condition projection
        cond_dim = diffusion_step_embed_dim * 2  # time + global condition
        self.cond_proj = nn.Linear(cond_dim, down_dims[0])
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        in_dim = down_dims[0]
        for out_dim in down_dims[1:]:
            self.encoder.append(nn.Sequential(
                nn.GroupNorm(n_groups, in_dim),
                nn.ReLU(),
                nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(n_groups, out_dim),
                nn.ReLU(),
                nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2),
            ))
            in_dim = out_dim
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i, out_dim in enumerate(reversed(down_dims[:-1])):
            self.decoder.append(nn.Sequential(
                nn.GroupNorm(n_groups, in_dim),
                nn.ReLU(),
                nn.ConvTranspose1d(in_dim, out_dim, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(n_groups, out_dim),
                nn.ReLU(),
                nn.ConvTranspose1d(out_dim, out_dim, kernel_size, padding=kernel_size//2),
            ))
            in_dim = out_dim
        
        # Output projection
        self.output_proj = nn.Conv1d(down_dims[0], input_dim, 1)
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, input_dim]
            time: Time tensor [B]
            global_cond: Global condition tensor [B, global_cond_dim]
            
        Returns:
            Output tensor [B, T, input_dim]
        """
        # Transpose for conv1d: [B, T, input_dim] -> [B, input_dim, T]
        x = x.transpose(1, 2)
        
        # Time embedding
        time_embed = self.time_embed(time.unsqueeze(-1))  # [B, time_embed_dim]
        
        # Global condition embedding
        global_embed = self.global_cond_embed(global_cond)  # [B, time_embed_dim]
        
        # Combine conditions
        cond = torch.cat([time_embed, global_embed], dim=1)  # [B, 2 * time_embed_dim]
        cond_proj = self.cond_proj(cond).unsqueeze(-1)  # [B, down_dims[0], 1]
        
        # Input projection
        x = self.input_proj(x)  # [B, down_dims[0], T]
        
        # Add condition
        x = x + cond_proj
        
        # Encoder
        encoder_outputs = [x]
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            skip = encoder_outputs[-(i+2)]  # Skip connection
            x = layer(x)
            x = x + skip  # Add skip connection
        
        # Output projection
        x = self.output_proj(x)
        
        # Transpose back: [B, input_dim, T] -> [B, T, input_dim]
        x = x.transpose(1, 2)
        
        return x


class StreamingFlowUnetHybridImagePolicy(StreamingFlowPolicy):
    """Streaming Flow Policy with U-Net for hybrid image observations."""

    def __init__(
        self,
        *,
        shape_meta: Dict[str, Any],
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        sigma0: float = 0.0,
        sigma1: float = 0.0,
        num_integration_steps: int = 100,
        # Vision encoder parameters
        crop_shape: Optional[List[int]] = None,
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        # U-Net parameters
        diffusion_step_embed_dim: int = 128,
        down_dims: List[int] = [512, 1024, 2048],
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ):
        """Initialize Streaming Flow U-Net Hybrid Image Policy.
        
        Args:
            shape_meta: Shape metadata for observations and actions
            horizon: Prediction horizon
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps to execute
            sigma0: Standard deviation at t=0
            sigma1: Standard deviation at t=1
            num_integration_steps: Number of ODE integration steps
            crop_shape: Image crop shape
            obs_encoder_group_norm: Use group norm in observation encoder
            eval_fixed_crop: Use fixed crop during evaluation
            diffusion_step_embed_dim: Dimension of time embedding
            down_dims: U-Net down dimensions
            kernel_size: Convolution kernel size
            n_groups: Number of groups for group norm
            cond_predict_scale: Whether to predict scale in conditioning
            device: Device to run on
        """
        
        # Get action dimension from shape meta
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Build observation encoder
        obs_config = {
            'crop_shape': crop_shape,
            'obs_encoder_group_norm': obs_encoder_group_norm,
            'eval_fixed_crop': eval_fixed_crop,
        }
        
        obs_encoder = get_obs_encoder(
            shape_meta=shape_meta,
            n_obs_steps=n_obs_steps,
            device=device,
            **obs_config
        )
        
        # Get encoded observation dimension
        with torch.no_grad():
            # Create dummy observations to get encoded dimension
            dummy_obs = {}
            for key, spec in shape_meta['obs'].items():
                shape = [1, n_obs_steps] + list(spec['shape'])
                dummy_obs[key] = torch.zeros(shape, device=device)
            
            encoded_obs = obs_encoder(dummy_obs)
            obs_dim = encoded_obs.shape[1]
        
        # Create velocity network (U-Net)
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        
        # Dummy action space for base class
        class DummyActionSpace:
            def __init__(self, shape):
                self.shape = shape
        
        action_space = DummyActionSpace(action_shape)
        
        # Initialize base class
        super().__init__(
            action_space=action_space,
            velocity_net=velocity_net,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            sigma0=sigma0,
            sigma1=sigma1,
            num_integration_steps=num_integration_steps,
            device=device,
            **kwargs,
        )
        
        # Store additional attributes
        self.obs_encoder = obs_encoder
        self.shape_meta = shape_meta
        self.crop_shape = crop_shape
        
        log.info(f"StreamingFlowUnetHybridImagePolicy initialized with obs_dim={obs_dim}, "
                f"action_dim={action_dim}")

    def _velocity_wrapper(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Wrapper for velocity network compatible with NeuralODE.
        
        Args:
            t: Time tensor [batch_size] or scalar
            x: State tensor [batch_size, action_dim * horizon + obs_dim]
            
        Returns:
            Velocity tensor [batch_size, action_dim * horizon + obs_dim]
        """
        batch_size = x.shape[0]
        
        # Split state into action and observation parts
        action_part = x[:, :self.action_dim * self.horizon]  # [B, action_dim * horizon]
        obs_part = x[:, self.action_dim * self.horizon:]     # [B, obs_dim]
        
        # Reshape action part to [B, horizon, action_dim]
        action_seq = action_part.reshape(batch_size, self.horizon, self.action_dim)
        
        # Expand time to match batch size
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif len(t) == 1:
            t = t.expand(batch_size)
            
        # Call velocity network
        action_velocity = self.velocity_net(action_seq, t, obs_part)
        
        # Reshape back to [B, action_dim * horizon]
        action_velocity_flat = action_velocity.reshape(batch_size, -1)
        
        # Observations don't change during integration (zero velocity)
        obs_velocity = torch.zeros_like(obs_part)
        
        # Concatenate action velocity and observation velocity
        full_velocity = torch.cat([action_velocity_flat, obs_velocity], dim=1)
        
        return full_velocity

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict action given observations.
        
        Args:
            obs_dict: Dictionary of observations
            
        Returns:
            Dictionary containing predicted actions
        """
        # Normalize observations if normalizer is available and has observation keys
        if (self.obs_normalizer is not None and 
            hasattr(self.obs_normalizer, 'params_dict') and 
            any(key in self.obs_normalizer.params_dict for key in obs_dict.keys())):
            nobs = self.obs_normalizer.normalize(obs_dict)
        else:
            nobs = obs_dict
            
        # Get batch size
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        
        # Process observations for encoding (following BFN pattern)
        from diffusion_policy.common.pytorch_util import dict_apply
        this_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        
        # Encode observations
        obs_features = self.obs_encoder(this_obs)  # [B * n_obs_steps, obs_feature_dim]
        obs_features = obs_features.reshape(batch_size, -1)  # [B, obs_dim]
        obs_features = obs_features.to(self._device)  # Ensure correct device
        
        # Sample initial noise
        action_noise = torch.randn(
            batch_size, self.action_dim * self.horizon,
            device=self._device, dtype=obs_features.dtype
        ) * self.sigma0
        
        # Concatenate action noise and observations for ODE integration
        x0 = torch.cat([action_noise, obs_features], dim=1)  # [B, action_dim * horizon + obs_dim]
        
        # Integration time span
        t_span = torch.linspace(0.0, 1.0, self.num_integration_steps + 1, device=self._device)
        
        # Integrate ODE
        with torch.no_grad():
            ode_result = self.neural_ode(x0, t_span)
            # NeuralODE returns (t_eval, sol) tuple
            if isinstance(ode_result, tuple):
                trajectory = ode_result[1]  # Get the solution trajectory
            else:
                trajectory = ode_result
        
        # Extract final action sequence
        final_state = trajectory[-1]  # [B, action_dim * horizon + obs_dim]
        action_pred = final_state[:, :self.action_dim * self.horizon]  # [B, action_dim * horizon]
        
        # Reshape to [B, horizon, action_dim]
        action_pred = action_pred.reshape(batch_size, self.horizon, self.action_dim)
        
        # Denormalize actions if normalizer is available
        if self.action_normalizer is not None:
            action_pred = self.action_normalizer.unnormalize(action_pred)
        
        return {
            'action': action_pred,
            'action_pred': action_pred
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss tensor
        """
        # Get observations and actions from batch
        # Use the same pattern as BFN policies
        obs_dict = batch['obs']  # Direct access to nested observation dict
        actions = batch['action']  # [B, horizon, action_dim]
        
        batch_size = actions.shape[0]
        
        # Normalize observations and actions if normalizers are available
        if (self.obs_normalizer is not None and 
            hasattr(self.obs_normalizer, 'params_dict') and 
            any(key in self.obs_normalizer.params_dict for key in obs_dict.keys())):
            nobs = self.obs_normalizer.normalize(obs_dict)
        else:
            nobs = obs_dict
            
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)
        
        # Process observations for encoding (following BFN pattern)
        # Apply temporal slicing to observations
        from diffusion_policy.common.pytorch_util import dict_apply
        this_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        
        # Encode observations
        obs_features = self.obs_encoder(this_obs)  # [B * n_obs_steps, obs_feature_dim]
        obs_features = obs_features.reshape(batch_size, -1)  # [B, obs_dim]
        
        # Ensure observation features are on the correct device
        obs_features = obs_features.to(self._device)
        
        # Sample random time steps
        t = torch.rand(batch_size, device=self._device)  # [B]
        
        # Sample noise
        z0 = torch.randn(batch_size, self.action_dim * self.horizon, device=self._device)
        
        # Compute noisy action trajectory
        actions_flat = actions.reshape(batch_size, -1)  # [B, action_dim * horizon]
        
        # Linear interpolation with noise schedule
        actions_t = (
            self.sigma0 * z0 + 
            actions_flat + 
            self.sigma_r * t.unsqueeze(1) * z0
        )
        
        # Predict velocity
        velocity_pred = self.velocity_net(
            actions_t.reshape(batch_size, self.horizon, self.action_dim),
            t,
            obs_features
        )
        
        # Target velocity (derivative of the flow)
        target_velocity = actions_flat + self.sigma_r * z0  # Simplified target
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(
            velocity_pred.reshape(batch_size, -1),
            target_velocity
        )
        
        return loss

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward inference method.
        
        Args:
            obs: Observations dictionary
            deterministic: Whether to sample deterministically (unused for streaming flow)
            **kwargs: Additional arguments
            
        Returns:
            Action tensor of shape [B, ActionDim] (only first action executed)
        """
        # Get full action sequence using predict_action
        result = self.predict_action(obs)
        action_sequence = result['action']  # [B, horizon, action_dim]
        
        # Return only the first action for execution
        return action_sequence[:, 0]  # [B, action_dim]

    def get_params(self):
        """Get trainable parameters."""
        params = list(self.velocity_net.parameters()) + list(self.obs_encoder.parameters())
        return params

    def reset(self):
        """Reset policy state between episodes."""
        pass