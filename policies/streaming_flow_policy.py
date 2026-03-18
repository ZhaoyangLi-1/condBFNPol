"""Streaming Flow Policy implementation for condBFNPol.

This module implements a policy wrapper for Streaming Flow Networks.
It follows the same pattern as other policies in the condBFNPol framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from torchdyn.core import NeuralODE

from policies.base import BasePolicy

# Try importing from local utils first, fallback to diffusion_policy
try:
    from utils.normalizer import LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

__all__ = ["StreamingFlowPolicy", "StreamingFlowConfig"]


class StreamingFlowConfig(NamedTuple):
    """Configuration for Streaming Flow Policy hyperparameters."""
    
    sigma0: float = 0.0
    sigma1: float = 0.1
    k: float = 0.0
    pred_horizon: int = 16
    num_integration_steps: int = 100


class StreamingFlowVectorField(nn.Module):
    """Neural ODE vector field wrapper for streaming flow inference."""

    def __init__(self, velocity_net: nn.Module, action_dim: int, horizon: int):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.horizon = horizon

    @staticmethod
    def _expand_time(t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.dim() == 0:
            return t.expand(batch_size)
        if t.numel() == 1:
            return t.reshape(1).expand(batch_size)
        return t

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute action-state velocity for ODE integration."""
        batch_size = x.shape[0]

        action_part = x[:, :self.action_dim * self.horizon]
        obs_part = x[:, self.action_dim * self.horizon:]
        action_seq = action_part.reshape(batch_size, self.horizon, self.action_dim)
        t = self._expand_time(t, batch_size)

        action_velocities = []
        for i in range(self.horizon):
            for j in range(self.action_dim):
                single_action = action_seq[:, i, j:j+1]
                dummy_z = torch.zeros_like(single_action)
                state = torch.stack([single_action, dummy_z], dim=1)

                vel = self.velocity_net(
                    x=state,
                    time=t,
                    global_cond=obs_part,
                )
                action_velocities.append(vel[:, 0, 0])

        action_velocity = torch.stack(action_velocities, dim=1)
        obs_velocity = torch.zeros_like(obs_part)
        return torch.cat([action_velocity.reshape(batch_size, -1), obs_velocity], dim=1)


class StreamingFlowPolicy(BasePolicy):
    """Policy wrapper for Streaming Flow Networks with Action Chunking."""

    def __init__(
        self,
        *,
        action_space: Any,
        velocity_net: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        sigma0: float = 0.0,
        sigma1: float = 0.1,
        k: float = 0.0,
        num_integration_steps: int = 100,
        action_normalizer: Optional[LinearNormalizer] = None,
        obs_normalizer: Optional[LinearNormalizer] = None,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ):
        """Initialize Streaming Flow Policy.
        
        Args:
            action_space: Action space definition
            velocity_net: Neural network for velocity field
            horizon: Total prediction horizon
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps to execute
            sigma0: Standard deviation at t=0
            sigma1: Standard deviation at t=1
            num_integration_steps: Number of ODE integration steps
            action_normalizer: Action normalizer
            obs_normalizer: Observation normalizer
            device: Device to run on
        """
        super().__init__(action_space)
        
        self.action_space = action_space
        self.velocity_net = velocity_net
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.k = k
        self.num_integration_steps = num_integration_steps
        
        # PushT stochastic SFP uses the unstabilized latent schedule.
        assert 0 <= sigma0 <= sigma1, (
            f"sigma0 ({sigma0}) must be <= sigma1 ({sigma1})"
        )
        self.sigma_r = np.sqrt(np.square(sigma1) - np.square(sigma0))
        
        # Normalizers
        self.action_normalizer = action_normalizer
        self.obs_normalizer = obs_normalizer
        
        # Device handling
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        
        # Get action dimension from action space
        if hasattr(action_space, 'shape'):
            self.action_dim = action_space.shape[0]
        else:
            # Fallback for different action space types
            self.action_dim = action_space.n if hasattr(action_space, 'n') else 2
            
        # Neural ODE will be created during inference, not here
        
        log.info(f"StreamingFlowPolicy initialized with action_dim={self.action_dim}, "
                f"sigma0={sigma0}, sigma1={sigma1}, horizon={horizon}")

    def _make_vector_field(self) -> nn.Module:
        return StreamingFlowVectorField(
            velocity_net=self.velocity_net,
            action_dim=self.action_dim,
            horizon=self.horizon,
        )

    def _make_neural_ode(self) -> NeuralODE:
        return NeuralODE(self._make_vector_field(), solver='dopri5')

    # Backward-compatible debug/test hooks.
    def _velocity_wrapper(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._make_vector_field()(t, x, **kwargs)

    @property
    def neural_ode(self) -> NeuralODE:
        return self._make_neural_ode()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict action given observations.
        
        Args:
            obs_dict: Dictionary of observations
            
        Returns:
            Dictionary containing predicted actions
        """
        batch_size = list(obs_dict.values())[0].shape[0]
        
        # Normalize observations if normalizer is available
        if self.obs_normalizer is not None:
            obs_dict = self.obs_normalizer.normalize(obs_dict)
        
        # Concatenate observations
        obs_features = []
        for key in sorted(obs_dict.keys()):
            obs = obs_dict[key]
            if len(obs.shape) > 2:  # Handle image observations
                obs = obs.flatten(start_dim=1)
            obs_features.append(obs)
        
        obs_concat = torch.cat(obs_features, dim=1)  # [B, obs_dim]
        obs_dim = obs_concat.shape[1]
        
        # Sample initial noise
        action_noise = torch.randn(
            batch_size, self.action_dim * self.horizon,
            device=self._device, dtype=obs_concat.dtype
        ) * self.sigma0
        
        # Concatenate action noise and observations for ODE integration
        x0 = torch.cat([action_noise, obs_concat], dim=1)  # [B, action_dim * horizon + obs_dim]
        
        # Create neural ODE for integration
        neural_ode = self._make_neural_ode()
        
        # Integration time span
        t_span = torch.linspace(0.0, 1.0, self.num_integration_steps + 1, device=self._device)
        
        # Integrate ODE
        with torch.no_grad():
            ode_result = neural_ode(x0, t_span)
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
        """Compute training loss following original streaming flow implementation.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss tensor
        """
        # Get observations and actions from batch
        obs_dict = batch['obs'] if 'obs' in batch else {k: v for k, v in batch.items() if k.startswith('obs')}
        actions = batch['action']  # [B, horizon, action_dim]
        
        batch_size = actions.shape[0]
        
        # Process observations (take only the observation steps, not action steps)
        if isinstance(obs_dict, dict) and len(obs_dict) > 0:
            # Handle nested observation dict
            obs_features = []
            for key in sorted(obs_dict.keys()):
                obs = obs_dict[key]
                if len(obs.shape) > 3:  # Image observations [B, T, C, H, W]
                    obs = obs[:, :self.n_obs_steps]  # Take only obs steps
                    obs = obs.reshape(batch_size, -1)  # Flatten
                elif len(obs.shape) == 3:  # Vector observations [B, T, D]
                    obs = obs[:, :self.n_obs_steps]  # Take only obs steps  
                    obs = obs.reshape(batch_size, -1)  # Flatten
                else:
                    obs = obs.reshape(batch_size, -1)
                obs_features.append(obs)
            obs_concat = torch.cat(obs_features, dim=1).to(self._device)
        else:
            # Fallback for simple observations
            obs_concat = torch.randn(batch_size, 10, device=self._device)  # dummy obs
        
        # Generate training samples following original streaming flow algorithm
        losses = []
        
        for i in range(batch_size):
            action_seq = actions[i]  # [horizon, action_dim]
            obs_i = obs_concat[i].unsqueeze(0)  # [1, obs_dim]
            
            # Create trajectory from action sequence (following original implementation)
            traj_times = torch.linspace(0, 1, self.horizon, device=self._device)
            
            # Sample random time and compute trajectory values
            time = torch.rand(1, device=self._device)  # Random time in [0,1]
            
            # Sample a random action index from the sequence  
            action_idx = torch.randint(0, self.horizon, (1,)).item()
            ξt = action_seq[action_idx]  # [action_dim] - current action
            
            # Compute trajectory derivative (velocity) using finite differences
            if action_idx > 0 and action_idx < self.horizon - 1:
                ξ̇t = (action_seq[action_idx + 1] - action_seq[action_idx - 1]) / 2.0  # Central difference
            elif action_idx == 0:
                ξ̇t = action_seq[1] - action_seq[0]  # Forward difference
            else:
                ξ̇t = action_seq[-1] - action_seq[-2]  # Backward difference
                
            # Sample noise
            z0 = torch.randn_like(ξt)  # [action_dim]
            
            # Sample action following original formulation: 
            # at = ξt + ε_a0 + σr * time * z0
            ε_a0 = self.sigma0 * torch.randn_like(ξt)
            at = ξt + ε_a0 + self.sigma_r * time * z0  # [action_dim]
            
            # Sample latent following: zt = (1 - (1-σ1) * time) * z0 + time * ξt
            sigma1 = self.sigma1 if hasattr(self, 'sigma1') else 0.1  # fallback
            zt = (1 - (1-sigma1) * time) * z0 + time * ξt  # [action_dim]
            
            # Compute target velocities following original equations:
            # va = ξ̇t + σr * z0
            # vz = ξt + time * ξ̇t - (1 - σ1) * z0
            va = ξ̇t + self.sigma_r * z0  # [action_dim]
            vz = ξt + time * ξ̇t - (1 - sigma1) * z0  # [action_dim]
            
            # Concatenate a and z for network input
            x = torch.stack([at, zt], dim=0).unsqueeze(0)  # [1, 2, action_dim]
            v_target = torch.stack([va, vz], dim=0).unsqueeze(0)  # [1, 2, action_dim]
            
            # Predict velocity
            v_pred = self.velocity_net(
                sample=x, 
                timestep=time.expand(1), 
                global_cond=obs_i
            )  # [1, 2, action_dim]
            
            # Compute loss for this sample
            sample_loss = torch.nn.functional.mse_loss(v_pred, v_target)
            losses.append(sample_loss)
        
        # Average loss across batch
        loss = torch.stack(losses).mean()
        return loss

    def set_normalizers(self, action_normalizer=None, obs_normalizer=None):
        """Set normalizers for the policy."""
        if action_normalizer is not None:
            self.action_normalizer = action_normalizer
        if obs_normalizer is not None:
            self.obs_normalizer = obs_normalizer

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward inference method.
        
        Args:
            obs: Observations
            deterministic: Whether to sample deterministically (unused for streaming flow)
            **kwargs: Additional arguments
            
        Returns:
            Action tensor of shape [B, ActionDim] (only first action executed)
        """
        # Convert obs to dict if it's a tensor
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': obs}
        else:
            obs_dict = obs
            
        # Get full action sequence using predict_action
        result = self.predict_action(obs_dict)
        action_sequence = result['action']  # [B, horizon, action_dim]
        
        # Return only the first action for execution
        return action_sequence[:, 0]  # [B, action_dim]

    def get_params(self):
        """Get trainable parameters."""
        return self.velocity_net.parameters()
