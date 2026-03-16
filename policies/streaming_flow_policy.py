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
from pydrake.all import PiecewisePolynomial

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
    sigma1: float = 0.0
    pred_horizon: int = 16
    num_integration_steps: int = 100


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
        sigma1: float = 0.0,
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
        self.num_integration_steps = num_integration_steps
        
        # Compute sigma_r for the noise schedule
        assert 0 <= sigma0 <= sigma1, "sigma0 must be <= sigma1"
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
            
        # Initialize neural ODE for integration
        self.neural_ode = NeuralODE(self._velocity_wrapper, solver='dopri5')
        
        log.info(f"StreamingFlowPolicy initialized with action_dim={self.action_dim}, "
                f"sigma0={sigma0}, sigma1={sigma1}, horizon={horizon}")

    def _velocity_wrapper(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Wrapper for velocity network to be compatible with NeuralODE.
        
        Args:
            t: Time tensor [batch_size]
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
        obs_dict = {k: v for k, v in batch.items() if k.startswith('obs')}
        actions = batch['action']  # [B, horizon, action_dim]
        
        batch_size = actions.shape[0]
        
        # Normalize data if normalizers are available
        if self.obs_normalizer is not None:
            obs_dict = self.obs_normalizer.normalize(obs_dict)
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)
        
        # Concatenate observations
        obs_features = []
        for key in sorted(obs_dict.keys()):
            obs = obs_dict[key]
            if len(obs.shape) > 2:  # Handle image observations
                obs = obs.flatten(start_dim=1)
            obs_features.append(obs)
        
        obs_concat = torch.cat(obs_features, dim=1)  # [B, obs_dim]
        
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
            obs_concat
        )
        
        # Target velocity (derivative of the flow)
        # For streaming flow, the target is the derivative of the trajectory
        target_velocity = actions_flat + self.sigma_r * z0  # Simplified target
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(
            velocity_pred.reshape(batch_size, -1),
            target_velocity
        )
        
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