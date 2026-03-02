"""Streaming Flow Policy implementation for condBFNPol framework."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Add streaming flow policy to path
PROJECT_ROOT = Path(__file__).parent.parent
SFP_PATH = PROJECT_ROOT / "streaming_flow_policy"
if str(SFP_PATH) not in sys.path:
    sys.path.insert(0, str(SFP_PATH))

try:
    from core.sfp_latent_base import StreamingFlowPolicyLatentBase
    from core.sfp_latent import StreamingFlowPolicyLatent
    from pydrake.all import Trajectory, PiecewisePolynomial
except ImportError as e:
    print(f"Warning: Could not import streaming flow policy modules: {e}")
    StreamingFlowPolicyLatentBase = None
    StreamingFlowPolicyLatent = None
    Trajectory = None
    PiecewisePolynomial = None

from .base import BasePolicy


class StreamingFlowPolicy(BasePolicy):
    """Streaming Flow Policy wrapper for condBFNPol framework.
    
    This class wraps the streaming flow policy to be compatible with the 
    condBFNPol training and evaluation pipeline.
    """
    
    def __init__(
        self,
        action_space: Any,
        shape_meta: Dict[str, Any],
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        trajectories: Optional[List] = None,
        prior: Optional[List[float]] = None,
        sigma0: float = 0.1,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs
    ):
        """Initialize Streaming Flow Policy.
        
        Args:
            action_space: Environment action space
            shape_meta: Shape metadata for observations and actions
            horizon: Planning horizon
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps to execute
            trajectories: List of demonstration trajectories
            prior: Prior probabilities for trajectories
            sigma0: Initial Gaussian tube standard deviation
            device: Device to run on
            dtype: Data type
        """
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.sigma0 = sigma0
        
        # Get action dimension from shape_meta
        action_shape = shape_meta["action"]["shape"]
        if isinstance(action_shape, (list, tuple)):
            self.action_dim = action_shape[0]
        else:
            self.action_dim = action_shape
            
        # Initialize with dummy trajectories if not provided
        if trajectories is None or prior is None:
            self.trajectories = self._create_dummy_trajectories()
            if len(self.trajectories) > 0:
                self.prior = [1.0 / len(self.trajectories)] * len(self.trajectories)
            else:
                self.prior = [1.0]  # Default single trajectory
        else:
            self.trajectories = trajectories
            self.prior = prior
            
        # Initialize the streaming flow policy core
        if StreamingFlowPolicyLatent is not None:
            self.sfp_core = StreamingFlowPolicyLatent(
                dim=self.action_dim,
                trajectories=self.trajectories,
                prior=self.prior,
                σ0=self.sigma0
            )
        else:
            self.sfp_core = None
            print("Warning: StreamingFlowPolicyLatent not available, using dummy implementation")
            
    def _create_dummy_trajectories(self) -> List:
        """Create dummy trajectories for initialization."""
        if PiecewisePolynomial is None:
            print("Warning: PyDrake not available, using empty trajectory list")
            return []
            
        try:
            # Create simple linear trajectories as examples
            trajectories = []
            time_points = np.linspace(0, 1, 10)
            
            for i in range(3):  # Create 3 dummy trajectories
                # Create random waypoints
                waypoints = np.random.randn(self.action_dim, len(time_points)) * 0.1
                traj = PiecewisePolynomial.FirstOrderHold(time_points, waypoints)
                trajectories.append(traj)
                
            return trajectories
        except Exception as e:
            print(f"Warning: Failed to create dummy trajectories: {e}")
            return []
        
    def forward(
        self,
        obs: Dict[str, Tensor],
        **kwargs
    ) -> Dict[str, Tensor]:
        """Forward pass of the streaming flow policy.
        
        Args:
            obs: Observation dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predicted actions
        """
        batch_size = self._get_batch_size(obs)
        device = self._get_device()
        
        if self.sfp_core is None:
            # Dummy implementation when sfp_core is not available
            actions = torch.randn(
                batch_size, self.n_action_steps, self.action_dim,
                device=device, dtype=self._dtype
            )
            return {"action": actions}
            
        # For now, implement a simple version
        # In a full implementation, this would:
        # 1. Process observations to determine current state
        # 2. Use SFP to predict action sequence
        # 3. Return formatted actions
        
        actions = torch.randn(
            batch_size, self.n_action_steps, self.action_dim,
            device=device, dtype=self._dtype
        )
        
        return {"action": actions}
        
    def predict_action(
        self,
        obs: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Predict action for environment interaction.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            Action dictionary
        """
        # Convert numpy to torch if needed
        obs_torch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_torch[key] = torch.from_numpy(value).to(
                    device=self.device, dtype=self._dtype
                )
            else:
                obs_torch[key] = value.to(device=self.device, dtype=self._dtype)
                
        # Add batch dimension if missing
        for key, value in obs_torch.items():
            if value.ndim == len(self.shape_meta["obs"][key]["shape"]):
                obs_torch[key] = value.unsqueeze(0)
                
        with torch.no_grad():
            result = self.forward(obs_torch)
            
        # Convert back to numpy and remove batch dimension
        action = result["action"].cpu().numpy()
        if action.shape[0] == 1:
            action = action[0]
            
        return {"action": action}
        
    def compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute training loss.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss dictionary
        """
        # For streaming flow policy, we don't train in the traditional sense
        # The policy is based on demonstration trajectories
        # This is a placeholder for compatibility
        
        loss = torch.tensor(0.0, device=self.device)
        return {"loss": loss}
        
    def _get_batch_size(self, obs: Dict[str, Tensor]) -> int:
        """Get batch size from observations."""
        for key, value in obs.items():
            return value.shape[0]
        return 1
        
    def _get_device(self) -> torch.device:
        """Get device from model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self.device)