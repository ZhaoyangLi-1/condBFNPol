import numpy as np
import torch as t
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any

from bfn_utils import str_to_torch_dtype

__all__ = ["BasePolicy"]


class BasePolicy(nn.Module, ABC):
    """Shared functionality for robotics policies.

    This class handles device / dtype management, numpy↔torch conversion and
    action clipping. Concrete policies only need to implement `forward`.
    """

    def __init__(
        self,
        action_space: Any,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
    ):
        super().__init__()
        self.action_space = action_space
        self.device = t.device(device)
        self.dtype = str_to_torch_dtype(dtype)
        self.clip_actions = clip_actions

    @abstractmethod
    def forward(
        self, obs: t.Tensor, *, deterministic: bool = False, **kwargs: Any
    ) -> t.Tensor:
        """Return a batch of actions for a batch of observations.

        Args:
            obs: observations with leading batch dimension.
            deterministic: whether to return the mean action (if applicable).
        """
        raise NotImplementedError

    def _to_tensor(self, obs: Any) -> t.Tensor:
        """Convert numpy arrays to tensors on the correct device / dtype."""
        if isinstance(obs, t.Tensor):
            return obs.to(device=self.device, dtype=self.dtype)
        return t.as_tensor(obs, device=self.device, dtype=self.dtype)

    def _clip(self, action: t.Tensor) -> t.Tensor:
        """Clamp actions to the bounds of the action space when available."""
        if not self.clip_actions:
            return action
        low = getattr(self.action_space, "low", None)
        high = getattr(self.action_space, "high", None)
        if low is None or high is None:
            return action
        low_t = t.as_tensor(low, device=action.device, dtype=action.dtype)
        high_t = t.as_tensor(high, device=action.device, dtype=action.dtype)
        return t.clamp(action, low_t, high_t)

    @t.inference_mode()
    def act(
        self,
        obs: Any,
        *,
        deterministic: bool = False,
        return_torch: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | t.Tensor:
        """Convenience method for environment interaction.

        Args:
            obs: observation from the environment (numpy or tensor).
            deterministic: whether to request a deterministic action.
            return_torch: when True, keep the action as a tensor on device.
            **kwargs: forwarded to `forward`, e.g. conditioning information.
        """
        obs_t = self._to_tensor(obs)
        obs_space_shape = getattr(self.action_space, "shape", None)
        expectation_ndim = None if obs_space_shape is None else len(obs_space_shape)

        batch_added = False
        if obs_t.ndim == 1 or obs_t.ndim == expectation_ndim:
            obs_t = obs_t.unsqueeze(0)
            batch_added = True

        action = self.forward(obs_t, deterministic=deterministic, **kwargs)
        action = self._clip(action)

        if return_torch:
            return action

        action_np = (
            action.squeeze(0).detach().cpu().numpy()
            if batch_added
            else action.detach().cpu().numpy()
        )
        return action_np
