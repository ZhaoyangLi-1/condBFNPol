"""Conditional BFN-backed policy implementations."""

from __future__ import annotations

import torch as t
import torch.nn as nn

from typing import Any, Iterable, Optional, Sequence
import contextlib

from bfn_utils import default, exists
from conditional_bfn import ContinuousBFN
from networks.base import BFNetwork, RandomOrLearnedSinusoidalPosEmb
from policies.base import BasePolicy

__all__ = ["ConditionalBFNBackbone", "ConditionalBFNPolicy"]


class ConditionalBFNBackbone(BFNetwork):
    """Lightweight MLP backbone that conditions on observations."""

    def __init__(
        self,
        *,
        action_dim: int,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        time_emb_dim: int = 64,
        cond_drop_prob: float = 0.1,
        random_time_emb: bool = False,
    ):
        super().__init__(is_conditional_model=True)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.cond_dim = obs_dim  # used for validation in ContinuousBFN
        self.cond_is_discrete = False
        self.cond_drop_prob = cond_drop_prob

        # Time embeddings mirror the style used in network modules.
        self.time_mlp = nn.Sequential(
            RandomOrLearnedSinusoidalPosEmb(time_emb_dim, random_time_emb),
            nn.Linear(time_emb_dim + 1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Observation encoder.
        hidden_dims = (
            tuple(hidden_dims) if isinstance(hidden_dims, Iterable) else (hidden_dims,)
        )
        obs_layers = []
        last_dim = obs_dim
        for dim in hidden_dims:
            obs_layers.extend([nn.Linear(last_dim, dim), nn.GELU()])
            last_dim = dim
        self.cond_proj = nn.Sequential(*obs_layers)
        self._cond_dim = last_dim

        # Core predictor.
        core_layers = []
        core_in = action_dim + time_emb_dim + self._cond_dim
        for dim in hidden_dims:
            core_layers.extend([nn.Linear(core_in, dim), nn.GELU()])
            core_in = dim
        core_layers.append(nn.Linear(core_in, action_dim))
        self.core = nn.Sequential(*core_layers)

    def forward(
        self,
        x: t.Tensor,
        time: t.Tensor,
        cond: Optional[t.Tensor],
        cond_drop_prob: Optional[float] = None,
    ) -> t.Tensor:
        batch = x.shape[0]
        if time.shape == (1,):
            time = time.expand(batch)

        time_emb = self.time_mlp(time[:, None])
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(cond):
            cond_flat = cond.view(batch, -1).to(dtype=x.dtype, device=x.device)
            cond_emb = self.cond_proj(cond_flat)
            if cond_drop_prob > 0.0:
                keep_mask = t.rand((batch,), device=x.device) < (1 - cond_drop_prob)
                cond_emb = cond_emb * keep_mask[:, None]
        else:
            cond_emb = t.zeros((batch, self._cond_dim), device=x.device, dtype=x.dtype)

        h = t.cat((x.view(batch, -1), time_emb, cond_emb), dim=-1)
        out = self.core(h)
        return out.view_as(x)


class ConditionalBFNPolicy(BasePolicy):
    """Policy that samples actions via a conditional ContinuousBFN."""

    def __init__(
        self,
        *,
        action_space: Any,
        obs_dim: int,
        action_dim: int,
        backbone: Optional[BFNetwork] = None,
        hidden_dims: Sequence[int] = (256, 256),
        time_emb_dim: int = 64,
        sigma_1: float = 0.001,
        n_timesteps: int = 20,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
        deterministic_seed: int = 0,
        random_time_emb: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sigma_1 = sigma_1
        self.n_timesteps = n_timesteps
        self.cond_scale = cond_scale
        self.rescaled_phi = rescaled_phi
        self.deterministic_seed = deterministic_seed

        self.backbone = backbone or ConditionalBFNBackbone(
            action_dim=action_dim,
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            time_emb_dim=time_emb_dim,
            random_time_emb=random_time_emb,
        )
        self.bfn = ContinuousBFN(
            dim=action_dim, net=self.backbone, device_str=device, dtype_str=dtype
        )

    def _flatten_obs(self, obs: t.Tensor) -> t.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        return obs.view(obs.size(0), -1)

    def _sample_actions(
        self,
        cond: t.Tensor,
        *,
        cond_scale: Optional[float],
        rescaled_phi: Optional[float],
    ) -> t.Tensor:
        actions = self.bfn.sample(
            n_samples=1,
            sigma_1=self.sigma_1,
            n_timesteps=self.n_timesteps,
            cond=cond,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )
        return actions.squeeze(1)

    def forward(
        self,
        obs: t.Tensor,
        *,
        deterministic: bool = False,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> t.Tensor:
        cond = self._flatten_obs(obs)
        cond_scale = default(cond_scale, self.cond_scale)
        rescaled_phi = default(rescaled_phi, self.rescaled_phi)

        if deterministic:
            ctx = (
                t.random.fork_rng(devices=[self.device])
                if isinstance(self.device, t.device) and self.device.type == "cuda"
                else contextlib.nullcontext()
            )
            with ctx:
                t.manual_seed(self.deterministic_seed)
                return self._sample_actions(
                    cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi
                )

        return self._sample_actions(
            cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi
        )
