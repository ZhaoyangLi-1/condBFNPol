"""Guided Bayesian Flow Network (Flow Matching) policy implementation.

This module implements a policy that generates action sequences using Flow Matching
(mathematically equivalent to Continuous BFNs in this context).

Key Features:
- Flow Matching Objective (MSE on Vector Field).
- Integrated ResNet-MLP Backbone (FiLM/AdaLN) for vector tasks.
- Configurable Normalization and Observation Encoding.
- Action Chunking support.
"""

from __future__ import annotations

import collections
import contextlib
import logging
from typing import Any, Callable, NamedTuple, Optional, Deque, Union, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import BasePolicy
from networks.base import BFNetwork, SinusoidalPosEmb

# Try local imports for factories
try:
    from utils.normalizer import get_normalizer, LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    def get_normalizer(type_str="linear", **kwargs):
        return LinearNormalizer()


try:
    from networks.obs_encoders import get_obs_encoder
except ImportError:

    def get_obs_encoder(type_str, **kwargs):
        return None


log = logging.getLogger(__name__)

__all__ = ["GuidedBFNPolicy", "GuidanceConfig", "HorizonConfig", "BackboneConfig"]


# =============================================================================
# Configurations
# =============================================================================


class GuidanceConfig(NamedTuple):
    """Configuration for diffusion/flow guidance mechanisms."""

    steps: int = 20
    cfg_scale: float = 1.0
    grad_scale: float = 0.0


class HorizonConfig(NamedTuple):
    """Configuration for planning and execution horizons."""

    obs_history: int = 2
    prediction: int = 16
    execution: int = 8


class BackboneConfig(NamedTuple):
    """Configuration for the integrated MLP/ResNet Backbone."""

    hidden_dim: int = 256
    depth: int = 3
    time_emb_dim: int = 128
    cond_drop_prob: float = 0.1
    dropout: float = 0.0
    conditioning_type: str = "film"  # 'film', 'adaln', 'concat'


# =============================================================================
# Internal Backbone Components (Copied for self-containment)
# =============================================================================


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.act = nn.Mish()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, scale, shift):
        residual = x
        x = self.norm(x) * (1 + scale) + shift
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x, scale, shift):
        residual = x
        x = self.norm(x)
        x = x * (1 + scale) + shift
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class ConcatBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x, **kwargs):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class IntegratedFlowBackbone(BFNetwork):
    """Robust MLP Backbone for Flow Matching."""

    def __init__(self, action_dim: int, cond_dim: int, config: BackboneConfig):
        super().__init__(is_conditional_model=True)
        self.action_dim = action_dim
        self.config = config
        self.cond_is_discrete = False

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.time_emb_dim),
            nn.Linear(config.time_emb_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        input_dim = (
            action_dim + config.hidden_dim * 2
            if config.conditioning_type == "concat"
            else action_dim
        )
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        if config.conditioning_type in ["film", "adaln"]:
            self.mod_gen = nn.Linear(
                config.hidden_dim * 2, config.hidden_dim * 2 * config.depth
            )
            block_cls = AdaLNBlock if config.conditioning_type == "adaln" else FiLMBlock
            self.blocks = nn.ModuleList(
                [
                    block_cls(config.hidden_dim, config.dropout)
                    for _ in range(config.depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    ConcatBlock(config.hidden_dim, config.dropout)
                    for _ in range(config.depth)
                ]
            )

        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, action_dim)

    def forward(self, x, time, cond=None, **kwargs):
        t_emb = self.time_mlp(time * 1000.0)  # Robust scaling
        if cond is None:
            cond = torch.zeros(
                (x.shape[0], self.config.hidden_dim), device=x.device
            )  # Fallback
        c_emb = self.cond_mlp(cond)

        if self.config.conditioning_type == "concat":
            h = self.input_proj(torch.cat([x, t_emb, c_emb], dim=-1))
            for block in self.blocks:
                h = block(h)
        else:
            h = self.input_proj(x)
            context = torch.cat([t_emb, c_emb], dim=-1)
            params = self.mod_gen(context).view(
                x.shape[0], self.config.depth, 2, self.config.hidden_dim
            )
            for i, block in enumerate(self.blocks):
                h = block(h, params[:, i, 0], params[:, i, 1])

        return self.output_proj(self.output_norm(h))


# =============================================================================
# Policy Implementation
# =============================================================================


class GuidedBFNPolicy(BasePolicy):
    """Guided Policy using Flow Matching with optional integrated backbone."""

    def __init__(
        self,
        *,
        action_space: Any,
        action_dim: int,
        # Optional custom network
        network: Optional[nn.Module] = None,
        # Factories
        normalizer_type: str = "linear",
        obs_encoder_type: str = "identity",
        obs_encoder_config: Optional[Dict] = None,
        # Configs
        horizons: Optional[HorizonConfig] = None,
        guidance: Optional[GuidanceConfig] = None,
        backbone_config: Optional[BackboneConfig] = None,
        # Legacy / Overrides
        obs_encoder: Optional[Callable] = None,
        cond_from_obs: Optional[Callable] = None,
        deterministic_seed: int = 42,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        # Fallback args
        obs_horizon_T_o: int = 2,
        action_pred_horizon_T_p: int = 16,
        action_exec_horizon_T_a: int = 8,
        bfn_timesteps: int = 20,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        self.horizons = horizons or HorizonConfig(
            obs_history=obs_horizon_T_o,
            prediction=action_pred_horizon_T_p,
            execution=action_exec_horizon_T_a,
        )
        self.guidance = guidance or GuidanceConfig(
            steps=bfn_timesteps,
            cfg_scale=kwargs.get("cfg_guidance_scale_w", 1.0),
            grad_scale=kwargs.get("grad_guidance_scale_alpha", 0.0),
        )

        self.backbone_config = backbone_config or BackboneConfig(
            hidden_dim=kwargs.get("hidden_dims", (256,))[0]
            if "hidden_dims" in kwargs
            else 256,
            depth=4,
            time_emb_dim=128,
        )

        self.action_dim = action_dim
        self.deterministic_seed = deterministic_seed

        # 1. Components
        self.normalizer = get_normalizer(normalizer_type)

        if obs_encoder is not None:
            self.obs_encoder = obs_encoder
        else:
            enc_kwargs = obs_encoder_config if obs_encoder_config else {}
            self.obs_encoder = get_obs_encoder(obs_encoder_type, **enc_kwargs)

        # 2. Network
        # Determine total flat dim based on horizon
        self.flat_dim = action_dim * self.horizons.prediction

        self.network = network
        if self.network is None:
            # Build default MLP backbone
            obs_dim = kwargs.get("obs_dim", 2)  # Fallback for PointMaze

            self.network = IntegratedFlowBackbone(
                action_dim=self.flat_action_dim
                if hasattr(self, "flat_action_dim")
                else self.flat_dim,
                cond_dim=obs_dim,
                config=self.backbone_config,
            )
            log.info(
                f"Built IntegratedFlowBackbone: Act={self.flat_dim}, Obs={obs_dim}, Type={self.backbone_config.conditioning_type}"
            )
        else:
            log.info("Using Custom Network for GuidedBFNPolicy")

        # Buffers
        self.obs_buffer: Deque[torch.Tensor] = collections.deque(
            maxlen=self.horizons.obs_history
        )
        self._c_uncond_token: Optional[torch.Tensor] = None

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())
        self._ensure_normalizer_device()
        log.info(
            f"GuidedBFNPolicy normalizer loaded. Params: {len(self.normalizer.params_dict)}"
        )

    def _ensure_normalizer_device(self) -> None:
        """Keep LinearNormalizer parameters on the target device after fit/load."""
        if not isinstance(self.normalizer, nn.Module):
            return

        target_device = getattr(self, "_device", None)
        if target_device is None:
            return

        try:
            first_param = next(self.normalizer.parameters())
        except StopIteration:
            return

        if first_param.device != target_device:
            self.normalizer.to(device=target_device)

    def _encode_obs(self, obs: Union[torch.Tensor, Dict]) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in obs.items()
            }
            if self.obs_encoder:
                enc = self.obs_encoder(obs)
                # If encoder flattened time into batch (e.g., images with shape [B, T, ...]),
                # reshape back and pool over time so cond matches batch dim.
                if "image" in obs and isinstance(obs["image"], torch.Tensor):
                    batch_size = obs["image"].shape[0]
                    if enc.shape[0] % batch_size == 0 and enc.shape[0] != batch_size:
                        t = enc.shape[0] // batch_size
                        enc = enc.view(batch_size, t, -1).mean(dim=1)
                return enc
            return torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=1)

        obs = obs.to(self.device, dtype=self.dtype)
        if self.obs_encoder:
            return self.obs_encoder(obs)
        return obs.view(obs.size(0), -1)

    def _predict_x1(self, x_t, t_val, cond):
        B = x_t.shape[0]

        # FIX: Handle scalar vs tensor time
        if isinstance(t_val, torch.Tensor) and t_val.ndim > 0:
            t_tensor = t_val  # Use directly if [B]
        else:
            t_tensor = torch.full((B,), t_val, device=self.device, dtype=self.dtype)

        try:
            return self.network(x_t, t_tensor, cond=cond)
        except TypeError:
            return self.network(x_t, t_tensor, cond)

    def _ode_step(self, x_t, t_val, dt, cond):
        pred_x1 = self._predict_x1(x_t, t_val, cond)
        denom = max(1 - t_val, 1e-5)
        velocity = (pred_x1 - x_t) / denom
        return x_t + velocity * dt

    def plan(self, obs: Any) -> torch.Tensor:
        cond = self._encode_obs(obs)
        B = cond.shape[0]

        # Init from Prior N(0, 1)
        x = torch.randn((B, self.flat_dim), device=self.device, dtype=self.dtype)
        dt = 1.0 / self.guidance.steps

        for step in range(self.guidance.steps):
            t_val = step / self.guidance.steps
            x = self._ode_step(x, t_val, dt, cond)

        return x

    def forward(
        self, obs: Any, *, deterministic: bool = False, **kwargs
    ) -> torch.Tensor:
        # Keep normalizer parameters aligned with the policy device for inference.
        self._ensure_normalizer_device()

        # 1. Normalize (Robust)
        if len(self.normalizer.params_dict) > 0:
            if isinstance(obs, dict):
                try:
                    obs = self.normalizer["obs"].normalize(obs)
                except (KeyError, AttributeError):
                    pass
            else:
                try:
                    obs = self.normalizer["obs"].normalize(obs)
                except (KeyError, AttributeError):
                    try:
                        obs = self.normalizer.normalize(obs)
                    except RuntimeError:
                        pass

        if deterministic:
            # Fixed seed logic if needed
            pass

        # 2. Plan (Flow Matching Inference)
        actions_flat = self.plan(obs)

        # 3. Reshape & Unnormalize
        B = actions_flat.shape[0]
        actions = actions_flat.view(B, self.horizons.prediction, self.action_dim)

        # Unnormalize (Robust)
        try:
            if len(self.normalizer.params_dict) > 0:
                if "action" in self.normalizer.params_dict:
                    actions_dict = self.normalizer.unnormalize({"action": actions})
                    actions = actions_dict["action"]
                else:
                    actions = self.normalizer.unnormalize(actions)
        except RuntimeError:
            pass

        return actions

    def compute_loss(self, batch: Any) -> torch.Tensor:
        # Make sure normalizer params (added during fit) follow the policy device.
        self._ensure_normalizer_device()

        # Standardize tuple batches -> dict for normalizer compatibility.
        if isinstance(batch, (list, tuple)):
            batch = {"obs": batch[0], "action": batch[1]}

        # 1. Normalize (Safe Check)
        # Use try-except to catch "Not initialized" error from library normalizer
        try:
            if len(self.normalizer.params_dict) > 0:
                # Handle nested obs dicts by normalizing only the keys we have stats for.
                if (
                    isinstance(batch, dict)
                    and "obs" in batch
                    and isinstance(batch["obs"], dict)
                ):
                    obs_dict = batch["obs"]
                    if (
                        "agent_pos" in obs_dict
                        and "agent_pos" in self.normalizer.params_dict
                    ):
                        obs_dict["agent_pos"] = self.normalizer["agent_pos"].normalize(
                            obs_dict["agent_pos"]
                        )
                    batch["obs"] = obs_dict

                    if "action" in batch and "action" in self.normalizer.params_dict:
                        batch["action"] = self.normalizer["action"].normalize(
                            batch["action"]
                        )
                else:
                    try:
                        batch = self.normalizer.normalize(batch)
                    except KeyError:
                        # Missing fields (e.g., nested obs dict). Skip normalization gracefully.
                        if not getattr(self, "_warned", False):
                            log.warning(
                                "GuidedBFNPolicy: Normalizer missing keys for given batch. Skipping normalization."
                            )
                            self._warned = True
            else:
                if not getattr(self, "_warned", False):
                    log.warning(
                        "GuidedBFNPolicy: Training without normalization! Loss may explode."
                    )
                    self._warned = True
        except RuntimeError:
            # Catch "RuntimeError: Not initialized"
            if not getattr(self, "_warned", False):
                log.warning(
                    "GuidedBFNPolicy: Normalizer raised RuntimeError (Not initialized). Skipping normalization."
                )
                self._warned = True

        obs_t = self._to_tensor(batch["obs"])
        x1 = self._to_tensor(batch["action"])

        # Flatten [B, T, D] -> [B, T*D]
        if x1.ndim == 3:
            x1 = x1.reshape(x1.shape[0], -1)

        cond = self._encode_obs(obs_t)
        B = x1.shape[0]

        # Flow Matching Training
        t_val = torch.rand((B,), device=self.device, dtype=self.dtype)
        x0 = torch.randn_like(x1)

        # Interpolant
        t_exp = t_val.view(B, *([1] * (x1.ndim - 1)))
        x_t = (1 - t_exp) * x0 + t_exp * x1

        pred_x1 = self._predict_x1(x_t, t_val, cond)
        return F.mse_loss(pred_x1, x1)

    def to(self, *args, **kwargs):
        """Moves policy and internal buffers to the specified device."""
        device = kwargs.get("device", None)
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
                break

        if device is not None:
            self._device = torch.device(device)

        # Move all registered submodules (network, normalizer, encoders, etc.)
        super().to(*args, **kwargs)
        if hasattr(self.obs_encoder, "to"):
            self.obs_encoder.to(*args, **kwargs)
        if hasattr(self.normalizer, "to"):
            self.normalizer.to(*args, **kwargs)

        if self._c_uncond_token is not None and device is not None:
            self._c_uncond_token = self._c_uncond_token.to(self._device)

        return self
