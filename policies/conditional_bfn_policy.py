"""Conditional BFN Policy with Integrated MLP Backbone.

This module implements a production-ready Conditional Bayesian Flow Network (BFN)
policy tailored for state-based control tasks. It integrates a robust MLP
backbone directly, handling time embeddings, conditioning, and action chunking
seamlessly.

Key Features:
- Integrated ResNet-MLP Backbone (High Capacity).
- Automatic Action Chunking (Horizon handling).
- Built-in Input/Output Normalization.
- Robust Time Scaling for Sinusoidal Embeddings.
"""

from __future__ import annotations

import logging
import contextlib
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
import einops

from policies.base import BasePolicy
from networks.conditional_bfn import ContinuousBFN
from networks.base import BFNetwork, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb
from utils.bfn_utils import default, exists

# Try local import first, fallback to library
try:
    from networks.normalizer import get_normalizer, LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    # Dummy factory fallback
    def get_normalizer(type_str="linear", **kwargs):
        return LinearNormalizer()


# Try local encoder factory
try:
    from networks.obs_encoders import get_obs_encoder
except ImportError:

    def get_obs_encoder(type_str, **kwargs):
        return None


log = logging.getLogger(__name__)

__all__ = [
    "ConditionalBFNPolicy",
    "BackboneConfig",
    "BFNConfig",
]


# =============================================================================
# Configurations
# =============================================================================


class BackboneConfig(NamedTuple):
    """Configuration for the integrated MLP/ResNet Backbone."""

    hidden_dim: int = 256
    depth: int = 3
    time_emb_dim: int = 128
    cond_drop_prob: float = 0.1
    dropout: float = 0.0
    # Choose between 'film', 'adaln', 'concat'
    conditioning_type: str = "film"


class BFNConfig(NamedTuple):
    """Configuration for BFN Sampling."""

    sigma_1: float = 0.001
    n_timesteps: int = 20
    cond_scale: Optional[float] = None
    rescaled_phi: Optional[float] = None
    deterministic_seed: int = 42


# =============================================================================
# Internal Backbone Components
# =============================================================================


class AdaLNBlock(nn.Module):
    """Residual Block using Adaptive Layer Norm (AdaLN)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.act = nn.Mish()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x) * (1 + scale) + shift
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class FiLMBlock(nn.Module):
    """Residual Block using Feature-wise Linear Modulation (FiLM)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x * (1 + scale) + shift
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class ConcatBlock(nn.Module):
    """Standard Residual Block (Conditioning is concatenated at input)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class IntegratedBFNBackbone(BFNetwork):
    """Robust MLP Backbone with selectable conditioning."""

    def __init__(self, action_dim: int, cond_dim: int, config: BackboneConfig):
        super().__init__(is_conditional_model=True)
        self.action_dim = action_dim
        self.cond_dim = cond_dim
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

        if config.conditioning_type == "concat":
            input_dim = action_dim + config.hidden_dim * 2
        else:
            input_dim = action_dim

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

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(time * 1000.0)

        if cond is None:
            cond = torch.zeros((x.shape[0], self.cond_dim), device=x.device)
        c_emb = self.cond_mlp(cond)

        if self.config.conditioning_type == "concat":
            x_in = torch.cat([x, t_emb, c_emb], dim=-1)
            h = self.input_proj(x_in)
            for block in self.blocks:
                h = block(h)
        else:
            h = self.input_proj(x)
            context = torch.cat([t_emb, c_emb], dim=-1)
            params = self.mod_gen(context)
            params = params.view(
                x.shape[0], self.config.depth, 2, self.config.hidden_dim
            )
            for i, block in enumerate(self.blocks):
                scale, shift = params[:, i, 0], params[:, i, 1]
                h = block(h, scale, shift)

        h = self.output_norm(h)
        out = self.output_proj(h)
        return out


# =============================================================================
# Policy Implementation
# =============================================================================


class ConditionalBFNPolicy(BasePolicy):
    """High-performance BFN Policy with integrated normalization and chunking."""

    def __init__(
        self,
        *,
        action_space: Any,
        action_dim: int,
        obs_dim: int,
        network: Optional[nn.Module] = None,
        n_action_steps: int = 1,
        # Factories
        normalizer_type: str = "linear",
        obs_encoder_type: str = "identity",
        obs_encoder_config: Optional[Dict] = None,
        # Configs
        backbone_config: Optional[BackboneConfig] = None,
        bfn_config: Optional[BFNConfig] = None,
        # Direct Override
        obs_encoder: Optional[
            Callable[[Union[torch.Tensor, Dict]], torch.Tensor]
        ] = None,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_action_steps = n_action_steps
        self.flat_action_dim = action_dim * n_action_steps

        # Defaults
        self.backbone_config = backbone_config or BackboneConfig(
            hidden_dim=kwargs.get("hidden_dims", (256,))[0]
            if "hidden_dims" in kwargs
            else 256,
            depth=4,
            time_emb_dim=128,
            conditioning_type="film",
        )
        self.bfn_config = bfn_config or BFNConfig(
            sigma_1=kwargs.get("sigma_1", 0.005),
            n_timesteps=kwargs.get("n_timesteps", 20),
            cond_scale=kwargs.get("cond_scale", None),
            deterministic_seed=kwargs.get("deterministic_seed", 42),
        )

        # 1. Components (Factory)
        self.normalizer = get_normalizer(normalizer_type)

        # Resolve Encoder
        if obs_encoder is not None:
            self.obs_encoder = obs_encoder
        else:
            enc_kwargs = obs_encoder_config if obs_encoder_config else {}
            self.obs_encoder = get_obs_encoder(obs_encoder_type, **enc_kwargs)

        # 2. Backbone
        self.net = network
        if self.net is None:
            self.net = IntegratedBFNBackbone(
                action_dim=self.flat_action_dim,
                cond_dim=obs_dim,
                config=self.backbone_config,
            )
        elif not hasattr(self.net, "action_dim"):
            try:
                self.net.action_dim = self.flat_action_dim
            except AttributeError:
                pass

        # 3. BFN Wrapper
        self.bfn = ContinuousBFN(
            dim=self.flat_action_dim, net=self.net, device_str=device, dtype_str=dtype
        )

        mode_str = (
            self.backbone_config.conditioning_type if network is None else "Custom"
        )
        log.info(
            f"Init ConditionalBFN ({mode_str}, {normalizer_type}, {obs_encoder_type}): Act={self.flat_action_dim}, Obs={obs_dim}"
        )

    def set_normalizer(self, normalizer: Any) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())
        log.info(f"Normalizer loaded: {list(self.normalizer.params_dict.keys())}")

    def _encode_obs(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in obs.items()
            }
            if self.obs_encoder is not None:
                return self.obs_encoder(obs)
            if "obs" in obs:
                obs = obs["obs"]
            elif "state" in obs:
                obs = obs["state"]
            else:
                obs = list(obs.values())[0]
        obs = obs.to(self.device, dtype=self.dtype)

        # Apply explicit encoder if available
        if self.obs_encoder is not None:
            return self.obs_encoder(obs)

        return obs.view(obs.size(0), -1)

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        *,
        deterministic: bool = False,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> torch.Tensor:
        # Normalize
        if len(self.normalizer.params_dict) > 0:
            try:
                obs = self.normalizer["obs"].normalize(obs)
            except (KeyError, AttributeError):
                obs = self.normalizer.normalize(obs)

        cond = self._encode_obs(obs)
        c_scale = default(cond_scale, self.bfn_config.cond_scale)
        r_phi = default(rescaled_phi, self.bfn_config.rescaled_phi)

        rng_ctx = contextlib.nullcontext()
        if deterministic:
            rng_ctx = (
                torch.random.fork_rng(devices=[self.device])
                if self.device.type == "cuda"
                else torch.random.fork_rng()
            )

        with rng_ctx:
            if deterministic:
                torch.manual_seed(self.bfn_config.deterministic_seed)
            actions_flat = self.bfn.sample(
                n_samples=1,
                sigma_1=self.bfn_config.sigma_1,
                n_timesteps=self.bfn_config.n_timesteps,
                cond=cond,
                cond_scale=c_scale,
                rescaled_phi=r_phi,
            ).squeeze(1)

        B = actions_flat.shape[0]
        actions = actions_flat.view(B, self.n_action_steps, self.action_dim)

        # Unnormalize
        if len(self.normalizer.params_dict) > 0:
            if "action" in self.normalizer.params_dict:
                # Robust dict-based unnormalization
                actions_dict = self.normalizer.unnormalize({"action": actions})
                actions = actions_dict["action"]
            else:
                actions = self.normalizer.unnormalize(actions)

        return actions

    def compute_loss(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            batch = {"obs": batch[0], "action": batch[1]}

        if len(self.normalizer.params_dict) > 0:
            batch = self.normalizer.normalize(batch)
        else:
            if not getattr(self, "_warned", False):
                log.warning("Training without normalization! Loss may explode.")
                self._warned = True

        obs_t = self._to_tensor(batch["obs"])
        actions_t = self._to_tensor(batch["action"])

        if actions_t.ndim == 3:
            B, T, D = actions_t.shape
            actions_t = actions_t.reshape(B, -1)

        cond = self._encode_obs(obs_t)

        loss = self.bfn.loss(
            actions_t,
            cond=cond,
            sigma_1=self.bfn_config.sigma_1,
            cond_scale=self.bfn_config.cond_scale,
            rescaled_phi=self.bfn_config.rescaled_phi,
        )
        return loss.mean()

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        if isinstance(self.obs_encoder, nn.Module):
            self.obs_encoder.to(*args, **kwargs)

        device = kwargs.get("device", None)
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
                break
        if device:
            self._device = torch.device(device)
        return self
