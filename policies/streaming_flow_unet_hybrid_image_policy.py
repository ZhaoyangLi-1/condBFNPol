"""Streaming Flow policy integrated into condBFNPol.

This implementation follows the public streaming-flow-policy PushT algorithm:
training supervises the conditional velocity field at a random flow time, and
inference integrates that learned vector field with a Neural ODE to produce an
action trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer

from networks.streaming_flow_unet import StreamingFlowConditionalUnet1D
from policies.base import BasePolicy

try:
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.models.base_nets as rm_base_nets

    try:
        import robomimic.models.obs_core as rm_obs_core
    except Exception:
        rm_obs_core = None

    RM_CropRandomizer = getattr(rm_base_nets, "CropRandomizer", None)
    if RM_CropRandomizer is None and rm_obs_core is not None:
        RM_CropRandomizer = getattr(rm_obs_core, "CropRandomizer", None)

    import diffusion_policy.model.vision.crop_randomizer as dmvc

    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False
    RM_CropRandomizer = None

__all__ = ["StreamingFlowUnetHybridImagePolicy"]


class SimpleObsEncoder(nn.Module):
    """Fallback observation encoder used when robomimic is unavailable."""

    def __init__(self, obs_shape_meta: Dict[str, Any], output_dim: int = 512):
        super().__init__()
        input_dim = 0
        for attr in obs_shape_meta.values():
            input_dim += int(np.prod(attr["shape"]))

        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [obs_dict[key].flatten(start_dim=1) for key in sorted(obs_dict.keys())]
        return self.net(torch.cat(parts, dim=1))

    def output_shape(self) -> tuple[int]:
        return (self.output_dim,)


class StreamingFlowVectorField(nn.Module):
    """torchdyn-compatible vector field wrapper.

    Matches the original VectorFieldWrapper from streaming-flow-policy.
    Accepts x with shape ``(state_tokens, action_dim)`` (single sample) or
    ``(B, state_tokens, action_dim)`` (batched).
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        obs_cond: torch.Tensor,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.obs_cond = obs_cond

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x: (state_tokens, action_dim) or (B, state_tokens, action_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.numel() == 1:
            t = t.reshape(1).expand(batch_size)

        velocity = self.velocity_net(
            sample=x,
            timestep=t,
            global_cond=self.obs_cond,
        )

        if squeeze:
            velocity = velocity.squeeze(0)
        return velocity


class StreamingFlowUnetHybridImagePolicy(BasePolicy):
    """Image-conditioned Streaming Flow policy with unified train/eval interface."""

    RUNTIME_INIT_ACTION_KEY = "__streaming_flow_init_action__"

    def __init__(
        self,
        *,
        shape_meta: Dict[str, Any],
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        sigma0: float = 0.0,
        sigma1: float = 0.1,
        flow_mode: str = "stochastic",
        integration_steps_per_action: int = 6,
        initial_action_mode: str = "auto",
        initial_action_keys: Optional[Sequence[str]] = None,
        crop_shape: Optional[tuple[int, int]] = None,
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        ode_solver: str = "dopri5",
        ode_atol: float = 1e-4,
        ode_rtol: float = 1e-4,
        device: Union[torch.device, str] = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        action_representation: str = "unspecified",
        k: float = 0.0,
        **kwargs,
    ):
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Streaming Flow currently expects 1D action vectors."
        action_dim = action_shape[0]

        super().__init__(
            action_space=None,
            device=str(device),
            dtype=dtype,
            clip_actions=clip_actions,
        )

        if flow_mode not in {"stochastic", "deterministic"}:
            raise ValueError(f"Unsupported flow_mode={flow_mode!r}.")
        if flow_mode == "stochastic" and sigma0 > sigma1:
            raise ValueError("Stochastic Streaming Flow requires sigma0 <= sigma1.")
        if initial_action_mode not in {"auto", "obs", "zeros"}:
            raise ValueError(f"Unsupported initial_action_mode={initial_action_mode!r}.")
        if action_representation not in {"unspecified", "delta", "absolute_xy"}:
            raise ValueError(
                "Unsupported action_representation="
                f"{action_representation!r}."
            )

        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = {}
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        if HAS_ROBOMIMIC:
            obs_encoder = self._build_robomimic_encoder(
                obs_config=obs_config,
                obs_key_shapes=obs_key_shapes,
                action_dim=action_dim,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
            obs_feature_dim = obs_encoder.output_shape()[0]
        else:
            obs_encoder = SimpleObsEncoder(obs_shape_meta)
            obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.model = StreamingFlowConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            fc_timesteps=2 if flow_mode == "stochastic" else 1,
        )
        self.velocity_net = self.model
        self.normalizer = LinearNormalizer()

        self.shape_meta = shape_meta
        self.obs_shape_meta = obs_shape_meta
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.flow_mode = flow_mode
        self.state_tokens = 2 if flow_mode == "stochastic" else 1
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.sigma_r = math.sqrt(max(self.sigma1**2 - self.sigma0**2, 0.0))
        # Backward compat: old checkpoints store num_integration_steps (total),
        # convert to per-action equivalent.
        if 'num_integration_steps' in kwargs:
            legacy_val = int(kwargs.pop('num_integration_steps'))
            integration_steps_per_action = max(1, legacy_val // max(horizon - 1, 1))
        self.integration_steps_per_action = int(integration_steps_per_action)
        self.initial_action_mode = initial_action_mode
        self.initial_action_keys = list(initial_action_keys) if initial_action_keys else None
        self.ode_solver = ode_solver
        self.ode_atol = float(ode_atol)
        self.ode_rtol = float(ode_rtol)
        self.obs_feature_dim = obs_feature_dim
        self.action_representation = action_representation
        self.k = float(k)
        self.kwargs = kwargs

    @classmethod
    def get_runtime_init_action_key(cls) -> str:
        return cls.RUNTIME_INIT_ACTION_KEY

    def _build_robomimic_encoder(
        self,
        obs_config: Dict[str, List[str]],
        obs_key_shapes: Dict[str, List[int]],
        action_dim: int,
        crop_shape: Optional[tuple[int, int]],
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> nn.Module:
        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                crop_h, crop_w = crop_shape
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = crop_h
                        modality.obs_randomizer_kwargs.crop_width = crop_w

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )
        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop and RM_CropRandomizer is not None:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, RM_CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=getattr(x, "num_crops", 1),
                    pos_enc=getattr(x, "pos_enc", False),
                ),
            )

        return obs_encoder

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            obs_dict = {"obs": obs}
        else:
            obs_dict = obs
        return self.predict_action(obs_dict)["action"]

    def _encode_observation_condition(
        self,
        nobs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, -1)

    def _resolve_initial_action_keys(self) -> Optional[List[str]]:
        if self.initial_action_keys is not None:
            return self.initial_action_keys

        matching_lowdim = []
        for key, attr in self.obs_shape_meta.items():
            if attr.get("type", "low_dim") != "low_dim":
                continue
            if int(np.prod(attr["shape"])) == self.action_dim:
                matching_lowdim.append(key)
        matching_lowdim.sort()
        return matching_lowdim[:1] if matching_lowdim else None

    def _extract_initial_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).shape[0]
        runtime_init_action = obs_dict.get(self.RUNTIME_INIT_ACTION_KEY, None)
        if runtime_init_action is not None:
            if runtime_init_action.dim() == 1:
                runtime_init_action = runtime_init_action.unsqueeze(0)
            elif runtime_init_action.dim() >= 3:
                runtime_init_action = runtime_init_action.reshape(
                    runtime_init_action.shape[0], -1, runtime_init_action.shape[-1]
                )[:, -1, :]
            runtime_init_action = runtime_init_action.to(device=device, dtype=dtype)
            if runtime_init_action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Runtime init action dim mismatch: expected {self.action_dim}, "
                    f"got {runtime_init_action.shape[-1]}."
                )
            return self.normalizer["action"].normalize(runtime_init_action)

        keys = self._resolve_initial_action_keys()

        if keys:
            parts = []
            for key in keys:
                if key not in obs_dict:
                    if self.initial_action_mode == "auto":
                        keys = None
                        parts = []
                        break
                    raise KeyError(f"Initial action key {key!r} missing from observations.")
                value = obs_dict[key][:, self.n_obs_steps - 1, ...]
                parts.append(value.reshape(batch_size, -1))

            if parts:
                raw_init_action = torch.cat(parts, dim=-1).to(device=device, dtype=dtype)
                if raw_init_action.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"Initial action dim mismatch: expected {self.action_dim}, "
                        f"got {raw_init_action.shape[-1]} from keys {keys}."
                    )
                return self.normalizer["action"].normalize(raw_init_action)

        if self.initial_action_mode == "obs":
            raise ValueError(
                "initial_action_mode='obs' requires low-dim observation keys that map to action_dim. "
                f"Configured keys: {self.initial_action_keys!r}"
            )

        return torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)

    def _interpolate_action_trajectory(
        self,
        action_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, action_dim = action_seq.shape
        if horizon == 1:
            xi_t = action_seq[:, 0]
            xi_dot_t = torch.zeros_like(xi_t)
            return xi_t, xi_dot_t

        scaled_t = t * float(horizon - 1)
        idx0 = scaled_t.floor().long().clamp(min=0, max=horizon - 2)
        idx1 = idx0 + 1
        alpha = (scaled_t - idx0.to(dtype=scaled_t.dtype)).unsqueeze(-1)
        batch_idx = torch.arange(batch_size, device=action_seq.device)

        x0 = action_seq[batch_idx, idx0]
        x1 = action_seq[batch_idx, idx1]
        xi_t = x0 + alpha * (x1 - x0)
        xi_dot_t = (x1 - x0) * float(horizon - 1)
        assert xi_t.shape == (batch_size, action_dim)
        return xi_t, xi_dot_t

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])

        cond = self._encode_observation_condition(nobs)
        batch_size = naction.shape[0]
        device = naction.device
        dtype = naction.dtype

        t = torch.rand(batch_size, device=device, dtype=dtype)
        xi_t, xi_dot_t = self._interpolate_action_trajectory(naction, t)

        if self.flow_mode == "stochastic":
            z0 = torch.randn_like(xi_t)
            epsilon_a0 = self.sigma0 * torch.randn_like(xi_t)
            t_col = t.unsqueeze(-1)

            a_t = xi_t + epsilon_a0 + self.sigma_r * t_col * z0
            z_t = (1.0 - (1.0 - self.sigma1) * t_col) * z0 + t_col * xi_t
            v_a = xi_dot_t + self.sigma_r * z0
            v_z = xi_t + t_col * xi_dot_t - (1.0 - self.sigma1) * z0

            state = torch.stack((a_t, z_t), dim=1)
            target = torch.stack((v_a, v_z), dim=1)
        else:
            x_t = xi_t + self.sigma0 * torch.randn_like(xi_t)
            state = x_t.unsqueeze(1)
            target = xi_dot_t.unsqueeze(1)

        pred = self.velocity_net(
            sample=state,
            timestep=t,
            global_cond=cond,
        )
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def _integrate_normalized_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        device = cond.device
        dtype = cond.dtype
        init_action = self._extract_initial_action(obs_dict, device=device, dtype=dtype)

        if self.flow_mode == "stochastic":
            init_latent = torch.randn_like(init_action)
            # Stack as (B, 2, ACTION_DIM) — matches original torch.stack((a0, z0), dim=0)
            x0 = torch.stack((init_action, init_latent), dim=1)
        else:
            x0 = init_action.unsqueeze(1)  # (B, 1, ACTION_DIM)

        # Match original streaming-flow-policy: integration_steps_per_action
        num_future_actions = self.horizon - 1
        total_steps = num_future_actions * self.integration_steps_per_action
        total_integration_steps = 1 + total_steps
        t_span = torch.linspace(0.0, 1.0, total_integration_steps, device=device, dtype=dtype)
        select_idx = torch.arange(
            0, total_integration_steps, self.integration_steps_per_action, device=device
        )

        vector_field = StreamingFlowVectorField(
            velocity_net=self.velocity_net,
            obs_cond=cond,
        )
        ode_solver = NeuralODE(
            vector_field=vector_field,
            solver=self.ode_solver,
            sensitivity="adjoint",
            atol=self.ode_atol,
            rtol=self.ode_rtol,
        )
        # trajectory: (total_integration_steps, B, state_tokens, ACTION_DIM)
        trajectory = ode_solver.trajectory(x=x0, t_span=t_span)

        # Select action keypoints; index 0 on state_tokens dim = action component
        action_traj = trajectory.index_select(0, select_idx)[:, :, 0, :]
        # (num_points, B, ACTION_DIM) → (B, num_points, ACTION_DIM)
        return action_traj.permute(1, 0, 2).contiguous()

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_inputs = {
            key: value
            for key, value in obs_dict.items()
            if key != self.RUNTIME_INIT_ACTION_KEY
        }
        nobs = self.normalizer.normalize(obs_inputs)
        cond = self._encode_observation_condition(nobs)

        naction_full = self._integrate_normalized_action(obs_dict, cond)
        action_pred = self.normalizer["action"].unnormalize(naction_full)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def reset(self):
        pass

    def set_actions(self, action: torch.Tensor):
        del action

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return {
            "obs_encoder": self.obs_encoder.state_dict(),
            "model": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        obs_state = state_dict.get("obs_encoder")
        model_state = state_dict.get("model")
        norm_state = state_dict.get("normalizer")

        if obs_state is not None and model_state is not None:
            self.obs_encoder.load_state_dict(obs_state, strict=strict)
            self.model.load_state_dict(model_state, strict=strict)
            if norm_state is not None:
                self.normalizer.load_state_dict(norm_state, strict=strict)
            return

        return super().load_state_dict(state_dict, strict=strict)
