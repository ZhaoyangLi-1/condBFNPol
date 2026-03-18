"""Streaming Flow hybrid image policy aligned with original PushT SFP."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from policies.streaming_flow_policy import StreamingFlowPolicy

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

log = logging.getLogger(__name__)

__all__ = ["StreamingFlowUnetHybridImagePolicy"]


class SinusoidalPosEmb(nn.Module):
    """Original SFP timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Linear1d(nn.Module):
    """Original SFP fully-connected temporal mixing used for 2-token states."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, timesteps = x.size()
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        return x.reshape(batch_size, channels, timesteps)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class SFPConditionalUnet1D(nn.Module):
    """Original streaming-flow-policy U-Net with optional fc_timesteps mixing."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        fc_timesteps: Optional[int] = None,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
        ])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            downsample = (
                Downsample1d(dim_out)
                if fc_timesteps is None
                else Linear1d(fc_timesteps * dim_out)
            )
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                downsample if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            upsample = (
                Upsample1d(dim_in)
                if fc_timesteps is None
                else Linear1d(fc_timesteps * dim_in)
            )
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                upsample if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        skip_connections = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skip_connections.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class SimpleObsEncoder(nn.Module):
    """Fallback encoder used only when robomimic is unavailable."""

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

    def output_shape(self):
        return (self.output_dim,)


class StreamingFlowHybridVectorField(nn.Module):
    """torchdyn-compatible vector field for the PushT stochastic SFP state."""

    def __init__(self, velocity_net: nn.Module, obs_cond: torch.Tensor):
        super().__init__()
        self.velocity_net = velocity_net
        self.obs_cond = obs_cond

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.numel() == 1:
            t = t.reshape(1).expand(batch_size)

        state = x.reshape(batch_size, 2, -1)
        velocity = self.velocity_net(
            sample=state,
            timestep=t,
            global_cond=self.obs_cond,
        )
        return velocity.reshape(batch_size, -1)


class StreamingFlowUnetHybridImagePolicy(StreamingFlowPolicy):
    """Streaming Flow policy that follows the original PushT stochastic SFP."""

    def __init__(
        self,
        *,
        shape_meta: Dict[str, Any],
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        sigma0: float = 0.0,
        sigma1: float = 0.1,
        k: float = 0.0,
        num_integration_steps: int = 100,
        crop_shape: Optional[tuple[int, int]] = None,
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ):
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Streaming flow hybrid policy expects 1D actions."
        action_dim = action_shape[0]

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

        velocity_net = SFPConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            fc_timesteps=2,
        )

        class DummyActionSpace:
            def __init__(self, shape):
                self.shape = shape

        super().__init__(
            action_space=DummyActionSpace(action_shape),
            velocity_net=velocity_net,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            sigma0=sigma0,
            sigma1=sigma1,
            k=k,
            num_integration_steps=num_integration_steps,
            device=device,
            **kwargs,
        )

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.obs_feature_dim = obs_feature_dim
        self.crop_shape = crop_shape
        self.normalizer = None
        self.action_init_key = self._infer_action_init_key(obs_shape_meta, action_dim)
        self._cached_obs_cond: Optional[torch.Tensor] = None

        if self.k != 0.0:
            log.warning(
                "PushT stochastic SFP ignores k, but config provided k=%s. "
                "The original algorithm does not use the stabilized schedule.",
                self.k,
            )

        log.info(
            "StreamingFlowUnetHybridImagePolicy initialized with obs_feature_dim=%s, "
            "action_dim=%s, action_init_key=%s",
            obs_feature_dim,
            action_dim,
            self.action_init_key,
        )

    def _build_robomimic_encoder(
        self,
        obs_config: Dict[str, Any],
        obs_key_shapes: Dict[str, Any],
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
                    num_groups=x.num_features // 16,
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
        elif eval_fixed_crop:
            log.warning(
                "eval_fixed_crop=True but robomimic CropRandomizer was not found; skipping replacement."
            )

        return obs_encoder

    @staticmethod
    def _infer_action_init_key(obs_shape_meta: Dict[str, Any], action_dim: int) -> Optional[str]:
        if "agent_pos" in obs_shape_meta:
            agent_shape = obs_shape_meta["agent_pos"]["shape"]
            if len(agent_shape) == 1 and agent_shape[0] == action_dim:
                return "agent_pos"

        candidates = []
        for key, attr in obs_shape_meta.items():
            if attr.get("type", "low_dim") != "low_dim":
                continue
            shape = attr["shape"]
            if len(shape) == 1 and shape[0] == action_dim:
                candidates.append(key)

        if len(candidates) == 1:
            return candidates[0]
        return None

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.obs_normalizer is not None and hasattr(self.obs_normalizer, "params_dict"):
            valid_keys = set(getattr(self.obs_normalizer, "params_dict", {}).keys())
            if valid_keys and valid_keys.issuperset(obs_dict.keys()):
                obs_dict = self.obs_normalizer.normalize(obs_dict)

        batch_size = next(iter(obs_dict.values())).shape[0]
        this_obs = dict_apply(
            obs_dict,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_obs)
        obs_cond = obs_features.reshape(batch_size, -1)
        self._cached_obs_cond = obs_cond
        return obs_cond

    def _get_current_action(self, obs_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.action_init_key is None or self.action_init_key not in obs_dict:
            return None
        return obs_dict[self.action_init_key][:, self.n_obs_steps - 1]

    def _normalize_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_normalizer is None:
            return action
        return self.action_normalizer.normalize(action)

    def _unnormalize_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_normalizer is None:
            return action
        return self.action_normalizer.unnormalize(action)

    def _match_first_action_to_observation(
        self,
        actions: torch.Tensor,
        current_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if current_action is None:
            return actions

        current_action = current_action.to(device=actions.device, dtype=actions.dtype)

        matched_actions = actions.clone()
        mismatch = ~torch.isclose(
            matched_actions[:, 0],
            current_action,
            atol=1e-6,
            rtol=0.0,
        ).all(dim=-1)
        if mismatch.any():
            matched_actions[mismatch, 0] = current_action[mismatch]
        return matched_actions

    def _interpolate_trajectory(
        self,
        actions: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.horizon == 1:
            xi_t = actions[:, 0]
            xi_dot_t = torch.zeros_like(xi_t)
            return xi_t, xi_dot_t

        scaled_time = time.clamp(0.0, 1.0) * (self.horizon - 1)
        idx0 = torch.floor(scaled_time).long().clamp(0, self.horizon - 2)
        idx1 = idx0 + 1
        alpha = (scaled_time - idx0.to(time.dtype)).view(-1, 1)

        gather_shape = (-1, 1, self.action_dim)
        left = actions.gather(
            dim=1,
            index=idx0.view(-1, 1, 1).expand(*gather_shape),
        ).squeeze(1)
        right = actions.gather(
            dim=1,
            index=idx1.view(-1, 1, 1).expand(*gather_shape),
        ).squeeze(1)

        xi_t = (1.0 - alpha) * left + alpha * right
        xi_dot_t = (right - left) * (self.horizon - 1)
        return xi_t, xi_dot_t

    def _integration_steps_per_action(self) -> int:
        if self.horizon <= 1:
            return 1
        return max(self.num_integration_steps // (self.horizon - 1), 1)

    def _default_obs_cond(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._cached_obs_cond is not None and self._cached_obs_cond.shape[0] == batch_size:
            return self._cached_obs_cond.to(device=device, dtype=dtype)
        return torch.zeros(
            batch_size,
            self.obs_feature_dim * self.n_obs_steps,
            device=device,
            dtype=dtype,
        )

    def _make_neural_ode(self, obs_cond: Optional[torch.Tensor] = None) -> NeuralODE:
        if obs_cond is None:
            obs_cond = self._default_obs_cond(
                batch_size=1,
                device=self.device,
                dtype=self.dtype,
            )
        return NeuralODE(
            vector_field=StreamingFlowHybridVectorField(self.velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

    def _velocity_wrapper(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        obs_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        squeeze_batch = x.dim() == 1
        if squeeze_batch:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        state_dim = 2 * self.action_dim
        full_dim = x.shape[1]
        state = x[:, :state_dim]

        if obs_cond is None:
            if full_dim > state_dim:
                tail_dim = self.obs_feature_dim * self.n_obs_steps
                if full_dim >= state_dim + tail_dim:
                    obs_cond = x[:, -tail_dim:]
                else:
                    obs_cond = self._default_obs_cond(
                        batch_size=batch_size,
                        device=x.device,
                        dtype=x.dtype,
                    )
            else:
                obs_cond = self._default_obs_cond(
                    batch_size=batch_size,
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            obs_cond = obs_cond.to(device=x.device, dtype=x.dtype)

        velocity = StreamingFlowHybridVectorField(self.velocity_net, obs_cond)(t, state)
        if full_dim > state_dim:
            padded_velocity = torch.zeros_like(x)
            padded_velocity[:, :state_dim] = velocity
            velocity = padded_velocity

        if squeeze_batch:
            velocity = velocity.squeeze(0)
        return velocity

    @property
    def neural_ode(self) -> NeuralODE:
        return self._make_neural_ode()

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_cond = self._encode_obs(obs_dict)
        batch_size = obs_cond.shape[0]
        dtype = obs_cond.dtype
        device = obs_cond.device

        current_action = self._get_current_action(obs_dict)
        if current_action is None:
            a0 = torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
        else:
            a0 = self._normalize_action_tensor(current_action.to(device=device, dtype=dtype))

        z0 = torch.randn_like(a0)
        x0 = torch.cat([a0, z0], dim=-1)

        num_actions = self.horizon
        integration_steps_per_action = self._integration_steps_per_action()
        num_future_actions = num_actions - 1
        if self.horizon <= 1:
            t_max = 0.0
        else:
            t_max = num_future_actions / (self.horizon - 1)

        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(
            0.0,
            t_max,
            total_integration_steps,
            device=device,
            dtype=dtype,
        )
        select_indices = torch.arange(
            0,
            total_integration_steps,
            integration_steps_per_action,
            device=device,
        )

        ode_solver = self._make_neural_ode(obs_cond)
        ode_result = ode_solver(x0, t_span)
        trajectory = ode_result[1] if isinstance(ode_result, tuple) else ode_result

        naction_pred = trajectory.index_select(0, select_indices)[:, :, : self.action_dim]
        naction_pred = naction_pred.transpose(0, 1)
        action_pred = self._unnormalize_action_tensor(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict = batch["obs"]
        raw_actions = batch["action"]
        current_action = self._get_current_action(obs_dict)
        raw_actions = self._match_first_action_to_observation(raw_actions, current_action)

        nactions = self._normalize_action_tensor(raw_actions)
        obs_cond = self._encode_obs(obs_dict)

        batch_size = nactions.shape[0]
        dtype = nactions.dtype
        device = nactions.device
        time = torch.rand(batch_size, device=device, dtype=dtype)

        xi_t, xi_dot_t = self._interpolate_trajectory(nactions, time)
        z0 = torch.randn_like(xi_t)
        eps_a0 = self.sigma0 * torch.randn_like(xi_t)
        time_expanded = time.view(-1, 1)

        a_t = xi_t + eps_a0 + self.sigma_r * time_expanded * z0
        z_t = (1.0 - (1.0 - self.sigma1) * time_expanded) * z0 + time_expanded * xi_t

        v_a = xi_dot_t + self.sigma_r * z0
        v_z = xi_t + time_expanded * xi_dot_t - (1.0 - self.sigma1) * z0

        state = torch.stack([a_t, z_t], dim=1)
        velocity_target = torch.stack([v_a, v_z], dim=1)
        velocity_pred = self.velocity_net(
            sample=state,
            timestep=time,
            global_cond=obs_cond,
        )

        return torch.nn.functional.mse_loss(velocity_pred, velocity_target)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del deterministic, kwargs
        return self.predict_action(obs)["action"]

    def get_params(self):
        return list(self.velocity_net.parameters()) + list(self.obs_encoder.parameters())

    # ==================== Normalizer (BFN-compatible interface) ====================

    def set_normalizer(self, normalizer):
        """Set normalizer matching the BFN workspace interface.

        ``normalizer`` is a :class:`LinearNormalizer` whose keys correspond to
        the dataset fields (``'action'``, per-obs-key, …).  We store the full
        normalizer and wire up the ``action_normalizer`` / ``obs_normalizer``
        shortcuts used by the rest of the policy.
        """
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        if not hasattr(self, "normalizer") or self.normalizer is None:
            self.normalizer = LinearNormalizer()

        self.normalizer.load_state_dict(normalizer.state_dict())

        # Wire up convenience handles used by _normalize_action_tensor etc.
        self.action_normalizer = self.normalizer["action"]
        self.obs_normalizer = self.normalizer

    # ==================== State Dict ====================

    def state_dict(self):
        sd = {
            "obs_encoder": self.obs_encoder.state_dict(),
            "velocity_net": self.velocity_net.state_dict(),
        }
        if hasattr(self, "normalizer") and self.normalizer is not None:
            sd["normalizer"] = self.normalizer.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        self.obs_encoder.load_state_dict(state_dict["obs_encoder"])
        self.velocity_net.load_state_dict(state_dict["velocity_net"])
        if "normalizer" in state_dict:
            from diffusion_policy.model.common.normalizer import LinearNormalizer

            if not hasattr(self, "normalizer") or self.normalizer is None:
                self.normalizer = LinearNormalizer()
            self.normalizer.load_state_dict(state_dict["normalizer"])
            self.action_normalizer = self.normalizer["action"]
            self.obs_normalizer = self.normalizer

    def reset(self):
        pass
