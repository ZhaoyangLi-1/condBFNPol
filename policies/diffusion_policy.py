"""Generic Diffusion Policy implementation using Diffusers.

This module implements a Denoising Diffusion Probabilistic Model (DDPM) policy.
It treats policy learning as a conditional generative modeling problem.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from policies.base import BasePolicy

__all__ = ["DiffusionPolicy", "HorizonConfig", "InferenceConfig"]


class HorizonConfig(NamedTuple):
    """Configuration for temporal horizons."""

    planning_horizon: int
    obs_history: int
    action_prediction: int
    execution: int = 1


class InferenceConfig(NamedTuple):
    """Configuration for the diffusion inference process."""

    num_steps: Optional[int] = None
    condition_mode: Literal["local", "global", "inpaint"] = "local"
    pred_action_steps_only: bool = False
    oa_step_convention: bool = False


class DiffusionPolicy(BasePolicy):
    """Diffusion-based policy wrapper.

    Wraps a diffusion model (e.g., conditional U-Net or Transformer) and a
    noise scheduler to perform iterative action generation.
    """

    def __init__(
        self,
        *,
        action_space: Any,
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        action_dim: int,
        obs_dim: Optional[int] = None,
        # Configuration Objects
        horizons: Optional[HorizonConfig] = None,
        inference: Optional[InferenceConfig] = None,
        # Legacy / Direct Arguments
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        condition_mode: str = "local",
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        num_inference_steps: Optional[int] = None,
        # Base Args
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        # --- Configuration Setup ---
        self.horizons = horizons or HorizonConfig(
            planning_horizon=horizon,
            obs_history=n_obs_steps,
            action_prediction=n_action_steps,
            execution=n_action_steps,
        )

        self.inference = inference or InferenceConfig(
            num_steps=num_inference_steps,
            condition_mode=condition_mode,  # type: ignore
            pred_action_steps_only=pred_action_steps_only,
            oa_step_convention=oa_step_convention,
        )

        # --- Components ---
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.kwargs = kwargs

        # Resolve inference steps
        if self.inference.num_steps is None:
            self.inference = self.inference._replace(
                num_steps=noise_scheduler.config.num_train_timesteps
            )

        # --- Mask Generator ---
        extra_obs_dim = (
            0
            if self.inference.condition_mode in ("local", "global")
            else (obs_dim or 0)
        )
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=extra_obs_dim,
            max_n_obs_steps=self.horizons.obs_history,
            fix_obs_steps=True,
            action_visible=False,
        )

    # -------- Condition Builders --------

    def _build_condition(self, obs: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        B, T_obs, D_obs = obs.shape
        T_plan = self.horizons.planning_horizon
        D_act = self.action_dim
        device, dtype = obs.device, obs.dtype

        traj_len = (
            self.horizons.action_prediction
            if self.inference.pred_action_steps_only
            else T_plan
        )

        if self.inference.condition_mode == "inpaint":
            cond_data = torch.zeros(
                (B, traj_len, D_act + D_obs), device=device, dtype=dtype
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :T_obs, D_act:] = obs[:, :T_obs]
            cond_mask[:, :T_obs, D_act:] = True
            return {
                "cond_data": cond_data,
                "cond_mask": cond_mask,
                "local_cond": None,
                "global_cond": None,
            }

        elif self.inference.condition_mode == "local":
            cond_data = torch.zeros((B, traj_len, D_act), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            local_cond = obs[:, :T_obs].clone()
            return {
                "cond_data": cond_data,
                "cond_mask": cond_mask,
                "local_cond": local_cond,
                "global_cond": None,
            }

        elif self.inference.condition_mode == "global":
            cond_data = torch.zeros((B, traj_len, D_act), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            global_cond = obs[:, :T_obs].reshape(B, -1)
            return {
                "cond_data": cond_data,
                "cond_mask": cond_mask,
                "local_cond": None,
                "global_cond": global_cond,
            }

        raise ValueError("Unreachable condition mode state.")

    # -------- Inference --------

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.inference.num_steps)

        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            # Map global_cond to 'cond' if the model expects it (e.g. BFN-style Unet)
            # Standard Diffusion Policy UNet expects global_cond, but our unified Unet uses cond.
            # We pass both to be safe or check model signature?
            # Our robust Unet accepts **kwargs, so passing both is safe.
            # Ideally, we pass 'cond' if using our Unet.

            model_output = self.model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
                cond=global_cond,  # Alias for unified Unet
            )

            step_output = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            )
            trajectory = step_output.prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict

        # 1. Normalize
        if len(self.normalizer.params_dict) > 0:
            nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        else:
            nobs = obs_dict["obs"]

        # Ensure obs is [B, T, D]
        if nobs.ndim == 2:
            nobs = nobs.unsqueeze(1)

        nobs = nobs[:, : self.horizons.obs_history]
        cond_parts = self._build_condition(nobs)

        sample = self.conditional_sample(
            cond_parts["cond_data"],
            cond_parts["cond_mask"],
            local_cond=cond_parts["local_cond"],
            global_cond=cond_parts["global_cond"],
            **self.kwargs,
        )

        naction_pred = sample[..., : self.action_dim]

        # Unnormalize
        if len(self.normalizer.params_dict) > 0:
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
        else:
            action_pred = naction_pred

        if self.inference.pred_action_steps_only:
            action = action_pred
        else:
            start = self.horizons.obs_history
            if self.inference.oa_step_convention:
                start = max(0, start - 1)
            end = start + self.horizons.action_prediction
            action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def forward(
        self, obs: Any, *, deterministic: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        input_obs = obs if isinstance(obs, dict) else {"obs": obs}
        # Ensure tensor
        if isinstance(input_obs["obs"], torch.Tensor):
            input_obs["obs"] = input_obs["obs"].to(self.device, self.dtype)

        result = self.predict_action(input_obs)
        action = result["action"]

        if action.shape[0] == 1:
            action = action[0]
        return action

    def compute_loss(self, batch: Any) -> torch.Tensor:
        # 1. Unpack & Normalize
        if isinstance(batch, (list, tuple)):
            batch = {"obs": batch[0], "action": batch[1]}

        if len(self.normalizer.params_dict) > 0:
            nbatch = self.normalizer.normalize(batch)
        else:
            nbatch = batch

        obs = nbatch["obs"]
        action = nbatch["action"]

        # Fix Obs shape [B, D] -> [B, 1, D]
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        # Fix Action shape [B, T*D] -> [B, T, D] (from DataModule chunking)
        if action.ndim == 2:
            B = action.shape[0]
            T = self.horizons.planning_horizon
            action = action.view(B, T, self.action_dim)

        obs = obs[:, : self.horizons.obs_history]

        cond_parts = self._build_condition(obs)
        trajectory = cond_parts["cond_data"].clone()

        # Fill trajectory with ground truth actions
        T_action = min(action.shape[1], trajectory.shape[1])
        trajectory[:, :T_action, : self.action_dim] = action[:, :T_action]

        if self.inference.pred_action_steps_only:
            loss_mask = torch.ones_like(trajectory, dtype=torch.bool)
        else:
            loss_mask = ~cond_parts["cond_mask"]

        noise = torch.randn(
            trajectory.shape, device=trajectory.device, dtype=trajectory.dtype
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_traj = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        pred = self.model(
            noisy_traj,
            timesteps,
            local_cond=cond_parts["local_cond"],
            global_cond=cond_parts["global_cond"],
            cond=cond_parts["global_cond"],  # Alias for unified Unet
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.flatten(1).mean(dim=1).mean()

        return loss
