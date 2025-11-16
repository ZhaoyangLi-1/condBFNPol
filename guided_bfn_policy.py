"""Guided Bayesian Flow Network Policy for visuomotor planning."""

from __future__ import annotations

import collections
import contextlib
from typing import Optional

import torch as t


class GuidedBFNPolicy:
    """
    Guided Bayesian Flow Network Policy for visuomotor trajectory planning,
    synthesized from diffusion-style conditional guidance and gradient guidance.
    """

    def __init__(
        self,
        *,
        backbone_transformer: t.nn.Module,  # BFN network Psi
        vision_encoder: t.nn.Module,  # e.g., ResNet producing features
        action_dim: int,
        # BFN parameters
        bfn_timesteps: int = 100,
        # Receding horizon parameters
        obs_horizon_T_o: int = 2,
        action_pred_horizon_T_p: int = 16,
        action_exec_horizon_T_a: int = 8,
        # Guidance parameters
        cfg_uncond_prob: float = 0.1,
        cfg_guidance_scale_w: float = 7.5,
        grad_guidance_scale_alpha: float = 0.01,
        device: str = "cpu",
    ):
        self.network = backbone_transformer  # Psi
        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        self.device = t.device(device)

        # Horizons
        self.T_o = obs_horizon_T_o
        self.T_p = action_pred_horizon_T_p
        self.T_a = action_exec_horizon_T_a

        # BFN steps
        self.N_steps = bfn_timesteps

        # Guidance params
        self.p_uncond = cfg_uncond_prob
        self.w = cfg_guidance_scale_w
        self.alpha = grad_guidance_scale_alpha

        # State buffer for RHC
        self.obs_buffer = collections.deque(maxlen=self.T_o)

        # Null token for unconditional pass (simple zero embedding)
        out_dim = getattr(self.vision_encoder, "output_dim", None)
        if out_dim is None:
            raise ValueError("vision_encoder must expose output_dim attribute")
        self.c_uncond_token = t.zeros(1, self.T_o, out_dim, device=self.device)

    # --- BFN helper functions ---

    def bfn_get_sender_receiver(self, x: t.Tensor, t_frac: t.Tensor):
        """Returns noisy sender sample and loss weight (simplified schedule)."""
        sigma_t = 0.01**t_frac
        noise = t.randn_like(x)
        y_noisy = x * sigma_t + noise * (1 - sigma_t**2).sqrt()
        loss_weight = 1.0 / (0.01 ** (2 * t_frac))
        return y_noisy, loss_weight

    def bfn_ode_step(self, A_t_plus_dt: t.Tensor, A_guided: t.Tensor, dt: float):
        """Simple Euler update toward guided prediction."""
        return A_t_plus_dt + (A_guided - A_t_plus_dt) * dt

    def compute_trajectory_cost(
        self, action_sequence: t.Tensor, goal_state: t.Tensor
    ) -> t.Tensor:
        """Differentiable cost over an action sequence (L2 to goal on final action)."""
        predicted_final_action = action_sequence[:, -1, :]
        return t.linalg.norm(predicted_final_action - goal_state)

    # --- Main planning interface ---

    def _encode_obs(self) -> t.Tensor:
        """Encode the observation buffer to conditioning embeddings."""
        if len(self.obs_buffer) == 0:
            raise ValueError("obs_buffer is empty; call act with observations first.")
        obs_list = list(self.obs_buffer)
        # Pad if buffer not full
        while len(obs_list) < self.T_o:
            obs_list.insert(0, obs_list[0])
        obs_batch = (
            t.stack(obs_list, dim=0).unsqueeze(0).to(self.device)
        )  # [1, T_o, ...]
        # Flatten batch/time for encoder, then reshape back if encoder expects batch
        # Assumes encoder operates on (B*T_o, C, H, W) or (B*T_o, D)
        b, t_h = obs_batch.shape[0], obs_batch.shape[1]
        flat = obs_batch.view(b * t_h, *obs_batch.shape[2:])
        feats = self.vision_encoder(flat)
        feats = feats.view(b, t_h, -1)
        return feats

    def _guided_prediction(
        self, A: t.Tensor, cond: Optional[t.Tensor], t_frac: float
    ) -> t.Tensor:
        """Forward through the network with optional classifier-free and gradient guidance."""
        t_tensor = t.full((A.size(0),), t_frac, device=A.device, dtype=A.dtype)

        # Conditional forward
        if hasattr(self.network, "forward_with_cond_scale") and cond is not None:
            pred = self.network.forward_with_cond_scale(
                A, t_tensor, cond, cond_scale=self.w
            )
        else:
            pred = self.network(A, t_tensor, cond)

        return pred

    def plan(self, obs: t.Tensor, goal_state: Optional[t.Tensor] = None) -> t.Tensor:
        """Plan an action sequence given current observation and optional goal."""
        self.obs_buffer.append(obs.to(self.device))
        cond = self._encode_obs() if len(self.obs_buffer) > 0 else None

        A = t.randn((1, self.T_p, self.action_dim), device=self.device)
        dt = 1.0 / max(self.N_steps, 1)

        for step in range(self.N_steps, 0, -1):
            t_frac = step / self.N_steps

            # Conditional prediction
            pred = self._guided_prediction(A, cond, t_frac)

            # Gradient guidance toward goal
            if goal_state is not None and self.alpha > 0.0:
                pred.requires_grad_(True)
                cost = self.compute_trajectory_cost(pred, goal_state.to(self.device))
                cost.backward()
                grad = pred.grad if pred.grad is not None else 0.0
                guided = pred - self.alpha * grad
                pred = guided.detach()

            A = self.bfn_ode_step(A, pred, dt)

        return A.squeeze(0)  # [T_p, action_dim]

    def act(self, obs: t.Tensor, goal_state: Optional[t.Tensor] = None) -> t.Tensor:
        """Return the first action from a planned trajectory."""
        traj = self.plan(obs, goal_state)
        return traj[: self.T_a]

    def to(self, *args, **kwargs):
        """Move internal modules/tensors to device/dtype."""
        self.network.to(*args, **kwargs)
        self.vision_encoder.to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is not None:
            self.device = t.device(device)
            self.c_uncond_token = self.c_uncond_token.to(self.device)
        return self

    def state_dict(self):
        return {
            "network": self.network.state_dict(),
            "vision_encoder": self.vision_encoder.state_dict(),
            "c_uncond_token": self.c_uncond_token,
            "action_dim": self.action_dim,
            "T_o": self.T_o,
            "T_p": self.T_p,
            "T_a": self.T_a,
            "N_steps": self.N_steps,
            "w": self.w,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.vision_encoder.load_state_dict(state["vision_encoder"])
        if "c_uncond_token" in state:
            self.c_uncond_token = state["c_uncond_token"].to(self.device)
        # hyperparams not strictly required to load weights
        self.action_dim = state.get("action_dim", self.action_dim)
        self.T_o = state.get("T_o", self.T_o)
        self.T_p = state.get("T_p", self.T_p)
        self.T_a = state.get("T_a", self.T_a)
        self.N_steps = state.get("N_steps", self.N_steps)
        self.w = state.get("w", self.w)
        self.alpha = state.get("alpha", self.alpha)
        return self

    def eval(self):
        self.network.eval()
        self.vision_encoder.eval()
        return self

    def train(self, mode: bool = True):
        self.network.train(mode)
        self.vision_encoder.train(mode)
        return self

    @t.no_grad()
    def predict_action_sequence(
        self,
        current_obs_sequence: t.Tensor,
        goal_state: t.Tensor,
    ) -> t.Tensor:
        """
        Generate an action sequence using dual guidance (CFG + gradient) as in Guided-BFNs.

        Args:
            current_obs_sequence: tensor of observations over the horizon, expected shape
                compatible with the vision_encoder. Must include batch dimension.
            goal_state: tensor representing the goal for gradient guidance.
        """
        B = current_obs_sequence.shape[0]

        # Encode conditioning and unconditional tokens
        c_cond = self.vision_encoder(current_obs_sequence.to(self.device))
        c_uncond = self.c_uncond_token.to(c_cond.device)
        if c_uncond.shape[1] != c_cond.shape[1]:
            # tile/resize unconditional token to match time dims
            c_uncond = c_uncond.expand(B, c_cond.shape[1], -1)

        # Start from noise
        A_t = t.randn(B, self.T_p, self.action_dim, device=c_cond.device)

        time_steps = t.linspace(0.0, 1.0, self.N_steps + 1, device=c_cond.device)

        for i in range(self.N_steps):
            t_curr = time_steps[i]
            t_next = time_steps[i + 1]
            dt = (t_next - t_curr).item()
            t_tensor = t.full((B,), t_curr, device=c_cond.device)

            # Stage 1: CFG
            pred_uncond = self.network(A_t, t_tensor, c_uncond)
            pred_cond = self.network(A_t, t_tensor, c_cond)
            A_pred_cfg = pred_uncond + self.w * (pred_cond - pred_uncond)

            # Stage 2: Gradient guidance
            if self.alpha > 0.0:
                with t.enable_grad():
                    A_pred_grad = A_pred_cfg.clone().requires_grad_(True)
                    cost = self.compute_trajectory_cost(
                        A_pred_grad, goal_state.to(c_cond.device)
                    )
                    grad_cost = t.autograd.grad(cost, A_pred_grad, allow_unused=True)[0]
                if grad_cost is None:
                    grad_cost = 0.0
                A_guided_final = A_pred_cfg - self.alpha * grad_cost
            else:
                A_guided_final = A_pred_cfg

            # BFN ODE step
            A_t = self.bfn_ode_step(A_t, A_guided_final, dt)

        return A_t.detach()
