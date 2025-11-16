"""Train a simple GuidedBFN-style policy on PushT via behavior cloning."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Tuple

import hydra
import numpy as np
import torch as t
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from environments import PushTEnv
from guided_bfn_policy import GuidedBFNPolicy
from networks.unet import Unet

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class DataConfig:
    episodes: int = 50
    max_steps: int = 200
    batch_size: int = 128
    val_split: float = 0.1


@dataclass
class ModelConfig:
    emb_dim: int = 128
    hidden_dim: int = 256
    bfn_timesteps: int = 50
    T_o: int = 2
    T_p: int = 4
    T_a: int = 1
    cfg_uncond_prob: float = 0.1
    cfg_guidance_scale_w: float = 3.0
    grad_guidance_scale_alpha: float = 0.0
    use_unet: bool = False
    unet_dim: int = 64


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 50
    seed: int = 42
    device: str = "cpu"
    log_interval: int = 1
    ckpt_path: str | None = "checkpoints/guided_pusht_bfn.pt"


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "guided-pusht-bfn"
    entity: str | None = None
    run_name: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    env_name: str = "gym_pusht/PushT-v0"


class VisionEncoderMLP(t.nn.Module):
    def __init__(self, obs_dim: int, emb_dim: int):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(obs_dim, emb_dim),
            t.nn.ReLU(),
            t.nn.Linear(emb_dim, emb_dim),
        )
        self.output_dim = emb_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)


class BackboneMLP(t.nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        inp = action_dim + cond_dim + 1
        self.net = t.nn.Sequential(
            t.nn.Linear(inp, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, A: t.Tensor, t_tensor: t.Tensor, cond: t.Tensor) -> t.Tensor:
        B, T_p, _ = A.shape
        cond_last = cond[:, -1, :]
        cond_ex = cond_last[:, None, :].expand(B, T_p, cond_last.size(-1))
        t_ex = t_tensor.view(B, 1, 1).expand(B, T_p, 1)
        x = t.cat([A, cond_ex, t_ex], dim=-1)
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.array(obs[k]).ravel() for k in sorted(obs.keys())]
        return np.concatenate(parts, axis=0)
    return np.array(obs).ravel()


def collect_dataset(cfg: Config, env: PushTEnv):
    obs_buf = []
    act_buf = []
    for _ in range(cfg.data.episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        while not (done or truncated) and steps < cfg.data.max_steps:
            action = env.action_space.sample()  # random BC placeholder
            obs_buf.append(flatten_obs(obs).astype(np.float32))
            act_buf.append(np.array(action, dtype=np.float32).ravel())
            obs, reward, done, truncated, _ = env.step(action)
            steps += 1
    return np.stack(obs_buf), np.stack(act_buf)


def build_loaders(cfg: Config, obs: np.ndarray, acts: np.ndarray):
    n = obs.shape[0]
    split = int(n * (1 - cfg.data.val_split))

    def to_loader(o, a, shuffle):
        ds = TensorDataset(t.tensor(o), t.tensor(a))
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle)

    return to_loader(obs[:split], acts[:split], True), to_loader(obs[split:], acts[split:], False)


@hydra.main(config_name="guided_pusht", config_path="config", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    set_seed(cfg.train.seed)
    device = t.device(cfg.train.device)

    if PushTEnv is None:
        raise ImportError("PushTEnv not available. Install gymnasium and gym-pusht.")

    env = PushTEnv(env_name=cfg.env_name, render_mode=None)
    obs_np, acts_np = collect_dataset(cfg, env)
    obs_dim = obs_np.shape[1]
    act_dim = acts_np.shape[1]

    train_loader, val_loader = build_loaders(cfg, obs_np, acts_np)

    use_unet = getattr(cfg.model, "use_unet", False)
    obs_shape = env.observation_space.shape
    if use_unet:
        if len(obs_shape) == 3:
            h, w, c = obs_shape
        elif len(obs_shape) == 4:
            _, h, w, c = obs_shape
        else:
            print(f"Unet mode requested but obs shape {obs_shape} is not image-like; falling back to MLP.")
            use_unet = False

    if use_unet:
        unet = Unet(dim=cfg.model.unet_dim, channels=c, out_dim=act_dim, num_classes=None).to(device)
        opt = t.optim.Adam(unet.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        policy = None
    else:
        vision = VisionEncoderMLP(obs_dim, cfg.model.emb_dim).to(device)
        backbone = BackboneMLP(act_dim, cfg.model.emb_dim, cfg.model.hidden_dim).to(device)
        policy = GuidedBFNPolicy(
            backbone_transformer=backbone,
            vision_encoder=vision,
            action_dim=act_dim,
            bfn_timesteps=cfg.model.bfn_timesteps,
            obs_horizon_T_o=cfg.model.T_o,
            action_pred_horizon_T_p=cfg.model.T_p,
            action_exec_horizon_T_a=cfg.model.T_a,
            cfg_uncond_prob=cfg.model.cfg_uncond_prob,
            cfg_guidance_scale_w=cfg.model.cfg_guidance_scale_w,
            grad_guidance_scale_alpha=cfg.model.grad_guidance_scale_alpha,
            device=device,
        ).to(device)
        opt = t.optim.Adam(policy.network.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    wandb_run = None
    if cfg.wandb.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; set wandb.use_wandb=false or install wandb.")
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    for epoch in range(1, cfg.train.epochs + 1):
        total_loss = 0.0
        batches = 0
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            if use_unet:
                # reshape obs to NCHW
                if obs.dim() == 4:  # [B, H, W, C]
                    obs_img = obs.permute(0, 3, 1, 2)
                elif obs.dim() == 3:  # [B, C, H, W]
                    obs_img = obs
                else:
                    raise ValueError(f"Unexpected obs dim {obs.shape}")
                t_tensor = t.zeros(obs.size(0), device=device)
                out = unet(obs_img, t_tensor)
                logits = out.mean(dim=(-2, -1))  # spatial mean to action_dim
            else:
                obs_seq = obs.unsqueeze(1).repeat(1, cfg.model.T_o, 1)
                cond = vision(obs_seq.view(obs_seq.size(0) * cfg.model.T_o, -1)).view(
                    obs_seq.size(0), cfg.model.T_o, -1
                )
                A_init = t.zeros(obs.size(0), cfg.model.T_p, act_dim, device=device)
                t_tensor = t.zeros(obs.size(0), device=device)
                pred = policy.network(A_init, t_tensor, cond)
                logits = pred[:, 0, :]

            loss = t.nn.functional.mse_loss(logits, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)

        val_loss = 0.0
        val_batches = 0
        with t.inference_mode():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                if use_unet:
                    if obs.dim() == 4:
                        obs_img = obs.permute(0, 3, 1, 2)
                    elif obs.dim() == 3:
                        obs_img = obs
                    else:
                        raise ValueError(f"Unexpected obs dim {obs.shape}")
                    t_tensor = t.zeros(obs.size(0), device=device)
                    out = unet(obs_img, t_tensor)
                    logits = out.mean(dim=(-2, -1))
                else:
                    obs_seq = obs.unsqueeze(1).repeat(1, cfg.model.T_o, 1)
                    cond = vision(obs_seq.view(obs_seq.size(0) * cfg.model.T_o, -1)).view(
                        obs_seq.size(0), cfg.model.T_o, -1
                    )
                    A_init = t.zeros(obs.size(0), cfg.model.T_p, act_dim, device=device)
                    t_tensor = t.zeros(obs.size(0), device=device)
                    pred = policy.network(A_init, t_tensor, cond)
                    logits = pred[:, 0, :]
                val_loss += t.nn.functional.mse_loss(logits, act).item()
                val_batches += 1
        val_loss = val_loss / max(val_batches, 1)

        if epoch % cfg.train.log_interval == 0:
            print(f"[{epoch}/{cfg.train.epochs}] loss={avg_loss:.4f} val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"loss": avg_loss, "val_loss": val_loss, "epoch": epoch})

    if cfg.train.ckpt_path:
        dirpath = os.path.dirname(cfg.train.ckpt_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if use_unet:
            t.save(unet.state_dict(), cfg.train.ckpt_path)
        else:
            t.save(policy.state_dict(), cfg.train.ckpt_path)
        print(f"Saved checkpoint to {cfg.train.ckpt_path}")
        if wandb_run is not None:
            artifact = wandb.Artifact("guided-pusht-bfn", type="model")
            artifact.add_file(cfg.train.ckpt_path)
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
