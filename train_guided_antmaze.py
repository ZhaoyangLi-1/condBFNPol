"""Train a simple GuidedBFN-style policy on AntMaze via behavior cloning."""

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

from environments import AntMazeEnv
from guided_bfn_policy import GuidedBFNPolicy

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class DataConfig:
    episodes: int = 50
    max_steps: int = 300
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
    ckpt_path: str | None = "checkpoints/guided_antmaze_bfn.pt"


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "guided-antmaze-bfn"
    entity: str | None = None
    run_name: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    env_name: str = "AntMaze_UMaze-v4"


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


def collect_dataset(cfg: Config, env: AntMazeEnv, obs_dim: int, act_dim: int):
    def flatten_obs(o):
        if isinstance(o, dict):
            parts = []
            for k in sorted(o.keys()):
                v = o[k]
                parts.append(np.array(v).ravel())
            return np.concatenate(parts, axis=0)
        return np.array(o).ravel()

    obs_buf = []
    act_buf = []
    for _ in range(cfg.data.episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        while not (done or truncated) and steps < cfg.data.max_steps:
            action = (
                env.action_space.sample()
            )  # random policy; replace with expert if available
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

    return to_loader(obs[:split], acts[:split], True), to_loader(
        obs[split:], acts[split:], False
    )


@hydra.main(config_name="guided_antmaze", config_path="config", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    set_seed(cfg.train.seed)
    device = t.device(cfg.train.device)

    env = AntMazeEnv(env_name=cfg.env_name, render_mode=None)
    # Infer dims from a sample observation/action
    sample_obs, _ = env.reset()
    if isinstance(sample_obs, dict):
        obs_dim = sum(np.prod(np.array(v).shape) for v in sample_obs.values())
    else:
        obs_dim = int(np.prod(np.array(sample_obs).shape))
    sample_act = env.action_space.sample()
    act_dim = int(np.prod(np.array(sample_act).shape))

    obs_np, acts_np = collect_dataset(cfg, env, obs_dim, act_dim)
    train_loader, val_loader = build_loaders(cfg, obs_np, acts_np)

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

    opt = t.optim.Adam(
        policy.network.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    wandb_run = None
    if cfg.wandb.use_wandb:
        if wandb is None:
            raise ImportError(
                "wandb not installed; set wandb.use_wandb=false or install wandb."
            )
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    for epoch in range(1, cfg.train.epochs + 1):
        policy.network.train()
        vision.train()
        total_loss = 0.0
        batches = 0
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            # build cond
            obs_seq = obs.unsqueeze(1).repeat(1, cfg.model.T_o, 1)
            cond = vision(obs_seq.view(obs_seq.size(0) * cfg.model.T_o, -1)).view(
                obs_seq.size(0), cfg.model.T_o, -1
            )
            A_init = t.zeros(obs.size(0), cfg.model.T_p, act_dim, device=device)
            t_tensor = t.zeros(obs.size(0), device=device)
            pred = policy.network(A_init, t_tensor, cond)  # [B, T_p, act_dim]
            logits = pred[:, 0, :]
            loss = t.nn.functional.mse_loss(logits, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)

        # Eval
        policy.network.eval()
        vision.eval()
        val_loss = 0.0
        val_batches = 0
        with t.inference_mode():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
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
            print(
                f"[{epoch}/{cfg.train.epochs}] loss={avg_loss:.4f} val_loss={val_loss:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log({"loss": avg_loss, "val_loss": val_loss, "epoch": epoch})

    if cfg.train.ckpt_path:
        dirpath = os.path.dirname(cfg.train.ckpt_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        t.save(policy.state_dict(), cfg.train.ckpt_path)
        print(f"Saved checkpoint to {cfg.train.ckpt_path}")
        if wandb_run is not None:
            artifact = wandb.Artifact("guided-antmaze-bfn", type="model")
            artifact.add_file(cfg.train.ckpt_path)
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
