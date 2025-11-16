"""Train a conditional BFN policy on AntMaze via behavior cloning."""

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
from policies import ConditionalBFNPolicy

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
    hidden_dims: Tuple[int, ...] = (256, 256)
    time_emb_dim: int = 64
    sigma_1: float = 0.001
    n_timesteps: int = 20
    cond_scale: float | None = None
    rescaled_phi: float | None = None


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
    ckpt_path: str | None = "checkpoints/antmaze_bfn.pt"


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "antmaze-bfn"
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


def collect_dataset(cfg: Config, env: AntMazeEnv):
    obs_buf = []
    act_buf = []
    for _ in range(cfg.data.episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        while not (done or truncated) and steps < cfg.data.max_steps:
            action = env.action_space.sample()  # placeholder policy; replace with expert data if available
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


@hydra.main(config_name="antmaze_bfn", config_path="config", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    set_seed(cfg.train.seed)
    device = t.device(cfg.train.device)

    env = AntMazeEnv(env_name=cfg.env_name, render_mode=None)
    obs_np, acts_np = collect_dataset(cfg, env)
    obs_dim = obs_np.shape[1]
    act_dim = acts_np.shape[1]

    train_loader, val_loader = build_loaders(cfg, obs_np, acts_np)

    policy = ConditionalBFNPolicy(
        action_space=env.action_space,
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=list(cfg.model.hidden_dims),
        time_emb_dim=cfg.model.time_emb_dim,
        sigma_1=cfg.model.sigma_1,
        n_timesteps=cfg.model.n_timesteps,
        cond_scale=cfg.model.cond_scale,
        rescaled_phi=cfg.model.rescaled_phi,
        device=cfg.train.device,
        dtype="float32",
    ).to(device)

    opt = t.optim.Adam(policy.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

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
        policy.train()
        total_loss = 0.0
        batches = 0
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            # Use backbone directly for supervised regression
            x_init = t.zeros(obs.size(0), act_dim, device=device)
            time = t.zeros(obs.size(0), device=device)
            pred = policy.backbone(x_init, time, obs)  # predict action from obs conditioning
            loss = t.nn.functional.mse_loss(pred, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)

        policy.eval()
        val_loss = 0.0
        val_batches = 0
        with t.inference_mode():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                x_init = t.zeros(obs.size(0), act_dim, device=device)
                time = t.zeros(obs.size(0), device=device)
                pred = policy.backbone(x_init, time, obs)
                val_loss += t.nn.functional.mse_loss(pred, act).item()
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
        t.save(policy.state_dict(), cfg.train.ckpt_path)
        print(f"Saved checkpoint to {cfg.train.ckpt_path}")
        if wandb_run is not None:
            artifact = wandb.Artifact("antmaze-bfn", type="model")
            artifact.add_file(cfg.train.ckpt_path)
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
