"""Train a conditional BFN policy on the MazeEnv using Hydra + Weights & Biases."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import hydra
import numpy as np
import torch as t
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

from environments import MazeEnv
from policies import ConditionalBFNBackbone, ConditionalBFNPolicy


# --------------------
# Config
# --------------------
@dataclass
class DataConfig:
    episodes: int = 200
    max_steps: int = 50
    batch_size: int = 64
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
class TrainConfig:
    epochs: int = 50
    seed: int = 42
    device: str = "cpu"
    log_interval: int = 1
    ckpt_path: str | None = None


@dataclass
class OptimizerConfig:
    type: str = "adam"  # adam, adamw, sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: Optional[float] = None  # for SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam/AdamW


@dataclass
class SchedulerConfig:
    type: Optional[str] = None  # step, cosine, or None
    step_size: int = 50
    gamma: float = 0.5
    T_max: int = 100


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "maze-bfn"
    entity: str | None = None
    run_name: str | None = None
    resume: bool = False
    mode: str | None = None
    tags: Tuple[str, ...] | None = None
    id: str | None = None
    group: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    maze_seed: int | None = None


# --------------------
# Utils
# --------------------
Action = int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def bfs_shortest_action(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Action:
    """Return first action along a shortest path from start to goal using BFS."""
    h, w = grid.shape
    q = [(start, [])]
    visited = {start}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        (y, x), path = q.pop(0)
        if (y, x) == goal:
            return path[0] if path else 0
        for idx, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if grid[ny, nx] == "#":
                continue
            if (ny, nx) in visited:
                continue
            visited.add((ny, nx))
            q.append(((ny, nx), path + [idx]))
    return 0  # fallback (should not happen on valid mazes)


def collect_dataset(cfg: Config, env: MazeEnv) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out expert trajectories (shortest path) to build supervised dataset."""
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    for _ in range(cfg.data.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated) and steps < cfg.data.max_steps:
            y, x = env._pos  # use env internals to compute expert action
            action = bfs_shortest_action(env.grid, (y, x), tuple(env.goal_pos))
            one_hot = np.zeros(env.action_space.n, dtype=np.float32)
            one_hot[action] = 1.0
            obs_list.append(obs.astype(np.float32))
            act_list.append(one_hot)

            obs, _, done, truncated, _ = env.step(action)
            steps += 1
    return np.stack(obs_list), np.stack(act_list)


def build_dataloaders(cfg: Config, obs: np.ndarray, acts: np.ndarray):
    n = obs.shape[0]
    split = int(n * (1 - cfg.data.val_split))
    train_obs, val_obs = obs[:split], obs[split:]
    train_acts, val_acts = acts[:split], acts[split:]

    def to_loader(o, a, shuffle: bool):
        ds = TensorDataset(t.tensor(o), t.tensor(a))
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle)

    return to_loader(train_obs, train_acts, True), to_loader(val_obs, val_acts, False)


def evaluate(policy: ConditionalBFNPolicy, loader: DataLoader, device: t.device) -> float:
    policy.eval()
    correct = 0
    total = 0
    with t.inference_mode():
        for obs, acts in loader:
            obs = obs.to(device)
            acts = acts.to(device)
            preds = policy(obs, deterministic=True)
            pred_actions = preds.argmax(dim=-1)
            target_actions = acts.argmax(dim=-1)
            correct += (pred_actions == target_actions).sum().item()
            total += obs.size(0)
    policy.train()
    return correct / max(total, 1)


def build_optimizer(policy: ConditionalBFNPolicy, cfg: Config):
    opt_cfg = cfg.optimizer
    lr = opt_cfg.lr
    wd = opt_cfg.weight_decay
    opt_type = opt_cfg.type.lower()

    if opt_type == "adam":
        opt = t.optim.Adam(policy.parameters(), lr=lr, weight_decay=wd, betas=opt_cfg.betas)
    elif opt_type == "adamw":
        opt = t.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd, betas=opt_cfg.betas)
    elif opt_type == "sgd":
        momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.0
        opt = t.optim.SGD(policy.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_cfg.type}")

    sched_cfg = cfg.scheduler
    scheduler = None
    if sched_cfg.type:
        stype = sched_cfg.type.lower()
        if stype == "step":
            scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
        elif stype == "cosine":
            scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_cfg.T_max)
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_cfg.type}")

    return opt, scheduler


# --------------------
# Hydra entrypoint
# --------------------
@hydra.main(config_name="config", config_path=None, version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    train_cfg: TrainConfig = cfg.train
    model_cfg: ModelConfig = cfg.model
    data_cfg: DataConfig = cfg.data
    wandb_cfg: WandbConfig = cfg.wandb

    set_seed(train_cfg.seed)
    device = t.device(train_cfg.device)

    env = MazeEnv(max_steps=data_cfg.max_steps, seed=cfg.maze_seed)
    obs_np, acts_np = collect_dataset(cfg, env)
    train_loader, val_loader = build_dataloaders(cfg, obs_np, acts_np)

    policy = ConditionalBFNPolicy(
        action_space=env.action_space,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=model_cfg.hidden_dims,
        time_emb_dim=model_cfg.time_emb_dim,
        sigma_1=model_cfg.sigma_1,
        n_timesteps=model_cfg.n_timesteps,
        cond_scale=model_cfg.cond_scale,
        rescaled_phi=model_cfg.rescaled_phi,
        device=train_cfg.device,
    ).to(device)

    opt, scheduler = build_optimizer(policy, cfg)

    wandb_run = None
    if wandb_cfg.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; install or disable wandb logging.")
        wandb_run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=wandb_cfg.resume,
            mode=wandb_cfg.mode,
            tags=list(wandb_cfg.tags) if wandb_cfg.tags is not None else None,
            id=wandb_cfg.id,
            group=wandb_cfg.group,
        )

    for epoch in range(1, train_cfg.epochs + 1):
        policy.train()
        total_loss = 0.0
        total_batches = 0
        for obs, acts in train_loader:
            obs = obs.to(device)
            acts = acts.to(device)
            loss = policy.bfn.loss(acts, cond=obs).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        val_acc = evaluate(policy, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        if epoch % train_cfg.log_interval == 0:
            current_lr = opt.param_groups[0]["lr"]
            log_msg = f"[{epoch}/{train_cfg.epochs}] loss={avg_loss:.4f} val_acc={val_acc:.3f} lr={current_lr:.2e}"
            print(log_msg)
            if wandb_cfg.use_wandb:
                wandb.log({"loss": avg_loss, "val_acc": val_acc, "lr": current_lr, "epoch": epoch})

    if train_cfg.ckpt_path:
        dirpath = os.path.dirname(train_cfg.ckpt_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        t.save(policy.state_dict(), train_cfg.ckpt_path)
        print(f"Saved checkpoint to {train_cfg.ckpt_path}")

        if wandb_run is not None:
            artifact = wandb.Artifact("maze-bfn-policy", type="model")
            artifact.add_file(train_cfg.ckpt_path)
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
