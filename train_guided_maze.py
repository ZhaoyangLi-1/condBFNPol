"""Train a simple GuidedBFN-style policy on MazeEnv using MLP encoder/backbone."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import hydra
import numpy as np
import torch as t
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from environments import MazeEnv
from guided_bfn_policy import GuidedBFNPolicy

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# ---------------- Configs ---------------- #


@dataclass
class DataConfig:
    episodes: int = 200
    max_steps: int = 50
    batch_size: int = 64
    val_split: float = 0.1


@dataclass
class ModelConfig:
    obs_dim: int = 2
    action_dim: int = 4
    emb_dim: int = 32
    hidden_dim: int = 128
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
    ckpt_path: str | None = "checkpoints/guided_maze_bfn.pt"


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "guided-maze-bfn"
    entity: str | None = None
    run_name: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    maze_seed: int | None = None


# --------------- Components -------------- #


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
        # x: [B*T_o, obs_dim]
        return self.net(x)


class BackboneMLP(t.nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        inp = action_dim + cond_dim + 1  # + time scalar
        self.net = t.nn.Sequential(
            t.nn.Linear(inp, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, A: t.Tensor, t_tensor: t.Tensor, cond: t.Tensor) -> t.Tensor:
        # A: [B, T_p, action_dim]; cond: [B, T_o, cond_dim]; t_tensor: [B]
        B, T_p, a_dim = A.shape
        # Use last cond embedding (most recent obs)
        cond_last = cond[:, -1, :]  # [B, cond_dim]
        cond_ex = cond_last[:, None, :].expand(B, T_p, cond_last.size(-1))
        t_ex = t_tensor.view(B, 1, 1).expand(B, T_p, 1)
        x = t.cat([A, cond_ex, t_ex], dim=-1)
        out = self.net(x)
        return out


# --------------- Dataset utils ------------ #


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def bfs_shortest_action(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> int:
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
    return 0


def collect_dataset(cfg: Config, env: MazeEnv):
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    for _ in range(cfg.data.episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        while not (done or truncated) and steps < cfg.data.max_steps:
            y, x = env._pos
            action = bfs_shortest_action(env.grid, (y, x), tuple(env.goal_pos))
            one_hot = np.zeros(env.action_space.n, dtype=np.float32)
            one_hot[action] = 1.0
            obs_list.append(obs.astype(np.float32))
            act_list.append(one_hot)
            obs, _, done, truncated, _ = env.step(action)
            steps += 1
    return np.stack(obs_list), np.stack(act_list)


def build_loaders(cfg: Config, obs: np.ndarray, acts: np.ndarray):
    n = obs.shape[0]
    split = int(n * (1 - cfg.data.val_split))

    def to_loader(o, a, shuffle):
        ds = TensorDataset(t.tensor(o), t.tensor(a))
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle)

    return to_loader(obs[:split], acts[:split], True), to_loader(
        obs[split:], acts[split:], False
    )


# --------------- Training ----------------- #


@hydra.main(config_name="guided_config", config_path="config", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    set_seed(cfg.train.seed)
    device = t.device(cfg.train.device)

    env = MazeEnv(max_steps=cfg.data.max_steps, seed=cfg.maze_seed)
    obs_np, acts_np = collect_dataset(cfg, env)
    train_loader, val_loader = build_loaders(cfg, obs_np, acts_np)

    vision = VisionEncoderMLP(cfg.model.obs_dim, cfg.model.emb_dim).to(device)
    backbone = BackboneMLP(
        cfg.model.action_dim, cfg.model.emb_dim, cfg.model.hidden_dim
    ).to(device)

    policy = GuidedBFNPolicy(
        backbone_transformer=backbone,
        vision_encoder=vision,
        action_dim=cfg.model.action_dim,
        bfn_timesteps=cfg.model.bfn_timesteps,
        obs_horizon_T_o=cfg.model.T_o,
        action_pred_horizon_T_p=cfg.model.T_p,
        action_exec_horizon_T_a=cfg.model.T_a,
        cfg_uncond_prob=cfg.model.cfg_uncond_prob,
        cfg_guidance_scale_w=cfg.model.cfg_guidance_scale_w,
        grad_guidance_scale_alpha=cfg.model.grad_guidance_scale_alpha,
        device=device,
    )
    policy.to(device)

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
            # Build fake time and cond to predict a single-step action sequence
            obs_seq = obs.unsqueeze(1).repeat(1, cfg.model.T_o, 1)  # [B, T_o, obs_dim]
            cond = vision(obs_seq.view(obs_seq.size(0) * cfg.model.T_o, -1)).view(
                obs_seq.size(0), cfg.model.T_o, -1
            )
            A_init = t.zeros(
                obs.size(0), cfg.model.T_p, cfg.model.action_dim, device=device
            )
            t_tensor = t.zeros(obs.size(0), device=device)
            pred = policy.network(A_init, t_tensor, cond)  # [B, T_p, action_dim]
            logits = pred[:, 0, :]  # use first step
            loss = t.nn.functional.cross_entropy(logits, act.argmax(dim=-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)

        # Eval
        policy.network.eval()
        vision.eval()
        correct = 0
        total = 0
        with t.inference_mode():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                obs_seq = obs.unsqueeze(1).repeat(1, cfg.model.T_o, 1)
                cond = vision(obs_seq.view(obs_seq.size(0) * cfg.model.T_o, -1)).view(
                    obs_seq.size(0), cfg.model.T_o, -1
                )
                A_init = t.zeros(
                    obs.size(0), cfg.model.T_p, cfg.model.action_dim, device=device
                )
                t_tensor = t.zeros(obs.size(0), device=device)
                pred = policy.network(A_init, t_tensor, cond)
                logits = pred[:, 0, :]
                pred_actions = logits.argmax(dim=-1)
                target = act.argmax(dim=-1)
                correct += (pred_actions == target).sum().item()
                total += obs.size(0)
        val_acc = correct / max(total, 1)

        if epoch % cfg.train.log_interval == 0:
            print(
                f"[{epoch}/{cfg.train.epochs}] loss={avg_loss:.4f} val_acc={val_acc:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log({"loss": avg_loss, "val_acc": val_acc, "epoch": epoch})

    # Save and upload
    if cfg.train.ckpt_path:
        dirpath = os.path.dirname(cfg.train.ckpt_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        t.save(policy.state_dict(), cfg.train.ckpt_path)
        print(f"Saved checkpoint to {cfg.train.ckpt_path}")
        if wandb_run is not None:
            artifact = wandb.Artifact("guided-maze-bfn", type="model")
            artifact.add_file(cfg.train.ckpt_path)
            wandb_run.log_artifact(artifact)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
