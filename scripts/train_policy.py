#!/usr/bin/env python

"""Hydra-based policy training entrypoint with logging and callbacks.

Supports two data modes:
1. DataModule (Recommended): Encapsulates dataset generation, normalization, and splitting.
   Required for complex tasks like BFN on PointMaze.
2. Manual Collection: Iterates environment with random/expert actions for simple testing.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import random
import socket
import warnings
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, TensorDataset

from environments import AntMazeEnv, MazeEnv, PushTEnv
from tasks.policy_training import PolicyTraining

# Enable logging for debugging
faulthandler.enable(all_threads=False)

warnings.filterwarnings(
    "ignore",
    "The '\\w+_dataloader' does not have many workers",
    module="lightning",
)
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)

log = logging.getLogger("policy_train")


def store_job_info(config: DictConfig):
    """Attach job metadata (host/pid/slurm ids) to the config for logging."""
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


def get_callbacks(config: DictConfig, logger):
    monitor_metric = (
        config.monitor.get("metric", "val/loss_epoch")
        if config.get("monitor")
        else "val/loss_epoch"
    )
    monitor_mode = config.monitor.get("mode", "min") if config.get("monitor") else "min"

    # Handle checkpoint path safely
    if config.get("train") and config.train.get("ckpt_path"):
        ckpt_base = Path(config.train.ckpt_path).parent
    else:
        ckpt_base = Path("checkpoints")

    dirpath = ckpt_base
    if logger is not None and hasattr(logger, "name") and hasattr(logger, "version"):
        dirpath = ckpt_base / f"{logger.name}/{logger.version}"

    callbacks = [
        ModelCheckpoint(
            dirpath=str(dirpath),
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            monitor=monitor_metric,
            mode=monitor_mode,
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(),
    ]

    if config.get("early_stopping") is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            monitor=monitor_metric,
            mode=monitor_mode,
        )
        callbacks.append(stopper)
    return callbacks


# -------------------------- Utilities ---------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.array(obs[k]).ravel() for k in sorted(obs.keys())]
        return np.concatenate(parts, axis=0)
    return np.array(obs).ravel()


def collect_dataset(cfg: DictConfig, env) -> tuple[np.ndarray, np.ndarray]:
    """Collects data manually if no DataModule is provided."""
    obs_buf, act_buf = [], []

    # Safely access nested config
    episodes = cfg.data.episodes if cfg.get("data") else 50
    max_steps = cfg.data.max_steps if cfg.get("data") else 200

    log.info(f"Collecting {episodes} episodes (Manual Mode)...")

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated) and steps < max_steps:
            if hasattr(env, "get_expert_action"):
                action = env.get_expert_action(obs)
            else:
                action = env.action_space.sample()

            obs_buf.append(flatten_obs(obs).astype(np.float32))

            action_arr = np.array(action)
            if np.issubdtype(action_arr.dtype, np.floating):
                action_arr = action_arr.astype(np.float32)
            act_buf.append(action_arr.ravel())

            obs, _, done, truncated, _ = env.step(action)
            steps += 1

    return np.stack(obs_buf), np.stack(act_buf)


def build_loaders(cfg: DictConfig, obs: np.ndarray, acts: np.ndarray):
    n = obs.shape[0]
    val_split = cfg.data.val_split if cfg.get("data") else 0.1
    batch_size = cfg.data.batch_size if cfg.get("data") else 64

    split = int(n * (1 - val_split))

    def to_loader(o, a, shuffle):
        obs_t = torch.tensor(o, dtype=torch.float32)
        acts_t = torch.tensor(a)
        if acts_t.is_floating_point():
            acts_t = acts_t.float()
        ds = TensorDataset(obs_t, acts_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return to_loader(obs[:split], acts[:split], True), to_loader(
        obs[split:], acts[split:], False
    )


def make_env(cfg: DictConfig):
    env_cfg = cfg.get("environment") or cfg.get("env")
    if env_cfg is None:
        # Fallback for simple configs that might rely on defaults not yet loaded
        raise ValueError("Config must define `environment` or `env`.")

    max_steps = cfg.data.max_steps if cfg.get("data") else 200
    render_mode = env_cfg.render_mode

    if env_cfg.kind == "maze":
        return MazeEnv(max_steps=max_steps, render_mode=render_mode)

    if env_cfg.kind == "point_maze":
        from environments.point_maze import PointMazeEnv

        return PointMazeEnv(render_mode=render_mode, max_steps=max_steps)

    if env_cfg.kind == "antmaze":
        if AntMazeEnv is None:
            raise ImportError("AntMazeEnv not available; install gymnasium[robotics].")
        return AntMazeEnv(env_name=env_cfg.name, render_mode=render_mode)

    if env_cfg.kind == "pusht":
        if PushTEnv is None:
            raise ImportError(
                "PushTEnv not available; install gymnasium and gym-pusht."
            )
        return PushTEnv(env_name=env_cfg.name, render_mode=render_mode)

    raise ValueError(f"Unknown env kind: {env_cfg.kind}")


# --------------------------- Main -------------------------------- #
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig):
    # Handle seed safely
    seed = config.train.seed if config.get("train") else 42
    set_seed(seed)

    store_job_info(config)
    OmegaConf.resolve(config)

    logger = None
    if config.get("logging") and config.logging.get("wandb") is not None:
        logger = instantiate(
            config.logging.wandb,
            _target_="lightning.pytorch.loggers.WandbLogger",
            resume=(config.logging.wandb.get("mode", "") == "online") and "allow",
            log_model=False,
        )

    log.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision(config.get("matmul_precision", "high"))
    torch.set_num_threads(1)

    # --- Data Pipeline Selection ---
    datamodule: Optional[LightningDataModule] = None
    train_loader = None
    val_loader = None

    if config.get("datamodule") is not None:
        log.info("Instantiating DataModule")
        datamodule = instantiate(config.datamodule)
    else:
        log.info("No DataModule found. Falling back to manual environment collection.")
        env = make_env(config)
        obs_np, acts_np = collect_dataset(config, env)
        train_loader, val_loader = build_loaders(config, obs_np, acts_np)

    # --- Policy & Trainer ---
    log.info("Instantiating policy module")

    # Check if 'task' is defined in config (New Style) or if we build PolicyTraining manually
    if config.get("task"):
        module = instantiate(config.task)
    else:
        # Legacy/Fallback support
        module = PolicyTraining(
            policy=config.policy,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            compile=config.get("compile", False),
        )

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, logger)

    # FIX: Use instantiate for Trainer to handle _target_ correctly
    # We pass explicit overrides for critical params if they exist in 'train' config section,
    # otherwise we rely on the 'trainer' config defaults.
    trainer_kwargs = {}
    if config.get("train"):
        if config.train.get("epochs"):
            trainer_kwargs["max_epochs"] = config.train.epochs
        if config.train.get("log_every_n_steps"):
            trainer_kwargs["log_every_n_steps"] = config.train.log_every_n_steps

    trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=[SLURMEnvironment(auto_requeue=False)],
        **trainer_kwargs,
    )

    log.info("Starting training!")

    trainer.fit(
        module,
        datamodule=datamodule,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Handle checkpointing
    ckpt_path = None
    if config.get("train") and config.train.get("ckpt_path"):
        ckpt_path = Path(config.train.ckpt_path)

    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_path))
        log.info("Saved checkpoint to %s", ckpt_path)

    if logger is not None:
        logger.finalize("success")


if __name__ == "__main__":
    main()
