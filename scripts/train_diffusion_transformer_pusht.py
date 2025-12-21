"""Train a low-dim Diffusion Transformer policy on PushT agent_pos observations.

Matches the provided low-dim transformer config:
- horizon=16, n_obs_steps=2, n_action_steps=8
- Transformer: 8 layers, 4 heads, emb=256, p_drop_attn=0.01, causal attn on
- DDPMScheduler: 100 steps, squaredcos_cap_v2
- AdamW lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-1
- EMA: inv_gamma=1.0, power=0.75, max=0.9999

Monitoring: train/val loss, optional eval rollouts in PushTEnv, CSV/JSONL logs, optional W&B.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from environments.pusht import PushTEnv
from networks.transformer_for_diffusion import TransformerForDiffusion
from policies.diffusion_policy import DiffusionPolicy, HorizonConfig, InferenceConfig
from utils.ema import EMAModel


def main():
    parser = argparse.ArgumentParser(description="Train low-dim Diffusion Transformer on PushT agent_pos.")
    parser.add_argument("--config-file", type=str, default=None, help="Optional YAML config to load defaults.")
    parser.add_argument("--dataset", type=str, default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"])
    parser.add_argument("--wandb-project", type=str, default="dp-pusht-lowdim")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate in env every N epochs (0=disable).")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=300)
    args = parser.parse_args()

    # Config file support (YAML). CLI overrides config defaults.
    defaults = parser.parse_args([])
    if args.config_file is not None:
        cfg_path = Path(args.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        import yaml

        with cfg_path.open("r") as f:
            cfg_data = yaml.safe_load(f) or {}
        allowed = {
            "dataset",
            "epochs",
            "batch_size",
            "num_workers",
            "checkpoint_every",
            "output_dir",
            "wandb_mode",
            "wandb_project",
            "wandb_entity",
            "wandb_name",
            "eval_every",
            "eval_episodes",
            "eval_max_steps",
        }
        for k, v in cfg_data.items():
            if k in allowed and getattr(args, k) == getattr(defaults, k):
                setattr(args, k, v)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    HORIZON = 16
    N_ACTION_STEPS = 8
    N_OBS_STEPS = 2
    ACTION_DIM = 2  # PushT action dim
    OBS_DIM = 2  # use agent_pos only
    LR = 1e-4
    BETAS = (0.9, 0.95)
    WEIGHT_DECAY = 1e-1
    ZARR_PATH = args.dataset

    print(f"--- Training Diffusion Transformer (PushT low-dim) on {DEVICE} ---")
    if not os.path.exists(ZARR_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {ZARR_PATH}. Please download pusht_cchi_v7_replay.zarr."
        )

    # Run directory
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        out_dir = Path(f"results/pusht_diffusion_transformer/{stamp}")
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    jsonl_path = out_dir / "train_log.jsonl"

    # Dataset (we only use agent_pos for obs)
    dataset = PushTImageDataset(
        zarr_path=ZARR_PATH,
        horizon=HORIZON,
        pad_before=1,
        pad_after=7,
    )
    val_size = max(1, int(0.02 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=True,
    )

    # Normalizer (actions + agent_pos)
    normalizer = LinearNormalizer()
    print("Fitting normalizer on subset...")
    subset_indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
    subset_batch = [dataset[i] for i in subset_indices]
    collate_obs = []
    collate_act = []
    for x in subset_batch:
        collate_act.append(x["action"])
        collate_obs.append(x["obs"]["agent_pos"])
    fit_data = {"action": np.stack(collate_act), "obs": np.stack(collate_obs)}
    normalizer.fit(fit_data)

    # Model (Transformer)
    model = TransformerForDiffusion(
        input_dim=ACTION_DIM,
        output_dim=ACTION_DIM,
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        cond_dim=OBS_DIM,
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.01,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0,
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        prediction_type="epsilon",
        clip_sample=True,
    )

    policy = DiffusionPolicy(
        action_space=None,
        model=model,
        noise_scheduler=noise_scheduler,
        action_dim=ACTION_DIM,
        obs_dim=OBS_DIM,
        horizons=HorizonConfig(
            planning_horizon=HORIZON,
            obs_history=N_OBS_STEPS,
            action_prediction=N_ACTION_STEPS,
            execution=N_ACTION_STEPS,
        ),
        inference=InferenceConfig(
            num_steps=100,
            condition_mode="global",
            pred_action_steps_only=False,
            oa_step_convention=False,
        ),
        device=str(DEVICE),
    ).to(DEVICE)
    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = EMAModel(
        policy,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    )

    # Logging setup
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "eval_return", "eval_len"])
    if not jsonl_path.exists():
        jsonl_path.touch()

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": ZARR_PATH,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": LR,
                "betas": BETAS,
                "weight_decay": WEIGHT_DECAY,
                "horizon": HORIZON,
                "n_obs_steps": N_OBS_STEPS,
                "n_action_steps": N_ACTION_STEPS,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "eval_max_steps": args.eval_max_steps,
            },
        )

    def batch_to_tensors(batch: Dict) -> Dict:
        obs = batch["obs"]["agent_pos"].to(DEVICE, non_blocking=True)
        act = batch["action"].to(DEVICE, non_blocking=True)
        return {"obs": obs, "action": act}

    # Training loop
    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            b = batch_to_tensors(batch)
            loss = policy.compute_loss(b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            ema.step(policy)
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_avg = epoch_loss / len(train_loader)

        # Validation with EMA
        policy_state = policy.state_dict()
        policy.load_state_dict(ema.averaged_model.state_dict(), strict=False)
        val_loss = 0.0
        n_val = 0
        policy.eval()
        with torch.no_grad():
            for batch in val_loader:
                b = batch_to_tensors(batch)
                loss = policy.compute_loss(b)
                val_loss += loss.item()
                n_val += 1
        val_avg = val_loss / max(1, n_val)
        policy.load_state_dict(policy_state, strict=False)

        eval_ret = None
        eval_len = None
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            eval_ret, eval_len = run_eval(policy, normalizer, DEVICE, args.eval_episodes, args.eval_max_steps)

        print(
            f"Epoch {epoch+1} | train_loss: {train_avg:.4f} | val_loss (EMA): {val_avg:.4f} | eval_return: {eval_ret}"
        )
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_avg, val_avg, eval_ret, eval_len])
        with jsonl_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_avg,
                        "val_loss": val_avg,
                        "eval_return": eval_ret,
                        "eval_len": eval_len,
                    }
                )
                + "\n"
            )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_avg,
                    "val/loss": val_avg,
                    "eval/return": eval_ret,
                    "eval/len": eval_len,
                },
                step=epoch + 1,
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "ema_state": ema.state_dict(),
                    "normalizer": normalizer.state_dict(),
                },
                out_dir / f"ckpt_ep{epoch+1}.pt",
            )

    # Final save
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "ema_state": ema.state_dict(),
            "normalizer": normalizer.state_dict(),
        },
        out_dir / "ckpt_final.pt",
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Training complete. Artifacts in {out_dir}")


@torch.no_grad()
def run_eval(policy, normalizer, device, episodes: int, max_steps: int):
    """Simple eval: run episodes in PushTEnv using agent_pos obs only."""
    try:
        env = PushTEnv(render_mode=None)
    except Exception:
        return None, None

    rets = []
    lens = []
    policy.eval()
    policy.set_normalizer(normalizer)

    for _ in range(episodes):
        obs, _ = env.reset()
        ret = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            obs_t = torch.as_tensor(obs["agent_pos"], device=device, dtype=torch.float32)
            obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # [1,1,2]
            action_chunk = policy({"obs": obs_t})
            if action_chunk.dim() == 3:
                action_np = action_chunk[0, 0].detach().cpu().numpy()
            else:
                action_np = action_chunk.detach().cpu().numpy()
            obs, reward, term, trunc, _ = env.step(action_np)
            ret += float(reward)
            done = term or trunc
            steps += 1
        rets.append(ret)
        lens.append(steps)
    env.close()
    return float(np.mean(rets)), float(np.mean(lens))


if __name__ == "__main__":
    main()
