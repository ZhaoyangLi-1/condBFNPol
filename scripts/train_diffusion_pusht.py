"""Train a Diffusion Policy on PushT with monitoring (loss logs, optional W&B).

Key settings aligned to the hybrid DP config:
- horizon=16, n_obs_steps=2, n_action_steps=8
- UNet down_dims [512, 1024, 2048], kernel_size=5, n_groups=8
- Crop 84x84 from 96x96, ImageNet norm, group norm encoder
- EMA (inv_gamma=1.0, power=0.75, min=0.0, max=0.9999)
- AdamW lr=1e-4, betas=(0.95, 0.999), wd=1e-6

Monitoring:
- Train/val loss every epoch
- CSV + JSONL logs in the run directory
- Optional W&B logging (--wandb-mode online/offline)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from environments.pusht import PushTEnv
from networks.multi_image_obs_encoder import MultiImageObsEncoder
from networks.unet import Unet
from policies.diffusion_policy import DiffusionPolicy, HorizonConfig, InferenceConfig
from utils.ema import EMAModel


def get_resnet_backbone():
    import torchvision

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()
    return model


def encode_obs(encoder: nn.Module, obs_dict: dict, n_obs_steps: int) -> torch.Tensor:
    """Encode obs dict to [B, T, D]."""
    emb = encoder(obs_dict)  # [B*T, D]
    B = obs_dict["image"].shape[0]
    emb = emb.view(B, n_obs_steps, -1)
    return emb


@torch.no_grad()
def evaluate_policy(
    policy: DiffusionPolicy,
    vision_encoder: nn.Module,
    obs_projector: nn.Module,
    device: torch.device,
    n_obs_steps: int,
    n_action_steps: int,
    episodes: int = 3,
    max_steps: int = 300,
) -> dict:
    """Run a few eval episodes in PushT and return mean return/length."""
    try:
        env = PushTEnv(render_mode=None)
    except Exception:
        return {"return_mean": None, "len_mean": None}

    def prep_obs(obs):
        img = torch.as_tensor(obs["image"], device=device, dtype=torch.float32)
        agent = torch.as_tensor(obs["agent_pos"], device=device, dtype=torch.float32)
        return img, agent

    rets = []
    lens = []
    policy_state = policy.state_dict()
    policy.eval()

    # Use EMA weights if present
    ema_state = None
    if hasattr(policy, "ema_model") and getattr(policy, "ema_model", None) is not None:
        ema_state = policy.state_dict()

    for _ in range(episodes):
        obs, _ = env.reset()
        imgs = deque(maxlen=n_obs_steps)
        agents = deque(maxlen=n_obs_steps)

        img0, agent0 = prep_obs(obs)
        for _ in range(n_obs_steps):
            imgs.append(img0)
            agents.append(agent0)

        ret = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            img_stack = torch.stack(list(imgs), dim=0).unsqueeze(0)  # [1,T,C,H,W]
            agent_stack = torch.stack(list(agents), dim=0).unsqueeze(0)  # [1,T,2]

            obs_dict = {
                "image": img_stack.to(device),
                "agent_pos": agent_stack.to(device),
            }
            obs_emb = encode_obs(vision_encoder, obs_dict, n_obs_steps)
            obs_emb = obs_projector(obs_emb)

            action_chunk = policy({"obs": obs_emb})
            if action_chunk.dim() == 3:
                action_np = action_chunk[0, 0].detach().cpu().numpy()
            else:
                action_np = action_chunk.detach().cpu().numpy()

            obs, reward, term, trunc, _ = env.step(action_np)
            ret += float(reward)
            done = term or trunc
            steps += 1

            img_next, agent_next = prep_obs(obs)
            imgs.append(img_next)
            agents.append(agent_next)

        rets.append(ret)
        lens.append(steps)

    if ema_state is not None:
        policy.load_state_dict(ema_state, strict=False)
    else:
        policy.load_state_dict(policy_state, strict=False)

    env.close()
    return {
        "return_mean": float(np.mean(rets)) if len(rets) > 0 else None,
        "len_mean": float(np.mean(lens)) if len(lens) > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on PushT.")
    parser.add_argument("--config-file", type=str, default=None, help="Optional YAML/JSON config to load defaults.")
    parser.add_argument("--dataset", type=str, default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"])
    parser.add_argument("--wandb-project", type=str, default="dp-pusht")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate in env every N epochs (0=disable).")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=300)
    # Capture defaults to allow config override while letting CLI win
    defaults = parser.parse_args([])
    args = parser.parse_args()

    # Optional config file (YAML/JSON). CLI takes precedence over config values.
    if args.config_file is not None:
        cfg_path = Path(args.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for --config-file") from e

        with cfg_path.open("r") as f:
            cfg_data = yaml.safe_load(f)
        if cfg_data is None:
            cfg_data = {}
        allowed_keys = {
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
            if k not in allowed_keys:
                continue
            # Only apply config value if CLI left it at default
            if getattr(args, k) == getattr(defaults, k):
                setattr(args, k, v)

    # ---- Hyperparameters (matching reference) ----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    HORIZON = 16
    N_ACTION_STEPS = 8
    N_OBS_STEPS = 2
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.epochs
    LR = 1e-4
    BETAS = (0.95, 0.999)
    WEIGHT_DECAY = 1e-6
    ZARR_PATH = args.dataset

    print(f"--- Training Diffusion Policy (PushT) on {DEVICE} ---")
    if not os.path.exists(ZARR_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {ZARR_PATH}. Please download pusht_cchi_v7_replay.zarr."
        )

    # ---- Run directory ----
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        out_dir = Path(f"results/pusht_diffusion/{stamp}")
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    jsonl_path = out_dir / "train_log.jsonl"

    # ---- Run directory ----
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        out_dir = Path(f"results/pusht_diffusion/{stamp}")
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    jsonl_path = out_dir / "train_log.jsonl"

    # ---- Dataset ----
    dataset = PushTImageDataset(
        zarr_path=ZARR_PATH,
        horizon=HORIZON,
        pad_before=1,
        pad_after=7,
    )
    # Split train/val for monitoring
    val_size = max(1, int(0.02 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
    )

    # ---- Models ----
    obs_dim = 128
    shape_meta = {
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2], "type": "action"},
    }

    vision_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet_backbone(),
        resize_shape=[96, 96],
        crop_shape=[84, 84],
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        output_dim=obs_dim,
        feature_aggregation=None,
    ).to(DEVICE)

    # Project encoded obs to a fixed obs_dim (handles larger encoder outputs)
    obs_projector = nn.LazyLinear(obs_dim).to(DEVICE)

    # ---- Normalizer (fit on subset, includes encoded obs) ----
    normalizer = LinearNormalizer()
    print("Fitting normalizer on subset...")
    vision_encoder.eval()
    subset_indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
    subset_batch = [dataset[i] for i in subset_indices]
    collate_obs = []
    collate_act = []
    collate_agent = []
    with torch.no_grad():
        for x in subset_batch:
            collate_act.append(x["action"])
            collate_agent.append(x["obs"]["agent_pos"])

            # encode obs for normalization; accept np arrays or tensors
            img = x["obs"]["image"]
            agent = x["obs"]["agent_pos"]
            img_t = (
                torch.from_numpy(img)
                if isinstance(img, np.ndarray)
                else img.clone().detach()
            )
            agent_t = (
                torch.from_numpy(agent)
                if isinstance(agent, np.ndarray)
                else agent.clone().detach()
            )
            obs_tmp = {
                "image": img_t.unsqueeze(0).to(DEVICE),
                "agent_pos": agent_t.unsqueeze(0).to(DEVICE),
            }
            emb = encode_obs(vision_encoder, obs_tmp, N_OBS_STEPS)
            emb = obs_projector(emb)
            collate_obs.append(emb.cpu().numpy())

    fit_data = {"action": np.stack(collate_act), "obs": np.stack(collate_obs)}
    try:
        fit_data["agent_pos"] = np.stack(collate_agent)
    except Exception:
        pass
    normalizer.fit(fit_data)
    vision_encoder.train()

    model = Unet(
        input_dim=2,  # action_dim
        # Match cond dim to flattened obs (n_obs_steps * obs_dim)
        global_cond_dim=obs_dim * N_OBS_STEPS,
        diffusion_step_embed_dim=128,
        down_dims=[512, 1024, 2048],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
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
        action_dim=2,
        obs_dim=obs_dim,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ema = EMAModel(
        policy,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    )

    # ---- Logging setup ----
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
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
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

    # ---- Training ----
    for epoch in range(EPOCHS):
        policy.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in pbar:
            # Move to device
            batch["action"] = batch["action"].to(DEVICE, non_blocking=True)
            for k in batch["obs"]:
                batch["obs"][k] = batch["obs"][k].to(DEVICE, non_blocking=True)

            # Encode obs to low-dim
            obs_emb = encode_obs(vision_encoder, batch["obs"], N_OBS_STEPS)
            obs_emb = obs_projector(obs_emb)
            batch_enc = {"obs": obs_emb, "action": batch["action"]}

            loss = policy.compute_loss(batch_enc)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            ema.step(policy)

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_avg = epoch_loss / len(train_loader)

        # Validation with EMA weights
        policy.eval()
        val_loss = 0.0
        n_val = 0
        # swap in EMA weights
        policy_state = policy.state_dict()
        policy.load_state_dict(ema.averaged_model.state_dict(), strict=False)
        with torch.no_grad():
            for batch in val_loader:
                batch["action"] = batch["action"].to(DEVICE, non_blocking=True)
                for k in batch["obs"]:
                    batch["obs"][k] = batch["obs"][k].to(DEVICE, non_blocking=True)
                obs_emb = encode_obs(vision_encoder, batch["obs"], N_OBS_STEPS)
                obs_emb = obs_projector(obs_emb)
                batch_enc = {"obs": obs_emb, "action": batch["action"]}
                loss = policy.compute_loss(batch_enc)
                val_loss += loss.item()
                n_val += 1
        # restore training weights
        policy.load_state_dict(policy_state, strict=False)
        val_avg = val_loss / max(1, n_val)

        eval_ret = None
        eval_len = None
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            eval_stats = evaluate_policy(
                policy,
                vision_encoder,
                obs_projector,
                device=torch.device(DEVICE),
                n_obs_steps=N_OBS_STEPS,
                n_action_steps=N_ACTION_STEPS,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
            )
            eval_ret = eval_stats["return_mean"]
            eval_len = eval_stats["len_mean"]

        print(f"Epoch {epoch+1} | train_loss: {train_avg:.4f} | val_loss (EMA): {val_avg:.4f}")
        # append metrics
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

        # Save checkpoint periodically
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


if __name__ == "__main__":
    main()
