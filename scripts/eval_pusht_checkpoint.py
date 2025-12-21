"""Evaluate a PushT checkpoint and save rollout visualizations.

Usage:
    python scripts/eval_pusht_checkpoint.py --ckpt results/pusht/ckpt_ep100.pt

Outputs GIFs/PNGs to the chosen output directory (default: results/pusht/eval_ckpt).
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Dict, List

import imageio
import numpy as np
import torch

from environments.pusht import PushTEnv
from networks.multi_image_obs_encoder import MultiImageObsEncoder
from networks.unet import Unet
from policies.guided_bfn_policy import GuidedBFNPolicy, GuidanceConfig, HorizonConfig
from diffusion_policy.model.common.normalizer import LinearNormalizer
import torchvision
import torch.nn as nn


def get_resnet_backbone():
    """ResNet18 backbone without final FC (matches train script defaults)."""
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()
    return model


def build_policy(device: torch.device) -> GuidedBFNPolicy:
    """Rebuild the PushT GuidedBFNPolicy to match the training setup."""
    chunk_size = 16
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
        resize_shape=[96, 96],
        crop_shape=[84, 84],
        rgb_model=get_resnet_backbone(),
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        output_dim=obs_dim,
        feature_aggregation=None,
    )

    network = Unet(
        input_dim=2 * chunk_size,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )

    policy = GuidedBFNPolicy(
        action_space=None,
        action_dim=2,
        network=network,
        obs_encoder=vision_encoder,
        normalizer_type="linear",
        horizons=HorizonConfig(obs_history=2, prediction=chunk_size, execution=8),
        guidance=GuidanceConfig(steps=20),
        device=str(device),
    ).to(device)

    return policy


def load_checkpoint(ckpt_path: Path, device: torch.device) -> GuidedBFNPolicy:
    policy = build_policy(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    policy.load_state_dict(state_dict, strict=False)

    if "normalizer" in ckpt:
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(ckpt["normalizer"])
        policy.set_normalizer(normalizer)

    policy.eval()
    return policy


def _to_chw(image: np.ndarray) -> np.ndarray:
    """Ensure image is [C, H, W] float32 in [0, 1]."""
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4):
        chw = image
    elif image.ndim == 3:
        chw = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    if chw.dtype != np.float32:
        chw = chw.astype(np.float32)
    if chw.max() > 1.0:
        chw = chw / 255.0
    return chw


def _extract_image_agentpos(obs, env) -> tuple[np.ndarray, np.ndarray]:
    """Handle different PushT obs formats (dict or flat array)."""
    if isinstance(obs, dict):
        img = obs.get("image", None)
        agent_pos = obs.get("agent_pos", None)
    else:
        img = None
        agent_pos = obs

    if img is None:
        img = env.render()
    if agent_pos is None:
        agent_pos = obs if not isinstance(obs, dict) else obs

    agent_pos = np.array(agent_pos, dtype=np.float32).reshape(-1)
    # Ensure expected 2D agent_pos to match training (x, y)
    if agent_pos.size > 2:
        agent_pos = agent_pos[:2]
    return img, agent_pos


@torch.no_grad()
def rollout_episode(
    policy: GuidedBFNPolicy, device: torch.device, max_steps: int
) -> List[np.ndarray]:
    env = PushTEnv(render_mode="rgb_array")
    obs, _ = env.reset()

    obs_hist = deque(maxlen=2)
    frames: List[np.ndarray] = []

    for _ in range(max_steps):
        img_raw, agent_pos = _extract_image_agentpos(obs, env)
        if img_raw is None:
            break
        img_chw = _to_chw(np.array(img_raw))
        obs_hist.append(img_chw)
        while len(obs_hist) < obs_hist.maxlen:
            obs_hist.append(img_chw)

        img_stack = torch.from_numpy(np.stack(list(obs_hist), axis=0)).unsqueeze(0)
        agent_pos_t = torch.from_numpy(agent_pos.astype(np.float32))
        agent_pos_stack = (
            agent_pos_t.unsqueeze(0).unsqueeze(0).repeat(1, img_stack.shape[1], 1)
        )

        obs_dict: Dict[str, torch.Tensor] = {
            "image": img_stack.to(device),
            "agent_pos": agent_pos_stack.to(device),
        }

        action_chunk = policy(obs_dict, deterministic=True)
        action = action_chunk[0, 0].cpu().numpy()

        obs, _, term, trunc, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if term or trunc:
            break

    env.close()
    return frames


def save_gif(frames: List[np.ndarray], path: Path, fps: int = 15):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PushT checkpoint and visualize rollout.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt file.")
    parser.add_argument("--output-dir", type=str, default="results/pusht/eval_ckpt", help="Where to save GIFs.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to roll out.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy = load_checkpoint(ckpt_path, device)
    print(f"Loaded checkpoint from {ckpt_path}")

    out_dir = Path(args.output_dir)
    for ep in range(args.episodes):
        frames = rollout_episode(policy, device, args.max_steps)
        if len(frames) == 0:
            print(f"Episode {ep}: no frames collected; check environment render_mode.")
            continue
        gif_path = out_dir / f"ep{ep:03d}.gif"
        save_gif(frames, gif_path)


if __name__ == "__main__":
    main()
