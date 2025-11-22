"""Visualize Guided BFN PushT checkpoints.

Fixes:
- Correctly passes 'agent_pos' to the encoder to prevent KeyError.
- Matches architecture to standard training (Encoder handles fusion).
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision

from diffusion_policy.model.common.normalizer import LinearNormalizer
from environments.pusht import PushTEnv
from networks.multi_image_obs_encoder import MultiImageObsEncoder
from networks.unet import Unet
from policies.guided_bfn_policy import GuidedBFNPolicy, GuidanceConfig, HorizonConfig


def _get_resnet_backbone() -> nn.Module:
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()
    return model


def build_pusht_policy(device: str, obs_dim: int, obs_history: int) -> GuidedBFNPolicy:
    chunk_size = 16

    # Calculate feature dimension per step
    # If ObsDim=128 and History=2, then FeatDim=64.
    feat_dim = obs_dim // obs_history

    print(
        f"Building Policy: Total ObsDim={obs_dim}, History={obs_history} -> Per-Step Encoder Dim={feat_dim}"
    )

    shape_meta = {
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2], "type": "action"},
    }

    vision_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=_get_resnet_backbone(),
        resize_shape=[96, 96],
        crop_shape=[84, 84],
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        output_dim=feat_dim,  # Encoder projects Image+Pos to this dim
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
        obs_encoder=None,
        normalizer_type="linear",
        horizons=HorizonConfig(
            obs_history=obs_history, prediction=chunk_size, execution=8
        ),
        guidance=GuidanceConfig(steps=20),
        device=device,
    ).to(device)

    policy.vision_encoder = vision_encoder.to(device)
    return policy


def _latest_checkpoint(base_dir: Path) -> Path:
    candidates = list(base_dir.glob("ckpt_ep*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {base_dir}")

    def _epoch_num(path: Path) -> int:
        match = re.search(r"ep(\d+)", path.stem)
        return int(match.group(1)) if match else -1

    candidates.sort(key=_epoch_num)
    return candidates[-1]


def load_checkpoint(policy: GuidedBFNPolicy, ckpt_path: Path, device: str) -> None:
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    try:
        policy.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"[ERROR] Weight mismatch! {e}")
        print("Tip: Use --obs_dim 128 --history 2 (or check inspect_ckpt.py)")
        exit(1)

    normalizer = LinearNormalizer()
    if "normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer"])
    policy.set_normalizer(normalizer)
    policy.eval()


def _to_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3:
        return image
    if image.ndim == 3 and image.shape[-1] == 3:
        return np.transpose(image, (2, 0, 1))
    if image.ndim == 4 and image.shape[-1] == 3:
        return np.transpose(image[0], (2, 0, 1))
    raise ValueError(f"Unexpected image shape {image.shape}")


def _extract_image_and_pos(env_obs):
    if isinstance(env_obs, dict):
        img = np.array(env_obs.get("image", env_obs.get("observation")))
        pos = np.array(env_obs.get("agent_pos", env_obs.get("state", np.zeros(2))[:2]))
        return img, pos
    if isinstance(env_obs, (list, tuple)):
        return np.array(env_obs[0]), np.array(env_obs[1]) if len(
            env_obs
        ) > 1 else np.zeros(2)
    if isinstance(env_obs, np.ndarray):
        return env_obs, np.zeros(2)
    raise TypeError(f"Unsupported observation type: {type(env_obs)}")


def prepare_obs_embedding(obs, buffer, policy, fallback_image=None):
    raw_img, raw_pos = _extract_image_and_pos(obs)
    try:
        img = _to_chw(np.array(raw_img))
    except ValueError:
        img = _to_chw(np.array(fallback_image))

    img_tensor = torch.from_numpy(img).float()
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0
    agent_pos = torch.from_numpy(np.array(raw_pos)).float()

    buffer.append({"image": img_tensor, "agent_pos": agent_pos})
    while len(buffer) < buffer.maxlen:
        buffer.appendleft(buffer[0])

    stacked_img = (
        torch.stack([o["image"] for o in buffer], dim=0).unsqueeze(0).to(policy.device)
    )
    stacked_pos = (
        torch.stack([o["agent_pos"] for o in buffer], dim=0)
        .unsqueeze(0)
        .to(policy.device)
    )

    # Pass FULL dict to encoder
    obs_input = {"image": stacked_img, "agent_pos": stacked_pos}

    # Encoder returns flattened features: [B*T, Dim] or [B, T*Dim] depending on encoder
    # MultiImageObsEncoder outputs [B, Dim] if configured with output_dim
    # BUT here input is 5D [B, T, C, H, W], so encoder flattens B*T -> [B*T, Dim]
    feats = policy.vision_encoder(obs_input)

    # Reshape [B*T, Dim] -> [B, T*Dim]
    # B=1, T=2
    return feats.view(1, -1)


def rollout_episode(policy, max_steps, render_stride):
    try:
        env = PushTEnv(render_mode="rgb_array")
    except Exception:
        print("PushTEnv not found.")
        return [], False

    obs, _ = env.reset()
    frames = []
    buffer = deque(maxlen=policy.horizons.obs_history)

    first_frame = env.render()
    if first_frame is not None:
        frames.append(first_frame)

    success = False
    steps = 0

    with torch.no_grad():
        while steps < max_steps:
            obs_dict = prepare_obs_embedding(
                obs,
                buffer,
                policy,
                fallback_image=first_frame
                if steps == 0
                else frames[-1]
                if frames
                else None,
            )

            # Plan using flattened condition
            action_chunk_flat = policy.plan(obs_dict)
            action_chunk_flat = policy.normalizer["action"].unnormalize(
                action_chunk_flat
            )
            action_chunk = action_chunk_flat.view(-1, 2)
            action = action_chunk[0].cpu().numpy()

            obs, _, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc
            success = bool(info.get("success", False)) or term

            frame = env.render()
            if frame is not None and steps % render_stride == 0:
                frames.append(frame)

            if done:
                break

    env.close()
    return frames, success


def save_visuals(frames, save_dir, stem, fps):
    save_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    frames_uint8 = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]
    imageio.mimsave(save_dir / f"{stem}.gif", frames_uint8, duration=1 / fps)
    strip = np.concatenate(frames_uint8[:: max(1, len(frames) // 8)], axis=1)
    imageio.imwrite(save_dir / f"{stem}_strip.png", strip)
    print(f"Saved visuals to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--obs_dim", type=int, default=128, help="Total Obs Dim")
    parser.add_argument("--history", type=int, default=2, help="Obs History")
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--save-dir", type=Path, default=Path("results/pusht/vis"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    ckpt_path = args.ckpt if args.ckpt else _latest_checkpoint(Path("results/pusht"))

    print(f"Using {device}")

    policy = build_pusht_policy(device, args.obs_dim, args.history)
    load_checkpoint(policy, ckpt_path, device)

    frames, success = rollout_episode(policy, args.max_steps, args.render_stride)

    tag = f"{ckpt_path.stem}_vis"
    save_visuals(frames, args.save_dir, tag, 10)
    print(f"Result: {'SUCCESS' if success else 'FAIL'}")


if __name__ == "__main__":
    main()
