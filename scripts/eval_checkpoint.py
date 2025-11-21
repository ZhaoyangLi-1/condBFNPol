"""Evaluation script for BFN Policy Checkpoints.

This script loads a trained Lightning checkpoint (.ckpt), reconstructs the 
BFN Policy and Network, and runs evaluation episodes with visualization.

Usage:
    # If trained with default settings (512 width, 4 depth)
    python scripts/eval_checkpoint.py --ckpt checkpoints/last.ckpt

    # If trained with "Stabilized" settings (256 width, 3 depth)
    python scripts/eval_checkpoint.py --ckpt checkpoints/last.ckpt --width 256 --depth 3
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import project modules
from environments.point_maze import PointMazeEnv
from policies.bfn_policy import BFNPolicy, BFNConfig
from networks.resnet import Resnet
from diffusion_policy.model.common.normalizer import LinearNormalizer


def load_policy_from_checkpoint(
    ckpt_path, device="cpu", hidden_dim=512, depth=4, chunk_size=8
):
    """Loads BFNPolicy from a Lightning checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    print(
        f"Reconstructing Network with: Width={hidden_dim}, Depth={depth}, Chunk={chunk_size}"
    )

    # weights_only=False required for OmegaConf/DictConfig globals in Lightning ckpts
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint file: {e}")
        sys.exit(1)

    # 1. Reconstruct Architecture
    action_dim = 2
    obs_dim = 2
    flat_action_dim = action_dim * chunk_size

    network = Resnet(
        dim=flat_action_dim,
        cond_dim=obs_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        time_dim=128,
        dropout=0.0,
    )

    policy = BFNPolicy(
        action_space=None,
        network=network,
        action_dim=action_dim,
        n_action_steps=chunk_size,
        obs_encoder=None,
        config=BFNConfig(sigma_1=0.005, n_timesteps=20, deterministic_seed=42),
        device=device,
    )

    # 2. Load Weights
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("policy."):
            new_key = k.replace("policy.", "", 1)
            new_state_dict[new_key] = v

    try:
        keys = policy.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print("\n[ERROR] Architecture Mismatch!")
        print(f"The checkpoint architecture does not match the eval arguments.")
        print(f"Error details: {e}")
        print("\nSUGGESTION: Try running with different --width or --depth flags.")
        print(
            f"Example: python scripts/eval_checkpoint.py --ckpt {ckpt_path} --width 256 --depth 3"
        )
        sys.exit(1)

    # 3. Verify Normalizer
    if len(policy.normalizer.params_dict) == 0:
        print(
            "[WARNING] Normalizer is empty! Checkpoint might not contain normalization stats."
        )
    else:
        print(
            f"Normalizer stats loaded successfully (Keys: {list(policy.normalizer.params_dict.keys())})."
        )

    policy.eval()
    policy.to(device)
    return policy


def run_eval_episode(policy, env, seed=42, chunk_size=8):
    """Runs a single episode using Receding Horizon Control."""
    obs, _ = env.reset(seed=seed)
    traj = [env.pos.copy()]

    done = False
    steps = 0
    success = False  # Track success locally

    with torch.no_grad():
        while not done and steps < 300:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)

            # Forward
            action_chunk = policy(obs_t, deterministic=True)

            # Execute First Action
            next_action = action_chunk[0, 0].cpu().numpy()

            obs, _, term, trunc, _ = env.step(next_action)
            traj.append(env.pos.copy())

            done = term or trunc
            steps += 1

            if term:
                success = True  # Goal reached

    return np.array(traj), success


def visualize_trajectory(env, traj, save_path="eval_vis.png"):
    plt.figure(figsize=(6, 6))

    img = env.render()
    if img is not None:
        plt.imshow(img)

    # Swap x/y for plotting on image (row=y, col=x)
    plt.plot(traj[:, 1] * 10, traj[:, 0] * 10, "w-", linewidth=3, label="Path")

    plt.scatter(
        traj[0, 1] * 10,
        traj[0, 0] * 10,
        c="g",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    plt.scatter(
        traj[-1, 1] * 10,
        traj[-1, 0] * 10,
        c="r",
        s=100,
        marker="x",
        label="End",
        zorder=5,
    )

    plt.legend()
    plt.title("Checkpoint Evaluation")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--width", type=int, default=512, help="Network hidden dim")
    parser.add_argument("--depth", type=int, default=4, help="Network depth")
    parser.add_argument("--chunk", type=int, default=8, help="Action chunk size")
    parser.add_argument("--seed", type=int, default=123, help="Eval seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = load_policy_from_checkpoint(
        args.ckpt,
        device=device,
        hidden_dim=args.width,
        depth=args.depth,
        chunk_size=args.chunk,
    )

    env = PointMazeEnv(max_steps=300, render_mode="rgb_array")

    print(f"Running evaluation with seed {args.seed}...")
    traj, success = run_eval_episode(policy, env, seed=args.seed, chunk_size=args.chunk)

    result_str = "SUCCESS" if success else "FAIL"
    print(f"Result: {result_str}")

    save_name = f"eval_{Path(args.ckpt).stem}_seed{args.seed}.png"
    visualize_trajectory(env, traj, save_path=save_name)


if __name__ == "__main__":
    main()
