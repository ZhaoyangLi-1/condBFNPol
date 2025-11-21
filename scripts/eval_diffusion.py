"""Evaluation script for Diffusion Policy Checkpoints on PointMaze.

Loads a trained Diffusion Policy checkpoint, runs inference (denoising),
and visualizes the resulting trajectories.

Usage:
    python scripts/eval_diffusion.py --ckpt checkpoints/last.ckpt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm  # Added for visibility

# Project Imports
from environments.point_maze import PointMazeEnv
from policies.diffusion_policy import DiffusionPolicy, HorizonConfig, InferenceConfig
from networks.unet import Unet
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def load_policy_from_checkpoint(ckpt_path, device="cpu", num_inference_steps=100):
    """Reconstructs DiffusionPolicy from a Lightning checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)

    # 1. Config & Hyperparams (Must match training config!)
    action_dim = 2
    obs_dim = 2
    chunk_size = 16

    # 2. Reconstruct Components
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )

    model = Unet(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )

    policy = DiffusionPolicy(
        action_space=None,
        model=model,
        noise_scheduler=noise_scheduler,
        action_dim=action_dim,
        obs_dim=obs_dim,
        # Config Objects
        horizons=HorizonConfig(
            planning_horizon=chunk_size,
            obs_history=1,
            action_prediction=chunk_size,
            execution=1,
        ),
        inference=InferenceConfig(
            num_steps=num_inference_steps,  # Allow overriding steps for speed
            condition_mode="global",
            pred_action_steps_only=False,
        ),
        device=device,
    )

    # 3. Load Weights
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
        print(f"[ERROR] Architecture Mismatch: {e}")
        sys.exit(1)

    # 4. Check Normalizer
    if len(policy.normalizer.params_dict) == 0:
        print("[WARNING] Normalizer is empty! Checkpoint might be invalid.")
    else:
        print("Normalizer stats loaded.")

    policy.eval()
    policy.to(device)
    return policy


def run_eval_episode(policy, env, seed=42):
    obs, _ = env.reset(seed=seed)
    traj = [env.pos.copy()]

    done = False
    steps = 0
    success = False

    print(f"Simulating Episode (Seed {seed})...")
    # Added Progress Bar
    with tqdm(total=300, desc="Steps") as pbar:
        with torch.no_grad():
            while not done and steps < 300:
                # Prepare Obs
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)

                # Forward (Predict Action Chunk)
                # The updated Policy logic handles moving tensors internally if implemented robustly,
                # but let's ensure the policy wrapper call is safe.
                # Policy forward calls predict_action -> conditional_sample.
                # We rely on the policy class updates we made.

                action_seq = policy(obs_t, deterministic=True)

                # To Numpy
                action_seq_np = action_seq.cpu().numpy()

                # Receding Horizon Control: Execute ONLY the first action
                if action_seq_np.ndim > 1:
                    next_action = action_seq_np[0]
                else:
                    next_action = action_seq_np

                # Step
                obs, _, term, trunc, _ = env.step(next_action)
                traj.append(env.pos.copy())

                done = term or trunc
                steps += 1
                pbar.update(1)

                if term:
                    success = True

    return np.array(traj), success


def visualize_trajectory(env, traj, save_path="eval_diff_vis.png"):
    plt.figure(figsize=(6, 6))

    img = env.render()
    if img is not None:
        plt.imshow(img)

    plt.plot(
        traj[:, 1] * 10, traj[:, 0] * 10, "c-", linewidth=3, label="Diffusion Path"
    )
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
    plt.title("Diffusion Policy Evaluation")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=123, help="Eval seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Diffusion inference steps (reduce for speed)",
    )
    args = parser.parse_args()

    # Select best device (MPS for Mac, CUDA for Nvidia, CPU otherwise)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS acceleration (Mac).")
    else:
        device = "cpu"
        print("Using CPU (Expect slow generation).")

    policy = load_policy_from_checkpoint(
        args.ckpt, device=device, num_inference_steps=args.steps
    )
    env = PointMazeEnv(max_steps=300, render_mode="rgb_array")

    traj, success = run_eval_episode(policy, env, seed=args.seed)

    result_str = "SUCCESS" if success else "FAIL"
    print(f"Result: {result_str}")

    save_name = f"eval_diffusion_{Path(args.ckpt).stem}_seed{args.seed}.png"
    visualize_trajectory(env, traj, save_path=save_name)


if __name__ == "__main__":
    main()
