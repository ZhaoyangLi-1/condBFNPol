"""Standalone training script for BFN on PointMaze (Robust).

Changes for Success:
1. Architecture: Hidden Dim 1024, Depth 6 (High Capacity).
2. Training: 500 Epochs (BFNs need long training).
3. Sigma: 0.01 (Standard stability value).
4. Timesteps: 100 (Better inference precision).
5. Batch Size: 1024 (Stable gradients).
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import random

# Project Imports
from environments.point_maze import PointMazeEnv
from policies.bfn_policy import BFNPolicy, BFNConfig
from networks.resnet import Resnet
from diffusion_policy.model.common.normalizer import LinearNormalizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_chunked_dataset(obs_list, act_list, chunk_size=8):
    """Converts lists of episodes into chunked sequence data."""
    chunked_obs = []
    chunked_acts = []

    for ep_obs, ep_acts in zip(obs_list, act_list):
        T = len(ep_obs)
        for t in range(T):
            chunked_obs.append(ep_obs[t])

            # Actions from t to t + chunk_size
            start = t
            end = min(t + chunk_size, T)
            chunk = ep_acts[start:end]

            # Pad if chunk is too short
            if len(chunk) < chunk_size:
                pad_len = chunk_size - len(chunk)
                last_act = chunk[-1:]
                padding = np.repeat(last_act, pad_len, axis=0)
                chunk = np.concatenate([chunk, padding], axis=0)

            chunked_acts.append(chunk.flatten())

    return (
        torch.tensor(np.stack(chunked_obs), dtype=torch.float32),
        torch.tensor(np.stack(chunked_acts), dtype=torch.float32),
    )


def main():
    # --- 1. Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ACTION_DIM = 2
    OBS_DIM = 2
    CHUNK_SIZE = 16  # Increased context
    FLAT_ACTION_DIM = ACTION_DIM * CHUNK_SIZE

    BATCH_SIZE = 1024  # Large batch for stable gradients
    EPOCHS = 300  # Needs time to converge
    LR = 2e-4
    SIGMA_1 = 0.01  # Standard robust value
    TIMESTEPS = 50  # More inference steps

    set_seed(42)
    print(f"Training on {DEVICE} | Chunk={CHUNK_SIZE} | Batch={BATCH_SIZE}")

    # --- 2. Data Collection ---
    # Increase step size slightly to make expert trajectories cleaner
    env = PointMazeEnv(max_steps=300, step_size=0.2, render_mode="rgb_array")

    print("Collecting Expert Demonstrations...")
    all_ep_obs = []
    all_ep_acts = []

    # Collect 2000 episodes for dense coverage
    for _ in tqdm(range(2000), desc="Collecting Data"):
        obs, _ = env.reset()
        done = False
        ep_obs = []
        ep_acts = []

        while not done:
            action = env.get_expert_action()
            ep_obs.append(obs)
            ep_acts.append(action)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        if len(ep_obs) > 0:
            all_ep_obs.append(np.stack(ep_obs))
            all_ep_acts.append(np.stack(ep_acts))

    print("Chunking dataset...")
    obs_data, act_data = create_chunked_dataset(all_ep_obs, all_ep_acts, CHUNK_SIZE)
    print(f"Dataset Size: {len(obs_data)} transitions")

    # --- 3. Normalization ---
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": obs_data, "action": act_data})
    norm_data = normalizer.normalize({"obs": obs_data, "action": act_data})

    train_ds = TensorDataset(norm_data["obs"], norm_data["action"])
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- 4. Policy Setup ---
    # High-Capacity ResNet
    network = Resnet(
        dim=FLAT_ACTION_DIM,
        cond_dim=OBS_DIM,
        hidden_dim=1024,  # Very wide
        depth=6,  # Very deep
        time_dim=256,
        dropout=0.1,  # Slight regularization
    )

    policy = BFNPolicy(
        action_space=env.action_space,
        network=network,
        action_dim=FLAT_ACTION_DIM,
        obs_encoder=None,
        config=BFNConfig(
            sigma_1=SIGMA_1,
            n_timesteps=TIMESTEPS,  # Slower but more accurate inference
            deterministic_seed=42,
        ),
        device=DEVICE,
    ).to(DEVICE)

    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # --- 5. Training Loop ---
    loss_history = []
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        policy.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch_obs, batch_acts in pbar:
            batch_obs = batch_obs.to(DEVICE)
            batch_acts = batch_acts.to(DEVICE)

            cond = policy._encode_obs(batch_obs)

            # BFN Loss
            loss = policy.bfn.loss(batch_acts, cond=cond, sigma_1=SIGMA_1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

        # Validation every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Save checkpoint
            ckpt_path = f"checkpoints/bfn_maze_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": normalizer.state_dict(),  # Save normalizer explicitly
                },
                ckpt_path,
            )

            evaluate_agent(policy, env, epoch + 1, chunk_size=CHUNK_SIZE)

    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.title("BFN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.savefig("training_loss.png")
    print("Saved loss plot.")


def evaluate_agent(policy, env, epoch, chunk_size):
    policy.eval()
    seeds = [100, 200, 300]  # Test different starts

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        traj = [env.pos.copy()]
        done = False
        steps = 0
        success = False

        with torch.no_grad():
            while not done and steps < 300:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)
                if policy.normalizer is not None:
                    obs_t = policy.normalizer["obs"].normalize(obs_t)

                cond = policy._encode_obs(obs_t)

                # Sample
                action_flat = policy._sample_actions(
                    cond, cond_scale=None, rescaled_phi=None
                )

                if policy.normalizer is not None:
                    action_flat = policy.normalizer["action"].unnormalize(action_flat)

                # Reshape [1, Chunk*Dim] -> [Chunk, Dim]
                action_chunk = action_flat.view(chunk_size, 2)

                # Receding Horizon: Execute first action
                next_action = action_chunk[0].cpu().numpy()

                obs, _, term, trunc, _ = env.step(next_action)
                traj.append(env.pos.copy())
                done = term or trunc
                steps += 1
                if term:
                    success = True

        print(
            f"--> Eval Epoch {epoch} (Seed {seed}): {'SUCCESS' if success else 'FAIL'} (Steps: {steps})"
        )

        # Visualize
        traj = np.array(traj)
        plt.figure()
        img = env.render()
        if img is not None:
            plt.imshow(img)
        plt.plot(traj[:, 1] * 10, traj[:, 0] * 10, "r-", linewidth=2)
        plt.scatter(traj[0, 1] * 10, traj[0, 0] * 10, c="g", marker="o", label="Start")
        plt.scatter(traj[-1, 1] * 10, traj[-1, 0] * 10, c="r", marker="x", label="End")
        plt.title(f"Epoch {epoch} Seed {seed} - {'Success' if success else 'Fail'}")
        plt.savefig(f"eval_epoch_{epoch}_seed_{seed}.png")
        plt.close()


if __name__ == "__main__":
    main()
