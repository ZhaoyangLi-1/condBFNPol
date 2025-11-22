"""Standalone training script for GuidedBFNPolicy on PointMaze (Stabilized).

Changes:
1. Saves 'best' checkpoint based on eval success.
2. Increases data diversity (random starts).
3. Tuned hyperparameters for stability.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import sys

# Project Imports
from environments.point_maze import PointMazeEnv
from policies.guided_bfn_policy import (
    GuidedBFNPolicy,
    BackboneConfig,
    GuidanceConfig,
    HorizonConfig,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_chunked_dataset(obs_list, act_list, chunk_size=8):
    chunked_obs, chunked_acts = [], []
    for ep_obs, ep_acts in zip(obs_list, act_list):
        T = len(ep_obs)
        for t in range(T):
            chunked_obs.append(ep_obs[t])
            start = t
            end = min(t + chunk_size, T)
            chunk = ep_acts[start:end]
            if len(chunk) < chunk_size:
                pad = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            chunked_acts.append(chunk.flatten())
    return (
        torch.tensor(np.stack(chunked_obs), dtype=torch.float32),
        torch.tensor(np.stack(chunked_acts), dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    set_seed(args.seed)

    print(f"--- Training Guided BFN (Flow Matching) on {DEVICE} ---")

    # --- 1. Data Collection ---
    # Use a separate env for data collection to allow randomization
    env = PointMazeEnv(max_steps=300, step_size=0.2, render_mode="rgb_array")

    print("Collecting Expert Data (Randomized Starts)...")
    all_ep_obs, all_ep_acts = [], []

    # Collect 2000 episodes with randomized seeds for diversity
    rng = np.random.default_rng(args.seed)
    for _ in tqdm(range(2000), desc="Collecting"):
        ep_seed = int(rng.integers(0, 100000))
        obs, _ = env.reset(seed=ep_seed)  # Random start
        done = False
        ep_obs, ep_acts = [], []

        while not done:
            action = env.get_expert_action()
            ep_obs.append(obs)
            ep_acts.append(action)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        if len(ep_obs) > 0:
            all_ep_obs.append(np.stack(ep_obs))
            all_ep_acts.append(np.stack(ep_acts))

    obs_data, act_data = create_chunked_dataset(all_ep_obs, all_ep_acts, args.chunk)
    print(f"Dataset Size: {len(obs_data)} transitions")

    # --- 2. Policy Setup ---
    policy = GuidedBFNPolicy(
        action_space=env.action_space,
        action_dim=2,
        normalizer_type="linear",
        obs_encoder_type="identity",
        horizons=HorizonConfig(obs_history=1, prediction=args.chunk, execution=8),
        guidance=GuidanceConfig(steps=args.steps),
        backbone_config=BackboneConfig(
            hidden_dim=args.width,
            depth=args.depth,
            time_emb_dim=128,
            conditioning_type="film",
            dropout=0.1,  # Add dropout for regularization
        ),
        device=DEVICE,
    ).to(DEVICE)

    # --- 3. Fit Normalizer ---
    print("Fitting Normalizer...")
    policy.normalizer.fit({"obs": obs_data, "action": act_data})

    train_ds = TensorDataset(obs_data, act_data)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Lower LR and higher weight decay for stability
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 4. Training Loop ---
    loss_history = []
    os.makedirs("results/guided", exist_ok=True)
    best_success_rate = -1.0

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_obs, batch_acts in pbar:
            batch_obs = batch_obs.to(DEVICE)
            batch_acts = batch_acts.to(DEVICE)

            loss = policy.compute_loss((batch_obs, batch_acts))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        loss_history.append(avg)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0:
            success_rate = evaluate(policy, env, epoch + 1, args)

            # Save Best Checkpoint
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                torch.save(
                    {"state_dict": policy.state_dict(), "config": args},
                    f"results/guided/ckpt_best.pt",
                )
                print(f"New Best Model Saved! (Success Rate: {success_rate:.2f})")

    # Save Final
    torch.save(
        {"state_dict": policy.state_dict(), "config": args},
        f"results/guided/ckpt_final.pt",
    )

    plt.figure()
    plt.plot(loss_history)
    plt.title("Guided BFN Loss")
    plt.yscale("log")
    plt.savefig("results/guided/loss_history.png")
    plt.close()


def evaluate(policy, env, epoch, args):
    policy.eval()
    seeds = [100, 200, 300, 400, 500]  # 5 evaluation episodes
    successes = 0

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        traj = [env.pos.copy()]
        done = False
        steps = 0

        with torch.no_grad():
            while not done and steps < 300:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)
                action_chunk = policy(obs_t, deterministic=True)
                action = action_chunk[0, 0].cpu().numpy()

                obs, _, term, trunc, _ = env.step(action)
                traj.append(env.pos.copy())
                done = term or trunc
                steps += 1
                if term:
                    successes += 1

        # Plot the first seed only
        if seed == seeds[0]:
            traj = np.array(traj)
            plt.figure()
            img = env.render()
            if img is not None:
                plt.imshow(img)
            plt.plot(traj[:, 1] * 10, traj[:, 0] * 10, "c-", linewidth=2)
            plt.title(f"Eval Epoch {epoch}")
            plt.savefig(f"results/guided/eval_ep{epoch}.png")
            plt.close()

    rate = successes / len(seeds)
    print(f"Eval Success Rate: {rate:.2f}")
    return rate


if __name__ == "__main__":
    main()
