"""Standalone training script for GuidedBFNPolicy on PointMaze (Robust).

This script implements a high-capacity training loop for Flow Matching.
It includes explicit normalization checks and improved hyperparameters
to prevent mode collapse.

Usage:
    python scripts/train_guided_bfn.py
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
    HorizonConfig
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
        torch.tensor(np.stack(chunked_acts), dtype=torch.float32)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300) # Increased epochs
    parser.add_argument("--batch_size", type=int, default=1024) # Larger batch size
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--width", type=int, default=512) # Increased width
    parser.add_argument("--depth", type=int, default=6)   # Increased depth
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): DEVICE = "mps"
    
    set_seed(args.seed)
    
    print(f"--- Training Guided BFN (Flow Matching) on {DEVICE} ---")

    # --- 1. Data Collection ---
    # Increase step size slightly to make trajectories cleaner
    env = PointMazeEnv(max_steps=300, step_size=0.2, render_mode="rgb_array")
    
    print("Collecting Expert Data...")
    all_ep_obs, all_ep_acts = [], []
    
    # Collect 2000 episodes for dense coverage
    for _ in tqdm(range(2000), desc="Collecting"):
        obs, _ = env.reset()
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
        
        horizons=HorizonConfig(
            obs_history=1,
            prediction=args.chunk,
            execution=8
        ),
        guidance=GuidanceConfig(
            steps=args.steps,
            cfg_scale=1.0,
            grad_scale=0.0
        ),
        backbone_config=BackboneConfig(
            hidden_dim=args.width,
            depth=args.depth,
            time_emb_dim=128,
            conditioning_type="film",
            dropout=0.0
        ),
        device=DEVICE
    ).to(DEVICE)
    
    # --- 3. Fit Normalizer & Verify ---
    print("Fitting Normalizer...")
    policy.normalizer.fit({"obs": obs_data, "action": act_data})
    # Normalizer parameters are created on CPU during fit; align with policy device.
    policy._ensure_normalizer_device()
    
    # Verify normalization quality
    norm_acts = policy.normalizer['action'].normalize(act_data)
    print(f"Action Stats: Mean={act_data.mean():.3f}, Std={act_data.std():.3f}")
    print(f"Norm Stats:   Mean={norm_acts.mean():.3f}, Std={norm_acts.std():.3f}")
    if norm_acts.abs().max() > 1.1:
        print("[WARNING] Normalization bounds exceed [-1, 1]. Check data distribution.")

    train_ds = TensorDataset(obs_data, act_data)
    # Use multiple workers and pinned memory for speed
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Use 3e-4 LR, which is standard for Flow Matching
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- 4. Training Loop ---
    loss_history = []
    os.makedirs("results/guided", exist_ok=True)
    
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
            # Gradient Clipping is crucial for Flow Matching stability
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg = epoch_loss / len(train_loader)
        loss_history.append(avg)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

        # Eval every 20 epochs
        if (epoch + 1) % 5 == 0:
            evaluate(policy, env, epoch+1, args)

    # Save Final Artifacts
    plt.figure()
    plt.plot(loss_history)
    plt.title("Guided BFN Loss")
    plt.yscale('log')
    plt.savefig("results/guided/loss_history.png")
    plt.close()
    
    torch.save({
        'state_dict': policy.state_dict(),
        'config': args 
    }, f"results/guided/ckpt_final.pt")
    print(f"Training complete. Results in results/guided/")

def evaluate(policy, env, epoch, args):
    policy.eval()
    # Check multiple seeds to ensure robustness
    seeds = [100, 200, 300]
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        traj = [env.pos.copy()]
        done = False
        steps = 0
        success = False
        
        with torch.no_grad():
            while not done and steps < 300:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)
                
                # Forward -> Plan (ODE) -> Unnormalize
                action_chunk = policy(obs_t, deterministic=True)
                action = action_chunk[0, 0].cpu().numpy()
                
                obs, _, term, trunc, _ = env.step(action)
                traj.append(env.pos.copy())
                done = term or trunc
                steps += 1
                if term: success = True

        # Plot one representative trajectory
        if seed == seeds[0]:
            traj = np.array(traj)
            plt.figure()
            img = env.render()
            if img is not None: plt.imshow(img)
            plt.plot(traj[:,1]*10, traj[:,0]*10, 'c-', linewidth=2, label='Guided Flow')
            plt.scatter(traj[0,1]*10, traj[0,0]*10, c='g', marker='o')
            plt.scatter(traj[-1,1]*10, traj[-1,0]*10, c='r', marker='x')
            plt.legend()
            plt.title(f"Eval Epoch {epoch} - {'Success' if success else 'Fail'}")
            plt.savefig(f"results/guided/eval_ep{epoch}.png")
            plt.close()

if __name__ == "__main__":
    main()
