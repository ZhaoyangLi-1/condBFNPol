"""Flexible training script for ConditionalBFNPolicy on PointMaze.

Use this script to test different architectural combinations:
- Encoders: Identity vs MLP
- Conditioning: FiLM vs AdaLN vs Concat
- Normalizers: Linear vs Gaussian vs CDF

Usage Examples:
    # Baseline (Identity + FiLM + Linear)
    python scripts/train_conditional_bfn.py

    # MLP Encoder + AdaLN + Gaussian Normalizer
    python scripts/train_conditional_bfn.py --encoder mlp --cond adaln --norm gaussian

    # Concat Conditioning + CDF Normalizer
    python scripts/train_conditional_bfn.py --cond concat --norm cdf
"""

import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

# Project Imports
from environments.point_maze import PointMazeEnv
from policies.conditional_bfn_policy import (
    ConditionalBFNPolicy,
    BackboneConfig,
    BFNConfig,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Import factory to pre-fit normalizer if needed (though Policy handles it usually)
# We use Policy's internal normalizer management for simplicity.


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
    # Architecture Flags
    parser.add_argument(
        "--encoder",
        type=str,
        default="identity",
        choices=["identity", "mlp"],
        help="Observation Encoder",
    )
    parser.add_argument(
        "--enc_dim", type=int, default=64, help="Output dim for MLP encoder"
    )
    parser.add_argument(
        "--cond",
        type=str,
        default="film",
        choices=["film", "adaln", "concat"],
        help="Conditioning Type",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="linear",
        choices=["linear", "gaussian", "cdf"],
        help="Normalizer Type",
    )

    # Training Flags
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"\n--- Training Config ---")
    print(
        f"Encoder:    {args.encoder} (Dim: {args.enc_dim if args.encoder == 'mlp' else 'Raw'})"
    )
    print(f"Condition:  {args.cond}")
    print(f"Normalizer: {args.norm}")
    print(f"Backbone:   {args.width}w x {args.depth}d")
    print(f"Device:     {DEVICE}")
    print("-----------------------\n")

    # --- 1. Data Collection ---
    env = PointMazeEnv(max_steps=200, render_mode="rgb_array")
    print("Collecting Expert Data...")
    all_ep_obs, all_ep_acts = [], []

    # 500 episodes is usually enough for PointMaze with BFN
    for _ in tqdm(range(500), desc="Collecting"):
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

    # --- 2. Policy Setup ---
    # Determine Policy Obs Dim based on Encoder
    raw_obs_dim = 2
    policy_obs_dim = args.enc_dim if args.encoder == "mlp" else raw_obs_dim

    # Encoder Config (for factory)
    enc_config = {}
    if args.encoder == "mlp":
        enc_config = {"input_dim": raw_obs_dim, "output_dim": args.enc_dim}

    policy = ConditionalBFNPolicy(
        action_space=env.action_space,
        action_dim=2,
        obs_dim=policy_obs_dim,  # Policy sees encoded dim
        n_action_steps=args.chunk,
        # Factories
        normalizer_type=args.norm,
        obs_encoder_type=args.encoder,
        obs_encoder_config=enc_config,
        # Backbone Config
        backbone_config=BackboneConfig(
            hidden_dim=args.width,
            depth=args.depth,
            time_emb_dim=128,
            conditioning_type=args.cond,
            dropout=0.0,
        ),
        # BFN Config
        bfn_config=BFNConfig(
            sigma_1=args.sigma, n_timesteps=20, deterministic_seed=args.seed
        ),
        device=DEVICE,
    ).to(DEVICE)

    # --- 3. Fit Normalizer ---
    # We fit the policy's internal normalizer manually here
    # because we aren't using the DataModule pipeline.
    print("Fitting Normalizer...")
    policy.normalizer.fit({"obs": obs_data, "action": act_data})

    # Create Dataset (Apply normalization for training efficiency?
    # No, let policy handle it internally in forward/compute_loss for safety test)
    # But for speed, pre-normalization is better. Let's stick to Policy Internal (raw data).
    train_ds = TensorDataset(obs_data, act_data)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 4. Training ---
    loss_history = []

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_obs, batch_acts in pbar:
            batch_obs = batch_obs.to(DEVICE)
            batch_acts = batch_acts.to(DEVICE)

            # Pass raw batch (tuple) to compute_loss
            # Policy will Normalize -> Encode -> Loss
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

        # Quick Eval
        if (epoch + 1) % 5 == 0:
            evaluate(policy, env, epoch + 1, args)

    # Save Artifacts
    os.makedirs("results", exist_ok=True)
    run_name = f"{args.encoder}_{args.cond}_{args.norm}"

    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Loss: {run_name}")
    plt.savefig(f"results/loss_{run_name}.png")
    plt.close()

    # Save Checkpoint (for full eval script)
    torch.save(
        {
            "state_dict": policy.state_dict(),  # Contains normalizer params too
            "config": args,
        },
        f"results/ckpt_{run_name}.pt",
    )
    print(f"Saved checkpoint to results/ckpt_{run_name}.pt")


def evaluate(policy, env, epoch, args):
    policy.eval()
    obs, _ = env.reset(seed=123)
    traj = [env.pos.copy()]
    done = False
    steps = 0

    with torch.no_grad():
        while not done and steps < 200:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(policy.device)
            action_chunk = policy(obs_t, deterministic=True)
            action = action_chunk[0, 0].cpu().numpy()

            obs, _, term, trunc, _ = env.step(action)
            traj.append(env.pos.copy())
            done = term or trunc
            steps += 1

    traj = np.array(traj)
    plt.figure()
    img = env.render()
    if img is not None:
        plt.imshow(img)
    plt.plot(traj[:, 1] * 10, traj[:, 0] * 10, "w-", linewidth=2)
    plt.title(f"Eval {epoch}: {args.encoder}/{args.cond}/{args.norm}")
    plt.savefig(f"results/eval_{args.encoder}_{args.cond}_{args.norm}_ep{epoch}.png")
    plt.close()


if __name__ == "__main__":
    main()
