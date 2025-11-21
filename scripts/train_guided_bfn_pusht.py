"""Standalone training script for Guided BFN (Flow Matching) on PushT Image.

Pipeline:
1. Load Zarr Dataset (PushT Human Demonstrations).
2. Init Vision Encoder (ResNet18) + U-Net Backbone.
3. Train GuidedBFNPolicy.
4. Evaluate in Gym Environment.

Usage:
    python scripts/train_guided_bfn_pusht.py
"""

import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import collections

# Project Imports
from environments.pusht import PushTEnv
from policies.guided_bfn_policy import GuidedBFNPolicy, GuidanceConfig, HorizonConfig
from networks.unet import Unet
from networks.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


def get_resnet_backbone():
    """Gets a ResNet18 backbone, removing the final FC layer."""
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    # Remove FC, we want features
    model.fc = nn.Identity()
    return model


def main():
    # --- Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"

    CHUNK_SIZE = 16
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    ZARR_PATH = "data/pusht_cchi_v7_replay.zarr"

    print(f"--- Training Guided BFN (PushT) on {DEVICE} ---")

    # --- 1. Load Dataset ---
    if not os.path.exists(ZARR_PATH):
        print(f"Dataset not found at {ZARR_PATH}.")
        print(
            "Please download it: https://diffusion-policy.cs.columbia.edu/data/training/pusht_cchi_v7_replay.zarr.zip"
        )
        return

    print("Loading PushT Image Dataset...")
    dataset = PushTImageDataset(
        zarr_path=ZARR_PATH, horizon=CHUNK_SIZE, pad_before=1, pad_after=7
    )

    # Create Normalizer from dataset statistics
    normalizer = LinearNormalizer()
    # PushT dataset stats are usually pre-computed or we fit on a subset
    print("Fitting normalizer on dataset...")
    # Simple fit on a subset to avoid RAM explosion
    subset_indices = np.random.choice(
        len(dataset), min(1000, len(dataset)), replace=False
    )
    subset_batch = [dataset[i] for i in subset_indices]

    # Collate subset
    collate_obs = collections.defaultdict(list)
    collate_act = []
    for x in subset_batch:
        collate_act.append(x["action"])
        for k, v in x["obs"].items():
            collate_obs[k].append(v)

    # Fit only on low-dim fields to avoid nested-dict issues (images are not normalized here).
    fit_data = {"action": np.stack(collate_act)}
    if "agent_pos" in collate_obs:
        fit_data["agent_pos"] = np.stack(collate_obs["agent_pos"])
    normalizer.fit(fit_data)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- 2. Setup Components ---

    # Vision Encoder
    obs_dim = 128  # Project ResNet features to this dim
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
        output_dim=obs_dim,  # Linear projection output
        feature_aggregation=None,
    )

    # U-Net Backbone
    # Input: Flat action chunk [16*2 = 32]
    # Condition: Vision features [128]
    network = Unet(
        input_dim=2 * CHUNK_SIZE,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )

    # Policy
    policy = GuidedBFNPolicy(
        action_space=None,
        action_dim=2,
        network=network,
        obs_encoder=vision_encoder,
        normalizer_type="linear",
        horizons=HorizonConfig(
            obs_history=2,  # Dataset provides 2 frames history typically
            prediction=CHUNK_SIZE,
            execution=8,
        ),
        guidance=GuidanceConfig(steps=20),
        device=DEVICE,
    ).to(DEVICE)

    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 3. Training Loop ---
    loss_history = []
    os.makedirs("results/pusht", exist_ok=True)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        policy.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            # Batch is {'obs': {'image': ...}, 'action': ...}
            # Move to device
            batch["action"] = batch["action"].to(DEVICE, non_blocking=True)
            for k in batch["obs"]:
                batch["obs"][k] = batch["obs"][k].to(DEVICE, non_blocking=True)

            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg = epoch_loss / len(dataloader)
        loss_history.append(avg)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

        # Eval every 10 epochs
        if (epoch + 1) % 5 == 0:
            evaluate_pusht(policy, epoch + 1)

            # Save Checkpoint
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "normalizer": normalizer.state_dict(),
                },
                f"results/pusht/ckpt_ep{epoch+1}.pt",
            )

    plt.plot(loss_history)
    plt.savefig("results/pusht/loss.png")


def evaluate_pusht(policy, epoch):
    """Runs one episode in PushT env."""
    try:
        env = PushTEnv(render_mode="rgb_array")
    except Exception:
        print("Skipping eval: PushTEnv not found.")
        return

    policy.eval()
    obs, _ = env.reset()
    done = False
    images = []

    # Queue for observation history (if dataset uses n_obs_steps=2)
    # Simple approach: repeat first obs if needed, or policy handles it.
    # MultiImageObsEncoder expects [B, T, C, H, W] if n_obs_steps > 1
    # We will assume simple inference for now.

    with torch.no_grad():
        while not done and len(images) < 300:
            # Prepare Image Obs
            # env.reset returns {'image': [1, C, H, W], 'agent_pos': ...} (tensors from our wrapper)
            # or numpy dict depending on wrapper version.

            # Assuming standard Gym wrap: returns numpy dict
            # We need to tensorize
            img = torch.from_numpy(obs["image"]).float().to(policy.device)
            # [C, H, W] -> [1, 1, C, H, W] (Batch, Time, ...)
            # If dataset pad_before=1, it expects 2 frames.
            # Let's manually stack for simplicity:
            img = img.unsqueeze(0).unsqueeze(0)
            # If encoder expects 2 frames, we might need to repeat.
            # MultiImageObsEncoder handles [B, T, ...] by flattening to [B*T, ...]

            obs_dict = {
                "image": img,
                "agent_pos": torch.from_numpy(obs["agent_pos"])
                .float()
                .to(policy.device)
                .unsqueeze(0)
                .unsqueeze(0),
            }

            # Predict
            action_chunk = policy(obs_dict, deterministic=True)
            action = action_chunk[0, 0].cpu().numpy()

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            images.append(env.render())

    # Save video/gif
    print(f"Eval Epoch {epoch}: {len(images)} steps.")
    #
    # Save strip of images
    if len(images) > 0:
        strip = np.concatenate(images[::20], axis=1)  # Every 20th frame
        plt.figure(figsize=(15, 3))
        plt.imshow(strip)
        plt.title(f"PushT Eval Epoch {epoch}")
        plt.axis("off")
        plt.savefig(f"results/pusht/eval_ep{epoch}.png")
        plt.close()


if __name__ == "__main__":
    main()
