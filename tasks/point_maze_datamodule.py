"""LightningDataModule for PointMaze Expert Data.

This module bridges the gap between the PointMaze environment and the generic
PolicyTraining module. It handles:
1. Collecting expert demonstrations (Oracle).
2. Chunking actions (Horizon > 1).
3. Fitting the Normalizer (but yielding RAW data to the policy).
"""

from __future__ import annotations

import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Dict

from environments.point_maze import PointMazeEnv
from diffusion_policy.model.common.normalizer import LinearNormalizer


class PointMazeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_episodes: int = 1000,
        chunk_size: int = 8,
        batch_size: int = 256,
        max_steps: int = 200,
        seed: int = 42,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_episodes = n_episodes
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seed = seed
        self.num_workers = num_workers

        self.normalizer = LinearNormalizer()
        self.train_dataset = None
        self.val_dataset = None

    def get_normalizer(self):
        """Exposes normalizer to PolicyTraining via setup hook."""
        return self.normalizer

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        # 1. Collect Expert Data
        env = PointMazeEnv(max_steps=self.max_steps, render_mode="rgb_array")
        all_ep_obs = []
        all_ep_acts = []

        print(f"Collecting {self.n_episodes} expert episodes for PointMaze...")
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_episodes):
            ep_seed = int(rng.integers(0, 100000))
            obs, _ = env.reset(seed=ep_seed)
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

        # 2. Create Chunked Dataset (RAW Data)
        obs_data, act_data = self._chunk_data(all_ep_obs, all_ep_acts)

        # 3. Fit Normalizer (But do NOT apply it here)
        # The Policy will apply normalization internally during training.
        self.normalizer.fit({"obs": obs_data, "action": act_data})
        print("Normalizer fitted on raw data.")

        # 4. Split Train/Val (Raw Data)
        dataset = TensorDataset(obs_data, act_data)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        print(
            f"Data prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val chunks."
        )

    def _chunk_data(self, obs_list, act_list):
        chunked_obs = []
        chunked_acts = []

        for ep_obs, ep_acts in zip(obs_list, act_list):
            T = len(ep_obs)
            for t in range(T):
                # Observation at t
                chunked_obs.append(ep_obs[t])

                # Action Chunk [t, t+k]
                start = t
                end = min(t + self.chunk_size, T)
                chunk = ep_acts[start:end]

                # Pad if at end of episode
                if len(chunk) < self.chunk_size:
                    pad = np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)

                # Flatten: (Chunk, D) -> (Chunk*D)
                chunked_acts.append(chunk.flatten())

        return (
            torch.tensor(np.stack(chunked_obs), dtype=torch.float32),
            torch.tensor(np.stack(chunked_acts), dtype=torch.float32),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
