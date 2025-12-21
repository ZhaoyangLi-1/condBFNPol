"""Compute simple mask-based metrics on PushT rollout videos.

Inspired by diffusion_policy's real_pusht_metrics script. Uses the last frame
of a reference video to derive a target mask, then measures IoU and coverage
per episode video and aggregates results.

Usage:
    python scripts/pusht_video_metrics.py \
        --reference path/to/reference.mp4 \
        --input-dir results/pusht_diffusion/<run>/videos \
        --camera-idx 0 \
        --n-workers 8 \
        --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import av
import cv2
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits


def get_t_mask(img: np.ndarray, hsv_ranges=None) -> np.ndarray:
    """Create a binary mask from HSV ranges (default tuned for PushT T-block)."""
    if hsv_ranges is None:
        hsv_ranges = [
            [0, 255],
            [130, 216],
            [150, 230],
        ]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.ones(img.shape[:2], dtype=bool)
    for c in range(len(hsv_ranges)):
        l, h = hsv_ranges[c]
        mask &= (l <= hsv_img[..., c])
        mask &= (hsv_img[..., c] <= h)
    return mask


def get_mask_metrics(target_mask: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    total = float(np.sum(target_mask))
    i = float(np.sum(target_mask & mask))
    u = float(np.sum(target_mask | mask))
    iou = i / u if u > 0 else 0.0
    coverage = i / total if total > 0 else 0.0
    return {"iou": iou, "coverage": coverage}


def get_video_metrics(video_path: Path, target_mask: np.ndarray, use_tqdm=True):
    threadpool_limits(1)
    cv2.setNumThreads(1)

    metrics = collections.defaultdict(list)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        iterator = tqdm(container.decode(stream), total=stream.frames) if use_tqdm else container.decode(stream)
        for frame in iterator:
            img = frame.to_ndarray(format="rgb24")
            mask = get_t_mask(img)
            metric = get_mask_metrics(target_mask=target_mask, mask=mask)
            for k, v in metric.items():
                metrics[k].append(v)
    return metrics


def worker(args):
    return get_video_metrics(*args)


def main():
    parser = argparse.ArgumentParser(description="Compute PushT video IoU/coverage metrics.")
    parser.add_argument("--reference", "-r", required=True, type=str, help="Reference video; last frame defines target mask.")
    parser.add_argument("--input-dir", "-i", required=True, type=str, help="Directory containing episode subdirs with <camera_idx>.mp4.")
    parser.add_argument("--camera-idx", "-c", default=0, type=int, help="Camera index filename to evaluate (default 0).")
    parser.add_argument("--n-workers", "-n", default=8, type=int, help="Parallel workers.")
    parser.add_argument("--wandb-mode", default="disabled", choices=["disabled", "online", "offline"])
    parser.add_argument("--wandb-project", default="dp-pusht-metrics")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    # Target mask from reference last frame
    with av.open(args.reference) as container:
        stream = container.streams.video[0]
        last_frame = None
        for frame in tqdm(container.decode(stream), total=stream.frames):
            last_frame = frame
    if last_frame is None:
        raise RuntimeError("Failed to read reference video.")
    last_img = last_frame.to_ndarray(format="rgb24")
    target_mask = get_t_mask(last_img)

    input_dir = Path(args.input_dir)
    episode_video_path_map = {}
    for vid_dir in input_dir.glob("*/"):
        if not vid_dir.is_dir():
            continue
        try:
            episode_idx = int(vid_dir.stem)
        except ValueError:
            continue
        video_path = vid_dir.joinpath(f"{args.camera_idx}.mp4")
        if video_path.exists():
            episode_video_path_map[episode_idx] = video_path

    episode_idxs = sorted(episode_video_path_map.keys())
    print(f"Found videos for episodes: {episode_idxs}")

    with mp.Pool(args.n_workers) as pool:
        worker_args = [(episode_video_path_map[idx], target_mask) for idx in episode_idxs]
        results = pool.map(worker, worker_args)

    episode_metric_map = {idx: result for idx, result in zip(episode_idxs, results)}

    # Aggregate
    agg_map = collections.defaultdict(list)
    for metric in episode_metric_map.values():
        for key, value in metric.items():
            agg_map[f"max/{key}"].append(np.max(value))
            agg_map[f"last/{key}"].append(value[-1])

    final_metric = {key: float(np.mean(val)) for key, val in agg_map.items()}

    # Save
    with input_dir.joinpath("metrics_agg.json").open("w") as f:
        json.dump(final_metric, f, sort_keys=True, indent=2)
    with input_dir.joinpath("metrics_raw.json").open("w") as f:
        json.dump(episode_metric_map, f, sort_keys=True, indent=2)
    print("Saved metrics to metrics_agg.json and metrics_raw.json")

    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
        )
        wandb_run.log(final_metric)
        wandb_run.finish()


if __name__ == "__main__":
    main()
