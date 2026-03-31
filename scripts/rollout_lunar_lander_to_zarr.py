"""
Rollout + Expert Quality Visualization for Hybrid SAC LunarLander
=================================================================

Two modes:

1. Rollout → zarr + figures (default):
    python scripts/rollout_and_visualize.py \
        --model-path /path/to/best_model.pt \
        --output-path /path/to/replay.zarr \
        --output-dir  /path/to/figures \
        --n-episodes 200 \
        --n-traj 15

2. Figures only from existing zarr:
    python scripts/rollout_and_visualize.py \
        --zarr-path /path/to/replay.zarr \
        --output-dir /path/to/figures

Zarr structure saved:
    data/
        action                  (N, 2)       float32  low-level env action
        img                     (N, H, W, 3) float32  RGB in [0, 1]
        state                   (N, 8)       float32  state vector
        hybrid_action_discrete  (N,)         int64    discrete branch id k
        hybrid_action_params    (N, 2)       float32  padded x_k
        hybrid_action_flat      (N, D)       float32  flat actor output
    meta/
        episode_ends            (n_episodes,) int64
"""

from __future__ import annotations

# ── MuJoCo / gymnasium_robotics patch ────────────────────────────────────────
# Must appear before ANY `import gymnasium` to prevent mujoco_py crash.
import importlib.metadata as _im
_orig_ep = _im.entry_points
def _filtered_ep(*a, **kw):
    eps = _orig_ep(*a, **kw)
    if kw.get("group") != "gymnasium.envs":
        return eps
    filtered = [e for e in eps if "gymnasium_robotics" not in getattr(e, "value", "")]
    try:
        return type(eps)(filtered)
    except Exception:
        return filtered
_im.entry_points = _filtered_ep
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
# Color palette  (white background)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg":       "#ffffff",
    "bg2":      "#f5f5f7",
    "border":   "#d0d0d8",
    "text":     "#1a1a2e",
    "muted":    "#6b6b80",
    "accent":   "#5a4fcf",
    "teal":     "#0d9488",
    "amber":    "#d97706",
    "coral":    "#ea580c",
    "rose":     "#e11d48",
    "green":    "#16a34a",
}

ACTION_COLORS = {
    0: PALETTE["muted"],
    1: PALETTE["teal"],
    2: PALETTE["amber"],
    3: PALETTE["coral"],
}
ACTION_NAMES = {0: "coast", 1: "main engine", 2: "left boost", 3: "right boost"}


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg2"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.6,
        "grid.alpha":        0.8,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "legend.facecolor":  PALETTE["bg"],
        "legend.edgecolor":  PALETTE["border"],
        "legend.labelcolor": PALETTE["text"],
        "figure.dpi":        150,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Policy + env loading helpers (shared by rollout and visualize)
# ══════════════════════════════════════════════════════════════════════════════

def _load_env_and_policy(model_path: str, use_image_obs: bool,
                          img_size: tuple, sub_steps: int, device):
    """Load HybridLunarLander and Policy directly, bypassing __init__.py."""
    import torch
    import importlib.util as _ilu

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load HybridLunarLander directly to skip environments/__init__.py
    ll_path = os.path.join(project_root, "environments", "lunar_lander.py")
    spec = _ilu.spec_from_file_location("environments.lunar_lander", ll_path)
    ll_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(ll_mod)
    HybridLunarLander = ll_mod.HybridLunarLander

    from scripts.train_hybrid_sac_lunar_lander import HybridActionLayout, Policy

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    network_cfg = cfg.get("network", {})
    sac_cfg = cfg.get("sac", {})

    env = HybridLunarLander(
        use_image_obs=use_image_obs,
        img_size=tuple(img_size),
        sub_steps=sub_steps,
    )
    action_layout = HybridActionLayout.from_env_spec(env.get_action_spec())

    # Support both old (state obs) and new (dict obs with "state" key)
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        obs_dim = int(obs_space["state"].shape[0])
    else:
        obs_dim = int(obs_space.shape[0])

    policy = Policy(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(network_cfg.get("hidden_dims", [256, 256])),
        activation=network_cfg.get("activation", "relu"),
        weights_init=network_cfg.get("weights_init", "xavier"),
        bias_init=network_cfg.get("bias_init", "zeros"),
        log_std_min=float(sac_cfg.get("log_std_min", -5.0)),
        log_std_max=float(sac_cfg.get("log_std_max", 0.0)),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    return env, policy, action_layout, ckpt


# ══════════════════════════════════════════════════════════════════════════════
# Zarr rollout
# ══════════════════════════════════════════════════════════════════════════════

def _filter_episodes(arrays, episode_ends, episode_rewards, min_reward):
    keep = {k: [] for k in arrays}
    keep_ends, keep_rewards = [], []
    prev, kept = 0, 0
    for idx, end in enumerate(episode_ends):
        if episode_rewards[idx] >= min_reward:
            for k, v in arrays.items():
                keep[k].extend(v[prev:end])
            kept += end - prev
            keep_ends.append(kept)
            keep_rewards.append(episode_rewards[idx])
        prev = end
    if not keep_ends:
        print("WARNING: no episodes passed filter — saving all.")
        return arrays, episode_ends, episode_rewards
    return keep, keep_ends, keep_rewards


def rollout_to_zarr(args) -> dict:
    """Roll out policy, save zarr, AND return viz data — single pass, same seeds."""
    import torch
    import zarr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub_steps = args.sub_steps if args.sub_steps is not None else 1
    env, policy, action_layout, _ = _load_env_and_policy(
        model_path=args.model_path,
        use_image_obs=True,
        img_size=tuple(args.img_size),
        sub_steps=sub_steps,
        device=device,
    )

    rng = np.random.default_rng(args.collect_seed)
    arrays = {k: [] for k in
              ["action", "img", "state",
               "hybrid_action_discrete", "hybrid_action_params", "hybrid_action_flat"]}
    episode_ends: list[int] = []
    episode_rewards: list[float] = []
    episode_seeds: list[int] = []
    total_steps = 0

    # Viz accumulators — built during the SAME rollout loop
    trajectories: list = []          # (x, y) arrays for first n_traj episodes
    all_discrete: list = []
    all_params_k1: list = []
    all_params_k2: list = []
    all_params_k3: list = []
    ep_action_seqs: list = []

    print(f"Collecting {args.n_episodes} episodes → {args.output_path}")

    for ep in range(args.n_episodes):
        seed = abs(int(rng.normal(args.seed_mean, args.seed_std)))
        episode_seeds.append(seed)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        ep_reward = 0.0
        ep_steps = 0
        ep_traj_x: list = []
        ep_traj_y: list = []
        ep_actions: list = []

        while not (terminated or truncated):
            if isinstance(obs, dict):
                state = obs["state"].astype(np.float32)
                image = obs["image"].astype(np.float32) / 255.0
            else:
                state = obs.astype(np.float32)
                image = np.zeros((*args.img_size, 3), dtype=np.float32)

            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                flat_action_t, discrete_action_t = policy.act(
                    state_t, deterministic=args.deterministic)

            discrete_action = int(discrete_action_t.item())
            flat_action = flat_action_t.cpu().numpy()[0].astype(np.float32)
            hybrid_action = action_layout.flat_to_env_action(discrete_action, flat_action)
            low_level = env._convert_action(
                discrete_action, hybrid_action["x_k"]).astype(np.float32)

            # zarr arrays
            arrays["state"].append(state)
            arrays["img"].append(image)
            arrays["action"].append(low_level)
            arrays["hybrid_action_discrete"].append(discrete_action)
            arrays["hybrid_action_params"].append(hybrid_action["x_k"].astype(np.float32))
            arrays["hybrid_action_flat"].append(flat_action)

            # viz accumulators
            ep_traj_x.append(float(state[0]))
            ep_traj_y.append(float(state[1]))
            ep_actions.append(discrete_action)
            all_discrete.append(discrete_action)
            xk = hybrid_action["x_k"]
            if discrete_action == 1:
                all_params_k1.append(float(xk[0]))
            elif discrete_action == 2:
                all_params_k2.append([float(xk[0]), float(xk[1])])
            elif discrete_action == 3:
                all_params_k3.append([float(xk[0]), float(xk[1])])

            obs, reward, terminated, truncated, _ = env.step(hybrid_action)
            ep_reward += reward
            ep_steps += 1

        total_steps += ep_steps
        episode_ends.append(total_steps)
        episode_rewards.append(ep_reward)
        ep_action_seqs.append(ep_actions)
        if ep < args.n_traj:
            trajectories.append((np.array(ep_traj_x), np.array(ep_traj_y)))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:>4d}/{args.n_episodes} | seed={seed:>10d} | "
                  f"steps={ep_steps:>4d} | reward={ep_reward:>8.1f} | total={total_steps}")

    env.close()

    # ── Optional reward filter ────────────────────────────────────────────────
    if args.min_reward is not None:
        print(f"\nFiltering episodes with reward >= {args.min_reward} ...")
        arrays, episode_ends, episode_rewards = _filter_episodes(
            arrays, episode_ends, episode_rewards, args.min_reward)
        total_steps = episode_ends[-1] if episode_ends else 0
        print(f"  Kept {len(episode_ends)}/{args.n_episodes} episodes, {total_steps} steps")

    if not episode_ends:
        raise RuntimeError("No episodes collected.")

    # ── Save zarr ─────────────────────────────────────────────────────────────
    np_arrays = {
        "action":                 np.stack(arrays["action"]).astype(np.float32),
        "img":                    np.stack(arrays["img"]).astype(np.float32),
        "state":                  np.stack(arrays["state"]).astype(np.float32),
        "hybrid_action_discrete": np.asarray(arrays["hybrid_action_discrete"], dtype=np.int64),
        "hybrid_action_params":   np.stack(arrays["hybrid_action_params"]).astype(np.float32),
        "hybrid_action_flat":     np.stack(arrays["hybrid_action_flat"]).astype(np.float32),
    }
    episode_ends_np = np.asarray(episode_ends, dtype=np.int64)
    chunk_len = max(int(np.median(np.diff(np.concatenate([[0], episode_ends_np])))), 1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_path), mode="w")
    root.attrs.update({
        "policy_type":    "hybrid_sac",
        "action_names":   [action_layout.action_names[k]
                           for k in range(action_layout.num_discrete)],
        "flat_action_dim": int(action_layout.total_param_dim),
        "img_size":        list(args.img_size),
        "episode_seeds":   episode_seeds,       # store seeds for reproducibility
    })
    dg = root.create_group("data")
    dg.create_dataset("action",
        data=np_arrays["action"],
        chunks=(chunk_len, np_arrays["action"].shape[1]), dtype="float32")
    dg.create_dataset("img",
        data=np_arrays["img"],
        chunks=(chunk_len, *args.img_size, 3), dtype="float32")
    dg.create_dataset("state",
        data=np_arrays["state"],
        chunks=(chunk_len, np_arrays["state"].shape[1]), dtype="float32")
    dg.create_dataset("hybrid_action_discrete",
        data=np_arrays["hybrid_action_discrete"],
        chunks=(chunk_len,), dtype="int64")
    dg.create_dataset("hybrid_action_params",
        data=np_arrays["hybrid_action_params"],
        chunks=(chunk_len, np_arrays["hybrid_action_params"].shape[1]), dtype="float32")
    dg.create_dataset("hybrid_action_flat",
        data=np_arrays["hybrid_action_flat"],
        chunks=(chunk_len, np_arrays["hybrid_action_flat"].shape[1]), dtype="float32")
    mg = root.create_group("meta")
    mg.create_dataset("episode_ends",  data=episode_ends_np, dtype="int64")
    mg.create_dataset("episode_seeds",
        data=np.asarray(episode_seeds[:len(episode_ends)], dtype=np.int64))
    mg.create_dataset("episode_rewards",
        data=np.asarray(episode_rewards, dtype=np.float32))

    rewards_np = np.asarray(episode_rewards, dtype=np.float32)
    print(f"\nSaved zarr → {output_path}")
    for k, v in np_arrays.items():
        print(f"  data/{k:30s}: {v.shape}")
    print(f"  meta/episode_ends               : {episode_ends_np.shape}")
    print(f"  chunk_len={chunk_len}  episodes={len(episode_ends_np)}  "
          f"steps={total_steps}  reward={rewards_np.mean():.1f}±{rewards_np.std():.1f}")

    # ── Return viz data built during the same rollout ─────────────────────────
    episode_lengths = np.diff(np.concatenate([[0], episode_ends_np]))
    return {
        "episode_rewards": rewards_np,
        "episode_lengths": episode_lengths.astype(float),
        "trajectories":    trajectories,
        "all_discrete":    np.array(all_discrete),
        "all_params_k1":   np.array(all_params_k1) if all_params_k1 else np.array([]),
        "all_params_k2":   np.array(all_params_k2) if all_params_k2 else np.zeros((0, 2)),
        "all_params_k3":   np.array(all_params_k3) if all_params_k3 else np.zeros((0, 2)),
        "ep_action_seqs":  ep_action_seqs,
        "episode_seeds":   episode_seeds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data collection for visualization
# ══════════════════════════════════════════════════════════════════════════════

def collect_from_checkpoint(args) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub_steps = args.sub_steps if args.sub_steps is not None else 1
    env, policy, action_layout, _ = _load_env_and_policy(
        model_path=args.model_path,
        use_image_obs=False,
        img_size=tuple(args.img_size),
        sub_steps=sub_steps,
        device=device,
    )

    rng = np.random.default_rng(args.collect_seed)
    total_episodes = max(args.n_episodes, args.n_traj)

    episode_rewards, episode_lengths = [], []
    trajectories = []
    all_discrete = []
    all_params_k1, all_params_k2, all_params_k3 = [], [], []
    ep_action_seqs = []

    print(f"Rolling out {total_episodes} episodes for visualization...")
    for ep in range(total_episodes):
        seed = abs(int(rng.normal(args.seed_mean, args.seed_std)))
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        ep_traj_x, ep_traj_y, ep_actions = [], [], []

        while not done:
            if isinstance(obs, dict):
                state = obs["state"].astype(np.float32)
            else:
                state = obs.astype(np.float32)

            obs_t = torch.as_tensor(state, dtype=torch.float32,
                                    device=device).unsqueeze(0)
            with torch.no_grad():
                flat_action_t, discrete_action_t = policy.act(
                    obs_t, deterministic=args.deterministic)

            discrete_action = int(discrete_action_t.item())
            flat_action = flat_action_t.cpu().numpy()[0]
            hybrid_action = action_layout.flat_to_env_action(discrete_action, flat_action)

            ep_traj_x.append(float(state[0]))
            ep_traj_y.append(float(state[1]))
            ep_actions.append(discrete_action)
            all_discrete.append(discrete_action)

            xk = hybrid_action["x_k"]
            if discrete_action == 1:
                all_params_k1.append(float(xk[0]))
            elif discrete_action == 2:
                all_params_k2.append([float(xk[0]), float(xk[1])])
            elif discrete_action == 3:
                all_params_k3.append([float(xk[0]), float(xk[1])])

            obs, reward, terminated, truncated, _ = env.step(hybrid_action)
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(len(ep_actions))
        ep_action_seqs.append(ep_actions)
        if ep < args.n_traj:
            trajectories.append((np.array(ep_traj_x), np.array(ep_traj_y)))

        if (ep + 1) % 20 == 0:
            print(f"  {ep+1}/{total_episodes}  "
                  f"mean_reward={np.mean(episode_rewards):.1f}")

    env.close()
    return {
        "episode_rewards": np.array(episode_rewards),
        "episode_lengths": np.array(episode_lengths),
        "trajectories":    trajectories,
        "all_discrete":    np.array(all_discrete),
        "all_params_k1":   np.array(all_params_k1) if all_params_k1 else np.array([]),
        "all_params_k2":   np.array(all_params_k2) if all_params_k2 else np.zeros((0, 2)),
        "all_params_k3":   np.array(all_params_k3) if all_params_k3 else np.zeros((0, 2)),
        "ep_action_seqs":  ep_action_seqs,
    }


def collect_from_zarr(zarr_path: str, n_traj: int) -> dict:
    import zarr
    root = zarr.open(zarr_path, mode="r")
    states  = root["data/state"][:]
    discrete = root["data/hybrid_action_discrete"][:]
    params   = root["data/hybrid_action_params"][:]
    episode_ends = root["meta/episode_ends"][:]

    episode_starts  = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts

    episode_rewards, trajectories, ep_action_seqs = [], [], []
    all_params_k1, all_params_k2, all_params_k3 = [], [], []

    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep_states   = states[start:end]
        ep_discrete = discrete[start:end]
        ep_params   = params[start:end]

        ep_action_seqs.append(ep_discrete.tolist())
        if i < n_traj:
            trajectories.append((ep_states[:, 0], ep_states[:, 1]))

        for d, p in zip(ep_discrete, ep_params):
            if d == 1:
                all_params_k1.append(float(p[0]))
            elif d == 2:
                all_params_k2.append([float(p[0]), float(p[1])])
            elif d == 3:
                all_params_k3.append([float(p[0]), float(p[1])])

        episode_rewards.append(float(ep_states[-1, 1]))

    print("NOTE: zarr mode — reward is a proxy (final y). "
          "Use --model-path for accurate histogram.")
    return {
        "episode_rewards": np.array(episode_rewards),
        "episode_lengths": episode_lengths.astype(float),
        "trajectories":    trajectories,
        "all_discrete":    discrete,
        "all_params_k1":   np.array(all_params_k1) if all_params_k1 else np.array([]),
        "all_params_k2":   np.array(all_params_k2) if all_params_k2 else np.zeros((0, 2)),
        "all_params_k3":   np.array(all_params_k3) if all_params_k3 else np.zeros((0, 2)),
        "ep_action_seqs":  ep_action_seqs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Return histogram
# ══════════════════════════════════════════════════════════════════════════════

def plot_return_histogram(data: dict, output_dir: Path):
    rewards = data["episode_rewards"]
    n = len(rewards)
    mean_r, std_r = rewards.mean(), rewards.std()
    pct_pos  = 100.0 * (rewards > 0).sum() / n
    pct_high = 100.0 * (rewards > 200).sum() / n

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    n_bins = min(40, max(10, n // 5))
    counts, bins, patches = ax.hist(rewards, bins=n_bins,
                                    color=PALETTE["accent"], alpha=0.75,
                                    edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor(PALETTE["rose"])
            patch.set_alpha(0.7)
        elif left > 200:
            patch.set_facecolor(PALETTE["green"])
            patch.set_alpha(0.85)

    ax.axvline(mean_r, color=PALETTE["teal"],  lw=1.8, ls="--",
               label=f"mean = {mean_r:.1f}")
    ax.axvline(0,       color=PALETTE["rose"],  lw=1.2, ls=":", alpha=0.8,
               label="reward = 0")
    ax.axvline(200,     color=PALETTE["green"], lw=1.2, ls=":", alpha=0.8,
               label="reward = 200")

    ymax = counts.max() * 1.15
    ax.axvspan(rewards.min() - 5, 0,             alpha=0.06, color=PALETTE["rose"],  zorder=0)
    ax.axvspan(200,  rewards.max() + 5,           alpha=0.06, color=PALETTE["green"], zorder=0)

    stats_txt = (f"n = {n}\n"
                 f"mean ± std = {mean_r:.1f} ± {std_r:.1f}\n"
                 f"reward > 0   : {pct_pos:.1f}%\n"
                 f"reward > 200 : {pct_high:.1f}%")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=PALETTE["text"],
            bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                      boxstyle="round,pad=0.5", alpha=0.9))

    ax.legend(handles=[
        Patch(color=PALETTE["rose"],   label="negative"),
        Patch(color=PALETTE["accent"], label="moderate"),
        Patch(color=PALETTE["green"],  label="> 200 (expert)"),
    ] + ax.get_lines(), loc="upper left", fontsize=8)

    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.set_title("Fig 1 — Expert Return Distribution")
    ax.grid(True, axis="y")
    ax.set_ylim(0, ymax)
    fig.tight_layout()

    out = output_dir / "fig1_return_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Multi-trajectory overlay
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_overlay(data: dict, output_dir: Path):
    trajectories = data["trajectories"]
    n_traj = len(trajectories)
    cmap = plt.cm.plasma

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    for i, (xs, ys) in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(xs, ys, color=color, alpha=0.6, lw=0.9, zorder=2)
        ax.scatter(xs[0],  ys[0],  color=color, s=18, zorder=3, alpha=0.8)
        ax.scatter(xs[-1], ys[-1], color=color, marker="*", s=60, zorder=4)

    ax.axhline(0, color=PALETTE["border"], lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color=PALETTE["border"], lw=1.0, ls="--", alpha=0.7)
    ax.add_patch(plt.Rectangle((-0.2, -0.05), 0.4, 0.05,
                                color=PALETTE["green"], alpha=0.2, zorder=1))
    ax.text(0, -0.08, "landing pad", ha="center", va="top",
            color=PALETTE["green"], fontsize=8)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"2D Trajectory Overlay (n={n_traj})")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.15, 1.6)
    ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    for i, (xs, ys) in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        t = np.arange(len(xs))
        ax2.plot(t, xs, color=color, alpha=0.45, lw=0.8, ls="-")
        ax2.plot(t, ys, color=color, alpha=0.45, lw=0.8, ls="--")
    ax2.plot([], [], color=PALETTE["muted"], ls="-",  label="x")
    ax2.plot([], [], color=PALETTE["muted"], ls="--", label="y")
    ax2.axhline(0, color=PALETTE["green"], lw=0.8, ls=":", alpha=0.7)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Position")
    ax2.set_title("Position vs Time  (solid=x, dashed=y)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    fig.suptitle("Fig 2 — Multi-Trajectory Overlay  ★=landing point",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = output_dir / "fig2_trajectory_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Discrete action frequency heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_action_heatmap(data: dict, output_dir: Path, max_steps: int = 300):
    ep_action_seqs = data["ep_action_seqs"]
    n_actions = 4

    freq   = np.zeros((n_actions, max_steps), dtype=np.float64)
    counts = np.zeros(max_steps, dtype=np.float64)
    for seq in ep_action_seqs:
        for t, a in enumerate(seq):
            if t < max_steps:
                freq[a, t] += 1.0
                counts[t]  += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        freq_norm = np.where(counts > 0, freq / counts[np.newaxis, :], 0.0)

    last_t = np.where(counts > 0)[0]
    if len(last_t):
        freq_norm    = freq_norm[:, :last_t[-1] + 1]
        counts_trim  = counts[:last_t[-1] + 1]
    else:
        counts_trim = counts

    fig, (ax, ax_c) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08})
    fig.patch.set_facecolor(PALETTE["bg"])

    cmaps = [
        LinearSegmentedColormap.from_list("c0", [PALETTE["bg2"], PALETTE["muted"]]),
        LinearSegmentedColormap.from_list("c1", [PALETTE["bg2"], PALETTE["teal"]]),
        LinearSegmentedColormap.from_list("c2", [PALETTE["bg2"], PALETTE["amber"]]),
        LinearSegmentedColormap.from_list("c3", [PALETTE["bg2"], PALETTE["coral"]]),
    ]
    T = freq_norm.shape[1]
    for k in range(n_actions):
        row_y = n_actions - 1 - k
        for t in range(T):
            v = freq_norm[k, t]
            if v > 0.001:
                ax.barh(row_y, 1.0, left=t, height=0.9,
                        color=cmaps[k](v), linewidth=0)

    ax.set_yticks(range(n_actions))
    ax.set_yticklabels([ACTION_NAMES[n_actions - 1 - k] for k in range(n_actions)],
                       fontsize=10)
    for lbl, k_rev in zip(ax.get_yticklabels(), range(n_actions)):
        lbl.set_color(ACTION_COLORS[n_actions - 1 - k_rev])

    ax.set_xlim(0, T + 60)
    ax.set_ylim(-0.5, n_actions - 0.5)
    ax.set_xticks([])
    ax.set_title("Fig 3 — Discrete Action Frequency over Episode Steps", pad=8)
    for k in range(n_actions):
        ax.text(T + 2, n_actions - 1 - k, f"■ {ACTION_NAMES[k]}",
                color=ACTION_COLORS[k], va="center", fontsize=8)
    ax.grid(False)

    x = np.arange(len(counts_trim))
    ax_c.fill_between(x, counts_trim, color=PALETTE["accent"], alpha=0.35)
    ax_c.plot(x, counts_trim, color=PALETTE["accent"], lw=0.8)
    ax_c.set_ylabel("# eps", fontsize=8, color=PALETTE["muted"])
    ax_c.set_xlabel("Timestep")
    ax_c.set_xlim(0, T + 60)
    ax_c.yaxis.set_label_position("right")
    ax_c.yaxis.tick_right()
    ax_c.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    out = output_dir / "fig3_action_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Per-branch parameter scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_param_scatter(data: dict, output_dir: Path):
    k1 = data["all_params_k1"]
    k2 = data["all_params_k2"]
    k3 = data["all_params_k3"]
    all_d  = data["all_discrete"]
    n_tot  = max(len(all_d), 1)
    f0 = (all_d == 0).sum() / n_tot * 100
    f1 = (all_d == 1).sum() / n_tot * 100
    f2 = (all_d == 2).sum() / n_tot * 100
    f3 = (all_d == 3).sum() / n_tot * 100

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)

    # k=0
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(["coast\n(k=0)"], [f0], color=ACTION_COLORS[0], alpha=0.75, width=0.5)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Usage %", color=PALETTE["muted"])
    ax0.set_title("k=0 coast\n(no params)", fontsize=9, color=ACTION_COLORS[0])
    ax0.text(0, f0 + 2, f"{f0:.1f}%", ha="center", va="bottom",
             fontsize=11, color=ACTION_COLORS[0])
    ax0.grid(True, axis="y", alpha=0.4)

    # k=1 histogram
    ax1 = fig.add_subplot(gs[0, 1:3])
    if len(k1) > 0:
        ax1.hist(k1, bins=min(50, max(10, len(k1)//10)),
                 color=ACTION_COLORS[1], alpha=0.75,
                 edgecolor="white", lw=0.4)
        ax1.axvline(k1.mean(), color=PALETTE["text"], lw=1.4, ls="--",
                    label=f"mean={k1.mean():.2f}")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.4)
        _cluster_note(ax1, k1)
    else:
        ax1.text(0.5, 0.5, "no k=1 actions", ha="center", va="center",
                 transform=ax1.transAxes, color=PALETTE["muted"])
    ax1.set_title(f"k=1 main engine — throttle  ({f1:.1f}%)",
                  fontsize=9, color=ACTION_COLORS[1])
    ax1.set_xlabel("throttle  x_k[0] ∈ [-1, 1]")
    ax1.set_ylabel("Count")
    ax1.set_xlim(-1.1, 1.1)

    # k=1 boxplot
    ax1b = fig.add_subplot(gs[0, 3])
    if len(k1) > 0:
        ax1b.boxplot(k1, vert=True, patch_artist=True,
                     medianprops=dict(color=PALETTE["text"], lw=2),
                     boxprops=dict(facecolor=ACTION_COLORS[1], alpha=0.5),
                     whiskerprops=dict(color=PALETTE["muted"]),
                     capprops=dict(color=PALETTE["muted"]),
                     flierprops=dict(marker=".", ms=2,
                                     markerfacecolor=PALETTE["muted"]))
        ax1b.set_xticks([1])
        ax1b.set_xticklabels(["throttle"])
        ax1b.set_ylim(-1.1, 1.1)
        ax1b.grid(True, axis="y", alpha=0.4)
    ax1b.set_title("spread", fontsize=9, color=ACTION_COLORS[1])

    # k=2 scatter
    ax2 = fig.add_subplot(gs[1, 0:2])
    if len(k2) > 1:
        _scatter2d(ax2, k2[:, 0], k2[:, 1], ACTION_COLORS[2],
                   "intensity  x_k[0]", "duration_frac  x_k[1]")
        _cluster_note_2d(ax2, k2)
    else:
        ax2.text(0.5, 0.5, "no k=2 actions", ha="center", va="center",
                 transform=ax2.transAxes, color=PALETTE["muted"])
    ax2.set_title(f"k=2 left boost — intensity × duration  ({f2:.1f}%)",
                  fontsize=9, color=ACTION_COLORS[2])

    # k=3 scatter
    ax3 = fig.add_subplot(gs[1, 2:4])
    if len(k3) > 1:
        _scatter2d(ax3, k3[:, 0], k3[:, 1], ACTION_COLORS[3],
                   "intensity  x_k[0]", "duration_frac  x_k[1]")
        _cluster_note_2d(ax3, k3)
    else:
        ax3.text(0.5, 0.5, "no k=3 actions", ha="center", va="center",
                 transform=ax3.transAxes, color=PALETTE["muted"])
    ax3.set_title(f"k=3 right boost — intensity × duration  ({f3:.1f}%)",
                  fontsize=9, color=ACTION_COLORS[3])

    fig.text(0.5, 0.94, "Fig 4 — Per-Branch Continuous Parameter Distributions",
             ha="center", fontsize=12, fontweight="bold", color=PALETTE["text"])
    fig.text(0.5, 0.905,
             f"coast {f0:.1f}%   main {f1:.1f}%   left {f2:.1f}%   right {f3:.1f}%",
             ha="center", fontsize=9, color=PALETTE["muted"])

    out = output_dir / "fig4_param_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


def _scatter2d(ax, xs, ys, color, xlabel, ylabel, max_pts=4000):
    if len(xs) > max_pts:
        idx = np.random.choice(len(xs), max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    ax.scatter(xs, ys, c=color, alpha=0.3, s=7, linewidths=0)
    try:
        from scipy.stats import gaussian_kde
        for arr, span, fixed, horiz in [
            (xs, np.linspace(-1.05, 1.05, 200), -1.1, False),
            (ys, np.linspace(-1.05, 1.05, 200), -1.1, True),
        ]:
            if arr.std() > 1e-6:
                kde = gaussian_kde(arr)
                d = kde(span); d = d / d.max() * 0.18
                if not horiz:
                    ax.plot(span, np.full_like(span, fixed) + d, color=color, lw=1.0, alpha=0.7)
                else:
                    ax.plot(np.full_like(span, fixed) + d, span, color=color, lw=1.0, alpha=0.7)
    except Exception:
        pass
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.axhline(0, color=PALETTE["border"], lw=0.5, alpha=0.6)
    ax.axvline(0, color=PALETTE["border"], lw=0.5, alpha=0.6)
    ax.grid(True, alpha=0.4)


def _cluster_note(ax, arr):
    std = arr.std()
    ok = std < 0.35
    ax.text(0.02, 0.97,
            f"std={std:.3f}  {'← clustered ✓' if ok else '← spread (check)'}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=PALETTE["green"] if ok else PALETTE["amber"])


def _cluster_note_2d(ax, arr):
    sx, sy = arr[:, 0].std(), arr[:, 1].std()
    ok = sx < 0.4 and sy < 0.4
    ax.text(0.02, 0.97,
            f"std: ({sx:.2f}, {sy:.2f})  {'← clustered ✓' if ok else '← spread (check)'}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=PALETTE["green"] if ok else PALETTE["amber"])


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(data: dict):
    rewards  = data["episode_rewards"]
    lengths  = data["episode_lengths"]
    discrete = data["all_discrete"]
    n_tot    = max(len(discrete), 1)

    print("\n" + "=" * 60)
    print("  EXPERT QUALITY SUMMARY")
    print("=" * 60)
    print(f"  episodes         : {len(rewards)}")
    print(f"  return           : {rewards.mean():.1f} ± {rewards.std():.1f}")
    print(f"  return range     : [{rewards.min():.1f}, {rewards.max():.1f}]")
    print(f"  pct reward > 0   : {100*(rewards>0).mean():.1f}%")
    print(f"  pct reward > 200 : {100*(rewards>200).mean():.1f}%")
    print(f"  episode length   : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print()
    print("  Discrete action usage:")
    for k in range(4):
        pct = 100 * (discrete == k).sum() / n_tot
        print(f"    k={k} {ACTION_NAMES[k]:15s} : {pct:5.1f}%  {'█'*int(pct/2)}")
    print()
    k1 = data["all_params_k1"]
    if len(k1):
        print(f"  k=1 throttle : mean={k1.mean():.3f}  std={k1.std():.3f}")
    for lbl, arr in [("k=2", data["all_params_k2"]), ("k=3", data["all_params_k3"])]:
        if len(arr):
            print(f"  {lbl} intensity : mean={arr[:,0].mean():.3f}  std={arr[:,0].std():.3f}")
            print(f"  {lbl} dur_frac  : mean={arr[:,1].mean():.3f}  std={arr[:,1].std():.3f}")
    print()
    ok = rewards.mean() > 150 and (rewards > 0).mean() > 0.85 and lengths.std() < 100
    print("  ✓  Strong expert — safe for IL." if ok
          else "  ⚠  Quality uncertain — check figures.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Rollout Hybrid SAC to zarr + expert quality figures.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", type=str,
                     help="SAC checkpoint (.pt) — runs rollout + figures.")
    src.add_argument("--zarr-path",  type=str,
                     help="Pre-saved zarr — figures only, no rollout.")

    # zarr output (only used with --model-path)
    parser.add_argument("--output-path", type=str,
                        default="./lunar_lander_replay.zarr",
                        help="Where to save the zarr dataset.")

    # figures
    parser.add_argument("--output-dir",    type=str,   default="./expert_figures")
    parser.add_argument("--n-episodes",    type=int,   default=300)
    parser.add_argument("--n-traj",        type=int,   default=15)
    parser.add_argument("--max-steps",     type=int,   default=300)
    parser.add_argument("--deterministic", action="store_true",  default=True)
    parser.add_argument("--stochastic",    action="store_false", dest="deterministic")
    parser.add_argument("--seed-mean",     type=float, default=0.0)
    parser.add_argument("--seed-std",      type=float, default=10000.0)
    parser.add_argument("--collect-seed",  type=int,   default=42)
    parser.add_argument("--img-size",      type=int,   nargs=2, default=[96, 96])
    parser.add_argument("--sub-steps",     type=int,   default=None)
    parser.add_argument("--min-reward",    type=float, default=None,
                        help="Filter zarr: keep episodes with reward >= this.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    if args.model_path:
        # Single rollout → zarr + viz data simultaneously (same seeds)
        data = rollout_to_zarr(args)
    else:
        print(f"Loading zarr: {args.zarr_path}")
        data = collect_from_zarr(args.zarr_path, n_traj=args.n_traj)

    print_summary(data)

    print("\nGenerating figures...")
    plot_return_histogram(data, output_dir)
    plot_trajectory_overlay(data, output_dir)
    plot_action_heatmap(data, output_dir, max_steps=args.max_steps)
    plot_param_scatter(data, output_dir)

    print(f"\nAll figures → {output_dir}/")


if __name__ == "__main__":
    main()