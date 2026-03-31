"""
Expert Quality Visualization for Hybrid SAC LunarLander
========================================================

Generates 4 key figures to validate expert quality before IL training:
    1. Return distribution histogram (multi-episode)
    2. Multi-trajectory overlay (x/y positions over time)
    3. Discrete action frequency heatmap over episode steps
    4. Per-branch continuous parameter scatter plots

Usage (standalone):
    python scripts/visualize_expert.py \
        --model-path /path/to/best_model.pt \
        --n-episodes 100 \
        --n-traj 15 \
        --output-dir /path/to/figures

Usage (after rollout, from saved zarr):
    python scripts/visualize_expert.py \
        --zarr-path /path/to/lunar_lander_replay.zarr \
        --output-dir /path/to/figures
"""

from __future__ import annotations

# ── MuJoCo / gymnasium_robotics patch ────────────────────────────────────────
# Must run before ANY `import gymnasium` to prevent mujoco_py crash.
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
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Color palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0f1117",
    "bg2":      "#1a1d27",
    "border":   "#2a2d3a",
    "text":     "#e8e8f0",
    "muted":    "#7a7a9a",
    "accent":   "#7c6af7",    # purple
    "teal":     "#2dd4bf",
    "amber":    "#fbbf24",
    "coral":    "#f97316",
    "rose":     "#fb7185",
    "green":    "#4ade80",
}

# One color per discrete action
ACTION_COLORS = {
    0: PALETTE["muted"],     # coast     — gray
    1: PALETTE["teal"],      # main_engine — teal
    2: PALETTE["amber"],     # left_boost  — amber
    3: PALETTE["coral"],     # right_boost — coral
}
ACTION_NAMES = {0: "coast", 1: "main engine", 2: "left boost", 3: "right boost"}


def set_style():
    plt.rcParams.update({
        "figure.facecolor":    PALETTE["bg"],
        "axes.facecolor":      PALETTE["bg2"],
        "axes.edgecolor":      PALETTE["border"],
        "axes.labelcolor":     PALETTE["text"],
        "axes.titlecolor":     PALETTE["text"],
        "xtick.color":         PALETTE["muted"],
        "ytick.color":         PALETTE["muted"],
        "text.color":          PALETTE["text"],
        "grid.color":          PALETTE["border"],
        "grid.linewidth":      0.5,
        "grid.alpha":          0.6,
        "font.family":         "monospace",
        "font.size":           10,
        "axes.titlesize":      12,
        "axes.titleweight":    "bold",
        "axes.labelsize":      10,
        "legend.facecolor":    PALETTE["bg2"],
        "legend.edgecolor":    PALETTE["border"],
        "legend.labelcolor":   PALETTE["text"],
        "figure.dpi":          150,
    })


# ── Data collection ────────────────────────────────────────────────────────────

def collect_from_checkpoint(
    model_path: str,
    n_episodes: int,
    n_traj: int,
    deterministic: bool,
    seed_mean: float,
    seed_std: float,
    collect_seed: int,
    img_size: tuple[int, int],
    sub_steps: int,
) -> dict:
    """Roll out the policy and collect all data needed for visualization."""
    import torch
    import importlib.util as _ilu
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ll_path = os.path.join(_project_root, 'environments', 'lunar_lander.py')
    _spec = _ilu.spec_from_file_location('environments.lunar_lander', _ll_path)
    _ll_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ll_mod)
    HybridLunarLander = _ll_mod.HybridLunarLander
    from scripts.train_hybrid_sac_lunar_lander import HybridActionLayout, Policy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    network_cfg = cfg.get("network", {})
    sac_cfg = cfg.get("sac", {})

    env = HybridLunarLander(use_image_obs=False, img_size=img_size, sub_steps=sub_steps)
    action_layout = HybridActionLayout.from_env_spec(env.get_action_spec())

    obs_dim = int(env.observation_space.shape[0])
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

    rng = np.random.default_rng(collect_seed)
    total_episodes = max(n_episodes, n_traj)

    episode_rewards = []
    episode_lengths = []
    # Trajectory data for first n_traj episodes
    trajectories = []   # list of (T, 2) arrays: x, y per step
    # Action data accumulated across all episodes
    all_discrete = []
    all_params_k1 = []   # throttle
    all_params_k2 = []   # [intensity, dur]
    all_params_k3 = []   # [intensity, dur]
    # For heatmap: per-episode step action sequences (padded)
    ep_action_seqs = []

    print(f"Rolling out {total_episodes} episodes...")
    for ep in range(total_episodes):
        seed = abs(int(rng.normal(seed_mean, seed_std)))
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        ep_traj_x = []
        ep_traj_y = []
        ep_actions = []

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                flat_action_t, discrete_action_t = policy.act(
                    obs_t, deterministic=deterministic
                )
            discrete_action = int(discrete_action_t.item())
            flat_action = flat_action_t.cpu().numpy()[0]
            hybrid_action = action_layout.flat_to_env_action(discrete_action, flat_action)

            ep_traj_x.append(float(obs[0]))
            ep_traj_y.append(float(obs[1]))
            ep_actions.append(discrete_action)

            # Collect continuous params per branch
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

        if ep < n_traj:
            trajectories.append((np.array(ep_traj_x), np.array(ep_traj_y)))

        if (ep + 1) % 20 == 0:
            print(f"  {ep+1}/{total_episodes}  mean_reward={np.mean(episode_rewards):.1f}")

    env.close()

    return {
        "episode_rewards":  np.array(episode_rewards),
        "episode_lengths":  np.array(episode_lengths),
        "trajectories":     trajectories,
        "all_discrete":     np.array(all_discrete),
        "all_params_k1":    np.array(all_params_k1) if all_params_k1 else np.array([]),
        "all_params_k2":    np.array(all_params_k2) if all_params_k2 else np.zeros((0, 2)),
        "all_params_k3":    np.array(all_params_k3) if all_params_k3 else np.zeros((0, 2)),
        "ep_action_seqs":   ep_action_seqs,
    }


def collect_from_zarr(zarr_path: str, n_traj: int) -> dict:
    """Load pre-saved zarr rollout data."""
    import zarr

    root = zarr.open(zarr_path, mode="r")
    states = root["data/state"][:]
    discrete = root["data/hybrid_action_discrete"][:]
    params = root["data/hybrid_action_params"][:]
    episode_ends = root["meta/episode_ends"][:]

    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts

    # Reconstruct per-episode rewards from states isn't possible without reward,
    # so we compute a proxy: use episode length as a heuristic flag.
    # If rewards were saved as attrs or separate array, use those.
    # Here we just reconstruct what we can.

    episode_rewards = []
    trajectories = []
    ep_action_seqs = []
    all_params_k1, all_params_k2, all_params_k3 = [], [], []

    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep_states = states[start:end]
        ep_discrete = discrete[start:end]
        ep_params = params[start:end]

        # x = state[0], y = state[1]
        ep_traj_x = ep_states[:, 0]
        ep_traj_y = ep_states[:, 1]

        ep_action_seqs.append(ep_discrete.tolist())

        if i < n_traj:
            trajectories.append((ep_traj_x, ep_traj_y))

        for step_d, step_p in zip(ep_discrete, ep_params):
            if step_d == 1:
                all_params_k1.append(float(step_p[0]))
            elif step_d == 2:
                all_params_k2.append([float(step_p[0]), float(step_p[1])])
            elif step_d == 3:
                all_params_k3.append([float(step_p[0]), float(step_p[1])])

        # Simple proxy reward: estimate from final y-position and length
        # Use episode length as quality signal since we don't have rewards saved
        episode_rewards.append(float(ep_states[-1, 1]))  # placeholder

    # Warn user if rewards are proxies
    print("NOTE: reward loaded from zarr is a proxy (final y-pos). "
          "For accurate histogram, run with --model-path.")

    return {
        "episode_rewards":  np.array(episode_rewards),
        "episode_lengths":  episode_lengths.astype(float),
        "trajectories":     trajectories,
        "all_discrete":     discrete,
        "all_params_k1":    np.array(all_params_k1) if all_params_k1 else np.array([]),
        "all_params_k2":    np.array(all_params_k2) if all_params_k2 else np.zeros((0, 2)),
        "all_params_k3":    np.array(all_params_k3) if all_params_k3 else np.zeros((0, 2)),
        "ep_action_seqs":   ep_action_seqs,
    }


# ── Figure 1: Return Histogram ─────────────────────────────────────────────────

def plot_return_histogram(data: dict, output_dir: Path):
    rewards = data["episode_rewards"]
    n = len(rewards)
    mean_r = rewards.mean()
    std_r = rewards.std()
    pct_positive = 100.0 * (rewards > 0).sum() / n
    pct_high = 100.0 * (rewards > 200).sum() / n

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Histogram
    n_bins = min(40, max(10, n // 5))
    counts, bins, patches = ax.hist(
        rewards, bins=n_bins,
        color=PALETTE["accent"], alpha=0.85,
        edgecolor=PALETTE["bg"], linewidth=0.5,
    )

    # Color bars by quality zone
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor(PALETTE["rose"])
            patch.set_alpha(0.75)
        elif left > 200:
            patch.set_facecolor(PALETTE["green"])
            patch.set_alpha(0.9)

    # Reference lines
    ax.axvline(mean_r, color=PALETTE["teal"], linewidth=1.5, linestyle="--",
               label=f"mean = {mean_r:.1f}")
    ax.axvline(0, color=PALETTE["rose"], linewidth=1.0, linestyle=":", alpha=0.7,
               label="reward = 0")
    ax.axvline(200, color=PALETTE["green"], linewidth=1.0, linestyle=":", alpha=0.7,
               label="reward = 200")

    # Shaded zones
    ymax = counts.max() * 1.1
    ax.axvspan(rewards.min() - 5, 0, alpha=0.06, color=PALETTE["rose"], zorder=0)
    ax.axvspan(200, rewards.max() + 5, alpha=0.06, color=PALETTE["green"], zorder=0)

    # Stats annotation
    stats_txt = (
        f"n = {n}\n"
        f"mean ± std = {mean_r:.1f} ± {std_r:.1f}\n"
        f"reward > 0   : {pct_positive:.1f}%\n"
        f"reward > 200 : {pct_high:.1f}%"
    )
    ax.text(
        0.97, 0.97, stats_txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=PALETTE["text"],
        bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                  boxstyle="round,pad=0.5", alpha=0.9),
    )

    legend_patches = [
        Patch(color=PALETTE["rose"],   label="negative reward"),
        Patch(color=PALETTE["accent"], label="moderate"),
        Patch(color=PALETTE["green"],  label="> 200 (expert zone)"),
    ]
    ax.legend(
        handles=legend_patches + ax.get_lines(),
        loc="upper left", fontsize=8,
        framealpha=0.85,
    )

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


# ── Figure 2: Multi-trajectory Overlay ────────────────────────────────────────

def plot_trajectory_overlay(data: dict, output_dir: Path):
    trajectories = data["trajectories"]
    n_traj = len(trajectories)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Left panel: x, y trajectory paths in 2D space
    ax = axes[0]
    cmap = plt.cm.plasma
    for i, (xs, ys) in enumerate(trajectories):
        alpha = 0.55 + 0.45 * (i / max(n_traj - 1, 1))
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=0.9, zorder=2)
        # Start marker
        ax.scatter(xs[0], ys[0], color=color, s=18, zorder=3, alpha=0.8)
        # End marker
        ax.scatter(xs[-1], ys[-1], color=color, marker="*", s=55, zorder=4, alpha=0.95)

    # Landing pad reference
    ax.axhline(0, color=PALETTE["border"], linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(0, color=PALETTE["border"], linewidth=1.0, linestyle="--", alpha=0.6)
    ax.add_patch(plt.Rectangle((-0.2, -0.05), 0.4, 0.05,
                                color=PALETTE["green"], alpha=0.15, zorder=1))
    ax.text(0, -0.08, "landing pad", ha="center", va="top",
            color=PALETTE["green"], fontsize=8)

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"2D Trajectory Overlay (n={n_traj})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.15, 1.6)

    # Right panel: x and y over timestep
    ax2 = axes[1]
    for i, (xs, ys) in enumerate(trajectories):
        T = len(xs)
        t = np.arange(T)
        color = cmap(i / max(n_traj - 1, 1))
        ax2.plot(t, xs, color=color, alpha=0.45, linewidth=0.8, linestyle="-")
        ax2.plot(t, ys, color=color, alpha=0.45, linewidth=0.8, linestyle="--")

    # Legend proxies
    ax2.plot([], [], color=PALETTE["muted"], linestyle="-",  label="x position")
    ax2.plot([], [], color=PALETTE["muted"], linestyle="--", label="y position")
    ax2.axhline(0, color=PALETTE["green"], linewidth=0.8, linestyle=":", alpha=0.7)

    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Position")
    ax2.set_title("Position vs Time (solid=x, dashed=y)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Fig 2 — Multi-Trajectory Overlay  ★=landing point",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = output_dir / "fig2_trajectory_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ── Figure 3: Discrete Action Frequency Heatmap ───────────────────────────────

def plot_action_heatmap(data: dict, output_dir: Path, max_steps: int = 300):
    ep_action_seqs = data["ep_action_seqs"]
    n_actions = 4

    # Build frequency matrix: (n_actions, max_steps)
    freq = np.zeros((n_actions, max_steps), dtype=np.float64)
    counts = np.zeros(max_steps, dtype=np.float64)

    for seq in ep_action_seqs:
        for t, a in enumerate(seq):
            if t < max_steps:
                freq[a, t] += 1.0
                counts[t] += 1.0

    # Normalize per timestep
    with np.errstate(invalid="ignore", divide="ignore"):
        freq_norm = np.where(counts > 0, freq / counts[np.newaxis, :], 0.0)

    # Trim to last timestep with data
    last_t = np.where(counts > 0)[0]
    if len(last_t) > 0:
        freq_norm = freq_norm[:, : last_t[-1] + 1]
        counts_trim = counts[: last_t[-1] + 1]
    else:
        counts_trim = counts

    fig, (ax, ax_count) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor(PALETTE["bg"])

    # Custom colormap per action row
    cmaps = [
        LinearSegmentedColormap.from_list("coast",   ["#1a1d27", PALETTE["muted"]]),
        LinearSegmentedColormap.from_list("main",    ["#1a1d27", PALETTE["teal"]]),
        LinearSegmentedColormap.from_list("left",    ["#1a1d27", PALETTE["amber"]]),
        LinearSegmentedColormap.from_list("right",   ["#1a1d27", PALETTE["coral"]]),
    ]

    T = freq_norm.shape[1]
    x = np.arange(T)
    bar_h = 0.9

    for k in range(n_actions):
        row_y = n_actions - 1 - k  # flip so main_engine is near top
        vals = freq_norm[k]
        # Draw as colored bar segments
        for t in range(T):
            v = vals[t]
            if v > 0.001:
                color = cmaps[k](v)
                ax.barh(row_y, 1.0, left=t, height=bar_h,
                        color=color, linewidth=0)

    ax.set_yticks(range(n_actions))
    ax.set_yticklabels(
        [ACTION_NAMES[n_actions - 1 - k] for k in range(n_actions)],
        fontsize=10,
    )
    # Color the y tick labels
    for label, k_rev in zip(ax.get_yticklabels(), range(n_actions)):
        k = n_actions - 1 - k_rev
        label.set_color(ACTION_COLORS[k])

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, n_actions - 0.5)
    ax.set_xticks([])
    ax.set_title("Fig 3 — Discrete Action Frequency over Episode Steps", pad=8)

    # Colorbar-like legend
    for k in range(n_actions):
        ax.text(T + 2, n_actions - 1 - k, f"■ {ACTION_NAMES[k]}",
                color=ACTION_COLORS[k], va="center", fontsize=8)

    ax.set_xlim(0, T + 60)
    ax.grid(False)

    # Bottom: episode count at each timestep
    ax_count.fill_between(x, counts_trim, color=PALETTE["accent"], alpha=0.4)
    ax_count.plot(x, counts_trim, color=PALETTE["accent"], linewidth=0.8)
    ax_count.set_ylabel("# eps", fontsize=8, color=PALETTE["muted"])
    ax_count.set_xlabel("Timestep")
    ax_count.set_xlim(0, T + 60)
    ax_count.yaxis.set_label_position("right")
    ax_count.yaxis.tick_right()
    ax_count.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "fig3_action_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ── Figure 4: Per-Branch Parameter Scatter ────────────────────────────────────

def plot_param_scatter(data: dict, output_dir: Path):
    k1 = data["all_params_k1"]   # (N,)    throttle
    k2 = data["all_params_k2"]   # (N, 2)  intensity, dur
    k3 = data["all_params_k3"]   # (N, 2)  intensity, dur

    all_discrete = data["all_discrete"]
    n_total = len(all_discrete)
    freq_k1 = (all_discrete == 1).sum() / n_total * 100
    freq_k2 = (all_discrete == 2).sum() / n_total * 100
    freq_k3 = (all_discrete == 3).sum() / n_total * 100
    freq_k0 = (all_discrete == 0).sum() / n_total * 100

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.06, right=0.97, top=0.88, bottom=0.08,
    )

    # ── k=0 coast: just a usage frequency bar ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(["coast\n(k=0)"], [freq_k0],
            color=ACTION_COLORS[0], alpha=0.8, width=0.5)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Usage %", color=PALETTE["muted"])
    ax0.set_title(f"k=0 coast\n(no params)", fontsize=9, color=ACTION_COLORS[0])
    ax0.text(0, freq_k0 + 2, f"{freq_k0:.1f}%",
             ha="center", va="bottom", fontsize=11, color=ACTION_COLORS[0])
    ax0.grid(True, axis="y", alpha=0.4)

    # ── k=1 main_engine: throttle histogram ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1:3])
    if len(k1) > 0:
        n_bins = min(50, max(10, len(k1) // 10))
        ax1.hist(k1, bins=n_bins, color=ACTION_COLORS[1], alpha=0.85,
                 edgecolor=PALETTE["bg"], linewidth=0.4)
        ax1.axvline(k1.mean(), color="white", linewidth=1.2, linestyle="--",
                    label=f"mean={k1.mean():.2f}")
        ax1.axvspan(-1, -0.5, alpha=0.07, color=PALETTE["rose"])
        ax1.axvspan(0.5, 1, alpha=0.07, color=PALETTE["green"])
        ax1.set_xlabel("throttle  x_k[0] ∈ [-1, 1]")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.4)
        _add_cluster_note(ax1, k1, "throttle")
    else:
        ax1.text(0.5, 0.5, "no k=1 actions", ha="center", va="center",
                 transform=ax1.transAxes, color=PALETTE["muted"])
    ax1.set_title(
        f"k=1 main engine — throttle  ({freq_k1:.1f}% of steps)",
        fontsize=9, color=ACTION_COLORS[1],
    )
    ax1.set_xlim(-1.1, 1.1)

    # ── k=1 box plot on right ─────────────────────────────────────────────
    ax1b = fig.add_subplot(gs[0, 3])
    if len(k1) > 0:
        bp = ax1b.boxplot(k1, vert=True, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2),
                          boxprops=dict(facecolor=ACTION_COLORS[1], alpha=0.6),
                          whiskerprops=dict(color=PALETTE["muted"]),
                          capprops=dict(color=PALETTE["muted"]),
                          flierprops=dict(marker=".", markersize=2,
                                          markerfacecolor=PALETTE["muted"]))
        ax1b.set_xticks([1])
        ax1b.set_xticklabels(["throttle"])
        ax1b.set_ylim(-1.1, 1.1)
        ax1b.set_ylabel("value")
        ax1b.grid(True, axis="y", alpha=0.4)
    ax1b.set_title("spread", fontsize=9, color=ACTION_COLORS[1])

    # ── k=2 left_boost: 2D scatter ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0:2])
    if len(k2) > 1:
        _scatter_2d(ax2, k2[:, 0], k2[:, 1], ACTION_COLORS[2],
                    "intensity  x_k[0]", "duration_frac  x_k[1]")
        _add_cluster_note_2d(ax2, k2)
    else:
        ax2.text(0.5, 0.5, "no k=2 actions", ha="center", va="center",
                 transform=ax2.transAxes, color=PALETTE["muted"])
    ax2.set_title(
        f"k=2 left boost — intensity × duration  ({freq_k2:.1f}% of steps)",
        fontsize=9, color=ACTION_COLORS[2],
    )

    # ── k=3 right_boost: 2D scatter ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2:4])
    if len(k3) > 1:
        _scatter_2d(ax3, k3[:, 0], k3[:, 1], ACTION_COLORS[3],
                    "intensity  x_k[0]", "duration_frac  x_k[1]")
        _add_cluster_note_2d(ax3, k3)
    else:
        ax3.text(0.5, 0.5, "no k=3 actions", ha="center", va="center",
                 transform=ax3.transAxes, color=PALETTE["muted"])
    ax3.set_title(
        f"k=3 right boost — intensity × duration  ({freq_k3:.1f}% of steps)",
        fontsize=9, color=ACTION_COLORS[3],
    )

    # Overall action usage bar at the top
    fig.text(0.5, 0.94,
             "Fig 4 — Per-Branch Continuous Parameter Distributions",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=PALETTE["text"])
    fig.text(0.5, 0.905,
             f"coast {freq_k0:.1f}%   main {freq_k1:.1f}%   "
             f"left {freq_k2:.1f}%   right {freq_k3:.1f}%",
             ha="center", va="center", fontsize=9, color=PALETTE["muted"])

    out = output_dir / "fig4_param_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


def _scatter_2d(ax, xs, ys, color, xlabel, ylabel, max_pts=4000):
    if len(xs) > max_pts:
        idx = np.random.choice(len(xs), max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    ax.scatter(xs, ys, c=color, alpha=0.25, s=6, linewidths=0)
    # Marginal density lines
    from scipy.stats import gaussian_kde
    try:
        for arr, span, fixed, horiz in [
            (xs, np.linspace(-1.05, 1.05, 200), -1.1, False),
            (ys, np.linspace(-1.05, 1.05, 200), -1.1, True),
        ]:
            if arr.std() > 1e-6:
                kde = gaussian_kde(arr)
                density = kde(span)
                density = density / density.max() * 0.18
                if not horiz:
                    ax.plot(span, np.full_like(span, fixed) + density,
                            color=color, linewidth=1.0, alpha=0.7)
                else:
                    ax.plot(np.full_like(span, fixed) + density, span,
                            color=color, linewidth=1.0, alpha=0.7)
    except Exception:
        pass

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=PALETTE["border"], linewidth=0.5, alpha=0.5)
    ax.axvline(0, color=PALETTE["border"], linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)


def _add_cluster_note(ax, arr, name):
    """Print std to indicate clustering vs uniform."""
    std = arr.std()
    note = f"std={std:.3f}"
    if std < 0.35:
        note += "  ← clustered ✓"
        color = PALETTE["green"]
    else:
        note += "  ← spread (check)"
        color = PALETTE["amber"]
    ax.text(0.02, 0.97, note, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=color)


def _add_cluster_note_2d(ax, arr):
    std_x = arr[:, 0].std()
    std_y = arr[:, 1].std()
    note = f"std: ({std_x:.2f}, {std_y:.2f})"
    clustered = std_x < 0.4 and std_y < 0.4
    color = PALETTE["green"] if clustered else PALETTE["amber"]
    suffix = "  ← clustered ✓" if clustered else "  ← spread (check)"
    ax.text(0.02, 0.97, note + suffix, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=color)


# ── Summary stats print ────────────────────────────────────────────────────────

def print_summary(data: dict):
    rewards = data["episode_rewards"]
    lengths = data["episode_lengths"]
    discrete = data["all_discrete"]
    n_total = len(discrete)

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
        pct = 100 * (discrete == k).sum() / n_total
        bar = "█" * int(pct / 2)
        print(f"    k={k} {ACTION_NAMES[k]:15s} : {pct:5.1f}%  {bar}")
    print()
    k1 = data["all_params_k1"]
    if len(k1) > 0:
        print(f"  k=1 throttle     : mean={k1.mean():.3f}  std={k1.std():.3f}")
    for label, arr in [("k=2", data["all_params_k2"]), ("k=3", data["all_params_k3"])]:
        if len(arr) > 0:
            print(f"  {label} intensity   : mean={arr[:,0].mean():.3f}  std={arr[:,0].std():.3f}")
            print(f"  {label} dur_frac    : mean={arr[:,1].mean():.3f}  std={arr[:,1].std():.3f}")

    # Expert verdict
    print()
    verdict_ok = (
        rewards.mean() > 150
        and (rewards > 0).mean() > 0.85
        and lengths.std() < 100
    )
    if verdict_ok:
        print("  ✓  Agent looks like a strong expert — safe to use for IL.")
    else:
        print("  ⚠  Expert quality uncertain — check figures before proceeding.")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expert quality visualization for Hybrid SAC LunarLander.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", type=str, help="Path to SAC checkpoint (.pt).")
    src.add_argument("--zarr-path",  type=str, help="Path to pre-saved zarr rollout.")

    parser.add_argument("--output-dir",    type=str,   default="./expert_figures")
    parser.add_argument("--n-episodes",    type=int,   default=100,
                        help="Episodes to roll out (model-path mode).")
    parser.add_argument("--n-traj",        type=int,   default=15,
                        help="Trajectories to show in overlay plot.")
    parser.add_argument("--max-steps",     type=int,   default=300,
                        help="Max timestep shown in action heatmap.")
    parser.add_argument("--deterministic", action="store_true",  default=True)
    parser.add_argument("--stochastic",    action="store_false", dest="deterministic")
    parser.add_argument("--seed-mean",     type=float, default=0.0)
    parser.add_argument("--seed-std",      type=float, default=10000.0)
    parser.add_argument("--collect-seed",  type=int,   default=42)
    parser.add_argument("--img-size",      type=int,   nargs=2, default=[96, 96])
    parser.add_argument("--sub-steps",     type=int,   default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()

    if args.model_path:
        print(f"Collecting data from checkpoint: {args.model_path}")
        data = collect_from_checkpoint(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            n_traj=args.n_traj,
            deterministic=args.deterministic,
            seed_mean=args.seed_mean,
            seed_std=args.seed_std,
            collect_seed=args.collect_seed,
            img_size=tuple(args.img_size),
            sub_steps=args.sub_steps,
        )
    else:
        print(f"Loading data from zarr: {args.zarr_path}")
        data = collect_from_zarr(args.zarr_path, n_traj=args.n_traj)

    print_summary(data)

    print("\nGenerating figures...")
    plot_return_histogram(data, output_dir)
    plot_trajectory_overlay(data, output_dir)
    plot_action_heatmap(data, output_dir, max_steps=args.max_steps)
    plot_param_scatter(data, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")
    print("  fig1_return_histogram.png")
    print("  fig2_trajectory_overlay.png")
    print("  fig3_action_heatmap.png")
    print("  fig4_param_scatter.png")


if __name__ == "__main__":
    main()
'''
python scripts/visualize_expert.py \
    --model-path /path/to/best_model.pt \
    --n-episodes 100 \
    --n-traj 15
'''