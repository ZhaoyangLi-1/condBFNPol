"""
Roll out a trained Hybrid SAC policy on HybridLunarLander and save
trajectories as a zarr dataset.

Hybrid action semantics:
    A complete hybrid action is a pair `(k, x_k)`.
    - `hybrid_action_discrete` stores the discrete branch id `k`
    - `hybrid_action_params` stores the padded continuous parameter vector `x_k`

    In the current HybridLunarLander environment:
    - `k = 0` -> `coast`
      no continuous parameters are used, so `x_k` is effectively ignored
    - `k = 1` -> `main_engine`
      `x_k[0]` is `throttle`, `x_k[1]` is unused padding
    - `k = 2` -> `left_boost`
      `x_k = [intensity, duration_frac]`
    - `k = 3` -> `right_boost`
      `x_k = [intensity, duration_frac]`

Saved zarr structure:
    data/
        action:                  (N, 2) float32   executed low-level env action
        img:                     (N, H, W, 3) float32 RGB image in [0, 1]
        state:                   (N, 8) float32 state vector
        hybrid_action_discrete:  (N,) int64      discrete branch id k (use this for action)
        hybrid_action_params:    (N, 2) float32  padded x_k used by env (use this for action)
        hybrid_action_flat:      (N, D) float32  shared flat actor output 
    meta/
        episode_ends:            (n_episodes,) int64

`data/action` remains 2D so existing continuous-action consumers can keep
working. The additional hybrid-action arrays preserve the full policy output.

Usage:
    python scripts/rollout_lunar_lander_to_zarr.py \
        --model-path /scr2/zhaoyang/lunar_lander_outputs/hybrid_sac_lunar_lander/checkpoints/best_model.pt \
        --n-episodes 200 \
        --output-path /scr2/zhaoyang/BFN_data/lunar_lander/lunar_lander_replay.zarr
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.lunar_lander import HybridLunarLander
from scripts.train_hybrid_sac_lunar_lander import HybridActionLayout, Policy


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "policy" not in ckpt:
        raise ValueError(
            f"Checkpoint {model_path} does not look like a Hybrid SAC checkpoint: "
            "missing 'policy' key."
        )
    if "cfg" not in ckpt:
        raise ValueError(
            f"Checkpoint {model_path} is missing 'cfg'; cannot reconstruct policy."
        )
    return ckpt


def build_env(args, ckpt_cfg: dict) -> HybridLunarLander:
    env_cfg = ckpt_cfg.get("env", {})
    sub_steps = args.sub_steps if args.sub_steps is not None else env_cfg.get("sub_steps", 1)
    return HybridLunarLander(
        use_image_obs=True,
        img_size=tuple(args.img_size),
        sub_steps=int(sub_steps),
    )


def load_policy(
    ckpt: dict,
    env: HybridLunarLander,
    device: torch.device,
) -> tuple[Policy, HybridActionLayout]:
    cfg = ckpt["cfg"]
    action_layout = HybridActionLayout.from_env_spec(env.get_action_spec())

    saved_layout = ckpt.get("action_layout", {})
    if saved_layout:
        if int(saved_layout.get("num_discrete", -1)) != action_layout.num_discrete:
            raise ValueError("Checkpoint/action-space discrete branch count mismatch.")
        if int(saved_layout.get("total_param_dim", -1)) != action_layout.total_param_dim:
            raise ValueError("Checkpoint/action-space flat continuous dim mismatch.")

    obs_space = env.observation_space["state"]
    obs_dim = int(obs_space.shape[0])

    network_cfg = cfg.get("network", {})
    sac_cfg = cfg.get("sac", {})
    hidden_dims = list(network_cfg.get("hidden_dims", [256, 256]))

    policy = Policy(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=hidden_dims,
        activation=network_cfg.get("activation", "relu"),
        weights_init=network_cfg.get("weights_init", "xavier"),
        bias_init=network_cfg.get("bias_init", "zeros"),
        log_std_min=float(sac_cfg.get("log_std_min", -5.0)),
        log_std_max=float(sac_cfg.get("log_std_max", 0.0)),
    ).to(device)

    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, action_layout


def filter_episodes(arrays: dict, episode_ends: list[int], episode_rewards: list[float], min_reward: float):
    keep = {key: [] for key in arrays.keys()}
    keep_ends = []
    keep_rewards = []
    prev_end = 0
    kept_steps = 0

    for ep_idx, end in enumerate(episode_ends):
        if episode_rewards[ep_idx] >= min_reward:
            for key, values in arrays.items():
                keep[key].extend(values[prev_end:end])
            ep_len = end - prev_end
            kept_steps += ep_len
            keep_ends.append(kept_steps)
            keep_rewards.append(episode_rewards[ep_idx])
        prev_end = end

    if len(keep_ends) == 0:
        print("WARNING: No episodes passed the filter. Saving all episodes instead.")
        return arrays, episode_ends, episode_rewards
    return keep, keep_ends, keep_rewards


def rollout_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model_path, device)
    ckpt_cfg = ckpt.get("cfg", {})

    env = build_env(args, ckpt_cfg)
    policy, action_layout = load_policy(ckpt, env, device)

    rng = np.random.default_rng(args.collect_seed)

    arrays = {
        "action": [],
        "img": [],
        "state": [],
        "hybrid_action_discrete": [],
        "hybrid_action_params": [],
        "hybrid_action_flat": [],
    }
    episode_ends: list[int] = []
    episode_rewards: list[float] = []
    total_steps = 0

    print(f"Collecting {args.n_episodes} episodes from Hybrid SAC checkpoint ...")

    for ep in range(args.n_episodes):
        seed = abs(int(rng.normal(args.seed_mean, args.seed_std)))
        obs, _ = env.reset(seed=seed)

        terminated = truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (terminated or truncated):
            state = obs["state"].astype(np.float32)
            image = obs["image"].astype(np.float32) / 255.0

            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                flat_action_t, discrete_action_t, _, _, _ = policy.get_action(
                    state_t,
                    deterministic=args.deterministic,
                )

            discrete_action = int(discrete_action_t.item())
            flat_action = flat_action_t.cpu().numpy()[0].astype(np.float32)
            hybrid_action = action_layout.flat_to_env_action(discrete_action, flat_action)
            low_level_action = env._convert_action(discrete_action, hybrid_action["x_k"]).astype(np.float32)

            arrays["state"].append(state)
            arrays["img"].append(image)
            arrays["action"].append(low_level_action)
            arrays["hybrid_action_discrete"].append(discrete_action)
            arrays["hybrid_action_params"].append(hybrid_action["x_k"].astype(np.float32))
            arrays["hybrid_action_flat"].append(flat_action)

            obs, reward, terminated, truncated, _ = env.step(hybrid_action)
            ep_reward += reward
            ep_steps += 1

        total_steps += ep_steps
        episode_ends.append(total_steps)
        episode_rewards.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Episode {ep + 1:>4d}/{args.n_episodes} | "
                f"steps={ep_steps:>4d} | reward={ep_reward:>8.1f} | total_steps={total_steps}"
            )

    env.close()

    if args.min_reward is not None:
        print(f"\nFiltering episodes with reward >= {args.min_reward} ...")
        arrays, episode_ends, episode_rewards = filter_episodes(
            arrays=arrays,
            episode_ends=episode_ends,
            episode_rewards=episode_rewards,
            min_reward=args.min_reward,
        )
        total_steps = episode_ends[-1] if episode_ends else 0
        print(f"  Kept {len(episode_ends)}/{args.n_episodes} episodes, {total_steps} steps")

    if len(episode_ends) == 0:
        raise RuntimeError("No episodes collected.")

    np_arrays = {
        "action": np.stack(arrays["action"]).astype(np.float32),
        "img": np.stack(arrays["img"]).astype(np.float32),
        "state": np.stack(arrays["state"]).astype(np.float32),
        "hybrid_action_discrete": np.asarray(arrays["hybrid_action_discrete"], dtype=np.int64),
        "hybrid_action_params": np.stack(arrays["hybrid_action_params"]).astype(np.float32),
        "hybrid_action_flat": np.stack(arrays["hybrid_action_flat"]).astype(np.float32),
    }
    episode_ends_np = np.asarray(episode_ends, dtype=np.int64)

    ep_lens = np.diff(np.concatenate([[0], episode_ends_np]))
    chunk_len = max(int(np.median(ep_lens)), 1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(output_path), mode="w")
    root.attrs["policy_type"] = "hybrid_sac"
    root.attrs["action_names"] = [action_layout.action_names[k] for k in range(action_layout.num_discrete)]
    root.attrs["flat_action_dim"] = int(action_layout.total_param_dim)
    root.attrs["img_size"] = list(args.img_size)

    data = root.create_group("data")
    data.create_dataset(
        "action",
        data=np_arrays["action"],
        chunks=(chunk_len, np_arrays["action"].shape[1]),
        dtype="float32",
    )
    data.create_dataset(
        "img",
        data=np_arrays["img"],
        chunks=(chunk_len, *args.img_size, 3),
        dtype="float32",
    )
    data.create_dataset(
        "state",
        data=np_arrays["state"],
        chunks=(chunk_len, np_arrays["state"].shape[1]),
        dtype="float32",
    )
    data.create_dataset(
        "hybrid_action_discrete",
        data=np_arrays["hybrid_action_discrete"],
        chunks=(chunk_len,),
        dtype="int64",
    )
    data.create_dataset(
        "hybrid_action_params",
        data=np_arrays["hybrid_action_params"],
        chunks=(chunk_len, np_arrays["hybrid_action_params"].shape[1]),
        dtype="float32",
    )
    data.create_dataset(
        "hybrid_action_flat",
        data=np_arrays["hybrid_action_flat"],
        chunks=(chunk_len, np_arrays["hybrid_action_flat"].shape[1]),
        dtype="float32",
    )

    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=episode_ends_np, dtype="int64")

    rewards = np.asarray(episode_rewards, dtype=np.float32)
    print(f"\nSaved zarr to: {output_path}")
    print(f"  data/action                 : {np_arrays['action'].shape}")
    print(f"  data/img                    : {np_arrays['img'].shape}")
    print(f"  data/state                  : {np_arrays['state'].shape}")
    print(f"  data/hybrid_action_discrete : {np_arrays['hybrid_action_discrete'].shape}")
    print(f"  data/hybrid_action_params   : {np_arrays['hybrid_action_params'].shape}")
    print(f"  data/hybrid_action_flat     : {np_arrays['hybrid_action_flat'].shape}")
    print(f"  meta/episode_ends           : {episode_ends_np.shape}")
    print(f"  chunk_len                   : {chunk_len}")
    print(f"  episodes                    : {len(episode_ends_np)}")
    print(f"  total steps                 : {total_steps}")
    print(f"  reward                      : {rewards.mean():.1f} +/- {rewards.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Roll out Hybrid SAC LunarLander policy to zarr.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Hybrid SAC checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/scr2/zhaoyang/BFN_data/lunar_lander/lunar_lander_replay.zarr",
    )
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--img-size", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--sub-steps", type=int, default=None)
    parser.add_argument("--seed-mean", type=float, default=0.0)
    parser.add_argument("--seed-std", type=float, default=10000.0)
    parser.add_argument(
        "--collect-seed",
        type=int,
        default=42,
        help="RNG seed for sampling episode reset seeds.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions during rollout.",
    )
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Only keep episodes with reward >= this threshold.",
    )
    args = parser.parse_args()
    rollout_and_save(args)


if __name__ == "__main__":
    main()
