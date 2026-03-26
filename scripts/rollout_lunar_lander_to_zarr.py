"""
Roll out a trained PPO policy on HybridLunarLanderV2 and save trajectories
as a zarr dataset compatible with diffusion_policy / BFN training.

Output zarr structure (matches pusht_cchi_v7_replay.zarr):
    data/
        action:  (N, 2)        float32   continuous action u
        img:     (N, 96, 96, 3) float32  RGB image in [0, 1]
        state:   (N, 8)        float32   8-dim state vector
    meta/
        episode_ends: (n_episodes,) int64

Usage:
    python scripts/rollout_lunar_lander_to_zarr.py \
        --model-path outputs/ppo_lunar_lander/best_model.pt \
        --n-episodes 200 \
        --output-path /scr2/zhaoyang/BFN_data/lunar_lander/lunar_lander_replay.zarr
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import zarr
from environments.lunar_lander import HybridLunarLanderV2


# ── Re-use the same ActorCritic from training ────────────────────────────────
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.shared(obs)
        mean = self.policy_mean(h)
        std = self.policy_log_std.exp().expand_as(mean)
        value = self.value_head(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            action = torch.tanh(dist.sample())
        return action


def load_policy(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    ac = ActorCritic(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden=ckpt["hidden_dim"],
    ).to(device)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    return ac


# ── Rollout & save ────────────────────────────────────────────────────────────

def rollout_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.model_path, device)

    env = HybridLunarLanderV2(
        use_image_obs=True,
        img_size=tuple(args.img_size),
        trigger_side=True,
    )

    rng = np.random.default_rng(args.collect_seed)

    all_actions = []
    all_imgs = []
    all_states = []
    episode_ends = []
    total_steps = 0
    total_reward_list = []

    print(f"Collecting {args.n_episodes} episodes ...")

    for ep in range(args.n_episodes):
        seed = abs(int(rng.normal(args.seed_mean, args.seed_std)))
        obs, _ = env.reset(seed=seed)

        terminated = truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (terminated or truncated):
            state = obs["state"]    # (8,)
            image = obs["image"]    # (H, W, 3) uint8

            # Store observation + action at this timestep
            all_states.append(state.astype(np.float32))
            all_imgs.append(image.astype(np.float32) / 255.0)

            # Policy acts on state
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy.get_action(state_t, deterministic=args.deterministic)
                action_np = action.cpu().numpy()[0].astype(np.float32)

            action_np = np.clip(action_np, -1.0, 1.0)
            all_actions.append(action_np)

            # Step env
            hybrid_action = {"mode": 0, "u": action_np}
            obs, reward, terminated, truncated, _ = env.step(hybrid_action)
            ep_reward += reward
            ep_steps += 1

        total_steps += ep_steps
        episode_ends.append(total_steps)
        total_reward_list.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1:>4d}/{args.n_episodes} | "
                  f"steps={ep_steps:>4d} | reward={ep_reward:>7.1f} | "
                  f"total_steps={total_steps}")

    env.close()

    # ── Filter low-reward episodes (optional) ─────────────────────────────────
    if args.min_reward is not None:
        print(f"\nFiltering episodes with reward >= {args.min_reward} ...")
        keep_actions, keep_imgs, keep_states = [], [], []
        keep_ends = []
        prev_end = 0
        kept_steps = 0
        kept_eps = 0
        for ep_idx, end in enumerate(episode_ends):
            if total_reward_list[ep_idx] >= args.min_reward:
                ep_len = end - prev_end
                keep_actions.extend(all_actions[prev_end:end])
                keep_imgs.extend(all_imgs[prev_end:end])
                keep_states.extend(all_states[prev_end:end])
                kept_steps += ep_len
                keep_ends.append(kept_steps)
                kept_eps += 1
            prev_end = end

        if kept_eps == 0:
            print("WARNING: No episodes passed the filter! Saving all episodes.")
        else:
            all_actions = keep_actions
            all_imgs = keep_imgs
            all_states = keep_states
            episode_ends = keep_ends
            total_steps = kept_steps
            print(f"  Kept {kept_eps}/{args.n_episodes} episodes, {total_steps} steps")

    # ── Stack into arrays ─────────────────────────────────────────────────────
    all_actions = np.stack(all_actions)                         # (N, 2)
    all_imgs = np.stack(all_imgs)                               # (N, H, W, 3)
    all_states = np.stack(all_states)                           # (N, 8)
    episode_ends = np.array(episode_ends, dtype=np.int64)       # (n_eps,)

    # ── Chunk size = median episode length ────────────────────────────────────
    ep_lens = np.diff(np.concatenate([[0], episode_ends]))
    chunk_len = max(int(np.median(ep_lens)), 1)

    # ── Write zarr ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    root = zarr.open(args.output_path, mode="w")

    data = root.create_group("data")
    data.create_dataset("action", data=all_actions,
                        chunks=(chunk_len, all_actions.shape[1]),
                        dtype="float32")
    data.create_dataset("img", data=all_imgs,
                        chunks=(chunk_len, *args.img_size, 3),
                        dtype="float32")
    data.create_dataset("state", data=all_states,
                        chunks=(chunk_len, all_states.shape[1]),
                        dtype="float32")

    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=episode_ends, dtype="int64")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSaved zarr to: {args.output_path}")
    print(f"  data/action : {all_actions.shape}")
    print(f"  data/img    : {all_imgs.shape}")
    print(f"  data/state  : {all_states.shape}")
    print(f"  meta/episode_ends : {episode_ends.shape}")
    print(f"  chunk_len   : {chunk_len}")
    print(f"  episodes    : {len(episode_ends)}")
    print(f"  total steps : {total_steps}")
    rewards = np.array(total_reward_list[:len(episode_ends)])
    print(f"  reward      : {rewards.mean():.1f} +/- {rewards.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Rollout PPO → zarr for imitation learning")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained PPO model .pt file")
    parser.add_argument("--output-path", type=str,
                        default="/scr2/zhaoyang/BFN_data/lunar_lander/lunar_lander_replay.zarr")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--img-size", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--seed-mean", type=float, default=0.0)
    parser.add_argument("--seed-std", type=float, default=10000.0)
    parser.add_argument("--collect-seed", type=int, default=42,
                        help="RNG seed for the collection seed generator")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy for rollout")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--min-reward", type=float, default=None,
                        help="Only keep episodes with reward >= this value (e.g. 200)")
    args = parser.parse_args()
    rollout_and_save(args)


if __name__ == "__main__":
    main()
