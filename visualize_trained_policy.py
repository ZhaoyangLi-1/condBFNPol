"""Roll out a trained conditional BFN policy on the MazeEnv and visualize it."""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch as t

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

from environments import MazeEnv
from policies import ConditionalBFNPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a trained BFN policy on MazeEnv.")
    p.add_argument("--ckpt", type=str, required=False, help="Path to policy checkpoint (.pt).")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes to roll out.")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode.")
    p.add_argument("--render-mode", type=str, default="ansi", choices=["ansi", "human", "rgb_array"])
    p.add_argument("--sleep", type=float, default=0.1, help="Delay between frames (seconds).")
    p.add_argument("--device", type=str, default="cpu", help="Torch device to run the policy on.")
    return p.parse_args()


def render_text(env: MazeEnv, step: int, reward: float, terminated: bool, truncated: bool):
    text = env.render()
    print(f"\nStep {step} | reward={reward:.3f} | done={terminated} | truncated={truncated}")
    print(text)


def render_rgb(env: MazeEnv, title: str, ax) -> None:
    frame = env.render()
    ax.clear()
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")
    plt.pause(0.001)


def load_policy(device: str, ckpt: Optional[str], env: MazeEnv) -> ConditionalBFNPolicy:
    policy = ConditionalBFNPolicy(
        action_space=env.action_space,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    ).to(device)
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state = t.load(ckpt, map_location=device)
        policy.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt}")
    policy.eval()
    return policy


def policy_step(policy: ConditionalBFNPolicy, obs: np.ndarray) -> int:
    obs_t = t.tensor(obs, device=policy.device, dtype=t.float32).unsqueeze(0)
    logits = policy(obs_t, deterministic=True)
    action = int(t.argmax(logits, dim=-1).item())
    return action


def main():
    args = parse_args()
    env = MazeEnv(max_steps=args.max_steps, render_mode=args.render_mode)
    policy = load_policy(args.device, args.ckpt, env)

    if args.render_mode == "rgb_array" and plt is None:
        raise ImportError("matplotlib is required for rgb_array rendering. Try --render-mode ansi.")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        step = 0
        done = False
        truncated = False

        ax = None
        if args.render_mode == "rgb_array" and plt is not None:
            _, ax = plt.subplots()
            render_rgb(env, f"Episode {ep} Step {step}", ax=ax)
        elif args.render_mode in ("human", "ansi"):
            render_text(env, step, reward=0.0, terminated=False, truncated=False)

        while not (done or truncated):
            action = policy_step(policy, obs)
            obs, reward, done, truncated, _ = env.step(action)
            step += 1

            if args.render_mode == "rgb_array" and plt is not None:
                render_rgb(env, f"Episode {ep} Step {step}", ax=ax)
            elif args.render_mode in ("human", "ansi"):
                render_text(env, step, reward, done, truncated)

            time.sleep(args.sleep)

    env.close()
    if plt is not None and args.render_mode == "rgb_array":
        plt.show(block=True)


if __name__ == "__main__":
    main()
