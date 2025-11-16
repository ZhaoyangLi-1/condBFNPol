"""Quick visualization script for the MazeEnv."""

from __future__ import annotations

import argparse
import time
from typing import Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

from environments import MazeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the MazeEnv.")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to roll out."
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Maximum steps per episode."
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="ansi",
        choices=["human", "ansi", "rgb_array"],
        help="Render mode to use.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.25, help="Delay between rendered frames."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    return parser.parse_args()


def render_text(
    env: MazeEnv, step: int, reward: float, terminated: bool, truncated: bool
):
    text = env.render()
    print(
        f"\nStep {step} | reward={reward:.3f} | done={terminated} | truncated={truncated}"
    )
    print(text)


def render_rgb(env: MazeEnv, title: str, ax) -> None:
    frame = env.render()
    ax.clear()
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")
    plt.pause(0.001)


def main():
    args = parse_args()
    env = MazeEnv(
        max_steps=args.max_steps, seed=args.seed, render_mode=args.render_mode
    )

    if args.render_mode == "rgb_array" and plt is None:
        raise ImportError(
            "matplotlib is required for rgb_array rendering. Try --render-mode ansi."
        )

    for ep in range(args.episodes):
        obs, info = env.reset()
        if args.render_mode in ("human", "ansi"):
            render_text(env, step=0, reward=0.0, terminated=False, truncated=False)

        ax = None
        if args.render_mode == "rgb_array" and plt is not None:
            _, ax = plt.subplots()
            render_rgb(env, f"Episode {ep} Step 0", ax=ax)

        done = False
        truncated = False
        step = 0
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step += 1

            if args.render_mode in ("human", "ansi"):
                render_text(env, step, reward, done, truncated)
            elif args.render_mode == "rgb_array" and plt is not None:
                render_rgb(env, f"Episode {ep} Step {step}", ax=ax)

            time.sleep(args.sleep)

    env.close()
    if plt is not None and args.render_mode == "rgb_array":
        plt.show(block=True)


if __name__ == "__main__":
    main()
