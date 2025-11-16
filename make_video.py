"""Generate a rollout video of a trained policy on MazeEnv (conditional or guided)."""

from __future__ import annotations

import argparse
import os
import imageio.v2 as imageio
import torch as t
import numpy as np

from environments import MazeEnv, AntMazeEnv, PushTEnv
from policies import ConditionalBFNPolicy
from guided_bfn_policy import GuidedBFNPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create rollout video for MazeEnv policy.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt).")
    p.add_argument(
        "--out",
        type=str,
        default="maze_rollout.gif",
        help="Output video file (gif recommended).",
    )
    p.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to record."
    )
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode.")
    p.add_argument(
        "--fps", type=int, default=4, help="Frames per second for the video."
    )
    p.add_argument(
        "--device", type=str, default="cpu", help="Torch device to run policy."
    )
    p.add_argument(
        "--guided", action="store_true", help="Load a GuidedBFNPolicy checkpoint."
    )
    p.add_argument(
        "--antmaze",
        action="store_true",
        help="Use AntMaze environment instead of grid Maze.",
    )
    p.add_argument(
        "--pusht",
        action="store_true",
        help="Use PushT environment instead of grid Maze.",
    )
    p.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Environment name if using AntMaze/PushT.",
    )
    return p.parse_args()


def load_conditional_policy(
    device: str, ckpt: str, env: MazeEnv
) -> ConditionalBFNPolicy:
    policy = ConditionalBFNPolicy(
        action_space=env.action_space,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    ).to(device)
    state = t.load(ckpt, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def load_guided_policy(device: str, ckpt: str) -> GuidedBFNPolicy:
    state = t.load(ckpt, map_location=device)

    # Infer dims from weights
    enc_w0 = state["vision_encoder"]["net.0.weight"]
    obs_dim = enc_w0.shape[1]
    emb_dim = enc_w0.shape[0]

    net_w0 = state["network"]["net.0.weight"]
    hidden_dim = net_w0.shape[0]
    input_dim = net_w0.shape[1]

    action_dim = state.get("action_dim", state["network"]["net.4.weight"].shape[0])
    cond_dim = input_dim - action_dim - 1

    class VisionEncoderMLP(t.nn.Module):
        def __init__(self, obs_dim: int, emb_dim: int):
            super().__init__()
            self.net = t.nn.Sequential(
                t.nn.Linear(obs_dim, emb_dim),
                t.nn.ReLU(),
                t.nn.Linear(emb_dim, emb_dim),
            )
            self.output_dim = emb_dim

        def forward(self, x: t.Tensor) -> t.Tensor:
            return self.net(x)

    class BackboneMLP(t.nn.Module):
        def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int):
            super().__init__()
            inp = action_dim + cond_dim + 1
            self.net = t.nn.Sequential(
                t.nn.Linear(inp, hidden_dim),
                t.nn.GELU(),
                t.nn.Linear(hidden_dim, hidden_dim),
                t.nn.GELU(),
                t.nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, A: t.Tensor, t_tensor: t.Tensor, cond: t.Tensor) -> t.Tensor:
            B, T_p, _ = A.shape
            cond_last = cond[:, -1, :]
            cond_ex = cond_last[:, None, :].expand(B, T_p, cond_last.size(-1))
            t_ex = t_tensor.view(B, 1, 1).expand(B, T_p, 1)
            x = t.cat([A, cond_ex, t_ex], dim=-1)
            return self.net(x)

    vision = VisionEncoderMLP(obs_dim, emb_dim)
    backbone = BackboneMLP(action_dim, cond_dim, hidden_dim)

    policy = GuidedBFNPolicy(
        backbone_transformer=backbone,
        vision_encoder=vision,
        action_dim=action_dim,
        bfn_timesteps=state.get("N_steps", 50),
        obs_horizon_T_o=state.get("T_o", 2),
        action_pred_horizon_T_p=state.get("T_p", 4),
        action_exec_horizon_T_a=state.get("T_a", 1),
        cfg_uncond_prob=state.get("p_uncond", 0.1),
        cfg_guidance_scale_w=state.get("w", 3.0),
        grad_guidance_scale_alpha=state.get("alpha", 0.0),
        device=device,
    )
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


def main():
    args = parse_args()

    if args.antmaze or args.pusht:
        if not args.guided:
            raise ValueError(
                "AntMaze/PushT playback currently supported only for guided checkpoints; pass --guided."
            )
        if args.antmaze:
            env_name = args.env_name or "AntMaze_UMaze-v4"
            env = AntMazeEnv(env_name=env_name, render_mode="rgb_array")
        else:
            env_name = args.env_name or "gym_pusht/PushT-v0"
            env = PushTEnv(env_name=env_name, render_mode="rgb_array")
    else:
        env = MazeEnv(max_steps=args.max_steps, render_mode="rgb_array")

    if args.guided:
        policy = load_guided_policy(args.device, args.ckpt)
    else:
        policy = load_conditional_policy(args.device, args.ckpt, env)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames = []

    with t.inference_mode():
        for _ in range(args.episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            frames.append(env.render())

            while not (done or truncated):
                if args.guided:
                    obs_arr = obs
                    if isinstance(obs_arr, dict):
                        parts = [
                            np.array(obs_arr[k]).ravel() for k in sorted(obs_arr.keys())
                        ]
                        obs_arr = np.concatenate(parts, axis=0)
                    obs_seq = (
                        t.tensor(obs_arr, device=policy.device, dtype=t.float32)
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )
                    obs_seq = obs_seq.repeat(1, policy.T_o, 1)
                    goal = t.zeros(1, policy.action_dim, device=policy.device)
                    traj = policy.predict_action_sequence(obs_seq, goal)
                    if args.antmaze or args.pusht:
                        action = traj[0, 0, :].detach().cpu().numpy()
                    else:
                        logits = traj[:, 0, :]
                        action = int(t.argmax(logits, dim=-1).item())
                else:
                    obs_t = t.tensor(
                        obs, device=policy.device, dtype=t.float32
                    ).unsqueeze(0)
                    logits = policy(obs_t, deterministic=True)
                    action = int(t.argmax(logits, dim=-1).item())
                obs, reward, done, truncated, _ = env.step(action)
                frames.append(env.render())

    imageio.mimsave(args.out, frames, fps=args.fps)
    env.close()
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
