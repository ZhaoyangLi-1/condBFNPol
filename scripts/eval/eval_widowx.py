#!/usr/bin/env python3
"""Evaluate trained diffusion/BFN policies on WidowX real robot.

This script mirrors the control loop style in `example_widowx.py`, but loads a
policy checkpoint trained in this project and runs closed-loop rollout on
WidowX.

Example:
    python scripts/eval/eval_widowx.py \
        --checkpoint_path /path/to/checkpoints/epoch=0200-xxx.ckpt \
        --ip localhost \
        --port 5556 \
        --device cuda:0 \
        --num_timesteps 120 \
        --action_horizon 4
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from absl import app, flags, logging
from omegaconf import OmegaConf

try:
    import imageio
except ImportError:  # optional; only needed when saving videos
    imageio = None

try:
    import dill
except ImportError:  # fallback for environments without dill
    import pickle as dill

try:
    import cv2
except ImportError:  # optional dependency
    cv2 = None


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIFFUSION_POLICY_ROOT = PROJECT_ROOT / "src" / "diffusion-policy"
if DIFFUSION_POLICY_ROOT.exists() and str(DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

# Optional local Octo examples path (for envs.widowx_env import)
DEFAULT_OCTO_EXAMPLES = PROJECT_ROOT.parent / "octo" / "examples"
if DEFAULT_OCTO_EXAMPLES.exists() and str(DEFAULT_OCTO_EXAMPLES) not in sys.path:
    sys.path.append(str(DEFAULT_OCTO_EXAMPLES))


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to .ckpt checkpoint.", required=True)
flags.DEFINE_string("device", "cuda:0", "Torch device.")
flags.DEFINE_bool("use_ema", True, "Use EMA model from workspace when available.")

flags.DEFINE_string("ip", "localhost", "IP address of the robot service.")
flags.DEFINE_integer("port", 5556, "Port of the robot service.")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial EEP xyz.")
flags.DEFINE_bool("blocking", False, "Use blocking controller.")
flags.DEFINE_integer("image_size", 256, "Image size passed to WidowX client.")
flags.DEFINE_float("step_duration", 0.2, "Control period in seconds.")
flags.DEFINE_integer("sticky_gripper_num_steps", 1, "Sticky gripper setting.")

flags.DEFINE_integer("num_episodes", 1, "Number of episodes; <=0 means run forever.")
flags.DEFINE_integer("num_timesteps", 120, "Max timesteps per episode.")
flags.DEFINE_integer("action_horizon", 4, "How many predicted actions to execute each policy query.")
flags.DEFINE_bool("interactive", True, "Wait for keyboard input before each episode.")

flags.DEFINE_integer("policy_n_timesteps", -1, "Override BFN sampling steps (>0 enables).")
flags.DEFINE_integer(
    "policy_num_inference_steps",
    -1,
    "Override diffusion num_inference_steps (>0 enables).",
)

flags.DEFINE_bool("show_image", False, "Show camera image during rollout.")
flags.DEFINE_string("video_save_path", None, "Directory to save rollout videos.")

flags.DEFINE_spaceseplist(
    "image_obs_source_keys",
    ["image_primary", "image", "camera_0", "camera_1"],
    "Prioritized raw obs keys for image observations.",
)
flags.DEFINE_spaceseplist(
    "lowdim_obs_source_keys",
    ["robot_eef_pose", "proprio", "state", "robot_state"],
    "Prioritized raw obs keys for low-dimensional observations.",
)

flags.DEFINE_bool(
    "append_gripper",
    False,
    "If true and policy action_dim==6, append one constant gripper dimension.",
)
flags.DEFINE_float("gripper_value", 1.0, "Gripper value used when append_gripper=True.")


# Bridge/Octo-style WidowX defaults
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
}


# Resolver used by some saved OmegaConf configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_workspace_and_policy(
    checkpoint_path: Path,
    device: str,
    use_ema: bool,
) -> Tuple[Any, Any, Any]:
    """Load workspace + policy from checkpoint payload."""
    payload = torch.load(
        checkpoint_path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )

    # Snapshot path fallback: payload is already a workspace object.
    if not isinstance(payload, dict) or "cfg" not in payload:
        workspace = payload
        cfg = workspace.cfg
    else:
        cfg = payload["cfg"]
        workspace_target = cfg.get("_target_", None)
        if workspace_target is None:
            raise ValueError("Checkpoint cfg does not contain `_target_` for workspace.")

        import hydra

        workspace_cls = hydra.utils.get_class(workspace_target)
        workspace = workspace_cls(cfg, output_dir=str(checkpoint_path.parent.parent))
        if hasattr(workspace, "load_payload"):
            workspace.load_payload(payload)
        else:
            workspace.load_checkpoint(path=str(checkpoint_path))

    policy = getattr(workspace, "model", None)
    ema_model = getattr(workspace, "ema_model", None)
    if use_ema and (ema_model is not None):
        policy = ema_model
        logging.info("Using EMA model for evaluation.")
    else:
        logging.info("Using non-EMA model for evaluation.")

    if policy is None:
        raise ValueError("Could not find model in workspace.")

    policy.to(device)
    policy.eval()
    return workspace, policy, cfg


def _get_shape_meta(cfg: Any) -> Dict[str, Any]:
    shape_meta = cfg.get("shape_meta", None)
    if shape_meta is None and cfg.get("task", None) is not None:
        shape_meta = cfg.task.get("shape_meta", None)
    if shape_meta is None:
        raise ValueError("Cannot find shape_meta in checkpoint cfg.")
    return shape_meta


def _is_image_like(x: Any) -> bool:
    arr = np.asarray(x)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        return False
    return (arr.shape[-1] in (1, 3)) or (arr.shape[0] in (1, 3))


def _is_vector_like(x: Any) -> bool:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return False
    if arr.ndim == 1:
        return True
    if arr.ndim >= 2:
        return arr.shape[-1] > 0
    return False


def _extract_latest_image(raw_value: Any) -> np.ndarray:
    arr = np.asarray(raw_value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected image ndim={arr.ndim}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_policy_image(image_hwc: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    c, h, w = target_shape
    if image_hwc.shape[0] != h or image_hwc.shape[1] != w:
        if cv2 is not None:
            image_hwc = cv2.resize(image_hwc, (w, h), interpolation=cv2.INTER_AREA)
        else:
            image_hwc = np.asarray(
                Image.fromarray(image_hwc).resize((w, h), resample=Image.BILINEAR)
            )
    if c == 1 and image_hwc.shape[-1] == 3:
        if cv2 is not None:
            image_hwc = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2GRAY)[..., None]
        else:
            image_hwc = np.mean(image_hwc.astype(np.float32), axis=-1, keepdims=True).astype(np.uint8)
    elif c == 3 and image_hwc.shape[-1] == 1:
        image_hwc = np.repeat(image_hwc, 3, axis=-1)
    image = image_hwc.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def _extract_latest_vector(raw_value: Any) -> np.ndarray:
    arr = np.asarray(raw_value, dtype=np.float32)
    if arr.ndim >= 2:
        arr = arr[-1]
    return arr.reshape(-1).astype(np.float32)


def _choose_image_source(
    raw_obs: Dict[str, Any],
    policy_key: str,
    prioritized_keys: List[str],
    used_sources: set,
) -> str:
    candidates = [policy_key] + list(prioritized_keys)
    candidates += [k for k in raw_obs.keys() if "image" in k.lower()]
    for key in candidates:
        if key in raw_obs and key not in used_sources and _is_image_like(raw_obs[key]):
            return key
    for key in candidates:
        if key in raw_obs and _is_image_like(raw_obs[key]):
            return key
    raise KeyError(f"No image source found for policy key `{policy_key}`. Raw keys: {list(raw_obs.keys())}")


def _choose_lowdim_source(
    raw_obs: Dict[str, Any],
    policy_key: str,
    prioritized_keys: List[str],
) -> str:
    candidates = [policy_key] + list(prioritized_keys)
    candidates += [
        k
        for k in raw_obs.keys()
        if any(token in k.lower() for token in ("state", "pose", "eef", "proprio", "joint"))
    ]
    for key in candidates:
        if key in raw_obs and _is_vector_like(raw_obs[key]):
            return key
    raise KeyError(f"No low-dim source found for policy key `{policy_key}`. Raw keys: {list(raw_obs.keys())}")


def _build_obs_source_map(
    raw_obs: Dict[str, Any],
    obs_shape_meta: Dict[str, Any],
) -> Dict[str, str]:
    source_map: Dict[str, str] = {}
    used_image_sources: set = set()
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            source = _choose_image_source(raw_obs, key, FLAGS.image_obs_source_keys, used_image_sources)
            used_image_sources.add(source)
        else:
            source = _choose_lowdim_source(raw_obs, key, FLAGS.lowdim_obs_source_keys)
        source_map[key] = source
    return source_map


def _convert_raw_obs_to_policy_obs(
    raw_obs: Dict[str, Any],
    obs_shape_meta: Dict[str, Any],
    source_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, attr in obs_shape_meta.items():
        source = source_map[key]
        shape = tuple(int(x) for x in attr["shape"])
        obs_type = attr.get("type", "low_dim")

        if source not in raw_obs:
            raise KeyError(f"Raw observation missing source key `{source}` for policy key `{key}`.")

        if obs_type == "rgb":
            image = _extract_latest_image(raw_obs[source])
            out[key] = _to_policy_image(image, shape)
        else:
            vec = _extract_latest_vector(raw_obs[source])
            target_dim = int(np.prod(shape))
            if vec.size < target_dim:
                vec = np.pad(vec, (0, target_dim - vec.size), mode="constant")
            elif vec.size > target_dim:
                vec = vec[:target_dim]
            out[key] = vec.reshape(shape).astype(np.float32)
    return out


def _history_to_torch_obs(
    obs_history: Dict[str, Deque[np.ndarray]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obs_batch: Dict[str, torch.Tensor] = {}
    for key, dq in obs_history.items():
        arr = np.stack(list(dq), axis=0)  # [T, ...]
        arr = np.expand_dims(arr, axis=0)  # [1, T, ...]
        obs_batch[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return obs_batch


def _extract_display_image(raw_obs: Dict[str, Any], source_map: Dict[str, str]) -> Optional[np.ndarray]:
    for key in source_map.values():
        if key in raw_obs and _is_image_like(raw_obs[key]):
            return _extract_latest_image(raw_obs[key])
    for key in FLAGS.image_obs_source_keys:
        if key in raw_obs and _is_image_like(raw_obs[key]):
            return _extract_latest_image(raw_obs[key])
    for key, value in raw_obs.items():
        if _is_image_like(value):
            return _extract_latest_image(value)
    return None


def _reset_env(env: Any) -> Dict[str, Any]:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env: Any, action: np.ndarray) -> Tuple[Dict[str, Any], bool]:
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, _, terminated, truncated, _ = out
            done = bool(terminated) or bool(truncated)
            return obs, done
        if len(out) == 4:
            obs, _, done, _ = out
            return obs, bool(done)
    raise RuntimeError("Unexpected env.step output format.")


def _create_widowx_env() -> Any:
    try:
        from envs.widowx_env import WidowXGym
        from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
    except ImportError as exc:
        raise ImportError(
            "Failed to import WidowX environment dependencies. "
            "Ensure `widowx_envs` is installed and Octo `examples` path is available."
        ) from exc

    start_state = None
    if FLAGS.initial_eep is not None:
        initial_eep = np.asarray([float(x) for x in FLAGS.initial_eep], dtype=np.float32)
        if initial_eep.shape[0] != 3:
            raise ValueError(f"--initial_eep must have 3 values, got {initial_eep}")
        start_state = np.concatenate([initial_eep, np.array([0, 0, 0, 1], dtype=np.float32)])

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["move_duration"] = FLAGS.step_duration
    if start_state is not None:
        env_params["start_state"] = list(start_state)

    client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    client.init(env_params, image_size=FLAGS.image_size)

    env = WidowXGym(
        client,
        FLAGS.image_size,
        FLAGS.blocking,
        FLAGS.sticky_gripper_num_steps,
    )
    return env


def _apply_step_overrides(policy: Any) -> None:
    if FLAGS.policy_n_timesteps > 0 and hasattr(policy, "n_timesteps"):
        policy.n_timesteps = int(FLAGS.policy_n_timesteps)
        logging.info("Override BFN n_timesteps=%d", policy.n_timesteps)

    if FLAGS.policy_num_inference_steps > 0 and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(FLAGS.policy_num_inference_steps)
        logging.info("Override diffusion num_inference_steps=%d", policy.num_inference_steps)
        if hasattr(policy, "noise_scheduler"):
            policy.noise_scheduler.set_timesteps(policy.num_inference_steps)


def _save_video(frames: List[np.ndarray], step_duration: float) -> None:
    if FLAGS.video_save_path is None or len(frames) == 0:
        return
    if imageio is None:
        raise ImportError("Saving videos requires `imageio`. Install with `pip install imageio`.")
    os.makedirs(FLAGS.video_save_path, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(FLAGS.video_save_path, f"{ts}.mp4")
    fps = max(1, int(round(1.0 / step_duration)))
    imageio.mimsave(save_path, frames, fps=fps)
    logging.info("Saved rollout video: %s", save_path)


def _run_episode(
    env: Any,
    policy: Any,
    obs_shape_meta: Dict[str, Any],
    n_obs_steps: int,
    device: torch.device,
) -> None:
    obs_history: Dict[str, Deque[np.ndarray]] = {
        k: deque(maxlen=n_obs_steps) for k in obs_shape_meta.keys()
    }

    obs = _reset_env(env)
    source_map = _build_obs_source_map(obs, obs_shape_meta)
    logging.info("Observation source map: %s", source_map)

    first_policy_obs = _convert_raw_obs_to_policy_obs(obs, obs_shape_meta, source_map)
    for key, value in first_policy_obs.items():
        for _ in range(n_obs_steps):
            obs_history[key].append(value.copy())

    if hasattr(policy, "reset"):
        try:
            policy.reset()
        except Exception:
            pass

    if FLAGS.interactive:
        cmd = input("Press [Enter] to start episode, or type 'q' to quit: ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            raise KeyboardInterrupt

    t = 0
    frames: List[np.ndarray] = []
    last_control_time = time.time()

    while t < FLAGS.num_timesteps:
        obs_batch = _history_to_torch_obs(obs_history, device=device)

        infer_start = time.time()
        with torch.no_grad():
            result = policy.predict_action(obs_batch)
        infer_ms = (time.time() - infer_start) * 1000.0

        if "action" not in result:
            raise KeyError("Policy predict_action result missing `action` key.")

        action_seq = result["action"][0].detach().cpu().numpy()
        if action_seq.ndim == 1:
            action_seq = action_seq[None, :]
        execute_steps = min(int(FLAGS.action_horizon), action_seq.shape[0])
        logging.info("t=%d infer=%.1fms execute=%d", t, infer_ms, execute_steps)

        done = False
        for k in range(execute_steps):
            if not FLAGS.blocking:
                now = time.time()
                sleep_dt = FLAGS.step_duration - (now - last_control_time)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
            last_control_time = time.time()

            action = np.asarray(action_seq[k], dtype=np.float32)
            if FLAGS.append_gripper and action.shape[0] == 6:
                action = np.concatenate([action, np.array([FLAGS.gripper_value], dtype=np.float32)])

            obs, done = _step_env(env, action)
            policy_obs = _convert_raw_obs_to_policy_obs(obs, obs_shape_meta, source_map)
            for key, value in policy_obs.items():
                obs_history[key].append(value.copy())

            frame = _extract_display_image(obs, source_map)
            if frame is not None:
                frames.append(frame)
                if FLAGS.show_image and cv2 is not None:
                    cv2.imshow("widowx_eval", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            t += 1
            if done or t >= FLAGS.num_timesteps:
                logging.info("Episode stopped (done=%s, t=%d).", done, t)
                break

        if done:
            break

    _save_video(frames, FLAGS.step_duration)


def main(_: Any) -> None:
    logging.set_verbosity(logging.INFO)
    if FLAGS.show_image and cv2 is None:
        raise ImportError("`--show_image` requires OpenCV (`opencv-python`).")

    ckpt_path = Path(FLAGS.checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(FLAGS.device)
    workspace, policy, cfg = _load_workspace_and_policy(
        checkpoint_path=ckpt_path,
        device=str(device),
        use_ema=FLAGS.use_ema,
    )
    del workspace  # not used after loading

    _apply_step_overrides(policy)

    shape_meta = _get_shape_meta(cfg)
    obs_shape_meta = shape_meta["obs"]
    n_obs_steps = int(getattr(policy, "n_obs_steps", cfg.get("n_obs_steps", 2)))
    logging.info("n_obs_steps=%d", n_obs_steps)
    logging.info("Policy obs keys=%s", list(obs_shape_meta.keys()))

    env = _create_widowx_env()
    episode_idx = 0

    try:
        while FLAGS.num_episodes <= 0 or episode_idx < FLAGS.num_episodes:
            logging.info("=== Episode %d ===", episode_idx)
            _run_episode(
                env=env,
                policy=policy,
                obs_shape_meta=obs_shape_meta,
                n_obs_steps=n_obs_steps,
                device=device,
            )
            episode_idx += 1
    finally:
        if FLAGS.show_image and cv2 is not None:
            cv2.destroyAllWindows()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    app.run(main)
