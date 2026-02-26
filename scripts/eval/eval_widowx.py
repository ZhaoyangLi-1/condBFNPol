#!/usr/bin/env python3
"""Evaluate condBFNPol-trained checkpoints on WidowX.

This script borrows the control flow of scripts/eval/example_widowx.py but
loads PyTorch workspace checkpoints trained in this repository.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root and local diffusion_policy to path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Add likely local WidowX package roots as well.
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "src" / "diffusion-policy",
    PROJECT_ROOT.parent / "bridge_data_robot_pusht",
    PROJECT_ROOT.parent / "bridge_data_robot_pusht" / "widowx_envs",
    PROJECT_ROOT.parent / "bridge_data_robot_pusht" / "widowx_envs" / "multicam_server" / "src",
    PROJECT_ROOT.parent / "bridge_data_robot_pusht" / "widowx_envs",
    PROJECT_ROOT.parent / "bridge_data_robot_pusht" / "widowx_envs" / "multicam_server" / "src",
    Path.home() / "bridge_data_robot_pusht",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs" / "multicam_server" / "src",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs" / "multicam_server" / "src",
]:
    sp = str(p)
    if p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)

from absl import app, flags, logging
import cv2
import dill
import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint",
    None,
    "Checkpoint file or run directory (can pass multiple for policy selection).",
    required=True,
)
flags.DEFINE_string("device", "cuda:0", "Torch device, e.g. cuda:0 or cpu")
flags.DEFINE_bool("use_ema", True, "Use ema_model when available")
flags.DEFINE_integer(
    "num_inference_steps",
    -1,
    "Override diffusion num_inference_steps when > 0",
)
flags.DEFINE_integer(
    "bfn_n_timesteps",
    -1,
    "Override BFN n_timesteps when > 0",
)

flags.DEFINE_integer("im_size", 128, "WidowX service image size")
flags.DEFINE_integer("num_timesteps", 120, "Number of control steps per rollout")
flags.DEFINE_integer("num_rollouts", 1, "Number of rollouts; <=0 means infinite")
flags.DEFINE_float("step_duration", 0.2, "Control period in seconds")
flags.DEFINE_integer("act_exec_horizon", 8, "How many planned actions to execute")

flags.DEFINE_bool("blocking", False, "Use blocking controller")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_string("ip", "localhost", "Robot IP")
flags.DEFINE_integer("port", 5556, "Robot port")

flags.DEFINE_bool("show_image", False, "Show camera image")
flags.DEFINE_string("video_save_path", None, "Directory to save rollout videos")
flags.DEFINE_bool("no_pitch_roll", False, "Zero out pitch/roll action dims")
flags.DEFINE_bool("no_yaw", False, "Zero out yaw action dim")

flags.DEFINE_integer(
    "sticky_gripper_num_steps",
    1,
    "Consecutive close/open predictions required to toggle gripper state",
)
flags.DEFINE_float(
    "gripper_close_threshold",
    0.5,
    "Action value below this threshold is treated as close",
)
flags.DEFINE_spaceseplist(
    "fixed_std",
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Per-dim Gaussian noise std added to action",
)

flags.DEFINE_string(
    "widowx_envs_path",
    "/home/liralab-widowx/bridge_data_robot/widowx_envs",
    "Path that contains widowx_envs and utils.py",
)

CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]


@dataclass
class LoadedPolicy:
    name: str
    checkpoint_path: str
    policy: Any
    shape_meta: Dict[str, Any]
    n_obs_steps: int
    n_action_steps: int


def _load_widowx_dependencies():
    widowx_envs_path = Path(FLAGS.widowx_envs_path).expanduser()
    if widowx_envs_path.is_dir() and str(widowx_envs_path) not in sys.path:
        sys.path.insert(0, str(widowx_envs_path))

    try:
        from widowx_envs.widowx_env_service import (  # type: ignore
            WidowXClient,
            WidowXConfigs,
            WidowXStatus,
        )
    except Exception as exc:
        raise ImportError(
            f"Failed to import widowx_envs from {widowx_envs_path}. "
            "Set --widowx_envs_path correctly."
        ) from exc

    state_to_eep = None
    utils_py = widowx_envs_path / "utils.py"
    if utils_py.is_file():
        spec = importlib.util.spec_from_file_location("widowx_eval_utils", str(utils_py))
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            state_to_eep = getattr(module, "state_to_eep", None)

    if state_to_eep is None:
        try:
            from utils import state_to_eep as imported_state_to_eep  # type: ignore

            state_to_eep = imported_state_to_eep
        except Exception as exc:
            raise ImportError(
                "Failed to import state_to_eep from widowx utils.py."
            ) from exc

    return WidowXClient, WidowXStatus, WidowXConfigs, state_to_eep


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _resolve_checkpoint_path(path_or_dir: str) -> str:
    p = Path(path_or_dir).expanduser()
    if p.is_file():
        return str(p)

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")

    latest_ckpt = p / "checkpoints" / "latest.ckpt"
    if latest_ckpt.is_file():
        return str(latest_ckpt)

    candidates: List[Path] = []
    ckpt_dir = p / "checkpoints"
    if ckpt_dir.is_dir():
        candidates.extend(ckpt_dir.glob("*.ckpt"))
    candidates.extend(p.glob("*.ckpt"))

    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt found in {path_or_dir} or {path_or_dir}/checkpoints"
        )

    candidates.sort(key=lambda x: x.stat().st_mtime)
    return str(candidates[-1])


def _load_policy_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[Any, Dict[str, Any], int, int]:
    payload = torch.load(
        open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if FLAGS.use_ema and hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model
    elif hasattr(workspace, "model"):
        policy = workspace.model
    else:
        raise AttributeError(
            "Workspace has neither 'model' nor available 'ema_model'."
        )

    policy = policy.to(device)
    policy.eval()

    if FLAGS.num_inference_steps > 0 and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(FLAGS.num_inference_steps)
    if FLAGS.bfn_n_timesteps > 0 and hasattr(policy, "n_timesteps"):
        policy.n_timesteps = int(FLAGS.bfn_n_timesteps)

    shape_meta_cfg = _cfg_get(cfg, "shape_meta", None)
    if shape_meta_cfg is None:
        task_cfg = _cfg_get(cfg, "task", None)
        if task_cfg is not None:
            shape_meta_cfg = _cfg_get(task_cfg, "shape_meta", None)
    if shape_meta_cfg is None:
        raise KeyError("Cannot find shape_meta in checkpoint cfg")

    if OmegaConf.is_config(shape_meta_cfg):
        shape_meta = OmegaConf.to_container(shape_meta_cfg, resolve=True)
    else:
        shape_meta = shape_meta_cfg
    n_obs_steps = int(getattr(policy, "n_obs_steps", _cfg_get(cfg, "n_obs_steps", 1)))
    n_action_steps = int(
        getattr(policy, "n_action_steps", _cfg_get(cfg, "n_action_steps", 1))
    )
    return policy, shape_meta, n_obs_steps, n_action_steps


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {arr.shape}")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] != 3:
        raise ValueError(f"Image channel mismatch, expected 3 channels, got {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _extract_widowx_rgb_obs(raw_obs: Dict[str, Any], im_size: int) -> np.ndarray:
    if "full_image" in raw_obs and raw_obs["full_image"] is not None:
        return _to_uint8_rgb(np.asarray(raw_obs["full_image"]))

    if "image" in raw_obs and raw_obs["image"] is not None:
        img = np.asarray(raw_obs["image"])
        if img.ndim == 1:
            side = int(im_size)
            if side <= 0 or img.size != 3 * side * side:
                inferred_side = int(round(np.sqrt(img.size / 3.0)))
                if inferred_side > 0 and 3 * inferred_side * inferred_side == img.size:
                    side = inferred_side
                else:
                    raise ValueError(
                        f"Cannot reshape flattened image of size {img.size} with im_size={im_size}"
                    )
            img = img.reshape(3, side, side).transpose(1, 2, 0)
            return _to_uint8_rgb(img)

        if img.ndim == 3:
            return _to_uint8_rgb(img)

    raise ValueError("WidowX observation does not contain a parsable image")


def _resize_and_format_rgb(base_rgb: np.ndarray, shape: Any) -> np.ndarray:
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"Unexpected rgb shape metadata: {shape}")
    c, h, w = [int(v) for v in shape]
    if c != 3:
        raise ValueError(f"Only 3-channel RGB observations are supported, got {shape}")

    img = base_rgb
    if img.shape[0] != h or img.shape[1] != w:
        interp = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_LINEAR
        img = cv2.resize(img, (w, h), interpolation=interp)

    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    return np.moveaxis(img, -1, 0)


def _extract_low_dim(
    raw_obs: Dict[str, Any], key: str, shape: Any, state_vec: np.ndarray
) -> np.ndarray:
    if not isinstance(shape, (list, tuple)):
        shape = [int(shape)]
    target_shape = tuple(int(v) for v in shape)
    target_dim = int(np.prod(target_shape))

    if key in raw_obs and raw_obs[key] is not None:
        arr = np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
    elif key == "proprio" and state_vec.size > 0:
        arr = state_vec
    elif key.endswith("eef_pos") and state_vec.size >= 3:
        arr = state_vec[:3]
    elif key.endswith("eef_quat") and state_vec.size >= 7:
        arr = state_vec[3:7]
    elif "gripper" in key and state_vec.size >= 2:
        arr = state_vec[-2:]
    elif key == "agent_pos" and state_vec.size >= 2:
        arr = state_vec[:2]
    elif "pose" in key and state_vec.size >= target_dim:
        arr = state_vec[:target_dim]
    elif state_vec.size > 0:
        arr = state_vec
    else:
        arr = np.zeros(target_dim, dtype=np.float32)

    if arr.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded

    return arr[:target_dim].reshape(target_shape)


def _build_policy_obs_frame(
    raw_obs: Dict[str, Any], obs_shape_meta: Dict[str, Any], im_size: int
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    base_rgb = _extract_widowx_rgb_obs(raw_obs, im_size)
    state_vec = np.asarray(raw_obs.get("state", []), dtype=np.float32).reshape(-1)

    obs_frame: Dict[str, np.ndarray] = {}
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        shape = attr.get("shape", None)
        if shape is None:
            raise KeyError(f"Missing shape metadata for obs key: {key}")

        if obs_type == "rgb":
            obs_frame[key] = _resize_and_format_rgb(base_rgb, shape)
        else:
            obs_frame[key] = _extract_low_dim(raw_obs, key, shape, state_vec)

    return obs_frame, base_rgb


def _stack_obs_history(obs_history: deque) -> Dict[str, np.ndarray]:
    if len(obs_history) == 0:
        raise ValueError("obs_history is empty")

    keys = list(obs_history[0].keys())
    stacked = {
        key: np.stack([obs[key] for obs in obs_history], axis=0).astype(np.float32)
        for key in keys
    }
    return stacked


def _predict_action_sequence(
    policy: Any, obs_np: Dict[str, np.ndarray], device: torch.device
) -> np.ndarray:
    obs_torch = {
        k: torch.from_numpy(v).unsqueeze(0).to(device=device, dtype=torch.float32)
        for k, v in obs_np.items()
    }

    with torch.no_grad():
        result = policy.predict_action(obs_torch)
        if "action" not in result:
            raise KeyError("policy.predict_action output missing 'action' key")
        actions = result["action"][0].detach().to("cpu").numpy()

    if actions.ndim == 1:
        actions = actions[None]

    return actions


def _save_rollout_video(images: List[np.ndarray], policy_name: str):
    if FLAGS.video_save_path is None or len(images) == 0:
        return

    os.makedirs(FLAGS.video_save_path, exist_ok=True)
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = policy_name.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(
        FLAGS.video_save_path,
        f"{curr_time}_{safe_name}_sticky_{FLAGS.sticky_gripper_num_steps}.mp4",
    )
    fps = max(1.0, (1.0 / FLAGS.step_duration) * 3.0)
    imageio.mimsave(save_path, images, fps=fps)
    print(f"Saved video to: {save_path}")


def main(_):
    np.set_printoptions(suppress=True)
    logging.set_verbosity(logging.WARNING)

    fixed_std = np.array([float(x) for x in FLAGS.fixed_std], dtype=np.float64)
    if fixed_std.size == 0:
        fixed_std = np.zeros(7, dtype=np.float64)
    if FLAGS.act_exec_horizon <= 0:
        raise ValueError("--act_exec_horizon must be >= 1")

    device = torch.device(FLAGS.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, fallback to CPU")
        device = torch.device("cpu")

    WidowXClient, WidowXStatus, WidowXConfigs, state_to_eep = _load_widowx_dependencies()

    policies: List[LoadedPolicy] = []
    for idx, ckpt_input in enumerate(FLAGS.checkpoint):
        resolved_ckpt = _resolve_checkpoint_path(ckpt_input)
        policy, shape_meta, n_obs_steps, n_action_steps = _load_policy_from_checkpoint(
            resolved_ckpt, device
        )

        ckpt_path = Path(resolved_ckpt)
        name = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        policies.append(
            LoadedPolicy(
                name=name,
                checkpoint_path=resolved_ckpt,
                policy=policy,
                shape_meta=shape_meta,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
            )
        )
        print(
            f"Loaded policy[{idx}]: {name}, n_obs_steps={n_obs_steps}, "
            f"n_action_steps={n_action_steps}"
        )

    if not policies:
        raise RuntimeError("No policy loaded")

    assert isinstance(FLAGS.initial_eep, list)
    initial_eep = [float(e) for e in FLAGS.initial_eep]
    start_state = np.concatenate([initial_eep, [0, 0, 0, 1]], axis=0)

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": CAMERA_TOPICS,
            "override_workspace_boundaries": WORKSPACE_BOUNDS,
            "move_duration": float(FLAGS.step_duration),
        }
    )
    env_params["state_state"] = list(start_state)

    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)

    rollout_idx = 0
    while FLAGS.num_rollouts <= 0 or rollout_idx < FLAGS.num_rollouts:
        if len(policies) == 1:
            policy_idx = 0
        else:
            print("Available policies:")
            for i, loaded in enumerate(policies):
                print(f"{i}) {loaded.name}")
            policy_idx = int(input("Select policy index: "))

        loaded = policies[policy_idx]
        policy = loaded.policy

        obs_shape_meta = loaded.shape_meta.get("obs", None)
        if obs_shape_meta is None:
            raise KeyError("shape_meta missing 'obs' field")

        if FLAGS.show_image:
            obs = widowx_client.get_observation()
            while obs is None:
                print("Waiting for observations...")
                time.sleep(1.0)
                obs = widowx_client.get_observation()
            preview = _extract_widowx_rgb_obs(obs, FLAGS.im_size)
            cv2.imshow("img_view", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)

        input("Press [Enter] to start rollout.")

        widowx_client.reset()
        time.sleep(2.5)

        if FLAGS.initial_eep is not None:
            initial_pose = state_to_eep(initial_eep, 0)
            widowx_client.move_gripper(1.0)  # open gripper
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(initial_pose, duration=1.5)

        if hasattr(policy, "reset"):
            try:
                policy.reset()
            except Exception:
                pass

        last_tstep = time.time()
        images: List[np.ndarray] = []
        obs_hist = deque(maxlen=max(1, loaded.n_obs_steps))

        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0

        t = 0
        try:
            while t < FLAGS.num_timesteps:
                if not FLAGS.blocking and time.time() <= last_tstep + FLAGS.step_duration:
                    time.sleep(0.001)
                    continue

                raw_obs = widowx_client.get_observation()
                if raw_obs is None:
                    print("WARNING retrying to get observation...")
                    continue

                obs_frame, vis_rgb = _build_policy_obs_frame(
                    raw_obs=raw_obs,
                    obs_shape_meta=obs_shape_meta,
                    im_size=FLAGS.im_size,
                )

                if FLAGS.show_image:
                    cv2.imshow("img_view", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if len(obs_hist) == 0:
                    obs_hist.extend([obs_frame] * max(1, loaded.n_obs_steps))
                else:
                    obs_hist.append(obs_frame)

                stacked_obs = _stack_obs_history(obs_hist)
                last_tstep = time.time()

                actions = _predict_action_sequence(
                    policy=policy,
                    obs_np=stacked_obs,
                    device=device,
                )

                exec_horizon = min(int(FLAGS.act_exec_horizon), actions.shape[0])
                for i in range(exec_horizon):
                    action = np.asarray(actions[i], dtype=np.float64).reshape(-1)
                    if action.size < 7:
                        raise ValueError(
                            f"Policy action dim {action.size} < 7, cannot execute on WidowX"
                        )
                    if action.size > 7:
                        action = action[:7]

                    if fixed_std.size == action.size:
                        action += np.random.normal(0.0, fixed_std)
                    elif fixed_std.size == 1:
                        action += np.random.normal(0.0, fixed_std.item(), size=action.shape)

                    # sticky gripper logic
                    if (action[-1] < FLAGS.gripper_close_threshold) != is_gripper_closed:
                        num_consecutive_gripper_change_actions += 1
                    else:
                        num_consecutive_gripper_change_actions = 0

                    if (
                        num_consecutive_gripper_change_actions
                        >= FLAGS.sticky_gripper_num_steps
                    ):
                        is_gripper_closed = not is_gripper_closed
                        num_consecutive_gripper_change_actions = 0

                    action[-1] = 0.0 if is_gripper_closed else 1.0

                    if FLAGS.no_pitch_roll:
                        action[3] = 0.0
                        action[4] = 0.0
                    if FLAGS.no_yaw:
                        action[5] = 0.0

                    widowx_client.step_action(action, blocking=FLAGS.blocking)

                    images.append(vis_rgb)
                    t += 1
                    if t >= FLAGS.num_timesteps:
                        break

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)

        _save_rollout_video(images, loaded.name)
        rollout_idx += 1


if __name__ == "__main__":
    app.run(main)
