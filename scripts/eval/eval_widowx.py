#!/usr/bin/env python3
"""Evaluate condBFNPol-trained checkpoints on WidowX.

This script borrows the control flow of scripts/eval/example_widowx.py but
loads PyTorch workspace checkpoints trained in this repository.

python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/diffusion_real_pusht.ckpt \
  --widowx_envs_path /scr2/zhaoyang/bridge_data_robot_pusht/widowx_envs \
  --action_mode 3trans3rot \
  --step_duration 0.1 \
  --act_exec_horizon 8 \
  --im_size 480 \
  --widowx_init_timeout_ms 180000 \
  --widowx_init_retries 8 \
  --video_save_path /data/BFN_data/diffusion_results \
  -- save_two_camera_videos
  

python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/bfn_real_pusht.ckpt \
  --widowx_envs_path /scr2/zhaoyang/bridge_data_robot_pusht/widowx_envs \
  --action_mode 2trans \
  --step_duration 0.1 \
  --act_exec_horizon 8 \
  --im_size 480 \
  --widowx_init_timeout_ms 180000 \
  --widowx_init_retries 8 \
  --video_save_path /data/BFN_data/bfn_results \
  save_two_camera_videos

"""

from __future__ import annotations

import importlib.util
import os
import re
import select
import socket
import sys
import termios
import time
import traceback
import tty
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

# flags.DEFINE_multi_string(
#     "checkpoint",
#     None,
#     "Checkpoint file or run directory (can pass multiple for policy selection).",
#     required=True,
# )
flags.DEFINE_multi_string(
    "checkpoint",
    "/data/BFN_data/checkpoints/diffusion_real_pusht.ckpt",
    "Checkpoint file or run directory (can pass multiple for policy selection).",
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

flags.DEFINE_integer("im_size", 480, "WidowX service image size")
flags.DEFINE_integer("num_rollouts", 1, "Number of rollouts; <=0 means infinite")
flags.DEFINE_float("step_duration", 0.1, "Control period in seconds")
flags.DEFINE_float(
    "max_duration",
    60.0,
    "Max duration for each rollout in seconds.",
)
flags.DEFINE_spaceseplist(
    "term_pose_xy",
    [],
    "Optional fixed termination XY. Leave empty to auto-align to post-reset robot pose.",
)
flags.DEFINE_float(
    "term_dist_thresh",
    0.03,
    "EEF XY distance threshold to consider entering termination area (meters).",
)
flags.DEFINE_float(
    "term_hold_sec",
    0.5,
    "Required dwell time in termination area before auto-termination (seconds).",
)
flags.DEFINE_float(
    "robot_exec_hz",
    10.0,
    "Robot command execution frequency in Hz after linear interpolation.",
)
flags.DEFINE_integer("act_exec_horizon", 8, "How many planned actions to execute")
flags.DEFINE_bool(
    "allow_unpaced_action_chunks",
    False,
    "If false, cap act_exec_horizon to 1 when blocking=False to avoid over-commanding.",
)

flags.DEFINE_bool("blocking", True, "Use blocking controller")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.02], "Initial position")
flags.DEFINE_bool(
    "run_initial_absolute_move",
    False,
    "After reset, call absolute move([x,y,z,roll,pitch,yaw]) to initial_eep. "
    "Disabled by default because [x,y,z,0,0,0] is often IK-infeasible on WidowX.",
)
flags.DEFINE_integer(
    "max_initial_move_retries",
    3,
    "Maximum retries for optional absolute initial move.",
)
flags.DEFINE_string("ip", "localhost", "Robot IP")
flags.DEFINE_integer("port", 5556, "Robot port")
flags.DEFINE_integer(
    "widowx_init_retries",
    5,
    "Number of init retries when WidowX returns a non-success status.",
)
flags.DEFINE_integer(
    "widowx_init_timeout_ms",
    120000,
    "Req/rep timeout used during WidowX init RPC (milliseconds).",
)
flags.DEFINE_integer(
    "widowx_rpc_timeout_ms",
    10000,
    "Req/rep timeout used after init for regular RPCs (milliseconds).",
)
flags.DEFINE_float(
    "widowx_init_retry_sleep",
    1.0,
    "Seconds to sleep between init retries.",
)
flags.DEFINE_integer(
    "widowx_reset_retries",
    3,
    "Number of retries for reset() when WidowX returns non-success status.",
)
flags.DEFINE_integer(
    "widowx_reset_timeout_ms",
    30000,
    "Req/rep timeout used during reset RPC (milliseconds).",
)
flags.DEFINE_float(
    "widowx_reset_retry_sleep",
    1.0,
    "Seconds to sleep between reset retries.",
)
flags.DEFINE_enum(
    "action_mode",
    "2trans",
    ["2trans", "3trans", "3trans1rot", "3trans3rot"],
    "WidowX env action_mode for step_action payload projection.",
)
flags.DEFINE_bool("lock_z", True, "Enable Z locking in env params")
flags.DEFINE_float("fixed_z_height", 0.02, "Fixed z height used when lock_z is True")
flags.DEFINE_float("neutral_z_height", 0.02, "Neutral z height")
flags.DEFINE_float("fixed_gripper", 0.0, "Fixed gripper command for 2trans env mode")
flags.DEFINE_spaceseplist(
    "camera_topics",
    ["/D435/color/image_raw", "/blue/image_raw"],
    "ROS RGB image topics passed to WidowX env init.",
)
flags.DEFINE_bool(
    "skip_move_to_neutral",
    False,
    "Pass through skip_move_to_neutral to WidowX env init.",
)

flags.DEFINE_bool("show_image", False, "Show camera image")
flags.DEFINE_string("video_save_path", None, "Directory to save rollout videos")
flags.DEFINE_bool(
    "save_two_camera_videos",
    False,
    "Also save separate cam0/cam1 videos when video_save_path is set.",
)
flags.DEFINE_bool("no_pitch_roll", False, "Zero out pitch/roll action dims")
flags.DEFINE_bool("no_yaw", False, "Zero out yaw action dim")
flags.DEFINE_float(
    "safety_max_xy_delta",
    1,
    "Safety clip for |dx|,|dy| in 2trans mode (meters). Set <=0 to disable.",
)
flags.DEFINE_float(
    "safety_max_xyz_delta",
    1,
    "Safety clip for |dx|,|dy|,|dz| in non-2trans modes (meters). Set <=0 to disable.",
)
flags.DEFINE_float(
    "safety_max_rot_delta",
    1,
    "Safety clip for |droll|,|dpitch|,|dyaw| in non-2trans modes (radians). Set <=0 to disable.",
)

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

WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]


@dataclass
class LoadedPolicy:
    name: str
    checkpoint_path: str
    policy: Any
    shape_meta: Dict[str, Any]
    n_obs_steps: int
    n_action_steps: int
    action_input_std: np.ndarray | None = None


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

    return WidowXClient, WidowXStatus, WidowXConfigs


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _status_name(status: Any, WidowXStatus: Any) -> str:
    for name in ("SUCCESS", "NO_CONNECTION", "EXECUTION_FAILURE", "NOT_INITIALIZED"):
        if hasattr(WidowXStatus, name) and status == getattr(WidowXStatus, name):
            return name
    return str(status)


def _set_widowx_reqrep_timeout_ms(widowx_client: Any, timeout_ms: int) -> None:
    """Best-effort update of underlying req/rep timeout used by widowx_envs."""
    try:
        action_client = getattr(widowx_client, "_WidowXClient__client", None)
        if action_client is None:
            return
        reqrep_client = getattr(action_client, "client", None)
        if reqrep_client is None:
            return
        reqrep_client.timeout_ms = int(timeout_ms)
        reqrep_client.reset_socket()
    except Exception:
        # Keep compatibility across SDK variants that may hide these internals.
        pass


def _tcp_port_reachable(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.05, float(timeout_s))):
            return True
    except Exception:
        return False


def _init_widowx_with_retry(
    widowx_client: Any,
    env_params: Dict[str, Any],
    image_size: int,
    WidowXStatus: Any,
) -> Any:
    retries = max(1, int(FLAGS.widowx_init_retries))
    retry_sleep = max(0.0, float(FLAGS.widowx_init_retry_sleep))
    init_timeout_ms = max(1, int(FLAGS.widowx_init_timeout_ms))
    rpc_timeout_ms = max(1, int(FLAGS.widowx_rpc_timeout_ms))

    _set_widowx_reqrep_timeout_ms(widowx_client, init_timeout_ms)
    last_status = None

    for attempt in range(1, retries + 1):
        print(
            f"[INFO] WidowX init attempt {attempt}/{retries} "
            f"(timeout={init_timeout_ms} ms, server={FLAGS.ip}:{FLAGS.port})"
        )
        t0 = time.time()
        last_status = widowx_client.init(env_params, image_size=image_size)
        elapsed = time.time() - t0
        if last_status == WidowXStatus.SUCCESS:
            _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
            return last_status

        print(
            f"[WARN] WidowX init attempt {attempt}/{retries} failed with "
            f"status={_status_name(last_status, WidowXStatus)} "
            f"after {elapsed:.2f}s."
        )
        no_connection = getattr(WidowXStatus, "NO_CONNECTION", None)
        if last_status == no_connection:
            print(
                "[HINT] No response from WidowX action server. "
                "Make sure `widowx_env_service --server` is running "
                f"and reachable at {FLAGS.ip}:{FLAGS.port}."
            )
        if attempt < retries and retry_sleep > 0.0:
            time.sleep(retry_sleep)

    _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
    return last_status


def _reset_widowx_with_retry(
    widowx_client: Any, WidowXStatus: Any, i_traj: int | None = None
) -> Any:
    retries = max(1, int(FLAGS.widowx_reset_retries))
    retry_sleep = max(0.0, float(FLAGS.widowx_reset_retry_sleep))
    reset_timeout_ms = max(1, int(FLAGS.widowx_reset_timeout_ms))
    rpc_timeout_ms = max(1, int(FLAGS.widowx_rpc_timeout_ms))

    _set_widowx_reqrep_timeout_ms(widowx_client, max(reset_timeout_ms, rpc_timeout_ms))
    last_status = None
    warned_itraj_fallback = False
    for attempt in range(1, retries + 1):
        if i_traj is None:
            last_status = widowx_client.reset()
        else:
            try:
                last_status = widowx_client.reset(itraj=int(i_traj))
            except TypeError:
                if not warned_itraj_fallback:
                    print(
                        "[WARN] widowx_client.reset(itraj=...) is not supported by "
                        "this widowx_envs version; falling back to reset()."
                    )
                    warned_itraj_fallback = True
                last_status = widowx_client.reset()
        if last_status == WidowXStatus.SUCCESS:
            break
        print(
            f"[WARN] WidowX reset attempt {attempt}/{retries} failed with "
            f"status={_status_name(last_status, WidowXStatus)}."
        )
        if attempt < retries and retry_sleep > 0.0:
            time.sleep(retry_sleep)

    _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
    return last_status


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


def _extract_action_input_std_from_state_dict(
    state_dict: Dict[str, Any],
) -> np.ndarray | None:
    std = state_dict.get("normalizer.params_dict.action.input_stats.std", None)
    if isinstance(std, torch.Tensor):
        return std.detach().to("cpu").numpy().astype(np.float64)

    normalizer = state_dict.get("normalizer", None)
    if isinstance(normalizer, dict):
        nested_std = normalizer.get("params_dict.action.input_stats.std", None)
        if isinstance(nested_std, torch.Tensor):
            return nested_std.detach().to("cpu").numpy().astype(np.float64)

    return None


def _load_policy_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[Any, Dict[str, Any], int, int, np.ndarray | None]:
    payload = torch.load(
        open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload.get("state_dicts", {})
    load_candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    if FLAGS.use_ema and isinstance(state_dicts.get("ema_model"), dict):
        load_candidates.append(("ema_model", state_dicts["ema_model"]))
    if isinstance(state_dicts.get("model"), dict):
        load_candidates.append(("model", state_dicts["model"]))

    if not load_candidates:
        raise KeyError(
            f"Checkpoint has no model/ema_model state_dicts: {checkpoint_path}"
        )

    loaded_source = None
    action_input_std = None
    last_load_error: Exception | None = None
    for source, state in load_candidates:
        try:
            action_input_std = _extract_action_input_std_from_state_dict(state)
            state = {
                k.replace("_orig_mod.", ""): v
                for k, v in state.items()
            }
            try:
                policy.load_state_dict(state, strict=False)
            except TypeError:
                # Some custom classes only accept (state_dict)
                policy.load_state_dict(state)
            loaded_source = source
            break
        except Exception as exc:
            last_load_error = exc

    if loaded_source is None:
        raise RuntimeError(
            "Failed to load checkpoint state_dict into policy model."
        ) from last_load_error

    policy = policy.to(device)
    policy.eval()
    print(f"[INFO] Loaded weights from state_dicts['{loaded_source}']")

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
    return policy, shape_meta, n_obs_steps, n_action_steps, action_input_std


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


def _reshape_flattened_image(flat: np.ndarray, im_size: int) -> np.ndarray:
    side = int(im_size)
    if side <= 0 or flat.size != 3 * side * side:
        inferred_side = int(round(np.sqrt(flat.size / 3.0)))
        if inferred_side > 0 and 3 * inferred_side * inferred_side == flat.size:
            side = inferred_side
        else:
            raise ValueError(
                f"Cannot reshape flattened image of size {flat.size} with im_size={im_size}"
            )
    img = flat.reshape(3, side, side).transpose(1, 2, 0)
    return _to_uint8_rgb(img)


def _extract_widowx_rgb_sources(raw_obs: Dict[str, Any], im_size: int) -> Dict[str, np.ndarray]:
    sources: Dict[str, np.ndarray] = {}

    # Fast image transfer keys used by widowx_env_service.
    for key in ("external_img", "over_shoulder_img", "wrist_img"):
        if key in raw_obs and raw_obs[key] is not None:
            sources[key] = _to_uint8_rgb(np.asarray(raw_obs[key]))

    # Optional full_image from other wrappers: shape usually (N, H, W, C).
    if "full_image" in raw_obs and raw_obs["full_image"] is not None:
        full_image = np.asarray(raw_obs["full_image"])
        if full_image.ndim == 4:
            for i in range(full_image.shape[0]):
                sources[f"full_image_{i}"] = _to_uint8_rgb(full_image[i])
        elif full_image.ndim == 3:
            sources["full_image_0"] = _to_uint8_rgb(full_image)

    # "image" may be flattened CHW float image(s) from BridgeDataRailRLPrivateWidowX.
    if "image" in raw_obs and raw_obs["image"] is not None:
        image_arr = np.asarray(raw_obs["image"])
        if image_arr.ndim == 1:
            sources.setdefault("image_0", _reshape_flattened_image(image_arr, im_size))
        elif image_arr.ndim == 2:
            for i in range(image_arr.shape[0]):
                try:
                    sources.setdefault(f"image_{i}", _reshape_flattened_image(image_arr[i], im_size))
                except Exception:
                    continue
        elif image_arr.ndim == 3:
            # Could be single HWC/CHW or a stack of CHW.
            if image_arr.shape[-1] == 3:
                sources.setdefault("image_0", _to_uint8_rgb(image_arr))
            elif image_arr.shape[0] == 3:
                sources.setdefault("image_0", _to_uint8_rgb(image_arr))
            else:
                for i in range(image_arr.shape[0]):
                    try:
                        sources.setdefault(f"image_{i}", _to_uint8_rgb(image_arr[i]))
                    except Exception:
                        continue
        elif image_arr.ndim == 4:
            for i in range(image_arr.shape[0]):
                try:
                    sources.setdefault(f"image_{i}", _to_uint8_rgb(image_arr[i]))
                except Exception:
                    continue

    if not sources:
        raise ValueError(
            "WidowX observation does not contain image keys among "
            "{external_img, over_shoulder_img, wrist_img, full_image, image}"
        )
    return sources


def _rgb_key_to_camera_index(obs_key: str) -> int | None:
    match = re.search(r"(\d+)$", obs_key)
    if match:
        return int(match.group(1))
    return None


def _select_rgb_source_for_obs_key(obs_key: str, rgb_sources: Dict[str, np.ndarray]) -> np.ndarray:
    key_lower = obs_key.lower()
    # semantic fallback first
    if "wrist" in key_lower and "wrist_img" in rgb_sources:
        return rgb_sources["wrist_img"]
    if "shoulder" in key_lower and "over_shoulder_img" in rgb_sources:
        return rgb_sources["over_shoulder_img"]

    cam_idx = _rgb_key_to_camera_index(obs_key)
    preferred_order: List[str] = []
    if cam_idx == 0:
        preferred_order = ["external_img", "full_image_0", "image_0", "over_shoulder_img", "wrist_img"]
    elif cam_idx == 1:
        preferred_order = ["over_shoulder_img", "full_image_1", "image_1", "external_img", "wrist_img"]
    elif cam_idx == 2:
        preferred_order = ["wrist_img", "full_image_2", "image_2", "external_img", "over_shoulder_img"]

    for source_key in preferred_order:
        if source_key in rgb_sources:
            return rgb_sources[source_key]

    # Last fallback: first available source (stable order by key).
    return rgb_sources[sorted(rgb_sources.keys())[0]]


def _extract_widowx_rgb_obs(raw_obs: Dict[str, Any], im_size: int) -> np.ndarray:
    rgb_sources = _extract_widowx_rgb_sources(raw_obs, im_size)
    if "external_img" in rgb_sources:
        return rgb_sources["external_img"]
    if "over_shoulder_img" in rgb_sources:
        return rgb_sources["over_shoulder_img"]
    return rgb_sources[sorted(rgb_sources.keys())[0]]


def _build_camera_topics_from_flags() -> List[Dict[str, str]]:
    topics: List[Dict[str, str]] = []
    for topic in FLAGS.camera_topics:
        # Support both:
        #   --camera_topics=/a,/b
        #   --camera_topics="/a /b"
        #   --camera_topics=/a --camera_topics=/b
        pieces = re.split(r"[,\s]+", str(topic).strip())
        for piece in pieces:
            topic_str = piece.strip()
            if topic_str:
                topics.append({"name": topic_str})
    if not topics:
        raise ValueError("--camera_topics must include at least one non-empty topic.")
    return topics


def _extract_two_camera_rgb(raw_obs: Dict[str, Any], im_size: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    rgb_sources = _extract_widowx_rgb_sources(raw_obs, im_size)

    cam0 = None
    for key in ("external_img", "full_image_0", "image_0", "over_shoulder_img", "wrist_img"):
        if key in rgb_sources:
            cam0 = rgb_sources[key]
            break
    if cam0 is None and rgb_sources:
        cam0 = rgb_sources[sorted(rgb_sources.keys())[0]]

    cam1 = None
    for key in ("over_shoulder_img", "full_image_1", "image_1", "external_img", "wrist_img"):
        if key in rgb_sources and rgb_sources[key] is not cam0:
            cam1 = rgb_sources[key]
            break
    if cam1 is None and rgb_sources:
        for key in sorted(rgb_sources.keys()):
            candidate = rgb_sources[key]
            if candidate is not cam0:
                cam1 = candidate
                break

    return cam0, cam1


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
    rgb_sources = _extract_widowx_rgb_sources(raw_obs, im_size)
    vis_rgb = _extract_widowx_rgb_obs(raw_obs, im_size)
    state_vec = np.asarray(raw_obs.get("state", []), dtype=np.float32).reshape(-1)

    obs_frame: Dict[str, np.ndarray] = {}
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        shape = attr.get("shape", None)
        if shape is None:
            raise KeyError(f"Missing shape metadata for obs key: {key}")

        if obs_type == "rgb":
            key_rgb = _select_rgb_source_for_obs_key(key, rgb_sources)
            obs_frame[key] = _resize_and_format_rgb(key_rgb, shape)
        else:
            obs_frame[key] = _extract_low_dim(raw_obs, key, shape, state_vec)

    return obs_frame, vis_rgb


def _project_action_to_env_mode(action_7d: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "2trans":
        return action_7d[:2]
    if action_mode == "3trans":
        return np.array([action_7d[0], action_7d[1], action_7d[2], action_7d[6]], dtype=np.float64)
    if action_mode == "3trans1rot":
        return np.array(
            [action_7d[0], action_7d[1], action_7d[2], action_7d[5], action_7d[6]],
            dtype=np.float64,
        )
    if action_mode == "3trans3rot":
        return action_7d
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def _safety_clip_action(action_7d: np.ndarray, action_mode: str) -> np.ndarray:
    action = np.asarray(action_7d, dtype=np.float64).copy()

    if action_mode == "2trans":
        if FLAGS.safety_max_xy_delta > 0:
            lim = float(FLAGS.safety_max_xy_delta)
            action[:2] = np.clip(action[:2], -lim, lim)
        # 2trans should only use planar translation deltas.
        action[2:6] = 0.0
        return action

    if FLAGS.safety_max_xyz_delta > 0:
        lim_xyz = float(FLAGS.safety_max_xyz_delta)
        action[:3] = np.clip(action[:3], -lim_xyz, lim_xyz)
    if FLAGS.safety_max_rot_delta > 0:
        lim_rot = float(FLAGS.safety_max_rot_delta)
        action[3:6] = np.clip(action[3:6], -lim_rot, lim_rot)
    return action


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


def _enter_terminal_cbreak_mode() -> Tuple[int | None, Any]:
    if not sys.stdin.isatty():
        return None, None
    try:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return fd, old_attrs
    except Exception:
        return None, None


def _restore_terminal_mode(fd: int | None, old_attrs: Any) -> None:
    if fd is None or old_attrs is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    except Exception:
        pass


def _poll_stop_key(show_image: bool, stdin_fd: int | None) -> bool:
    if show_image:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")):
            return True

    if stdin_fd is not None:
        try:
            ready, _, _ = select.select([stdin_fd], [], [], 0.0)
        except Exception:
            ready = []
        if ready:
            try:
                ch = os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
            except Exception:
                ch = ""
            if ch.lower() == "s":
                return True
    return False


def _extract_eef_xy(raw_obs: Dict[str, Any]) -> np.ndarray | None:
    for key in ("eef_pos", "ee_pos", "agent_pos", "state", "proprio"):
        if key not in raw_obs or raw_obs[key] is None:
            continue
        arr = np.asarray(raw_obs[key], dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
    return None


def _get_current_eef_xy_with_retry(
    widowx_client: Any,
    max_wait_sec: float = 2.0,
    poll_interval_sec: float = 0.05,
) -> np.ndarray | None:
    deadline = time.monotonic() + max(0.1, float(max_wait_sec))
    poll_interval_sec = max(0.01, float(poll_interval_sec))
    while time.monotonic() < deadline:
        raw_obs = widowx_client.get_observation()
        if raw_obs is not None:
            eef_xy = _extract_eef_xy(raw_obs)
            if eef_xy is not None:
                return np.asarray(eef_xy, dtype=np.float64).reshape(-1)[:2]
        time.sleep(poll_interval_sec)
    return None


def _interpolate_actions_to_robot_rate(
    prev_action: np.ndarray | None,
    next_action: np.ndarray,
    num_substeps: int,
) -> List[np.ndarray]:
    num_substeps = max(1, int(num_substeps))
    next_action = np.asarray(next_action, dtype=np.float64).reshape(-1)

    if prev_action is None or num_substeps == 1:
        return [next_action.copy() for _ in range(num_substeps)]

    prev_action = np.asarray(prev_action, dtype=np.float64).reshape(-1)
    interpolated: List[np.ndarray] = []
    for k in range(1, num_substeps + 1):
        alpha = float(k) / float(num_substeps)
        sub_action = (1.0 - alpha) * prev_action + alpha * next_action
        # Keep the gripper command discrete/sticky at the current policy step value.
        if sub_action.size > 0 and next_action.size > 0:
            sub_action[-1] = next_action[-1]
        interpolated.append(sub_action)
    return interpolated


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
    if FLAGS.step_duration <= 0:
        raise ValueError("--step_duration must be > 0")
    if FLAGS.robot_exec_hz <= 0:
        raise ValueError("--robot_exec_hz must be > 0")

    policy_hz = 1.0 / float(FLAGS.step_duration)
    interp_substeps = max(1, int(round(float(FLAGS.robot_exec_hz) / policy_hz)))
    robot_exec_hz = policy_hz * float(interp_substeps)
    robot_step_duration = float(FLAGS.step_duration) / float(interp_substeps)
    requested_exec_hz = float(FLAGS.robot_exec_hz)
    if abs(robot_exec_hz - requested_exec_hz) > 1e-6:
        print(
            "[WARN] Requested --robot_exec_hz is not an integer multiple of policy Hz "
            f"({policy_hz:.6f}). Using nearest feasible robot Hz={robot_exec_hz:.6f} "
            f"with {interp_substeps} substeps."
        )
    print(
        "[INFO] Control rate setup: "
        f"policy_hz={policy_hz:.6f}, robot_exec_hz={robot_exec_hz:.6f}, "
        f"interp_substeps={interp_substeps}, robot_step_duration={robot_step_duration:.6f}s"
    )
    term_pose_xy_override: np.ndarray | None = None
    if len(FLAGS.term_pose_xy) > 0:
        term_pose_xy_override = np.array([float(v) for v in FLAGS.term_pose_xy], dtype=np.float64).reshape(-1)
        if term_pose_xy_override.size < 2:
            raise ValueError("--term_pose_xy must contain at least two numbers: x y")
        term_pose_xy_override = term_pose_xy_override[:2]
    if FLAGS.term_dist_thresh <= 0:
        raise ValueError("--term_dist_thresh must be > 0")
    if FLAGS.term_hold_sec <= 0:
        raise ValueError("--term_hold_sec must be > 0")
    if FLAGS.max_duration <= 0:
        raise ValueError("--max_duration must be > 0")

    device = torch.device(FLAGS.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, fallback to CPU")
        device = torch.device("cpu")

    WidowXClient, WidowXStatus, WidowXConfigs = _load_widowx_dependencies()

    policies: List[LoadedPolicy] = []
    for idx, ckpt_input in enumerate(FLAGS.checkpoint):
        resolved_ckpt = _resolve_checkpoint_path(ckpt_input)
        policy, shape_meta, n_obs_steps, n_action_steps, action_input_std = _load_policy_from_checkpoint(
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
                action_input_std=action_input_std,
            )
        )
        print(
            f"Loaded policy[{idx}]: {name}, n_obs_steps={n_obs_steps}, "
            f"n_action_steps={n_action_steps}"
        )
        if isinstance(action_input_std, np.ndarray):
            print(
                "[INFO] Action input std from checkpoint: "
                f"{np.array2string(action_input_std, precision=6, suppress_small=True)}"
            )

    if not policies:
        raise RuntimeError("No policy loaded")

    assert isinstance(FLAGS.initial_eep, list)
    initial_eep = [float(e) for e in FLAGS.initial_eep]
    camera_topics = _build_camera_topics_from_flags()
    print(f"[INFO] Using camera topics: {[cam['name'] for cam in camera_topics]}")

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": camera_topics,
            "override_workspace_boundaries": WORKSPACE_BOUNDS,
            "move_duration": robot_step_duration,
            "action_mode": FLAGS.action_mode,
            "skip_move_to_neutral": bool(FLAGS.skip_move_to_neutral),
            "move_to_rand_start_freq": -1,
            "fix_zangle": 0.1,
            "adaptive_wait": True,
            "fixed_z_height": float(FLAGS.fixed_z_height),
            "neutral_z_height": float(FLAGS.neutral_z_height),
            "z_lock_feedback_gain": 0.2,
            "z_lock_max_delta": 0.0015,
            "z_lock_deadband": 0.002,
            "xy_action_deadband": 0.0015,
            "vr_vertical_reject_ratio": 0.6,
            "vr_xy_step_deadband": 0.0015,
            "vr_xy_step_clip": 0.008,
            "vr_xy_scale": 0.9,
            "fixed_gripper": float(FLAGS.fixed_gripper),
            "lock_z": bool(FLAGS.lock_z),
            "action_clipping": None,
        }
    )
    if not _tcp_port_reachable(FLAGS.ip, FLAGS.port, timeout_s=1.0):
        print(
            "[WARN] Preflight TCP check failed for WidowX server at "
            f"{FLAGS.ip}:{FLAGS.port}. "
            "Init may timeout unless the action server is started."
        )
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    init_status = _init_widowx_with_retry(
        widowx_client=widowx_client,
        env_params=env_params,
        image_size=FLAGS.im_size,
        WidowXStatus=WidowXStatus,
    )
    if init_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            "WidowX init failed after "
            f"{max(1, int(FLAGS.widowx_init_retries))} attempts with "
            f"status={_status_name(init_status, WidowXStatus)}. "
            f"Check server reachability at {FLAGS.ip}:{FLAGS.port}."
        )
    if FLAGS.initial_eep is not None and not FLAGS.run_initial_absolute_move:
        print(
            "[INFO] Using collect-style reset path (reset with trajectory index). "
            "Optional absolute move is disabled."
        )
    
    rollout_idx = 0
    stop_requested = False
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
        if isinstance(loaded.action_input_std, np.ndarray) and loaded.action_input_std.size >= 7:
            # Many Bridge Push-T checkpoints are effectively XY-only despite action shape [7].
            if np.all(np.abs(loaded.action_input_std[2:]) < 1e-12) and FLAGS.action_mode != "2trans":
                print(
                    "[WARN] Checkpoint action stats indicate only XY deltas were trained "
                    f"(std={np.array2string(loaded.action_input_std, precision=6, suppress_small=True)}). "
                    "Consider --action_mode=2trans for closer train/eval matching."
                )

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

        reset_status = _reset_widowx_with_retry(
            widowx_client=widowx_client,
            WidowXStatus=WidowXStatus,
            i_traj=rollout_idx,
        )
        if reset_status != WidowXStatus.SUCCESS:
            print(
                "[ERROR] Reset failed with status="
                f"{_status_name(reset_status, WidowXStatus)}, skip rollout."
            )
            rollout_idx += 1
            continue

        if FLAGS.initial_eep is not None and FLAGS.run_initial_absolute_move:
            widowx_client.move_gripper(1.0)  # open gripper
            initial_pose = np.array(
                [initial_eep[0], initial_eep[1], initial_eep[2], 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            move_status = None
            retries = max(1, int(FLAGS.max_initial_move_retries))
            for attempt in range(1, retries + 1):
                move_status = widowx_client.move(initial_pose, duration=1.5)
                if move_status == WidowXStatus.SUCCESS:
                    break
                print(
                    f"[WARN] initial absolute move failed "
                    f"(attempt {attempt}/{retries}, "
                    f"status={_status_name(move_status, WidowXStatus)})."
                )
            if move_status != WidowXStatus.SUCCESS:
                print(
                    "[ERROR] initial absolute move failed; "
                    "set --run_initial_absolute_move=false (recommended) or adjust pose."
                )
                rollout_idx += 1
                continue
        fallback_term_pose_xy = np.array(initial_eep[:2], dtype=np.float64)
        if term_pose_xy_override is not None:
            rollout_term_pose_xy = term_pose_xy_override.copy()
            print(
                "[INFO] Using fixed termination XY from --term_pose_xy: "
                f"{np.array2string(rollout_term_pose_xy, precision=6, suppress_small=True)}"
            )
        else:
            rollout_term_pose_xy = _get_current_eef_xy_with_retry(
                widowx_client=widowx_client,
                max_wait_sec=2.0,
                poll_interval_sec=0.05,
            )
            if rollout_term_pose_xy is None:
                rollout_term_pose_xy = fallback_term_pose_xy
                print(
                    "[WARN] Failed to read post-reset EEF XY; "
                    f"fallback termination XY={rollout_term_pose_xy.tolist()} from --initial_eep."
                )
            else:
                print(
                    "[INFO] Auto-aligned termination XY to post-reset robot pose: "
                    f"{np.array2string(rollout_term_pose_xy, precision=6, suppress_small=True)}"
                )

        if hasattr(policy, "reset"):
            try:
                policy.reset()
            except Exception:
                pass

        last_tstep = time.time()
        last_exec_tstep = last_tstep - robot_step_duration
        images: List[np.ndarray] = []
        cam0_images: List[np.ndarray] = []
        cam1_images: List[np.ndarray] = []
        obs_hist = deque(maxlen=max(1, loaded.n_obs_steps))

        prev_policy_action: np.ndarray | None = None
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        warned_unpaced_horizon = False

        t = 0
        stop_this_rollout = False
        stop_reason: str | None = None
        rollout_t_start = time.monotonic()
        term_area_start_timestamp = float("inf")
        warned_missing_eef_xy = False
        stdin_fd, stdin_old_attrs = _enter_terminal_cbreak_mode()
        print(
            "[INFO] Stop modes: press 's', timeout by --max_duration, "
            "or dwell in termination area."
        )
        try:
            while True:
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
                if FLAGS.save_two_camera_videos:
                    cam0_rgb, cam1_rgb = _extract_two_camera_rgb(raw_obs, FLAGS.im_size)
                    if cam0_rgb is not None:
                        cam0_images.append(cam0_rgb)
                    if cam1_rgb is not None:
                        cam1_images.append(cam1_rgb)

                if FLAGS.show_image:
                    cv2.imshow("img_view", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
                if _poll_stop_key(FLAGS.show_image, stdin_fd):
                    print("[INFO] Stop requested by user; ending rollout early.")
                    stop_this_rollout = True
                    stop_requested = True
                    stop_reason = "manual"
                    break
                elapsed = time.monotonic() - rollout_t_start
                if elapsed > float(FLAGS.max_duration):
                    print("[INFO] Terminated by timeout.")
                    stop_this_rollout = True
                    stop_reason = "timeout"
                    break

                eef_xy = _extract_eef_xy(raw_obs)
                if eef_xy is None:
                    if not warned_missing_eef_xy:
                        print("[WARN] Could not extract EEF XY from observation; disabling term-area stop.")
                        warned_missing_eef_xy = True
                    term_area_start_timestamp = float("inf")
                else:
                    dist = np.linalg.norm(eef_xy - rollout_term_pose_xy, axis=-1)
                    now = time.monotonic()
                    if dist < float(FLAGS.term_dist_thresh):
                        if term_area_start_timestamp > now:
                            term_area_start_timestamp = now
                        elif (now - term_area_start_timestamp) > float(FLAGS.term_hold_sec):
                            print("[INFO] Terminated by reaching termination area.")
                            stop_this_rollout = True
                            stop_reason = "term_area"
                            break
                    else:
                        term_area_start_timestamp = float("inf")

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
                if (
                    not FLAGS.blocking
                    and not FLAGS.allow_unpaced_action_chunks
                    and exec_horizon > 1
                ):
                    if not warned_unpaced_horizon:
                        print(
                            "[WARN] blocking=False with act_exec_horizon>1 can over-command the robot; "
                            "capping executed horizon to 1. "
                            "Set --allow_unpaced_action_chunks=true to override."
                        )
                        warned_unpaced_horizon = True
                    exec_horizon = 1
                for i in range(exec_horizon):
                    raw_action = np.asarray(actions[i], dtype=np.float64).reshape(-1)
                    action = np.zeros((7,), dtype=np.float64)
                    copy_dim = min(7, raw_action.size)
                    action[:copy_dim] = raw_action[:copy_dim]
                    if raw_action.size == 2 and FLAGS.action_mode == "2trans":
                        action[6] = float(FLAGS.fixed_gripper)

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

                    action = _safety_clip_action(action, FLAGS.action_mode)
                    sub_actions = _interpolate_actions_to_robot_rate(
                        prev_action=prev_policy_action,
                        next_action=action,
                        num_substeps=interp_substeps,
                    )
                    for sub_action in sub_actions:
                        if not FLAGS.blocking:
                            wait_s = (last_exec_tstep + robot_step_duration) - time.time()
                            if wait_s > 0.0:
                                time.sleep(wait_s)
                        sub_action = _safety_clip_action(sub_action, FLAGS.action_mode)
                        env_action = _project_action_to_env_mode(sub_action, FLAGS.action_mode)
                        step_status = widowx_client.step_action(
                            env_action, blocking=FLAGS.blocking
                        )
                        if step_status != WidowXStatus.SUCCESS:
                            raise RuntimeError(
                                "WidowX step_action failed: status="
                                f"{_status_name(step_status, WidowXStatus)}, "
                                f"env_action={env_action.tolist()}"
                            )
                        last_exec_tstep = time.time()
                        if _poll_stop_key(FLAGS.show_image, stdin_fd):
                            print("[INFO] Stop requested by user; ending rollout early.")
                            stop_this_rollout = True
                            stop_requested = True
                            stop_reason = "manual"
                            break
                        if (time.monotonic() - rollout_t_start) > float(FLAGS.max_duration):
                            print("[INFO] Terminated by timeout.")
                            stop_this_rollout = True
                            stop_reason = "timeout"
                            break
                    if stop_this_rollout:
                        break

                    prev_policy_action = action.copy()

                    images.append(vis_rgb)
                    t += 1
                    if stop_this_rollout:
                        break
                if stop_this_rollout:
                    break

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            _restore_terminal_mode(stdin_fd, stdin_old_attrs)

        _save_rollout_video(images, loaded.name)
        if FLAGS.save_two_camera_videos:
            _save_rollout_video(cam0_images, f"{loaded.name}_cam0")
            _save_rollout_video(cam1_images, f"{loaded.name}_cam1")
        if stop_reason == "manual":
            print("[INFO] Rollout ended by user request (S key).")
        elif stop_reason == "timeout":
            print("[INFO] Rollout ended by timeout condition.")
        elif stop_reason == "term_area":
            print("[INFO] Rollout ended by termination-area condition.")
        rollout_idx += 1
        if stop_requested:
            print("[INFO] Evaluation stopped by user request.")
            break


if __name__ == "__main__":
    app.run(main)
