#!/usr/bin/env python3
"""Evaluate trained Diffusion/BFN policies on real WidowX.

This script is intentionally minimal and conservative:
- Loads policy architecture from benchmark config.
- Loads .ckpt weights (Diffusion or BFN).
- Runs WidowX rollout with bridge_data_robot-compatible relative actions.

Typical usage:

python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/diffusion_pusht_real.ckpt \
  --method auto \
  --config-source checkpoint \
  --camera-topics /D435/color/image_raw,/blue/image_raw \
  --num-timesteps 120
  
python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/bfn_pusht_real.ckpt \
  --method auto \
  --config-source checkpoint \
  --camera-topics /D435/color/image_raw,/blue/image_raw \
  --num-timesteps 120
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root and local dependency roots before importing local packages.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "src" / "diffusion-policy",
    PROJECT_ROOT.parent / "bridge_data_robot",
    PROJECT_ROOT.parent / "bridge_data_robot" / "widowx_envs",
    PROJECT_ROOT.parent / "bridge_data_robot" / "widowx_envs" / "multicam_server" / "src",
    Path.home() / "bridge_data_robot",
    Path.home() / "bridge_data_robot" / "widowx_envs",
    Path.home() / "bridge_data_robot" / "widowx_envs" / "multicam_server" / "src",
]:
    sp = str(p)
    if p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)

import hydra
try:
    from experiments.widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("experiments"):
        from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
    else:
        raise

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    from scipy.spatial.transform import Rotation as SciRotation
except Exception:
    SciRotation = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    from widowx_envs.utils import transformation_utils as wx_tr
except Exception:
    wx_tr = None

# Keep WidowX init consistent with example_widowx.py.
DEFAULT_WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
DEFAULT_INITIAL_EEP = [0.3, 0.0, 0.15]
WIDOWX_DEFAULT_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)

# Keep startup behavior close to example_widowx.py defaults.
STEP_DURATION = 0.1
MOVE_DURATION = 0.09
ACT_EXEC_HORIZON = 8
RPC_TIMEOUT_MS = 8000
STARTUP_TIMEOUT = 180.0
STARTUP_POLL_INTERVAL = 1.0
NEUTRAL_RETRIES = 60

# bridge_data_robot RobotBaseEnv action limits for 3trans3rot + gripper
REL_ACTION_LOW = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0], dtype=np.float32)
REL_ACTION_HIGH = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0], dtype=np.float32)


def _register_omegaconf_resolvers() -> None:
    # Keep parity with training-time Hydra resolvers used in benchmark configs.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now",
        lambda fmt="%Y-%m-%d": datetime.now().strftime(fmt),
        replace=True,
    )


def _stdin_has_data() -> bool:
    try:
        import select

        return select.select([sys.stdin], [], [], 0)[0] != []
    except Exception:
        return False


def _prep_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _status_is_success(status: Any) -> bool:
    if status is None:
        # Some variants use side effects only and return nothing.
        return True
    if isinstance(status, np.generic):
        status = status.item()
    if isinstance(status, bool):
        return status
    if isinstance(status, (int, float)):
        # WidowXStatus.SUCCESS is 1 in known implementations.
        return int(status) == 1
    s = str(status).strip().lower()
    if not s:
        return True
    return (
        "success" in s
        and "no_connection" not in s
        and "not_initialized" not in s
        and "failure" not in s
    )


def _set_widowx_rpc_timeout_ms(widowx_client: Any, timeout_ms: int) -> bool:
    """
    Best-effort override of edgeml REQ/REP client timeout.

    widowx_envs.WidowXClient wraps an internal ActionClient -> ReqRepClient
    with default timeout ~800ms, which is too short for slow init/reset calls.
    """
    try:
        action_client = getattr(widowx_client, "_WidowXClient__client", None)
        req_client = getattr(action_client, "client", None)
        if req_client is None:
            return False
        if not hasattr(req_client, "timeout_ms") or not hasattr(req_client, "reset_socket"):
            return False
        req_client.timeout_ms = int(timeout_ms)
        req_client.reset_socket()
        return True
    except Exception:
        return False


def _state_to_eep_transform(xyz_coor: List[float], zangle: float = 0.0) -> np.ndarray:
    """Mirror example_widowx.py state_to_eep() for move() initialization."""
    xyz = np.asarray(xyz_coor, dtype=np.float64).reshape(-1)
    if xyz.shape[0] != 3:
        raise ValueError(f"Expected xyz length 3, got {xyz.shape}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = xyz
    if SciRotation is None:
        raise RuntimeError("scipy is required for state_to_eep transform conversion.")
    rz = SciRotation.from_euler("z", float(zangle)).as_matrix()
    tf[:3, :3] = rz.dot(WIDOWX_DEFAULT_ROTATION)
    return tf


def _move_to_neutral(
    widowx_client: Any,
    initial_eep: List[float],
    blocking: bool = True,
    retries: int = 1,
    retry_interval_s: float = 0.5,
) -> None:
    neutral_tf = _state_to_eep_transform(initial_eep, zangle=0.0)
    errors: List[str] = []
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            # Keep parity with example_widowx.py startup:
            # reset -> open gripper -> move(state_to_eep(initial_eep, 0), duration=1.5)
            reset_fn = getattr(widowx_client, "reset", None)
            if callable(reset_fn):
                status = reset_fn()
                if not _status_is_success(status):
                    errors.append(f"reset returned non-success status {status!r}")
                    raise RuntimeError("reset failed")
                time.sleep(2.5)

            move_gripper_fn = getattr(widowx_client, "move_gripper", None)
            if callable(move_gripper_fn):
                move_gripper_fn(1.0)

            move_fn = getattr(widowx_client, "move", None)
            if callable(move_fn):
                status = move_fn(neutral_tf, duration=1.5, blocking=blocking)
                if _status_is_success(status):
                    return
                errors.append(f"move returned non-success status {status!r}")
            else:
                errors.append("move() is not available on WidowXClient")
        except Exception as e:
            errors.append(f"startup neutral sequence failed with {type(e).__name__}: {e}")

        # Compatibility fallback for variants exposing go_to_neutral only.
        go_to_neutral_fn = getattr(widowx_client, "go_to_neutral", None)
        if callable(go_to_neutral_fn):
            try:
                status = go_to_neutral_fn()
                if _status_is_success(status):
                    return
                errors.append(f"go_to_neutral returned non-success status {status!r}")
            except Exception as e:
                errors.append(f"go_to_neutral failed with {type(e).__name__}: {e}")

        if attempt < retries:
            if attempt == 1 or attempt % 5 == 0:
                print(f"[INFO] Waiting for WidowX neutral/reset to become available (attempt {attempt}/{retries})...")
            time.sleep(max(0.01, float(retry_interval_s)))

    if not errors:
        raise AttributeError("WidowXClient does not support reset()/move()/go_to_neutral().")
    raise RuntimeError(
        "Failed to move to neutral after retries. "
        f"Last error: {errors[-1]}"
    )


def _wait_for_startup_observation(
    widowx_client: Any,
    timeout_s: float,
    poll_interval_s: float,
) -> Dict[str, Any]:
    deadline = time.time() + max(0.0, float(timeout_s))
    poll_interval_s = max(0.01, float(poll_interval_s))
    attempts = 0
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        attempts += 1
        try:
            obs = widowx_client.get_observation()
            if obs is not None:
                if attempts > 1:
                    print(f"[INFO] WidowX observation stream is ready after {attempts} attempts.")
                return obs
        except Exception as e:
            last_error = e

        if attempts == 1 or attempts % max(1, int(round(5.0 / poll_interval_s))) == 0:
            print("[INFO] Waiting for WidowX initialization (controller/cameras) to finish...")
        time.sleep(poll_interval_s)

    if last_error is not None:
        raise TimeoutError(
            f"Timed out waiting {timeout_s:.1f}s for first observation. Last error: {type(last_error).__name__}: {last_error}"
        )
    raise TimeoutError(f"Timed out waiting {timeout_s:.1f}s for first observation from WidowX server.")


def _resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Checkpoint file must end with .ckpt, got: {p}")
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    candidates = sorted((p / "checkpoints").glob("*.ckpt"))
    if not candidates:
        candidates = sorted(p.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt file found under: {p}")

    # Prefer newest by modification time.
    return max(candidates, key=lambda x: x.stat().st_mtime)


def _load_payload(ckpt_path: Path) -> Dict[str, Any]:
    import dill

    with ckpt_path.open("rb") as f:
        try:
            payload = torch.load(f, map_location="cpu", pickle_module=dill, weights_only=False)
        except TypeError:
            # older torch may not support weights_only arg
            payload = torch.load(f, map_location="cpu", pickle_module=dill)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint format (not dict): {ckpt_path}")
    return payload


def _infer_method_from_cfg(cfg: Any) -> Optional[str]:
    try:
        target = str(cfg.policy._target_).lower()
    except Exception:
        target = str(cfg).lower()
    if "bfn" in target:
        return "bfn"
    if "diffusion" in target:
        return "diffusion"
    return None


def _get_benchmark_cfg_path(method: str, override_path: Optional[str]) -> Path:
    if override_path is not None:
        p = Path(override_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"benchmark config not found: {p}")
        return p
    if method == "bfn":
        return (PROJECT_ROOT / "config" / "benchmark_bfn_pusht_real.yaml").resolve()
    if method == "diffusion":
        return (PROJECT_ROOT / "config" / "benchmark_diffusion_pusht_real.yaml").resolve()
    raise ValueError(f"Unknown method: {method}")


def _sanitize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("_orig_mod.", "")
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    return cleaned


def _select_policy_state_dict(payload: Dict[str, Any], prefer_ema: bool) -> Dict[str, torch.Tensor]:
    state_dicts = payload.get("state_dicts", None)
    if isinstance(state_dicts, dict) and state_dicts:
        if prefer_ema and "ema_model" in state_dicts:
            state = state_dicts["ema_model"]
        elif "model" in state_dicts:
            state = state_dicts["model"]
        elif "ema_model" in state_dicts:
            state = state_dicts["ema_model"]
        else:
            raise KeyError("No 'model' or 'ema_model' in checkpoint state_dicts.")
        if not isinstance(state, dict):
            raise RuntimeError("Selected model state is not a dict.")
        return _sanitize_state_dict_keys(state)

    # Fallback: direct policy state dict.
    if all(k in payload for k in ("obs_encoder", "model")):
        return _sanitize_state_dict_keys(payload)

    raise RuntimeError("Cannot find policy weights in checkpoint payload.")


def _load_policy_from_cfg(
    method: str,
    cfg: Any,
    policy_state: Dict[str, torch.Tensor],
    device: torch.device,
    policy_steps: Optional[int],
) -> Tuple[torch.nn.Module, Any]:
    _register_omegaconf_resolvers()
    # Resolve only the policy subtree to avoid unrelated training/logging-only interpolations.
    OmegaConf.resolve(cfg.policy)

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()

    # Optional override for inference steps.
    if policy_steps is not None:
        if method == "bfn" and hasattr(policy, "n_timesteps"):
            policy.n_timesteps = int(policy_steps)
        if method == "diffusion" and hasattr(policy, "num_inference_steps"):
            policy.num_inference_steps = int(policy_steps)
            if hasattr(policy, "noise_scheduler"):
                policy.noise_scheduler.set_timesteps(int(policy_steps))

    return policy, cfg


def _to_hwc_uint8(
    image: Any,
    target_hw: Tuple[int, int],
    flat_hint_size: Optional[int] = None,
) -> np.ndarray:
    target_h, target_w = target_hw
    arr = np.asarray(image)
    # Some WidowX transports add singleton batch/channel dims (e.g., [1, N] or [1, H, W, C]).
    arr = np.squeeze(arr)

    # Handle flattened pixel tables [H*W,3] or [3,H*W] when provided.
    if arr.ndim == 2:
        if arr.shape == (target_h * target_w, 3):
            arr = arr.reshape(target_h, target_w, 3)
        elif arr.shape == (3, target_h * target_w):
            arr = arr.reshape(3, target_h, target_w).transpose(1, 2, 0)

    if arr.ndim == 1:
        total = arr.size
        if total % 3 != 0:
            raise ValueError(f"Flat image length {total} is not divisible by 3.")
        if total == 3 * target_h * target_w:
            c, h, w = 3, target_h, target_w
        elif flat_hint_size is not None and total == 3 * flat_hint_size * flat_hint_size:
            c, h, w = 3, flat_hint_size, flat_hint_size
        else:
            side = int(round((total / 3) ** 0.5))
            if side * side * 3 != total:
                raise ValueError(f"Cannot infer HxW from flat image length {total}.")
            c, h, w = 3, side, side
        arr = arr.reshape(c, h, w).transpose(1, 2, 0)

    elif arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(1, 2, 0)
        if arr.shape[-1] != 3:
            raise ValueError(f"Image last dim must be 3 channels, got {arr.shape}.")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if np.max(arr) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    th, tw = target_hw
    if arr.shape[0] != th or arr.shape[1] != tw:
        if cv2 is not None:
            arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            # Fallback nearest-neighbor resize without extra dependencies.
            y_idx = np.linspace(0, arr.shape[0] - 1, th).astype(np.int32)
            x_idx = np.linspace(0, arr.shape[1] - 1, tw).astype(np.int32)
            arr = arr[y_idx][:, x_idx]

    return arr


def _extract_camera_source(obs: Dict[str, Any], key: str, fallback_idx: int) -> Optional[Any]:
    if key in obs:
        return obs[key]

    images = obs.get("images", None)
    if isinstance(images, dict):
        if key in images:
            return images[key]
        values = [images[k] for k in sorted(images.keys())]
        if fallback_idx < len(values):
            return values[fallback_idx]

    # Legacy widowx_env_service variants expose camera streams as:
    # external_img, over_shoulder_img, wrist_img
    legacy_cam_keys = ["external_img", "over_shoulder_img", "wrist_img"]
    if fallback_idx < len(legacy_cam_keys):
        v = obs.get(legacy_cam_keys[fallback_idx], None)
        if v is not None:
            return v

    if "full_image" in obs:
        full = obs["full_image"]
        arr = np.asarray(full)
        if arr.ndim == 4 and fallback_idx < arr.shape[0]:
            return arr[fallback_idx]
        return full
    if "image" in obs:
        return obs["image"]
    return None


def _build_single_step_obs(
    raw_obs: Dict[str, Any],
    obs_meta: Dict[str, Any],
    rgb_keys: List[str],
    lowdim_keys: List[str],
    flat_hint_size: Optional[int],
    warned_missing: set,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # RGB keys -> CHW float32 [0,1]
    first_rgb_chw = None
    for i, key in enumerate(rgb_keys):
        shape = tuple(obs_meta[key]["shape"])
        target_hw = (int(shape[1]), int(shape[2]))
        src = _extract_camera_source(raw_obs, key=key, fallback_idx=i)

        if src is None:
            if first_rgb_chw is None:
                raise RuntimeError(f"Cannot find source image for {key} in WidowX observation.")
            out[key] = first_rgb_chw.copy()
            if key not in warned_missing:
                print(f"[WARN] Missing {key}; duplicated first available camera image.")
                warned_missing.add(key)
            continue

        hwc = _to_hwc_uint8(src, target_hw=target_hw, flat_hint_size=flat_hint_size)
        chw = np.moveaxis(hwc.astype(np.float32) / 255.0, -1, 0)
        out[key] = chw
        if first_rgb_chw is None:
            first_rgb_chw = chw

    # low-dim keys
    for key in lowdim_keys:
        dim = int(obs_meta[key]["shape"][0])
        if key in raw_obs:
            vec = np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
        elif key == "robot_eef_pose" and "state" in raw_obs:
            # widowx_env_service returns state in euler-angle parameterization.
            # Our training data stores robot_eef_pose as xyz + rotvec; convert here.
            state_vec = np.asarray(raw_obs["state"], dtype=np.float32).reshape(-1)
            if state_vec.shape[0] < 6:
                raise RuntimeError(f"Raw 'state' has invalid shape {state_vec.shape}, expected >=6.")
            vec = _state_euler6_to_rotvec6(state_vec[:6])
        else:
            raise RuntimeError(f"Cannot find low-dim key '{key}' in WidowX observation.")

        if vec.shape[0] < dim:
            pad = np.zeros((dim - vec.shape[0],), dtype=np.float32)
            vec = np.concatenate([vec, pad], axis=0)
        elif vec.shape[0] > dim:
            vec = vec[:dim]
        out[key] = vec.astype(np.float32)

    return out


def _state_euler6_to_rotvec6(euler_pose: np.ndarray) -> np.ndarray:
    euler_pose = np.asarray(euler_pose, dtype=np.float64).reshape(-1)
    if euler_pose.shape[0] < 6:
        raise ValueError(f"Expected >=6 values for euler pose, got {euler_pose.shape}")

    xyz = euler_pose[:3]
    euler = euler_pose[3:6]
    if wx_tr is None or SciRotation is None:
        # Best-effort fallback if transformation helpers are unavailable.
        return np.concatenate([xyz, euler], axis=0).astype(np.float32)

    state7 = np.concatenate([xyz, euler, np.array([1.0], dtype=np.float64)], axis=0)
    tf, _ = wx_tr.state2transform(state7, WIDOWX_DEFAULT_ROTATION)
    rotvec = SciRotation.from_matrix(tf[:3, :3]).as_rotvec()
    return np.concatenate([xyz, rotvec], axis=0).astype(np.float32)


def _pose_rotvec6_to_transform(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.shape[0] < 6:
        raise ValueError(f"Expected >=6 values for pose, got {pose6.shape}")
    if SciRotation is None:
        raise RuntimeError("scipy is required for rotvec conversion but is unavailable.")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = SciRotation.from_rotvec(pose6[3:6]).as_matrix()
    tf[:3, 3] = pose6[:3]
    return tf


def _extract_current_pose_rotvec_from_obs(raw_obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if "robot_eef_pose" in raw_obs:
        vec = np.asarray(raw_obs["robot_eef_pose"], dtype=np.float32).reshape(-1)
        if vec.shape[0] >= 6:
            return vec[:6]
    if "state" in raw_obs:
        vec = np.asarray(raw_obs["state"], dtype=np.float32).reshape(-1)
        if vec.shape[0] >= 6:
            return _state_euler6_to_rotvec6(vec[:6])
    return None


def _infer_action_space(action7: np.ndarray) -> str:
    a = np.asarray(action7, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got {a.shape}")
    is_relative = (
        np.max(np.abs(a[:3])) <= 0.08
        and np.max(np.abs(a[3:6])) <= 0.4
    )
    return "relative" if is_relative else "absolute"


def _infer_action_space_from_policy(policy: Any) -> Optional[str]:
    """Infer action representation from normalizer stats when available."""
    try:
        stats = policy.normalizer["action"].get_input_stats()
        amin = np.asarray(stats["min"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        amax = np.asarray(stats["max"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if amin.shape[0] < 6 or amax.shape[0] < 6:
            return None
        # Relative WidowX deltas are typically bounded to ~0.05m / 0.25rad.
        rel_like = (
            np.max(np.abs(amin[:3])) <= 0.12
            and np.max(np.abs(amax[:3])) <= 0.12
            and np.max(np.abs(amin[3:6])) <= 0.6
            and np.max(np.abs(amax[3:6])) <= 0.6
        )
        return "relative" if rel_like else "absolute"
    except Exception:
        return None


def _absolute_action_to_relative(
    target_action7: np.ndarray,
    current_pose6_rotvec: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_action7, dtype=np.float64).reshape(-1)
    curr = np.asarray(current_pose6_rotvec, dtype=np.float64).reshape(-1)
    if target.shape[0] != 7:
        raise ValueError(f"Expected target action shape (7,), got {target.shape}")
    if curr.shape[0] < 6:
        raise ValueError(f"Expected current pose shape (>=6,), got {curr.shape}")

    # Primary path: exact transform math with bridge_data_robot utility.
    if wx_tr is not None and SciRotation is not None:
        tf_target = _pose_rotvec6_to_transform(target[:6])
        tf_curr = _pose_rotvec6_to_transform(curr[:6])
        tf_delta = tf_target.dot(np.linalg.inv(tf_curr))
        rel = wx_tr.transform2action_local(
            tf_delta,
            np.array([target[6]], dtype=np.float64),
            curr[:3],
        )
        return np.asarray(rel, dtype=np.float32).reshape(-1)

    # Fallback: simple component delta.
    rel = np.zeros((7,), dtype=np.float32)
    rel[:6] = (target[:6] - curr[:6]).astype(np.float32)
    rel[6] = float(target[6])
    return rel


def _clip_relative_action(action7: np.ndarray) -> np.ndarray:
    a = np.asarray(action7, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got {a.shape}")
    return np.clip(a, REL_ACTION_LOW, REL_ACTION_HIGH).astype(np.float32)


def _absolute_action_to_relative_from_target_tf(
    target_action7: np.ndarray,
    current_target_tf: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_action7, dtype=np.float64).reshape(-1)
    if target.shape[0] != 7:
        raise ValueError(f"Expected target action shape (7,), got {target.shape}")
    tf_target = _pose_rotvec6_to_transform(target[:6])
    tf_delta = tf_target.dot(np.linalg.inv(current_target_tf))
    if wx_tr is not None:
        rel = wx_tr.transform2action_local(
            tf_delta,
            np.array([target[6]], dtype=np.float64),
            current_target_tf[:3, 3],
        )
        return np.asarray(rel, dtype=np.float32).reshape(-1)

    # Fallback if bridge transformation utils are unavailable.
    if SciRotation is None:
        raise RuntimeError("Cannot convert absolute->relative action without scipy or widowx transform utils.")
    curr_pose = np.concatenate(
        [current_target_tf[:3, 3], SciRotation.from_matrix(current_target_tf[:3, :3]).as_rotvec()],
        axis=0,
    )
    return _absolute_action_to_relative(target_action7=target, current_pose6_rotvec=curr_pose)


def _apply_relative_action_to_pose(
    current_pose6_rotvec: np.ndarray,
    rel_action7: np.ndarray,
) -> np.ndarray:
    """Apply executed relative action to current pose estimate."""
    curr = np.asarray(current_pose6_rotvec, dtype=np.float64).reshape(-1)
    rel = np.asarray(rel_action7, dtype=np.float64).reshape(-1)
    if curr.shape[0] < 6 or rel.shape[0] != 7:
        raise ValueError(f"Invalid shapes for pose/action update: {curr.shape}, {rel.shape}")

    if wx_tr is not None and SciRotation is not None:
        curr_tf = _pose_rotvec6_to_transform(curr[:6])
        delta_tf, _ = wx_tr.action2transform_local(rel, curr[:3])
        next_tf = delta_tf.dot(curr_tf)
        next_rotvec = SciRotation.from_matrix(next_tf[:3, :3]).as_rotvec()
        return np.concatenate([next_tf[:3, 3], next_rotvec], axis=0).astype(np.float32)

    # Fallback approximation.
    out = curr[:6].astype(np.float32).copy()
    out[:6] = out[:6] + rel[:6].astype(np.float32)
    return out


def _stack_obs_history(history: deque, keys: List[str]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for key in keys:
        result[key] = np.stack([x[key] for x in history], axis=0).astype(np.float32)
    return result


def _resolve_control_frequencies(
    step_duration_s: Optional[float],
    policy_hz: float,
    robot_hz: float,
) -> Tuple[float, float, int]:
    if step_duration_s is not None:
        if float(step_duration_s) <= 0.0:
            raise ValueError(f"--step-duration must be > 0, got {step_duration_s}")
        policy_hz = 1.0 / float(step_duration_s)

    policy_hz = float(policy_hz)
    robot_hz = float(robot_hz)
    if policy_hz <= 0.0:
        raise ValueError(f"--policy-hz must be > 0, got {policy_hz}")
    if robot_hz <= 0.0:
        raise ValueError(f"--robot-hz must be > 0, got {robot_hz}")
    if robot_hz < policy_hz:
        raise ValueError(
            f"--robot-hz ({robot_hz}) must be >= --policy-hz ({policy_hz}) for interpolation."
        )

    ratio = robot_hz / policy_hz
    interp_substeps = int(round(ratio))
    if interp_substeps < 1:
        interp_substeps = 1
    if abs(ratio - interp_substeps) > 1e-4:
        raise ValueError(
            f"robot_hz/policy_hz must be an integer multiple; got ratio={ratio:.6f} "
            f"(policy_hz={policy_hz}, robot_hz={robot_hz})."
        )
    return policy_hz, robot_hz, interp_substeps


def _interpolate_actions(
    start_action: np.ndarray,
    end_action: np.ndarray,
    substeps: int,
) -> np.ndarray:
    start = np.asarray(start_action, dtype=np.float32).reshape(-1)
    end = np.asarray(end_action, dtype=np.float32).reshape(-1)
    if start.shape != end.shape:
        raise ValueError(f"Action shape mismatch for interpolation: {start.shape} vs {end.shape}")
    if substeps <= 1:
        return end[None, :]

    # Exclude alpha=0 so we don't resend the previous command.
    alphas = np.linspace(0.0, 1.0, num=substeps + 1, dtype=np.float32)[1:]
    return start[None, :] + (end - start)[None, :] * alphas[:, None]


def _print_runtime_config(
    method: str,
    cfg_source: str,
    cfg: Any,
    ckpt_path: Path,
    policy_hz: float,
    robot_hz: float,
    step_duration: float,
    move_duration: float,
):
    image_shape = list(cfg.image_shape)
    crop_shape = list(cfg.crop_shape)
    print("=" * 80)
    print("WidowX Eval Runtime Config")
    print(f"method               : {method}")
    print(f"checkpoint           : {ckpt_path}")
    print(f"config source        : {cfg_source}")
    print(f"horizon              : {int(cfg.horizon)}")
    print(f"n_obs_steps          : {int(cfg.n_obs_steps)}")
    print(f"n_action_steps       : {int(cfg.n_action_steps)}")
    print(f"image_shape (C,H,W)  : {image_shape}")
    print(f"crop_shape (H,W)     : {crop_shape}")
    print(f"action_dim           : {int(cfg.shape_meta.action.shape[0])}")
    print(f"policy_hz            : {policy_hz:.3f}")
    print(f"robot_hz             : {robot_hz:.3f}")
    print(f"control_step_duration: {step_duration:.3f}")
    print(f"env move_duration    : {move_duration:.3f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser("Evaluate trained Diffusion/BFN checkpoint on real WidowX.")
    parser.add_argument("--checkpoint", type=str, required=True, help=".ckpt file or run dir containing checkpoints/")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "bfn", "diffusion"],
        help="Policy type. 'auto' infers from checkpoint cfg.policy._target_.",
    )
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--camera-topics",
        type=str,
        default="/D435/color/image_raw,/blue/image_raw",
        help="Comma-separated ROS camera topics.",
    )
    parser.add_argument("--policy-hz", type=float, default=1.0 / STEP_DURATION, help="Policy control frequency (Hz).")
    parser.add_argument("--robot-hz", type=float, default=1.0 / STEP_DURATION, help="Robot control frequency (Hz).")
    parser.add_argument("--move-duration", type=float, default=MOVE_DURATION, help="WidowX move duration for each command.")
    parser.add_argument(
        "--act-exec-horizon",
        type=int,
        default=ACT_EXEC_HORIZON,
        help="Number of predicted actions to execute per inference.",
    )
    parser.add_argument("--gripper-value", type=float, default=None, help="If set, override action gripper command in [0,1].")
    parser.add_argument("--no-pitch-roll", action="store_true", help="Zero pitch/roll relative commands.")
    parser.add_argument("--no-yaw", action="store_true", help="Zero yaw relative commands.")
    parser.add_argument("--num-timesteps", type=int, default=120)
    parser.add_argument("--show-image", action="store_true")
    parser.add_argument("--video-save-path", type=str, default=None)
    parser.add_argument("--initial-eep", type=float, nargs=3, default=DEFAULT_INITIAL_EEP)
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default=None,
        help=(
            "Optional benchmark config path used when --config-source=benchmark "
            "(or as fallback when checkpoint cfg is unavailable)."
        ),
    )
    parser.add_argument(
        "--config-source",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "benchmark"],
        help=(
            "Policy/shape config source. "
            "'checkpoint' uses cfg embedded in the .ckpt for exact train-eval parity. "
            "'benchmark' uses benchmark YAML."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "WidowX service image_size for init. "
            "Default: max(H,W) from eval config image_shape; fallback 256."
        ),
    )
    args = parser.parse_args()

    if args.show_image and cv2 is None:
        print("[WARN] cv2 is not installed, disabling --show-image.")
        args.show_image = False

    if args.policy_hz <= 0.0:
        raise ValueError(f"--policy-hz must be > 0, got {args.policy_hz}")
    if args.robot_hz <= 0.0:
        raise ValueError(f"--robot-hz must be > 0, got {args.robot_hz}")
    if args.robot_hz + 1e-6 < args.policy_hz:
        raise ValueError(f"--robot-hz ({args.robot_hz}) must be >= --policy-hz ({args.policy_hz})")
    ratio = args.robot_hz / args.policy_hz
    if abs(ratio - round(ratio)) > 1e-4:
        raise ValueError(
            f"--robot-hz/--policy-hz must be integer multiple, got {ratio:.6f} "
            f"({args.robot_hz}/{args.policy_hz})"
        )

    step_duration = 1.0 / float(args.policy_hz)
    move_duration = float(args.move_duration)
    if move_duration <= 0.0:
        raise ValueError(f"--move-duration must be > 0, got {move_duration}")

    device = _prep_device("cuda")
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    payload = _load_payload(ckpt_path)

    payload_cfg = payload.get("cfg", None)
    payload_method = _infer_method_from_cfg(payload_cfg)
    method = args.method
    if method == "auto":
        if payload_method is None:
            raise RuntimeError("Cannot infer --method from checkpoint cfg. Please pass --method bfn|diffusion.")
        method = payload_method
    elif payload_method is not None and method != payload_method:
        raise ValueError(
            f"Checkpoint appears to be '{payload_method}' but --method was '{method}'. "
            "Use --method auto or match the checkpoint type."
        )

    benchmark_cfg_path = _get_benchmark_cfg_path(method=method, override_path=args.benchmark_config)
    eval_cfg = None
    cfg_source = ""
    if args.config_source == "checkpoint":
        if payload_cfg is None:
            raise RuntimeError(
                "Checkpoint has no cfg payload. "
                "Use --config-source benchmark and optionally --benchmark-config."
            )
        try:
            eval_cfg = copy.deepcopy(payload_cfg)
            # Basic structural checks used by this runner.
            _ = eval_cfg.policy
            _ = eval_cfg.shape_meta.action.shape
            _ = eval_cfg.shape_meta.obs
            _ = eval_cfg.n_obs_steps
            _ = eval_cfg.n_action_steps
            _ = eval_cfg.image_shape
            _ = eval_cfg.crop_shape
            cfg_source = "checkpoint cfg"
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint cfg is incomplete for eval: {type(e).__name__}: {e}. "
                "Use --config-source benchmark and optionally --benchmark-config."
            )
    else:
        eval_cfg = OmegaConf.load(str(benchmark_cfg_path))
        cfg_source = str(benchmark_cfg_path)

    policy_state = _select_policy_state_dict(payload, prefer_ema=True)
    policy, eval_cfg = _load_policy_from_cfg(
        method=method,
        cfg=eval_cfg,
        policy_state=policy_state,
        device=device,
        policy_steps=None,
    )

    _print_runtime_config(
        method=method,
        cfg_source=cfg_source,
        cfg=eval_cfg,
        ckpt_path=ckpt_path,
        policy_hz=float(args.policy_hz),
        robot_hz=float(args.robot_hz),
        step_duration=step_duration,
        move_duration=move_duration,
    )

    n_obs_steps = int(eval_cfg.n_obs_steps)
    n_action_steps = int(eval_cfg.n_action_steps)
    if hasattr(policy, "n_obs_steps") and int(policy.n_obs_steps) != n_obs_steps:
        raise RuntimeError(f"Policy n_obs_steps={policy.n_obs_steps} != benchmark n_obs_steps={n_obs_steps}")
    if hasattr(policy, "n_action_steps") and int(policy.n_action_steps) != n_action_steps:
        raise RuntimeError(
            f"Policy n_action_steps={policy.n_action_steps} != benchmark n_action_steps={n_action_steps}"
        )

    if args.act_exec_horizon <= 0:
        raise ValueError(f"--act-exec-horizon must be >= 1, got {args.act_exec_horizon}")
    act_exec_horizon = min(int(args.act_exec_horizon), n_action_steps)
    obs_meta = eval_cfg.shape_meta.obs
    rgb_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "low_dim"]
    all_obs_keys = rgb_keys + lowdim_keys
    action_dim = int(eval_cfg.shape_meta.action.shape[0])
    if action_dim != 7:
        raise RuntimeError(f"Expected action_dim=7 for WidowX, got {action_dim}.")

    topics = [x.strip() for x in args.camera_topics.split(",") if x.strip()]
    if not topics:
        raise ValueError("camera-topics cannot be empty")
    if len(rgb_keys) > len(topics):
        print(
            f"[WARN] Policy expects {len(rgb_keys)} RGB streams but only {len(topics)} camera topic(s) were provided. "
            "Missing streams will duplicate camera_0."
        )

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": [{"name": t} for t in topics],
            "override_workspace_boundaries": DEFAULT_WORKSPACE_BOUNDS,
            "move_duration": move_duration,
        }
    )
    start_state = list(np.concatenate([np.asarray(args.initial_eep, dtype=np.float32), [0, 0, 0, 1]]))
    # Keep init payload parity with example_widowx.py.
    env_params["state_state"] = start_state

    widowx_client = WidowXClient(host=args.ip, port=args.port)
    if _set_widowx_rpc_timeout_ms(widowx_client, RPC_TIMEOUT_MS):
        print(f"[INFO] Set WidowX RPC timeout to {RPC_TIMEOUT_MS} ms.")
    else:
        print("[WARN] Could not override WidowX RPC timeout; using library default.")

    if args.image_size is not None:
        init_image_size = int(args.image_size)
    else:
        try:
            im_shape = [int(x) for x in eval_cfg.image_shape]
            init_image_size = max(int(im_shape[1]), int(im_shape[2]))
        except Exception:
            init_image_size = 256
    if init_image_size <= 0:
        raise ValueError(f"--image-size must be > 0, got {init_image_size}")
    print(f"[INFO] widowx_client.init image_size={init_image_size}")

    init_status = widowx_client.init(env_params, image_size=init_image_size)
    if not _status_is_success(init_status):
        print(f"[WARN] widowx_client.init returned non-success status {init_status!r}; waiting for readiness anyway.")

    _wait_for_startup_observation(
        widowx_client,
        timeout_s=STARTUP_TIMEOUT,
        poll_interval_s=STARTUP_POLL_INTERVAL,
    )

    obs_hist: deque = deque(maxlen=n_obs_steps)
    warned_missing = set()
    saved_frames: List[np.ndarray] = []
    reset_requested = False
    action_space_mode: Optional[str] = _infer_action_space_from_policy(policy)
    if action_space_mode is not None:
        print(f"[INFO] Inferred model action space from normalizer stats: {action_space_mode}.")
    warned_missing_pose = False

    def _read_obs_retry() -> Dict[str, Any]:
        while True:
            try:
                obs = widowx_client.get_observation()
            except Exception as e:
                print(f"[WARN] get_observation() failed: {type(e).__name__}: {e}; retrying...")
                time.sleep(max(0.01, STARTUP_POLL_INTERVAL))
                continue
            if obs is not None:
                return obs
            print("[WARN] get_observation() returned None; retrying...")
            time.sleep(max(0.01, STARTUP_POLL_INTERVAL))

    try:
        try:
            _move_to_neutral(
                widowx_client,
                initial_eep=args.initial_eep,
                blocking=True,
                retries=NEUTRAL_RETRIES,
                retry_interval_s=STARTUP_POLL_INTERVAL,
            )
        except Exception as e:
            print(f"[WARN] Startup move-to-neutral failed: {e}. Continuing anyway.")

        time.sleep(0.5)
        input("Press [Enter] to start real-robot evaluation...")

        t = 0
        last_tstep = time.time() - step_duration
        while t < args.num_timesteps:
            now = time.time()
            if now < last_tstep + step_duration:
                time.sleep(0.001)
                continue

            raw_obs = _read_obs_retry()

            if args.show_image:
                vis_img = None
                if "full_image" in raw_obs:
                    arr = np.asarray(raw_obs["full_image"])
                    if arr.ndim == 4 and arr.shape[0] > 0:
                        vis_img = arr[0]
                    elif arr.ndim == 3:
                        vis_img = arr
                if vis_img is None and "external_img" in raw_obs:
                    vis_img = np.asarray(raw_obs["external_img"])
                if vis_img is not None and vis_img.ndim == 3 and vis_img.shape[-1] == 3:
                    bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("widowx_eval", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("r"), ord("R")):
                    reset_requested = True
                    print("[INFO] Reset requested by keyboard.")
                    break
            elif _stdin_has_data():
                try:
                    line = sys.stdin.readline().strip().lower()
                    if line == "r":
                        reset_requested = True
                        print("[INFO] Reset requested by stdin.")
                        break
                except Exception:
                    pass

            step_obs = _build_single_step_obs(
                raw_obs=raw_obs,
                obs_meta=obs_meta,
                rgb_keys=rgb_keys,
                lowdim_keys=lowdim_keys,
                flat_hint_size=None,
                warned_missing=warned_missing,
            )

            if len(obs_hist) == 0:
                obs_hist.extend([step_obs] * n_obs_steps)
            else:
                obs_hist.append(step_obs)

            model_obs_np = _stack_obs_history(obs_hist, keys=all_obs_keys)
            model_obs_t = {
                k: torch.from_numpy(v).unsqueeze(0).to(device=device, dtype=torch.float32)
                for k, v in model_obs_np.items()
            }

            with torch.no_grad():
                result = policy.predict_action(model_obs_t)
            action_seq = result["action"].detach().cpu().numpy()[0]
            if action_seq.ndim == 1:
                action_seq = action_seq[None]

            curr_pose = _extract_current_pose_rotvec_from_obs(raw_obs)
            curr_pose_exec = None if curr_pose is None else curr_pose.copy()
            exec_steps = min(act_exec_horizon, action_seq.shape[0], args.num_timesteps - t)
            for i in range(exec_steps):
                model_action = np.asarray(action_seq[i], dtype=np.float32).reshape(-1)
                if model_action.shape[0] != 7:
                    raise RuntimeError(
                        f"Expected 7D policy action for WidowX step_action, got shape {tuple(model_action.shape)}"
                    )

                if action_space_mode is None:
                    action_space_mode = _infer_action_space(model_action)
                    print(f"[INFO] Inferred model action space: {action_space_mode}.")

                if action_space_mode == "absolute":
                    if curr_pose_exec is None:
                        if not warned_missing_pose:
                            print("[WARN] Missing current pose in observation; skipping absolute->relative conversion this step.")
                            warned_missing_pose = True
                        continue
                    rel_action = _absolute_action_to_relative(model_action, curr_pose_exec)
                else:
                    rel_action = model_action.copy()

                if args.gripper_value is not None:
                    rel_action[6] = float(np.clip(args.gripper_value, 0.0, 1.0))
                if args.no_pitch_roll:
                    rel_action[3] = 0.0
                    rel_action[4] = 0.0
                if args.no_yaw:
                    rel_action[5] = 0.0

                rel_action = _clip_relative_action(rel_action)
                if action_space_mode == "absolute" and curr_pose_exec is not None:
                    curr_pose_exec = _apply_relative_action_to_pose(curr_pose_exec, rel_action)
                widowx_client.step_action(rel_action, blocking=True)
                last_tstep = time.time()

                cam0 = step_obs[rgb_keys[0]]
                frame = np.moveaxis(cam0, 0, -1)
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                saved_frames.append(frame)

                t += 1
                if t >= args.num_timesteps:
                    break

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C received, stopping rollout.")
    finally:
        try:
            _move_to_neutral(
                widowx_client,
                initial_eep=args.initial_eep,
                blocking=True,
                retries=max(1, min(5, NEUTRAL_RETRIES)),
                retry_interval_s=STARTUP_POLL_INTERVAL,
            )
        except Exception as e:
            print(f"[WARN] Failed to move to neutral: {e}")
        if args.show_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if args.video_save_path and saved_frames and not reset_requested:
        if imageio is None:
            raise ModuleNotFoundError("imageio is required for --video-save-path but is not installed.")
        out_dir = Path(args.video_save_path).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = out_dir / f"{ts}_{method}_widowx_eval.mp4"
        imageio.mimsave(str(out_path), saved_frames, fps=1.0 / step_duration)
        print(f"[INFO] Saved rollout video: {out_path}")


if __name__ == "__main__":
    main()
