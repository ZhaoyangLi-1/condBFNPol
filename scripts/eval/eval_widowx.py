#!/usr/bin/env python3
"""Evaluate trained Diffusion/BFN policies on real WidowX.

This script:
1) Loads policy architecture/hyperparameters from benchmark config
   - config/benchmark_diffusion_pusht_real.yaml
   - config/benchmark_bfn_pusht_real.yaml
2) Loads checkpoint weights from a .ckpt file (or run directory).
3) Runs real-robot rollout with WidowXClient using benchmark-aligned horizons.

# diffusion
python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/diffusion_real_pusht.ckpt \
  --method bfn \
  --policy-hz 10 \
  --robot-hz 30

# bfn
python scripts/eval/eval_widowx.py \
  --checkpoint /data/BFN_data/checkpoints/bfn_real_pusht.ckpt \
  --method bfn \
  --policy-hz 10 \
  --robot-hz 30
"""

from __future__ import annotations

import argparse
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
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None

DEFAULT_WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]


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


def _move_to_neutral(
    widowx_client: Any,
    initial_eep: List[float],
    blocking: bool = True,
    retries: int = 1,
    retry_interval_s: float = 0.5,
) -> None:
    neutral_pose = np.asarray(list(initial_eep) + [0.0, 0.0, 0.0], dtype=np.float32)
    errors: List[str] = []
    for attempt in range(1, max(1, int(retries)) + 1):
        for name, fn in (
            ("reset", lambda: getattr(widowx_client, "reset", None)),
            ("go_to_neutral", lambda: getattr(widowx_client, "go_to_neutral", None)),
            ("move", lambda: getattr(widowx_client, "move", None)),
        ):
            method = fn()
            if not callable(method):
                continue
            try:
                if name == "move":
                    status = method(neutral_pose, duration=1.0, blocking=blocking)
                else:
                    status = method()
                if _status_is_success(status):
                    return
                errors.append(f"{name} returned non-success status {status!r}")
            except Exception as e:
                errors.append(f"{name} failed with {type(e).__name__}: {e}")
        if attempt < retries:
            if attempt == 1 or attempt % 5 == 0:
                print(f"[INFO] Waiting for WidowX neutral/reset to become available (attempt {attempt}/{retries})...")
            time.sleep(max(0.01, float(retry_interval_s)))

    if not errors:
        raise AttributeError("WidowXClient does not support reset(), go_to_neutral(), or move().")
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


def _load_policy_from_benchmark(
    method: str,
    benchmark_cfg_path: Path,
    policy_state: Dict[str, torch.Tensor],
    device: torch.device,
    policy_steps: Optional[int],
) -> Tuple[torch.nn.Module, Any]:
    _register_omegaconf_resolvers()
    cfg = OmegaConf.load(str(benchmark_cfg_path))
    # Resolve only the policy subtree to avoid unrelated training/logging-only interpolations.
    OmegaConf.resolve(cfg.policy)

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(policy_state, strict=True)
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
    arr = np.asarray(image)

    if arr.ndim == 1:
        total = arr.size
        if total % 3 != 0:
            raise ValueError(f"Flat image length {total} is not divisible by 3.")
        target_h, target_w = target_hw
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
            vec = np.asarray(raw_obs["state"], dtype=np.float32).reshape(-1)
        else:
            raise RuntimeError(f"Cannot find low-dim key '{key}' in WidowX observation.")

        if vec.shape[0] < dim:
            pad = np.zeros((dim - vec.shape[0],), dtype=np.float32)
            vec = np.concatenate([vec, pad], axis=0)
        elif vec.shape[0] > dim:
            vec = vec[:dim]
        out[key] = vec.astype(np.float32)

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
    benchmark_cfg_path: Path,
    cfg: Any,
    ckpt_path: Path,
    policy_hz: float,
    robot_hz: float,
    interp_substeps: int,
):
    image_shape = list(cfg.image_shape)
    crop_shape = list(cfg.crop_shape)
    print("=" * 80)
    print("WidowX Eval Runtime Config")
    print(f"method               : {method}")
    print(f"checkpoint           : {ckpt_path}")
    print(f"benchmark config     : {benchmark_cfg_path}")
    print(f"horizon              : {int(cfg.horizon)}")
    print(f"n_obs_steps          : {int(cfg.n_obs_steps)}")
    print(f"n_action_steps       : {int(cfg.n_action_steps)}")
    print(f"image_shape (C,H,W)  : {image_shape}")
    print(f"crop_shape (H,W)     : {crop_shape}")
    print(f"action_dim           : {int(cfg.shape_meta.action.shape[0])}")
    print(f"policy_hz            : {policy_hz:.3f}")
    print(f"robot_hz             : {robot_hz:.3f}")
    print(f"interp_substeps      : {interp_substeps}")
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
    parser.add_argument("--benchmark-config", type=str, default=None, help="Override benchmark yaml path.")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer ema_model weights if present.",
    )
    parser.add_argument("--policy-steps", type=int, default=None, help="Override inference steps (BFN n_timesteps / Diffusion num_inference_steps).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # WidowX runtime
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--camera-topics", type=str, default="/blue/image_raw", help="Comma-separated ROS camera topics.")
    parser.add_argument("--client-image-size", type=int, default=None, help="Optional WidowX image_size init arg; used as flat image reshape hint.")
    parser.add_argument(
        "--step-duration",
        type=float,
        default=None,
        help="Deprecated override of policy period in seconds; if set, overrides --policy-hz.",
    )
    parser.add_argument("--policy-hz", type=float, default=10.0, help="Policy command rate before interpolation.")
    parser.add_argument(
        "--robot-hz",
        type=float,
        default=30.0,
        help="Robot execution rate after interpolation (must be integer multiple of --policy-hz).",
    )
    parser.add_argument("--num-timesteps", type=int, default=120)
    parser.add_argument("--act-exec-horizon", type=int, default=None, help="How many predicted actions to execute per inference. Default: benchmark n_action_steps.")
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--show-image", action="store_true")
    parser.add_argument("--video-save-path", type=str, default=None)
    parser.add_argument("--initial-eep", type=float, nargs=3, default=[0.3, 0.0, 0.15])
    parser.add_argument("--gripper-value", type=float, default=None, help="If set, override the 7th (gripper) component of 7D action.")
    parser.add_argument("--no-pitch-roll", action="store_true")
    parser.add_argument("--no-yaw", action="store_true")
    parser.add_argument(
        "--rpc-timeout-ms",
        type=int,
        default=5000,
        help="ZMQ RPC timeout for WidowX client calls (init/reset can exceed default 800ms).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=90.0,
        help="Max seconds to wait for first observation after init (slow robot/camera startup).",
    )
    parser.add_argument(
        "--startup-poll-interval",
        type=float,
        default=0.5,
        help="Polling interval while waiting for startup readiness.",
    )
    parser.add_argument(
        "--neutral-retries",
        type=int,
        default=20,
        help="How many times to retry neutral/reset when startup is still busy.",
    )
    args = parser.parse_args()

    if args.show_image and cv2 is None:
        print("[WARN] cv2 is not installed, disabling --show-image.")
        args.show_image = False

    policy_hz, robot_hz, interp_substeps = _resolve_control_frequencies(
        step_duration_s=args.step_duration,
        policy_hz=args.policy_hz,
        robot_hz=args.robot_hz,
    )
    robot_step_duration = 1.0 / robot_hz

    device = _prep_device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    payload = _load_payload(ckpt_path)

    # Infer policy method from checkpoint config if needed.
    payload_cfg = payload.get("cfg", None)
    method = args.method
    if method == "auto":
        inferred = _infer_method_from_cfg(payload_cfg)
        if inferred is None:
            raise RuntimeError("Cannot infer --method from checkpoint cfg. Please pass --method bfn|diffusion.")
        method = inferred

    benchmark_cfg_path = _get_benchmark_cfg_path(method=method, override_path=args.benchmark_config)
    policy_state = _select_policy_state_dict(payload, prefer_ema=args.use_ema)
    policy, benchmark_cfg = _load_policy_from_benchmark(
        method=method,
        benchmark_cfg_path=benchmark_cfg_path,
        policy_state=policy_state,
        device=device,
        policy_steps=args.policy_steps,
    )

    _print_runtime_config(
        method=method,
        benchmark_cfg_path=benchmark_cfg_path,
        cfg=benchmark_cfg,
        ckpt_path=ckpt_path,
        policy_hz=policy_hz,
        robot_hz=robot_hz,
        interp_substeps=interp_substeps,
    )

    # Benchmark-aligned hyperparameters.
    n_obs_steps = int(benchmark_cfg.n_obs_steps)
    n_action_steps = int(benchmark_cfg.n_action_steps)
    if hasattr(policy, "n_obs_steps") and int(policy.n_obs_steps) != n_obs_steps:
        raise RuntimeError(f"Policy n_obs_steps={policy.n_obs_steps} != benchmark n_obs_steps={n_obs_steps}")
    if hasattr(policy, "n_action_steps") and int(policy.n_action_steps) != n_action_steps:
        raise RuntimeError(
            f"Policy n_action_steps={policy.n_action_steps} != benchmark n_action_steps={n_action_steps}"
        )
    act_exec_horizon = int(args.act_exec_horizon) if args.act_exec_horizon is not None else n_action_steps

    obs_meta = benchmark_cfg.shape_meta.obs
    rgb_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "low_dim"]
    all_obs_keys = rgb_keys + lowdim_keys
    action_dim = int(benchmark_cfg.shape_meta.action.shape[0])

    # Init WidowX.
    topics = [x.strip() for x in args.camera_topics.split(",") if x.strip()]
    if not topics:
        raise ValueError("camera-topics cannot be empty")
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": [{"name": t} for t in topics],
            "override_workspace_boundaries": DEFAULT_WORKSPACE_BOUNDS,
            "move_duration": float(robot_step_duration),
        }
    )
    env_params["start_state"] = list(np.concatenate([np.asarray(args.initial_eep, dtype=np.float32), [0, 0, 0, 1]]))

    widowx_client = WidowXClient(host=args.ip, port=args.port)
    if _set_widowx_rpc_timeout_ms(widowx_client, args.rpc_timeout_ms):
        print(f"[INFO] Set WidowX RPC timeout to {int(args.rpc_timeout_ms)} ms.")
    else:
        print("[WARN] Could not override WidowX RPC timeout; using library default.")
    if args.client_image_size is None:
        init_status = widowx_client.init(env_params)
    else:
        init_status = widowx_client.init(env_params, image_size=int(args.client_image_size))
    if not _status_is_success(init_status):
        print(f"[WARN] widowx_client.init returned non-success status {init_status!r}; waiting for readiness anyway.")

    _wait_for_startup_observation(
        widowx_client,
        timeout_s=args.startup_timeout,
        poll_interval_s=args.startup_poll_interval,
    )

    obs_hist: deque = deque(maxlen=n_obs_steps)
    warned_missing = set()
    saved_frames: List[np.ndarray] = []
    reset_requested = False

    def _read_obs_retry() -> Dict[str, Any]:
        while True:
            try:
                obs = widowx_client.get_observation()
            except Exception as e:
                print(f"[WARN] get_observation() failed: {type(e).__name__}: {e}; retrying...")
                time.sleep(max(0.01, float(args.startup_poll_interval)))
                continue
            if obs is not None:
                return obs
            print("[WARN] get_observation() returned None; retrying...")
            time.sleep(max(0.01, float(args.startup_poll_interval)))

    try:
        try:
            _move_to_neutral(
                widowx_client,
                initial_eep=args.initial_eep,
                blocking=True,
                retries=args.neutral_retries,
                retry_interval_s=args.startup_poll_interval,
            )
        except Exception as e:
            print(f"[WARN] Startup move-to-neutral failed: {e}. Continuing anyway.")
        time.sleep(0.5)
        input("Press [Enter] to start real-robot evaluation...")

        t = 0
        prev_policy_action: Optional[np.ndarray] = None
        while t < args.num_timesteps:
            raw_obs = _read_obs_retry()

            if args.show_image and "full_image" in raw_obs:
                bgr = cv2.cvtColor(np.asarray(raw_obs["full_image"]), cv2.COLOR_RGB2BGR)
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
                flat_hint_size=args.client_image_size,
                warned_missing=warned_missing,
            )

            # bootstrap history
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

            exec_steps = min(act_exec_horizon, action_seq.shape[0], args.num_timesteps - t)
            for i in range(exec_steps):
                action = np.asarray(action_seq[i], dtype=np.float32).reshape(-1).copy()
                if action.shape[0] != 7:
                    raise RuntimeError(
                        f"Expected 7D policy action for WidowX step_action, got shape {tuple(action.shape)}"
                    )
                if args.gripper_value is not None:
                    action[6] = float(np.clip(args.gripper_value, 0.0, 1.0))

                if args.no_pitch_roll and action.shape[0] >= 5:
                    action[3] = 0.0
                    action[4] = 0.0
                if args.no_yaw and action.shape[0] >= 6:
                    action[5] = 0.0

                interp_start = prev_policy_action if prev_policy_action is not None else action
                interp_actions = _interpolate_actions(
                    start_action=interp_start,
                    end_action=action,
                    substeps=interp_substeps,
                )
                for robot_action in interp_actions:
                    robot_step_start = time.time()
                    widowx_client.step_action(robot_action, blocking=args.blocking)
                    if not args.blocking:
                        dt = time.time() - robot_step_start
                        if dt < robot_step_duration:
                            time.sleep(robot_step_duration - dt)

                prev_policy_action = action

                # Save visualization frame (camera_0).
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
                retries=max(1, min(5, args.neutral_retries)),
                retry_interval_s=args.startup_poll_interval,
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
        fps = float(policy_hz)
        imageio.mimsave(str(out_path), saved_frames, fps=fps)
        print(f"[INFO] Saved rollout video: {out_path}")


if __name__ == "__main__":
    main()
