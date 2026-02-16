#!/usr/bin/env python3

"""
python eval_widowx.py \
  --ckpt_path xx/xx.ckpt \
  --ip localhost \
  --port 5556 \
  --device cuda \
  --prefer_full_image \
  --act_exec_horizon 8 \
  --robot_action_dim 6 \
  --action_noise_std 0.0 \
  --sticky_gripper_num_steps 0 \
  --num_timesteps 120
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image
except ImportError:
    Image = None

# Add project root and local diffusion_policy to path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, PROJECT_ROOT / "src" / "diffusion-policy"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 0
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}


def _stdin_has_data() -> bool:
    try:
        import select

        return select.select([sys.stdin], [], [], 0)[0] != []
    except Exception:
        return False


def _prep_device(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resize_hwc_float01(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if img.shape[:2] == (target_h, target_w):
        return img

    if cv2 is not None:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if Image is not None:
        pil = Image.fromarray(np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8))
        pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
        return np.asarray(pil).astype(np.float32) / 255.0

    src_h, src_w = img.shape[:2]
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
    return img[y_idx][:, x_idx]


def _rgb_to_bgr_u8(rgb_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return rgb_u8[..., ::-1].copy()


def _is_ckpt_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".ckpt"


def _list_ckpts(path_like: str) -> List[Path]:
    path = Path(path_like)
    if _is_ckpt_file(path):
        return [path]
    if not path.is_dir():
        return []

    ckpts = list(path.glob("*.ckpt"))
    ckpt_dir = path / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts.extend(ckpt_dir.glob("*.ckpt"))

    dedup = sorted(set(ckpts), key=lambda p: p.stat().st_mtime)
    return dedup


def _discover_runs(root_dir: str) -> List[str]:
    root = Path(root_dir)
    if not root.is_dir():
        return []

    runs = set()
    for current, dirnames, filenames in os.walk(root):
        cur = Path(current)

        if cur.name == "checkpoints":
            if any(name.endswith(".ckpt") for name in filenames):
                runs.add(str(cur.parent))
            dirnames[:] = []
            continue

        has_local_ckpt = any(name.endswith(".ckpt") for name in filenames)
        ckpt_subdir = cur / "checkpoints"
        has_ckpt_subdir = ckpt_subdir.is_dir() and any(
            child.suffix == ".ckpt" for child in ckpt_subdir.iterdir() if child.is_file()
        )

        if has_local_ckpt or has_ckpt_subdir:
            runs.add(str(cur))
            dirnames[:] = []

    return sorted(runs)


def _strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _extract_shape_meta(cfg) -> Dict:
    shape_meta = OmegaConf.select(cfg, "shape_meta")
    if shape_meta is None:
        shape_meta = OmegaConf.select(cfg, "task.shape_meta")
    if shape_meta is None:
        raise KeyError("Cannot find shape_meta in checkpoint cfg.")

    shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
    if not isinstance(shape_meta, dict):
        raise TypeError("shape_meta must be a dict.")
    if "obs" not in shape_meta or "action" not in shape_meta:
        raise KeyError("shape_meta must contain obs and action.")
    return shape_meta


def _infer_policy_kind(policy: torch.nn.Module) -> str:
    name = type(policy).__name__.lower()
    if "bfn" in name:
        return "bfn"
    if "diffusion" in name:
        return "diffusion"
    return "policy"


def _load_policy_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    use_ema: bool = True,
):
    try:
        from hydra.utils import instantiate
    except ImportError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'hydra-core'. Install it to instantiate policy configs."
        ) from e

    try:
        import dill as dill_pickle
    except ImportError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'dill'. Install it to load .ckpt files."
        ) from e

    with open(ckpt_path, "rb") as f:
        payload = torch.load(
            f,
            pickle_module=dill_pickle,
            map_location="cpu",
            weights_only=False,
        )

    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(f"Checkpoint missing cfg: {ckpt_path}")

    policy_cfg = OmegaConf.select(cfg, "policy")
    if policy_cfg is None:
        raise KeyError(f"Checkpoint cfg missing policy section: {ckpt_path}")

    policy = instantiate(policy_cfg)

    state_dicts = payload.get("state_dicts", {})
    load_candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    if use_ema and isinstance(state_dicts.get("ema_model"), dict):
        load_candidates.append(("ema_model", state_dicts["ema_model"]))
    if isinstance(state_dicts.get("model"), dict):
        load_candidates.append(("model", state_dicts["model"]))

    if not load_candidates:
        raise KeyError(
            f"Checkpoint has no model/ema_model state_dicts: {ckpt_path}"
        )

    loaded_from = None
    last_error = None
    for source, state in load_candidates:
        try:
            state = _strip_orig_mod_prefix(state)
            policy.load_state_dict(state, strict=False)
            loaded_from = source
            break
        except Exception as e:
            last_error = e

    if loaded_from is None:
        raise RuntimeError(
            f"Failed to load checkpoint weights from {ckpt_path}: {last_error}"
        )

    policy.to(device)
    policy.eval()

    shape_meta = _extract_shape_meta(cfg)
    n_obs_steps = int(getattr(policy, "n_obs_steps", OmegaConf.select(cfg, "n_obs_steps") or 1))
    n_action_steps = int(
        getattr(policy, "n_action_steps", OmegaConf.select(cfg, "n_action_steps") or 1)
    )

    action_shape = shape_meta["action"].get("shape", None)
    if not action_shape:
        raise KeyError("shape_meta.action.shape missing")
    action_dim = int(action_shape[0])

    return {
        "policy": policy,
        "cfg": cfg,
        "shape_meta": shape_meta,
        "n_obs_steps": max(1, n_obs_steps),
        "n_action_steps": max(1, n_action_steps),
        "action_dim": action_dim,
        "kind": _infer_policy_kind(policy),
        "loaded_from": loaded_from,
    }


def _as_hwc_float01(image: np.ndarray, im_size: int) -> np.ndarray:
    arr = np.asarray(image)

    if arr.ndim == 1:
        expected = 3 * im_size * im_size
        if arr.size != expected:
            raise ValueError(
                f"Cannot reshape flat image of size {arr.size} into 3x{im_size}x{im_size}"
            )
        arr = arr.reshape(3, im_size, im_size).transpose(1, 2, 0)
    elif arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        elif arr.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image rank: {arr.ndim}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _select_rgb_source(
    obs: Dict,
    im_size: int,
    target_hw: Tuple[int, int],
    prefer_full_image: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_hw

    full_img = None
    if prefer_full_image and obs.get("full_image") is not None:
        try:
            full_img = _as_hwc_float01(obs["full_image"], im_size)
        except Exception:
            full_img = None

    small_img = None
    if obs.get("image") is not None:
        try:
            small_img = _as_hwc_float01(obs["image"], im_size)
        except Exception:
            small_img = None

    if full_img is None and small_img is None:
        raise RuntimeError("Observation has neither valid full_image nor image")

    if full_img is not None and small_img is not None:
        fh, fw = full_img.shape[:2]
        sh, sw = small_img.shape[:2]
        full_cost = abs(fh - target_h) + abs(fw - target_w)
        small_cost = abs(sh - target_h) + abs(sw - target_w)
        src = full_img if full_cost <= small_cost else small_img
    else:
        src = full_img if full_img is not None else small_img

    if src.shape[:2] != (target_h, target_w):
        src = _resize_hwc_float01(src, (target_h, target_w))

    chw = np.transpose(src, (2, 0, 1)).astype(np.float32)
    u8 = np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8)
    return chw, u8


def _extract_state(obs: Dict) -> np.ndarray:
    state = obs.get("state", None)
    if state is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _extract_lowdim_from_state(state: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    dim = int(np.prod(shape))
    out = np.zeros((dim,), dtype=np.float32)
    take = min(dim, state.shape[0])
    if take > 0:
        out[:take] = state[:take]
    return out.reshape(shape)


def _build_obs_frame(
    raw_obs: Dict,
    obs_shape_meta: Dict,
    im_size: int,
    prefer_full_image: bool,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    frame = {}
    state = _extract_state(raw_obs)
    display_rgb_u8 = None

    for key, attr in obs_shape_meta.items():
        key_type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape", []))

        if key_type == "rgb":
            if len(shape) != 3:
                raise ValueError(f"RGB obs key {key} expects 3D shape, got {shape}")
            _, h, w = shape
            chw, rgb_u8 = _select_rgb_source(
                obs=raw_obs,
                im_size=im_size,
                target_hw=(h, w),
                prefer_full_image=prefer_full_image,
            )
            frame[key] = chw
            if display_rgb_u8 is None:
                display_rgb_u8 = rgb_u8
        elif key_type == "low_dim":
            if not shape:
                raise ValueError(f"Low-dim obs key {key} has empty shape")
            frame[key] = _extract_lowdim_from_state(state, shape)
        else:
            raise ValueError(f"Unsupported obs type {key_type} for key {key}")

    if display_rgb_u8 is None:
        # Fallback for video/preview when shape_meta has no rgb key.
        try:
            img = _as_hwc_float01(raw_obs["full_image"], im_size)
            display_rgb_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        except Exception:
            display_rgb_u8 = None

    return frame, display_rgb_u8


def _stack_obs(frames: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = frames[0].keys()
    return {k: np.stack([f[k] for f in frames], axis=0) for k in keys}


@torch.no_grad()
def _predict_action_sequence(
    policy: torch.nn.Module,
    obs_np: Dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    obs_t = {
        k: torch.from_numpy(v).float().unsqueeze(0).to(device)
        for k, v in obs_np.items()
    }

    result = policy.predict_action(obs_t)

    if isinstance(result, dict):
        if "action" in result:
            action_t = result["action"]
        elif "action_pred" in result:
            action_t = result["action_pred"]
        else:
            tensor_values = [v for v in result.values() if torch.is_tensor(v)]
            if not tensor_values:
                raise RuntimeError("predict_action returned dict without tensor outputs")
            action_t = tensor_values[0]
    elif torch.is_tensor(result):
        action_t = result
    else:
        raise RuntimeError(f"Unsupported predict_action output type: {type(result)}")

    action_np = action_t.detach().cpu().numpy()
    if action_np.ndim == 1:
        action_np = action_np[None]
    elif action_np.ndim == 2:
        # Could be [Ta, Da] or [1, Da], both are acceptable.
        pass
    elif action_np.ndim >= 3:
        # Common case: [B, Ta, Da].
        action_np = action_np[0]
    else:
        raise RuntimeError(f"Unexpected action rank: {action_np.ndim}")

    if action_np.ndim == 1:
        action_np = action_np[None]

    return action_np.astype(np.float32)


def _adapt_action_dim(action: np.ndarray, target_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    if action.size == target_dim:
        return action
    if action.size > target_dim:
        return action[:target_dim].copy()

    out = np.zeros((target_dim,), dtype=np.float32)
    out[: action.size] = action

    # WidowX commonly expects gripper at index 6.
    if target_dim >= 7 and action.size < 7:
        out[6] = 1.0

    return out


def _postprocess_action(
    action: np.ndarray,
    no_pitch_roll: bool,
    no_yaw: bool,
) -> np.ndarray:
    if no_pitch_roll and action.size > 4:
        action[3] = 0.0
        action[4] = 0.0
    if no_yaw and action.size > 5:
        action[5] = 0.0
    return action


def _get_display_bgr(raw_obs: Dict, fallback_rgb_u8: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if raw_obs.get("full_image") is not None:
        try:
            rgb = _as_hwc_float01(raw_obs["full_image"], im_size=96)
            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            return _rgb_to_bgr_u8(rgb_u8)
        except Exception:
            pass

    if fallback_rgb_u8 is not None:
        try:
            return _rgb_to_bgr_u8(fallback_rgb_u8)
        except Exception:
            return None

    return None


def _build_run_list(args) -> List[str]:
    run_items: List[str] = []

    if args.ckpt_path:
        run_items.extend(args.ckpt_path)

    if args.save_dir:
        run_items.extend(args.save_dir)

    if args.runs_root:
        discovered = _discover_runs(args.runs_root)
        if not discovered:
            print(f"[WARN] No valid runs discovered under {args.runs_root}")
        run_items.extend(discovered)

    # Preserve order and deduplicate.
    seen = set()
    unique = []
    for item in run_items:
        resolved = str(Path(item).expanduser())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Real-robot WidowX eval for diffusion/BFN checkpoints"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="+",
        default=None,
        help="One or more run directories containing .ckpt files",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Root directory to recursively discover run directories",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        nargs="+",
        default=None,
        help="Direct checkpoint path(s)",
    )
    parser.add_argument("--im_size", type=int, default=96)
    parser.add_argument("--video_save_path", type=str, default=None)
    parser.add_argument("--num_timesteps", type=int, default=120)
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--initial_eep", type=float, nargs=3, default=[0.3, 0.0, 0.15])
    parser.add_argument("--act_exec_horizon", type=int, default=8)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--show_image", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--policy_index",
        type=int,
        default=None,
        help="Non-interactive run selection index (first iteration only)",
    )
    parser.add_argument(
        "--ckpt_index",
        type=int,
        default=None,
        help="Non-interactive checkpoint selection index (first iteration only)",
    )
    parser.add_argument(
        "--robot_action_dim",
        type=int,
        default=6,
        help="Action dimension sent to WidowX",
    )
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
        help="Gaussian action noise std applied before execution",
    )
    parser.add_argument(
        "--sticky_gripper_num_steps",
        type=int,
        default=STICKY_GRIPPER_NUM_STEPS,
        help="Consecutive toggle count for sticky gripper logic; <=0 disables",
    )
    parser.add_argument("--no_pitch_roll", action="store_true", default=NO_PITCH_ROLL)
    parser.add_argument("--no_yaw", action="store_true", default=NO_YAW)
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Load model weights instead of ema_model weights",
    )
    parser.add_argument(
        "--prefer_full_image",
        dest="prefer_full_image",
        action="store_true",
        help="Prefer full_image over image for RGB preprocessing",
    )
    parser.add_argument(
        "--no_prefer_full_image",
        dest="prefer_full_image",
        action="store_false",
        help="Prefer image over full_image for RGB preprocessing",
    )
    parser.set_defaults(prefer_full_image=True)

    args = parser.parse_args()

    try:
        from experiments.widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
    except ImportError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'experiments.widowx_envs'. Use the environment where WidowX SDK is installed."
        ) from e

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    if args.show_image and cv2 is None:
        print("[WARN] OpenCV is not installed; disabling --show_image")
        args.show_image = False

    device = _prep_device(args.device)
    widowx_client = None

    policy_index_used = False
    ckpt_index_used = False

    while True:
        run_candidates = _build_run_list(args)
        if not run_candidates:
            raise RuntimeError("Please provide --ckpt_path, --save_dir, or --runs_root")

        names = [Path(p).name for p in run_candidates]

        if len(run_candidates) == 1:
            selected_run_idx = 0
        else:
            if args.policy_index is not None and not policy_index_used:
                selected_run_idx = int(args.policy_index)
                policy_index_used = True
                if selected_run_idx < 0 or selected_run_idx >= len(run_candidates):
                    raise ValueError(
                        f"policy_index out of range: {selected_run_idx}, valid [0..{len(run_candidates)-1}]"
                    )
            else:
                print("runs:")
                for i, n in enumerate(names):
                    print(f"{i}) {n}")
                selected_run_idx = int(input("select run: "))

        chosen_run = run_candidates[selected_run_idx]
        ckpt_files = _list_ckpts(chosen_run)
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt found under: {chosen_run}")

        if len(ckpt_files) == 1:
            chosen_ckpt = ckpt_files[0]
        else:
            if args.ckpt_index is not None and not ckpt_index_used:
                ckpt_idx = int(args.ckpt_index)
                ckpt_index_used = True
                if ckpt_idx < 0 or ckpt_idx >= len(ckpt_files):
                    raise ValueError(
                        f"ckpt_index out of range: {ckpt_idx}, valid [0..{len(ckpt_files)-1}]"
                    )
            else:
                print("checkpoints:")
                for i, p in enumerate(ckpt_files):
                    print(f"{i}) {p.name}")
                try:
                    ckpt_idx = int(input("select ckpt (default latest): ") or str(len(ckpt_files) - 1))
                except Exception:
                    ckpt_idx = len(ckpt_files) - 1
                ckpt_idx = max(0, min(ckpt_idx, len(ckpt_files) - 1))

            chosen_ckpt = ckpt_files[ckpt_idx]

        print(f"Loading checkpoint: {chosen_ckpt}")
        picked = _load_policy_from_checkpoint(
            ckpt_path=str(chosen_ckpt),
            device=device,
            use_ema=(not args.no_ema),
        )

        policy = picked["policy"]
        shape_meta = picked["shape_meta"]
        obs_shape_meta = shape_meta["obs"]
        n_obs_steps = picked["n_obs_steps"]

        print(
            f"Loaded {picked['kind']} policy ({type(policy).__name__}), "
            f"weights={picked['loaded_from']}, n_obs_steps={n_obs_steps}, "
            f"n_action_steps={picked['n_action_steps']}, action_dim={picked['action_dim']}"
        )
        print(f"Observation keys: {list(obs_shape_meta.keys())}")

        if widowx_client is None:
            print("Initializing WidowX connection...")
            env_params = WidowXConfigs.DefaultEnvParams.copy()
            env_params.update(ENV_PARAMS)
            env_params["state_state"] = list(np.concatenate([args.initial_eep, [0, 0, 0, 1]]))
            widowx_client = WidowXClient(host=args.ip, port=args.port)
            widowx_client.init(env_params, image_size=args.im_size)
            print("WidowX connection established.")

        widowx_client.go_to_neutral()
        time.sleep(0.5)

        last_tstep = time.time()
        t = 0
        images = []
        obs_hist = deque(maxlen=n_obs_steps)

        is_gripper_closed = False
        consecutive_gripper_change = 0
        reset_requested = False

        if args.show_image:
            obs = widowx_client.get_observation()
            while obs is None:
                print("Waiting for observations...")
                time.sleep(1)
                obs = widowx_client.get_observation()
            bgr = _get_display_bgr(obs, None)
            if bgr is not None:
                cv2.imshow("img_view", bgr)
                cv2.waitKey(100)

        input("Press [Enter] to start evaluation.")

        try:
            while t < args.num_timesteps:
                if time.time() <= last_tstep + STEP_DURATION and not args.blocking:
                    continue

                raw_obs = widowx_client.get_observation()
                if raw_obs is None:
                    print("WARNING: retrying get_observation...")
                    continue

                curr_frame, frame_rgb_u8 = _build_obs_frame(
                    raw_obs=raw_obs,
                    obs_shape_meta=obs_shape_meta,
                    im_size=args.im_size,
                    prefer_full_image=args.prefer_full_image,
                )

                if args.show_image:
                    bgr_img = _get_display_bgr(raw_obs, frame_rgb_u8)
                    if bgr_img is not None:
                        cv2.imshow("img_view", bgr_img)
                        key = cv2.waitKey(10) & 0xFF
                        if key in (ord("r"), ord("R")):
                            reset_requested = True
                            print("[INFO] Reset requested via 'R' key")
                            break
                else:
                    if _stdin_has_data():
                        try:
                            line = sys.stdin.readline().strip()
                            if line.lower() == "r":
                                reset_requested = True
                                print("[INFO] Reset requested via stdin")
                                break
                        except Exception:
                            pass

                if len(obs_hist) == 0:
                    obs_hist.extend([curr_frame] * obs_hist.maxlen)
                else:
                    obs_hist.append(curr_frame)

                model_obs = _stack_obs(list(obs_hist))

                last_tstep = time.time()
                actions = _predict_action_sequence(policy, model_obs, device)
                exec_horizon = min(max(1, args.act_exec_horizon), actions.shape[0])

                for i in range(exec_horizon):
                    action = actions[i].copy()

                    if args.action_noise_std > 0:
                        action += np.random.normal(
                            loc=0.0,
                            scale=args.action_noise_std,
                            size=action.shape,
                        ).astype(np.float32)

                    action = _adapt_action_dim(action, args.robot_action_dim)

                    if args.sticky_gripper_num_steps > 0 and action.size > 6:
                        if (action[6] < 0.5) != is_gripper_closed:
                            consecutive_gripper_change += 1
                        else:
                            consecutive_gripper_change = 0

                        if consecutive_gripper_change >= args.sticky_gripper_num_steps:
                            is_gripper_closed = not is_gripper_closed
                            consecutive_gripper_change = 0

                        action[6] = 0.0 if is_gripper_closed else 1.0

                    action = _postprocess_action(
                        action,
                        no_pitch_roll=args.no_pitch_roll,
                        no_yaw=args.no_yaw,
                    )

                    widowx_client.step_action(action, blocking=args.blocking)

                    if frame_rgb_u8 is not None:
                        images.append(frame_rgb_u8)

                    t += 1
                    if t >= args.num_timesteps:
                        break

        except KeyboardInterrupt:
            print("[INFO] Ctrl+C received; moving robot to neutral...", file=sys.stderr)
            try:
                widowx_client.go_to_neutral()
            except Exception as e:
                print(f"[WARN] Failed to move to neutral: {e}", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(str(e), file=sys.stderr)

        if reset_requested:
            try:
                widowx_client.go_to_neutral()
            except Exception as e:
                print(f"[WARN] Failed to move to neutral on reset: {e}", file=sys.stderr)
            if args.show_image:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            continue

        if args.video_save_path is not None and images:
            try:
                import imageio
            except ImportError as e:
                raise ModuleNotFoundError(
                    "Missing dependency 'imageio'. Install it or disable --video_save_path."
                ) from e

            os.makedirs(args.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                args.video_save_path,
                f"{curr_time}_widowx_eval.mp4",
            )
            imageio.mimsave(save_path, images, fps=max(1.0, 1.0 / STEP_DURATION))
            print(f"Video saved to: {save_path}")

        try:
            choice = input("\nTest completed. Load another policy? [y/n] (default: y): ").strip().lower()
            if choice in ("n", "no"):
                print("Exiting...")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
