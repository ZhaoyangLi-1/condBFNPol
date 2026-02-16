"""
python eval_widowx.py \
  --checkpoint_path /data/BFN_data/checkpoints/diffusion_real_pusht.ckpt \
  --device cuda --im_size 480 --num_timesteps 120 --video_save_path /tmp/widowx_eval \
  --ip 127.0.0.1 --port 5556
"""

import os
import sys
import time
import traceback
import socket
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import dill
import hydra
import imageio
import numpy as np
import torch
from absl import app, flags, logging
from omegaconf import OmegaConf
from PIL import Image


def _extend_pythonpath_for_widowx() -> None:
    """Allow running this script without pre-configuring PYTHONPATH."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2] if len(script_path.parents) > 2 else script_path.parent
    workspace_root = script_path.parents[3] if len(script_path.parents) > 3 else script_path.parent
    home_dir = Path.home()
    candidate_paths = []

    env_project_root = os.environ.get("BFNPOL_ROOT")
    if env_project_root:
        env_project_root = Path(env_project_root)
        candidate_paths.append(env_project_root)
        candidate_paths.append(env_project_root / "src")
        candidate_paths.append(env_project_root / "src/diffusion-policy")

    env_dp_root = os.environ.get("DIFFUSION_POLICY_ROOT")
    if env_dp_root:
        candidate_paths.append(Path(env_dp_root))

    env_widowx_root = os.environ.get("WIDOWX_ENVS_ROOT")
    if env_widowx_root:
        candidate_paths.append(Path(env_widowx_root))

    env_src = os.environ.get("MULTICAM_SERVER_SRC")
    if env_src:
        candidate_paths.append(Path(env_src))

    candidate_paths.extend(
        [
            project_root,
            project_root / "src",
            project_root / "src/diffusion-policy",
            workspace_root / "bridge_data_robot/widowx_envs",
            workspace_root / "Documents/bridge_data_robot/widowx_envs",
            workspace_root / "bridge_data_robot/widowx_envs/multicam_server/src",
            workspace_root / "Documents/bridge_data_robot/widowx_envs/multicam_server/src",
            home_dir / "bridge_data_robot/widowx_envs",
            home_dir / "Documents/bridge_data_robot/widowx_envs",
            home_dir / "bridge_data_robot/widowx_envs/multicam_server/src",
            home_dir / "Documents/bridge_data_robot/widowx_envs/multicam_server/src",
        ]
    )

    existing_candidates: List[str] = []
    seen = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.is_dir():
            existing_candidates.append(candidate_str)

    for candidate_str in reversed(existing_candidates):
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_extend_pythonpath_for_widowx()

try:
    from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("widowx_envs"):
        raise ModuleNotFoundError(
            "widowx_envs is not importable. Install it, set WIDOWX_ENVS_ROOT, or "
            "place it at bridge_data_robot/widowx_envs."
        ) from exc
    raise

np.set_printoptions(suppress=True)
logging.set_verbosity(logging.WARNING)

OmegaConf.register_new_resolver("eval", eval, replace=True)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_path", None, "Path(s) to diffusion/BFN checkpoint .ckpt", required=True
)
flags.DEFINE_string("ip", "127.0.0.1", "WidowX service IP (use IPv4; localhost may resolve to ::1)")
flags.DEFINE_integer("port", 5556, "WidowX service port")
flags.DEFINE_string("device", "cuda", "Torch device (e.g. cuda, cuda:0, cpu)")
flags.DEFINE_bool("use_ema", True, "Load EMA policy when available")
flags.DEFINE_integer("im_size", 96, "WidowX env image size")
flags.DEFINE_string("video_save_path", None, "Directory to save rollout videos")
flags.DEFINE_string("goal_image_path", None, "Optional goal image path")
flags.DEFINE_integer("num_timesteps", 120, "Rollout horizon per episode")
flags.DEFINE_bool("blocking", False, "Use blocking controller")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal EEF xyz")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial EEF xyz")
flags.DEFINE_integer("act_exec_horizon", 1, "How many predicted actions to execute per cycle")
flags.DEFINE_bool("deterministic", False, "Use deterministic sampling by fixed torch seed")
flags.DEFINE_integer("seed", 0, "Random seed for numpy/torch")
flags.DEFINE_integer(
    "override_diffusion_steps",
    0,
    "If > 0 and policy has num_inference_steps, override it for eval.",
)
flags.DEFINE_integer(
    "override_bfn_steps",
    0,
    "If > 0 and policy has n_timesteps, override it for eval.",
)
flags.DEFINE_float("step_duration", 0.2, "Control step duration in seconds")
flags.DEFINE_integer(
    "env_action_dim",
    7,
    "Action dim expected by WidowX env. Policy output is truncated/padded to this dim.",
)
flags.DEFINE_bool("no_pitch_roll", False, "Zero out action[3:5]")
flags.DEFINE_bool("no_yaw", False, "Zero out action[5]")
flags.DEFINE_bool("sticky_gripper", True, "Enable sticky gripper state machine")
flags.DEFINE_integer(
    "sticky_gripper_num_steps", 1, "Consecutive toggle steps to change gripper state"
)
flags.DEFINE_float("gripper_close_threshold", 0.5, "Gripper close threshold")
flags.DEFINE_multi_string(
    "camera_topic", ["/blue/image_raw"], "Camera topics passed to WidowX env"
)

WORKSPACE_BOUNDS = np.array(
    [[0.1, -0.15, -0.1, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]], dtype=np.float64
)
FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
Use 0.2s step duration for non-blocking evaluation.
"""


def _status_name(status: int) -> str:
    for name in ("SUCCESS", "NO_CONNECTION", "EXECUTION_FAILURE", "NOT_INITIALIZED"):
        if getattr(WidowXStatus, name) == status:
            return name
    return str(status)


def _wait_for_obs(
    client: WidowXClient, timeout_s: float = 30.0, poll_s: float = 0.1
) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    obs = client.get_observation()
    while obs is None and time.time() < deadline:
        time.sleep(poll_s)
        obs = client.get_observation()
    if obs is None:
        raise TimeoutError(f"Timed out waiting for WidowX observation after {timeout_s:.1f}s")
    return obs


def _convert_client_obs(raw_obs: Dict[str, Any], im_size: int) -> Dict[str, Any]:
    image = raw_obs.get("external_img", None)
    if image is None:
        image = raw_obs.get("image", None)
    if image is None:
        raise KeyError(f"Observation has no image key. Keys: {list(raw_obs.keys())}")

    image = _image_to_hwc_uint8(image, im_size)
    image = _resize_hwc_uint8(image, im_size, im_size)
    state = np.asarray(
        raw_obs.get("state", np.zeros((7,), dtype=np.float32)), dtype=np.float32
    ).reshape(-1)
    return {"image": image, "state": state}


def _move_to_xyz(
    client: WidowXClient, xyz: np.ndarray, duration: float = 1.5, blocking: bool = True
) -> bool:
    pose = np.asarray([xyz[0], xyz[1], xyz[2], 0.0, 1.57, 0.0], dtype=np.float64)
    status = client.move(pose, duration=duration, blocking=blocking)
    if status != WidowXStatus.SUCCESS:
        logging.warning("WidowX move failed with status=%s", _status_name(status))
        return False
    return True


def _client_step(
    client: WidowXClient, env_action: np.ndarray, blocking: bool, im_size: int
) -> Tuple[Optional[Dict[str, Any]], bool]:
    status = client.step_action(np.asarray(env_action, dtype=np.float64), blocking=blocking)
    if status != WidowXStatus.SUCCESS:
        logging.warning("WidowX step_action failed with status=%s", _status_name(status))
        return None, True

    raw_obs = client.get_observation()
    if raw_obs is None:
        logging.warning("WidowX observation returned None after step_action.")
        return None, True
    return _convert_client_obs(raw_obs, im_size), False


def _to_container(x: Any) -> Any:
    if OmegaConf.is_config(x):
        return OmegaConf.to_container(x, resolve=True)
    return x


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_shape_meta(cfg: Any, policy: Any) -> Dict[str, Any]:
    shape_meta = _obj_get(cfg, "shape_meta", None)
    if shape_meta is None:
        task = _obj_get(cfg, "task", None)
        shape_meta = _obj_get(task, "shape_meta", None)
    if shape_meta is None:
        shape_meta = _obj_get(policy, "shape_meta", None)
    shape_meta = _to_container(shape_meta)
    if not isinstance(shape_meta, dict):
        shape_meta = {}
    if "obs" not in shape_meta:
        shape_meta["obs"] = {
            "image": {"shape": [3, FLAGS.im_size, FLAGS.im_size], "type": "rgb"},
            "state": {"shape": [7], "type": "low_dim"},
        }
    if "action" not in shape_meta:
        shape_meta["action"] = {"shape": [FLAGS.env_action_dim]}
    return shape_meta


def _extract_action_dim(shape_meta: Dict[str, Any], policy: Any) -> int:
    action_meta = _obj_get(shape_meta, "action", {})
    shape = _obj_get(action_meta, "shape", None)
    if shape is not None:
        shape = tuple(int(v) for v in list(shape))
        if len(shape) == 1:
            return int(shape[0])
    action_dim = _obj_get(policy, "action_dim", None)
    if action_dim is not None:
        return int(action_dim)
    return FLAGS.env_action_dim


def _image_to_hwc_uint8(image: Any, fallback_size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 1:
        if arr.size == 3 * fallback_size * fallback_size:
            arr = arr.reshape(3, fallback_size, fallback_size)
        else:
            side = int(round((arr.size / 3) ** 0.5))
            if side * side * 3 != arr.size:
                raise ValueError(f"Cannot infer image shape from flat array size={arr.size}")
            arr = arr.reshape(3, side, side)
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image ndim={arr.ndim}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape={arr.shape}")

    if arr.dtype != np.uint8:
        max_val = float(np.max(arr)) if arr.size else 0.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_hwc_uint8(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return image
    pil = Image.fromarray(image)
    pil = pil.resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(pil)


def _extract_low_dim(raw_obs: Dict[str, Any], key: str, raw_state: np.ndarray) -> np.ndarray:
    if key in raw_obs:
        arr = np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr

    key_lower = key.lower()
    if "agent_pos" in key_lower or ("pos" in key_lower and "pose" not in key_lower):
        return raw_state[:2]
    if "eef" in key_lower or "pose" in key_lower or "state" in key_lower or "proprio" in key_lower:
        return raw_state
    return raw_state


def _build_policy_obs(
    raw_obs: Dict[str, Any], shape_meta: Dict[str, Any], fallback_size: int
) -> Dict[str, np.ndarray]:
    obs_meta = _obj_get(shape_meta, "obs", {})
    obs_meta = _to_container(obs_meta)
    if not isinstance(obs_meta, dict):
        obs_meta = {}

    raw_image = _image_to_hwc_uint8(raw_obs["image"], fallback_size)
    raw_state = np.asarray(raw_obs.get("state", []), dtype=np.float32).reshape(-1)
    out: Dict[str, np.ndarray] = {}

    for key, attr in obs_meta.items():
        attr = _to_container(attr)
        if not isinstance(attr, dict):
            continue
        obs_type = str(attr.get("type", "low_dim"))
        shape = tuple(int(v) for v in list(attr.get("shape", [])))

        if obs_type == "rgb":
            src = raw_obs.get(key, raw_image)
            img = _image_to_hwc_uint8(src, fallback_size)
            if len(shape) != 3:
                raise ValueError(f"RGB key {key} expects shape [C,H,W], got {shape}")
            c, h, w = shape
            img = _resize_hwc_uint8(img, h, w)
            if img.shape[-1] != c:
                if c == 1:
                    img = np.mean(img, axis=-1, keepdims=True).astype(np.uint8)
                elif c == 3 and img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                else:
                    raise ValueError(f"Channel mismatch for {key}: got {img.shape[-1]}, expect {c}")
            img = img.astype(np.float32) / 255.0
            out[key] = np.moveaxis(img, -1, 0)
        else:
            vec = _extract_low_dim(raw_obs, key, raw_state)
            if len(shape) == 0:
                out[key] = vec.astype(np.float32)
            else:
                dim = int(np.prod(shape))
                dst = np.zeros((dim,), dtype=np.float32)
                n_copy = min(dim, vec.size)
                if n_copy > 0:
                    dst[:n_copy] = vec[:n_copy]
                out[key] = dst.reshape(shape)

    if not out:
        out = {
            "image": np.moveaxis(raw_image.astype(np.float32) / 255.0, -1, 0),
            "state": raw_state.astype(np.float32),
        }
    return out


def _stack_obs(obs_hist: deque) -> Dict[str, np.ndarray]:
    keys = obs_hist[0].keys()
    return {k: np.stack([obs[k] for obs in obs_hist], axis=0) for k in keys}


def _to_torch_obs_batch(
    obs_np: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(v).unsqueeze(0).to(device=device, dtype=torch.float32)
        for k, v in obs_np.items()
    }


def _extract_action_array(policy_output: Any) -> np.ndarray:
    action = policy_output
    if isinstance(policy_output, dict):
        if "action" in policy_output:
            action = policy_output["action"]
        else:
            action = None
            for value in policy_output.values():
                if torch.is_tensor(value):
                    action = value
                    break
            if action is None:
                raise RuntimeError("Policy output dict has no tensor action field.")

    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    else:
        action = np.asarray(action)

    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    return np.asarray(action, dtype=np.float64)


def _make_policy_name(checkpoint_path: str, cfg: Any) -> str:
    ckpt = Path(checkpoint_path)
    cfg_name = _obj_get(cfg, "name", None)
    if cfg_name:
        return f"{cfg_name}-{ckpt.stem}"
    if ckpt.parent.name == "checkpoints":
        return f"{ckpt.parent.parent.name}-{ckpt.stem}"
    return ckpt.stem


def _dedup_name(name: str, existing: Dict[str, Any]) -> str:
    if name not in existing:
        return name
    idx = 2
    while f"{name}_{idx}" in existing:
        idx += 1
    return f"{name}_{idx}"


def _load_policy(checkpoint_path: str, device: torch.device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    cfg_training = _obj_get(cfg, "training", None)
    cfg_use_ema = bool(_obj_get(cfg_training, "use_ema", False))
    use_ema = (
        FLAGS.use_ema
        and cfg_use_ema
        and hasattr(workspace, "ema_model")
        and workspace.ema_model is not None
    )

    policy = workspace.ema_model if use_ema else workspace.model
    if policy is None:
        raise RuntimeError(f"Policy is None for checkpoint: {checkpoint_path}")

    policy.to(device)
    policy.eval()

    if FLAGS.override_diffusion_steps > 0 and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(FLAGS.override_diffusion_steps)
        if hasattr(policy, "noise_scheduler"):
            policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    if FLAGS.override_bfn_steps > 0 and hasattr(policy, "n_timesteps"):
        policy.n_timesteps = int(FLAGS.override_bfn_steps)

    shape_meta = _extract_shape_meta(cfg, policy)
    n_obs_steps = int(_obj_get(policy, "n_obs_steps", _obj_get(cfg, "n_obs_steps", 1)))
    action_dim = _extract_action_dim(shape_meta, policy)
    infer_count = 0

    @torch.no_grad()
    def get_action(obs_dict_np: Dict[str, np.ndarray]) -> np.ndarray:
        nonlocal infer_count
        obs_torch = _to_torch_obs_batch(obs_dict_np, device)

        if FLAGS.deterministic:
            if device.type == "cuda":
                dev_idx = device.index if device.index is not None else torch.cuda.current_device()
                rng_ctx = torch.random.fork_rng(devices=[dev_idx])
            else:
                rng_ctx = torch.random.fork_rng()

            with rng_ctx:
                torch.manual_seed(FLAGS.seed + infer_count)
                out = policy.predict_action(obs_torch) if hasattr(policy, "predict_action") else policy(obs_torch)
        else:
            out = policy.predict_action(obs_torch) if hasattr(policy, "predict_action") else policy(obs_torch)

        infer_count += 1
        return _extract_action_array(out)

    return {
        "cfg": cfg,
        "policy": policy,
        "shape_meta": shape_meta,
        "n_obs_steps": max(n_obs_steps, 1),
        "action_dim": action_dim,
        "get_action": get_action,
    }


def _candidate_ips(user_ip: str, port: int) -> List[str]:
    """
    Build a list of IPs to try. Handles cases where the service binds to IPv6 ::1,
    or where localhost resolves differently between host/container.
    """
    cands: List[str] = []

    def add(x: Optional[str]) -> None:
        if x and x not in cands:
            cands.append(x)

    add(user_ip)
    add("127.0.0.1")
    add("localhost")  # may resolve to ::1
    add("::1")        # IPv6 loopback

    try:
        hostname = socket.gethostname()
        for _fam, _typ, _pro, _can, sockaddr in socket.getaddrinfo(hostname, None):
            add(sockaddr[0])
    except Exception:
        pass

    try:
        add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass

    return cands


def _connect_widowx_with_retry(
    env_params: dict,
    image_size: int,
    total_timeout_s: float = 120.0,
    per_ip_timeout_s: float = 30.0,
) -> WidowXClient:
    """
    Robust connect that retries BOTH:
      (1) WidowXClient(...) constructor (ActionClient connect)  <-- your crash happens here
      (2) client.init(...)
    so you can wait for server startup without crashing.
    """
    ips = _candidate_ips(FLAGS.ip, FLAGS.port)

    t_start = time.time()
    last_err: Optional[str] = None

    for ip in ips:
        print(f"[widowx] Trying {ip}:{FLAGS.port} ...")
        ip_start = time.time()
        attempt = 0

        while True:
            if time.time() - t_start > total_timeout_s:
                raise RuntimeError(
                    f"WidowX connect timeout after {total_timeout_s:.1f}s. "
                    f"Tried IPs={ips}. Last error: {last_err}"
                )

            if time.time() - ip_start > per_ip_timeout_s:
                print(f"[widowx] Giving up on {ip} after {per_ip_timeout_s:.1f}s. Last error: {last_err}")
                break

            attempt += 1
            sleep_s = min(2.0, 0.1 * (2 ** (attempt - 1)))

            # Retry constructor (ActionClient connect)
            try:
                client = WidowXClient(host=ip, port=FLAGS.port)
            except Exception as e:
                last_err = f"constructor failed: {repr(e)}"
                logging.warning("[widowx] %s (attempt %d). Sleeping %.2fs", last_err, attempt, sleep_s)
                time.sleep(sleep_s)
                continue

            # Retry init
            try:
                st = client.init(env_params, image_size=int(image_size))
                if st == WidowXStatus.SUCCESS:
                    print(f"[widowx] Connected on {ip}:{FLAGS.port}")
                    return client

                last_err = f"init status={_status_name(st)}"
                logging.warning("[widowx] %s (attempt %d). Sleeping %.2fs", last_err, attempt, sleep_s)

            except Exception as e:
                last_err = f"init exception: {repr(e)}"
                logging.warning("[widowx] %s (attempt %d). Sleeping %.2fs", last_err, attempt, sleep_s)

            if hasattr(client, "close"):
                try:
                    client.close()
                except Exception:
                    pass

            time.sleep(sleep_s)

    raise RuntimeError(f"WidowX connect failed for all candidates {ips}. Last error: {last_err}")


def _warmup_camera_or_die(client: WidowXClient, im_size: int, timeout_s: float = 60.0) -> None:
    """
    Explicitly wait for first *valid* image observation.
    This prevents hanging later at goal capture / rollout if camera never publishes.
    """
    print(f"[widowx] Waiting for first camera frame (timeout {timeout_s:.0f}s)...")
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None

    while time.time() < deadline:
        try:
            raw = client.get_observation()
            if raw is None:
                time.sleep(0.1)
                continue
            _ = _convert_client_obs(raw, im_size)  # validate
            print("[widowx] First camera frame received.")
            return
        except Exception as e:
            last_err = repr(e)
            time.sleep(0.2)

    raise RuntimeError(
        "Camera did not produce a valid frame within timeout.\n"
        f"  camera_topic(s): {FLAGS.camera_topic}\n"
        f"  last_error: {last_err}\n"
        "Likely causes: wrong topic name, no publisher, compressed-vs-raw mismatch, "
        "or camera driver not accessible inside container."
    )


def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    device = torch.device(FLAGS.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available, fallback to CPU.")
        device = torch.device("cpu")

    policies: Dict[str, Dict[str, Any]] = {}
    for checkpoint_path in FLAGS.checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        policy_data = _load_policy(checkpoint_path, device)
        name = _make_policy_name(checkpoint_path, policy_data["cfg"])
        name = _dedup_name(name, policies)
        policies[name] = policy_data
        print(
            f"Loaded {name}: n_obs_steps={policy_data['n_obs_steps']}, "
            f"action_dim={policy_data['action_dim']}"
        )

    if FLAGS.initial_eep is not None:
        initial_eep = [float(x) for x in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    if not FLAGS.blocking:
        assert abs(float(FLAGS.step_duration) - 0.2) < 1e-6, STEP_DURATION_MESSAGE

    camera_topics = [{"name": topic} for topic in FLAGS.camera_topic]
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": camera_topics,
            "override_workspace_boundaries": WORKSPACE_BOUNDS.tolist(),
            "move_duration": float(FLAGS.step_duration),
        }
    )
    if start_state is not None:
        env_params["start_state"] = list(start_state)

    # Robust connect (retries constructor + init)
    widowx_client = _connect_widowx_with_retry(env_params, image_size=int(FLAGS.im_size))

    # Optional: require first camera frame early
    _warmup_camera_or_die(widowx_client, im_size=int(FLAGS.im_size), timeout_s=60.0)

    image_goal: Optional[np.ndarray] = None
    if FLAGS.goal_image_path is not None:
        image_goal = np.asarray(
            Image.open(FLAGS.goal_image_path).convert("RGB").resize(
                (FLAGS.im_size, FLAGS.im_size), Image.BILINEAR
            )
        )

    while True:
        if image_goal is None:
            print("Taking a new goal...")
            ch = "y"
        else:
            ch = input("Taking a new goal? [y/n]")

        if ch == "y":
            if FLAGS.goal_eep is not None:
                goal_eep = [float(x) for x in FLAGS.goal_eep]
            else:
                low_bound = WORKSPACE_BOUNDS[0][:3] + 0.03
                high_bound = WORKSPACE_BOUNDS[1][:3] - 0.03
                goal_eep = np.random.uniform(low_bound, high_bound)

            gripper_status = widowx_client.move_gripper(1.0)
            if gripper_status != WidowXStatus.SUCCESS:
                logging.warning("WidowX move_gripper failed with status=%s", _status_name(gripper_status))

            try:
                if not _move_to_xyz(widowx_client, np.asarray(goal_eep, dtype=np.float64), duration=1.5):
                    continue
            except Exception:
                continue

            input("Press [Enter] when ready for taking the goal image. ")
            obs = _convert_client_obs(_wait_for_obs(widowx_client), FLAGS.im_size)
            image_goal = obs["image"]

        if len(policies) == 1:
            policy_idx = 0
            input("Press [Enter] to start.")
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        policy_data = policies[policy_name]
        get_action = policy_data["get_action"]
        shape_meta = policy_data["shape_meta"]
        n_obs_steps = policy_data["n_obs_steps"]

        try:
            reset_status = widowx_client.reset()
            if reset_status != WidowXStatus.SUCCESS:
                logging.warning("WidowX reset failed with status=%s", _status_name(reset_status))
                continue
            obs = _convert_client_obs(_wait_for_obs(widowx_client), FLAGS.im_size)
        except Exception:
            continue

        try:
            if FLAGS.initial_eep is not None:
                initial_eep = [float(x) for x in FLAGS.initial_eep]
                if not _move_to_xyz(widowx_client, np.asarray(initial_eep, dtype=np.float64), duration=1.5):
                    continue
                obs = _convert_client_obs(_wait_for_obs(widowx_client), FLAGS.im_size)
        except Exception:
            continue

        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        obs_hist = deque(maxlen=n_obs_steps)
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        episode_truncated = False
        last_env_action = np.zeros((FLAGS.env_action_dim,), dtype=np.float64)
        if FLAGS.env_action_dim > 0:
            last_env_action[-1] = 1.0

        try:
            if hasattr(policy_data["policy"], "reset"):
                policy_data["policy"].reset()

            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + FLAGS.step_duration or FLAGS.blocking:
                    image_obs = _image_to_hwc_uint8(obs["image"], FLAGS.im_size)
                    policy_obs = _build_policy_obs(obs, shape_meta, FLAGS.im_size)

                    if n_obs_steps > 1:
                        if len(obs_hist) == 0:
                            obs_hist.extend([policy_obs] * n_obs_steps)
                        else:
                            obs_hist.append(policy_obs)
                        policy_obs_stacked = _stack_obs(obs_hist)
                    else:
                        policy_obs_stacked = {k: np.expand_dims(v, axis=0) for k, v in policy_obs.items()}

                    last_tstep = time.time()
                    actions = get_action(policy_obs_stacked)
                    if actions.ndim == 1:
                        actions = actions[None]

                    n_exec = min(int(FLAGS.act_exec_horizon), actions.shape[0])
                    for i_exec in range(n_exec):
                        policy_action = np.asarray(actions[i_exec], dtype=np.float64).reshape(-1)
                        env_action = last_env_action.copy()

                        n_copy = min(policy_action.size, env_action.size)
                        if n_copy > 0:
                            env_action[:n_copy] = policy_action[:n_copy]

                        noise_std = np.zeros_like(env_action)
                        n_std = min(noise_std.size, FIXED_STD.size)
                        noise_std[:n_std] = FIXED_STD[:n_std]
                        env_action += np.random.normal(0.0, noise_std)

                        if FLAGS.sticky_gripper and env_action.size >= 7:
                            gripper_idx = env_action.size - 1
                            if n_copy > gripper_idx:
                                change = env_action[gripper_idx] < FLAGS.gripper_close_threshold
                                if change != is_gripper_closed:
                                    num_consecutive_gripper_change_actions += 1
                                else:
                                    num_consecutive_gripper_change_actions = 0

                                if num_consecutive_gripper_change_actions >= FLAGS.sticky_gripper_num_steps:
                                    is_gripper_closed = not is_gripper_closed
                                    num_consecutive_gripper_change_actions = 0

                            env_action[gripper_idx] = 0.0 if is_gripper_closed else 1.0

                        if FLAGS.no_pitch_roll and env_action.size >= 5:
                            env_action[3] = 0.0
                            env_action[4] = 0.0
                        if FLAGS.no_yaw and env_action.size >= 6:
                            env_action[5] = 0.0

                        obs, truncated = _client_step(widowx_client, env_action, FLAGS.blocking, FLAGS.im_size)
                        if truncated or obs is None:
                            episode_truncated = True
                            break
                        last_env_action = env_action

                        if image_goal is not None:
                            goals.append(image_goal)
                        images.append(image_obs)

                        t += 1
                        if t >= FLAGS.num_timesteps:
                            break

                    if episode_truncated:
                        break

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)

        if FLAGS.video_save_path is not None and len(images) > 0 and len(goals) == len(images):
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{FLAGS.sticky_gripper_num_steps}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / FLAGS.step_duration * 3)
            print(f"Saved video: {save_path}")


if __name__ == "__main__":
    app.run(main)
