from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


TRAJ_PATTERN = re.compile(r"^traj\d+$")
DEFAULT_REFERENCE_ROOT = Path("/scr2/zhaoyang/pusht_real/real_pusht_20230105")

# Option A: always store 7D actions: [x, y, z, rx, ry, rz, gripper]
# Gripper defaults to "open" if not available.
DEFAULT_GRIPPER_OPEN = 1.0


class Dummy(SimpleNamespace):
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return Dummy


def find_traj_dirs(root: Path) -> List[Path]:
    trajs = [p for p in root.iterdir() if p.is_dir() and TRAJ_PATTERN.match(p.name)]
    trajs.sort(key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    return trajs


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_policy_out(path: Path):
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transform(s) to 6D pose (x, y, z, rotvec).
    T: (..., 4, 4)
    Returns: (..., 6)
    """
    T = np.asarray(T)
    pos = T[..., :3, 3]
    rot = R.from_matrix(T[..., :3, :3]).as_rotvec()
    return np.concatenate([pos, rot], axis=-1)


def compute_pose_vel(T: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Compute 6D pose velocity from transforms and timestamps.
    Returns shape (N, 6) with last velocity repeated.
    """
    T = np.asarray(T)
    timestamps = np.asarray(timestamps)
    n = T.shape[0]
    if n < 2:
        return np.zeros((n, 6), dtype=np.float64)

    pos = T[:, :3, 3]
    dt = np.diff(timestamps)
    dt_safe = np.where(dt == 0, np.nan, dt)

    dpos = (pos[1:] - pos[:-1]) / dt_safe[:, None]

    rot_prev = R.from_matrix(T[:-1, :3, :3])
    rot_next = R.from_matrix(T[1:, :3, :3])
    rot_rel = rot_prev.inv() * rot_next
    ang_vel = rot_rel.as_rotvec() / dt_safe[:, None]

    dpos = np.nan_to_num(dpos)
    ang_vel = np.nan_to_num(ang_vel)

    vel = np.concatenate([dpos, ang_vel], axis=1)
    vel = np.vstack([vel, vel[-1]])
    return vel.astype(np.float64)


def estimate_hz(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return float("nan")
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    dt = np.median(diffs)
    if dt <= 0 or not np.isfinite(dt):
        return float("nan")
    return 1.0 / dt


def make_downsample_indices(length: int, factor: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if factor <= 1:
        return np.arange(length, dtype=np.int64)
    return np.arange(0, length, factor, dtype=np.int64)


def make_target_hz_indices(
    timestamps: np.ndarray,
    target_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to target_hz by selecting nearest source frames on a uniform time grid.

    Returns:
        - selected source indices
        - resampled timestamps on an exact uniform grid
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = timestamps.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.array([0], dtype=np.int64), timestamps.copy()

    dt = 1.0 / float(target_hz)
    start_t = float(timestamps[0])
    end_t = float(timestamps[-1])
    n_target = int(np.floor((end_t - start_t) / dt)) + 1
    target_ts = start_t + np.arange(n_target, dtype=np.float64) * dt

    right_idx = np.searchsorted(timestamps, target_ts, side="left")
    right_idx = np.clip(right_idx, 0, n - 1)
    left_idx = np.maximum(right_idx - 1, 0)
    use_left = np.abs(timestamps[left_idx] - target_ts) <= np.abs(timestamps[right_idx] - target_ts)
    raw_idx = np.where(use_left, left_idx, right_idx)

    selected = []
    last = -1
    for cand in raw_idx.tolist():
        idx = int(cand)
        if idx <= last:
            idx = last + 1
        if idx >= n:
            break
        selected.append(idx)
        last = idx

    if not selected:
        selected = [0]

    indices = np.asarray(selected, dtype=np.int64)
    resampled_timestamps = start_t + np.arange(len(indices), dtype=np.float64) * dt
    return indices, resampled_timestamps


def _as_gripper_series(x, target_len: int) -> np.ndarray:
    """Coerce gripper source to shape (T,) float64 with len alignment."""
    if x is None:
        raise ValueError("gripper source is None")
    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.full((target_len,), float(arr), dtype=np.float64)
    if arr.ndim >= 1:
        flat = arr.astype(np.float64).reshape((arr.shape[0], -1))[:, 0]
        if flat.shape[0] == target_len:
            return flat
        if flat.shape[0] == target_len - 1 and flat.shape[0] > 0:
            return np.concatenate([flat, flat[-1:]], axis=0)
        if flat.shape[0] == 1:
            return np.full((target_len,), float(flat[0]), dtype=np.float64)
        if flat.shape[0] > target_len:
            return flat[:target_len]
        if 1 < flat.shape[0] < target_len:
            pad = np.full((target_len - flat.shape[0],), float(flat[-1]), dtype=np.float64)
            return np.concatenate([flat, pad], axis=0)
    raise ValueError(f"Unsupported gripper shape {arr.shape} for target_len={target_len}")


def get_gripper_from_obs(obs: dict, target_len: int) -> np.ndarray | None:
    """
    Try to infer a per-step gripper "command/state" time series from obs_dict.
    This is intentionally permissive: checks multiple common key names.
    """
    candidate_keys = [
        "gripper",
        "gripper_cmd",
        "gripper_command",
        "gripper_action",
        "gripper_state",
        "gripper_pos",
        "gripper_position",
        "gripper_qpos",
        "eef_gripper",
        "eef_gripper_state",
    ]
    for k in candidate_keys:
        if k in obs:
            try:
                return _as_gripper_series(obs[k], target_len)
            except Exception:
                pass
    return None


def get_gripper_from_policy_out(policy_out, target_len: int) -> np.ndarray | None:
    """
    Try to infer a per-step gripper series from policy_out.
    Supports:
      - dict entries with common gripper keys
      - "actions" with >=7 dims where the 7th dim is gripper
    """
    if policy_out is None:
        return None
    if not (isinstance(policy_out, list) and len(policy_out) > 0):
        return None

    sample = policy_out[0]
    if not isinstance(sample, dict):
        return None

    # 1) If actions contain a 7th dim, use it.
    if "actions" in sample:
        try:
            a0 = np.asarray(sample["actions"])
            if a0.ndim >= 1 and a0.shape[-1] >= 7:
                g = np.stack([np.asarray(p["actions"])[6] for p in policy_out], axis=0)
                return _as_gripper_series(g, target_len)
        except Exception:
            pass

    # 2) Common explicit gripper keys
    candidate_keys = [
        "gripper",
        "gripper_cmd",
        "gripper_command",
        "gripper_action",
        "gripper_state",
        "gripper_open",
        "gripper_close",
    ]
    for k in candidate_keys:
        if k in sample:
            try:
                g = np.stack([np.asarray(p[k]).reshape(-1)[0] for p in policy_out], axis=0)
                return _as_gripper_series(g, target_len)
            except Exception:
                pass

    return None


def extract_action(
    policy_out,
    obs: dict,
    target_len: int,
    default_gripper: float = DEFAULT_GRIPPER_OPEN,
    arm_action_source: str = "auto",
) -> np.ndarray:
    """
    Build action array of shape (target_len, 7):
      [x, y, z, rotvec(3), gripper]

    Arm part:
      - arm_action_source='relative': use policy_out["actions"][:6]
      - arm_action_source='absolute': use policy_out["new_robot_transform"] -> 6D pose
      - arm_action_source='auto': prefer relative deltas when they look like bridge step_action.

    Gripper part:
      - Prefer policy_out gripper (if available)
      - Else use obs-derived gripper series (if available)
      - Else default open (broadcast)
    """
    # -------- arm (6d) --------
    if arm_action_source not in {"auto", "relative", "absolute"}:
        raise ValueError(f"Unsupported arm_action_source: {arm_action_source}")

    if policy_out is None:
        arm6 = np.zeros((target_len, 6), dtype=np.float64)
    else:
        if isinstance(policy_out, list) and len(policy_out) > 0:
            sample = policy_out[0]
            if not isinstance(sample, dict):
                raise ValueError("policy_out format not recognized")

            has_actions = "actions" in sample
            has_tf = "new_robot_transform" in sample

            if arm_action_source == "relative":
                if not has_actions:
                    raise ValueError("arm_action_source='relative' but policy_out has no 'actions' key")
                arm6 = np.stack([np.asarray(p["actions"])[:6] for p in policy_out], axis=0)
            elif arm_action_source == "absolute":
                if not has_tf:
                    raise ValueError("arm_action_source='absolute' but policy_out has no 'new_robot_transform' key")
                Ts = np.stack([p["new_robot_transform"] for p in policy_out], axis=0)
                arm6 = transform_to_pose(Ts)
            else:
                if has_actions:
                    rel6 = np.stack([np.asarray(p["actions"])[:6] for p in policy_out], axis=0)
                    rel_like = (
                        np.max(np.abs(rel6[:, :3])) <= 0.12
                        and np.max(np.abs(rel6[:, 3:6])) <= 0.6
                    )
                    if rel_like or not has_tf:
                        arm6 = rel6
                    else:
                        Ts = np.stack([p["new_robot_transform"] for p in policy_out], axis=0)
                        arm6 = transform_to_pose(Ts)
                elif has_tf:
                    Ts = np.stack([p["new_robot_transform"] for p in policy_out], axis=0)
                    arm6 = transform_to_pose(Ts)
                else:
                    raise ValueError("policy_out format not recognized")
        else:
            raise ValueError("policy_out format not recognized")

    # Length align for arm
    if arm6.shape[0] == target_len:
        arm6 = arm6.astype(np.float64)
    elif arm6.shape[0] == target_len - 1:
        arm6 = np.vstack([arm6, arm6[-1]]).astype(np.float64)
    elif arm6.shape[0] > target_len:
        arm6 = arm6[:target_len].astype(np.float64)
    elif arm6.shape[0] < target_len:
        pad = np.repeat(arm6[-1][None, :], target_len - arm6.shape[0], axis=0)
        arm6 = np.vstack([arm6, pad]).astype(np.float64)

    # -------- gripper (1d) --------
    g = get_gripper_from_policy_out(policy_out, target_len)
    if g is None:
        g = get_gripper_from_obs(obs, target_len)
    if g is None:
        g = np.full((target_len,), float(default_gripper), dtype=np.float64)

    g = g.reshape((target_len, 1)).astype(np.float64)

    # -------- concat --------
    act7 = np.concatenate([arm6, g], axis=1)
    if act7.shape != (target_len, 7):
        raise RuntimeError(f"Internal error: expected (T,7) got {act7.shape}")
    return act7


def detect_image_ext(img_dir: Path) -> str:
    for ext in ("jpg", "png", "jpeg"):
        if (img_dir / f"im_0.{ext}").exists():
            return ext
    for p in img_dir.iterdir():
        if p.is_file() and p.name.startswith("im_"):
            return p.suffix.lstrip(".")
    raise FileNotFoundError(f"No images found in {img_dir}")


def make_video_from_images(
    img_dir: Path,
    out_path: Path,
    fps: int,
    frame_stride: int = 1,
    input_fps: float | None = None,
    frame_indices: np.ndarray | None = None,
) -> None:
    ext = detect_image_ext(img_dir)
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.size == 0:
            raise ValueError(f"Empty frame_indices for {img_dir}")

        with tempfile.TemporaryDirectory(prefix="formalize_frames_", dir=str(out_path.parent)) as tmp_dir:
            tmp_path = Path(tmp_dir)
            for out_idx, src_idx in enumerate(frame_indices.tolist()):
                src = img_dir / f"im_{src_idx}.{ext}"
                if not src.exists():
                    raise FileNotFoundError(f"Missing source frame: {src}")
                dst = tmp_path / f"im_{out_idx}.{ext}"
                os.symlink(src, dst)

            pattern = str(tmp_path / f"im_%d.{ext}")
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
        return

    pattern = str(img_dir / f"im_%d.{ext}")
    src_fps = fps
    if frame_stride > 1:
        if input_fps is not None and np.isfinite(input_fps):
            src_fps = input_fps
        else:
            src_fps = fps * frame_stride
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(src_fps),
        "-start_number",
        "0",
        "-i",
        pattern,
    ]
    if frame_stride > 1:
        vf = f"select='not(mod(n\\,{frame_stride}))',setpts=N*{frame_stride}/FRAME_RATE/TB"
        cmd += ["-vf", vf, "-r", str(fps)]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "22",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def build_videos(
    traj_dirs: List[Path],
    out_videos_dir: Path,
    fps: int,
    num_videos: int | None,
    duplicate_with_symlink: bool,
    frame_stride: int = 1,
    input_fps: float | None = None,
    frame_indices_per_episode: List[np.ndarray] | None = None,
) -> None:
    for ep_idx, traj_dir in enumerate(tqdm(traj_dirs, desc="Videos", unit="traj")):
        ep_out = out_videos_dir / str(ep_idx)
        ep_out.mkdir(parents=True, exist_ok=True)
        frame_indices = None
        if frame_indices_per_episode is not None:
            frame_indices = frame_indices_per_episode[ep_idx]

        img_dirs = sorted(
            [p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("images")],
            key=lambda p: int(re.findall(r"\d+", p.name)[0]) if re.findall(r"\d+", p.name) else 0,
        )

        for cam_idx, img_dir in enumerate(img_dirs):
            out_path = ep_out / f"{cam_idx}.mp4"
            if out_path.exists():
                continue
            make_video_from_images(
                img_dir,
                out_path,
                fps,
                frame_stride=frame_stride,
                input_fps=input_fps,
                frame_indices=frame_indices,
            )

        if num_videos is not None and len(img_dirs) < num_videos:
            if len(img_dirs) == 0:
                raise RuntimeError(f"No image directories found in {traj_dir}")
            for cam_idx in range(len(img_dirs), num_videos):
                src_cam = cam_idx % len(img_dirs)
                src_path = ep_out / f"{src_cam}.mp4"
                dst_path = ep_out / f"{cam_idx}.mp4"
                if dst_path.exists():
                    continue
                if duplicate_with_symlink:
                    os.symlink(src_path.name, dst_path)
                else:
                    shutil.copyfile(src_path, dst_path)


def parse_default_output(src_root: Path) -> Path:
    metadata_path = src_root / "collection_metadata.json"
    date_tag = "custom"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            date_time = meta.get("date_time", "").strip("_")
            date_tag = date_time.split("_")[0].replace("-", "") or date_tag
        except Exception:
            pass
    return Path("/scr2/zhaoyang/pusht_real") / f"real_pusht_{date_tag}"


def find_episode_video_dirs(videos_dir: Path) -> List[Path]:
    episode_dirs = [p for p in videos_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    episode_dirs.sort(key=lambda p: int(p.name))
    return episode_dirs


def count_image_cameras(traj_dir: Path) -> int:
    img_dirs = [p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("images")]
    return len(img_dirs)


def infer_reference_format(reference_root: Path) -> dict:
    """
    Infer formatting knobs from an existing dataset root:
    - target_hz / fps from replay_buffer.zarr/data/timestamp
    - chunk_len from replay_buffer.zarr/data/timestamp chunks
    - meta_chunk_len from replay_buffer.zarr/meta/episode_ends chunks
    - num_videos from videos/<episode>/*.mp4 count
    """
    spec = {
        "target_hz": None,
        "fps": None,
        "chunk_len": None,
        "meta_chunk_len": None,
        "num_videos": None,
    }

    replay_path = reference_root / "replay_buffer.zarr"
    if replay_path.exists():
        root = zarr.open(str(replay_path), mode="r")
        if "data" in root and "timestamp" in root["data"]:
            ts_arr = root["data"]["timestamp"]
            hz = estimate_hz(ts_arr[:])
            if np.isfinite(hz):
                spec["target_hz"] = float(hz)
                spec["fps"] = int(round(hz))
            if ts_arr.chunks and len(ts_arr.chunks) >= 1:
                spec["chunk_len"] = int(ts_arr.chunks[0])
        if "meta" in root and "episode_ends" in root["meta"]:
            ep_arr = root["meta"]["episode_ends"]
            if ep_arr.chunks and len(ep_arr.chunks) >= 1:
                spec["meta_chunk_len"] = int(ep_arr.chunks[0])

    videos_dir = reference_root / "videos"
    if videos_dir.exists():
        episode_dirs = find_episode_video_dirs(videos_dir)
        if episode_dirs:
            spec["num_videos"] = len(list(episode_dirs[0].glob("*.mp4")))

    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Formalize BFN pusht_real data to diffusion-policy format")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/scr2/zhaoyang/BFN_data/pusht_real_raw"),
        help="Source BFN pusht_real directory",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/scr2/zhaoyang/BFN_data/pusht_real"),
        help="Output directory (defaults based on collection_metadata.json)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Reference formatted dataset to mimic when corresponding args are not provided.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable reference-format inference.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--no-videos", action="store_true", help="Skip video generation")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (default: match data rate after downsampling)")
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Number of video files per episode (duplicate if fewer cameras).",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Duplicate videos by copying instead of symlink",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="Zarr chunk length (time dimension).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor (keep every Nth frame).",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="Target sampling rate in Hz. Resamples each trajectory on a uniform timestamp grid.",
    )
    parser.add_argument(
        "--check-existing-hz",
        action="store_true",
        help="If output exists, report its estimated Hz and exit.",
    )
    parser.add_argument(
        "--default-gripper-open",
        type=float,
        default=DEFAULT_GRIPPER_OPEN,
        help="Default gripper command used when unavailable in policy_out/obs (Option A).",
    )
    parser.add_argument(
        "--arm-action-source",
        type=str,
        default="auto",
        choices=["auto", "relative", "absolute"],
        help=(
            "How to construct the 6D arm action from policy_out. "
            "'relative' matches widowx step_action deltas; "
            "'absolute' uses new_robot_transform poses."
        ),
    )

    args = parser.parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    print(f"Arm action source: {args.arm_action_source}")

    reference_spec = None
    if not args.no_reference:
        if args.reference.exists():
            reference_spec = infer_reference_format(args.reference)
            if args.target_hz is None and args.downsample == 1 and reference_spec["target_hz"] is not None:
                args.target_hz = reference_spec["target_hz"]
            if args.fps is None and args.target_hz is None and reference_spec["fps"] is not None:
                args.fps = int(reference_spec["fps"])
            if args.chunk_len is None and reference_spec["chunk_len"] is not None:
                args.chunk_len = int(reference_spec["chunk_len"])
            print(
                "Reference format inferred from "
                f"{args.reference}: target_hz={reference_spec['target_hz']}, "
                f"fps={reference_spec['fps']}, num_videos={reference_spec['num_videos']}, "
                f"chunk_len={reference_spec['chunk_len']}, meta_chunk_len={reference_spec['meta_chunk_len']}"
            )
        else:
            print(f"Reference path not found, skip inference: {args.reference}")

    if args.target_hz is not None and args.target_hz <= 0:
        raise ValueError("--target-hz must be > 0")
    if args.target_hz is not None and args.downsample != 1:
        raise ValueError("Use either --target-hz or --downsample, not both")
    if args.target_hz is not None and args.fps is None:
        args.fps = int(round(args.target_hz))

    src_root: Path = args.src
    dst_root = args.dst if args.dst is not None else parse_default_output(src_root)

    if dst_root.exists():
        replay_path = dst_root / "replay_buffer.zarr"
        existing_hz = None
        if replay_path.exists():
            try:
                root = zarr.open(str(replay_path), mode="r")
                existing_hz = estimate_hz(root["data"]["timestamp"][:])
            except Exception:
                existing_hz = None
        if args.check_existing_hz:
            msg = f"Existing output: {dst_root}"
            if existing_hz is not None and np.isfinite(existing_hz):
                msg += f" (estimated {existing_hz:.2f} Hz)"
            print(msg)
            return
        if args.overwrite:
            shutil.rmtree(dst_root)
        else:
            hz_msg = ""
            if existing_hz is not None and np.isfinite(existing_hz):
                hz_msg = f" (estimated {existing_hz:.2f} Hz)"
            raise FileExistsError(f"Output directory exists: {dst_root}{hz_msg}")

    traj_dirs = find_traj_dirs(src_root)
    if not traj_dirs:
        raise FileNotFoundError(f"No traj directories found under {src_root}")
    if args.num_videos is None and not args.no_videos:
        src_num_videos = count_image_cameras(traj_dirs[0])
        if src_num_videos > 0:
            args.num_videos = src_num_videos
            print(f"Video camera count inferred from source: num_videos={args.num_videos}")
        elif reference_spec is not None and reference_spec.get("num_videos") is not None:
            args.num_videos = int(reference_spec["num_videos"])
            print(f"Video camera count fallback to reference: num_videos={args.num_videos}")

    all_timestamp = []
    all_joint = []
    all_joint_vel = []
    all_eef_pose = []
    all_eef_pose_vel = []
    all_stage = []
    all_action = []
    episode_ends = []

    total = 0
    resolved_downsample = None
    resolved_src_hz = None
    resolved_eff_hz = None
    frame_indices_per_episode = []

    for traj_dir in tqdm(traj_dirs, desc="Trajs", unit="traj"):
        obs_path = traj_dir / "obs_dict.pkl"
        pol_path = traj_dir / "policy_out.pkl"

        obs = load_pickle(obs_path)
        policy_out = load_policy_out(pol_path) if pol_path.exists() else None

        timestamps_full = np.asarray(obs["time_stamp"], dtype=np.float64)
        qpos = np.asarray(obs["qpos"], dtype=np.float64)
        qvel = np.asarray(obs["qvel"], dtype=np.float64)
        stage = np.asarray(obs.get("task_stage", np.zeros(len(timestamps_full))), dtype=np.int64)
        if stage.ndim == 0:
            stage = np.full(len(timestamps_full), int(stage), dtype=np.int64)
        eef_T = np.asarray(obs["eef_transform"], dtype=np.float64)

        if len(timestamps_full) != len(eef_T):
            raise ValueError(
                f"Length mismatch in {traj_dir}: timestamps {len(timestamps_full)} vs eef_transform {len(eef_T)}"
            )
        if len(qpos) != len(timestamps_full) or len(qvel) != len(timestamps_full) or len(stage) != len(timestamps_full):
            raise ValueError(
                f"Length mismatch in {traj_dir}: "
                f"timestamps {len(timestamps_full)} qpos {len(qpos)} qvel {len(qvel)} stage {len(stage)}"
            )

        if resolved_src_hz is None:
            resolved_src_hz = estimate_hz(timestamps_full)

        if resolved_downsample is None:
            if args.target_hz is not None:
                if resolved_src_hz is None or not np.isfinite(resolved_src_hz):
                    raise ValueError("Unable to estimate source Hz for --target-hz")
                resolved_downsample = 1
                resolved_eff_hz = float(args.target_hz)
                print(
                    f"Target Hz {args.target_hz:.2f} with timestamp resampling "
                    f"(source {resolved_src_hz:.2f} Hz)"
                )
            else:
                resolved_downsample = args.downsample
                if resolved_src_hz is not None and np.isfinite(resolved_src_hz):
                    resolved_eff_hz = resolved_src_hz / resolved_downsample
                    if resolved_downsample > 1:
                        print(
                            f"Downsample {resolved_downsample}x "
                            f"(source {resolved_src_hz:.2f} Hz -> {resolved_eff_hz:.2f} Hz)"
                        )

        # >>> Option A change: action_full is now (T, 7)
        action_full = extract_action(
            policy_out,
            obs=obs,
            target_len=len(timestamps_full),
            default_gripper=float(args.default_gripper_open),
            arm_action_source=str(args.arm_action_source),
        )

        if args.target_hz is not None:
            idx, timestamps = make_target_hz_indices(timestamps_full, args.target_hz)
        else:
            idx = make_downsample_indices(len(timestamps_full), resolved_downsample)
            timestamps = timestamps_full[idx]

        qpos = qpos[idx]
        qvel = qvel[idx]
        stage = stage[idx]
        eef_T = eef_T[idx]
        action = action_full[idx]  # (len(idx), 7)
        frame_indices_per_episode.append(idx)

        eef_pose = transform_to_pose(eef_T)
        eef_pose_vel = compute_pose_vel(eef_T, timestamps)

        all_timestamp.append(timestamps)
        all_joint.append(qpos)
        all_joint_vel.append(qvel)
        all_eef_pose.append(eef_pose)
        all_eef_pose_vel.append(eef_pose_vel)
        all_stage.append(stage)
        all_action.append(action)

        total += len(timestamps)
        episode_ends.append(total)

    timestamp = np.concatenate(all_timestamp, axis=0)
    robot_joint = np.concatenate(all_joint, axis=0)
    robot_joint_vel = np.concatenate(all_joint_vel, axis=0)
    robot_eef_pose = np.concatenate(all_eef_pose, axis=0)
    robot_eef_pose_vel = np.concatenate(all_eef_pose_vel, axis=0)
    stage = np.concatenate(all_stage, axis=0)
    action = np.concatenate(all_action, axis=0)  # (N, 7)

    replay_path = dst_root / "replay_buffer.zarr"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
    max_ep_len = int(np.max(np.diff([0] + episode_ends)))
    chunk_len = args.chunk_len or max_ep_len
    chunk_len = int(max(1, min(chunk_len, len(timestamp))))

    root = zarr.open(str(replay_path), mode="w")
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    data_grp.create_dataset("timestamp", data=timestamp, chunks=(chunk_len,), compressor=compressor)
    data_grp.create_dataset("robot_joint", data=robot_joint, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_joint_vel", data=robot_joint_vel, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_eef_pose", data=robot_eef_pose, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_eef_pose_vel", data=robot_eef_pose_vel, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("stage", data=stage, chunks=(chunk_len,), compressor=compressor)

    # >>> Option A change: write (N, 7) actions
    data_grp.create_dataset("action", data=action, chunks=(chunk_len, 7), compressor=compressor)

    meta_chunk_len = len(episode_ends)
    if reference_spec is not None and reference_spec.get("meta_chunk_len") is not None:
        meta_chunk_len = int(reference_spec["meta_chunk_len"])
    meta_chunk_len = int(max(1, meta_chunk_len))

    meta_grp.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(meta_chunk_len,),
        compressor=None,
    )

    if not args.no_videos:
        video_fps = args.fps
        if video_fps is None:
            if resolved_eff_hz is not None and np.isfinite(resolved_eff_hz):
                video_fps = int(round(resolved_eff_hz))
            elif resolved_src_hz is not None and np.isfinite(resolved_src_hz):
                video_fps = int(round(resolved_src_hz))
            else:
                video_fps = 30
        videos_dir = dst_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        build_videos(
            traj_dirs,
            videos_dir,
            fps=video_fps,
            num_videos=args.num_videos,
            duplicate_with_symlink=not args.no_symlink,
            frame_stride=resolved_downsample,
            input_fps=resolved_src_hz,
            frame_indices_per_episode=frame_indices_per_episode,
        )

    print(f"Done. Output written to {dst_root}")


if __name__ == "__main__":
    main()

"""
python dataset/formalize_data.py --src /scr2/zhaoyang/BFN_data/pusht_real_raw --dst /scr2/zhaoyang/BFN_data/pusht_real --overwrite --target-hz 10  --arm-action-source relative
"""
