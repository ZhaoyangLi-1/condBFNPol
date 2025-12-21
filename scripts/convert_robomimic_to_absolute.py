"""Convert robomimic HDF5 demonstrations to absolute position control.

The script builds absolute target actions from the *next* observation
(`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`) so policies can
train on position control instead of velocity deltas.

Example:
    python scripts/convert_robomimic_to_absolute.py \\
        --input data/robomimic/lift/ph/lift_ph.hdf5 \\
        --output data/robomimic/lift/ph/lift_ph_abs.hdf5

Supported tasks: Lift-Ph, Square-Ph, Square-Mh (and other robomimic tasks
with the same observation keys).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


DEFAULT_KEYS = dict(
    pos="robot0_eef_pos",
    quat="robot0_eef_quat",
    grip="robot0_gripper_qpos",
)


def infer_action_components(
    obs_grp, keys=DEFAULT_KEYS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.array(obs_grp[keys["pos"]])
    quat = np.array(obs_grp[keys["quat"]])
    grip = np.array(obs_grp[keys["grip"]]).reshape(-1)
    if grip.size == 0:
        grip = np.zeros(1, dtype=np.float32)
    return pos, quat, grip


def build_absolute_actions(obs_grp, T: int, keys=DEFAULT_KEYS) -> np.ndarray:
    """Use next observation as absolute target for step t."""
    actions_abs: List[np.ndarray] = []
    for t in range(T):
        obs_next = obs_grp[str(t + 1)] if str(t + 1) in obs_grp else obs_grp[str(t)]
        pos, quat, grip = infer_action_components(obs_next, keys=keys)
        act = np.concatenate([pos, quat, grip], axis=0).astype(np.float32)
        actions_abs.append(act)
    return np.stack(actions_abs, axis=0)


def copy_structure_and_replace_actions(
    src_file: h5py.File, dst_file: h5py.File, keys=DEFAULT_KEYS
):
    data_grp = src_file["data"]
    out_data = dst_file.require_group("data")

    for ep_name in data_grp.keys():
        ep_src = data_grp[ep_name]
        ep_dst = out_data.require_group(ep_name)

        # Copy everything except actions
        for k in ep_src.keys():
            if k == "actions":
                continue
            ep_src.copy(ep_src[k], ep_dst, name=k)

        obs_grp = ep_dst["observations"]
        T = len(ep_src["actions"])
        abs_actions = build_absolute_actions(obs_grp, T, keys=keys)

        # Match original action shape if needed
        target_dim = ep_src["actions"].shape[1]
        if abs_actions.shape[1] != target_dim:
            if abs_actions.shape[1] > target_dim:
                abs_actions = abs_actions[:, :target_dim]
            else:
                pad = np.zeros((T, target_dim - abs_actions.shape[1]), dtype=np.float32)
                abs_actions = np.concatenate([abs_actions, pad], axis=1)

        ep_dst.create_dataset("actions", data=abs_actions, compression="gzip")

    # Copy attributes and mark control mode
    for k, v in src_file.attrs.items():
        dst_file.attrs[k] = v
    dst_file.attrs["control_mode"] = "absolute_pose"


def main():
    parser = argparse.ArgumentParser(
        description="Convert robomimic HDF5 actions to absolute position control."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input robomimic HDF5.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 path (will be overwritten).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Make a temp copy to preserve any external links; then rewrite actions.
    tmp_path = args.output.with_suffix(".tmp.hdf5")
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copy2(args.input, tmp_path)

    with h5py.File(tmp_path, "r") as src, h5py.File(args.output, "w") as dst:
        copy_structure_and_replace_actions(src, dst, keys=DEFAULT_KEYS)

    tmp_path.unlink(missing_ok=True)
    print(f"Wrote absolute-action dataset to {args.output}")


if __name__ == "__main__":
    main()
