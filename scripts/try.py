#!/usr/bin/env python3
"""Offline sanity test for trained BFN/Diffusion checkpoints on real PushT data.

Examples:
    python scripts/test_trained_model.py \
        --checkpoint /scr2/zhaoyang/diffusion_real_pusht.ckpt \
        --config /scr2/zhaoyang/condBFNPol/config/benchmark_bfn_pusht_real.yaml \
        --dataset-path /scr2/zhaoyang/BFN_data/pusht_real \
        --split train \
        --num-batches 10 \
        --batch-size 16 \
        --device cuda

    python scripts/test_trained_model.py \
        --checkpoint /path/to/epoch=0100-val_loss=0.123.ckpt \
        --config /scr2/zhaoyang/condBFNPol/config/benchmark_diffusion_pusht_real.yaml \
        --dataset-path /scr2/zhaoyang/BFN_data/pusht_real
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make local project modules importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [PROJECT_ROOT, PROJECT_ROOT / "src" / "diffusion-policy"]:
    _sp = str(_p)
    if _p.exists() and _sp not in sys.path:
        sys.path.insert(0, _sp)


def _force_project_utils_package() -> None:
    """Ensure top-level `utils` resolves to this repository's utils package."""
    project_utils_dir = PROJECT_ROOT / "utils"
    init_py = project_utils_dir / "__init__.py"
    if not init_py.is_file():
        return

    existing = sys.modules.get("utils")
    existing_file = str(getattr(existing, "__file__", "") or "")
    existing_paths = [str(p) for p in (getattr(existing, "__path__", []) or [])]

    project_utils_str = str(project_utils_dir.resolve())
    points_to_project = (
        existing is not None
        and (
            project_utils_str in existing_file
            or any(project_utils_str in p for p in existing_paths)
        )
    )
    if points_to_project:
        return

    spec = importlib.util.spec_from_file_location(
        "utils",
        str(init_py),
        submodule_search_locations=[str(project_utils_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)


_force_project_utils_package()

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained checkpoint and test output quality on PushT real dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file path (.ckpt) or run directory containing checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Config YAML path. If omitted, use checkpoint embedded cfg. "
            "Typical values: "
            "/scr2/zhaoyang/condBFNPol/config/benchmark_bfn_pusht_real.yaml or "
            "/scr2/zhaoyang/condBFNPol/config/benchmark_diffusion_pusht_real.yaml"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/scr2/zhaoyang/BFN_data/pusht_real",
        help="Path to real PushT dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Dataset split for testing.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of dataloader batches to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used in test dataloader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=None,
        help=(
            "Optional quality threshold. If set, quality check passes only when "
            "mean action MSE <= max-mse."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save a JSON report.",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help=(
            "Directory to save visualization figures. "
            "Default: ./test_outputs/<ckpt_name>_<timestamp>"
        ),
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization generation.",
    )
    parser.add_argument(
        "--hist-max-points",
        type=int,
        default=200000,
        help="Max points used in histogram plot (random subsample if too large).",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="Prefer ema_model weights if checkpoint contains them.",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Force using model weights even if ema_model exists.",
    )
    parser.set_defaults(use_ema=True)
    return parser.parse_args()


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def resolve_checkpoint_path(path_or_dir: str) -> Path:
    p = Path(path_or_dir).expanduser()
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")

    latest_ckpt = p / "checkpoints" / "latest.ckpt"
    if latest_ckpt.is_file():
        return latest_ckpt

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
    return candidates[-1]


def load_cfg(args: argparse.Namespace, payload: Dict[str, Any]) -> Any:
    if args.config is not None:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = payload.get("cfg", None)
        if cfg is None:
            raise KeyError(
                "Checkpoint has no embedded cfg. Please provide --config explicitly."
            )
    return copy.deepcopy(cfg)


def pick_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def instantiate_policy(cfg: Any, payload: Dict[str, Any], use_ema: bool) -> Tuple[Any, str]:
    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    if use_ema and isinstance(state_dicts.get("ema_model"), dict):
        candidates.append(("ema_model", state_dicts["ema_model"]))
    if isinstance(state_dicts.get("model"), dict):
        candidates.append(("model", state_dicts["model"]))
    if not candidates:
        raise KeyError("Checkpoint has no state_dicts['model'] or state_dicts['ema_model'].")

    def _clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            (k.replace("_orig_mod.", "") if isinstance(k, str) else k): v
            for k, v in state_dict.items()
        }

    def _load_state_dict_flexible(module: Any, state_dict: Dict[str, Any]) -> None:
        load_fn = module.load_state_dict
        supports_strict = False
        try:
            supports_strict = "strict" in inspect.signature(load_fn).parameters
        except Exception:
            supports_strict = False

        if supports_strict:
            try:
                load_fn(state_dict, strict=False)
                return
            except TypeError:
                # Some custom implementations expose a non-standard signature.
                pass

        load_fn(state_dict)

    def _extract_prefixed_state(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        pfx = prefix + "."
        for k, v in state.items():
            if isinstance(k, str) and k.startswith(pfx):
                out[k[len(pfx):]] = v
        return out

    def _try_component_load(state: Dict[str, Any]) -> bool:
        # This handles custom policy checkpoints where full policy state loading
        # is not directly compatible with policy.load_state_dict(...).
        if not hasattr(policy, "obs_encoder") or not hasattr(policy, "model"):
            return False

        cleaned = _clean_state_dict(state)

        # Format A: nested dict, e.g. {"obs_encoder": {...}, "model": {...}, ...}
        nested_obs = cleaned.get("obs_encoder")
        nested_model = cleaned.get("model")
        nested_norm = cleaned.get("normalizer")
        if isinstance(nested_obs, dict) and isinstance(nested_model, dict):
            _load_state_dict_flexible(policy.obs_encoder, _clean_state_dict(nested_obs))
            _load_state_dict_flexible(policy.model, _clean_state_dict(nested_model))
            if hasattr(policy, "normalizer") and isinstance(nested_norm, dict):
                _load_state_dict_flexible(policy.normalizer, _clean_state_dict(nested_norm))
            return True

        # Format B: flat dict with prefixes, e.g. "obs_encoder.xxx", "model.xxx"
        obs_pref = _extract_prefixed_state(cleaned, "obs_encoder")
        model_pref = _extract_prefixed_state(cleaned, "model")
        norm_pref = _extract_prefixed_state(cleaned, "normalizer")
        if obs_pref and model_pref:
            _load_state_dict_flexible(policy.obs_encoder, obs_pref)
            _load_state_dict_flexible(policy.model, model_pref)
            if hasattr(policy, "normalizer") and norm_pref:
                _load_state_dict_flexible(policy.normalizer, norm_pref)
            return True

        # Format C: payload stores components separately in state_dicts.
        obs_sep = state_dicts.get("obs_encoder")
        model_sep = state_dicts.get("model")
        norm_sep = state_dicts.get("normalizer", payload.get("normalizer"))

        if isinstance(obs_sep, dict):
            obs_sep_clean = _clean_state_dict(obs_sep)
            obs_sep_pref = _extract_prefixed_state(obs_sep_clean, "obs_encoder")
            if obs_sep_pref:
                obs_sep_clean = obs_sep_pref
        else:
            obs_sep_clean = {}

        # candidate itself might be pure model weights, use that first
        model_candidate_clean = cleaned
        model_candidate_pref = _extract_prefixed_state(model_candidate_clean, "model")
        if model_candidate_pref:
            model_candidate_clean = model_candidate_pref

        if not model_candidate_clean and isinstance(model_sep, dict):
            model_candidate_clean = _clean_state_dict(model_sep)
            model_sep_pref = _extract_prefixed_state(model_candidate_clean, "model")
            if model_sep_pref:
                model_candidate_clean = model_sep_pref

        if obs_sep_clean and model_candidate_clean:
            _load_state_dict_flexible(policy.obs_encoder, obs_sep_clean)
            _load_state_dict_flexible(policy.model, model_candidate_clean)
            if hasattr(policy, "normalizer") and isinstance(norm_sep, dict):
                norm_clean = _clean_state_dict(norm_sep)
                norm_pref = _extract_prefixed_state(norm_clean, "normalizer")
                if norm_pref:
                    norm_clean = norm_pref
                _load_state_dict_flexible(policy.normalizer, norm_clean)
            return True

        return False

    last_error: Exception | None = None
    for source, state in candidates:
        cleaned_state = _clean_state_dict(state)
        try:
            _load_state_dict_flexible(policy, cleaned_state)
            return policy, source
        except Exception as exc:
            last_error = exc
            try:
                if _try_component_load(cleaned_state):
                    print(f"[INFO] Loaded checkpoint via component fallback from '{source}'.")
                    return policy, source
            except Exception as exc_component:
                last_error = exc_component

    raise RuntimeError("Failed to load policy state_dict from checkpoint.") from last_error


def build_dataset(cfg: Any, dataset_path: str, split: str):
    cfg.dataset_path = dataset_path
    cfg.task.dataset.dataset_path = dataset_path
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if split == "val":
        if not hasattr(dataset, "get_validation_dataset"):
            raise AttributeError("Dataset has no get_validation_dataset() for split=val.")
        dataset = dataset.get_validation_dataset()
    return dataset


def build_dataloader(
    cfg: Any,
    dataset,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    loader_cfg = cfg.val_dataloader if split == "val" else cfg.dataloader
    loader_kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    loader_kwargs["batch_size"] = batch_size
    loader_kwargs["shuffle"] = False
    loader_kwargs["num_workers"] = num_workers
    if num_workers == 0:
        loader_kwargs["persistent_workers"] = False
    return DataLoader(dataset, **loader_kwargs)


def align_pred_and_gt(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    n_obs_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_action.ndim == 3 and gt_action.ndim == 3:
        if pred_action.shape[-1] != gt_action.shape[-1]:
            raise ValueError(
                f"Action dim mismatch: pred {pred_action.shape[-1]} vs gt {gt_action.shape[-1]}"
            )
        if pred_action.shape[1] == gt_action.shape[1]:
            return pred_action, gt_action
        start = max(int(n_obs_steps) - 1, 0)
        end = start + pred_action.shape[1]
        if end <= gt_action.shape[1]:
            return pred_action, gt_action[:, start:end, :]
        t_common = min(pred_action.shape[1], gt_action.shape[1])
        return pred_action[:, :t_common, :], gt_action[:, :t_common, :]

    pred_flat = pred_action.reshape(pred_action.shape[0], -1)
    gt_flat = gt_action.reshape(gt_action.shape[0], -1)
    d_common = min(pred_flat.shape[1], gt_flat.shape[1])
    return pred_flat[:, :d_common], gt_flat[:, :d_common]


def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def to_numpy_detached(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def save_visualizations(
    metrics: List[Dict[str, Any]],
    error_chunks: List[np.ndarray],
    error_by_dim_chunks: List[np.ndarray],
    first_vis_payload: Dict[str, Any],
    vis_dir: Path,
    hist_max_points: int,
) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip visualization. {exc}")
        return []

    vis_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    # 1) Metrics curves.
    batch_idx = [m["batch_idx"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    mses = [m["action_mse"] for m in metrics]
    l1s = [m["action_l1"] for m in metrics]
    finite_ratios = [m["finite_ratio"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    axes[0].plot(batch_idx, losses, marker="o")
    axes[0].set_title("Loss per Batch")
    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch_idx, mses, marker="o", color="tab:orange")
    axes[1].set_title("Action MSE per Batch")
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(batch_idx, l1s, marker="o", color="tab:green")
    axes[2].set_title("Action L1 per Batch")
    axes[2].set_xlabel("Batch")
    axes[2].set_ylabel("L1")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(batch_idx, finite_ratios, marker="o", color="tab:red")
    axes[3].set_title("Finite Ratio per Batch")
    axes[3].set_xlabel("Batch")
    axes[3].set_ylabel("Finite ratio")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    p = vis_dir / "metrics_curves.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(str(p))

    # 2) Error histogram.
    if error_chunks:
        all_errors = np.concatenate(error_chunks, axis=0)
        if all_errors.size > hist_max_points:
            idx = np.random.choice(all_errors.size, size=hist_max_points, replace=False)
            all_errors = all_errors[idx]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(all_errors, bins=80, alpha=0.85, color="tab:blue")
        ax.set_title("Action Error Distribution (pred - gt)")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = vis_dir / "action_error_hist.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(str(p))

    # 3) Error by action dimension.
    if error_by_dim_chunks:
        err_by_dim = np.concatenate(error_by_dim_chunks, axis=0)  # [N, D]
        d = err_by_dim.shape[1]
        fig, axes = plt.subplots(d, 1, figsize=(10, 2.3 * d), sharex=False)
        if d == 1:
            axes = [axes]
        for i in range(d):
            axes[i].hist(err_by_dim[:, i], bins=60, alpha=0.85)
            axes[i].set_title(f"Error Distribution - Action dim {i}")
            axes[i].set_xlabel("Error")
            axes[i].set_ylabel("Count")
            axes[i].grid(True, alpha=0.3)
        fig.tight_layout()
        p = vis_dir / "action_error_by_dim.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(str(p))

    # 4) Pred vs GT trajectory for first sample.
    pred_first = first_vis_payload.get("pred_action")
    gt_first = first_vis_payload.get("gt_action")
    if isinstance(pred_first, np.ndarray) and isinstance(gt_first, np.ndarray):
        if pred_first.ndim == 2 and gt_first.ndim == 2:
            t = np.arange(pred_first.shape[0])
            d = pred_first.shape[1]
            fig, axes = plt.subplots(d, 1, figsize=(10, 2.3 * d), sharex=True)
            if d == 1:
                axes = [axes]
            for i in range(d):
                axes[i].plot(t, gt_first[:, i], label="gt", linewidth=2)
                axes[i].plot(t, pred_first[:, i], label="pred", linewidth=2, linestyle="--")
                axes[i].set_ylabel(f"a[{i}]")
                axes[i].grid(True, alpha=0.3)
                if i == 0:
                    axes[i].legend(loc="best")
            axes[-1].set_xlabel("Action step")
            fig.suptitle("First Sample: Predicted vs Ground Truth Action")
            fig.tight_layout()
            p = vis_dir / "pred_vs_gt_first_sample.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            saved.append(str(p))

    # 5) Observation preview for first sample (if RGB obs exists).
    obs_images: Dict[str, np.ndarray] = first_vis_payload.get("obs_images", {})
    if obs_images:
        keys = list(obs_images.keys())
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for i, k in enumerate(keys):
            img = obs_images[k]
            axes[i].imshow(img, cmap=None if img.ndim == 3 else "gray")
            axes[i].set_title(f"Obs preview: {k}")
            axes[i].axis("off")
        fig.tight_layout()
        p = vis_dir / "observation_preview.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(str(p))

    return saved


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    if not Path(args.dataset_path).expanduser().is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")

    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset:    {args.dataset_path}")
    print("=" * 80)

    payload = torch.load(
        ckpt_path.open("rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = load_cfg(args, payload)
    policy, loaded_source = instantiate_policy(cfg, payload, args.use_ema)

    device = pick_device(args.device)
    policy = policy.to(device)
    policy.eval()

    dataset = build_dataset(cfg, args.dataset_path, args.split)
    dataloader = build_dataloader(
        cfg=cfg,
        dataset=dataset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    n_obs_steps = int(cfg_get(cfg, "n_obs_steps", getattr(policy, "n_obs_steps", 1)))
    n_action_steps = int(cfg_get(cfg, "n_action_steps", getattr(policy, "n_action_steps", -1)))
    action_dim = int(cfg.shape_meta.action.shape[0])

    metrics: List[Dict[str, Any]] = []
    error_chunks: List[np.ndarray] = []
    error_by_dim_chunks: List[np.ndarray] = []
    first_vis_payload: Dict[str, Any] = {
        "obs_images": {},
        "pred_action": None,
        "gt_action": None,
    }
    basic_pass = True

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs = batch["obs"]
            gt_action = batch["action"]

            loss_val = float(policy.compute_loss(batch).item())
            result = policy.predict_action(obs)
            if "action" not in result:
                raise KeyError("policy.predict_action output missing key 'action'.")
            pred_action = result["action"]

            pred_eval, gt_eval = align_pred_and_gt(
                pred_action=pred_action,
                gt_action=gt_action,
                n_obs_steps=n_obs_steps,
            )
            error_tensor = (pred_eval - gt_eval).float()
            mse_val = float(F.mse_loss(pred_eval, gt_eval).item())
            l1_val = float(F.l1_loss(pred_eval, gt_eval).item())
            finite_ratio = float(torch.isfinite(pred_action).float().mean().item())

            error_chunks.append(to_numpy_detached(error_tensor.reshape(-1)))
            if error_tensor.ndim == 3:
                error_by_dim_chunks.append(to_numpy_detached(error_tensor.reshape(-1, error_tensor.shape[-1])))

            shape_ok = (
                pred_action.ndim == 3
                and pred_action.shape[-1] == action_dim
                and (
                    n_action_steps < 0
                    or pred_action.shape[1] == n_action_steps
                )
            )
            finite_ok = finite_ratio == 1.0
            batch_pass = shape_ok and finite_ok
            basic_pass = basic_pass and batch_pass

            metrics.append(
                {
                    "batch_idx": batch_idx,
                    "loss": loss_val,
                    "action_mse": mse_val,
                    "action_l1": l1_val,
                    "finite_ratio": finite_ratio,
                    "pred_shape": list(pred_action.shape),
                    "gt_shape": list(gt_action.shape),
                    "pred_min": float(pred_action.min().item()),
                    "pred_max": float(pred_action.max().item()),
                    "pred_abs_mean": float(pred_action.abs().mean().item()),
                    "shape_ok": bool(shape_ok),
                    "finite_ok": bool(finite_ok),
                    "batch_pass": bool(batch_pass),
                }
            )

            if first_vis_payload["pred_action"] is None and pred_eval.ndim == 3 and gt_eval.ndim == 3:
                first_vis_payload["pred_action"] = to_numpy_detached(pred_eval[0])
                first_vis_payload["gt_action"] = to_numpy_detached(gt_eval[0])
                if isinstance(obs, dict):
                    for key, value in obs.items():
                        if not isinstance(value, torch.Tensor):
                            continue
                        if value.ndim == 5 and value.shape[2] in (1, 3):
                            obs_t_idx = min(max(n_obs_steps - 1, 0), value.shape[1] - 1)
                            img = value[0, obs_t_idx].detach().float().cpu()
                            img = img.permute(1, 2, 0).numpy()
                            if img.shape[-1] == 1:
                                img = img[..., 0]
                            img = np.clip(img, 0.0, 1.0)
                            first_vis_payload["obs_images"][key] = img

    summary = {
        "checkpoint": str(ckpt_path),
        "config": str(args.config) if args.config is not None else "<from_checkpoint>",
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "device": str(device),
        "loaded_state_dict": loaded_source,
        "num_batches_evaluated": len(metrics),
        "loss_mean": safe_mean([m["loss"] for m in metrics]),
        "loss_std": float(np.std([m["loss"] for m in metrics])) if metrics else float("nan"),
        "action_mse_mean": safe_mean([m["action_mse"] for m in metrics]),
        "action_mse_std": float(np.std([m["action_mse"] for m in metrics])) if metrics else float("nan"),
        "action_l1_mean": safe_mean([m["action_l1"] for m in metrics]),
        "action_l1_std": float(np.std([m["action_l1"] for m in metrics])) if metrics else float("nan"),
        "finite_ratio_mean": safe_mean([m["finite_ratio"] for m in metrics]),
        "basic_pass": bool(basic_pass and len(metrics) > 0),
        "max_mse_threshold": args.max_mse,
        "quality_pass": None,
        "overall_pass": None,
        "visualizations": [],
    }

    quality_pass = None
    if args.max_mse is not None:
        quality_pass = summary["action_mse_mean"] <= float(args.max_mse)
    summary["quality_pass"] = quality_pass
    summary["overall_pass"] = (
        summary["basic_pass"] if quality_pass is None else bool(summary["basic_pass"] and quality_pass)
    )

    if not args.no_vis:
        if args.vis_dir is not None:
            vis_dir = Path(args.vis_dir).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = Path.cwd() / "test_outputs" / f"{ckpt_path.stem}_{ts}"
        saved_figs = save_visualizations(
            metrics=metrics,
            error_chunks=error_chunks,
            error_by_dim_chunks=error_by_dim_chunks,
            first_vis_payload=first_vis_payload,
            vis_dir=vis_dir,
            hist_max_points=args.hist_max_points,
        )
        summary["visualizations"] = saved_figs
        if saved_figs:
            print("\nSaved visualizations:")
            for p in saved_figs:
                print(f"- {p}")

    print()
    print("Test Summary")
    print("-" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"summary": summary, "batch_metrics": metrics}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to: {out_path}")

    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



"""
python -u scripts/try.py \
        --checkpoint /ariesdv0/zhaoyang/BFN_outputs/2026.02.27/08.39.14_diffusion_real_image_seed42/checkpoints/epoch=0056-val_loss=0.056.ckpt \
        --config /ariesdv0/zhaoyang/condBFNPol/config/benchmark_diffusion_pusht_real.yaml \
        --dataset-path /ariesdv0/zhaoyang/BFN_data/pusht_real \
        --split train \
        --num-batches 20 \
        --batch-size 64 \
        --device cuda \
        --vis-dir /ariesdv0/zhaoyang/test_bfn/diffusion \
        --output-json /ariesdv0/zhaoyang/test_bfn/diffusion/diffusion.json


python -u scripts/try.py \
        --checkpoint /ariesdv0/zhaoyang/BFN_outputs/2026.02.27/08.39.49_bfn_real_image_seed42/checkpoints/epoch=0088-val_loss=0.006.ckpt \
        --config /ariesdv0/zhaoyang/condBFNPol/config/benchmark_bfn_pusht_real.yaml \
        --dataset-path /ariesdv0/zhaoyang/BFN_data/pusht_real \
        --split train \
        --num-batches 20 \
        --batch-size 64 \
        --device cuda \
        --vis-dir /ariesdv0/zhaoyang/test_bfn/bfn \
        --output-json /ariesdv0/zhaoyang/test_bfn/bfn/bfn.json

"""