"""LightningModule for training a policy with supervised losses."""

from __future__ import annotations

import copy
import logging
import traceback  # Added
from typing import Any, Dict, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torchmetrics as tm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.ema import EMAModel

log = logging.getLogger(__name__)


class PolicyTraining(pl.LightningModule):
    """Generic LightningModule for behavior cloning style policy training.

    Expects batches shaped as (observations, actions) or Dictionaries.
    Delegates loss computation to the policy if available (required for BFN/Diffusion).
    """

    def __init__(
        self,
        policy: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig | None = None,
        compile: bool = False,
        datamodule: Any | None = None,
        ema: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("datamodule",))

        self.policy = instantiate(policy)
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.ema_cfg = ema

        # Metrics
        self.train_loss_metric = tm.MeanMetric()
        self.val_loss_metric = tm.MeanMetric()
        self.test_loss_metric = tm.MeanMetric()

        if compile:
            self.policy = torch.compile(self.policy)

        # Optional EMA model
        self.ema_model: EMAModel | None = None
        self._ema_backup_state: Dict[str, torch.Tensor] | None = None
        if self.ema_cfg is not None:
            ema_cfg_dict = OmegaConf.to_container(self.ema_cfg, resolve=True)
            self.ema_model = EMAModel(self.policy, **ema_cfg_dict)

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (after init, before train).

        Used to attach the normalizer from the DataModule to the Policy.
        """
        # Check if trainer exists and has a datamodule
        if hasattr(self, "trainer") and self.trainer.datamodule is not None:
            datamodule = self.trainer.datamodule
            # Look for 'get_normalizer' (common in Diffusion Policy codebases)
            # or public attribute 'normalizer'
            normalizer = None
            if hasattr(datamodule, "get_normalizer"):
                normalizer = datamodule.get_normalizer()
            elif hasattr(datamodule, "normalizer"):
                normalizer = datamodule.normalizer

            if normalizer is not None and hasattr(self.policy, "set_normalizer"):
                log.info(
                    f"Attaching normalizer from DataModule to Policy: {type(normalizer)}"
                )
                self.policy.set_normalizer(normalizer)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs, deterministic=False)

    def _compute_loss_fallback(
        self, pred: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Fallback loss calculation for simple regression/classification policies."""
        # Choose loss by action dtype
        if actions.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            if pred.dim() == actions.dim():
                # If pred is already class indices, convert to logits by adding last dim
                pred = pred
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(pred, actions.long())
        else:
            # Attempt to reshape actions if flat vs sequence
            # pred: [B, T, D], actions: [B, T*D]
            if pred.ndim == 3 and actions.ndim == 2:
                B, T, D = pred.shape
                if actions.shape[1] == T * D:
                    actions = actions.view(B, T, D)

            loss = torch.nn.functional.mse_loss(pred, actions)
        return loss

    def _step(self, batch: Any) -> torch.Tensor:
        """Unified step for Train/Val/Test."""

        # 1. Prefer policy-provided loss (Required for BFN / Diffusion)
        compute_loss = getattr(self.policy, "compute_loss", None)
        if callable(compute_loss):
            try:
                # Try passing the whole batch (BFNPolicy/DiffusionPolicy style)
                return compute_loss(batch)
            except TypeError:
                pass  # Incorrect signature
            except NotImplementedError:
                pass  # Not implemented
            except Exception as e:
                # CRITICAL DEBUG: Log the real error if compute_loss crashes!
                log.error(f"Policy.compute_loss failed with error: {e}")
                # traceback.print_exc() # Uncomment to see full trace in logs
                # We DO NOT re-raise here to allow fallback to attempt MSE,
                # but usually if compute_loss fails, fallback fails too.
                # For now, let's raise it to see the root cause.
                raise e

        # 2. Unpack Batch if we reached here
        if isinstance(batch, dict):
            obs = batch["obs"]
            actions = batch["action"]
        elif isinstance(batch, (list, tuple)):
            obs, actions = batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # 3. Try calling compute_loss with unpacked args (Legacy style)
        if callable(compute_loss):
            try:
                return compute_loss(obs, actions)
            except (TypeError, NotImplementedError):
                pass

        # 4. Fallback: Simple Forward -> MSE/CrossEntropy
        # (Not suitable for BFN/Diffusion, but good for simple MLPs)
        pred = self.policy(obs, deterministic=False)
        return self._compute_loss_fallback(pred, actions)

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        loss = self._step(batch)
        self.train_loss_metric.update(loss.detach())

        # Determine batch size for logging
        bs = 0
        if isinstance(batch, dict):
            bs = list(batch.values())[0].shape[0]
        elif isinstance(batch, (list, tuple)):
            bs = batch[0].shape[0]

        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self._step(batch)
        self.val_loss_metric.update(loss.detach())

        bs = 0
        if isinstance(batch, dict):
            bs = list(batch.values())[0].shape[0]
        elif isinstance(batch, (list, tuple)):
            bs = batch[0].shape[0]

        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self._step(batch)
        self.test_loss_metric.update(loss.detach())

        bs = 0
        if isinstance(batch, dict):
            bs = list(batch.values())[0].shape[0]
        elif isinstance(batch, (list, tuple)):
            bs = batch[0].shape[0]

        self.log("test/loss", loss, batch_size=bs)
        return loss

    def on_train_epoch_end(self):
        self.log("train/loss_epoch", self.train_loss_metric.compute(), prog_bar=True)
        self.train_loss_metric.reset()

    def on_validation_epoch_end(self):
        self.log("val/loss_epoch", self.val_loss_metric.compute(), prog_bar=True)
        self.val_loss_metric.reset()

    def on_test_epoch_end(self):
        self.log("test/loss_epoch", self.test_loss_metric.compute(), prog_bar=True)
        self.test_loss_metric.reset()

    def configure_optimizers(self):
        """Configures optimizers and schedulers, handling Lightning-specific args."""
        opt = instantiate(self.optimizer_cfg, self.policy.parameters())

        if self.lr_scheduler_cfg is None:
            return {"optimizer": opt}

        # --- FIX: Safe Parameter Extraction ---
        # Convert to a mutable container to pop keys safely
        cfg = OmegaConf.to_container(self.lr_scheduler_cfg, resolve=True)

        # Extract keys that Lightning needs but PyTorch schedulers don't accept
        interval = cfg.pop("interval", "epoch")
        frequency = cfg.pop("frequency", 1)
        monitor = cfg.pop("monitor", "val/loss_epoch")

        # The 'name' key is sometimes used for logic but is not a scheduler arg
        _ = cfg.pop("name", None)

        # Instantiate scheduler with remaining args (e.g. T_max, eta_min, _target_)
        scheduler = instantiate(cfg, opt)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
            "monitor": monitor,
        }

        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}

    def configure_callbacks(self):
        return []

    # ---------------- EMA Utilities ---------------- #
    def _maybe_update_ema(self):
        if self.ema_model is not None:
            self.ema_model.step(self.policy)

    def _swap_in_ema_weights(self):
        if self.ema_model is None:
            return
        self._ema_backup_state = copy.deepcopy(self.policy.state_dict())
        self.policy.load_state_dict(
            self.ema_model.averaged_model.state_dict(), strict=False
        )

    def _restore_policy_weights(self):
        if self.ema_model is None or self._ema_backup_state is None:
            return
        self.policy.load_state_dict(self._ema_backup_state, strict=False)
        self._ema_backup_state = None

    def on_validation_epoch_start(self):
        self._swap_in_ema_weights()

    def on_validation_epoch_end(self):
        self._restore_policy_weights()

    def on_test_epoch_start(self):
        self._swap_in_ema_weights()

    def on_test_epoch_end(self):
        self._restore_policy_weights()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )
        self._maybe_update_ema()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_model is not None:
            checkpoint["ema_state"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        ema_state = checkpoint.get("ema_state")
        if ema_state is not None:
            if self.ema_model is None:
                # Recreate with defaults if config is absent
                self.ema_model = EMAModel(self.policy)
            self.ema_model.load_state_dict(ema_state)
