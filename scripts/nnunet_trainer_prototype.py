"""
Custom nnUNet trainers that insert PrototypeFusionBlock3D at the encoder bottleneck.

Two classes are registered here so nnUNet can discover them via -tr flag:
  nnUNetTrainerPrototype          — standard nnUNetPlans  OR  nnUNetResEncUNetMPlans
  MedNeXtTrainerM_kernel3Prototype — MedNeXtTrainerM_kernel3 + prototype block

Deployment: the K8s job copies this file into nnunetv2's trainer variants directory
before calling nnUNetv2_train, so nnUNet's class-discovery mechanism finds the classes.

The PrototypeFusionBlock3D is imported from /workspace/prototype_nnunet_evidence/,
which is baked into the Docker image.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch import autocast

# Make prototype_module_3d importable from /workspace/prototype_nnunet_evidence/
_PROTO_DIR = "/workspace/prototype_nnunet_evidence"
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

from prototype_module_3d import PrototypeFusionBlock3D  # noqa: E402

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


# ── Shared mixin ──────────────────────────────────────────────────────────────

class _PrototypeMixin:
    """
    Inserts PrototypeFusionBlock3D at encoder.stages[-1] (bottleneck) via a
    forward hook and adds a prototype auxiliary loss alongside nnUNet's Dice+CE.

    Hyperparameters (override via subclass attributes):
      prototype_loss_weight  — weight of the auxiliary prototype CE loss
      prototypes_per_class   — number of prototype vectors per class
      prototype_num_heads    — attention heads in PrototypeCrossAttention3D
    """

    prototype_loss_weight: float = 0.1
    prototypes_per_class: int = 8
    prototype_num_heads: int = 4

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_prototype(self) -> None:
        feature_dim = self._probe_bottleneck_dim()
        num_classes = self.label_manager.num_segmentation_heads

        self.prototype_block = PrototypeFusionBlock3D(
            feature_dim=feature_dim,
            num_classes=num_classes,
            prototypes_per_class=self.prototypes_per_class,
            num_heads=self.prototype_num_heads,
        ).to(self.device)

        print(
            f"[Prototype] feature_dim={feature_dim}  num_classes={num_classes}  "
            f"prototypes_per_class={self.prototypes_per_class}"
        )

        # Add prototype parameters as a new optimizer param group
        self.optimizer.add_param_group({
            "params": list(self.prototype_block.parameters()),
            "lr": self.initial_lr,
            "weight_decay": self.weight_decay,
        })

        self._hook_cache: dict = {}
        bottleneck = self.network.encoder.stages[-1]
        self._hook_handle = bottleneck.register_forward_hook(self._prototype_hook)

    def _probe_bottleneck_dim(self) -> int:
        """Run one dummy forward to learn the bottleneck channel count."""
        cache: dict = {}

        def _probe(module, inp, out):
            cache["channels"] = out.shape[1]

        handle = self.network.encoder.stages[-1].register_forward_hook(_probe)
        self.network.eval()
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.num_input_channels,
                *self.configuration_manager.patch_size,
                device=self.device,
            )
            try:
                self.network(dummy)
            except Exception:
                pass
        handle.remove()
        return cache.get("channels", 320)

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _prototype_hook(self, module, inp, output):
        proto_out = self.prototype_block(output)
        self._hook_cache["class_evidence"] = proto_out["class_evidence"]
        return proto_out["fused_features"]

    # ── Auxiliary loss ────────────────────────────────────────────────────────

    def _compute_prototype_loss(self, class_evidence, target) -> torch.Tensor:
        """
        Supervise prototype class evidence at bottleneck resolution.
        class_evidence: (B, num_classes, D, H, W)
        target: list[Tensor] (deep supervision) or Tensor, integer labels
        """
        seg = target[0] if isinstance(target, (list, tuple)) else target
        B, C, D, H, W = class_evidence.shape
        seg_ds = (
            F.interpolate(seg.float(), size=(D, H, W), mode="nearest")
            .long()
            .squeeze(1)
        )
        return F.cross_entropy(class_evidence, seg_ds, ignore_index=-1)

    # ── Training step ─────────────────────────────────────────────────────────

    def _prototype_train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, (list, tuple)):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self._hook_cache.clear()
        self.optimizer.zero_grad(set_to_none=True)

        ctx = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with ctx:
            output = self.network(data)
            main_loss = self.loss(output, target)

            proto_loss = torch.tensor(0.0, device=self.device)
            if "class_evidence" in self._hook_cache:
                proto_loss = self._compute_prototype_loss(
                    self._hook_cache["class_evidence"], target
                )

            total_loss = main_loss + self.prototype_loss_weight * proto_loss

        all_params = (
            list(self.network.parameters())
            + list(self.prototype_block.parameters())
        )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.optimizer.step()

        return {"loss": total_loss.detach().cpu().numpy()}

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        proto_path = filename.replace(".pth", "_prototype.pth")
        torch.save(self.prototype_block.state_dict(), proto_path)

    def load_checkpoint(self, filename: str) -> None:
        super().load_checkpoint(filename)
        proto_path = filename.replace(".pth", "_prototype.pth")
        if os.path.exists(proto_path):
            self.prototype_block.load_state_dict(
                torch.load(proto_path, map_location="cpu", weights_only=True)
            )


# ── Concrete trainers ──────────────────────────────────────────────────────────

class nnUNetTrainerPrototype(_PrototypeMixin, nnUNetTrainer):
    """
    Standard nnUNet trainer + prototype block.
    Works with both nnUNetPlans and nnUNetResEncUNetMPlans.

    Usage:
      nnUNetv2_train 002 3d_fullres FOLD -tr nnUNetTrainerPrototype --npz
      nnUNetv2_train 002 3d_fullres FOLD -tr nnUNetTrainerPrototype -p nnUNetResEncUNetMPlans --npz
    """

    def initialize(self) -> None:
        super().initialize()
        self._setup_prototype()

    def train_step(self, batch: dict) -> dict:
        return self._prototype_train_step(batch)


# MedNeXt trainer is optional — only registered if mednext is installed
try:
    from mednext import MedNeXtTrainerM_kernel3 as _MedNeXtBase
except ImportError:
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mednext_trainer import (
            MedNeXtTrainerM_kernel3 as _MedNeXtBase,
        )
    except ImportError:
        _MedNeXtBase = None

if _MedNeXtBase is not None:

    class MedNeXtTrainerM_kernel3Prototype(_PrototypeMixin, _MedNeXtBase):
        """
        MedNeXt-M (kernel=3) trainer + prototype block.

        Usage:
          nnUNetv2_train 002 3d_fullres FOLD -tr MedNeXtTrainerM_kernel3Prototype --npz
        """

        def initialize(self) -> None:
            super().initialize()
            self._setup_prototype()

        def train_step(self, batch: dict) -> dict:
            return self._prototype_train_step(batch)

else:
    print(
        "[Prototype] WARNING: mednext not found — "
        "MedNeXtTrainerM_kernel3Prototype is not available."
    )
