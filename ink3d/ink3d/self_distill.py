"""Teacher-generated training labels.

Every training step, the labels stored on disk are replaced wholesale by the
output of one or two frozen teachers. The stored ``inklabels`` are not mixed
in or used as a prior — for this recipe they only matter during validation.

The procedure, per sample:

1. Predict with the primary teacher, optionally averaged over 8 mirror TTAs.
2. If the crop's *raw* intensity statistics say it is bright and flat
   (``mean > mean_hi`` and ``std < std_lo``), average in a second teacher and
   switch to the ensemble threshold.
3. Zero the probability wherever raw intensity is below
   ``input_mask_threshold`` — dark voxels cannot be ink.
4. Binarise at the selected threshold.

Statistics come from the raw, un-normalized, un-augmented crop; using the
normalized copy would make the thresholds meaningless.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import flip_spatial, mirror_axes
from .model import build_model


def load_frozen_model(
    checkpoint: str | Path, device: torch.device, *, in_channels: int = 1,
    out_channels: int = 1,
) -> nn.Module:
    """Load a teacher and put it beyond the reach of training."""
    model = build_model(in_channels=in_channels, out_channels=out_channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("ema", "model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def predict_with_tta(
    model: nn.Module, image: torch.Tensor, *, tta: bool = True
) -> torch.Tensor:
    """Sigmoid probabilities, averaged over mirror variants."""
    variants = mirror_axes(3) if tta else [()]
    total = torch.zeros_like(image, dtype=torch.float32)
    for axes in variants:
        flipped = flip_spatial(image, axes)
        logits = model(flipped)
        probs = torch.sigmoid(logits.float())
        total += flip_spatial(probs, axes)
    return total / len(variants)


class SelfDistillLabeler:
    """Replaces a batch's labels with teacher predictions."""

    def __init__(
        self,
        primary: nn.Module,
        ensemble: nn.Module | None,
        *,
        primary_threshold: float = 0.17647,
        ensemble_threshold: float = 0.15686,
        mean_hi: float = 105.0,
        std_lo: float = 30.0,
        input_mask_threshold: float = 50.0,
        tta: bool = True,
    ) -> None:
        self.primary = primary
        self.ensemble = ensemble
        self.primary_threshold = float(primary_threshold)
        self.ensemble_threshold = float(ensemble_threshold)
        self.mean_hi = float(mean_hi)
        self.std_lo = float(std_lo)
        self.input_mask_threshold = float(input_mask_threshold)
        self.tta = bool(tta)

    @torch.no_grad()
    def __call__(
        self,
        raw_image: torch.Tensor,
        normalized_image: torch.Tensor,
        raw_mean: torch.Tensor,
        raw_std: torch.Tensor,
    ) -> torch.Tensor:
        """Return binary labels shaped like ``normalized_image``.

        ``raw_image`` is the un-normalized crop, used only for gating and
        statistics; the teachers see ``normalized_image``.
        """
        probs = predict_with_tta(self.primary, normalized_image, tta=self.tta)

        # Per-sample choice of teacher, so one bright crop in a batch does not
        # drag the others onto the ensemble path.
        use_ensemble = (raw_mean > self.mean_hi) & (raw_std < self.std_lo)
        while use_ensemble.ndim < probs.ndim:
            use_ensemble = use_ensemble.unsqueeze(-1)

        if self.ensemble is not None and bool(use_ensemble.any()):
            other = predict_with_tta(self.ensemble, normalized_image, tta=self.tta)
            probs = torch.where(use_ensemble, 0.5 * (probs + other), probs)

        threshold = torch.where(
            use_ensemble,
            torch.full_like(probs, self.ensemble_threshold),
            torch.full_like(probs, self.primary_threshold),
        )

        probs = probs * (raw_image > self.input_mask_threshold).to(probs.dtype)
        return (probs > threshold).to(normalized_image.dtype)


def build_labeler(config, device: torch.device) -> SelfDistillLabeler | None:
    """Construct a labeler from config, or None when self-distillation is off."""
    sd = config.self_distill
    if not sd.enabled:
        return None
    if sd.primary_ckpt is None:
        raise ValueError("dynamic_label.enabled requires primary_ckpt")

    primary = load_frozen_model(
        sd.primary_ckpt, device, in_channels=config.in_channels,
        out_channels=config.out_channels,
    )
    ensemble = (
        load_frozen_model(
            sd.ensemble_ckpt, device, in_channels=config.in_channels,
            out_channels=config.out_channels,
        )
        if sd.ensemble_ckpt is not None
        else None
    )
    return SelfDistillLabeler(
        primary,
        ensemble,
        primary_threshold=sd.primary_threshold,
        ensemble_threshold=sd.ensemble_threshold,
        mean_hi=sd.mean_hi,
        std_lo=sd.std_lo,
        input_mask_threshold=sd.input_mask_threshold,
        tta=sd.tta,
    )
