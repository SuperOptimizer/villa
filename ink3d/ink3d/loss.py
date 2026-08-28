"""Dice + label-smoothed BCE.

Matches ``LabelSmoothedDCAndBCELoss``:

    total = weight_ce * masked_mean(BCE(logits, smoothed_target))
          + weight_dice * (-soft_dice(sigmoid(logits), target))

Both terms honour an ignore mask. The Dice term is computed per sample and
averaged over the batch, not pooled across it.

Note the sign: the Dice term contributes the *negative* dice coefficient, so
a perfect overlap drives it to -1 rather than 0. Total loss is therefore
negative once the model is doing well, which is expected.
"""

from __future__ import annotations

import torch
from torch import nn


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Negative soft Dice, averaged over the batch.

    ``logits`` and ``target`` are both [B, C, Z, Y, X]; ``mask`` is [B, 1, ...]
    and marks voxels that count. Summation runs over the spatial axes only, so
    each sample contributes its own Dice score.
    """
    probs = torch.sigmoid(logits)
    axes = tuple(range(2, logits.ndim))

    if mask is not None:
        intersect = (probs * target * mask).sum(axes)
        sum_pred = (probs * mask).sum(axes)
        sum_gt = (target * mask).sum(axes)
    else:
        intersect = (probs * target).sum(axes)
        sum_pred = probs.sum(axes)
        sum_gt = target.sum(axes)

    denominator = torch.clip(sum_gt + sum_pred + smooth, min=1e-8)
    dice = (2 * intersect + smooth) / denominator
    return -dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted sum of the two terms above.

    ``ignore_mask`` is the inverse of the supervision mask: 1 marks a voxel
    that must not contribute. With ``force_full_supervision`` the caller
    passes an all-zero mask, and every voxel counts.
    """

    def __init__(
        self,
        *,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        bce_label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= bce_label_smoothing <= 1.0:
            raise ValueError(
                f"bce_label_smoothing must be in [0, 1], got {bce_label_smoothing}"
            )
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.bce_label_smoothing = float(bce_label_smoothing)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _smooth(self, target: torch.Tensor) -> torch.Tensor:
        if self.bce_label_smoothing == 0.0:
            return target
        s = self.bce_label_smoothing
        return target * (1.0 - s) + 0.5 * s

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.float()
        mask = None if ignore_mask is None else (1.0 - ignore_mask.float())

        dice = soft_dice_loss(logits, target, mask)

        elementwise = self.bce(logits, self._smooth(target))
        if mask is not None:
            # A single mean over every counted voxel in the batch, matching the
            # reference; not a mean of per-sample means.
            ce = (elementwise * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce = elementwise.mean()

        return self.weight_ce * ce + self.weight_dice * dice
