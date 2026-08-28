"""Minimal 3D ink detection and surface prediction.

A from-scratch PyTorch implementation of the 3D U-Net used for the June 2026
model releases, covering training (including self-distillation) and
sliding-window inference.

Dependencies are torch, numpy and zarr.
"""

from .config import Config
from .loss import DiceBCELoss, soft_dice_loss
from .model import InkUNet, build_model

__all__ = ["Config", "DiceBCELoss", "soft_dice_loss", "InkUNet", "build_model"]
__version__ = "0.1.0"
