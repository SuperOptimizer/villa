"""Sliding-window inference.

Patches are predicted at ``patch_size`` on a stride of
``patch_size * (1 - overlap)``, weighted by a separable Gaussian, and summed
into an accumulator alongside their weights. Dividing the two at the end
gives a properly normalised blend regardless of how many windows happened to
cover a given voxel.

The Gaussian is what keeps patch seams from showing: a uniform window would
weight a voxel at the very edge of a patch — where the model has the least
context — as heavily as one at the centre.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import flip_spatial, mirror_axes, percentile_minmax


def gaussian_weight(
    patch_size: tuple[int, int, int], sigma_scale: float = 0.125
) -> np.ndarray:
    """Separable 3D Gaussian, peak-normalised to 1.

    The small floor keeps a voxel covered only by patch corners from dividing
    by zero.
    """
    windows = []
    for size in patch_size:
        sigma = size * sigma_scale
        coords = np.arange(size, dtype=np.float64)
        centre = (size - 1) / 2.0
        windows.append(np.exp(-0.5 * ((coords - centre) / sigma) ** 2))

    weight = windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]
    weight = weight / weight.max()
    return np.maximum(weight, 1e-4).astype(np.float32)


def _starts(extent: int, patch: int, stride: int) -> list[int]:
    """Window origins covering ``extent``, with the last one flush to the end."""
    if extent <= patch:
        return [0]
    positions = list(range(0, extent - patch + 1, stride))
    if positions[-1] != extent - patch:
        positions.append(extent - patch)
    return positions


@torch.no_grad()
def predict_volume(
    model,
    volume: np.ndarray,
    *,
    patch_size: tuple[int, int, int] = (256, 256, 256),
    overlap: float = 0.5,
    tta: bool = False,
    device: torch.device | None = None,
    normalize: bool = True,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
) -> np.ndarray:
    """Return per-voxel ink probability in [0, 1], shaped like ``volume``.

    ``volume`` is a raw ZYX array; normalization is applied per patch, matching
    training, rather than once over the whole volume.
    """
    device = device or next(model.parameters()).device
    model.eval()

    volume = np.asarray(volume)
    shape = volume.shape
    stride = tuple(max(1, int(round(size * (1.0 - overlap)))) for size in patch_size)
    weight = gaussian_weight(patch_size)

    accumulator = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    variants = mirror_axes(3) if tta else [()]

    for z in _starts(shape[0], patch_size[0], stride[0]):
        for y in _starts(shape[1], patch_size[1], stride[1]):
            for x in _starts(shape[2], patch_size[2], stride[2]):
                box = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                patch = volume[box]
                if patch.shape != tuple(patch_size):
                    continue

                array = (
                    percentile_minmax(patch, percentile_lower, percentile_upper)
                    if normalize
                    else patch.astype(np.float32)
                )
                tensor = torch.from_numpy(array)[None, None].to(device)

                probs = torch.zeros_like(tensor, dtype=torch.float32)
                for axes in variants:
                    logits = model(flip_spatial(tensor, axes))
                    probs += flip_spatial(torch.sigmoid(logits.float()), axes)
                probs /= len(variants)

                prediction = probs[0, 0].cpu().numpy()
                accumulator[box] += prediction * weight
                weights[box] += weight

    return np.clip(accumulator / np.maximum(weights, 1e-8), 0.0, 1.0)


def to_uint8(probability: np.ndarray) -> np.ndarray:
    """Quantise [0, 1] probabilities to the uint8 range used on disk."""
    return np.round(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_multiscale_zarr(
    path: str,
    probability: np.ndarray,
    *,
    levels: int = 6,
    chunks: tuple[int, int, int] = (65, 128, 128),
) -> None:
    """Write a uint8 pyramid, halving Y and X per level.

    Z is left alone: these volumes are already thin in Z relative to the
    surface extent, and halving it discards more than it saves.
    """
    import zarr

    root = zarr.open_group(path, mode="w")
    data = to_uint8(probability)
    datasets = []
    for level in range(levels):
        root.create_array(
            name=str(level), shape=data.shape, chunks=chunks, dtype="uint8",
        )[:] = data
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 2.0**level, 2.0**level]}
                ],
            }
        )
        if level + 1 < levels:
            data = data[:, ::2, ::2]

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets,
        }
    ]
