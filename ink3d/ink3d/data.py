"""Volume access, normalization, and patch assembly.

The on-disk layout this reads:

``surface-volume`` / scroll volume
    A zarr group of resolution levels keyed ``"0"``..``"5"``, or a bare 3D
    array. ZYX, uint8.

label / mask zarrs
    Same multiscale shape, ``(65, H, W)`` uint8 per level. The label itself
    occupies a single Z plane (index 32) and every other plane is zero — the
    volume exists so the label lines up with a 65-slice window of the surface
    volume, not because it is volumetric data. Code that treats all 65 planes
    as signal will train on 64 planes of nothing.
"""

from __future__ import annotations

import numpy as np

LABEL_DEPTH = 65
LABEL_SLICE = 32


# --------------------------------------------------------------------------
# zarr access
# --------------------------------------------------------------------------

def open_volume(path: str, scale: int = 0):
    """Open a volume and return the array for one resolution level.

    Accepts a bare array or a multiscale group; ``scale`` indexes the pyramid
    and is ignored for a bare array.
    """
    import zarr

    root = zarr.open(path, mode="r")
    if hasattr(root, "shape"):
        return root
    key = str(int(scale))
    if key not in root:
        available = sorted(root.array_keys())
        raise KeyError(f"level {key!r} not in {path!r}; have {available!r}")
    return root[key]


def read_bbox(
    array,
    origin: tuple[int, int, int],
    shape: tuple[int, int, int],
    *,
    fill_value: int | float = 0,
) -> np.ndarray:
    """Read a ZYX box, zero-padding wherever it runs past an edge.

    Reads near a boundary are normal rather than exceptional here — patch
    centres are chosen from surface geometry, not from the array's interior —
    so this pads instead of raising.
    """
    origin = tuple(int(v) for v in origin)
    shape = tuple(int(v) for v in shape)

    starts, stops, pads = [], [], []
    for axis, (start, size) in enumerate(zip(origin, shape)):
        limit = int(array.shape[axis])
        stop = start + size
        clipped_start, clipped_stop = max(start, 0), min(stop, limit)
        if clipped_stop < clipped_start:
            clipped_stop = clipped_start
        starts.append(clipped_start)
        stops.append(clipped_stop)
        pads.append((clipped_start - start, stop - clipped_stop))

    region = array[
        starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]
    ]
    region = np.asarray(region)

    if any(before or after for before, after in pads):
        region = np.pad(region, pads, mode="constant", constant_values=fill_value)
    return region


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def percentile_minmax(
    image: np.ndarray, lower: float = 1.0, upper: float = 99.0
) -> np.ndarray:
    """Clip to the given percentiles, then rescale that span onto [0, 1].

    A degenerate span (flat or non-finite crop) yields zeros rather than
    dividing by ~0.
    """
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, lower))
    hi = float(np.percentile(image, upper))
    scale = hi - lo
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    out = np.clip(image, lo, hi)
    out = (out - lo) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def labeled_coverage(label: np.ndarray) -> float:
    """Fraction of non-zero voxels — the patch-acceptance criterion."""
    if label.size == 0:
        return 0.0
    return float(np.count_nonzero(label)) / float(label.size)


# --------------------------------------------------------------------------
# test-time augmentation
# --------------------------------------------------------------------------

def mirror_axes(ndim_spatial: int = 3) -> list[tuple[int, ...]]:
    """All 2**n spatial flip combinations, identity first."""
    from itertools import combinations

    axes = list(range(ndim_spatial))
    out: list[tuple[int, ...]] = []
    for r in range(len(axes) + 1):
        out.extend(combinations(axes, r))
    return out


def flip_spatial(tensor, axes: tuple[int, ...]):
    """Flip a [B, C, Z, Y, X] tensor over the named spatial axes."""
    if not axes:
        return tensor
    return tensor.flip([a + 2 for a in axes])


__all__ = [
    "LABEL_DEPTH",
    "LABEL_SLICE",
    "open_volume",
    "read_bbox",
    "percentile_minmax",
    "labeled_coverage",
    "mirror_axes",
    "flip_spatial",
]
