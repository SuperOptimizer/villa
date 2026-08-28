"""Reading the tifxyz surface format.

A tifxyz surface is a directory holding a quad-grid of 3D points:

    x.tif, y.tif, z.tif   single-channel, same size; pixel (row, col) is one
                          grid vertex, together giving P = (X, Y, Z)
    meta.json             {"format": "tifxyz", "scale": [sx, sy], ...}
    mask.tif              optional; channel 0 < 255 invalidates a vertex

Validity follows the format's two rules: ``(-1, -1, -1)`` is the invalid
sentinel, and any vertex with ``Z <= 0`` is invalidated at load time
regardless of X and Y. A mask may be an integer multiple of the grid
resolution, in which case any invalid mask pixel invalidates the vertex it
maps onto; non-integer ratios are ignored, matching the reference loader.

Points are stored XYZ but every consumer here works in ZYX, so ``positions()``
returns ZYX.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_tiff(path: Path) -> np.ndarray:
    """Read a single-channel TIFF as float32, via whichever backend exists."""
    try:
        import tifffile

        data = tifffile.imread(str(path))
    except ImportError:
        try:
            from PIL import Image

            with Image.open(path) as handle:
                data = np.array(handle)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "reading tifxyz needs either 'tifffile' or 'pillow' installed"
            ) from exc

    data = np.asarray(data)
    if data.ndim == 3:  # multi-channel: the format only defines channel 0
        data = data[..., 0]
    return data.astype(np.float32, copy=False)


class Tifxyz:
    """A tifxyz surface directory.

    Coordinate grids are read once and cached; ``mask.tif`` is applied at load
    time so ``valid`` reflects every invalidation rule at once.
    """

    def __init__(self, path: str | Path, *, apply_mask: bool = True) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"tifxyz surface must be a directory: {self.path}")

        meta_path = self.path / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        fmt = self.meta.get("format")
        if fmt is not None and fmt != "tifxyz":
            raise ValueError(f"{self.path} declares format {fmt!r}, expected 'tifxyz'")

        scale = self.meta.get("scale", [1.0, 1.0])
        self.scale = (float(scale[0]), float(scale[1]))

        x = _read_tiff(self.path / "x.tif")
        y = _read_tiff(self.path / "y.tif")
        z = _read_tiff(self.path / "z.tif")
        if not (x.shape == y.shape == z.shape):
            raise ValueError(
                f"x/y/z grids disagree in {self.path}: {x.shape}, {y.shape}, {z.shape}"
            )

        self.shape = x.shape
        self._positions = np.stack([z, y, x], axis=-1)  # ZYX

        valid = np.isfinite(self._positions).all(axis=-1)
        valid &= z > 0  # the format's load-time rule; also excludes the sentinel
        if apply_mask:
            valid &= self._mask_for(self.shape)
        self._valid = valid

        self._positions[~valid] = np.nan
        self._normals: np.ndarray | None = None

    def _mask_for(self, shape: tuple[int, int]) -> np.ndarray:
        """Validity from mask.tif, downsampled if it is an integer multiple."""
        mask_path = self.path / "mask.tif"
        if not mask_path.exists():
            return np.ones(shape, dtype=bool)

        mask = _read_tiff(mask_path)
        if mask.shape == shape:
            return mask >= 255

        factor_y, remainder_y = divmod(mask.shape[0], shape[0])
        factor_x, remainder_x = divmod(mask.shape[1], shape[1])
        if remainder_y or remainder_x or factor_y < 1 or factor_x < 1:
            # Non-integer ratios are skipped rather than guessed at.
            return np.ones(shape, dtype=bool)

        blocks = mask[: shape[0] * factor_y, : shape[1] * factor_x]
        blocks = blocks.reshape(shape[0], factor_y, shape[1], factor_x)
        # Any invalid pixel in a block invalidates the vertex it maps onto.
        return (blocks >= 255).all(axis=(1, 3))

    @property
    def positions(self) -> np.ndarray:
        """``(H, W, 3)`` ZYX coordinates; invalid vertices are NaN."""
        return self._positions

    @property
    def valid(self) -> np.ndarray:
        """``(H, W)`` validity mask."""
        return self._valid

    @property
    def normals(self) -> np.ndarray:
        """``(H, W, 3)`` unit normals, computed on first use."""
        if self._normals is None:
            from .geometry import surface_normals

            self._normals = surface_normals(self._positions, self._valid)
        return self._normals

    def window(self, y0: int, y1: int, x0: int, x1: int):
        """Positions, normals and validity over a grid window."""
        return (
            self._positions[y0:y1, x0:x1],
            self.normals[y0:y1, x0:x1],
            self._valid[y0:y1, x0:x1],
        )

    def __repr__(self) -> str:
        return (
            f"Tifxyz({self.path.name!r}, shape={self.shape}, "
            f"valid={int(self._valid.sum())}/{self._valid.size})"
        )
