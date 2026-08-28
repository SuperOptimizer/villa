"""Projecting flat surface labels into a native scroll-space crop.

Labels live on the flattened surface grid; training happens in scroll
coordinates. Each flat pixel carries a 3D position and a normal, so a label is
projected by stepping along the normal over a thickness and marking the voxels
it lands in.

Marking points alone leaves gaps: neighbouring surface pixels can be more than
one voxel apart in scroll space, and a normal offset spreads them further. Each
marked point is therefore connected by a 3D line to its right, down, and
diagonal neighbours, and to the previous offset step, so a projected sheet is
watertight.

Ink and background project at different thicknesses and are made mutually
exclusive; supervision is their union.
"""

from __future__ import annotations

import numpy as np


def _mark(output: np.ndarray, z: float, y: float, x: float, origin) -> None:
    zi = int(round(z)) - origin[0]
    yi = int(round(y)) - origin[1]
    xi = int(round(x)) - origin[2]
    if 0 <= zi < output.shape[0] and 0 <= yi < output.shape[1] and 0 <= xi < output.shape[2]:
        output[zi, yi, xi] = 1


def _draw_line(output: np.ndarray, start, end, origin) -> None:
    """Mark voxels along a segment, sampled at ~1 voxel spacing."""
    delta = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    steps = int(max(abs(delta[0]), abs(delta[1]), abs(delta[2])))
    if steps <= 0:
        _mark(output, start[0], start[1], start[2], origin)
        return
    if steps > 512:  # a degenerate normal should not stripe the whole crop
        return
    for i in range(steps + 1):
        t = i / steps
        _mark(
            output,
            start[0] + delta[0] * t,
            start[1] + delta[1] * t,
            start[2] + delta[2] * t,
            origin,
        )


def _offset(positions, normals, row, col, step):
    """Position of a flat pixel pushed ``step`` voxels along its normal."""
    pz, py, px = positions[row, col]
    if not (np.isfinite(pz) and np.isfinite(py) and np.isfinite(px)):
        return None
    if step == 0.0:
        return (float(pz), float(py), float(px))
    nz, ny, nx = normals[row, col]
    magnitude = float(np.sqrt(nz * nz + ny * ny + nx * nx))
    if magnitude < 1e-6 or not np.isfinite(magnitude):
        return None
    scale = step / magnitude
    return (float(pz + nz * scale), float(py + ny * scale), float(px + nx * scale))


def project_mask_along_normals(
    flat_mask: np.ndarray,
    positions_zyx: np.ndarray,
    normals_zyx: np.ndarray,
    valid_mask: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    half_thickness_voxels: float,
) -> np.ndarray:
    """Project a binary flat mask into the crop as a solid sheet.

    ``positions_zyx`` and ``normals_zyx`` are ``(H, W, 3)`` over the same grid
    as ``flat_mask``. Returns a boolean volume shaped to ``crop_bbox_zyx``.
    """
    if half_thickness_voxels < 0.0:
        raise ValueError("half_thickness_voxels must be >= 0")

    flat_mask = np.asarray(flat_mask) > 0
    positions = np.asarray(positions_zyx, dtype=np.float32)
    normals = np.asarray(normals_zyx, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    if positions.shape[:2] != flat_mask.shape or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must be (*{flat_mask.shape}, 3), got {positions.shape}"
        )
    if normals.shape != positions.shape:
        raise ValueError(f"normals {normals.shape} must match positions {positions.shape}")
    if valid.shape != flat_mask.shape:
        raise ValueError(f"valid {valid.shape} must match mask {flat_mask.shape}")

    z0, y0, x0, z1, y1, x1 = crop_bbox_zyx
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    origin = (z0, y0, x0)

    radius = int(np.ceil(half_thickness_voxels))
    rows, cols = flat_mask.shape

    for row in range(rows):
        for col in range(cols):
            if not flat_mask[row, col] or not valid[row, col]:
                continue

            previous = None
            for step in range(-radius, radius + 1):
                if abs(step) > half_thickness_voxels + 1e-6:
                    continue
                current = _offset(positions, normals, row, col, float(step))
                if current is None:
                    break

                _mark(output, current[0], current[1], current[2], origin)
                if previous is not None:
                    _draw_line(output, previous, current, origin)

                # Close the gaps to the neighbours that have not been walked yet.
                for dr, dc in ((0, 1), (1, 0), (1, 1)):
                    nr, nc = row + dr, col + dc
                    if nr >= rows or nc >= cols:
                        continue
                    if not flat_mask[nr, nc] or not valid[nr, nc]:
                        continue
                    neighbour = _offset(positions, normals, nr, nc, float(step))
                    if neighbour is not None:
                        _draw_line(output, current, neighbour, origin)

                previous = current

    return output > 0


def project_labels_and_supervision(
    *,
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    inklabels_flat: np.ndarray,
    supervision_flat: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    normals_zyx: np.ndarray,
    label_half_thickness: float,
    background_half_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ink and background into the crop as disjoint sheets.

    Background is supervised-but-not-ink. It usually projects thicker than ink,
    since a confident negative extends further from the sheet than the ink layer
    itself. Where the two overlap, ink wins.
    """
    labels = np.asarray(inklabels_flat) > 0
    supervision = np.asarray(supervision_flat) > 0
    background = supervision & ~labels

    labels_native = project_mask_along_normals(
        labels, positions_zyx, normals_zyx, valid_mask, crop_bbox_zyx,
        half_thickness_voxels=label_half_thickness,
    )
    background_native = project_mask_along_normals(
        background, positions_zyx, normals_zyx, valid_mask, crop_bbox_zyx,
        half_thickness_voxels=background_half_thickness,
    )
    background_native &= ~labels_native

    return (
        labels_native.astype(np.float32),
        (labels_native | background_native).astype(np.float32),
    )


def compute_native_crop_bbox(
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Centre a fixed-size crop on the valid surface points.

    The crop is always exactly ``target_shape_zyx``: an extent larger than the
    target is trimmed symmetrically, a smaller one padded symmetrically. The
    result may fall outside the volume, which the padded reader handles.
    """
    points = np.asarray(positions_zyx)[np.asarray(valid_mask, dtype=bool)]
    if points.size == 0:
        raise ValueError("no valid surface points in patch")

    mins = points.min(axis=0).astype(np.int64)
    maxs = points.max(axis=0).astype(np.int64)
    target = np.asarray(target_shape_zyx, dtype=np.int64)

    excess = (maxs - mins + 1) - target
    trim_before = np.maximum(excess, 0) // 2
    mins += trim_before
    maxs -= np.maximum(excess, 0) - trim_before

    deficit = target - (maxs - mins + 1)
    pad_before = np.maximum(deficit, 0) // 2
    mins -= pad_before
    maxs += np.maximum(deficit, 0) - pad_before

    return (
        int(mins[0]), int(mins[1]), int(mins[2]),
        int(maxs[0]) + 1, int(maxs[1]) + 1, int(maxs[2]) + 1,
    )


def surface_normals(positions_zyx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Estimate unit normals from the cross product of grid tangents.

    Central differences along the grid axes give two tangents; their cross
    product is the normal. Degenerate points get a unit z normal so callers
    always receive a finite vector — ``valid_mask`` marks which to trust.
    """
    positions = np.asarray(positions_zyx, dtype=np.float32)
    d_row = np.gradient(positions, axis=0)
    d_col = np.gradient(positions, axis=1)

    normals = np.cross(d_row, d_col)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)

    degenerate = (magnitude < 1e-6) | ~np.isfinite(magnitude)
    normals = np.where(degenerate, np.array([1.0, 0.0, 0.0], dtype=np.float32), normals)
    magnitude = np.where(degenerate, 1.0, magnitude)

    normals = normals / magnitude
    normals[~np.asarray(valid_mask, dtype=bool)] = 0.0
    return normals.astype(np.float32)
