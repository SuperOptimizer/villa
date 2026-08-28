"""Segments, patch selection, and sample assembly.

A segment is one labelled surface: a tifxyz directory plus label and mask zarrs
on the flattened grid, paired with the scroll volume the surface came from.

Sampling a patch means picking a window on the flat grid, projecting that
window's labels into scroll space along the surface normals, and reading the
matching image crop from the volume.

Patch windows are chosen once at startup on a downsampled grid, since candidate
windows are decided by label coverage and scanning at full resolution buys
nothing. Windows with too little ink are dropped: with
``patch_min_labeled_coverage`` at 0.02, a window needs 2% positive label to
train on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import open_volume, percentile_minmax, read_bbox
from .geometry import compute_native_crop_bbox, project_labels_and_supervision
from .tifxyz import Tifxyz


@dataclass
class Segment:
    """One labelled surface and the volume it belongs to."""

    surface: Path
    volume_path: str
    inklabels: Path
    supervision_mask: Path
    validation_mask: Path | None = None
    scale: int = 0

    @classmethod
    def discover(cls, segments_root: Path, volume_path: str, scale: int = 0):
        """Find segments under a root directory.

        A segment directory holds ``*_inklabels.zarr`` beside a tifxyz surface;
        anything missing labels or a surface is skipped rather than failing the
        run, since partially prepared directories are normal.
        """
        segments_root = Path(segments_root)
        found = []
        for inklabels in sorted(segments_root.rglob("*_inklabels.zarr")):
            directory = inklabels.parent
            stem = inklabels.name[: -len("_inklabels.zarr")]

            supervision = directory / f"{stem}_supervision_mask.zarr"
            if not supervision.exists():
                continue

            surface = directory / "surface" if (directory / "surface").is_dir() else directory
            if not (surface / "x.tif").exists():
                continue

            validation = directory / f"{stem}_validation_mask.zarr"
            found.append(
                cls(
                    surface=surface,
                    volume_path=volume_path,
                    inklabels=inklabels,
                    supervision_mask=supervision,
                    validation_mask=validation if validation.exists() else None,
                    scale=scale,
                )
            )
        return found


@dataclass(frozen=True)
class Patch:
    """A window on one segment's flat grid."""

    segment_index: int
    y0: int
    y1: int
    x0: int
    x1: int


def _flat_surface(array, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read the labelled Z plane of a label/mask zarr over a grid window.

    These volumes are (65, H, W) with the label on a single plane; the rest is
    padding so the label lines up with a Z window of the surface volume.
    """
    from .data import LABEL_SLICE

    if array.ndim == 3:
        plane = array[LABEL_SLICE, y0:y1, x0:x1]
    else:
        plane = array[y0:y1, x0:x1]
    return np.asarray(plane)


def find_patches(
    segments: list[Segment],
    *,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    min_labeled_coverage: float = 0.02,
    scan_scale: int = 4,
) -> list[Patch]:
    """Scan each segment for windows with enough labelled content."""
    patches: list[Patch] = []
    window_y, window_x = patch_size[1], patch_size[2]
    stride_y = max(1, int(round(window_y * (1.0 - overlap))))
    stride_x = max(1, int(round(window_x * (1.0 - overlap))))

    for index, segment in enumerate(segments):
        labels = open_volume(str(segment.inklabels), segment.scale)
        height, width = labels.shape[-2], labels.shape[-1]

        coarse = _flat_surface(labels, 0, height, 0, width)[::scan_scale, ::scan_scale]
        coarse = np.asarray(coarse) > 0

        for y0 in range(0, max(1, height - window_y + 1), stride_y):
            for x0 in range(0, max(1, width - window_x + 1), stride_x):
                block = coarse[
                    y0 // scan_scale : (y0 + window_y) // scan_scale,
                    x0 // scan_scale : (x0 + window_x) // scan_scale,
                ]
                if block.size == 0:
                    continue
                if float(np.count_nonzero(block)) / block.size < min_labeled_coverage:
                    continue
                patches.append(Patch(index, y0, min(y0 + window_y, height),
                                     x0, min(x0 + window_x, width)))
    return patches


class InkDataset(Dataset):
    """Native-space 3D crops with labels projected from the flat surface."""

    def __init__(
        self,
        segments: list[Segment],
        patches: list[Patch],
        *,
        patch_size: tuple[int, int, int] = (256, 256, 256),
        label_half_thickness: float = 3.0,
        background_half_thickness: float = 6.0,
        percentile_lower: float = 1.0,
        percentile_upper: float = 99.0,
        emit_raw: bool = False,
        seed: int = 0,
    ) -> None:
        if not patches:
            raise ValueError("no patches; check label coverage and segment paths")
        self.segments = segments
        self.patches = patches
        self.patch_size = tuple(patch_size)
        self.label_half_thickness = float(label_half_thickness)
        self.background_half_thickness = float(background_half_thickness)
        self.percentile_lower = float(percentile_lower)
        self.percentile_upper = float(percentile_upper)
        # Self-distillation needs the raw crop and its statistics alongside the
        # normalized one, since the teacher gates on raw intensity.
        self.emit_raw = bool(emit_raw)
        self.rng = random.Random(seed)
        self._cache: dict[int, Tifxyz] = {}
        self._volumes: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.patches)

    def _surface(self, index: int) -> Tifxyz:
        if index not in self._cache:
            self._cache[index] = Tifxyz(self.segments[index].surface)
        return self._cache[index]

    def _volume(self, segment: Segment):
        key = f"{segment.volume_path}@{segment.scale}"
        if key not in self._volumes:
            self._volumes[key] = open_volume(segment.volume_path, segment.scale)
        return self._volumes[key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patch = self.patches[index]
        segment = self.segments[patch.segment_index]
        surface = self._surface(patch.segment_index)

        positions, normals, valid = surface.window(patch.y0, patch.y1, patch.x0, patch.x1)
        if not valid.any():
            return self[(index + 1) % len(self)]

        try:
            crop_bbox = compute_native_crop_bbox(positions, valid, self.patch_size)
        except ValueError:
            return self[(index + 1) % len(self)]

        labels_flat = _flat_surface(
            open_volume(str(segment.inklabels), segment.scale),
            patch.y0, patch.y1, patch.x0, patch.x1,
        )
        supervision_flat = _flat_surface(
            open_volume(str(segment.supervision_mask), segment.scale),
            patch.y0, patch.y1, patch.x0, patch.x1,
        )

        labels, supervision = project_labels_and_supervision(
            positions_zyx=positions,
            valid_mask=valid,
            inklabels_flat=labels_flat,
            supervision_flat=supervision_flat,
            crop_bbox_zyx=crop_bbox,
            normals_zyx=normals,
            label_half_thickness=self.label_half_thickness,
            background_half_thickness=self.background_half_thickness,
        )

        raw = read_bbox(
            self._volume(segment),
            (crop_bbox[0], crop_bbox[1], crop_bbox[2]),
            self.patch_size,
        ).astype(np.float32)

        image = percentile_minmax(raw, self.percentile_lower, self.percentile_upper)

        sample = {
            "image": torch.from_numpy(image)[None],
            "inklabels": torch.from_numpy(labels)[None],
            "supervision_mask": torch.from_numpy(supervision)[None],
        }
        if self.emit_raw:
            sample["raw_image"] = torch.from_numpy(raw)[None]
            sample["raw_mean"] = torch.tensor(float(raw.mean()))
            sample["raw_std"] = torch.tensor(float(raw.std()))
        return sample


def build_dataloader(
    config,
    *,
    shuffle: bool = True,
    num_workers: int | None = None,
):
    """Build a training dataloader from a parsed config."""
    segments: list[Segment] = []
    for dataset in config.datasets:
        segments.extend(
            Segment.discover(dataset.segments_path, dataset.volume_path, dataset.volume_scale)
        )
    if not segments:
        raise ValueError(
            "no segments found; expected directories with *_inklabels.zarr beside "
            "a tifxyz surface"
        )

    patches = find_patches(
        segments,
        patch_size=config.patch_size,
        overlap=config.patch_overlap,
        min_labeled_coverage=config.patch_min_labeled_coverage,
        scan_scale=config.patch_finding_scale,
    )

    dataset = InkDataset(
        segments,
        patches,
        patch_size=config.patch_size,
        label_half_thickness=config.projection_half_thickness,
        background_half_thickness=config.projection_half_thickness * 2.0,
        percentile_lower=config.percentile_lower,
        percentile_upper=config.percentile_upper,
        emit_raw=config.self_distill.enabled,
        seed=config.seed,
    )

    workers = config.dataloader_workers if num_workers is None else num_workers
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        prefetch_factor=config.prefetch_factor if workers > 0 else None,
        pin_memory=True,
        drop_last=True,
    )
