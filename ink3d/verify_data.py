"""Checks for the geometry, tifxyz and dataset layers.

Builds a synthetic segment on disk (a flat sheet with a known ink stripe) and
runs it through the real reader and sampler.

Run: .venv/bin/python verify_data.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ink3d.dataset import InkDataset, Patch, Segment, find_patches
from ink3d.geometry import (
    compute_native_crop_bbox,
    project_labels_and_supervision,
    project_mask_along_normals,
    surface_normals,
)
from ink3d.tifxyz import Tifxyz

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")


# ---------------------------------------------------------------- geometry
print("geometry")

# A sheet lying flat at z=32, spanning y,x in [10, 26).
H = W = 16
positions = np.zeros((H, W, 3), dtype=np.float32)
positions[..., 0] = 32.0
positions[..., 1] = np.arange(10, 10 + H)[:, None]
positions[..., 2] = np.arange(10, 10 + W)[None, :]
valid = np.ones((H, W), dtype=bool)

normals = surface_normals(positions, valid)
check("normals are unit length", np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-5))
check("flat sheet has z-facing normals", abs(abs(normals[8, 8, 0]) - 1.0) < 1e-5,
      f"{normals[8, 8].tolist()}")

bbox = compute_native_crop_bbox(positions, valid, (8, 8, 8))
check("crop bbox matches requested shape",
      (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]) == (8, 8, 8), str(bbox))
big = compute_native_crop_bbox(positions, valid, (32, 32, 32))
check("undersized extent pads to target",
      (big[3] - big[0], big[4] - big[1], big[5] - big[2]) == (32, 32, 32))
check("crop centres on the surface", big[0] < 32 < big[3])

crop = (28, 6, 6, 38, 30, 30)   # 10 x 24 x 24 around the sheet
mask = np.zeros((H, W), dtype=bool)
mask[4:12, 4:12] = True

thin = project_mask_along_normals(mask, positions, normals, valid, crop,
                                  half_thickness_voxels=0.0)
check("zero thickness projects one plane", thin.sum() > 0 and thin.shape == (10, 24, 24),
      f"{int(thin.sum())} voxels")
occupied_z = np.unique(np.nonzero(thin)[0])
check("zero thickness stays on one z plane", len(occupied_z) == 1, str(occupied_z.tolist()))

thick = project_mask_along_normals(mask, positions, normals, valid, crop,
                                   half_thickness_voxels=3.0)
z_span = np.unique(np.nonzero(thick)[0])
check("thickness spreads along the normal", len(z_span) == 7, f"{len(z_span)} planes")
check("thicker projection is a superset", bool((thick | thin == thick).all()))

# A sheet whose points are 2 voxels apart would leave gaps without line-filling.
sparse = positions.copy()
sparse[..., 1] = np.arange(10, 10 + H * 2, 2)[:, None]
sparse[..., 2] = np.arange(10, 10 + W * 2, 2)[None, :]
sparse_normals = surface_normals(sparse, valid)
wide_crop = (28, 6, 6, 38, 46, 46)
filled = project_mask_along_normals(mask, sparse, sparse_normals, valid, wide_crop,
                                    half_thickness_voxels=0.0)
plane = filled[np.nonzero(filled)[0][0]]
rows = np.unique(np.nonzero(plane)[0])
check("neighbour lines close the gaps", len(rows) >= 14, f"{len(rows)} rows covered")

labels_flat = np.zeros((H, W), dtype=np.uint8)
labels_flat[4:8, 4:12] = 1
supervision_flat = np.zeros((H, W), dtype=np.uint8)
supervision_flat[4:12, 4:12] = 1

ink, supervision = project_labels_and_supervision(
    positions_zyx=positions, valid_mask=valid,
    inklabels_flat=labels_flat, supervision_flat=supervision_flat,
    crop_bbox_zyx=crop, normals_zyx=normals,
    label_half_thickness=1.0, background_half_thickness=3.0,
)
check("ink and supervision project", ink.sum() > 0 and supervision.sum() > 0,
      f"ink {int(ink.sum())}, sup {int(supervision.sum())}")
check("supervision covers ink", bool(((ink > 0) <= (supervision > 0)).all()))
check("background is thicker than ink", supervision.sum() > ink.sum() * 1.5,
      f"{supervision.sum() / max(ink.sum(), 1):.2f}x")

empty = project_labels_and_supervision(
    positions_zyx=positions, valid_mask=np.zeros_like(valid),
    inklabels_flat=labels_flat, supervision_flat=supervision_flat,
    crop_bbox_zyx=crop, normals_zyx=normals,
    label_half_thickness=1.0, background_half_thickness=3.0,
)
check("no valid points projects nothing", empty[0].sum() == 0)

# ---------------------------------------------------------------- fixture
root = Path(tempfile.mkdtemp(prefix="ink3d_"))
try:
    print("\ntifxyz")
    try:
        import tifffile
        have_tiff = True
    except ImportError:
        have_tiff = False

    if not have_tiff:
        print("  (skipped — needs tifffile or pillow)")
    else:
        surface_dir = root / "seg001" / "surface"
        surface_dir.mkdir(parents=True)
        GH, GW = 64, 64
        gx = np.tile(np.arange(100, 100 + GW, dtype=np.float32), (GH, 1))
        gy = np.tile(np.arange(200, 200 + GH, dtype=np.float32)[:, None], (1, GW))
        gz = np.full((GH, GW), 300.0, dtype=np.float32)
        gz[:4, :] = -1.0   # invalid band: the Z <= 0 rule must drop it
        tifffile.imwrite(surface_dir / "x.tif", gx)
        tifffile.imwrite(surface_dir / "y.tif", gy)
        tifffile.imwrite(surface_dir / "z.tif", gz)
        (surface_dir / "meta.json").write_text(
            json.dumps({"format": "tifxyz", "scale": [1.0, 1.0]})
        )

        surface = Tifxyz(surface_dir)
        check("grid shape read", surface.shape == (GH, GW), str(surface.shape))
        check("Z<=0 vertices invalidated", not surface.valid[:4].any())
        check("remaining vertices valid", surface.valid[4:].all())
        check("invalid positions are NaN", bool(np.isnan(surface.positions[:4]).all()))
        check("positions are ZYX", abs(surface.positions[10, 10, 0] - 300.0) < 1e-4,
              str(surface.positions[10, 10].tolist()))
        check("normals computed on demand", surface.normals.shape == (GH, GW, 3))

        mask = np.full((GH, GW), 255, dtype=np.uint8)
        mask[:, :8] = 0
        tifffile.imwrite(surface_dir / "mask.tif", mask)
        masked = Tifxyz(surface_dir)
        check("mask.tif invalidates vertices", not masked.valid[:, :8].any())
        check("mask leaves the rest valid", masked.valid[4:, 8:].all())

        hi_res = np.full((GH * 2, GW * 2), 255, dtype=np.uint8)
        hi_res[:, :16] = 0
        tifffile.imwrite(surface_dir / "mask.tif", hi_res)
        scaled = Tifxyz(surface_dir)
        check("integer-multiple mask downsamples", not scaled.valid[:, :8].any())
        (surface_dir / "mask.tif").unlink()

        # ------------------------------------------------------ dataset
        print("\ndataset")
        import zarr

        segment_dir = root / "seg001"

        def write_label(name, array):
            store = zarr.open_group(str(segment_dir / name), mode="w")
            data = np.zeros((65, GH, GW), dtype=np.uint8)
            data[32] = array
            store.create_array(name="0", shape=data.shape, chunks=(65, 32, 32),
                               dtype="uint8")[:] = data

        ink_plane = np.zeros((GH, GW), dtype=np.uint8)
        ink_plane[20:44, 20:44] = 1
        write_label("seg001_inklabels.zarr", ink_plane)
        supervision_plane = np.zeros((GH, GW), dtype=np.uint8)
        supervision_plane[16:48, 16:48] = 1
        write_label("seg001_supervision_mask.zarr", supervision_plane)

        volume_path = root / "volume.zarr"
        volume_group = zarr.open_group(str(volume_path), mode="w")
        rng = np.random.default_rng(0)
        volume_group.create_array(name="0", shape=(400, 300, 200), chunks=(64, 64, 64),
                                  dtype="uint8")[:] = rng.integers(
            0, 255, (400, 300, 200), dtype=np.uint8)

        segments = Segment.discover(root, str(volume_path), 0)
        check("segment discovered", len(segments) == 1, f"{len(segments)} found")
        check("surface located", segments[0].surface == surface_dir)
        check("validation mask absent", segments[0].validation_mask is None)

        patches = find_patches(segments, patch_size=(32, 32, 32), overlap=0.5,
                              min_labeled_coverage=0.02, scan_scale=4)
        check("patches found over labelled area", len(patches) > 0, f"{len(patches)}")

        dataset = InkDataset(segments, patches, patch_size=(32, 32, 32),
                             label_half_thickness=2.0, background_half_thickness=4.0,
                             emit_raw=True)
        sample = dataset[0]
        check("sample has the expected keys",
              {"image", "inklabels", "supervision_mask", "raw_image",
               "raw_mean", "raw_std"} <= set(sample))
        check("image is [1,32,32,32]", tuple(sample["image"].shape) == (1, 32, 32, 32),
              str(tuple(sample["image"].shape)))
        check("labels match image shape",
              sample["inklabels"].shape == sample["image"].shape)
        check("image normalized to [0,1]",
              0.0 <= float(sample["image"].min()) and float(sample["image"].max()) <= 1.0,
              f"[{float(sample['image'].min()):.3f}, {float(sample['image'].max()):.3f}]")
        check("raw image keeps its scale", float(sample["raw_image"].max()) > 1.0,
              f"max {float(sample['raw_image'].max()):.1f}")
        check("raw stats are finite",
              bool(torch.isfinite(sample["raw_mean"]) and torch.isfinite(sample["raw_std"])))
        check("labels are binary",
              set(np.unique(sample["inklabels"].numpy()).tolist()) <= {0.0, 1.0})
        check("ink projected into the crop", float(sample["inklabels"].sum()) > 0,
              f"{int(sample['inklabels'].sum())} voxels")
        check("supervision covers ink",
              bool((sample["inklabels"] <= sample["supervision_mask"]).all()))

        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        check("batches collate", tuple(batch["image"].shape) == (2, 1, 32, 32, 32),
              str(tuple(batch["image"].shape)))
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for name in FAIL:
    print("  failed:", name)
sys.exit(1 if FAIL else 0)
