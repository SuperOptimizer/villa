"""End-to-end smoke test.

Builds a synthetic segment where ink is a bright slab inside darker noise, runs
real training steps against it, and checks the loss falls. Then loads the saved
checkpoint back and runs inference.

The model is deliberately shrunk to 3 stages: the real 7-stage network needs
128^3 patches, which is too slow to train on CPU/MPS in a test.

Run: .venv/bin/python verify_e2e.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import ink3d.model as model_module
from ink3d.dataset import InkDataset, Segment, find_patches
from ink3d.infer import predict_volume
from ink3d.loss import DiceBCELoss
from ink3d.train import ModelEMA, cosine_schedule_with_warmup, prepare_loss_inputs

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")


device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"device: {device}\n")

root = Path(tempfile.mkdtemp(prefix="ink3d_e2e_"))
try:
    import tifffile
    import zarr

    # ------------------------------------------------------------ fixture
    print("fixture")
    GH = GW = 64
    SURFACE_Z = 64
    surface_dir = root / "seg" / "surface"
    surface_dir.mkdir(parents=True)

    tifffile.imwrite(surface_dir / "x.tif",
                     np.tile(np.arange(32, 32 + GW, dtype=np.float32), (GH, 1)))
    tifffile.imwrite(surface_dir / "y.tif",
                     np.tile(np.arange(32, 32 + GH, dtype=np.float32)[:, None], (1, GW)))
    tifffile.imwrite(surface_dir / "z.tif",
                     np.full((GH, GW), float(SURFACE_Z), dtype=np.float32))
    (surface_dir / "meta.json").write_text(
        json.dumps({"format": "tifxyz", "scale": [1.0, 1.0]}))

    ink_plane = np.zeros((GH, GW), dtype=np.uint8)
    ink_plane[16:48, 16:48] = 1
    supervision_plane = np.ones((GH, GW), dtype=np.uint8)

    for name, plane in (("seg_inklabels.zarr", ink_plane),
                        ("seg_supervision_mask.zarr", supervision_plane)):
        group = zarr.open_group(str(root / "seg" / name), mode="w")
        data = np.zeros((65, GH, GW), dtype=np.uint8)
        data[32] = plane
        group.create_array(name="0", shape=data.shape, chunks=(65, 32, 32),
                           dtype="uint8")[:] = data

    # Ink is learnable: bright where labelled, dark elsewhere.
    rng = np.random.default_rng(0)
    volume = rng.integers(40, 90, (128, 128, 128)).astype(np.uint8)
    ys, xs = np.nonzero(ink_plane)
    for y, x in zip(ys, xs):
        volume[SURFACE_Z - 2 : SURFACE_Z + 3, 32 + y, 32 + x] = rng.integers(180, 255, 5)

    volume_group = zarr.open_group(str(root / "volume.zarr"), mode="w")
    volume_group.create_array(name="0", shape=volume.shape, chunks=(64, 64, 64),
                              dtype="uint8")[:] = volume

    segments = Segment.discover(root, str(root / "volume.zarr"), 0)
    check("segment discovered", len(segments) == 1)

    patches = find_patches(segments, patch_size=(32, 32, 32), overlap=0.5,
                           min_labeled_coverage=0.02, scan_scale=2)
    check("patches found", len(patches) > 0, f"{len(patches)}")

    dataset = InkDataset(segments, patches, patch_size=(32, 32, 32),
                         label_half_thickness=2.0, background_half_thickness=4.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                         num_workers=0, drop_last=True)
    sample = dataset[0]
    check("ink present in sample", float(sample["inklabels"].sum()) > 0,
          f"{int(sample['inklabels'].sum())} voxels")
    check("ink is a minority class", float(sample["inklabels"].mean()) < 0.5,
          f"{float(sample['inklabels'].mean()):.3f}")

    # ------------------------------------------------------------ training
    print("\ntraining")
    # Three stages so a 32^3 patch does not collapse the bottleneck.
    original = (model_module.FEATURES, model_module.BLOCKS,
                model_module.STRIDES, model_module.MIN_INPUT_SIZE)
    model_module.FEATURES = (16, 32, 64)
    model_module.BLOCKS = (1, 2, 2)
    model_module.STRIDES = ((1, 1, 1), (2, 2, 2), (2, 2, 2))
    model_module.MIN_INPUT_SIZE = 8

    try:
        torch.manual_seed(0)
        model = model_module.build_model().to(device)
        model.apply(model_module.InitWeightsHe(neg_slope=0.2))

        criterion = DiceBCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.99,
                                    nesterov=True, weight_decay=3e-5)
        scheduler = cosine_schedule_with_warmup(optimizer, 5, 60)
        ema = ModelEMA(model, decay=0.99, start_step=10)

        losses = []
        iterator = iter(loader)
        model.train()
        for step in range(1, 61):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch["image"])
            logits, target, ignore = prepare_loss_inputs(
                logits.float(), batch, force_full_supervision=True)
            loss = criterion(logits, target.float(), ignore)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(model, step)
            losses.append(loss.item())

        first, last = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
        check("all losses finite", all(np.isfinite(losses)))
        check("loss decreases", last < first, f"{first:+.4f} -> {last:+.4f}")
        check("training reaches negative loss", last < 0, f"{last:+.4f}")

        checkpoint = root / "ckpt.pth"
        torch.save({"step": 60, "model": model.state_dict(),
                    "ema": ema.state_dict()}, checkpoint)
        check("checkpoint written", checkpoint.exists())

        # -------------------------------------------------------- reload
        print("\ncheckpoint round trip")
        reloaded = model_module.build_model()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = reloaded.load_state_dict(payload["model"], strict=True), None
        check("state_dict loads strictly", missing is not None or True)
        reloaded.to(device).eval()

        model.eval()
        probe = dataset[0]["image"][None].to(device)
        with torch.no_grad():
            check("reloaded model matches original",
                  torch.allclose(model(probe), reloaded(probe), atol=1e-5))

        ema_model = model_module.build_model()
        ema_model.load_state_dict(payload["ema"], strict=True)
        check("EMA weights load strictly", True)

        # -------------------------------------------------------- inference
        print("\ninference")
        prediction = predict_volume(reloaded, volume[:96, :96, :96],
                                    patch_size=(32, 32, 32), overlap=0.5,
                                    device=device)
        check("prediction covers the volume", prediction.shape == (96, 96, 96),
              str(prediction.shape))
        check("probabilities in [0,1]",
              prediction.min() >= 0.0 and prediction.max() <= 1.0,
              f"[{prediction.min():.4f}, {prediction.max():.4f}]")
        check("prediction is not constant", prediction.std() > 1e-4,
              f"std {prediction.std():.4f}")

        # The bright slab should score above the dark background.
        ink_region = prediction[SURFACE_Z - 2 : SURFACE_Z + 3, 48:80, 48:80].mean()
        background = prediction[8:16, 8:24, 8:24].mean()
        check("ink scores above background", ink_region > background,
              f"ink {ink_region:.4f} vs bg {background:.4f}")

        tta_prediction = predict_volume(reloaded, volume[:64, :64, :64],
                                        patch_size=(32, 32, 32), overlap=0.5,
                                        tta=True, device=device)
        check("TTA inference runs", tta_prediction.shape == (64, 64, 64))
        check("TTA output stays in range",
              tta_prediction.min() >= 0.0 and tta_prediction.max() <= 1.0)
    finally:
        (model_module.FEATURES, model_module.BLOCKS,
         model_module.STRIDES, model_module.MIN_INPUT_SIZE) = original
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for name in FAIL:
    print("  failed:", name)
sys.exit(1 if FAIL else 0)
