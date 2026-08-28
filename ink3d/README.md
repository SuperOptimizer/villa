# ink3d

Minimal 3D U-Net for ink detection and surface prediction, rewritten from
scratch. Replaces ~13,000 lines of config-driven framework code with ~1,900
lines on `torch` + `numpy` + `zarr`.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

Python 3.14 free-threaded builds segfault on `import torch`; use 3.12.

## Use

```bash
python -m ink3d train config.json
python -m ink3d infer config.json --checkpoint ckpt.pth \
    --volume scroll.zarr --out prediction.zarr --tta
```

Runs on CUDA, MPS (Apple Metal), or CPU, selected automatically. Autocast is
enabled on CUDA only — bf16 on MPS is unreliable enough that silently wrong
numerics are the worse trade.

## Layout

| Module | Contents |
|---|---|
| `model.py` | 7-stage U-Net: residual encoder, plain-conv decoder |
| `loss.py` | Dice + label-smoothed BCE with an ignore mask |
| `train.py` | Training loop, EMA, cosine schedule with warmup |
| `dataset.py` | Segment discovery, patch finding, sample assembly |
| `geometry.py` | Projecting flat labels into scroll space along normals |
| `tifxyz.py` | The tifxyz surface format |
| `data.py` | Zarr reads, percentile normalization, mirror TTA |
| `self_distill.py` | Teacher-generated labels |
| `infer.py` | Sliding-window inference with Gaussian blending |

## Architecture

Autoconfigure for a 256³ patch resolves to seven stages:

| stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| stride | 1 | 2 | 2 | 2 | 2 | 2 | 2 |
| features | 32 | 64 | 128 | 256 | 320 | 320 | 320 |
| blocks | 1 | 3 | 4 | 6 | 6 | 6 | 6 |

141.89M parameters. Details that matter for loading existing checkpoints:

- The encoder is residual (`BasicBlockD`), the decoder is not. Downsampling
  happens in the first conv of each stage, not in a pooling layer.
- In a residual block, `conv2` has no activation — the block activates once
  after the residual add.
- Skip-path 1×1 projections are always bias-free, regardless of the main path.
- `InstanceNorm3d` is affine without running stats, so norm layers contribute
  weight and bias and no buffers.
- All six segmentation heads are built even with deep supervision off, so a
  checkpoint trained with it stays loadable.

Six downsampling stages divide the input by 64, so **inputs must be at least
128³** — anything smaller collapses the bottleneck to a single voxel, which
InstanceNorm rejects.

## Data

A segment directory holds label zarrs beside a tifxyz surface:

```
segment/
├── surface/{x,y,z}.tif, meta.json   quad-grid of 3D points
├── seg_inklabels.zarr               (65, H, W) uint8, label on plane 32
├── seg_supervision_mask.zarr
└── seg_validation_mask.zarr         optional
```

Labels are 2D grids stored inside a 65-slice volume so they align with a Z
window of the surface volume; only plane 32 carries signal. Training projects
them into scroll space along the surface normals, drawing lines between
neighbouring points so the projected sheet has no gaps.

## Training recipe

From the reference config: SGD at lr 0.01, momentum 0.99, Nesterov, weight
decay 3e-5; cosine schedule with 5000 warmup steps over 250k iterations; EMA
decay 0.9995 from step 1000; bf16; Dice + BCE weighted 1:1 with 0.1 BCE label
smoothing.

With `dynamic_label.kind: "self_distill"`, stored labels are replaced every
step by teacher predictions: a primary checkpoint, plus a second averaged in
for crops whose raw intensity is bright and flat (`mean > 105`, `std < 30`),
gated on raw intensity > 50 and binarized at 0.176 (0.157 for the ensemble).
Combined with `force_full_supervision`, every voxel is supervised, since the
teacher labels all of them. Validation keeps the real supervision mask.

## Deliberate omissions

- `dice_label_smoothing` is a no-op for single-channel targets — it only ever
  applied on the Dice loss's index-encoded branch, which a 1-channel ink target
  never reaches. Implementing it would be decorative.
- The DINOv2/dinovol backbone is not needed. `dino_guided_*` names the
  checkpoint lineage, not the code path: those configs set
  `kind: "self_distill"` and load no DINO backbone. Only a `dino_guided` kind
  would, and none of the shipped recipes use it.
- MedNeXt, Primus, multi-task decoders, deep-supervision loss weighting,
  stitched forward, and label dilation are all off in the reference configs.

## Verification

```bash
.venv/bin/python verify.py        # 45 checks: model, loss, schedule, EMA, config
.venv/bin/python verify_data.py   # 37 checks: geometry, tifxyz, dataset
.venv/bin/python verify_e2e.py    # 17 checks: trains on synthetic data
```

`verify_e2e.py` trains a shrunken model on a synthetic segment where ink is a
bright slab, and asserts the loss falls and the model separates ink from
background — so a regression that leaves shapes intact but breaks learning
still fails.

**Not yet verified against a released checkpoint.** The architecture is derived
from the reference implementation's autoconfigure path, not from a
`state_dict` key dump. Loading `scrollprize/ink_3d_dino_guided` with
`strict=True` is the check that would confirm the parameter count and key
names; until then, treat checkpoint compatibility as unconfirmed.
