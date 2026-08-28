"""End-to-end checks against known-good values.

Run: .venv/bin/python verify.py
"""

import math
import sys

import numpy as np
import torch

from ink3d.config import Config
from ink3d.data import mirror_axes, percentile_minmax, flip_spatial, read_bbox
from ink3d.infer import gaussian_weight, predict_volume
from ink3d.loss import DiceBCELoss, soft_dice_loss
from ink3d.model import build_model
from ink3d.train import ModelEMA, cosine_schedule_with_warmup, prepare_loss_inputs

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


dev = device()
print(f"device: {dev}   torch {torch.__version__}\n")

# ---------------------------------------------------------------- model
print("model")
m = build_model().to(dev).eval()
n_params = sum(p.numel() for p in m.parameters())
x = torch.randn(1, 1, 128, 128, 128, device=dev)
with torch.no_grad():
    y = m(x)
check("forward preserves spatial shape", tuple(y.shape) == (1, 1, 128, 128, 128), str(tuple(y.shape)))
check("single output channel", y.shape[1] == 1)

sd = m.state_dict()
check("encoder key naming", "shared_encoder.stem.convs.0.conv.weight" in sd)
check("decoder key naming", "task_decoders.ink.seg_layers.5.weight" in sd)
check("six seg layers built", sum(1 for k in sd if "seg_layers" in k and k.endswith("weight")) == 6)
check("no norm running stats", not any("running_mean" in k for k in sd))
check("skip projections biasless", "shared_encoder.stages.1.blocks.0.skip.1.conv.bias" not in sd)
check("stage 0 has no skip module", not any(k.startswith("shared_encoder.stages.0.blocks.0.skip") for k in sd))
print(f"       params {n_params/1e6:.2f}M, {len(sd)} state_dict entries")

m.deep_supervision = True
with torch.no_grad():
    ds = m(x)
check("deep supervision yields 6 maps", isinstance(ds, list) and len(ds) == 6)
check("deep supervision scales halve", tuple(ds[1].shape[-3:]) == (64, 64, 64))
m.deep_supervision = False

# ---------------------------------------------------------------- loss
print("\nloss")
logits = torch.full((2, 1, 8, 8, 8), 20.0)   # sigmoid ~ 1
target = torch.ones((2, 1, 8, 8, 8))
d = soft_dice_loss(logits, target)
check("perfect overlap -> dice ~ -1", abs(d.item() + 1.0) < 1e-3, f"{d.item():.6f}")

d_bad = soft_dice_loss(torch.full((2, 1, 8, 8, 8), -20.0), target)
check("no overlap -> dice ~ 0", abs(d_bad.item()) < 1e-2, f"{d_bad.item():.6f}")

crit = DiceBCELoss(weight_ce=1.0, weight_dice=1.0, bce_label_smoothing=0.1)
smoothed = crit._smooth(torch.tensor([0.0, 1.0]))
check("smoothing maps 0->0.05, 1->0.95",
      torch.allclose(smoothed, torch.tensor([0.05, 0.95]), atol=1e-6), str(smoothed.tolist()))

# An all-ignore mask must not produce NaN through the 1e-8 clamp.
loss_ignored = crit(logits, target, torch.ones_like(target))
check("fully-ignored batch stays finite", torch.isfinite(loss_ignored).item())

# Ignored voxels must not influence the result.
lg = torch.randn(2, 1, 4, 4, 4)
tg = (torch.rand(2, 1, 4, 4, 4) > 0.5).float()
ig = torch.zeros_like(tg); ig[..., 2:] = 1.0
lg2 = lg.clone(); lg2[..., 2:] = 99.0
check("ignored region excluded from BCE",
      abs(crit(lg, tg, ig).item() - crit(lg2, tg, ig).item()) < 0.35,
      f"{crit(lg, tg, ig).item():.4f} vs {crit(lg2, tg, ig).item():.4f}")

# ---------------------------------------------------------------- schedule
print("\nschedule")
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
sched = cosine_schedule_with_warmup(opt, num_warmup_steps=5000, num_training_steps=250000)
lrs = []
for step in range(250000):
    lrs.append(opt.param_groups[0]["lr"])
    sched.step()
check("lr starts at 0", lrs[0] == 0.0)
check("peak lr at end of warmup", abs(lrs[5000] - 0.01) < 1e-9, f"{lrs[5000]:.6f}")
check("lr decays to ~0", lrs[-1] < 1e-6, f"{lrs[-1]:.3e}")
check("warmup is linear", abs(lrs[2500] - 0.005) < 1e-9, f"{lrs[2500]:.6f}")
mid = lrs[5000 + (250000 - 5000) // 2]
check("half-way lr is ~half peak", abs(mid - 0.005) < 1e-4, f"{mid:.6f}")

# ---------------------------------------------------------------- EMA
print("\nEMA")
small = build_model()
ema = ModelEMA(small, decay=0.9995, start_step=1000)
key = "shared_encoder.stem.convs.0.conv.weight"

with torch.no_grad():
    small.state_dict()[key].fill_(1.0)
ema.update(small, step=10)
check("tracks exactly before start_step",
      torch.allclose(ema.state_dict()[key], torch.ones_like(ema.state_dict()[key])))

with torch.no_grad():
    small.state_dict()[key].fill_(2.0)
ema.update(small, step=1000)
got = ema.state_dict()[key].flatten()[0].item()
want = 1.0 * 0.9995 + 2.0 * 0.0005
check("decays at the configured rate", abs(got - want) < 1e-6, f"{got:.6f} vs {want:.6f}")

# ---------------------------------------------------------------- data
print("\ndata")
img = np.concatenate([np.full(100, 10.0), np.linspace(0, 200, 800), np.full(100, 250.0)])
norm = percentile_minmax(img, 1.0, 99.0)
check("normalization lands in [0,1]", norm.min() >= 0.0 and norm.max() <= 1.0,
      f"[{norm.min():.3f}, {norm.max():.3f}]")
check("flat input normalizes to zeros", np.all(percentile_minmax(np.full(64, 7.0)) == 0.0))

axes = mirror_axes(3)
check("8 mirror variants", len(axes) == 8, str(len(axes)))
check("identity variant first", axes[0] == ())
t = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
check("double flip is identity", torch.equal(flip_spatial(flip_spatial(t, (0, 2)), (0, 2)), t))

arr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
padded = read_bbox(arr, (-1, -1, -1), (3, 3, 3))
check("out-of-bounds read pads with zeros",
      padded.shape == (3, 3, 3) and padded[0, 0, 0] == 0 and padded[1, 1, 1] == arr[0, 0, 0])

# ---------------------------------------------------------------- inference
print("\ninference")
w = gaussian_weight((16, 16, 16))
check("gaussian peaks at centre", abs(w[8, 8, 8] - 1.0) < 1e-6, f"{w[8,8,8]:.6f}")
check("gaussian decays to the corner", w[0, 0, 0] < 0.02, f"{w[0,0,0]:.6f}")
check("gaussian strictly positive", w.min() > 0)

class Constant(torch.nn.Module):
    """Returns a fixed logit everywhere, so blending is checkable in closed form."""
    def __init__(self, value): super().__init__(); self.value = value
    def forward(self, x): return torch.full_like(x, self.value)
    def parameters(self, recurse=True): return iter([torch.nn.Parameter(torch.zeros(1))])

const = Constant(0.0)
out = predict_volume(const, np.random.rand(32, 32, 32).astype(np.float32) * 255,
                     patch_size=(16, 16, 16), overlap=0.5, device=torch.device("cpu"))
check("blend of constant 0.5 stays 0.5", np.allclose(out, 0.5, atol=1e-4),
      f"[{out.min():.4f}, {out.max():.4f}]")
check("output covers the volume", out.shape == (32, 32, 32))

# ---------------------------------------------------------------- training step
print("\ntraining step")
batch = {
    "image": torch.randn(1, 1, 128, 128, 128, device=dev),
    "inklabels": (torch.rand(1, 1, 128, 128, 128, device=dev) > 0.7).float(),
    "supervision_mask": torch.ones(1, 1, 128, 128, 128, device=dev),
}
lg, tg, ig = prepare_loss_inputs(torch.randn(1, 1, 128, 128, 128, device=dev), batch,
                                force_full_supervision=True)
check("force_full_supervision ignores nothing", float(ig.sum()) == 0.0)

batch["supervision_mask"][..., 64:] = 0
_, _, ig2 = prepare_loss_inputs(torch.randn(1, 1, 128, 128, 128, device=dev), batch,
                                force_full_supervision=False)
check("supervision mask drives the ignore mask", float(ig2.sum()) > 0)

model = build_model().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True,
                      weight_decay=3e-5)
crit = DiceBCELoss()
before = [p.detach().clone() for p in model.parameters()]
logits = model(batch["image"])
lg, tg, ig = prepare_loss_inputs(logits.float(), batch, force_full_supervision=True)
loss = crit(lg, tg.float(), ig)
opt.zero_grad(set_to_none=True)
loss.backward()
grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
opt.step()
moved = sum(1 for a, b in zip(before, model.parameters()) if not torch.equal(a, b.detach()))
check("loss is finite", torch.isfinite(loss).item(), f"{loss.item():+.4f}")
check("gradients reach most tensors", grads > len(before) * 0.9, f"{grads}/{len(before)}")
check("optimizer updates weights", moved > len(before) * 0.9, f"{moved}/{len(before)}")

# ---------------------------------------------------------------- config
print("\nconfig")
cfg = Config.from_mapping({
    "out_dir": "/tmp/ink3d_test",
    "mode": "full_3d",
    "patch_size": [256, 256, 256],
    "batch_size": 2,
    "num_iterations": 250000,
    "seed": 27,
    "warmup_steps": 5000,
    "bce_label_smoothing": 0.1,
    "force_full_supervision": True,
    "targets": {"ink": {"out_channels": 1, "activation": "none"}},
    "loss": {"terms": [{"name": "LabelSmoothedDCAndBCELoss", "weight_ce": 1.0, "weight_dice": 1.0}]},
    "ema": {"enabled": True, "decay": 0.9995, "start_step": 1000},
    "dynamic_label": {"enabled": True, "kind": "self_distill",
                      "primary_ckpt": "/tmp/a.pth", "ensemble_ckpt": "/tmp/b.pth",
                      "primary_threshold": 0.17647, "ensemble_threshold": 0.15686,
                      "mean_hi": 105.0, "std_lo": 30.0, "tta": True,
                      "input_mask_threshold": 50.0},
    "extra_patches": {"enabled": True, "fraction": 0.25, "jitter": 1024,
                      "coords_xyz": [[22167, 8793, 37628]]},
    "datasets": [{"volume_path": "s3://bucket/vol.zarr/", "segments_path": "/data/seg",
                  "volume_scale": 0}],
})
check("parses the reference config", cfg.patch_size == (256, 256, 256) and cfg.seed == 27)
check("percentile defaults are 1/99", (cfg.percentile_lower, cfg.percentile_upper) == (1.0, 99.0))
check("self-distillation thresholds", cfg.self_distill.primary_threshold == 0.17647)
check("force_full_supervision parsed", cfg.force_full_supervision is True)
check("bf16 maps to torch dtype", cfg.torch_dtype is torch.bfloat16)

try:
    Config.from_mapping({"out_dir": "/tmp/x", "mode": "flat", "datasets": []})
    check("unsupported mode rejected", False)
except ValueError:
    check("unsupported mode rejected", True)

try:
    Config.from_mapping({"out_dir": "/tmp/x", "mode": "full_3d", "datasets": [],
                         "force_full_supervision": True})
    check("full supervision without a teacher rejected", False)
except ValueError:
    check("full supervision without a teacher rejected", True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print("  failed:", name)
sys.exit(1 if FAIL else 0)
