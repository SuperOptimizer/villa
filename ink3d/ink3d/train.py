"""Training loop.

Single-process, single-device. The reference implementation carried a
distributed-training abstraction whose only visible effect at one GPU was to
make the scheduler step twice; that is gone.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from .config import Config
from .loss import DiceBCELoss
from .model import InitWeightsHe, build_model
from .self_distill import build_labeler


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to zero.

    Reimplemented rather than pulled from ``diffusers`` so the dependency
    footprint stays at torch/numpy/zarr; the curve is identical to
    ``get_cosine_schedule_with_warmup``.
    """

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average over the full state_dict.

    Buffers and integer tensors are copied rather than interpolated; only
    floating-point entries decay.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995, start_step: int = 1000):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        if step < self.start_step:
            # Track the weights exactly until the average starts, so the EMA
            # never carries the random initialisation forward.
            self.module.load_state_dict(model.state_dict())
            return
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            shadow = ema_state[name]
            if shadow.is_floating_point():
                shadow.lerp_(value.detach().to(shadow.dtype), 1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_dict(self):
        return self.module.state_dict()


# --------------------------------------------------------------------------
# batch preparation
# --------------------------------------------------------------------------

def prepare_loss_inputs(
    logits: torch.Tensor,
    batch: dict,
    *,
    force_full_supervision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align predictions with targets and build the ignore mask.

    With ``force_full_supervision`` every voxel is supervised, because the
    teacher labels all of them. Validation deliberately does not pass this
    flag: held-out metrics stay scoped to genuinely labelled voxels.
    """
    target = batch["inklabels"]
    spatial = tuple(int(v) for v in batch["image"].shape[-3:])
    if tuple(int(v) for v in logits.shape[-3:]) != spatial:
        logits = torch.nn.functional.interpolate(
            logits, size=spatial, mode="trilinear", align_corners=True
        )

    if force_full_supervision:
        ignore = torch.zeros_like(target)
    else:
        ignore = (batch["supervision_mask"] <= 0).to(dtype=target.dtype)
    return logits, target, ignore


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

def train(
    config: Config,
    train_loader,
    val_loader=None,
    *,
    device: torch.device | None = None,
) -> nn.Module:
    """Run training and return the trained model."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(config.seed)

    model = build_model(
        in_channels=config.in_channels, out_channels=config.out_channels
    )
    model.apply(InitWeightsHe(neg_slope=0.2))
    if config.checkpoint is not None:
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
    model.to(device)

    criterion = DiceBCELoss(
        weight_ce=config.weight_ce,
        weight_dice=config.weight_dice,
        bce_label_smoothing=config.bce_label_smoothing,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay,
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.num_iterations
    )
    ema = (
        ModelEMA(model, config.ema.decay, config.ema.start_step)
        if config.ema.enabled
        else None
    )
    labeler = build_labeler(config, device)

    amp_dtype = config.torch_dtype
    use_amp = device.type == "cuda" and amp_dtype is not torch.float32

    config.out_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)
    model.train()

    for step in range(1, config.num_iterations + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        if labeler is not None:
            batch["inklabels"] = labeler(
                batch["raw_image"],
                batch["image"],
                batch["raw_mean"],
                batch["raw_std"],
            )

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(batch["image"])

        # Loss runs in fp32 regardless of the forward dtype.
        logits, target, ignore = prepare_loss_inputs(
            logits.float(), batch, force_full_supervision=config.force_full_supervision
        )
        loss = criterion(logits, target.float(), ignore)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if ema is not None and step % config.ema.update_every_steps == 0:
            ema.update(model, step)

        if step % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:>7d}  loss {loss.item():+.4f}  lr {lr:.2e}", flush=True)

        if val_loader is not None and step % config.val_every == 0:
            evaluate(
                ema.module if (ema and config.ema.validate) else model,
                val_loader,
                criterion,
                device,
                max_steps=config.val_steps,
            )
            model.train()

        if step % config.save_every == 0:
            save_checkpoint(config.out_dir / f"ckpt_{step:06d}.pth", model, ema, step)

    save_checkpoint(config.out_dir / "ckpt_final.pth", model, ema, config.num_iterations)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, max_steps: int = 8) -> float:
    """Mean validation loss over a bounded number of batches."""
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(loader):
        if index >= max_steps:
            break
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["image"]).float()
        # Validation keeps the stored supervision mask.
        logits, target, ignore = prepare_loss_inputs(
            logits, batch, force_full_supervision=False
        )
        total += criterion(logits, target.float(), ignore).item()
        count += 1
    mean = total / max(count, 1)
    print(f"  val loss {mean:+.4f}  ({count} batches)", flush=True)
    return mean


def save_checkpoint(path: Path, model, ema, step: int) -> None:
    payload = {"step": step, "model": model.state_dict()}
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)
    print(f"  saved {path.name}", flush=True)
