"""Command line entry points.

    python -m ink3d train config.json
    python -m ink3d infer config.json --checkpoint ckpt.pth --volume vol.zarr --out pred.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_train(args) -> int:
    from .config import Config
    from .dataset import build_dataloader
    from .train import train

    config = Config.from_json(args.config)
    device = _device(args.device)
    print(f"training on {device}: {config.num_iterations} iterations, "
          f"batch {config.batch_size}, patch {config.patch_size}")

    loader = build_dataloader(config, num_workers=args.workers)
    print(f"{len(loader.dataset)} patches")

    train(config, loader, device=device)
    return 0


def cmd_infer(args) -> int:
    from .config import Config
    from .data import open_volume
    from .infer import predict_volume, write_multiscale_zarr
    from .model import build_model

    config = Config.from_json(args.config) if args.config else None
    device = _device(args.device)

    model = build_model(
        in_channels=config.in_channels if config else 1,
        out_channels=config.out_channels if config else 1,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("ema", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(device).eval()

    volume = open_volume(args.volume, args.scale)
    print(f"volume {volume.shape} -> {args.out}")

    probability = predict_volume(
        model,
        volume[:],
        patch_size=config.patch_size if config else (256, 256, 256),
        overlap=args.overlap,
        tta=args.tta,
        device=device,
    )
    write_multiscale_zarr(args.out, probability)
    print("done")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ink3d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="train a model")
    train_parser.add_argument("config", type=Path)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--workers", type=int, default=None)
    train_parser.set_defaults(func=cmd_train)

    infer_parser = sub.add_parser("infer", help="run sliding-window inference")
    infer_parser.add_argument("config", type=Path, nargs="?")
    infer_parser.add_argument("--checkpoint", required=True, type=Path)
    infer_parser.add_argument("--volume", required=True)
    infer_parser.add_argument("--out", required=True)
    infer_parser.add_argument("--scale", type=int, default=0)
    infer_parser.add_argument("--overlap", type=float, default=0.5)
    infer_parser.add_argument("--tta", action="store_true")
    infer_parser.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
