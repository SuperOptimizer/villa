"""Run configuration.

One frozen dataclass tree per run, populated from the same JSON the previous
training scripts used. Only the keys this rewrite actually honours are read;
anything else in the file is ignored rather than silently half-applied.

Deliberately dropped from the old schema:

``dice_label_smoothing``
    A no-op for single-channel targets. The smoothing only ever applied on
    the index-encoded branch of the Dice loss, which a 1-channel ink target
    never reaches. Implementing it here would be decorative.
``use_stitched_forward`` / ``stitch_factor``
    Off in the reference configs.
``model_type`` / ``model_config.autoconfigure``
    There is one architecture. See model.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = True
    decay: float = 0.9995
    start_step: int = 1000
    update_every_steps: int = 1
    validate: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "EMAConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            decay=float(raw.get("decay", 0.9995)),
            start_step=int(raw.get("start_step", 1000)),
            update_every_steps=int(raw.get("update_every_steps", 1)),
            validate=bool(raw.get("validate", True)),
        )


@dataclass(frozen=True)
class SelfDistillConfig:
    """Teacher-generated labels.

    Two frozen copies of an earlier checkpoint. The ensemble is only consulted
    for crops whose *raw* intensity statistics fall in the band described by
    ``mean_hi``/``std_lo``; everything else uses the primary alone. Both
    thresholds are applied to a probability that has already been gated by
    ``input_mask_threshold`` on raw intensity.
    """

    enabled: bool
    primary_ckpt: Path | None
    ensemble_ckpt: Path | None
    primary_threshold: float = 0.17647
    ensemble_threshold: float = 0.15686
    mean_hi: float = 105.0
    std_lo: float = 30.0
    tta: bool = True
    input_mask_threshold: float = 50.0

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "SelfDistillConfig":
        raw = raw or {}
        enabled = bool(raw.get("enabled", False))
        kind = str(raw.get("kind", "self_distill")).strip().lower()
        if enabled and kind != "self_distill":
            raise ValueError(
                f"unsupported dynamic_label.kind {kind!r}; this rewrite implements "
                "'self_distill' only"
            )

        def _path(key: str) -> Path | None:
            value = raw.get(key)
            return Path(value) if value else None

        return cls(
            enabled=enabled,
            primary_ckpt=_path("primary_ckpt"),
            ensemble_ckpt=_path("ensemble_ckpt"),
            primary_threshold=float(raw.get("primary_threshold", 0.17647)),
            ensemble_threshold=float(raw.get("ensemble_threshold", 0.15686)),
            mean_hi=float(raw.get("mean_hi", 105.0)),
            std_lo=float(raw.get("std_lo", 30.0)),
            tta=bool(raw.get("tta", True)),
            input_mask_threshold=float(raw.get("input_mask_threshold", 50.0)),
        )


@dataclass(frozen=True)
class ExtraPatchesConfig:
    """Hand-picked patch centres, mixed in at a fixed rate."""

    enabled: bool = False
    fraction: float = 0.0
    jitter: int = 0
    coords_xyz: tuple[tuple[int, int, int], ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "ExtraPatchesConfig":
        raw = raw or {}
        coords = tuple(
            (int(x), int(y), int(z)) for x, y, z in raw.get("coords_xyz", []) or []
        )
        return cls(
            enabled=bool(raw.get("enabled", False)) and bool(coords),
            fraction=float(raw.get("fraction", 0.0)),
            jitter=int(raw.get("jitter", 0)),
            coords_xyz=coords,
        )


@dataclass(frozen=True)
class DatasetConfig:
    volume_path: str
    segments_path: Path
    volume_scale: int = 0

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "DatasetConfig":
        return cls(
            volume_path=str(raw["volume_path"]),
            segments_path=Path(raw["segments_path"]),
            volume_scale=int(raw.get("volume_scale", 0)),
        )


@dataclass(frozen=True)
class Config:
    out_dir: Path
    datasets: tuple[DatasetConfig, ...]

    patch_size: tuple[int, int, int] = (256, 256, 256)
    batch_size: int = 2
    num_iterations: int = 250_000
    seed: int = 27

    patch_overlap: float = 0.5
    patch_min_labeled_coverage: float = 0.02
    patch_finding_scale: int = 4
    projection_half_thickness: float = 3.0

    learning_rate: float = 0.01
    momentum: float = 0.99
    nesterov: bool = True
    weight_decay: float = 3e-5
    warmup_steps: int = 5000
    grad_clip: float = 0.0

    weight_ce: float = 1.0
    weight_dice: float = 1.0
    bce_label_smoothing: float = 0.1

    percentile_lower: float = 1.0
    percentile_upper: float = 99.0

    mixed_precision: str = "bf16"
    dataloader_workers: int = 8
    prefetch_factor: int = 8

    val_every: int = 200
    val_steps: int = 8
    save_every: int = 1000

    in_channels: int = 1
    out_channels: int = 1
    force_full_supervision: bool = False
    checkpoint: Path | None = None

    ema: EMAConfig = field(default_factory=EMAConfig)
    self_distill: SelfDistillConfig = field(
        default_factory=lambda: SelfDistillConfig(False, None, None)
    )
    extra_patches: ExtraPatchesConfig = field(default_factory=ExtraPatchesConfig)

    @property
    def torch_dtype(self):
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
            self.mixed_precision
        ]

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Config":
        mode = str(raw.get("mode", "full_3d"))
        if mode != "full_3d":
            raise ValueError(
                f"unsupported mode {mode!r}; this rewrite implements 'full_3d' only"
            )

        patch = tuple(int(v) for v in raw.get("patch_size", (256, 256, 256)))
        if len(patch) != 3:
            raise ValueError(f"patch_size must have three entries, got {patch!r}")

        targets: Mapping[str, Any] = raw.get("targets") or {"ink": {}}
        if len(targets) != 1:
            raise ValueError(
                "this rewrite supports exactly one target; "
                f"got {sorted(targets)!r}"
            )
        (target_cfg,) = targets.values()

        loss_terms: Sequence[Mapping[str, Any]] = (raw.get("loss") or {}).get("terms") or [{}]
        if len(loss_terms) != 1:
            raise ValueError("this rewrite supports exactly one loss term")
        loss = loss_terms[0]

        self_distill = SelfDistillConfig.parse(raw.get("dynamic_label"))
        force_full = bool(raw.get("force_full_supervision", False))
        if force_full and not self_distill.enabled:
            raise ValueError(
                "force_full_supervision requires dynamic_label.enabled with "
                "kind='self_distill'"
            )

        checkpoint = raw.get("checkpoint")

        return cls(
            out_dir=Path(raw["out_dir"]),
            datasets=tuple(DatasetConfig.parse(d) for d in raw.get("datasets", ())),
            patch_size=(patch[0], patch[1], patch[2]),
            batch_size=int(raw.get("batch_size", 2)),
            num_iterations=int(raw.get("num_iterations", 250_000)),
            seed=int(raw.get("seed", 27)),
            patch_overlap=float(raw.get("patch_overlap", 0.5)),
            patch_min_labeled_coverage=float(
                raw.get("patch_min_labeled_coverage", 0.02)
            ),
            patch_finding_scale=int(raw.get("patch_finding_scale", 4)),
            projection_half_thickness=float(
                (raw.get("full_3d") or {}).get("projection_half_thickness", 3.0)
            ),
            warmup_steps=int(raw.get("warmup_steps", 5000)),
            grad_clip=float(raw.get("grad_clip", 0.0)),
            weight_ce=float(loss.get("weight_ce", 1.0)),
            weight_dice=float(loss.get("weight_dice", 1.0)),
            bce_label_smoothing=float(raw.get("bce_label_smoothing", 0.1)),
            percentile_lower=float(raw.get("percentile_lower", 1.0)),
            percentile_upper=float(raw.get("percentile_upper", 99.0)),
            mixed_precision=str(raw.get("mixed_precision", "bf16")),
            dataloader_workers=int(raw.get("dataloader_workers", 8)),
            prefetch_factor=int(raw.get("prefetch_factor", 8)),
            val_every=int(raw.get("val_every", 200)),
            val_steps=int(raw.get("val_steps", 8)),
            save_every=int(raw.get("save_every", 1000)),
            in_channels=int(raw.get("in_channels", 1)),
            out_channels=int(target_cfg.get("out_channels", 1)),
            force_full_supervision=force_full,
            checkpoint=Path(checkpoint) if checkpoint else None,
            ema=EMAConfig.parse(raw.get("ema")),
            self_distill=self_distill,
            extra_patches=ExtraPatchesConfig.parse(raw.get("extra_patches")),
        )
