"""3D U-Net for ink detection and surface prediction.

The layout mirrors the nnU-Net "3d_fullres" autoconfigure result for a
256^3 patch with one input channel:

    stage      0     1     2     3     4     5     6
    stride   1,1,1 2,2,2 2,2,2 2,2,2 2,2,2 2,2,2 2,2,2
    features   32    64   128   256   320   320   320
    blocks      1     3     4     6     6     6     6

The encoder is residual (BasicBlockD) and the decoder is not (plain conv
blocks). That asymmetry is deliberate: it is what the released checkpoints
were trained with, and the module names below are chosen so their
state_dicts load without remapping.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURES = (32, 64, 128, 256, 320, 320, 320)
BLOCKS = (1, 3, 4, 6, 6, 6, 6)
STRIDES = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
KERNEL = (3, 3, 3)

# Six downsampling stages divide the input by 64, and InstanceNorm needs more
# than one spatial element to normalise over. Anything below this collapses the
# bottleneck to 1^3 and fails inside the norm layer.
MIN_INPUT_SIZE = 128


def _same_padding(kernel):
    return tuple((k - 1) // 2 for k in kernel)


class ConvNormAct(nn.Module):
    """conv -> InstanceNorm3d(affine) -> LeakyReLU, any of the last two optional.

    InstanceNorm3d is built without running stats, so a norm layer contributes
    exactly two tensors (weight, bias) to the state_dict and no buffers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel=KERNEL,
        stride=(1, 1, 1),
        *,
        bias: bool = True,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=tuple(kernel),
            stride=tuple(stride),
            padding=_same_padding(kernel),
            bias=bias,
        )
        self.norm = (
            nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5) if norm else None
        )
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.nonlin is not None:
            x = self.nonlin(x)
        return x


class BasicBlockD(nn.Module):
    """Residual block. The stride and any channel change live in conv1.

    conv2 has no activation of its own; the block activates once after the
    residual add. The skip path only becomes a module when it has work to do
    (downsample, project, or both), which keeps the state_dict free of
    identity-shaped entries.
    """

    def __init__(self, in_channels: int, out_channels: int, stride) -> None:
        super().__init__()
        stride = tuple(stride)
        self.conv1 = ConvNormAct(in_channels, out_channels, KERNEL, stride)
        self.conv2 = ConvNormAct(out_channels, out_channels, KERNEL, (1, 1, 1), act=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        downsamples = any(s != 1 for s in stride)
        projects = in_channels != out_channels
        if downsamples or projects:
            ops = []
            if downsamples:
                ops.append(nn.AvgPool3d(stride, stride))
            if projects:
                # bias=False here regardless of the main path's bias setting.
                ops.append(
                    ConvNormAct(
                        in_channels,
                        out_channels,
                        (1, 1, 1),
                        (1, 1, 1),
                        bias=False,
                        act=False,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.conv2(self.conv1(x))
        out = out + residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    def __init__(self, n_blocks: int, in_channels: int, out_channels: int, stride):
        super().__init__()
        blocks = [BasicBlockD(in_channels, out_channels, stride)]
        blocks += [
            BasicBlockD(out_channels, out_channels, (1, 1, 1))
            for _ in range(n_blocks - 1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class StackedConvBlocks(nn.Module):
    def __init__(self, n_convs: int, in_channels: int, out_channels: int):
        super().__init__()
        convs = [ConvNormAct(in_channels, out_channels)]
        convs += [ConvNormAct(out_channels, out_channels) for _ in range(n_convs - 1)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.output_channels = list(FEATURES)
        self.strides = list(STRIDES)

        self.stem = StackedConvBlocks(1, in_channels, FEATURES[0])

        stages = []
        prev = FEATURES[0]
        for i, features in enumerate(FEATURES):
            stages.append(StackedResidualBlocks(BLOCKS[i], prev, features, STRIDES[i]))
            prev = features
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    """Transpose-conv upsampling with concatenated skips.

    Segmentation heads are built for all six stages even when deep supervision
    is off, matching the checkpoints: a run trained with deep supervision has
    to stay loadable by a model running without it.
    """

    def __init__(self, encoder: Encoder, num_classes: int, *, n_conv_per_stage: int = 1):
        super().__init__()
        n_stages = len(encoder.output_channels)

        transpconvs, stages, seg_layers = [], [], []
        for s in range(1, n_stages):
            below = encoder.output_channels[-s]
            skip = encoder.output_channels[-(s + 1)]
            stride = encoder.strides[-s]

            transpconvs.append(
                nn.ConvTranspose3d(below, skip, kernel_size=stride, stride=stride, bias=True)
            )
            stages.append(StackedConvBlocks(n_conv_per_stage, skip * 2, skip))
            seg_layers.append(nn.Conv3d(skip, num_classes, kernel_size=1, bias=True))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.deep_supervision = False

    def forward(self, skips):
        x = skips[-1]
        outputs = []
        for i, (up, stage) in enumerate(zip(self.transpconvs, self.stages)):
            x = up(x)
            x = torch.cat((x, skips[-(i + 2)]), dim=1)
            x = stage(x)
            if self.deep_supervision or i == len(self.stages) - 1:
                outputs.append(self.seg_layers[i](x))
        outputs.reverse()
        return outputs if self.deep_supervision else outputs[0]


class InkUNet(nn.Module):
    """Encoder plus one decoder per target.

    A single-target model still nests its decoder under ``task_decoders`` so
    that the parameter names match multi-target checkpoints.
    """

    def __init__(self, in_channels: int = 1, targets=("ink",), out_channels: int = 1):
        super().__init__()
        self.shared_encoder = Encoder(in_channels)
        self.task_decoders = nn.ModuleDict(
            {name: Decoder(self.shared_encoder, out_channels) for name in targets}
        )
        self.targets = list(targets)

    @property
    def deep_supervision(self) -> bool:
        return any(d.deep_supervision for d in self.task_decoders.values())

    @deep_supervision.setter
    def deep_supervision(self, value: bool) -> None:
        for decoder in self.task_decoders.values():
            decoder.deep_supervision = bool(value)

    def forward(self, x):
        spatial = tuple(int(v) for v in x.shape[-3:])
        if any(size < MIN_INPUT_SIZE for size in spatial):
            raise ValueError(
                f"input {spatial} is too small: six downsampling stages need at "
                f"least {MIN_INPUT_SIZE} voxels per axis (training patches are "
                f"{256}^3). Smaller inputs collapse the bottleneck to a single "
                "voxel, which InstanceNorm cannot normalise."
            )
        skips = self.shared_encoder(x)
        outputs = {name: dec(skips) for name, dec in self.task_decoders.items()}
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs


class InitWeightsHe:
    """Kaiming-normal init with the negative slope used for training runs.

    Only relevant when training from scratch; loading a checkpoint overwrites
    everything this touches.
    """

    def __init__(self, neg_slope: float = 0.2) -> None:
        self.neg_slope = neg_slope

    def __call__(self, module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_model(in_channels: int = 1, out_channels: int = 1, targets=("ink",)) -> InkUNet:
    return InkUNet(in_channels=in_channels, targets=targets, out_channels=out_channels)
