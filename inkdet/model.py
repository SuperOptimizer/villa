import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DenseBlock3D(nn.Module):

    def __init__(self, channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.Conv3d(in_ch, growth_rate, 3, padding=1, bias=False),
                nn.GroupNorm(8, growth_rate),
                nn.GELU()
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class SimpleVolumetricModel(nn.Module):

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.init_conv = nn.Conv3d(in_channels, base_channels, 7, padding=3, bias=False)
        self.init_norm = nn.GroupNorm(16, base_channels)
        self.parallel1 = nn.Conv3d(base_channels, base_channels // 2, 3, padding=1)
        self.parallel2 = nn.Conv3d(base_channels, base_channels // 2, 5, padding=2)

        self.enc1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
            DenseBlock3D(base_channels, growth_rate=32, num_layers=4)
        )
        enc1_out = base_channels + 4 * 32
        self.down1 = nn.Conv3d(enc1_out, base_channels * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            DenseBlock3D(base_channels * 2, growth_rate=48, num_layers=6)
        )
        enc2_out = base_channels * 2 + 6 * 48
        self.down2 = nn.Conv3d(enc2_out, base_channels * 4, 4, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
            DenseBlock3D(base_channels * 4, growth_rate=64, num_layers=8),
            ResBlock3D(base_channels * 4 + 8 * 64),
            ResBlock3D(base_channels * 4 + 8 * 64),
        )
        bottleneck_out = base_channels * 4 + 8 * 64

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(bottleneck_out, base_channels * 2, 3, padding=1),
                nn.GroupNorm(16, base_channels * 2),
                nn.GELU(),
                nn.Conv3d(base_channels * 2, 1, 1)
            ) for _ in range(3)
        ])

        self.attention = nn.Sequential(
            nn.Conv3d(bottleneck_out, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = F.gelu(x)
        p1 = self.parallel1(x)
        p2 = self.parallel2(x)
        x = torch.cat([p1, p2], dim=1)
        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x2 = self.enc2(x2)
        x3 = self.down2(x2)
        x = self.bottleneck(x3)
        outputs = torch.stack([head(x) for head in self.output_heads], dim=1)
        weights = self.attention(x).unsqueeze(2)
        return (outputs * weights).sum(dim=1)


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        target = F.interpolate(target.unsqueeze(1), size=(16, 16, 16), mode='trilinear').squeeze(1)
        pred = pred.squeeze(1)
        pos_weight = (target == 0).sum() / (target == 1).sum().clamp(min=1)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal = ((1 - p_t) ** 2 * ce_loss).mean()
        return 0.3 * bce + 0.4 * dice + 0.3 * focal