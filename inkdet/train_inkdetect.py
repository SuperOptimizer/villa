import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import segmentation_models_pytorch as smp
import zarr
import cv2
import random

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from scipy.ndimage import rotate, zoom, gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
import glob

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, rotate, convolve1d
from scipy.interpolate import RegularGridInterpolator
import random

from torch.cuda.amp import GradScaler, autocast

from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.optim import AdamWFp8

THISDIR = os.path.abspath(os.path.dirname(__file__))
VESUVIUS_ROOT = "/vesuvius"
ZARR_PATH = f'{VESUVIUS_ROOT}/fragments.zarr'
MASKS_PATH = f'{VESUVIUS_ROOT}/train_scrolls'
OUTPUT_PATH = f'{VESUVIUS_ROOT}/inkdet_outputs/'

CHUNK_SIZE = 64
STRIDE = 128
ISO_THRESHOLD = 64
NUM_EPOCHS = 2000
LEARNING_RATE = 3e-4  # Increased from 3e-5
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01  # Increased from 1e-6
NUM_WORKERS = 16
SEED = 42
VALIDATION_SPLIT = 0.05
AUGMENT_CHANCE = 0.5
INKDETECT_MEAN = .1

OUTPUT_SIZE = 16  # Changed from 4 to 16

BATCH_SIZE = 40

# whether to randomly offset the y x dimensions of our training data so taht we arent always yielding the same
# CHUNK_SIZE aligned chunk
CHUNK_RANDOM_OFFSET = True

COMPILE_FULLGRAPH=True

def preprocess_chunk(chunk):
    chunk = skimage.exposure.equalize_hist(chunk / 255.0)
    chunk[chunk < ISO_THRESHOLD / 255.0] = 0.0
    return chunk


import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import random


class VolumetricAugmentations:
    """3D augmentations for volumetric data"""

    def __init__(self):
        pass

    def elastic_transform_3d(self, volume, mask, alpha=500, sigma=20):
        """3D Elastic deformation"""
        if random.random() < .1:
            shape = volume.shape

            # Generate random displacement fields
            random_state = np.random.RandomState(None)
            dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

            # Create coordinate arrays
            z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')

            # Add displacements
            indices = np.reshape(z + dz, (-1, 1)), \
                np.reshape(y + dy, (-1, 1)), \
                np.reshape(x + dx, (-1, 1))

            # Apply transformation
            volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
            mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

        return volume, mask

    def grid_distortion_3d(self, volume, mask, num_steps=5, distort_limit=0.3):
        """3D Grid distortion"""
        if random.random() < .1:
            shape = volume.shape

            # Create grid of control points
            grid_z = np.linspace(0, shape[0] - 1, num_steps)
            grid_y = np.linspace(0, shape[1] - 1, num_steps)
            grid_x = np.linspace(0, shape[2] - 1, num_steps)

            # Random displacements for control points
            distort_z = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
            distort_y = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
            distort_x = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))

            # Create coordinate arrays
            z_coords = np.arange(shape[0])
            y_coords = np.arange(shape[1])
            x_coords = np.arange(shape[2])

            # Interpolate displacements to full resolution
            # Create interpolators for each displacement field
            f_z = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_z)
            f_y = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_y)
            f_x = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_x)

            # Create full resolution mesh
            zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)

            # Get interpolated displacements
            dz = f_z(points).reshape(shape) * shape[0]
            dy = f_y(points).reshape(shape) * shape[1]
            dx = f_x(points).reshape(shape) * shape[2]

            # Apply displacements
            indices = np.reshape(zz + dz, (-1, 1)), \
                np.reshape(yy + dy, (-1, 1)), \
                np.reshape(xx + dx, (-1, 1))

            volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
            mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

        return volume, mask

    def anisotropic_gaussian_blur_3d(self, volume):
        """Gaussian blur with different sigma per axis"""
        if random.random() < AUGMENT_CHANCE:
            # Different blur amounts for different axes
            # Z-axis often has lower resolution, so less blur
            sigma_z = random.uniform(0.5, 1.0)
            sigma_xy = random.uniform(0.5, 1.0)

            volume = gaussian_filter(volume, sigma=(sigma_z, sigma_xy, sigma_xy))

        return volume

    def random_gamma_3d(self, volume, gamma_limit=(0.8, 1.2)):
        """Random gamma correction"""
        if random.random() < AUGMENT_CHANCE:
            gamma = random.uniform(gamma_limit[0], gamma_limit[1])

            # Avoid numerical issues with values close to 0
            volume = np.clip(volume, 1e-7, 1)
            volume = np.power(volume, gamma)

        return volume

    def random_flip_3d(self, volume, mask):
        """Random flips along each axis"""
        if random.random() < AUGMENT_CHANCE:
            volume = np.flip(volume, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < AUGMENT_CHANCE:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < AUGMENT_CHANCE:
            volume = np.flip(volume, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return volume, mask

    def random_rotation_3d(self, volume, mask):
        """Random rotation in 3D space"""
        if random.random() < AUGMENT_CHANCE:
            # Random angles for each axis
            angle_x = random.uniform(-180, 180)
            angle_y = random.uniform(-180, 180)
            angle_z = random.uniform(-180, 180)

            # Rotate volume
            volume = rotate(volume, angle_x, axes=(1, 2), reshape=False, order=1)
            volume = rotate(volume, angle_y, axes=(0, 2), reshape=False, order=1)
            volume = rotate(volume, angle_z, axes=(0, 1), reshape=False, order=1)

            # Rotate mask with nearest neighbor
            mask = rotate(mask, angle_x, axes=(1, 2), reshape=False, order=0)
            mask = rotate(mask, angle_y, axes=(0, 2), reshape=False, order=0)
            mask = rotate(mask, angle_z, axes=(0, 1), reshape=False, order=0)

        return volume, mask

    def random_brightness_contrast_3d(self, volume):
        """Adjust brightness and contrast"""
        if random.random() < AUGMENT_CHANCE:
            # Brightness
            brightness = random.uniform(-0.2, 0.2)
            volume = volume + brightness

            # Contrast
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(volume)
            volume = (volume - mean) * contrast + mean

        return volume

    def random_intensity_shift_3d(self, volume):
        """Fast intensity shifting with spatial variation"""
        if random.random() < AUGMENT_CHANCE:
            # Create smooth random shift field
            z_size, y_size, x_size = volume.shape
            # Use larger blocks for efficiency
            block_size = 16
            shift_map = np.random.uniform(-0.1, 0.1,
                                          (z_size // block_size + 1, y_size // block_size + 1,
                                           x_size // block_size + 1))

            # Upsample efficiently using repeat
            shift_map = np.repeat(np.repeat(np.repeat(shift_map, block_size, axis=0),
                                            block_size, axis=1), block_size, axis=2)
            shift_map = shift_map[:z_size, :y_size, :x_size]

            volume = volume + shift_map
        return volume

    def motion_blur_z_axis(self, volume):
        """Fast motion blur along Z-axis (scanning direction)"""
        if random.random() < AUGMENT_CHANCE:
            kernel_size = random.choice([3, 5, 7])
            kernel = np.ones(kernel_size) / kernel_size

            # Apply 1D convolution along Z-axis
            volume = convolve1d(volume, kernel, axis=0, mode='reflect')
        return volume

    def gradient_based_dropout(self, volume):
        """Drop regions based on gradient magnitude - targets edges"""
        if random.random() < AUGMENT_CHANCE:
            # Fast gradient approximation
            gz = np.abs(np.diff(volume, axis=0, prepend=volume[0:1]))
            gy = np.abs(np.diff(volume, axis=1, prepend=volume[:, 0:1]))
            gx = np.abs(np.diff(volume, axis=2, prepend=volume[:, :, 0:1]))

            gradient_mag = gz + gy + gx
            threshold = np.percentile(gradient_mag, random.uniform(70, 90))

            # Drop high gradient regions occasionally
            mask = gradient_mag > threshold
            if random.random() < 0.5:
                volume[mask] *= random.uniform(0.3, 0.7)
        return volume

    def random_noise_3d(self, volume):
        """Add random noise - now uses anisotropic blur"""
        if random.random() < AUGMENT_CHANCE:
            noise_type = random.choice(['gaussian', 'anisotropic_blur'])

            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(0.01, 0.05), volume.shape)
                volume = volume + noise
            else:  # anisotropic_blur
                volume = self.anisotropic_gaussian_blur_3d(volume)

        return volume

    def coarse_dropout_3d(self, volume):
        """3D coarse dropout"""
        if random.random() < AUGMENT_CHANCE:
            h, w, d = volume.shape
            n_holes = random.randint(1, 3)

            for _ in range(n_holes):
                hole_size = int(0.2 * min(h, w, d))
                x = random.randint(0, h - hole_size)
                y = random.randint(0, w - hole_size)
                z = random.randint(0, d - hole_size)

                volume[x:x + hole_size, y:y + hole_size, z:z + hole_size] = 0

        return volume

    def __call__(self, volume, mask):
        """Apply augmentations"""
        # Geometric transforms (apply to both volume and mask)
        volume, mask = self.random_flip_3d(volume, mask)
        volume, mask = self.random_rotation_3d(volume, mask)
        volume, mask = self.elastic_transform_3d(volume, mask)
        volume, mask = self.grid_distortion_3d(volume, mask)

        # Intensity transforms (apply only to volume, NOT mask)
        volume = self.random_brightness_contrast_3d(volume)
        volume = self.random_gamma_3d(volume)
        volume = self.random_intensity_shift_3d(volume)
        volume = self.motion_blur_z_axis(volume)
        volume = self.gradient_based_dropout(volume)
        volume = self.random_noise_3d(volume)
        volume = self.coarse_dropout_3d(volume)
        volume = skimage.exposure.equalize_hist(volume)
        # Ensure values are in valid range
        volume = np.clip(volume, 0, 1)
        mask = np.clip(mask, 0, 1)

        return volume, mask


# New model architecture
class ResBlock3D(nn.Module):
    """Efficient residual block with GroupNorm for better small-batch training"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()  # GELU often works better than ReLU for 3D

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DenseBlock3D(nn.Module):
    """Dense connections for more parameters and gradient flow"""

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
    """Larger encoder-decoder that maps 64³ -> 16³"""

    def __init__(self, in_channels=1, base_channels=32):  # Doubled base channels
        super().__init__()

        # Initial feature extraction with parallel paths
        self.init_conv = nn.Conv3d(in_channels, base_channels, 7, padding=3, bias=False)
        self.init_norm = nn.GroupNorm(16, base_channels)

        # Parallel feature extractors for multi-scale
        self.parallel1 = nn.Conv3d(base_channels, base_channels // 2, 3, padding=1)
        self.parallel2 = nn.Conv3d(base_channels, base_channels // 2, 5, padding=2)

        # Encoder: 64³ -> 32³ -> 16³
        self.enc1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
            DenseBlock3D(base_channels, growth_rate=32, num_layers=4)
        )

        enc1_out = base_channels + 4 * 32  # After DenseBlock
        self.down1 = nn.Conv3d(enc1_out, base_channels * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            DenseBlock3D(base_channels * 2, growth_rate=48, num_layers=6)
        )

        enc2_out = base_channels * 2 + 6 * 48  # After DenseBlock
        self.down2 = nn.Conv3d(enc2_out, base_channels * 4, 4, stride=2, padding=1)

        # Larger bottleneck at 16³
        self.bottleneck = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
            DenseBlock3D(base_channels * 4, growth_rate=64, num_layers=8),
            ResBlock3D(base_channels * 4 + 8 * 64),
            ResBlock3D(base_channels * 4 + 8 * 64),
        )

        bottleneck_out = base_channels * 4 + 8 * 64

        # Multi-head output for ensemble-like predictions
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(bottleneck_out, base_channels * 2, 3, padding=1),
                nn.GroupNorm(16, base_channels * 2),
                nn.GELU(),
                nn.Conv3d(base_channels * 2, 1, 1)
            ) for _ in range(3)
        ])

        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Conv3d(bottleneck_out, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Initial processing
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = F.gelu(x)

        # Parallel paths
        p1 = self.parallel1(x)
        p2 = self.parallel2(x)
        x = torch.cat([p1, p2], dim=1)

        # Encode: 64³ -> 32³ -> 16³
        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x2 = self.enc2(x2)
        x3 = self.down2(x2)

        # Process at 16³
        x = self.bottleneck(x3)

        # Multi-head output with attention weighting
        outputs = torch.stack([head(x) for head in self.output_heads], dim=1)
        weights = self.attention(x).unsqueeze(2)

        # Weighted combination
        return (outputs * weights).sum(dim=1)


class CombinedLoss(nn.Module):
    """Effective loss combination for volumetric segmentation"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Downsample target to 16³
        target = F.interpolate(target.unsqueeze(1), size=(16, 16, 16), mode='trilinear').squeeze(1)
        pred = pred.squeeze(1)

        # Weighted BCE for class imbalance
        pos_weight = (target == 0).sum() / (target == 1).sum().clamp(min=1)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

        # Dice loss for region overlap
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)

        # Focal loss for hard examples
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal = ((1 - p_t) ** 2 * ce_loss).mean()

        return 0.3 * bce + 0.4 * dice + 0.3 * focal


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class ZarrDataset(Dataset):
    def __init__(self, fragment_ids, mode):
        self.zarr_store = zarr.open(ZARR_PATH, mode='r')
        self.fragment_ids = fragment_ids
        self.mode = mode

        self.ink_mask_cache_2d = {}

        self.chunks = []
        for frag_id in fragment_ids:
            self._build_chunks_for_fragment(frag_id)
        print(f"Loaded {len(self.chunks)} chunks for {mode} dataset")

        if mode == 'train':
            self.augment = VolumetricAugmentations()
        else:
            self.augment = None

    def _load_and_cache_mask_2d(self, frag_id):
        if frag_id in self.ink_mask_cache_2d:
            return self.ink_mask_cache_2d[frag_id]

        # Try PNG first, then TIFF
        ink_mask_path = f"{THISDIR}/all_labels/{frag_id}_inklabels.png"
        if not os.path.exists(ink_mask_path):
            ink_mask_path = f"{THISDIR}/all_labels/{frag_id}_inklabels.tiff"
            if not os.path.exists(ink_mask_path):
                return None

        ink_mask = cv2.imread(ink_mask_path, 0)
        if ink_mask is not None:
            h, w = self.zarr_store[frag_id].shape[1:]
            ink_mask = cv2.resize(ink_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self.ink_mask_cache_2d[frag_id] = ink_mask
        return ink_mask

    def _build_chunks_for_fragment(self, frag_id):
        frag_data = self.zarr_store[frag_id]
        d, h, w = frag_data.shape

        ink_mask_2d = self._load_and_cache_mask_2d(frag_id)
        if ink_mask_2d is None:
            print(f"No ink mask found for {frag_id}, skipping")
            return

        frag_mask = cv2.imread(f"{MASKS_PATH}/{frag_id}/{frag_id}_mask.png", 0)
        frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for y in range(0, h - CHUNK_SIZE, STRIDE):
            for x in range(0, w - CHUNK_SIZE, STRIDE):
                chunk_frag_mask = frag_mask[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
                if np.all(chunk_frag_mask == 0):
                    continue

                # Check 2D ink mask for chunk selection
                chunk_ink_2d = ink_mask_2d[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
                has_ink = np.mean(chunk_ink_2d) > INKDETECT_MEAN
                if has_ink or self.mode == 'valid' or random.randint(1, 50) == 1:
                    self.chunks.append([frag_id, x, y])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        frag_id, x, y = self.chunks[idx]

        if self.mode == 'train' and (CHUNK_RANDOM_OFFSET and
                                     STRIDE < x < self.zarr_store[frag_id].shape[2] - STRIDE and
                                     STRIDE < y < self.zarr_store[frag_id].shape[1] - STRIDE):
            yoff, xoff = random.randint(-STRIDE, STRIDE), random.randint(-STRIDE, STRIDE)
            ystart = yoff + y
            xstart = xoff + x
        else:
            ystart = y
            xstart = x

        chunk_3d = self.zarr_store[frag_id][:, ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE].astype(
            np.float32)
        ink_mask_2d = self.ink_mask_cache_2d[frag_id][ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE].astype(
            np.float32)

        ink_mask_3d = np.broadcast_to(ink_mask_2d[np.newaxis, :, :], (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)).copy()
        ink_mask_3d[chunk_3d < ISO_THRESHOLD] = 0

        chunk_3d = preprocess_chunk(chunk_3d)

        ink_mask_3d = ink_mask_3d / 255.0

        if self.augment is not None:
            chunk_3d, ink_mask_3d = self.augment(chunk_3d, ink_mask_3d)

        chunk_tensor = torch.from_numpy(chunk_3d).float()
        mask_tensor = torch.from_numpy(ink_mask_3d).float()

        if self.mode == 'valid':
            return chunk_tensor, mask_tensor, (xstart, ystart, xstart + CHUNK_SIZE, ystart + CHUNK_SIZE)
        return chunk_tensor, mask_tensor


class InkDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = SimpleVolumetricModel()
        self.loss_fn = CombinedLoss()

        # Track metrics
        self.val_losses = []

        print(f"Training Simple Volumetric Model from scratch")
        print(f"Processing {CHUNK_SIZE}³ chunks -> {OUTPUT_SIZE}³ outputs")

    def forward(self, x):
        # Add channel dimension if needed
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, D, H, W) -> (B, 1, D, H, W)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        #torch.compiler.cudagraph_mark_step_begin()
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        if torch.isnan(loss):
            print("Loss nan encountered")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #torch.compiler.cudagraph_mark_step_begin()
        x, y, xyxy = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.val_losses.append(loss.item())
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Simple early stopping signal
        if len(self.val_losses) > 10:
            recent_avg = sum(self.val_losses[-5:]) / 5
            older_avg = sum(self.val_losses[-10:-5]) / 5
            if recent_avg > older_avg * 0.98:  # Not improving
                self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'] * 0.5)
        self.val_losses = []

    def configure_optimizers(self):
        # AdamW with higher weight decay for better generalization
        optimizer = AdamWFp8(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95)  # Lower beta2 for more stable training
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,  # First cycle length
            T_mult=1,  # Keep cycles same length
            eta_min=MIN_LEARNING_RATE
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Find latest checkpoint
    checkpoint_path = None
    checkpoints = glob.glob(os.path.join(OUTPUT_PATH, f'best_simple_volumetric_*.ckpt'))
    if checkpoints:
        # Sort by epoch number
        checkpoint_path = max(checkpoints, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        print("Starting fresh training - no checkpoint found")

    zarr_store = zarr.open(ZARR_PATH, mode='r')
    all_fragments = list(zarr_store.keys())

    random.shuffle(all_fragments)
    n_valid = int(len(all_fragments) * VALIDATION_SPLIT)
    valid_fragments = []
    train_fragments = all_fragments[n_valid:]

    if '20231005123336' in train_fragments:
        train_fragments.remove('20231005123336')
        # valid_fragments.append('20231005123336')

    print(f"Total fragments: {len(all_fragments)}")
    print(f"Train fragments: {len(train_fragments)}")
    print(f"Valid fragments: {len(valid_fragments)}")

    train_dataset = ZarrDataset(train_fragments, mode='train')
    valid_dataset = ZarrDataset(valid_fragments, mode='valid')

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Valid chunks: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # To this:
    model = InkDetectionModel()
    config = Float8LinearConfig.from_recipe_name("tensorwise")
    convert_to_float8_training(model.model, config=config)  # Apply float8 to the inner model
    model.model = torch.compile(model.model, fullgraph=True, dynamic=False, mode='max-autotune')

    print(f"Using Simple Volumetric Model for 3D ink detection")
    print(f"Input: {CHUNK_SIZE}³ voxel chunks (float32, normalized 0-1)")
    print(f"Output: {OUTPUT_SIZE}³ predictions (logits for ink probability)")
    print(f"All spatial dimensions (Z, Y, X) are processed equally")

    # Uncomment if using Weights & Biases
    # wandb_logger = WandbLogger(project="vesuvius", name=f"simple_volumetric_ink_detection")

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        # logger=wandb_logger,
        default_root_dir=OUTPUT_PATH,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=[
            ModelCheckpoint(
                filename=f'best_simple_volumetric_{{epoch}}',
                dirpath=OUTPUT_PATH,
                save_top_k=-1  # Save all checkpoints
            ),
        ]
    )

    # Resume from checkpoint if available
    trainer.fit(model, train_loader, valid_loader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()