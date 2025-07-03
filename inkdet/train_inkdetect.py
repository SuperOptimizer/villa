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

THISDIR = os.path.abspath(os.path.dirname(__file__))
VESUVIUS_ROOT = "/vesuvius"
ZARR_PATH = f'{VESUVIUS_ROOT}/fragments.zarr'
MASKS_PATH = f'{VESUVIUS_ROOT}/train_scrolls'
OUTPUT_PATH = f'{VESUVIUS_ROOT}/inkdet_outputs/'

CHUNK_SIZE = 64
STRIDE = 64
ISO_THRESHOLD = 16
NUM_EPOCHS = 2000
LEARNING_RATE = 3e-5
MIN_LEARNING_RATE=1e-6
WEIGHT_DECAY = 1e-6
NUM_WORKERS=24
SEED=42
VALIDATION_SPLIT=0.05
AUGMENT_CHANCE = 0.5
INKDETECT_MEAN = .3

#for resnet50 and 24gb vram, bs=24
#for resnet10 and 24gb vram, bs=128

RESNET_DEPTH=10
RESNET_NORM=True
OUTPUT_SIZE=4

BATCH_SIZE=128

#whether to randomly offset the y x dimensions of our training data so taht we arent always yielding the same
#CHUNK_SIZE aligned chunk
CHUNK_RANDOM_OFFSET = True

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

def conv3x3x3(in_planes, out_planes, stride):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, downsample):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, downsample):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes, 1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, 1)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VolumetricResNet(nn.Module):
    def __init__(self, block, layers, base_planes=64):
        super().__init__()
        self.inplanes = base_planes

        # Feature dimensions for each stage
        self.planes = [base_planes, base_planes * 2, base_planes * 4, base_planes * 8]

        # Initial convolution - symmetric for all dimensions
        self.conv1 = nn.Conv3d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling - symmetric for all dimensions
        # Note: This is separate from the downsampling in residual blocks
        # MaxPool reduces spatial dims in the main path
        # Downsample adjusts residual connections to match
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(block, self.planes[0], layers[0])
        self.layer2 = self._make_layer(block, self.planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.planes[3], layers[3], stride=2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # Downsample is needed when:
        # 1. Spatial dimensions change (stride != 1)
        # 2. Number of channels change (inplanes != planes * expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []

        # First block in layer - may need downsampling
        layers.append(
            block(
                in_planes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample
            )
        )

        # Update inplanes for subsequent blocks
        self.inplanes = planes * block.expansion

        # Remaining blocks - no downsampling needed (same dims)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv + bn + relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)


        # ResNet stages
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def create_volumetric_resnet(depth, **kwargs):
    assert depth in [10, 18, 34, 50, 101, 152], f"Unsupported depth: {depth}"
    if depth == 10:
        return VolumetricResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif depth == 18:
        return VolumetricResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif depth == 34:
        return VolumetricResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif depth == 50:
        return VolumetricResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif depth == 101:
        return VolumetricResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif depth == 152:
        return VolumetricResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class Decoder3D(nn.Module):
    def __init__(self, encoder_dims, output_size):
        super().__init__()
        self.output_size = output_size

        # Build decoder layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    encoder_dims[i] + encoder_dims[i - 1],
                    encoder_dims[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm3d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))
        ])

        # Final logit layer
        self.logit = nn.Conv3d(encoder_dims[0], 1, kernel_size=1)

    def forward(self, feature_maps):
        # Decoder with skip connections
        for i in range(len(feature_maps) - 1, 0, -1):
            # Upsample higher level features
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode='trilinear', align_corners=False)
            # Concatenate with lower level features
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            # Process concatenated features
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        # Generate logits
        x = self.logit(feature_maps[0])

        # Resize to desired output size
        if x.shape[-3:] != (self.output_size, self.output_size, self.output_size):
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size, self.output_size),
                mode='trilinear',
                align_corners=False
            )

        return x



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
                if np.any(chunk_frag_mask == 0):
                    continue

                # Check 2D ink mask for chunk selection
                chunk_ink_2d = ink_mask_2d[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
                has_ink = np.mean(chunk_ink_2d) > INKDETECT_MEAN
                if has_ink or self.mode == 'valid' or random.randint(1, 5000) == 1:
                    self.chunks.append([frag_id, x, y])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        frag_id, x, y = self.chunks[idx]

        if self.mode == 'train' and (CHUNK_RANDOM_OFFSET and
                STRIDE < x < self.zarr_store[frag_id].shape[2] - STRIDE and
                STRIDE < y < self.zarr_store[frag_id].shape[1] - STRIDE):
            yoff, xoff = random.randint(-STRIDE,STRIDE), random.randint(-STRIDE,STRIDE)
            ystart = yoff + y
            xstart = xoff + x
        else:
            ystart = y
            xstart = x


        chunk_3d = self.zarr_store[frag_id][:, ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE].astype(np.float32)
        ink_mask_2d = self.ink_mask_cache_2d[frag_id][ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE].astype(np.float32)

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

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        self.backbone = create_volumetric_resnet(depth=RESNET_DEPTH)

        print(f"Training volumetric ResNet{RESNET_DEPTH} from scratch")

        # Get encoder dimensions by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.rand(1, 1, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)
            encoder_outputs = self.backbone(dummy_input)
            encoder_dims = [x.size(1) for x in encoder_outputs]
            print(f"ResNet{RESNET_DEPTH} encoder dimensions: {encoder_dims}")
            print(f"Processing {CHUNK_SIZE}³ chunks -> {OUTPUT_SIZE}³ outputs")

            # Show spatial dimension flow through network
            print("\nDimension flow through network:")
            print(f"Input: 1x{CHUNK_SIZE}x{CHUNK_SIZE}x{CHUNK_SIZE}")
            print(f"After conv1 (stride=2): {self.backbone.inplanes}x32x32x32")
            for i, feat in enumerate(encoder_outputs):
                print(f"After layer{i + 1}: {feat.shape[1]}x{feat.shape[2]}x{feat.shape[3]}x{feat.shape[4]}")

        # Create decoder
        self.decoder = Decoder3D(encoder_dims=encoder_dims, output_size=OUTPUT_SIZE)

        if RESNET_NORM:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        # Add channel dimension if needed
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, D, H, W) -> (B, 1, D, H, W)

        if RESNET_NORM:
            x = self.normalization(x)

        # Get feature maps from backbone
        feat_maps = self.backbone(x)

        # Decode to 3D mask
        pred_mask = self.decoder(feat_maps)

        return pred_mask  # (B, 1, 4, 4, 4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        outputs = self(x)

        # Downsample target to match output size
        y = F.interpolate(
            y.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE),
            mode='trilinear',
            align_corners=False
        )

        loss = self.loss_func(outputs, y)

        if torch.isnan(loss):
            print("Loss nan encountered")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, xyxy = batch

        outputs = self(x)

        # Downsample target to match output size
        y = F.interpolate(
            y.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE),
            mode='trilinear',
            align_corners=False
        )

        loss = self.loss_func(outputs, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, NUM_EPOCHS, eta_min=MIN_LEARNING_RATE
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine
        )
        return [optimizer], [scheduler]


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super().__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Find latest checkpoint
    checkpoint_path = None
    checkpoints = glob.glob(os.path.join(OUTPUT_PATH, f'best_volumetric_resnet{RESNET_DEPTH}_*.ckpt'))
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
        #valid_fragments.append('20231005123336')

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

    model = InkDetectionModel()
    model = torch.compile(model,fullgraph=False, dynamic=False)

    print(f"Using Volumetric ResNet{RESNET_DEPTH} model for 3D ink detection")
    print(f"Input: {CHUNK_SIZE}³ voxel chunks (float32, normalized 0-1)")
    print(f"Output: {OUTPUT_SIZE}³ predictions (logits for ink probability)")
    print(f"All spatial dimensions (Z, Y, X) are processed equally")

    # Uncomment if using Weights & Biases
    # wandb_logger = WandbLogger(project="vesuvius", name=f"volumetric_resnet{RESNET_DEPTH}_ink_detection")

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
                filename=f'best_volumetric_resnet{RESNET_DEPTH}_{{epoch}}',
                dirpath=OUTPUT_PATH,
                save_top_k=-1  # Save all checkpoints
            ),
        ]
    )

    # Resume from checkpoint if available
    trainer.fit(model, train_loader, valid_loader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()