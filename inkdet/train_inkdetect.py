from timesformer_pytorch import TimeSformer

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from warmup_scheduler import GradualWarmupScheduler

VESUVIUS_ROOT="/vesuvius"

class CFG:
    # Paths
    zarr_path = f'{VESUVIUS_ROOT}/fragments.zarr'
    masks_path = f'{VESUVIUS_ROOT}/train_scrolls'
    outputs_path = f'{VESUVIUS_ROOT}/inkdet_outputs/'
    model_dir = f'{outputs_path}/models/'

    # Model
    size = 64
    stride = 32  # For chunk sampling

    # Training

    # bs=16 for 64x64x64 fits well into 8gb vram and 32gb system ram
    batch_size = 16
    epochs = 20
    lr = 3e-5
    min_lr = 1e-6
    weight_decay = 1e-6
    num_workers = 2
    seed = 42

    # Validation split
    valid_ratio = 0.1  # Use 10% of fragments for validation


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')


class ZarrDataset(Dataset):
    def __init__(self, zarr_path, fragment_ids, masks_path, size=64, stride=32, mode='train'):
        self.zarr_store = zarr.open(zarr_path, mode='r')
        self.fragment_ids = fragment_ids
        self.masks_path = masks_path
        self.size = size
        self.stride = stride
        self.mode = mode

        # Cache for masks to avoid repeated disk reads
        self.mask_cache = {}

        # Build list of all valid chunks
        self.chunks = []
        for frag_id in fragment_ids:
            self._build_chunks_for_fragment(frag_id)

    def _load_and_cache_mask(self, frag_id):
        """Load mask once and cache it"""
        if frag_id in self.mask_cache:
            return self.mask_cache[frag_id]

        ink_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_inklabels.png")
        if not os.path.exists(ink_mask_path):
            ink_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_inklabels.tiff")
            if not os.path.exists(ink_mask_path):
                return None

        mask = cv2.imread(ink_mask_path, 0)
        if mask is not None:
            h, w = self.zarr_store[frag_id].shape[1:]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.float32) / 255.0
            self.mask_cache[frag_id] = mask
        return mask

    def _build_chunks_for_fragment(self, frag_id):
        # Load fragment shape from zarr
        frag_data = self.zarr_store[frag_id]
        z_size, h, w = frag_data.shape

        # Load and cache masks
        mask = self._load_and_cache_mask(frag_id)
        if mask is None:
            print(f"No ink mask found for {frag_id}, skipping")
            return

        # Load fragment mask
        frag_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_mask.png")
        if not os.path.exists(frag_mask_path):
            print(f"No fragment mask found for {frag_id}, skipping")
            return

        frag_mask = cv2.imread(frag_mask_path, 0)
        if frag_mask is None:
            print(f"Failed to load fragment mask for {frag_id}, skipping")
            return

        frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Generate chunk positions
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                # Check if chunk is entirely outside fragment
                chunk_frag_mask = frag_mask[y:y + self.size, x:x + self.size]
                if np.all(chunk_frag_mask == 0):
                    continue

                # Check if chunk has any ink (>5% coverage)
                chunk_mask = mask[y:y + self.size, x:x + self.size]
                if True:
                #if np.mean(chunk_mask) > 0.05 or self.mode == 'valid':
                    self.chunks.append({
                        'fragment_id': frag_id,
                        'x': x,
                        'y': y,
                        'has_ink': True
                    })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]
        frag_id = chunk_info['fragment_id']
        x, y = chunk_info['x'], chunk_info['y']

        # Load chunk from zarr (all 64 layers) - shape: (z, y, x)
        chunk_3d = self.zarr_store[frag_id][:, y:y + self.size, x:x + self.size]

        # Get mask chunk from cache
        mask = self.mask_cache[frag_id]
        mask_chunk = mask[y:y + self.size, x:x + self.size]

        # Normalize
        chunk_3d = chunk_3d.astype(np.float32) / 255.0

        # Convert to pytorch format
        chunk_tensor = torch.from_numpy(chunk_3d).float()  # Shape: (64, 64, 64)
        mask_tensor = torch.from_numpy(mask_chunk).float().unsqueeze(0)  # Shape: (1, 64, 64)

        if self.mode == 'valid':
            return chunk_tensor, mask_tensor, (x, y, x + self.size, y + self.size)
        return chunk_tensor, mask_tensor


class RegressionPLModel(pl.LightningModule):
    def __init__(self, size=64):
        super().__init__()
        self.save_hyperparameters()

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        # TimeSformer directly outputs 16 values (4x4 spatial)
        self.backbone = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=16,
            num_frames=64,  # z dimension as time
            num_classes=16,  # Direct 4x4 output
            channels=1,
            depth=8,
            heads=6,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1
        )

    def forward(self, x):
        # x shape: (batch, z, y, x) where z=64, y=64, x=64
        # Add channel dimension: (batch, 1, z, y, x)
        x = x.unsqueeze(1)  # (batch, 1, 64, 64, 64)

        # Permute to (batch, z, 1, y, x) - z becomes temporal dimension
        x = torch.permute(x, (0, 2, 1, 3, 4))

        # TimeSformer directly outputs 16 values
        x = self.backbone(x)  # (batch, 16)

        # Reshape to 2D spatial map
        x = x.view(-1, 1, 4, 4)  # (batch, 1, 4, 4)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, z, y, x), y: (batch, 1, y, x)
        # Downsample mask to match output size
        y = F.interpolate(y, size=(4, 4), mode='bilinear')
        outputs = self(x)
        loss = self.loss_func(outputs, y)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, xyxy = batch  # x: (batch, z, y, x), y: (batch, 1, y, x)
        # Downsample mask to match output size
        y = F.interpolate(y, size=(4, 4), mode='bilinear')
        outputs = self(x)
        loss = self.loss_func(outputs, y)
        self.log("val/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, CFG.epochs, eta_min=CFG.min_lr
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
    set_seed(CFG.seed)
    os.makedirs(CFG.model_dir, exist_ok=True)

    # Get all fragments from zarr
    zarr_store = zarr.open(CFG.zarr_path, mode='r')
    all_fragments = list(zarr_store.keys())

    # Split into train/valid
    random.shuffle(all_fragments)
    n_valid = int(len(all_fragments) * CFG.valid_ratio)
    valid_fragments = all_fragments[:n_valid]
    train_fragments = all_fragments[n_valid:]

    print(f"Total fragments: {len(all_fragments)}")
    print(f"Train fragments: {len(train_fragments)}")
    print(f"Valid fragments: {len(valid_fragments)}")

    # Create datasets
    train_dataset = ZarrDataset(
        CFG.zarr_path, train_fragments, CFG.masks_path,
        size=CFG.size, stride=CFG.stride, mode='train'
    )
    valid_dataset = ZarrDataset(
        CFG.zarr_path, valid_fragments, CFG.masks_path,
        size=CFG.size, stride=CFG.stride, mode='valid'
    )

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Valid chunks: {len(valid_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    #persistent_workers=True,  # Keep workers alive
    prefetch_factor=1  # Prefetch batches
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    #persistent_workers=True,  # Keep workers alive
    prefetch_factor=1  # Prefetch batches
    )

    # Setup model and training
    model = RegressionPLModel(size=CFG.size)

    #TODO: reenable when my setup isnt broken?
    #model = torch.compile(model)

    #wandb_logger = WandbLogger(project="vesuvius", name="zarr_training")

    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="gpu",
        devices=1,
        #logger=wandb_logger,
        default_root_dir=CFG.outputs_path,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        val_check_interval=1000,
        callbacks=[
            ModelCheckpoint(
                filename='best_{epoch}_{val/loss:.4f}',
                dirpath=CFG.model_dir,
                monitor='val/loss',
                mode='min',
                save_top_k=3
            )
        ]
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()