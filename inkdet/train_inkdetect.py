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
COMPILE=True

class CFG:
    zarr_path = f'{VESUVIUS_ROOT}/fragments.zarr'
    masks_path = f'{VESUVIUS_ROOT}/train_scrolls'
    outputs_path = f'{VESUVIUS_ROOT}/inkdet_outputs/'
    model_dir = f'{outputs_path}/models/'

    size = 64
    stride = 64

    batch_size = 16 # bs=16 for 64x64x64 fits well into 8gb vram and 32gb system ram
    epochs = 20
    lr = 3e-5
    min_lr = 1e-6
    weight_decay = 1e-6
    num_workers = 2
    seed = 42

    valid_ratio = 0.1


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class ZarrDataset(Dataset):
    def __init__(self, zarr_path, fragment_ids, masks_path, mode):
        self.zarr_store = zarr.open(zarr_path, mode='r')
        self.fragment_ids = fragment_ids
        self.masks_path = masks_path
        self.mode = mode

        self.ink_mask_cache = {}

        self.chunks = []
        for frag_id in fragment_ids:
            self._build_chunks_for_fragment(frag_id)
        print("done loading")

    def _load_and_cache_mask(self, frag_id):
        """Load mask once and cache it"""
        if frag_id in self.ink_mask_cache:
            return self.ink_mask_cache[frag_id]

        ink_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_inklabels.png")
        if not os.path.exists(ink_mask_path):
            ink_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_inklabels.tiff")
            if not os.path.exists(ink_mask_path):
                return None

        ink_mask = cv2.imread(ink_mask_path, 0)
        if ink_mask is not None:
            h, w = self.zarr_store[frag_id].shape[1:]
            ink_mask = cv2.resize(ink_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ink_mask = ink_mask.astype(np.float32) / 255.0
            self.ink_mask_cache[frag_id] = ink_mask
        return ink_mask

    def _build_chunks_for_fragment(self, frag_id):
        frag_data = self.zarr_store[frag_id]
        d, h, w = frag_data.shape

        ink_mask = self._load_and_cache_mask(frag_id)
        if ink_mask is None:
            print(f"No ink mask found for {frag_id}, skipping")
            return

        frag_mask_path = os.path.join(self.masks_path, frag_id, f"{frag_id}_mask.png")
        if not os.path.exists(frag_mask_path):
            print(f"No fragment mask found for {frag_id}, skipping")
            return

        frag_mask = cv2.imread(frag_mask_path, 0)
        if frag_mask is None:
            print(f"Failed to load fragment mask for {frag_id}, skipping")
            return

        frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for y in range(0, h - CFG.size, CFG.stride):
            for x in range(0, w - CFG.size, CFG.stride):
                chunk_frag_mask = frag_mask[y:y + CFG.size, x:x + CFG.size]
                if np.all(chunk_frag_mask == 0):
                    continue

                #TODO: we want to train on some chunks that have no ink, bit what's a good ratio of no ink to ink chunks?
                #for now lets just give a flat 10% chance to train on no ink chunks
                chunk_mask = ink_mask[y:y + CFG.size, x:x + CFG.size]
                has_ink = np.mean(chunk_mask) > 0.05
                if has_ink or self.mode == 'valid' or random.randint(1,10) == 1:
                    self.chunks.append([frag_id, x, y])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        frag_id, x, y = self.chunks[idx]

        chunk_3d = self.zarr_store[frag_id][:, y:y + CFG.size, x:x + CFG.size].astype(np.float32) / 255.0
        ink_mask = self.ink_mask_cache[frag_id][y:y + CFG.size, x:x + CFG.size]

        chunk_tensor = torch.from_numpy(chunk_3d).float()
        mask_tensor = torch.from_numpy(ink_mask).float().unsqueeze(0)

        if self.mode == 'valid':
            return chunk_tensor, mask_tensor, (x, y, x + CFG.size, y + CFG.size)
        return chunk_tensor, mask_tensor


class RegressionPLModel(pl.LightningModule):
    def __init__(self):
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
        if not COMPILE:
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, xyxy = batch  # x: (batch, z, y, x), y: (batch, 1, y, x)
        # Downsample mask to match output size
        y = F.interpolate(y, size=(4, 4), mode='bilinear')
        outputs = self(x)
        loss = self.loss_func(outputs, y)
        if not COMPILE:
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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

    zarr_store = zarr.open(CFG.zarr_path, mode='r')
    all_fragments = list(zarr_store.keys())

    random.shuffle(all_fragments)
    n_valid = int(len(all_fragments) * CFG.valid_ratio)
    valid_fragments = all_fragments[:n_valid]
    train_fragments = all_fragments[n_valid:]

    print(f"Total fragments: {len(all_fragments)}")
    print(f"Train fragments: {len(train_fragments)}")
    print(f"Valid fragments: {len(valid_fragments)}")

    train_dataset = ZarrDataset(CFG.zarr_path, train_fragments, CFG.masks_path, mode='train')
    valid_dataset = ZarrDataset(CFG.zarr_path, valid_fragments, CFG.masks_path, mode='valid')

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Valid chunks: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = RegressionPLModel()
    if COMPILE:
        model = torch.compile(model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})

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
                #monitor='val/loss',
                #mode='min',
                save_top_k=-1
            )
        ]
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()