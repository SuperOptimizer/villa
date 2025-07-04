import os
import numpy as np
from torch.utils.data import DataLoader
import zarr
import random
from tqdm import tqdm
import torch
import glob
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.optim import AdamWFp8

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleVolumetricModel, CombinedLoss
from dataset import ZarrDataset

THISDIR = os.path.abspath(os.path.dirname(__file__))
VESUVIUS_ROOT = "/vesuvius"
ZARR_PATH = f'{VESUVIUS_ROOT}/fragments.zarr'
MASKS_PATH = f'{VESUVIUS_ROOT}/train_scrolls'
OUTPUT_PATH = f'{VESUVIUS_ROOT}/inkdet_outputs/'
LABELS_PATH = f'{THISDIR}/all_labels/'

CHUNK_SIZE = 64
STRIDE = 128
ISO_THRESHOLD = 64
NUM_EPOCHS = 2000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01
NUM_WORKERS = 16
SEED = 42
VALIDATION_SPLIT = 0.0
AUGMENT_CHANCE = 0.5
INKDETECT_MEAN = .1
BATCH_SIZE = 32
COMPILE=False
FP8=False

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        is_distributed = True

        os.environ["NCCL_TIMEOUT"] = "180"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False

        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return rank, world_size, local_rank, is_distributed

def init_cuda():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, path, rank, is_distributed):
    if rank == 0:
        save_dict = {
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'optimizer_state_dict': optimizer.state_dict(),
        }

        if is_distributed:
            save_dict['model_state_dict'] = model.module.state_dict()
        else:
            save_dict['model_state_dict'] = model.state_dict()

        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(save_dict, path)

    if is_distributed:
        dist.barrier()


def load_checkpoint(path, model, optimizer=None, scheduler=None, rank=0, is_distributed=False):
    checkpoint = None

    if rank == 0:
        checkpoint = torch.load(path, map_location='cpu')

    if is_distributed:
        dist.barrier()
        checkpoint = [checkpoint] if rank == 0 else [None]
        dist.broadcast_object_list(checkpoint, src=0)
        checkpoint = checkpoint[0]

    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if is_distributed:
        dist.barrier()

    return epoch, train_losses, val_losses


def main():
    rank, world_size, local_rank, is_distributed = setup_distributed()
    set_seed(SEED + rank)

    if is_distributed:
        print(f"Running in distributed mode: Rank {rank}/{world_size}")
    else:
        print("Running in single GPU mode")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = torch.device(f'cuda:{local_rank}')

    checkpoint_path = None
    if rank == 0:
        checkpoints = glob.glob(os.path.join(OUTPUT_PATH, f'best_simple_volumetric_*.ckpt'))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
            print(f"Found checkpoint: {checkpoint_path}")

    if is_distributed:
        checkpoint_path = [checkpoint_path] if rank == 0 else [None]
        dist.broadcast_object_list(checkpoint_path, src=0)
        checkpoint_path = checkpoint_path[0]

    # Load data
    zarr_store = zarr.open(ZARR_PATH, mode='r')
    all_fragments = list(zarr_store.keys())
    #random.shuffle(all_fragments)
    n_valid = int(len(all_fragments) * VALIDATION_SPLIT)
    valid_fragments = []  # all_fragments[:n_valid]
    train_fragments = all_fragments[n_valid:]

    if '20231005123336' in train_fragments:
        train_fragments.remove('20231005123336')

    if rank == 0:
        print(f"Total fragments: {len(all_fragments)}")
        print(f"Train fragments: {len(train_fragments)}")
        print(f"Valid fragments: {len(valid_fragments)}")

    train_dataset = ZarrDataset(ZARR_PATH, LABELS_PATH, MASKS_PATH, train_fragments, 'train', CHUNK_SIZE, STRIDE, INKDETECT_MEAN, AUGMENT_CHANCE, ISO_THRESHOLD)
    valid_dataset = ZarrDataset(ZARR_PATH, LABELS_PATH, MASKS_PATH, valid_fragments, 'valid', CHUNK_SIZE, STRIDE, INKDETECT_MEAN, AUGMENT_CHANCE, ISO_THRESHOLD)

    if len(train_dataset) == 0:
        raise RuntimeError(f"Rank {rank} has no training data!")

    if is_distributed:
        effective_batch_size = BATCH_SIZE // world_size

        # Gather all dataset sizes for distributed mode
        local_size = torch.tensor(len(train_dataset), device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        if rank == 0:
            print(f"Train dataset sizes per rank:")
            for i, size in enumerate(all_sizes):
                print(f"  Rank {i}: {size.item()} chunks")

        # Use minimum size for balanced loading
        min_size = min(s.item() for s in all_sizes)
        if min_size == 0:
            raise RuntimeError("At least one rank has no data!")

        # Trim dataset to minimum size
        if len(train_dataset) > min_size:
            train_dataset.chunks = train_dataset.chunks[:min_size]

        if rank == 0:
            print(f"Using balanced size: {min_size} chunks per rank")

        # Create distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if len(valid_dataset) > 0 else None
    else:
        # Single GPU mode
        effective_batch_size = BATCH_SIZE
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=effective_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=8
    )

    model = SimpleVolumetricModel().to(device)

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Compile model - fixed to avoid DDP issues
    if is_distributed:
        # For DDP, compile before wrapping or compile the underlying module
        # Option 1: Don't compile with DDP (recommended for stability)
        # Option 2: Compile only the underlying module
        # model.module = torch.compile(model.module, fullgraph=True, dynamic=False, mode='default')
        pass  # Skip compilation with DDP for now
    else:
        # Single GPU mode - compile normally
        #model = torch.compile(model, fullgraph=True, dynamic=False, mode='max-autotune')
        pass

    loss_fn = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=MIN_LEARNING_RATE
    )

    start_epoch = 0
    train_losses = []
    val_losses = []
    if checkpoint_path and rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, rank, is_distributed
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")

    scaler = GradScaler('cuda')

    for epoch in range(start_epoch, NUM_EPOCHS):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_train_losses = []

        if is_distributed:
            dist.barrier()

        if is_distributed:
            local_steps = len(train_loader)
            steps_tensor = torch.tensor(local_steps, device=device)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.MIN)
            steps_per_epoch = int(steps_tensor.item())
        else:
            steps_per_epoch = len(train_loader)

        if rank == 0:
            print(f"Steps per epoch: {steps_per_epoch}")

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch} [Rank {rank}]", disable=(rank != 0))
        data_iter = iter(train_loader)

        optimizer.zero_grad()
        for step in pbar:
            try:
                x, y = next(data_iter)
            except StopIteration:
                # If we run out of data, restart the iterator
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            if rank == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if len(epoch_train_losses) > 0:
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        else:
            avg_train_loss = 0.0

        if is_distributed:
            avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = avg_train_loss_tensor.item()

        if rank == 0:
            train_losses.append(avg_train_loss)

        # Validation
        if valid_loader is not None and len(valid_loader) > 0:
            if is_distributed and valid_sampler is not None:
                valid_sampler.set_epoch(epoch)

            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for x, y, _ in tqdm(valid_loader, desc="Validation", disable=(rank != 0)):
                    x, y = x.to(device), y.to(device)
                    with autocast('cuda', dtype=torch.bfloat16):
                        pred = model(x)
                        loss = loss_fn(pred, y)
                    epoch_val_losses.append(loss.item())

            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else 0.0

            if is_distributed:
                avg_val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_val_loss = avg_val_loss_tensor.item()

            if rank == 0:
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        else:
            if rank == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

        # Save checkpoint
        if rank == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses, val_losses,
                os.path.join(OUTPUT_PATH, f'best_simple_volumetric_epoch={epoch}.ckpt'),
                rank, is_distributed
            )

        scheduler.step()

    if rank == 0:
        print("Training completed!")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    init_cuda()
    main()