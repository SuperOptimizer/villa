import os
import numpy as np
from torch.utils.data import  Dataset
import zarr
import cv2
import random
import torch
import skimage

from augment import VolumetricAugmentations


def preprocess_chunk(chunk, iso):
    chunk = skimage.exposure.equalize_hist(chunk / 255.0)
    chunk[chunk < iso / 255.0] = 0.0
    return chunk

class ZarrDataset(Dataset):
    def __init__(self, zarr_path, labels_path, masks_path, fragment_ids, mode, chunk_size, stride, ink_avg_threshold, augment_chance, iso):
        self.zarr_store = zarr.open(zarr_path, mode='r')
        self.labels_path = labels_path
        self.masks_path = masks_path
        self.fragment_ids = fragment_ids
        self.mode = mode
        self.chunk_size = chunk_size
        self.stride = stride
        self.ink_avg_threshold = ink_avg_threshold
        self.augment_chance = augment_chance
        self.iso = iso

        self.ink_mask_cache_2d = {}
        self.chunks = []
        for frag_id in fragment_ids:
            self._build_chunks_for_fragment(frag_id)
        if mode == 'train':
            self.augment = VolumetricAugmentations(self.augment_chance)
        else:
            self.augment = None

        print(f"Loaded {len(self.chunks)} chunks for {mode} dataset")

    def _load_and_cache_mask_2d(self, frag_id):
        if frag_id in self.ink_mask_cache_2d:
            return self.ink_mask_cache_2d[frag_id]
        ink_mask_path = f"{self.labels_path}/{frag_id}_inklabels.png"
        if not os.path.exists(ink_mask_path):
            ink_mask_path = f"{self.labels_path}/{frag_id}_inklabels.tiff"
            if not os.path.exists(ink_mask_path):
                return None
        ink_mask = cv2.imread(ink_mask_path, 0)
        if ink_mask is None:
            raise
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
        frag_mask = cv2.imread(f"{self.masks_path}/{frag_id}/{frag_id}_mask.png", 0)
        frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        for y in range(0, h - self.chunk_size, self.stride):
            for x in range(0, w - self.chunk_size, self.stride):
                chunk_frag_mask = frag_mask[y:y + self.chunk_size, x:x + self.chunk_size]
                if np.all(chunk_frag_mask == 0):
                    continue
                chunk_ink_2d = ink_mask_2d[y:y + self.chunk_size, x:x + self.chunk_size]
                has_ink = np.mean(chunk_ink_2d) > self.ink_avg_threshold
                #TODO: how many no ink samples should we show the model? here's where we get them
                if has_ink or self.mode == 'valid':
                    self.chunks.append([frag_id, x, y])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        frag_id, x, y = self.chunks[idx]
        if (self.mode == 'train' and
                self.stride < x < self.zarr_store[frag_id].shape[2] - self.stride and
                self.stride < y < self.zarr_store[frag_id].shape[1] - self.stride):
            yoff, xoff = random.randint(-self.stride, self.stride), random.randint(-self.stride, self.stride)
            ystart = yoff + y
            xstart = xoff + x
        else:
            ystart = y
            xstart = x
        chunk_3d = self.zarr_store[frag_id][:, ystart:ystart + self.chunk_size, xstart:xstart + self.chunk_size].astype(np.float32)
        ink_mask_2d = self.ink_mask_cache_2d[frag_id][ystart:ystart + self.chunk_size, xstart:xstart + self.chunk_size].astype(np.float32)
        ink_mask_3d = np.broadcast_to(ink_mask_2d[np.newaxis, :, :], (self.chunk_size, self.chunk_size, self.chunk_size)).copy()
        ink_mask_3d[chunk_3d < self.iso] = 0
        chunk_3d = preprocess_chunk(chunk_3d, self.iso)
        ink_mask_3d = ink_mask_3d / 255.0
        if self.augment is not None:
            chunk_3d, ink_mask_3d = self.augment.apply(chunk_3d, ink_mask_3d)
        chunk_tensor = torch.from_numpy(chunk_3d).float()
        mask_tensor = torch.from_numpy(ink_mask_3d).float()
        if self.mode == 'valid':
            return chunk_tensor, mask_tensor, (xstart, ystart, xstart + self.chunk_size, ystart + self.chunk_size)
        return chunk_tensor, mask_tensor