import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
import zarr
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset
from helpers import generate_positions

class InferenceDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            targets: List[Dict],
            model_info: Optional[Dict[str, Any]] = None,
            patch_size=None,
            input_format: str = 'zarr',
            overlap: float = 0.25,
            load_all: bool = False
            ):
        """
        Dataset for nnUNet inference on zarr arrays.
        
        Args:
            input_path: Path to the input zarr store
            targets: Output targets configuration
            model_info: Optional nnUNet model information dictionary
            patch_size: Patch size, can be overridden if model_info is provided
            input_format: Format of the input data ('zarr' supported currently)
            overlap: Overlap between patches (as a fraction)
            load_all: Whether to load the entire array into memory
        """
        self.input_path = input_path
        self.input_format = input_format
        self.targets = targets
        # Use model's patch size if available, otherwise use provided patch_size
        self.patch_size = model_info['patch_size'] if model_info is not None else patch_size
        self.overlap = overlap
        self.load_all = load_all
        self.model_info = model_info

        if input_format == 'zarr':
            self.input_array = zarr.open(self.input_path, mode='r')
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        if load_all:
            self.input_array = self.input_array[:]

        self.input_shape = self.input_array.shape
        self.input_dtype = self.input_array.dtype

        # Get expected number of input channels from model if available
        if model_info is not None:
            self.num_input_channels = model_info['num_input_channels']
        else:
            self.num_input_channels = 1  # Default to 1 channel if not specified

        # Verify that data dimensions match the model's expectations
        if len(self.input_shape) == 3:  # zarr is 3D (Z,Y,X) with implicit channel
            # Single channel implicit in data, make sure model expects 1 channel
            if self.num_input_channels != 1:
                raise ValueError(f"Model expects {self.num_input_channels} channels but input has 1 implicit channel")
        elif len(self.input_shape) == 4:  # zarr is 4D with explicit channels (C,Z,Y,X)
            # Check if channel count matches model expectations
            if self.input_shape[0] != self.num_input_channels:
                raise ValueError(f"Model expects {self.num_input_channels} channels but input has {self.input_shape[0]} channels")
            
        pZ, pY, pX = self.patch_size
        
        # Handle different input array dimensions
        if len(self.input_shape) == 3:  # 3D array (Z,Y,X)
            minz, maxz = 0, self.input_shape[0]
            miny, maxy = 0, self.input_shape[1]
            minx, maxx = 0, self.input_shape[2]
        elif len(self.input_shape) == 4:  # 4D array (C,Z,Y,X)
            minz, maxz = 0, self.input_shape[1]
            miny, maxy = 0, self.input_shape[2]
            minx, maxx = 0, self.input_shape[3]
        else:
            raise ValueError(f"Unsupported input shape: {self.input_shape}")

        # Generate all coordinates
        z_step = int(round(pZ * (1 - self.overlap)))
        y_step = int(round(pY * (1 - self.overlap)))
        x_step = int(round(pX * (1 - self.overlap)))

        z_positions = generate_positions(minz, maxz, pZ, z_step)
        y_positions = generate_positions(miny, maxy, pY, y_step)
        x_positions = generate_positions(minx, maxx, pX, x_step)

        self.all_positions = []
        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    self.all_positions.append((z, y, x))

    def __len__(self):
        return len(self.all_positions)

    def __getitem__(self, idx):
        z, y, x = self.all_positions[idx]
        
        # Handle different input array dimensions
        if len(self.input_shape) == 3:  # 3D array (Z,Y,X)
            patch = self.input_array[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            # Add channel dimension
            patch = patch[np.newaxis, ...]
        elif len(self.input_shape) == 4:  # 4D array (C,Z,Y,X)
            patch = self.input_array[:, z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
        
        # Apply appropriate normalization based on dtype
        if self.input_dtype == np.uint8:
            patch = patch.astype(np.float32) / 255.0
        elif self.input_dtype == np.uint16:
            patch = patch.astype(np.float32) / 65535.0
        else:
            # Already float32/float64, ensure it's float32
            patch = patch.astype(np.float32)
        
        # Apply nnUNet-specific preprocessing if model_info is available
        if self.model_info is not None:
            # Get intensity properties from nnUNet for proper normalization
            plans_manager = self.model_info.get('plans_manager')
            configuration_manager = self.model_info.get('configuration_manager')
            
            if plans_manager is not None and configuration_manager is not None:
                # Apply nnUNet preprocessing
                # Transpose to match nnUNet's expected format if needed
                # Here we assume the data is already in the correct format (C,Z,Y,X)
                pass
            else:
                # Apply a simple standardization if no specific nnUNet normalization is available
                for c in range(patch.shape[0]):
                    mean = np.mean(patch[c])
                    std = np.std(patch[c])
                    if std > 0:
                        patch[c] = (patch[c] - mean) / std
        else:
            # Simple z-score normalization when no model info is provided
            for c in range(patch.shape[0]):
                mean = np.mean(patch[c])
                std = np.std(patch[c])
                if std > 0:
                    patch[c] = (patch[c] - mean) / std
        
        patch = torch.from_numpy(patch)
        return {"image": patch, "index": idx}

