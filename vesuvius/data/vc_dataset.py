from typing import List, Dict, Optional, Any
import zarr
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.data.io.zarrio.zarr_cache import ZarrArrayLRUCache
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

class VCDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            targets: List[Dict],
            model_info: Optional[Dict[str, Any]] = None,
            patch_size=None,
            input_format: str = 'zarr',
            step_size: float = 0.5,
            load_all: bool = False,
            verbose: bool = False,
            cache_size: int = 256,
            max_cache_bytes: float = 4.0,
            mode: str = 'infer'
            ):
        """
        Dataset for nnUNet inference on zarr arrays.
        
        Args:
            input_path: Path to the input zarr store
            targets: Output targets configuration
            model_info: Optional nnUNet model information dictionary
            patch_size: Patch size, can be overridden if model_info is provided
            input_format: Format of the input data ('zarr' supported currently)
            step_size: Step size for sliding window as a fraction of patch size (default: 0.5, nnUNet default)
            load_all: Whether to load the entire array into memory
            verbose: Enable detailed output messages (default: False)
            cache_size: Size of the LRU cache for zarr chunks (default: 256)
            max_cache_bytes: Maximum memory in GB to use for zarr cache (default: 4.0)
        """
        self.input_path = input_path
        self.input_format = input_format
        self.targets = targets
        # Use model's patch size if available, otherwise use provided patch_size
        self.patch_size = model_info['patch_size'] if model_info is not None else patch_size
        self.step_size = step_size
        self.load_all = load_all
        self.model_info = model_info
        self.verbose = verbose
        self.cache_size = cache_size
        self.max_cache_bytes = max_cache_bytes
        self.mode = mode

        if input_format == 'zarr':
            # Open the zarr array
            zarr_root = zarr.open(self.input_path, mode='r')

            # Check if this is a multi-resolution zarr (has groups '0', '1', etc.)
            if hasattr(zarr_root, 'keys') and '0' in zarr_root.keys():
                # This is a multi-resolution zarr - we always use group '0' (highest resolution)
                if self.verbose:
                    print(f"Detected multi-resolution zarr. Using highest resolution (group '0')")
                self.input_array = zarr_root['0']
            else:
                # Regular zarr with array at the root
                if self.verbose:
                    print(f"Using zarr array at root level")
                self.input_array = zarr_root
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        if load_all:
            # Load the entire array into memory
            self.input_array = self.input_array[:]
            self.cache_enabled = False
            if self.verbose:
                print(f"Loaded entire input array into memory (load_all=True)")
        else:
            # Use LRU cache for zarr array access
            self.original_array = self.input_array
            self.input_array = ZarrArrayLRUCache(
                self.input_array, 
                max_size=self.cache_size, 
                max_bytes_gb=self.max_cache_bytes
            )
            self.cache_enabled = True
            if self.verbose:
                print(f"Using LRU cache for zarr array access with cache_size={self.cache_size}, max_cache_bytes={self.max_cache_bytes}GB")

        self.input_shape = self.input_array.shape
        self.input_dtype = self.input_array.dtype

        if mode == 'train':
            pass

        elif mode == 'infer':
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

            # Get the 3D spatial dimensions regardless of input shape
            image_size = self.get_input_shape()

            # Generate all coordinates using sliding window
            # self.step_size is a factor (e.g., 0.5 means 50% overlap)
            z_positions = compute_steps_for_sliding_window(image_size[0], pZ, self.step_size)
            y_positions = compute_steps_for_sliding_window(image_size[1], pY, self.step_size)
            x_positions = compute_steps_for_sliding_window(image_size[2], pX, self.step_size)

            # Print position information if the model_info contains verbose flag
            if self.model_info is not None and self.model_info.get('verbose', False):
                print(f"Computed patch positions with step_size={self.step_size}:")
                print(f"  - Input shape: {image_size}")
                print(f"  - Patch size: {self.patch_size}")
                print(f"  - Number of positions: z={len(z_positions)}, y={len(y_positions)}, x={len(x_positions)}")
                print(f"  - Total patches: {len(z_positions) * len(y_positions) * len(x_positions)}")

            self.all_positions = []
            for z in z_positions:
                for y in y_positions:
                    for x in x_positions:
                        self.all_positions.append((z, y, x))

    def get_input_shape(self):
        """
        Return the spatial dimensions (Z,Y,X) of the input array.
        For 4D arrays (C,Z,Y,X), returns just the spatial part (Z,Y,X).
        """
        # Handle both 3D array (Z,Y,X) and 4D array with channels (C,Z,Y,X)
        return self.input_shape if len(self.input_shape) == 3 else self.input_shape[1:]

    def set_distributed(self, rank: int, world_size: int):
        """
        Configure this dataset for distributed data parallel processing.

        This method divides the dataset's patch positions among processes,
        so each process works on a different subset of patches.

        Args:
            rank: The rank of this process (0 to world_size-1)
            world_size: Total number of processes
        """
        if world_size <= 1 or rank < 0 or rank >= world_size:
            # No need to distribute or invalid configuration
            return

        # Get total number of positions
        total_positions = len(self.all_positions)

        if self.verbose:
            print(f"Rank {rank}: Distributing {total_positions} positions among {world_size} processes")

        # Calculate positions for this rank (simple chunking)
        # Each rank gets approximately total_positions / world_size positions
        positions_per_rank = total_positions // world_size
        remainder = total_positions % world_size

        # Calculate start and end indices
        # Ranks with ID < remainder get one extra position
        start_idx = rank * positions_per_rank + min(rank, remainder)
        end_idx = start_idx + positions_per_rank + (1 if rank < remainder else 0)

        # Take only the positions assigned to this rank
        self.all_positions = self.all_positions[start_idx:end_idx]

        if self.verbose:
            print(f"Rank {rank}: Processing {len(self.all_positions)} positions ({start_idx} to {end_idx - 1})")

    def __len__(self):
        return len(self.all_positions)

    def __getitem__(self, idx):
        z, y, x = self.all_positions[idx]

        # Extract the patch based on array dimensionality
        if len(self.input_shape) == 3:  # 3D array (Z,Y,X)
            # Extract spatial patch and add channel dimension
            patch = self.input_array[z:z+self.patch_size[0],
                                     y:y+self.patch_size[1],
                                     x:x+self.patch_size[2]]
            patch = patch[np.newaxis, ...]  # Add channel dimension
        else:  # 4D array (C,Z,Y,X)
            # Extract patch with channels
            patch = self.input_array[:,
                                    z:z+self.patch_size[0],
                                    y:y+self.patch_size[1],
                                    x:x+self.patch_size[2]]

        # Apply appropriate normalization based on dtype
        if self.input_dtype == np.uint8:
            patch = patch.astype(np.float32) / 255.0
        elif self.input_dtype == np.uint16:
            patch = patch.astype(np.float32) / 65535.0
        else:
            # Already float32/float64, ensure it's float32
            patch = patch.astype(np.float32)

        # z-score normalization
        for c in range(patch.shape[0]):
            mean = np.mean(patch[c])
            std = np.std(patch[c])
            if std > 0:
                patch[c] = (patch[c] - mean) / std

        patch = torch.from_numpy(patch)

        # Create the position tuple with the 3D coordinates
        # Convert to integers to ensure consistency
        position = (int(z), int(y), int(x))

        # Debug statement - only for the first few items
        if idx < 3:
            print(f"Created position in __getitem__[{idx}]: {position}")

        return {
            "data": patch,  # Use key "data" for compatibility with inference.py
            "pos": position,  # Include the 3D position
            "index": idx
        }

    def get_cache_stats(self):
        """
        Get statistics about the LRU cache if enabled.

        Returns:
            Dict with cache statistics or None if cache is not enabled
        """
        if hasattr(self, 'cache_enabled') and self.cache_enabled:
            return self.input_array.get_stats()
        return None