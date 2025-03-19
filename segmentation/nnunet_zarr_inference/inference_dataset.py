import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
import zarr
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset

# Import helpers with fallback for different import scenarios
# We'll import specific functions only when needed in the code

class InferenceDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            targets: List[Dict],
            model_info: Optional[Dict[str, Any]] = None,
            patch_size=None,
            input_format: str = 'zarr',
            step_size: float = 0.5,
            load_all: bool = False,
            verbose: bool = False
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

        if input_format == 'zarr':
            # Open the zarr array
            zarr_root = zarr.open(self.input_path, mode='r')
            
            # Check if this is a multi-resolution zarr with groups named 0, 1, 2, etc.
            # Use hasattr to safely check if keys exist in the zarr root
            if hasattr(zarr_root, 'keys') and '0' in zarr_root.keys() and '1' in zarr_root.keys():
                # If we have both '0' and '1', it's likely a multi-resolution zarr
                if self.verbose:
                    print(f"Detected multi-resolution zarr. Using highest resolution (group '0')")
                
                # Now determine if '0' is a group or an array and handle accordingly
                if isinstance(zarr_root['0'], zarr.Group):
                    # It's a group with potentially multiple arrays
                    group_0 = zarr_root['0']
                    
                    # First look for direct array access
                    if hasattr(group_0, 'shape'):
                        # The group itself is array-like
                        if self.verbose:
                            print(f"Group '0' is directly accessible as an array")
                        self.input_array = group_0
                    else:
                        # Find the first array in the group
                        array_found = False
                        if hasattr(group_0, 'keys'):
                            # Use keys() method if available (safer)
                            for key in group_0.keys():
                                if hasattr(group_0[key], 'shape'):
                                    if self.verbose:
                                        print(f"Found array '{key}' in group '0'")
                                    self.input_array = group_0[key]
                                    array_found = True
                                    break
                        else:
                            # Fallback to direct iteration (might cause issues with some zarr layouts)
                            try:
                                for key in group_0:
                                    if hasattr(group_0[key], 'shape'):
                                        if self.verbose:
                                            print(f"Found array '{key}' in group '0'")
                                        self.input_array = group_0[key]
                                        array_found = True
                                        break
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error iterating group: {e}")
                        
                        if not array_found:
                            # Fallback to the group itself
                            if self.verbose:
                                print(f"No arrays found in group '0', using group directly")
                            self.input_array = group_0
                
                elif hasattr(zarr_root['0'], 'shape'):
                    # It's a multi-resolution zarr with arrays directly at the root
                    if self.verbose:
                        print(f"Using multi-resolution array '0'")
                    self.input_array = zarr_root['0']
                else:
                    # Unexpected structure
                    raise ValueError(f"Unsupported multi-resolution zarr structure: '0' exists but is not a group or array")
            else:
                # Not a multi-resolution zarr
                self.input_array = zarr_root
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

        # Generate all coordinates using nnUNet-style sliding window approach
        # Import compute_steps_for_sliding_window with same fallback strategy
        try:
            # First try relative imports (when running as a module)
            from nnunet_zarr_inference.helpers import compute_steps_for_sliding_window
        except ImportError:
            # Fallback for direct script execution
            from helpers import compute_steps_for_sliding_window
        
        # Use nnUNet's method to compute steps for sliding window
        if len(self.input_shape) == 3:  # 3D array (Z,Y,X)
            image_size = self.input_shape
        else:  # 4D array (C,Z,Y,X)
            image_size = self.input_shape[1:]
        
        # Use nnUNet's method to compute steps for sliding window
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
            
            # Print overlap information for each dimension
            for dim_name, positions, dim_size in zip(['Z', 'Y', 'X'], [z_positions, y_positions, x_positions], self.patch_size):
                if len(positions) > 1:
                    # Calculate overlap between first two patches as an example
                    first_end = positions[0] + dim_size
                    second_start = positions[1]
                    overlap = first_end - second_start
                    overlap_percent = (overlap / dim_size) * 100
                    
                    print(f"  - {dim_name}-axis: {len(positions)} positions, overlap={overlap} voxels ({overlap_percent:.1f}% of patch)")
                    
                    # Display a few example positions
                    num_to_show = min(4, len(positions))
                    print(f"    Example positions: {positions[:num_to_show]}{'...' if len(positions) > num_to_show else ''}")

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
                # Apply z-score normalization if no specific nnUNet normalization is available
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

