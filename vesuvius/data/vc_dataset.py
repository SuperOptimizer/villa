from typing import List, Dict, Optional, Any, Tuple, Union
import zarr
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re

from utils.models.helpers import compute_steps_for_sliding_window
from data.volume import Volume

class VCDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            targets: List[Dict],
            patch_size,
            num_input_channels: int = 2,
            input_format: str = 'zarr',
            step_size: float = 0.5,
            load_all: bool = False,
            verbose: bool = False,
            mode: str = 'infer',
            num_parts: int = 1,
            part_id: int = 0,
            # Volume-specific parameters
            scroll_id: Optional[Union[int, str]] = None,
            energy: Optional[int] = None, 
            resolution: Optional[float] = None, 
            segment_id: Optional[int] = None,
            cache: bool = True,
            normalize: bool = False,
            domain: Optional[str] = None,
            use_fsspec: bool = False,
            ):
        """
        Dataset for nnUNet inference on zarr arrays, with support for remote paths via Volume class.
        
        Args:
            input_path: Path to the input zarr store or an indicator for remote Volume
            targets: Output targets configuration
            patch_size: Patch size for extraction (tuple of 3 ints)
            num_input_channels: Number of input channels expected by the model
            input_format: Format of the input data ('zarr' or 'volume')
            step_size: Step size for sliding window as a fraction of patch size (default: 0.5, nnUNet default)
            load_all: Whether to load the entire array into memory
            verbose: Enable detailed output messages (default: False)
            mode: Mode of operation ('infer' or 'train')
            num_parts: Number of parts to split the dataset into (default: 1, no splitting)
            part_id: Which part to use (0-indexed, default: 0, must be < num_parts)
            
            # Volume-specific parameters (only needed when input_format='volume')
            scroll_id: ID of the scroll for Volume
            energy: Energy value for Volume
            resolution: Resolution value for Volume
            segment_id: ID of the segment for Volume
            cache: Whether to use caching with Volume (default: True)
            normalize: Whether to normalize Volume data (default: False)
            domain: Domain for Volume ('dl.ash2txt' or 'local')
            use_fsspec: Whether to use fsspec instead of TensorStore for faster data access (default: False)
        """
        self.input_path = input_path
        self.input_format = input_format
        self.targets = targets
        self.patch_size = patch_size
        self.step_size = step_size
        self.load_all = load_all
        self.num_input_channels = num_input_channels
        self.verbose = verbose
        self.mode = mode
        self.use_volume = False  # Will be set to True if we're using the Volume class
        
        # Data partitioning parameters
        if num_parts < 1:
            raise ValueError(f"num_parts must be >= 1, got {num_parts}")
        if part_id < 0 or part_id >= num_parts:
            raise ValueError(f"part_id must be between 0 and {num_parts-1}, got {part_id}")
            
        self.num_parts = num_parts
        self.part_id = part_id

        # Check if we should use Volume class
        # Either explicitly specified or if input_path contains scroll or segment indicators
        if input_format == 'volume' or (
            isinstance(input_path, str) and 
            (re.match(r'scroll\d+', input_path.lower()) or input_path.isdigit())
        ):
            self.use_volume = True
            
            # Determine volume type, scroll_id, and segment_id from input_path if not provided
            if scroll_id is None and segment_id is None:
                if re.match(r'scroll\d+', input_path.lower()):
                    # Format: 'scroll1', 'scroll2', etc.
                    type_value = input_path.lower()
                elif input_path.isdigit():
                    # Format: segment id as numeric string
                    type_value = input_path
                    segment_id = int(input_path)
                else:
                    type_value = input_path
            else:
                # Type can be determined from segment_id or scroll_id
                if segment_id is not None:
                    type_value = str(segment_id)  # Use segment_id as type
                else:
                    type_value = f"scroll{scroll_id}"  # Use scroll as type
            
            # Initialize Volume
            try:
                if self.verbose:
                    print(f"Initializing Volume with type={type_value}, scroll_id={scroll_id}, energy={energy}, resolution={resolution}, segment_id={segment_id}")
                
                self.volume = Volume(
                    type=type_value,
                    scroll_id=scroll_id,
                    energy=energy,
                    resolution=resolution,
                    segment_id=segment_id,
                    cache=cache,
                    normalize=normalize,
                    verbose=verbose,
                    domain=domain,
                    path=None if input_format == 'volume' else input_path,  # Only use input_path as path if not explicitly 'volume'
                    use_fsspec=use_fsspec
                )
                
                if self.verbose:
                    print(f"Successfully initialized Volume")
                    self.volume.meta()
                
                # Get volume's shape for the highest resolution (subvolume 0)
                self.input_shape = self.volume.shape(0)
                self.input_dtype = self.volume.dtype
                self.input_array = None  # We'll access data through volume directly
                
            except Exception as e:
                raise ValueError(f"Error initializing Volume from input_path '{input_path}': {str(e)}")
        
        elif input_format == 'zarr':
            # Original zarr-based loading
            self.use_volume = False
            
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

            if load_all:
                # Load the entire array into memory
                self.input_array = self.input_array[:]
                if self.verbose:
                    print(f"Loaded entire input array into memory (load_all=True)")
            else:
                # Just use the zarr array directly without caching
                if self.verbose:
                    print(f"Using direct zarr array access (no caching)")

            self.input_shape = self.input_array.shape
            self.input_dtype = self.input_array.dtype
            
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        if mode == 'train':
            pass

        elif mode == 'infer':
            # num_input_channels is already provided as a parameter
            pass

            pZ, pY, pX = self.patch_size

            # Get the 3D spatial dimensions regardless of input shape
            image_size = self.get_input_shape()

            # Generate all coordinates using sliding window
            # self.step_size is a factor (e.g., 0.5 means 50% overlap)
            z_positions = compute_steps_for_sliding_window(image_size[0], pZ, self.step_size)
            y_positions = compute_steps_for_sliding_window(image_size[1], pY, self.step_size)
            x_positions = compute_steps_for_sliding_window(image_size[2], pX, self.step_size)

            # Print position information if verbose is enabled
            if self.verbose:
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
                        
            # Apply dataset partitioning along Z-axis if requested
            if self.num_parts > 1:
                # Instead of partitioning by position index, we'll partition by Z coordinate
                # First, get the Z range from the volume
                max_z = image_size[0]
                
                # Calculate Z boundaries for each part
                z_per_part = max_z / self.num_parts  # This can be a float for even division
                
                # Calculate Z range for this part (part 0 is the bottom of the volume)
                z_start = int(z_per_part * self.part_id)
                z_end = int(z_per_part * (self.part_id + 1)) if self.part_id < self.num_parts - 1 else max_z
                
                if self.verbose:
                    print(f"Partitioning dataset along Z-axis: z_range=[0-{max_z}], num_parts={self.num_parts}, part_id={self.part_id}")
                    print(f"Z range for this part: {z_start} to {z_end}")
                
                # Filter positions to only include those in our Z range
                filtered_positions = []
                for pos in self.all_positions:
                    z, y, x = pos
                    z_patch_end = z + self.patch_size[0]  # End of the patch in Z direction
                    
                    # Include patch if:
                    # 1. Patch start is in our range, OR
                    # 2. Patch end is in our range, OR
                    # 3. Patch completely surrounds our range
                    if ((z_start <= z < z_end) or 
                        (z_start < z_patch_end <= z_end) or
                        (z <= z_start and z_patch_end >= z_end)):
                        filtered_positions.append(pos)
                
                # Update positions list
                self.all_positions = filtered_positions
                
                if self.verbose:
                    print(f"Partitioned dataset: using {len(self.all_positions)} positions in Z range [{z_start}-{z_end}]")
                    if self.all_positions:
                        print(f"First position: {self.all_positions[0]}")
                        print(f"Last position: {self.all_positions[-1]}")
                    else:
                        print("Warning: No positions in this partition!")

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
        
        Note: If dataset was already partitioned (num_parts > 1), this method
        will subdivide the current partition further among distributed processes.

        Args:
            rank: The rank of this process (0 to world_size-1)
            world_size: Total number of processes
        """
        if world_size <= 1 or rank < 0 or rank >= world_size:
            # No need to distribute or invalid configuration
            return

        # Get total number of positions in the current partition
        total_positions = len(self.all_positions)

        if self.verbose:
            if self.num_parts > 1:
                print(f"Rank {rank}: Distributing {total_positions} positions from Z-partition {self.part_id}/{self.num_parts} among {world_size} processes")
            else:
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
            print(f"Rank {rank}: Processing {len(self.all_positions)} positions (indices {start_idx} to {end_idx - 1})")
            if len(self.all_positions) > 0:
                # Get Z range of positions for this rank
                z_values = [pos[0] for pos in self.all_positions]
                min_z = min(z_values) if z_values else "N/A"
                max_z = max(z_values) if z_values else "N/A"
                print(f"Rank {rank}: Z-range: {min_z} to {max_z}")
                print(f"Rank {rank}: Sample positions: {self.all_positions[:2]} ... {self.all_positions[-2:] if len(self.all_positions) > 1 else []}")
            else:
                print(f"Rank {rank}: Warning - No positions assigned to this rank")

    def __len__(self):
        return len(self.all_positions)

    def __getitem__(self, idx):
        z, y, x = self.all_positions[idx]

        if self.use_volume:
            # Extract the patch from Volume
            pZ, pY, pX = self.patch_size
            
            # For Volume, we need to determine which subvolume (resolution level) to use
            # Default to the highest resolution (subvolume 0)
            subvolume_idx = 0
            
            # Get patch data using Volume's indexing capabilities
            # Volume class may accept coordinates in a different order than our dataset
            try:
                # Check if input has channels or if it's a single-channel 3D volume
                if len(self.input_shape) == 3:  # Single-channel implicit
                    # Extract spatial patch from Volume
                    if self.verbose and idx < 3:
                        print(f"Extracting patch at z={z}, y={y}, x={x} with size {self.patch_size}")
                    
                    # Extract the entire patch at once
                    # Volume's __getitem__ accepts indices as (x, y, z, subvolume_idx)
                    patch_data = np.zeros((pZ, pY, pX), dtype=np.float32)
                    
                    # Extract chunk at once for fsspec for better performance
                    if self.volume.use_fsspec:
                        try:
                            # Extract the entire patch at once for fsspec
                            if z + pZ <= self.input_shape[0] and y + pY <= self.input_shape[1] and x + pX <= self.input_shape[2]:
                                # Extract full patch at once
                                patch_data = self.volume[z:z+pZ, y:y+pY, x:x+pX].astype(np.float32)
                            else:
                                # Handle out-of-bounds using slice-by-slice approach
                                for i in range(pZ):
                                    if z + i < self.input_shape[0]:
                                        patch_data[i, :min(pY, self.input_shape[1]-y), :min(pX, self.input_shape[2]-x)] = \
                                            self.volume[z+i, y:min(y+pY, self.input_shape[1]), x:min(x+pX, self.input_shape[2])].astype(np.float32)
                            
                            # Apply appropriate normalization based on dtype (for fsspec)
                            if self.input_dtype == np.uint8:
                                patch_data = patch_data / 255.0
                            elif self.input_dtype == np.uint16:
                                patch_data = patch_data / 65535.0
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"Error extracting patch: {e}")
                    else:
                        # Original slice-by-slice extraction for TensorStore
                        for i in range(pZ):
                            if z + i < self.input_shape[0]:
                                for j in range(pY):
                                    if y + j < self.input_shape[1]:
                                        try:
                                            # Volume expects coordinates as z, y, x for the 3D case
                                            slice_data = self.volume[z + i, y + j, x:x + pX]
                                            patch_data[i, j, :min(pX, len(slice_data))] = slice_data[:min(pX, len(slice_data))]
                                        except (IndexError, ValueError) as e:
                                            if self.verbose and idx < 3:
                                                print(f"Warning: Error extracting slice at z={z+i}, y={y+j}, x={x}: {e}")
                    
                    # Add channel dimension (create C,Z,Y,X)
                    patch = patch_data[np.newaxis, ...]
                    
                else:  # Multi-channel explicit (C,Z,Y,X)
                    # For multi-channel, we need to extract each channel separately
                    num_channels = self.input_shape[0]
                    patch = np.zeros((num_channels, pZ, pY, pX), dtype=np.float32)
                    
                    # This implementation depends on how the Volume class handles multi-channel data
                    # You may need to adjust this based on your specific Volume implementation
                    for c in range(num_channels):
                        for i in range(pZ):
                            if z + i < self.input_shape[1]:  # Check Z bounds (input_shape now includes channel dim)
                                for j in range(pY):
                                    if y + j < self.input_shape[2]:  # Check Y bounds
                                        try:
                                            # Extract slice for this channel
                                            # Assume Volume can extract specific channels
                                            slice_data = self.volume[z + i, y + j, x:x + pX]
                                            # Assign to proper channel in the patch
                                            patch[c, i, j, :min(pX, len(slice_data))] = slice_data[:min(pX, len(slice_data))]
                                        except (IndexError, ValueError) as e:
                                            if self.verbose and idx < 3:
                                                print(f"Warning: Error extracting channel {c} slice at z={z+i}, y={y+j}, x={x}: {e}")
            
            except Exception as e:
                print(f"Error extracting patch from Volume at {z,y,x}: {str(e)}")
                # Return a zero-filled patch as a fallback
                if len(self.input_shape) == 3:
                    patch = np.zeros((1, pZ, pY, pX), dtype=np.float32)
                else:
                    patch = np.zeros((self.input_shape[0], pZ, pY, pX), dtype=np.float32)
        
        else:
            # Standard zarr array extraction (original implementation)
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

        # Handle data type normalization (same for both Volume and zarr)
        # Apply appropriate normalization based on dtype
        if not self.use_volume:  # Volume may already normalize based on normalize parameter
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
        if self.verbose:
            if idx < 3:
                print(f"Created position in __getitem__[{idx}]: {position}")

        return {
            "data": patch,  # Use key "data" for compatibility with inference.py
            "pos": position,  # Include the 3D position
            "index": idx
        }
