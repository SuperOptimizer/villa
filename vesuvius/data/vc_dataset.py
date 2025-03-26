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
            normalize: bool = True,
            normalization_scheme: str = 'zscore',
            return_as_type: str = 'np.float32',
            return_as_tensor: bool = True,
            domain: Optional[str] = None,
            use_fsspec: bool = False,
            ):
        """
        Dataset for nnUNet inference on zarr arrays, with support for local and remote data via Volume class.
        
        Args:
            input_path: Path to the input zarr store or an indicator for remote Volume
            targets: Output targets configuration
            patch_size: Patch size for extraction (tuple of 3 ints)
            num_input_channels: Number of input channels expected by the model
            input_format: Format of the input data ('zarr' or 'volume')
            step_size: Step size for sliding window as a fraction of patch size (default: 0.5, nnUNet default)
            load_all: Whether to load the entire array into memory (ignored when using Volume class)
            verbose: Enable detailed output messages (default: False)
            mode: Mode of operation ('infer' or 'train')
            num_parts: Number of parts to split the dataset into (default: 1, no splitting)
            part_id: Which part to use (0-indexed, default: 0, must be < num_parts)
            
            # Volume parameters
            scroll_id: ID of the scroll for Volume
            energy: Energy value for Volume
            resolution: Resolution value for Volume
            segment_id: ID of the segment for Volume
            cache: Whether to use caching with Volume (default: True)
            normalize: Whether to normalize Volume data to 0:1 values (default: True)
            normalization_scheme: Normalization scheme to apply ('none', 'zscore', 'minmax') (default: 'zscore')
            return_as_type: Type to return data as ('none', 'np.uint8', 'np.uint16', 'np.float16', 'np.float32') (default: 'np.float32')
            return_as_tensor: Whether to return data as torch tensor (default: True)
            domain: Domain for Volume ('dl.ash2txt' or 'local')
            use_fsspec: Whether to use fsspec instead of TensorStore for data access (default: False)
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
        
        # Data partitioning parameters
        if num_parts < 1:
            raise ValueError(f"num_parts must be >= 1, got {num_parts}")
        if part_id < 0 or part_id >= num_parts:
            raise ValueError(f"part_id must be between 0 and {num_parts-1}, got {part_id}")
            
        self.num_parts = num_parts
        self.part_id = part_id

        # Always use Volume class for all data access
        self.use_volume = True
        
        # Determine volume type, scroll_id, and segment_id from input_path if not provided
        if input_format == 'volume' or (
            isinstance(input_path, str) and 
            (re.match(r'scroll\d+', input_path.lower()) or input_path.isdigit())
        ):
            # This is a scroll or segment identifier
            if self.verbose:
                print(f"Using Volume class for scroll/segment identifier: {input_path}")
            
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
                    
            # When input_format is 'volume' or it's a scroll/segment ID, set path to None
            # This ensures Volume uses the config.yaml to find the remote path
            use_path = None
        else:
            # This is a path to a zarr file/store
            if self.verbose:
                print(f"Using Volume class for zarr path: {input_path}")
                
            # For regular zarr paths, use the zarr path as type
            # This allows Volume to handle both local and remote zarr files uniformly
            type_value = "zarr"
            use_path = input_path
        
        # Initialize Volume for all access
        try:
            if self.verbose:
                print(f"Initializing Volume with type={type_value}, scroll_id={scroll_id}, energy={energy}, resolution={resolution}, segment_id={segment_id}")
                print(f"Path={use_path}, use_fsspec={use_fsspec}, domain={domain}")
                
            self.volume = Volume(
                type=type_value,
                scroll_id=scroll_id,
                energy=energy,
                resolution=resolution,
                segment_id=segment_id,
                cache=cache,
                normalize=normalize,
                normalization_scheme=normalization_scheme,
                return_as_type=return_as_type,
                return_as_tensor=return_as_tensor,
                verbose=verbose,
                domain=domain,
                path=use_path,
                use_fsspec=use_fsspec
            )
            
            if self.verbose:
                print(f"Successfully initialized Volume")
                self.volume.meta()
            
            # Get volume's shape for the highest resolution (subvolume 0)
            self.input_shape = self.volume.shape(0)
            self.input_dtype = self.volume.dtype
            
        except Exception as e:
            raise ValueError(f"Error initializing Volume from input_path '{input_path}': {str(e)}")

        if mode == 'train':
            pass

        elif mode == 'infer':

            pZ, pY, pX = self.patch_size
            # Get the 3D spatial dimensions directly
            image_size = self.input_shape if len(self.input_shape) == 3 else self.input_shape[1:]

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

        pZ, pY, pX = self.patch_size

        # Get patch data using Volume
        try:
            # Check if input has channels or if it's a single-channel 3D volume
            if len(self.input_shape) == 3:  # Single-channel implicit
                # Extract spatial patch from Volume
                if self.verbose and idx < 3:
                    print(f"Extracting patch at z={z}, y={y}, x={x} with size {self.patch_size}")

                # Extract the entire patch at once
                # Volume's __getitem__ accepts indices as (x, y, z, subvolume_idx)
                patch_data = np.zeros((pZ, pY, pX), dtype=np.float32)

                # Extract patch data using uniform approach regardless of fsspec or TensorStore
                try:
                    # Define safe slices that don't go out of bounds
                    z_slice = slice(z, min(z + pZ, self.input_shape[0]))
                    y_slice = slice(y, min(y + pY, self.input_shape[1]))
                    x_slice = slice(x, min(x + pX, self.input_shape[2]))

                    if self.verbose and idx < 3:
                        print(f"Extracting patch with slices: z={z_slice}, y={y_slice}, x={x_slice}")
                        print(f"Volume access mode: {'fsspec/zarr' if self.volume.use_fsspec else 'TensorStore'}")

                    # Get the data using Volume's __getitem__ method
                    # Volume.__getitem__ handles the read().result() call for TensorStore
                    # And direct array access for fsspec/zarr
                    extracted_data = self.volume[z_slice, y_slice, x_slice]

                    # Copy data to the right location in our patch
                    # Calculate actual sizes fetched (may be smaller than requested due to bounds)
                    fetched_z = extracted_data.shape[0]
                    fetched_y = extracted_data.shape[1]
                    fetched_x = extracted_data.shape[2]

                    # Copy the data into our patch buffer
                    patch_data[:fetched_z, :fetched_y, :fetched_x] = extracted_data

                    if self.verbose and idx < 3:
                        print(f"Extracted data shape: {extracted_data.shape}")

                except Exception as e:
                    if self.verbose:
                        print(f"Error extracting patch: {e}")

                # Add channel dimension (create C,Z,Y,X)
                patch = patch_data[np.newaxis, ...]

            else:  # Multi-channel explicit (C,Z,Y,X)
                # For multi-channel, we need to extract each channel separately
                num_channels = self.input_shape[0]
                patch = np.zeros((num_channels, pZ, pY, pX), dtype=np.float32)

                # Use a unified approach for both fsspec and TensorStore
                try:
                    # Define safe slices that don't go out of bounds
                    z_slice = slice(z, min(z + pZ, self.input_shape[1]))  # input_shape[1] is Z for multi-channel
                    y_slice = slice(y, min(y + pY, self.input_shape[2]))  # input_shape[2] is Y for multi-channel
                    x_slice = slice(x, min(x + pX, self.input_shape[3]))  # input_shape[3] is X for multi-channel

                    if self.verbose and idx < 3:
                        print(f"Extracting multi-channel patch with slices: z={z_slice}, y={y_slice}, x={x_slice}")
                        print(f"Volume access mode: {'fsspec/zarr' if self.volume.use_fsspec else 'TensorStore'}")

                    # Try to extract all channels at once first
                    try:
                        # Attempt to extract the entire 4D chunk at once
                        all_data = self.volume[:, z_slice, y_slice, x_slice]

                        if self.verbose and idx < 3:
                            print(f"Extracted full multi-channel data shape: {all_data.shape}")

                        # Copy to our patch buffer
                        # We need to handle potential size differences if bounds were hit
                        c_size = min(all_data.shape[0], patch.shape[0])
                        z_size = min(all_data.shape[1], pZ)
                        y_size = min(all_data.shape[2], pY)
                        x_size = min(all_data.shape[3], pX)

                        patch[:c_size, :z_size, :y_size, :x_size] = all_data[:c_size, :z_size, :y_size, :x_size]

                    except Exception as e:
                        if self.verbose:
                            print(f"Full multi-channel extraction failed: {e}, falling back to per-channel extraction")

                        # Extract each channel separately
                        for c in range(num_channels):
                            # Volume.__getitem__ handles read().result() call for TensorStore
                            channel_data = self.volume[c, z_slice, y_slice, x_slice]

                            # Copy data to the right location in our patch
                            fetched_z = channel_data.shape[0]
                            fetched_y = channel_data.shape[1]
                            fetched_x = channel_data.shape[2]

                            # Copy the data into our patch buffer for this channel
                            patch[c, :fetched_z, :fetched_y, :fetched_x] = channel_data

                except Exception as e:
                    if self.verbose:
                        print(f"Multi-channel extraction error: {e}")

        except Exception as e:
            raise ValueError("Could not extract patch from Volume") from e

        # Assert that patch is already a torch tensor
        assert isinstance(patch, torch.Tensor), "Volume should return torch tensors when return_as_tensor=True"

        # Create the position tuple
        # Convert to integers
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
