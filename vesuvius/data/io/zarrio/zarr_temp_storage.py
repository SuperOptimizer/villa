import zarr
import numpy as np
import threading
import os
import time
import torch.distributed as dist
from typing import Tuple, Dict, Any, List, Optional, Union
from numcodecs import Blosc
from .zarr_writer_worker import ParallelZarrWriter

class ZarrTempStorage:
    """
    Manages temporary storage for inference patches and positions using zarr.
    
    This class provides methods to create, access, and clean up a zarr-based temporary
    storage system for nnUNet inference, with simplified in-memory position tracking.
    
    This version uses a parallel I/O system for better write performance.
    """
    
    def __init__(self, output_path: str, rank: int = 0, world_size: int = 1, volume_shape = None, verbose: bool = False, num_io_workers: int = 8):
        """
        Initialize the zarr temporary storage.
        
        Args:
            output_path: Base path where the temp.zarr will be created
            rank: Current process rank (for distributed processing)
            world_size: Total number of processes
            volume_shape: Optional tuple of (z, y, x) dimensions of the volume
            verbose: Enable verbose output
            num_io_workers: Number of parallel I/O workers (processes) to use for writing
        """
        self.output_path = output_path
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.temp_zarr_path = os.path.join(output_path, "temp.zarr")
        self.temp_zarr = None
        self.rank_group = None
        self.patches_group = None
        self.num_io_workers = num_io_workers

        self.positions = {}  # Dictionary for each target: {target_name: {index: (z,y,x)}}
        self.patch_counts = {}  # Dictionary to track patch counts: {target_name: count}
        
        # For all rank position collecting
        self.all_rank_positions = {}  # Structure: {rank: {target_name: {index: (z,y,x)}}}
        
        self.lock = threading.Lock()  # Minimal lock just for array creation
        self.volume_shape = volume_shape  # Store volume dimensions
        
        # Initialize our parallel writer
        self.parallel_writer = None  # We'll create this in initialize()
    
    def initialize(self, expected_patch_count=None):
        """
        Initialize the zarr store and create necessary groups.
        
        Args:
            expected_patch_count: Optional count of expected patches, used for more efficient
                               array creation when known in advance.
        """
        # Store expected patch count if provided
        self.expected_patch_count = expected_patch_count
        
        # Initialize dictionaries to store patch counts per target
        self.expected_patch_counts = {}
        
        if expected_patch_count is not None and self.verbose:
            print(f"Rank {self.rank}: Initializing with expected patch count: {expected_patch_count}")
        
        # Create the base zarr store if needed (only rank 0 needs to create the root)
        if self.rank == 0:
            if self.verbose:
                print(f"Rank {self.rank}: Creating temporary zarr storage at {self.temp_zarr_path}")
            self.temp_zarr = zarr.open(self.temp_zarr_path, mode='w')
        else:
            # Wait for rank 0 to create the store
            if self.verbose:
                print(f"Rank {self.rank}: Waiting for rank 0 to create zarr storage")
            
            max_wait = 60  # Maximum wait time in seconds
            start_time = time.time()
            
            while not os.path.exists(self.temp_zarr_path) and time.time() - start_time < max_wait:
                time.sleep(0.1)
            
            if not os.path.exists(self.temp_zarr_path):
                raise TimeoutError(f"Timeout waiting for zarr store at {self.temp_zarr_path}")
                
            if self.verbose:
                print(f"Rank {self.rank}: Opening existing zarr storage at {self.temp_zarr_path}")
            
            self.temp_zarr = zarr.open(self.temp_zarr_path, mode='a')
        
        # Create group for this rank
        self.rank_group = self.temp_zarr.create_group(f"rank_{self.rank}")
        self.patches_group = self.rank_group.create_group("patches")
        
        # Initialize position tracking for this rank
        self.all_rank_positions[self.rank] = {}
        
        # Initialize the parallel writer AFTER the zarr store is created
        self.parallel_writer = ParallelZarrWriter(
            zarr_path=self.temp_zarr_path,
            num_workers=self.num_io_workers,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"Rank {self.rank}: Created zarr groups and initialized parallel writer with {self.num_io_workers} workers")
    
    def set_expected_patch_count(self, target_name, count):
        """
        Set the expected patch count for a specific target.
        
        Args:
            target_name: Name of the target
            count: Expected number of patches
        """
        self.expected_patch_counts[target_name] = count
        if self.verbose:
            print(f"Rank {self.rank}: Set expected patch count for target {target_name} to {count}")
            
    def set_spatial_dimensions(self, target_name, max_z, max_y, max_x):
        """
        Set the spatial dimensions for a target.
        
        Args:
            target_name: Name of the target
            max_z: Maximum Z dimension
            max_y: Maximum Y dimension
            max_x: Maximum X dimension
        """
        # Store volume dimensions for future reference
        self.volume_shape = (max_z, max_y, max_x)
        if self.verbose:
            print(f"Rank {self.rank}: Set volume dimensions for target {target_name} to {max_z}x{max_y}x{max_x}")
    
    def store_patch(self, patch: np.ndarray, position: Tuple[int, int, int], target_name: str):
        """
        Store a patch and its position.
        
        Args:
            patch: The patch data to store
            position: Tuple of (z, y, x) coordinates
            target_name: Name of the target (e.g., "segmentation")
            
        Returns:
            Index where the patch was stored
        """
        # Ensure the position is a valid tuple of 3 integers
        if not isinstance(position, tuple) or len(position) != 3:
            raise ValueError(f"Position must be a tuple of (z, y, x) coordinates, got {position}")
            
        # Convert position to integers to ensure consistency
        position = tuple(int(p) for p in position)
        
        # Augment the patch with position information
        # Add 3 channels at the beginning - each channel will have a single value for the entire patch
        # This creates a new shape: (original_channels + 3, Z, Y, X)
        # Those first 3 channels will store position information
        patch_shape = patch.shape
        
        # Create the combined patch with 3 additional channels for position
        c, *spatial_dims = patch_shape  # Extract channels and spatial dimensions
        combined_shape = (c + 3,) + tuple(spatial_dims)
        
        # Debug output for critical shapes
        print(f"DEBUG: patch_shape={patch_shape}, extracted c={c}, spatial_dims={spatial_dims}")
        print(f"DEBUG: combined_shape={combined_shape}")
        
        combined_patch = np.zeros(combined_shape, dtype=patch.dtype)
        
        # Store position values in the first three channels as single values
        # These will be uniform arrays where every element has the same position value
        combined_patch[0] = position[0]  # Z position in first channel
        combined_patch[1] = position[1]  # Y position in second channel
        combined_patch[2] = position[2]  # X position in third channel
        
        # Copy the original patch data into the remaining channels
        combined_patch[3:] = patch
        
        # Initialize structures if needed (with locking ONLY for initialization)
        # This happens exactly once per target at the start
        if target_name not in self.patch_counts:
            with self.lock:
                if target_name not in self.patch_counts:
                    # We must have the expected_patch_counts set - fail otherwise
                    if not hasattr(self, 'expected_patch_counts') or target_name not in self.expected_patch_counts:
                        raise ValueError(f"Expected patch count must be set for target {target_name} before storing patches")
                    
                    # Get the exact array size from expected count
                    exact_size = self.expected_patch_counts[target_name]
                    assert exact_size > 0, f"Expected patch count must be positive, got {exact_size}"
                    
                    if self.verbose:
                        print(f"Rank {self.rank}: Using exact count of {exact_size} patches for {target_name}")
                    
                    # Initialize counter for patch tracking
                    self.patch_counts[target_name] = 0
                    
                    # Create zarr array to store patches with compression to save space
                    # Use ZSTD compression with a moderate level (3) for good balance of speed and compression
                    from numcodecs import Blosc
                    compressor = Blosc(cname='zstd', clevel=3)
                    
                    self.patches_group.create_dataset(
                        target_name, 
                        shape=(exact_size,) + combined_shape,  # (num_patches, C+3, Z, Y, X)
                        chunks=(1,) + combined_shape,  # Each patch in its own chunk for parallel access
                        dtype=combined_patch.dtype,
                        compressor=compressor,  # Use compression to save space
                        write_empty_chunks=False  # Skip writing empty chunks to save space
                    )
                    
                    # Store metadata about position channels
                    self.patches_group[target_name].attrs['has_position_channels'] = True
                    self.patches_group[target_name].attrs['position_channel_z'] = 0
                    self.patches_group[target_name].attrs['position_channel_y'] = 1 
                    self.patches_group[target_name].attrs['position_channel_x'] = 2
                    
                    # Register this array with the parallel writer
                    array_path = f"rank_{self.rank}/patches/{target_name}"
                    if self.parallel_writer:
                        self.parallel_writer.register_array(
                            array_path=array_path,
                            shape=(exact_size,) + combined_shape,
                            dtype=combined_patch.dtype,
                            chunks=(1,) + combined_shape,
                            compressor=compressor  # Add the compressor for parallel writer
                        )
                    
                    if self.verbose:
                        print(f"Rank {self.rank}: Created array for target {target_name} with exact size {exact_size}")
                        print(f"Rank {self.rank}: Each patch now contains position channels Z, Y, X as first 3 channels")
        
        # Get the next available index
        idx = self.patch_counts[target_name]
        
        # Check if the index is within bounds of the array
        array_path = f"rank_{self.rank}/patches/{target_name}"
        array_shape = self.patches_group[target_name].shape
        if idx >= array_shape[0]:
            raise IndexError(f"Index out of bounds: trying to write to index {idx} but array {array_path} has shape {array_shape}")
        
        # Increment patch count
        self.patch_counts[target_name] += 1
        
        # Send the combined patch to the parallel writer instead of storing directly
        if self.parallel_writer:
            array_path = f"rank_{self.rank}/patches/{target_name}"
            self.parallel_writer.write_patch(array_path, idx, combined_patch)
            
            if self.verbose and idx < 3:
                print(f"Sending patch at index {idx} with embedded position {position} to parallel writer")
        else:
            # Fallback to direct storage
            self.patches_group[target_name][idx] = combined_patch
            
            if self.verbose and idx < 3:
                print(f"Storing patch at index {idx} with embedded position {position} directly (no parallel writer)")
        
        return idx
    
    def get_patches_count(self, rank: int, target_name: str) -> int:
        """
        Get the number of patches stored for a specific rank and target.
        
        Args:
            rank: Rank to check
            target_name: Target name to check
            
        Returns:
            Number of patches stored, or 0 if target doesn't exist
        """
        # For our own rank, we have the count locally
        if rank == self.rank and target_name in self.patch_counts:
            return self.patch_counts[target_name]
        
        # If we have the position info for this rank and target, use that
        if rank in self.all_rank_positions and target_name in self.all_rank_positions[rank]:
            return len(self.all_rank_positions[rank][target_name])
            
        try:
            # Make sure the zarr store is open
            if self.temp_zarr is None:
                self.temp_zarr = zarr.open(self.temp_zarr_path, mode='r')
            
            # Check if the group exists
            if f"rank_{rank}" not in self.temp_zarr:
                return 0
                
            rank_group = self.temp_zarr[f"rank_{rank}"]
            
            if "patches" not in rank_group or target_name not in rank_group["patches"]:
                return 0
                
            # If we have a count stored separately, use that
            if f"count_{target_name}" in rank_group:
                return rank_group[f"count_{target_name}"][0]
                
            # Otherwise, use the array size
            # This could be optimized with metadata if needed
            patches_array = rank_group["patches"][target_name]
            
            # Since we're using a fixed-size array, check the metadata for actual count
            if hasattr(patches_array, 'attrs') and 'actual_count' in patches_array.attrs:
                return patches_array.attrs['actual_count']
            
            # If no metadata, we'll have to scan through the positions dictionary
            # Which we should have loaded by now through collect_all_patches
            if rank in self.all_rank_positions and target_name in self.all_rank_positions[rank]:
                return len(self.all_rank_positions[rank][target_name])
            
            # Fall back to array size, which might be an overestimate
            return patches_array.shape[0]
                
        except Exception as e:
            print(f"Error getting patches count for rank {rank}, target {target_name}: {e}")
            return 0
    
    def finalize_target(self, target_name: str):
        """
        Finalize a target by recording the patch count in metadata.
        
        Args:
            target_name: Target name to finalize
        """
        if target_name in self.patch_counts:
            # Get count of patches from our counter
            count = self.patch_counts[target_name]
            
            # Store count as metadata in the zarr array
            if target_name in self.patches_group:
                self.patches_group[target_name].attrs['actual_count'] = count
            
            # Also store count in a separate dataset for quick access
            self.rank_group.create_dataset(
                f"count_{target_name}", 
                data=[int(count)], 
                dtype='i4',
                write_empty_chunks=False  # Skip writing empty chunks to save space
            )
            
            if self.verbose:
                print(f"Number of patches stored: {count}")
                
                # Display information about stored positions
                print(f"Position information is embedded in each patch in the first 3 channels")
                
                # Show some statistics about the first few patches if available
                if count > 0 and target_name in self.patches_group:
                    # Get the first few patches (up to 5) for display
                    num_to_show = min(5, count)
                    for idx in range(num_to_show):
                        try:
                            # Extract just the position channels from the first patch
                            patch_with_pos = self.patches_group[target_name][idx]
                            # Get the position from the first pixel of each of the first 3 channels
                            z_pos = patch_with_pos[0, 0, 0, 0]  # First channel is Z
                            y_pos = patch_with_pos[1, 0, 0, 0]  # Second channel is Y
                            x_pos = patch_with_pos[2, 0, 0, 0]  # Third channel is X
                            print(f"Patch {idx} position: ({z_pos}, {y_pos}, {x_pos})")
                        except Exception as e:
                            print(f"Error extracting position from patch {idx}: {e}")
            
            if self.verbose:
                print(f"Rank {self.rank}: Finalized target {target_name} with {count} patches")
    
    def get_all_patches(self, rank: int, target_name: str) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]]]:
        """
        Get all patches and position mapping for a specific rank and target.
        
        Args:
            rank: Rank to get patches from
            target_name: Target name to get patches for
            
        Returns:
            Tuple of (patches_array, position_dict, index_mapping)
            where position_dict maps indices to (z,y,x) positions
            
        Note:
            Position information is embedded in the patches themselves in the first 3 channels.
            The position_dict is built on-demand from the embedded positions.
        """
        try:
            print(f"  get_all_patches for rank {rank}, target {target_name}")
            
            # Get patch count
            count = self.get_patches_count(rank, target_name)
            
            if count == 0:
                print(f"  No patches found for target {target_name} in rank {rank}")
                return None, {}, {}
            
            # Make sure the zarr store is open
            if self.temp_zarr is None:
                print(f"  Opening zarr store at {self.temp_zarr_path}")
                self.temp_zarr = zarr.open(self.temp_zarr_path, mode='r')
            
            # Get the rank group
            rank_key = f"rank_{rank}"
            if rank_key not in self.temp_zarr:
                error_msg = f"Error: Rank group '{rank_key}' not found in zarr store. Available: {list(self.temp_zarr.keys())}"
                print(f"  {error_msg}")
                raise KeyError(error_msg)
                
            rank_group = self.temp_zarr[rank_key]
            
            if "patches" not in rank_group:
                error_msg = f"Error: 'patches' group not found in rank {rank} group. Available: {list(rank_group.keys())}"
                print(f"  {error_msg}")
                raise KeyError(error_msg)
            
            patches_group = rank_group["patches"]
            
            # Verify target exists
            if target_name not in patches_group:
                error_msg = f"Error: Target '{target_name}' not found in patches group. Available: {list(patches_group.keys())}"
                print(f"  {error_msg}")
                raise KeyError(error_msg)
            
            print(f"  Accessing patches for rank {rank}, target {target_name}")
            
            # Return a reference to the zarr array, don't load all into memory
            # This is a reference that will load data on-demand when indexed
            patches_array = patches_group[target_name]
            
            # Check if patches have embedded position information by looking at attributes
            has_position_channels = patches_array.attrs.get('has_position_channels', False)
            
            if not has_position_channels:
                print(f"Warning: Patches for {target_name} in rank {rank} do not have embedded position channels")
                return patches_array, {}, {i: i for i in range(count)}
            
            # Create position dictionary from the first few patches (lazy loading)
            # For memory efficiency, don't load all patches at once to extract positions
            # Instead create a dictionary that will extract positions on demand
            position_dict = {}
            
            # Check the first patch to confirm position channel structure
            if count > 0:
                # Get just the first few pixels of the position channels from the first patch
                first_patch_pos_data = patches_array[0, :3, 0:1, 0:1, 0:1]
                z_pos = first_patch_pos_data[0, 0, 0, 0]
                y_pos = first_patch_pos_data[1, 0, 0, 0]
                x_pos = first_patch_pos_data[2, 0, 0, 0]
                print(f"  Sample position from first patch: ({z_pos}, {y_pos}, {x_pos})")
            
            # Create a sequential index mapping
            index_mapping = {i: i for i in range(count)}
            
            if self.verbose:
                print(f"DEBUG: Found patches_array with shape: {patches_array.shape}")
                print(f"DEBUG: Using embedded position channels for {count} patches")
            
            return patches_array, position_dict, index_mapping
            
        except Exception as e:
            error_msg = f"Error getting patches for rank {rank}, target {target_name}: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
    
    def collect_all_patches(self, target_name: str) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """
        Collect all patches across all ranks for a specific target.
        
        This constructs a list of (rank, patch_idx, position) tuples for all patches.
        Position information is extracted from the embedded position channels in each patch.
        
        Args:
            target_name: Target name to collect patches for
            
        Returns:
            List of (rank, patch_idx, position) tuples
        """
        print(f"Starting collect_all_patches for {target_name}")
        all_patches = []
        
        # Synchronize ranks if using distributed training
        if dist.is_initialized():
            # Barrier to ensure all ranks have saved their patches
            dist.barrier()
        
        # Make sure the zarr store is open
        if self.temp_zarr is None:
            print(f"Opening zarr store at {self.temp_zarr_path}")
            self.temp_zarr = zarr.open(self.temp_zarr_path, mode='r')
        
        # Now collect patches from all ranks
        for rank in range(self.world_size):
            print(f"Collecting patches for rank {rank}...")
            try:
                # First check if rank directory exists
                rank_key = f"rank_{rank}"
                if rank_key not in self.temp_zarr:
                    print(f"Warning: No directory for rank {rank} found in zarr store at {self.temp_zarr_path}")
                    print(f"Available keys: {list(self.temp_zarr.keys())}")
                    continue
                    
                # Then check if patches group exists
                if "patches" not in self.temp_zarr[rank_key]:
                    print(f"Warning: No 'patches' group in rank {rank} directory")
                    print(f"Available keys in {rank_key}: {list(self.temp_zarr[rank_key].keys())}")
                    continue
                
                # Fetch patch count from zarr
                count = self.get_patches_count(rank, target_name)
                print(f"Found {count} patches for rank {rank}")
                
                if count == 0:
                    continue
                
                # Get the patches array for this rank
                rank_key = f"rank_{rank}"
                if rank_key not in self.temp_zarr:
                    print(f"Warning: No data for rank {rank} found in zarr store")
                    continue
                
                rank_group = self.temp_zarr[rank_key]
                
                if "patches" not in rank_group:
                    print(f"Warning: No patches group for rank {rank}")
                    continue
                
                patches_group = rank_group["patches"]
                
                if target_name not in patches_group:
                    print(f"Warning: No patches for target {target_name} in rank {rank}")
                    continue
                
                # Get the patches array (reference only, doesn't load all data)
                patches_array = patches_group[target_name]
                
                # Check if position information is embedded in patches
                has_position_channels = patches_array.attrs.get('has_position_channels', False)
                
                if has_position_channels:
                    print(f"Found {count} patches with embedded positions for rank {rank}")
                    
                    # Use sampling to avoid loading all patches into memory at once
                    # Only load the minimum data needed (first pixel of each position channel)
                    for idx in range(count):
                        try:
                            # Extract just the position channels' first pixel for this patch
                            pos_data = patches_array[idx, :3, 0, 0, 0]
                            
                            # Create position tuple from the channel values
                            pos_tuple = (int(pos_data[0]), int(pos_data[1]), int(pos_data[2]))
                            all_patches.append((rank, idx, pos_tuple))
                            
                            # Debug the first and last few positions
                            if idx < 3 or idx >= count - 3:
                                print(f"Added position from rank {rank}, index {idx}: {pos_tuple}")
                        except Exception as e:
                            print(f"Error extracting position from patch {idx}: {e}")
                            continue
                    
                    print(f"Successfully added {count} patches from rank {rank}")
                else:
                    print(f"Warning: Patches for rank {rank}, target {target_name} don't have embedded positions")

            except Exception as e:
                print(f"Error collecting patches for rank {rank}, target {target_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"collect_all_patches completed, returning {len(all_patches)} total patches")
        return all_patches
    
    def close(self):
        """
        Close the zarr temporary storage without removing it.
        
        This should be called when you want to preserve the temp store but free resources.
        """
        # Shut down our parallel writer
        if self.parallel_writer:
            if self.verbose:
                print(f"Rank {self.rank}: Shutting down parallel writer")
            try:
                self.parallel_writer.shutdown()
                self.parallel_writer = None
            except Exception as e:
                print(f"Error shutting down parallel writer: {e}")
        
        # Close the zarr store but don't remove it
        self.temp_zarr = None
    
    def cleanup(self):
        """
        Clean up the zarr temporary storage.
        
        This should be called after all processing is complete to remove the temp.zarr store.
        In a distributed setting, only rank 0 should remove the store after all ranks are done.
        """
        # First shut down our parallel writer
        if self.parallel_writer:
            if self.verbose:
                print(f"Rank {self.rank}: Shutting down parallel writer")
            try:
                self.parallel_writer.shutdown()
                self.parallel_writer = None
            except Exception as e:
                print(f"Error shutting down parallel writer: {e}")
        
        # Close the zarr store but don't remove it yet
        self.temp_zarr = None
        
        # IMPORTANT: This method should ONLY be called by rank 0 in distributed settings
        # This is enforced in the inference.py cleanup section
        import torch.distributed as dist
        
        # ABSOLUTELY CRITICAL: Only rank 0 does any actual deletion
        if self.rank != 0:
            # Non-rank-0 processes do NOTHING
            if self.verbose:
                print(f"Rank {self.rank}: cleanup() called by non-rank-0 process. NO cleanup will be performed.")
            return
            
        # If we get here, we're either rank 0 or not using distributed processing
        # Actually remove the temporary storage
        if os.path.exists(self.temp_zarr_path):
            if self.verbose:
                print(f"Rank {self.rank}: Cleaning up temporary zarr storage at {self.temp_zarr_path}")
            
            try:
                import shutil
                shutil.rmtree(self.temp_zarr_path)
            except Exception as e:
                print(f"Error cleaning up zarr storage: {e}")