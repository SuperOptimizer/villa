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
        
        # Initialize structures if needed (with locking ONLY for initialization)
        # This happens exactly once per target at the start
        if target_name not in self.positions:
            with self.lock:
                if target_name not in self.positions:
                    # We must have the expected_patch_counts set - fail otherwise
                    if not hasattr(self, 'expected_patch_counts') or target_name not in self.expected_patch_counts:
                        raise ValueError(f"Expected patch count must be set for target {target_name} before storing patches")
                    
                    # Get the exact array size from expected count
                    exact_size = self.expected_patch_counts[target_name]
                    assert exact_size > 0, f"Expected patch count must be positive, got {exact_size}"
                    
                    if self.verbose:
                        print(f"Rank {self.rank}: Using exact count of {exact_size} patches for {target_name}")
                    
                    # Initialize positions dictionary for this target
                    self.positions[target_name] = {}
                    self.patch_counts[target_name] = 0
                    
                    # Initialize positions in the all rank positions dictionary
                    if self.rank not in self.all_rank_positions:
                        self.all_rank_positions[self.rank] = {}
                    self.all_rank_positions[self.rank][target_name] = {}
                    
                    # Get shape and dtype for patch array
                    patch_shape = patch.shape
                    patch_dtype = patch.dtype
                    
                    # Create a simple zarr array to store patches without compression for maximum write speed
                    # No compression for temp storage to maximize I/O throughput
                    self.patches_group.create_dataset(
                        target_name, 
                        shape=(exact_size,) + patch_shape,  # (num_patches, C, Z, Y, X)
                        chunks=(1,) + patch_shape,  # Each patch in its own chunk for parallel access
                        dtype=patch_dtype,
                        compressor=None,  # No compression for faster writes
                        write_empty_chunks=False  # Skip writing empty chunks to save space
                    )
                    
                    # Register this array with the parallel writer
                    array_path = f"rank_{self.rank}/patches/{target_name}"
                    if self.parallel_writer:
                        self.parallel_writer.register_array(
                            array_path=array_path,
                            shape=(exact_size,) + patch_shape,
                            dtype=patch_dtype,
                            chunks=(1,) + patch_shape
                        )
                    
                    if self.verbose:
                        print(f"Rank {self.rank}: Created array for target {target_name} with exact size {exact_size}")
                        print(f"Rank {self.rank}: Using simple dictionary to track positions")
        
        # Get the next available index
        idx = self.patch_counts[target_name]
        
        # Check if the index is within bounds of the array
        array_path = f"rank_{self.rank}/patches/{target_name}"
        array_shape = self.patches_group[target_name].shape
        if idx >= array_shape[0]:
            raise IndexError(f"Index out of bounds: trying to write to index {idx} but array {array_path} has shape {array_shape}")
        
        # Store position in our tracking dictionary
        self.positions[target_name][idx] = position
        self.all_rank_positions[self.rank][target_name][idx] = position
        
        # Increment patch count
        self.patch_counts[target_name] += 1
        
        # Send the patch to the parallel writer instead of storing directly
        if self.parallel_writer:
            array_path = f"rank_{self.rank}/patches/{target_name}"
            self.parallel_writer.write_patch(array_path, idx, patch)
            
            if self.verbose and idx < 3:
                print(f"Sending patch at index {idx} with position {position} to parallel writer")
        else:
            # Fallback to direct storage
            self.patches_group[target_name][idx] = patch
            
            if self.verbose and idx < 3:
                print(f"Storing patch at index {idx} with position {position} directly (no parallel writer)")
        
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
        Finalize a target by recording the positions in metadata.
        
        Args:
            target_name: Target name to finalize
        """
        if target_name in self.positions:
            # Get the positions for this target
            positions_dict = self.positions[target_name]
            
            # Calculate count of patches
            count = len(positions_dict)
            
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
            
            # Store positions in zarr for other ranks to access
            if 'positions' not in self.rank_group:
                positions_group = self.rank_group.create_group('positions')
            else:
                positions_group = self.rank_group['positions']
                
            # Convert positions dictionary to array for storage
            if count > 0:
                positions_array = np.zeros((count, 3), dtype=np.int32)
                for idx, pos in positions_dict.items():
                    if idx < count:  # Safety check
                        positions_array[idx] = np.array(pos, dtype=np.int32)
                
                # Save to zarr without compression for maximum write speed
                positions_group.create_dataset(
                    target_name,
                    data=positions_array,
                    dtype='i4',
                    compressor=None,  # No compression for faster writes
                    write_empty_chunks=False  # Skip writing empty chunks to save space
                )
                
                if self.verbose:
                    print(f"Saved {count} positions to zarr for target {target_name}")
            
            if self.verbose:
                print(f"Number of patches stored: {count}")
                if count > 0:
                    # Show first few positions
                    keys_to_show = sorted(positions_dict.keys())[:5] if len(positions_dict) >= 5 else sorted(positions_dict.keys())
                    first_positions = [(idx, positions_dict[idx]) for idx in keys_to_show]
                    print(f"First few positions: {first_positions}")
                    
                    # Show last few positions
                    last_keys = sorted(positions_dict.keys())[-5:] if len(positions_dict) >= 5 else sorted(positions_dict.keys())
                    last_positions = [(idx, positions_dict[idx]) for idx in last_keys]
                    print(f"Last few positions: {last_positions}")
            
            if self.verbose:
                print(f"Rank {self.rank}: Finalized target {target_name} with {count} patches and positions")
    
    def get_all_patches(self, rank: int, target_name: str) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]]]:
        """
        Get all patches and position mapping for a specific rank and target.
        
        Args:
            rank: Rank to get patches from
            target_name: Target name to get patches for
            
        Returns:
            Tuple of (patches_array, position_dict, index_mapping)
            where position_dict maps indices to (z,y,x) positions
        """
        try:
            print(f"  get_all_patches for rank {rank}, target {target_name}")
            
            # Get patch count
            count = self.get_patches_count(rank, target_name)
            
            if count == 0:
                print(f"  No patches found for target {target_name} in rank {rank}")
                return None, {}, {}
            
            # If this is our own rank, we can access the data directly
            if rank == self.rank and target_name in self.positions:
                print(f"  Getting patches from our own rank {rank}")
                
                # Get all positions from our dictionary
                position_dict = self.positions[target_name]
                
                if self.verbose:
                    print(f"  Found {len(position_dict)} positions")
                
                # Return a reference to the zarr array, don't load all into memory
                # This is a reference that will load data on-demand when indexed
                patches_array = self.patches_group[target_name]
                
                # Create a simple sequential index mapping (identity mapping)
                index_mapping = {i: i for i in range(count)}
                
                print(f"  Returning {count} patches for rank {rank}, target {target_name}")
                return patches_array, position_dict, index_mapping
            
            # Otherwise we're reading from another rank via zarr
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
            
            # Get position dictionary if we have it in memory
            if rank in self.all_rank_positions and target_name in self.all_rank_positions[rank]:
                position_dict = self.all_rank_positions[rank][target_name]
            else:
                # Try to load positions from zarr storage
                position_dict = {}
                try:
                    # Check if positions group exists
                    if "positions" in rank_group and target_name in rank_group["positions"]:
                        positions_dataset = rank_group["positions"][target_name]
                        
                        # Read all positions at once
                        positions_array = positions_dataset[:]
                        
                        # Check dataset shape
                        if len(positions_array.shape) != 2 or positions_array.shape[1] != 3:
                            print(f"Warning: Invalid positions dataset shape: {positions_array.shape}")
                        else:
                            # Convert to dictionary of tuples
                            for idx in range(positions_array.shape[0]):
                                pos = tuple(int(x) for x in positions_array[idx])
                                position_dict[idx] = pos
                            
                            print(f"Loaded {len(position_dict)} positions from zarr for rank {rank}")
                            
                            # Cache the positions for future use
                            if rank not in self.all_rank_positions:
                                self.all_rank_positions[rank] = {}
                            self.all_rank_positions[rank][target_name] = position_dict
                    else:
                        print(f"Warning: No position data for rank {rank}, target {target_name} in zarr storage")
                except Exception as e:
                    print(f"Error loading positions from zarr: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with empty dict as fallback
            
            # Create a simple sequential index mapping
            index_mapping = {i: i for i in range(count)}
            if self.verbose:
                print(f"DEBUG: Created patches_array with shape: {patches_array.shape}")
                print(f"DEBUG: Position dict has {len(position_dict)} entries")
            
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
        
        Args:
            target_name: Target name to collect patches for
            
        Returns:
            List of (rank, patch_idx, position) tuples
        """
        print(f"Starting collect_all_patches for {target_name}")
        all_patches = []
        
        # Share position information between all ranks if using distributed training
        if dist.is_initialized():
            # Create an explicit collective operation to share position data between ranks
            # This ensures all ranks have position information from all other ranks
            
            # First, collect our own positions
            if target_name in self.positions:
                # Using our own memory for our own rank
                positions_dict = self.positions[target_name]
                
                # Update our entry in all_rank_positions
                if self.rank not in self.all_rank_positions:
                    self.all_rank_positions[self.rank] = {}
                self.all_rank_positions[self.rank][target_name] = positions_dict
                
                if self.verbose:
                    print(f"Collected our own positions for rank {self.rank}, target {target_name}: {len(positions_dict)} entries")
                    
            # Barrier to ensure all ranks have processed their positions
            dist.barrier()
        
        # Now collect patches from all ranks
        for rank in range(self.world_size):
            print(f"Collecting patches for rank {rank}...")
            try:
                # First try to use in-memory data if available for our own rank
                if rank == self.rank and target_name in self.positions:
                    # Using our own memory for our own rank
                    positions_dict = self.positions[target_name]
                    count = len(positions_dict)
                    
                    print(f"Using in-memory positions for rank {self.rank}, found {count} patches")
                    
                    # Add each patch to the list with its position
                    for idx, pos in positions_dict.items():
                        try:
                            # Ensure position is a proper (z,y,x) tuple with exactly 3 integer values
                            if not isinstance(pos, tuple) or len(pos) != 3:
                                raise ValueError(f"Invalid position format: {pos}, expected a tuple of exactly 3 values")
                                
                            # Convert to integers to ensure consistency
                            pos_tuple = (int(pos[0]), int(pos[1]), int(pos[2]))
                            all_patches.append((rank, idx, pos_tuple))
                            
                            # Debug the first and last few positions
                            if self.verbose:
                                if idx < 3 or idx >= count - 3:
                                    print(f"Added position from rank {rank}, index {idx}: {pos}")
                        except Exception as e:
                            print(f"Error processing in-memory position: {e}")
                            raise  # Re-raise to abort processing - this is a critical error
                    
                    print(f"Successfully added {count} patches from rank {rank}")
                    continue  # Skip to next rank
                
                # Otherwise check if we already have the position data in memory
                if rank in self.all_rank_positions and target_name in self.all_rank_positions[rank]:
                    positions_dict = self.all_rank_positions[rank][target_name]
                    count = len(positions_dict)
                    
                    print(f"Using cached positions for rank {rank}, found {count} patches")
                    
                    # Add each patch to the list with its position
                    for idx, pos in positions_dict.items():
                        try:
                            # Ensure position is a proper tuple
                            if not isinstance(pos, tuple) or len(pos) != 3:
                                raise ValueError(f"Invalid position format: {pos}")
                                
                            # Convert to integers to ensure consistency
                            pos_tuple = (int(pos[0]), int(pos[1]), int(pos[2]))
                            all_patches.append((rank, idx, pos_tuple))
                            
                            # Debug the first and last few positions
                            if idx < 3 or idx >= count - 3:
                                print(f"Added position from rank {rank}, index {idx}: {pos}")
                        except Exception as e:
                            print(f"Error processing cached position: {e}")
                            continue
                    
                    print(f"Successfully added {count} patches from rank {rank}")
                    continue  # Skip to next rank
                
                # Otherwise fetch position data from zarr (for non-distributed case or fallback)
                count = self.get_patches_count(rank, target_name)
                print(f"Found {count} patches for rank {rank}")
                
                if count == 0:
                    continue
                
                try:
                    # First get the rank group from zarr
                    rank_key = f"rank_{rank}"
                    if rank_key not in self.temp_zarr:
                        print(f"Warning: No data for rank {rank} found in zarr store")
                        continue
                    
                    rank_group = self.temp_zarr[rank_key]
                    
                    # Check if positions group exists
                    if "positions" not in rank_group:
                        print(f"Warning: No positions group for rank {rank}")
                        continue
                    
                    # Check if target exists in positions group
                    positions_group = rank_group["positions"]
                    if target_name not in positions_group:
                        print(f"Warning: No positions for target {target_name} in rank {rank}")
                        continue
                    
                    # Get positions dataset
                    positions_dataset = positions_group[target_name]
                    
                    # Check dataset shape
                    if len(positions_dataset.shape) != 2 or positions_dataset.shape[1] != 3:
                        print(f"Warning: Invalid positions dataset shape: {positions_dataset.shape}")
                        continue
                    
                    # Read all positions at once and convert to tuples
                    positions_array = positions_dataset[:]
                    
                    # Add positions to the all_patches list
                    for idx in range(positions_array.shape[0]):
                        pos = tuple(int(x) for x in positions_array[idx])
                        all_patches.append((rank, idx, pos))
                        
                        # Debug logging for first and last few positions
                        if idx < 3 or idx >= positions_array.shape[0] - 3:
                            print(f"Added position from zarr for rank {rank}, index {idx}: {pos}")
                    
                    print(f"Successfully added {positions_array.shape[0]} patches from rank {rank} (loaded from zarr)")
                except Exception as e:
                    print(f"Error loading positions from zarr for rank {rank}: {e}")
                    import traceback
                    traceback.print_exc()

            except Exception as e:
                print(f"Error collecting patches for rank {rank}, target {target_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"collect_all_patches completed, returning {len(all_patches)} total patches")
        return all_patches
    
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