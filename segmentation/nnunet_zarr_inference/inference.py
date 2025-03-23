import os
import sys
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import torch.distributed as dist
from scipy.ndimage import gaussian_filter
import uuid
import queue
import time

# Add parent directory to sys.path for direct imports when running the script directly
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Now we can import as a module
    # print(f"Added {parent_dir} to sys.path")

# Import blending and activation
if __name__ == "__main__" or not __package__:
    # Direct script execution
    from nnunet_zarr_inference.blending_torch import (create_gaussian_weights_torch,
                                                      blend_patch_torch)
else:
    # Module import
    from .blending_torch import (create_gaussian_weights_torch,
                                 blend_patch_torch)

# Import our zarr cache and temp storage
if __name__ == "__main__" or not __package__:
    # Direct script execution
    from nnunet_zarr_inference.zarr_cache import ZarrArrayLRUCache
    from nnunet_zarr_inference.zarr_temp_storage import ZarrTempStorage
    from nnunet_zarr_inference.zarr_writer_worker import zarr_writer_worker
else:
    # Module import
    from .zarr_cache import ZarrArrayLRUCache
    from .zarr_temp_storage import ZarrTempStorage
    from .zarr_writer_worker import zarr_writer_worker

# Try to configure NumPy and related libraries to use multiple threads
try:
    # Determine optimal number of threads for various operations
    num_physical_cores = os.cpu_count() // 2 if os.cpu_count() else 4
    num_threads = min(8, num_physical_cores)

    # Configure NumPy threading
    np.set_num_threads(num_threads)

    # Try to configure OpenMP for libraries like MKL
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
except (AttributeError, ImportError):
    pass

# Determine if this is being run as a module or directly
import sys
import os

if __name__ == "__main__" or not __package__:
    # Add parent directory to sys.path for direct script execution
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import directly
    from nnunet_zarr_inference.load_nnunet_model import load_model, run_inference
    from nnunet_zarr_inference.inference_dataset import InferenceDataset
    from nnunet_zarr_inference.helpers import compute_steps_for_sliding_window_tuple
else:
    # Relative imports for package/module usage
    from .load_nnunet_model import load_model, run_inference
    from .inference_dataset import InferenceDataset
    from .helpers import compute_steps_for_sliding_window_tuple


class SequentialZarrNNUNetInference:
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 model_folder: str,
                 fold: Union[int, str] = 0,
                 checkpoint_name: str = 'checkpoint_final.pth',
                 patch_size: Optional[Tuple[int, int, int]] = None,
                 batch_size: int = 4,
                 step_size: float = 0.5,
                 num_dataloader_workers: int = 4,
                 num_write_workers: int = 4,
                 input_format: str = 'zarr',
                 load_all: bool = False,
                 device: str = 'cuda',
                 threshold: Optional[float] = None,
                 use_mirroring: bool = True,
                 max_tta_combinations: Optional[int] = None,
                 verbose: bool = False,
                 save_probability_maps: bool = True,
                 output_targets: Optional[List[Dict[str, Any]]] = None,
                 rank: int = 0,
                 edge_weight_boost: float = 0.5,
                 cache_size: int = 256,
                 max_cache_bytes: float = 4.0):
        """
        A sequential approach to nnUNet inference on zarr arrays.

        This implementation differs from the original ZarrNNUNetInferenceHandler by:
        1. Using a sequential writing approach to avoid chunk conflicts
        2. Writing each patch to a separate chunk in a temporary storage
        3. Only performing the Gaussian blending at the end during a final reduction pass

        Args:
            input_path: Path to the input zarr store
            output_path: Path to save the output zarr store
            model_folder: Path to the nnUNet model folder
            fold: Which fold to load (default: 0, can also be 'all')
            checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
            patch_size: Optional override for the patch size
            batch_size: Batch size for inference
            step_size: Step size for sliding window prediction as a fraction of patch_size (default: 0.5, nnUNet default)
            num_dataloader_workers: Number of workers for the DataLoader
            num_write_workers: Number of worker threads for asynchronous disk writes
            input_format: Format of the input data ('zarr' supported currently)
            load_all: Whether to load the entire array into memory
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Optional threshold value (0-100) for binarizing the probability map
            use_mirroring: Enable test time augmentation via mirroring (default: True, matches nnUNet default)
            max_tta_combinations: Maximum number of TTA combinations to use (default: None = auto-detect based on GPU memory)
            verbose: Enable detailed output messages during inference (default: False)
            save_probability_maps: Save full probability maps for multiclass segmentation (default: True, set to False to save space)
            output_targets: Optional list of output target configurations, each a dictionary with 'name' and other parameters
            rank: Process rank for distributed processing (default: 0, only process 0 will print verbose outputs)
            edge_weight_boost: Factor to boost Gaussian weights at patch edges (default: 0.5). Higher values reduce artifacts
                               at volume boundaries but may affect blending quality elsewhere. Set to 0 for original behavior.
            cache_size: Number of zarr chunks to cache in memory (default: 256)
            max_cache_bytes: Maximum memory in GB to use for zarr cache (default: 4.0)
        """
        self.input_path = input_path
        
        # Ensure output path ends with .zarr
        if not output_path.endswith('.zarr'):
            raise ValueError(f"Output path must end with '.zarr', got: {output_path}")
        self.output_path = output_path
        
        self.model_folder = model_folder
        self.fold = fold
        self.checkpoint_name = checkpoint_name
        self.batch_size = batch_size
        self.tile_step_size = step_size  # Using nnUNet's naming convention
        self.num_dataloader_workers = num_dataloader_workers
        self.input_format = input_format
        self.load_all = load_all
        self.device_str = device
        self.threshold = threshold
        self.use_mirroring = use_mirroring
        self.max_tta_combinations = max_tta_combinations
        self.verbose = verbose
        self.save_probability_maps = save_probability_maps
        self.rank = rank  # Store rank for distributed training awareness
        self.edge_weight_boost = edge_weight_boost  # Store edge weight boost factor
        self.cache_size = cache_size  # Number of zarr chunks to cache
        self.max_cache_bytes = max_cache_bytes  # Maximum cache size in GB

        if max_tta_combinations == 0:
            self.use_mirroring = False

        # Convert patch_size to tuple if it's a list or another sequence
        if patch_size is not None:
            self.patch_size = tuple(patch_size)

        # Use the provided output_targets or default to a standard configuration
        # Always use the list format expected by InferenceDataset
        self.targets = [
            {
                "name": "segmentation",  # Internal name for tracking
                "channels": 2,  # nnUNet always outputs at least 2 channels (background, foreground)
                "activation": "sigmoid",
                "nnunet_output_channels": 2  # This is the minimum number of channels
            }
        ]
        
        # Define output array name (when writing to the zarr file)
        # We'll use an empty string to write directly to the zarr root
        self.output_array_name = ""

        # Define foreground channel index for reference (not used for special handling)
        self.nnunet_foreground_channel = 1  # Second channel (index 1) is foreground in binary segmentation

        if self.verbose:
            print(f"Initialized with step_size={self.tile_step_size}")
            for target in self.targets:
                print(f"Output target '{target.get('name')}': {target}")

        # Determine rank for DDP
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Writer queue for patches
        max_queue = 300
        self.writer_queue = queue.Queue(maxsize=max_queue)

        # Load the nnUNet model
        self.model_info = None
        self.patch_size = patch_size  # This will be overridden if None and model is loaded

        # Number of writer threads
        self.num_write_workers = num_write_workers

        # For TTA, we'll use only the 3 primary axis for mirroring
        self.tta_directions = [(0,), (1,), (2,)]  # Only axis-aligned flips
        # Always use exactly 3 TTA combinations (one for each axis) for efficient inference
        self.max_tta_combinations = max_tta_combinations

        # Override any provided value to always use the 3 primary axis flips
        
        # Zarr temp storage - will be initialized in infer()
        self.temp_storage = None

    def _load_nnunet_model(self):
        """
        Load the nnUNet model and return model information.
        """
        try:
            # Only print from rank 0
            if self.rank == 0:
                print(f"Loading nnUNet model from {self.model_folder}, fold {self.fold}")

            if self.verbose and self.rank == 0:
                print(f"Test time augmentation (mirroring): {'enabled' if self.use_mirroring else 'disabled'}")

            # Set verbose to False for non-rank-0 processes to avoid duplicate messages
            local_verbose = self.verbose and self.rank == 0

            model_info = load_model(
                model_folder=self.model_folder,
                fold=self.fold,
                checkpoint_name=self.checkpoint_name,
                device=self.device_str,
                use_mirroring=self.use_mirroring,
                verbose=local_verbose,
                rank=self.rank  # Pass rank to load_model
            )

            # Use the model's patch size if none was specified
            if self.patch_size is None:
                # Always convert to tuple for consistency (model_info often has it as a list)
                self.patch_size = tuple(model_info['patch_size'])
                if self.verbose and self.rank == 0:
                    print(f"Using model's patch size: {self.patch_size}")

            # Report multiclass vs binary if rank 0
            if self.rank == 0:
                num_classes = model_info.get('num_seg_heads', 0)
                if num_classes > 2:
                    print(f"Detected multiclass model with {num_classes} classes from model_info")
                elif num_classes == 2:
                    print(f"Detected binary segmentation model from model_info")

            return model_info
        except Exception as e:
            print(f"Error loading nnUNet model (rank {self.rank}): {str(e)}")
            import traceback
            traceback.print_exc()
            raise


    def _blend_patches(self, patch_arrays, output_arrays, count_arrays):
        """
        Blend patches using a simplified z-chunking approach.
        Each patch is processed exactly once in the first chunk it intersects with.
        This eliminates boundary artifacts and simplifies the code.
        """
        if self.verbose:
            print("Starting patch blending phase...")

        # Ensure patch_size is a tuple
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # Create Gaussian blend weights - use device from model
        device = self.device_str

        # Adjusted nnUNet parameters for smoother blending
        sigma_scale = 1 / 8  # Increased from standard 1/8 (0.125) for smoother transitions
        value_scaling_factor = 10  # Standard nnUNet value

        if self.verbose:
            print(f"Using sigma_scale={sigma_scale}, value_scaling_factor={value_scaling_factor}")

        # Import intersects_chunk from blending_torch
        if __name__ == "__main__" or not __package__:
            from nnunet_zarr_inference.blending_torch import create_gaussian_weights_torch, blend_patch_torch, \
                intersects_chunk
        else:
            from .blending_torch import create_gaussian_weights_torch, blend_patch_torch, intersects_chunk

        # Create Gaussian weights with standard nnUNet parameters
        blend_weights = create_gaussian_weights_torch(
            patch_size_tuple,
            sigma_scale=sigma_scale,
            value_scaling_factor=value_scaling_factor,
            device=device,
            edge_weight_boost=0
        )

        # Process each target
        for target in self.targets:
            tgt_name = target.get("name")
            if tgt_name not in patch_arrays:
                print(f"Warning: No patches found for target {tgt_name}")
                continue

            # Collect all patches across ranks using our simpler approach
            print(f"Collecting patches for target {tgt_name} from all ranks...")
            
            # First, collect positions and patch data for all ranks
            all_patch_info = self.temp_storage.collect_all_patches(tgt_name)
            
            if self.verbose:
                print(f"Collected metadata for {len(all_patch_info)} patches for target {tgt_name} across all ranks")
            
            # Create a lookup dictionary for all patches - directly maps (rank, idx) -> patch data
            print(f"Fetching patch data for {len(all_patch_info)} patches...")
            patch_data_lookup = {}  # Dictionary to store patch data by (rank, idx)
            
            # Group patch info by rank for efficient fetching
            patches_by_rank = {}
            for rank_idx, idx, pos in all_patch_info:
                if rank_idx not in patches_by_rank:
                    patches_by_rank[rank_idx] = []
                patches_by_rank[rank_idx].append((idx, pos))
            
            # Store arrays and indices for deferred loading - don't load all patches in memory
            patch_arrays_by_rank = {}  # Store array references by rank
            
            for rank_idx, patches in patches_by_rank.items():
                try:
                    # Get the patches array and position dictionary in one call
                    # This returns a reference to the zarr array, not the actual data
                    patches_array, position_dict, _ = self.temp_storage.get_all_patches(rank_idx, tgt_name)
                    
                    if patches_array is not None:
                        # Store the array reference for this rank
                        patch_arrays_by_rank[rank_idx] = patches_array
                        print(f"Added reference to patches array for rank {rank_idx} with {len(patches)} patches")
                        
                        # Only store the mapping of (rank,idx) -> position, not the actual patch data
                        for idx, pos in patches:
                            # Just register the patch in the lookup dictionary with None to indicate it exists
                            # We'll load it on-demand when processing patches
                            patch_data_lookup[(rank_idx, idx)] = (rank_idx, idx)
                            
                            # Print diagnostics for the first few patches
                            if len(patch_data_lookup) <= 5 or len(patch_data_lookup) % 100 == 0:
                                print(f"Registered patch: rank={rank_idx}, idx={idx}, pos={pos}")
                    else:
                        print(f"No patches array found for rank {rank_idx}")
                except Exception as e:
                    print(f"Error setting up patch references for rank {rank_idx}: {e}")
            
            # Print diagnostic information about the lookup table
            print(f"Successfully registered {len(patch_data_lookup)} patches for deferred loading")
            
            # We'll modify the patch loading code to use these references when needed
            # Store the array references in the instance for later use
            self.patch_arrays_by_rank = patch_arrays_by_rank
            
            # Get array dimensions
            c, max_z, max_y, max_x = output_arrays[tgt_name].shape

            # Process in chunks to manage memory - standard approach
            chunk_size = 256  # Standard chunk size

            # Sort patches by all position components (z, y, x) for consistent ordering
            # This ensures a predictable spatial ordering regardless of how patches were stored
            all_patch_info.sort(key=lambda x: (x[2][0], x[2][1], x[2][2]))

            if self.verbose:
                print(f"Using chunk size {chunk_size} without overlap (nnU-Net standard approach)")
                print(f"Each patch will contribute to all chunks it intersects with")

            # Process patches in z-chunks - standard non-overlapping chunks
            total_chunks = (max_z + chunk_size - 1) // chunk_size
            print(f"Processing {total_chunks} z-chunks with {len(all_patch_info)} patches")
            chunk_start_time = time.time()
            
            for chunk_idx, z_start in enumerate(range(0, max_z, chunk_size)):
                z_end = min(z_start + chunk_size, max_z)
                print(f"Processing z-chunk {chunk_idx+1}/{total_chunks}: [{z_start}:{z_end}]...")
                chunk_process_start = time.time()

                # Load chunk data - but use direct reference when possible rather than copying
                # For output array - directly reference the zarr array and convert the slice to tensor
                output_chunk = output_arrays[tgt_name][:, z_start:z_end]
                count_chunk = count_arrays[tgt_name][z_start:z_end]
                
                # Create tensors directly from the zarr array views - avoid copying data in memory first
                # Use float16 for output tensor to reduce memory usage
                output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float16).contiguous()
                count_tensor = torch.as_tensor(count_chunk, device=device, dtype=torch.float16).contiguous()

                # Find ALL patches that intersect with this chunk
                chunk_patches = []
                for patch_idx, patch_info in enumerate(all_patch_info):
                    rank_idx, idx, (z, y, x) = patch_info

                    # Use the simpler intersection check
                    if intersects_chunk(z, y, x, patch_size_tuple, z_start, z_end):
                        chunk_patches.append((rank_idx, idx, (z, y, x)))

                if self.verbose:
                    print(f"Processing chunk [{z_start}:{z_end}] with {len(chunk_patches)} intersecting patches")

                # Process each patch that intersects with this chunk
                print(f"Processing chunk [{z_start}:{z_end}] - {len(chunk_patches)} patches...")
                patch_counter = 0
                
                for rank_idx, idx, (z, y, x) in chunk_patches:
                    # Debug progress periodically
                    patch_counter += 1
                    if patch_counter % 50 == 0 or patch_counter == 1 or patch_counter == len(chunk_patches):
                        print(f"Processing patch {patch_counter}/{len(chunk_patches)}")

                    # Get patch data from our pre-filled lookup dictionary
                    # Much faster than fetching from disk each time
                    if (rank_idx, idx) not in patch_data_lookup:
                        print(f"WARNING: Missing patch data for rank={rank_idx}, idx={idx}, pos={z,y,x}")
                        continue  # Skip if patch data not found
                    
                    # Debug the first few patches to see what's being processed
                    if patch_counter <= 5:
                        print(f"DEBUG: Processing patch rank={rank_idx}, idx={idx}, pos={z,y,x}")
                        
                    # 1. Calculate patch bounds in global coordinates (clamped to volume size)
                    patch_z_end = min(z + patch_size_tuple[0], max_z)
                    patch_y_end = min(y + patch_size_tuple[1], max_y)
                    patch_x_end = min(x + patch_size_tuple[2], max_x)

                    # 2. Calculate intersection with current chunk
                    global_target_z_start = max(z_start, z)
                    global_target_z_end = min(z_end, patch_z_end)

                    # Skip if no actual intersection (shouldn't happen due to earlier check)
                    if global_target_z_end <= global_target_z_start:
                        continue

                    # 3. Calculate chunk-relative coordinates for the intersection
                    target_z_start = global_target_z_start - z_start
                    target_z_end = global_target_z_end - z_start

                    # Y and X coordinates (these aren't chunked, so use full patch extent)
                    target_y_start = y
                    target_y_end = patch_y_end

                    target_x_start = x
                    target_x_end = patch_x_end

                    # 4. Calculate patch-relative coordinates for the intersection
                    patch_z_start_rel = global_target_z_start - z
                    patch_z_end_rel = global_target_z_end - z

                    patch_y_start_rel = 0
                    patch_y_end_rel = patch_y_end - y

                    patch_x_start_rel = 0
                    patch_x_end_rel = patch_x_end - x

                    # Skip if patch has invalid dimensions
                    if (patch_y_end_rel <= patch_y_start_rel or
                            patch_x_end_rel <= patch_x_start_rel or
                            patch_z_end_rel <= patch_z_start_rel):
                        continue

                    try:
                        # Check if patch exists in our lookup
                        if (rank_idx, idx) not in patch_data_lookup:
                            print(f"WARNING: Missing patch reference for rank={rank_idx}, idx={idx}, pos={z,y,x}")
                            continue
                            
                        # Instead of getting pre-loaded data, we now load it on demand
                        # This saves memory by only loading patches when they're needed
                        if rank_idx not in self.patch_arrays_by_rank:
                            print(f"WARNING: No patch array reference for rank={rank_idx}")
                            continue
                        
                        # Get the patch data directly from the zarr array (lazy loading)
                        patch_data = self.patch_arrays_by_rank[rank_idx][idx]
                        
                        # Convert patch to PyTorch tensor and blend
                        # Make contiguous for better GPU memory layout and performance
                        patch_tensor = torch.as_tensor(patch_data, device=device).contiguous()
                        
                        # Blend the patch
                        blend_patch_torch(
                            output_tensor, count_tensor,
                            patch_tensor, blend_weights,
                            target_z_start, target_z_end,
                            target_y_start, target_y_end,
                            target_x_start, target_x_end,
                            patch_z_start_rel, patch_z_end_rel,
                            patch_y_start_rel, patch_y_end_rel,
                            patch_x_start_rel, patch_x_end_rel
                        )
                        
                        # Help reduce memory pressure by explicitly removing any reference to the patch data
                        del patch_data
                    except Exception as e:
                        print(f"Error blending patch (rank={rank_idx}, idx={idx}): {str(e)}")
                        if patch_counter <= 5:  # Only show detailed errors for first few patches
                            import traceback
                            traceback.print_exc()

                # Copy blended data back to memory-mapped arrays - single CPU transfer
                # Use contiguous tensors to ensure efficient memory transfers
                output_arrays[tgt_name][:, z_start:z_end] = output_tensor.contiguous().cpu().numpy()
                count_arrays[tgt_name][z_start:z_end] = count_tensor.contiguous().cpu().numpy()

                # Clean up GPU memory
                del output_tensor
                del count_tensor
                # Only empty cache every few chunks to reduce overhead
                if chunk_idx % 4 == 0:
                    torch.cuda.empty_cache()
                
                # Report chunk processing time
                chunk_process_time = time.time() - chunk_process_start
                print(f"Completed z-chunk {chunk_idx+1}/{total_chunks} in {chunk_process_time:.2f} seconds")

            # Each patch should have contributed to all chunks it intersects with
            if self.verbose:
                print(f"Processed all chunk-patch intersections, total patches: {len(all_patch_info)}")
                print(f"Each patch was blended into every chunk it intersected with (standard nnU-Net approach)")

    def _finalize_arrays(self, output_arrays, count_arrays, final_arrays):
        """
        Divide the sum arrays by the count arrays to get the final result.
        Also handles thresholding and conversion to appropriate data type.

        If save_probability_maps is False, computes argmax over channels for multiclass segmentation.
        For binary segmentation with exactly 2 channels, extracts only channel 1 and applies sigmoid.

        This is done chunk by chunk to avoid loading entire large arrays into memory.
        Optimized with PyTorch for better performance on GPU.

        """
        if self.verbose:
            print("Finalizing arrays..")
            print("Information about arrays:")
            for tgt_name in output_arrays:
                print(
                    f"  - output_arrays[{tgt_name}] shape: {output_arrays[tgt_name].shape}, dtype: {output_arrays[tgt_name].dtype}")
                print(
                    f"  - count_arrays[{tgt_name}] shape: {count_arrays[tgt_name].shape}, dtype: {count_arrays[tgt_name].dtype}")
                print(
                    f"  - final_arrays[{tgt_name}] shape: {final_arrays[tgt_name].shape}, dtype: {final_arrays[tgt_name].dtype}")
                print(f"  - final_arrays[{tgt_name}] type: {type(final_arrays[tgt_name])}")
                print(f"  - save_probability_maps: {self.save_probability_maps}")

        # Get device from model
        device = self.device_str

        # Import the normalize_chunk_torch function
        if __name__ == "__main__" or not __package__:
            from nnunet_zarr_inference.blending_torch import normalize_chunk_torch
        else:
            from .blending_torch import normalize_chunk_torch

        for target in self.targets:
            tgt_name = target.get("name")
            if self.verbose:
                print(f"Processing {tgt_name}...")

            # Define threshold_val at the beginning for each target
            # This ensures it's available for all code paths
            threshold_val = None
            if self.threshold is not None:
                threshold_val = self.threshold / 100.0

            if self.verbose:
                print(f"Using threshold_val = {threshold_val} (from self.threshold = {self.threshold})")

            num_channels, z_max, y_max, x_max = output_arrays[tgt_name].shape

            # Identify segmentation types based on channel count
            is_multiclass = num_channels > 2
            is_binary = num_channels == 2

            # Find the target config from our list
            target_config = next((t for t in self.targets if t.get('name') == tgt_name), None)

            # Determine processing modes based on segmentation type and flags
            # Multiclass segmentation with argmax (when not saving probability maps)
            compute_argmax = not self.save_probability_maps and is_multiclass

            # Binary segmentation handling - always extract channel 1 and apply sigmoid
            extract_binary = is_binary

            if compute_argmax and self.verbose:
                print(f"Will compute argmax over {num_channels} channels for {tgt_name}")

            if extract_binary and self.verbose:
                if self.save_probability_maps:
                    print(
                        f"Binary segmentation detected: will use both channels with softmax and save probabilities")
                else:
                    print(f"Binary segmentation detected: will use both channels with softmax and threshold")

            # Use a chunk-based approach to avoid loading the entire volume in memory
            chunk_size = 256

            # Only use tqdm progress bar on rank 0
            z_range = range(0, z_max, chunk_size)
            if self.rank == 0:
                z_range = tqdm(z_range, desc=f"Finalizing {tgt_name}")

            for z_start in z_range:
                z_end = min(z_start + chunk_size, z_max)

                # Process each chunk separately to minimize memory usage
                # Get direct references to zarr array slices - avoid copying data
                output_chunk = output_arrays[tgt_name][:, z_start:z_end]
                count_chunk = count_arrays[tgt_name][z_start:z_end]

                # Convert to PyTorch tensors directly from the zarr array views
                # Use float16 for both tensors to reduce memory usage
                output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float16).contiguous()
                count_tensor = torch.as_tensor(count_chunk, device=device, dtype=torch.float16).contiguous()

                if self.verbose and z_start == 0:
                    print(f"Processing in chunks to minimize memory usage for {tgt_name}")

                # Determine processing approach based on configuration
                if compute_argmax:
                    # Multiclass case with argmax - compute argmax over channels
                    if self.verbose:
                        print(f"Computing argmax over {output_tensor.shape[0]} channels using torch.argmax")

                    # Ensure count_tensor is float32 for stable division
                    if count_tensor.dtype != torch.float32:
                        count_tensor = count_tensor.float()

                    # Create safe count tensor (avoid division by zero)
                    # Use a very small value (0.001) instead of 1.0 to maintain the dynamic range
                    safe_count_tensor = torch.clamp(count_tensor, min=1e-8)

                    if self.verbose:
                        print(f"Count tensor max value: {torch.max(count_tensor).item():.4f}")
                        print(f"Count tensor data type: {count_tensor.dtype}")

                    # Normalize directly in PyTorch using broadcasting
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Compute argmax along channel dimension (dim=0)
                    argmax_tensor = torch.argmax(normalized_tensor, dim=0)

                    # Reshape to add channel dimension and convert to uint8
                    # Create a new tensor with the correct shape and type
                    dest_tensor = torch.zeros((1,) + argmax_tensor.shape,
                                              dtype=torch.uint8,
                                              device=device)
                    dest_tensor[0] = argmax_tensor

                    if self.verbose:
                        print(f"About to write to zarr array - diagnostics:")
                        print(f"  - dest_tensor shape: {dest_tensor.shape}")
                        print(f"  - final_arrays[{tgt_name}] shape: {final_arrays[tgt_name].shape}")
                        print(f"  - dest_tensor has non-zero values: {torch.any(dest_tensor > 0).item()}, max={torch.max(dest_tensor).item()}")

                    # Move to CPU for transfer to zarr - use contiguous for more efficient transfer
                    dest_cpu = dest_tensor.contiguous().cpu().numpy()
                    
                    # Copy to the zarr array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                    if self.verbose:
                        print(f"Copied results to zarr: source max={torch.max(dest_tensor).item()}")

                elif extract_binary:
                    # Binary segmentation case - use both channels and apply softmax
                    if self.verbose:
                        print(f"Processing binary segmentation: using both channels and applying softmax")

                    # Debug the original output tensor
                    if self.verbose and self.rank == 0:
                        print(f"Binary output channels: {output_tensor.shape[0]}")
                        print(
                            f"Channel 0 (background) range: {torch.min(output_tensor[0]).item():.4f} to {torch.max(output_tensor[0]).item():.4f}")
                        print(
                            f"Channel 1 (foreground) range: {torch.min(output_tensor[1]).item():.4f} to {torch.max(output_tensor[1]).item():.4f}")

                    # Ensure count_tensor is float32 for stable division
                    if count_tensor.dtype != torch.float32:
                        count_tensor = count_tensor.float()

                    # Create safe count (avoid division by zero) using torch
                    # Use a very small value (0.001) instead of 1.0 to maintain the dynamic range
                    safe_count_tensor = torch.clamp(count_tensor, min=1e-8)

                    if self.verbose:
                        print(f"Count tensor max value: {torch.max(count_tensor).item():.4f}")
                        print(f"Count tensor data type: {count_tensor.dtype}")

                    # Normalize logits by count array (both channels)
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Debug normalized tensor
                    if self.verbose and self.rank == 0:
                        print(
                            f"Normalized tensor range: {torch.min(normalized_tensor).item():.4f} to {torch.max(normalized_tensor).item():.4f}")

                    # Apply softmax across channel dimension (0) to get proper probabilities for both classes
                    softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)

                    # Extract foreground probability (class 1)
                    foreground_prob = softmax_tensor[1]

                    # Debug softmax output
                    if self.verbose and self.rank == 0:
                        # Check raw normalized tensor values before softmax
                        print(f"Normalized tensor stats before softmax:")
                        print(
                            f"  - Background channel: min={torch.min(normalized_tensor[0]).item():.4f}, max={torch.max(normalized_tensor[0]).item():.4f}, mean={torch.mean(normalized_tensor[0]).item():.4f}")
                        print(
                            f"  - Foreground channel: min={torch.min(normalized_tensor[1]).item():.4f}, max={torch.max(normalized_tensor[1]).item():.4f}, mean={torch.mean(normalized_tensor[1]).item():.4f}")

                        # Show the channel difference - positive means foreground is predicted
                        diff = normalized_tensor[1] - normalized_tensor[0]
                        print(
                            f"  - Channel difference (fg-bg): min={torch.min(diff).item():.4f}, max={torch.max(diff).item():.4f}, mean={torch.mean(diff).item():.4f}")

                        # See how many pixels are near the decision boundary
                        boundary_pixels = ((diff > -0.1) & (diff < 0.1)).sum().item()
                        total_pixels = diff.numel()
                        print(
                            f"  - Pixels near decision boundary (±0.1): {boundary_pixels} ({boundary_pixels / total_pixels * 100:.2f}%)")

                        # Show the softmax results
                        print(f"Softmax results:")
                        print(
                            f"  - Background range: {torch.min(softmax_tensor[0]).item():.4f} to {torch.max(softmax_tensor[0]).item():.4f}")
                        print(
                            f"  - Foreground range: {torch.min(softmax_tensor[1]).item():.4f} to {torch.max(softmax_tensor[1]).item():.4f}")

                        # Show histogram of foreground probabilities
                        hist = torch.histc(foreground_prob, bins=10, min=0.0, max=1.0)
                        bin_edges = torch.linspace(0, 1, 11)
                        print(f"Foreground probability histogram:")
                        for i in range(10):
                            print(f"  - {bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i].item():.0f} pixels")

                    if self.save_probability_maps:
                        # When save_probability_maps is True, save foreground probability and binary mask
                        # Use argmax for binary mask for consistent boundary handling

                        # Save just the foreground probability channel (index 1)
                        prob_tensor = (softmax_tensor[1] * 255).to(torch.uint8)

                        # Generate binary mask with argmax for maximum consistency
                        # This is how nnUNet creates binary segmentations - argmax across channels
                        binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                        binary_tensor = binary_mask * 255  # Scale to 0/255 for visualization

                        # Create 2-channel output: foreground prob, binary mask
                        dest_tensor = torch.zeros((2,) + binary_tensor.shape,
                                                  dtype=torch.uint8,
                                                  device=device)
                        dest_tensor[0] = prob_tensor  # Channel 0: foreground probability
                        dest_tensor[1] = binary_tensor  # Channel 1: binary mask (0/255)

                        if self.verbose and self.rank == 0:
                            print(f"Saving 2-channel binary segmentation output:")
                            print(
                                f"  - Channel 0: Foreground probabilities (0-255), range: {torch.min(prob_tensor).item()}-{torch.max(prob_tensor).item()}")
                            print(
                                f"  - Channel 1: Binary mask (0/255), using argmax, unique values: {torch.unique(binary_tensor).cpu().numpy()}")
                            print(f"  - Final tensor shape: {dest_tensor.shape}")
                    else:
                        # When save_probability_maps is False: save only binary mask
                        # Use argmax for consistency with nnUNet approach
                        binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                        binary_tensor = binary_mask * 255  # Scale to 0/255
                        dest_tensor = binary_tensor.unsqueeze(0)  # Add channel dimension for single-channel output

                        if self.verbose:
                            print(f"Saving only binary mask from argmax (single channel)")

                    # Debug the data (on GPU) being written to zarr
                    if self.verbose and self.rank == 0:
                        print(f"Writing to zarr array: shape={dest_tensor.shape}, dtype={dest_tensor.dtype}")
                        if dest_tensor.shape[0] == 2:  # Binary segmentation with probabilities
                            print(f"  - Channel 0 (probabilities) range: {torch.min(dest_tensor[0]).item()}-{torch.max(dest_tensor[0]).item()}")
                            print(f"  - Channel 1 (binary mask) range: {torch.min(dest_tensor[1]).item()}-{torch.max(dest_tensor[1]).item()}")
                            unique_values = torch.unique(dest_tensor[1])
                            print(f"  - Unique values in binary mask: {unique_values.cpu().numpy()}")
                    
                    # Move to CPU for transfer to zarr - use contiguous for more efficient transfer
                    dest_cpu = dest_tensor.contiguous().cpu().numpy()

                    # Write the result to the final array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                else:
                    # Standard case for multiclass segmentation (saving full probability maps)
                    # Use channel count from the output tensor
                    c = output_tensor.shape[0]

                    # Debugging info
                    if self.verbose:
                        output_max = torch.max(output_tensor).item()
                        count_max = torch.max(count_tensor).item()
                        count_min = torch.min(count_tensor).item()
                        print(
                            f"Before normalization - output max: {output_max:.4f}, count range: {count_min:.4f}-{count_max:.4f}")
                        print(f"Processing segmentation with {c} channels")

                    # Ensure count_tensor is float32 for stable division
                    if count_tensor.dtype != torch.float32:
                        count_tensor = count_tensor.float()

                    # Create safe count tensor (avoid division by zero)
                    safe_count_tensor = torch.clamp(count_tensor, min=1e-8)

                    if self.verbose:
                        print(f"Count tensor max value: {torch.max(count_tensor).item():.4f}")
                        print(f"Count tensor data type: {count_tensor.dtype}")

                    # Normalize directly in PyTorch using broadcasting
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Apply softmax to get proper probabilities
                    softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)

                    # Convert to uint8 (0-255 range)
                    dest_tensor = (softmax_tensor * 255).to(torch.uint8)

                    # Apply threshold if provided
                    if threshold_val is not None:
                        # Convert threshold to uint8 scale
                        threshold_uint8 = int(threshold_val * 255)
                        dest_tensor = (dest_tensor >= threshold_uint8).to(torch.uint8) * 255

                    # Move to CPU for transfer to zarr - use contiguous for more efficient transfer
                    dest_cpu = dest_tensor.contiguous().cpu().numpy()

                    # Copy the results back to the zarr array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                # Clean up temporary tensors
                del normalized_tensor
                # Less frequent cleanup - only every 4th chunk
                if (z_start // chunk_size) % 4 == 0:
                    # Cleanup GPU memory
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # Final cleanup
            torch.cuda.empty_cache()

    def _process_model_outputs(self, outputs, positions):
        """
        Process model outputs and submit them to the writer queue.
        
        This method converts model outputs to the appropriate format and
        queues them for asynchronous storage in zarr via ZarrTempStorage.
        """
        # For nnUNet, the output is a tensor, not a dict - convert it
        if torch.is_tensor(outputs):
            # Convert single tensor output to dict based on targets
            processed_outputs = {}
            for target in self.targets:
                tgt_name = target.get("name")
                processed_outputs[tgt_name] = outputs

            if self.verbose:
                print(f"Converted tensor output to dict with keys: {list(processed_outputs.keys())}")
                print(f"Tensor shape: {outputs.shape}")
        else:
            # Already in dict format
            processed_outputs = outputs
            if self.verbose:
                print(f"Output already in dict format with keys: {list(processed_outputs.keys())}")

        # Get the expected patch shape
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # Debug outputs and positions
        if self.verbose:
            print(f"Number of positions: {len(positions)}")
            first_target = self.targets[0].get("name")
            if first_target in processed_outputs:
                print(f"Processed outputs for {first_target} shape: {processed_outputs[first_target].shape}")
            
        # Ensure positions is a list-like object
        positions_list = positions
        if isinstance(positions, torch.Tensor):
            # If positions is a tensor with batch dimension, convert to list
            if positions.dim() > 1 and positions.shape[0] > 1:
                positions_list = [positions[i] for i in range(positions.shape[0])]
            
        # Process each patch and add to the queue
        for i in range(processed_outputs[self.targets[0].get("name")].shape[0]):
            # Ensure we don't go beyond available positions
            if i >= len(positions_list):
                print(f"Warning: More outputs ({processed_outputs[self.targets[0].get('name')].shape[0]}) than positions ({len(positions_list)})")
                continue
                
            pos = positions_list[i]
            
            for target in self.targets:
                tgt_name = target.get("name")
                # Get the prediction tensor for this target
                try:
                    # Get model output as tensor first (keeping on CPU)
                    pred_tensor = processed_outputs[tgt_name][i].cpu()

                    # Store raw logits directly, without applying activation
                    # We'll apply softmax only once, after blending all patches
                    if tgt_name == "segmentation":
                        # Get number of channels
                        num_channels = pred_tensor.shape[0]

                        # Convert to numpy for zarr storage
                        pred_array = pred_tensor.numpy()

                        if self.verbose and i < 3:
                            print(f"Storing raw logits for {tgt_name} with {num_channels} channels")
                            print(f"  - Value range: {pred_tensor.min().item():.4f}-{pred_tensor.max().item():.4f}")
                            print(f"  - Dtype: {pred_tensor.dtype}")
                            print(f"  - Position: {pos}")
                    else:
                        # For non-segmentation targets, scale to uint8 range (0-255)
                        pred_array = (pred_tensor * 255).to(torch.uint8).numpy()

                    # Print shape information for the first few patches
                    if self.verbose and i < 3:
                        print(
                            f"Patch for {tgt_name} - output shape: {pred_array.shape}, expected shape: (C, {patch_size_tuple[0]}, {patch_size_tuple[1]}, {patch_size_tuple[2]})")
                        print(f"  - Value range: {pred_array.min()}-{pred_array.max()}, dtype: {pred_array.dtype}")

                    # Add patch to the writer queue
                    # The zarr_writer_worker will pick this up and store it in ZarrTempStorage
                    self.writer_queue.put((pred_array, pos, tgt_name))
                    
                except Exception as e:
                    print(f"Error processing output for target {tgt_name}, position {pos}: {str(e)}")
                    print(f"Output keys: {list(processed_outputs.keys())}")
                    if tgt_name in processed_outputs:
                        print(f"Output shape for {tgt_name}: {processed_outputs[tgt_name].shape}")
                        if i < processed_outputs[tgt_name].shape[0]:
                            print(f"Output shape at index {i}: {processed_outputs[tgt_name][i].shape}")
                    else:
                        print(f"Target {tgt_name} not found in outputs")
                    raise

    def infer(self):
        """
        Main inference method using zarr for all storage needs.
        
        This method handles:
        1. Loading the nnUNet model
        2. Setting up zarr arrays for output
        3. Processing input data in patches
        4. Blending patches together for final output
        5. Handling distributed processing across multiple GPUs
        """
        # Verify input path exists and is a valid zarr array
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        try:
            # Try to open the input zarr array to check if it's valid
            zarr.open(self.input_path, mode='r')
        except Exception as e:
            raise ValueError(f"Error opening input zarr array at {self.input_path}: {str(e)}")

        # Ensure output path ends with .zarr
        if not self.output_path.endswith('.zarr'):
            raise ValueError(f"Output path must end with '.zarr', got: {self.output_path}")

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Get world size for distributed processing
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Create a directory for temporary storage next to the output zarr file
        self.temp_dir = os.path.join(os.path.dirname(self.output_path), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize ZarrTempStorage for temporary patch storage
        # Use num_write_workers as the number of parallel I/O workers
        # This ensures we actually use the requested number of writer processes
        self.temp_storage = ZarrTempStorage(
            output_path=self.temp_dir,
            rank=self.rank,
            world_size=world_size,
            verbose=self.verbose,
            num_io_workers=self.num_write_workers
        )
        self.temp_storage.initialize()

        # Load the nnUNet model
        self.model_info = self._load_nnunet_model()
        network = self.model_info['network']
        
        try:
            # -------------------------------------------------------------
            # 1. Create output zarr arrays (only rank 0)
            # -------------------------------------------------------------
            final_arrays = {}  # Dictionary to store final output arrays
            
            if self.rank == 0:
                if os.path.isdir(self.output_path):
                    raise FileExistsError(f"Zarr store '{self.output_path}' already exists.")
                
                # Create a temporary dataset to determine the full output shape
                dataset_temp = InferenceDataset(
                    input_path=self.input_path,
                    targets=self.targets,
                    model_info=self.model_info,
                    patch_size=self.patch_size,
                    input_format=self.input_format,
                    step_size=self.tile_step_size,
                    load_all=self.load_all,
                    verbose=self.verbose,
                    cache_size=self.cache_size,
                    max_cache_bytes=self.max_cache_bytes
                )
                
                # Get full input shape using the proper method
                input_shape = dataset_temp.get_input_shape()
                if self.verbose:
                    print(f"Input shape: {input_shape}")
                
                # Create output zarr store - don't create the actual store yet
                if self.verbose:
                    print(f"Will create output zarr array at {self.output_path}")
                
                # Create output arrays for each target
                output_arrays = {}  # For accumulating outputs
                count_arrays = {}   # For counting overlapping patches
                
                for target in self.targets:
                    tgt_name = target.get("name")
                    tgt_type = target.get("type", "segmentation")
                    
                    # Create output arrays
                    if self.verbose:
                        print(f"Setting up output arrays for target: {tgt_name}")
                    
                    # Get the dimensions for the output array
                    z_max, y_max, x_max = input_shape
                    
                    # Determine number of channels based on target type
                    if tgt_type == "segmentation":
                        # For nnUNet, get number of segmentation heads
                        num_classes = self.model_info.get('num_seg_heads', 1)
                        out_shape = (num_classes, z_max, y_max, x_max)
                        
                        # Output format based on configuration
                        final_dtype = 'uint8'
                        final_shape = out_shape
                        
                        if not self.save_probability_maps and num_classes > 2:
                            # Argmax output for multiclass (single channel)
                            final_shape = (1, z_max, y_max, x_max)
                        elif not self.save_probability_maps and num_classes == 2:
                            # Binary output for binary segmentation
                            final_shape = (1, z_max, y_max, x_max)
                    else:
                        # For regression/other targets - use float32
                        out_shape = (1, z_max, y_max, x_max)
                        final_dtype = 'float32'
                        final_shape = out_shape
                    
                    # Create zarr arrays for sum and count
                    if self.verbose:
                        print(f"Creating zarr sum array with shape {out_shape}")
                        print(f"Creating zarr count array with shape {(z_max, y_max, x_max)}")
                    
                    # Create compressor for better performance
                    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
                    
                    # Ensure patch_size_tuple is defined and has correct dimensions
                    patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size
                    
                    # Create sum array in the temp zarr store with chunk size matching patch size
                    sum_array = self.temp_storage.temp_zarr.create_dataset(
                        f"sum_{tgt_name}", 
                        shape=out_shape,
                        chunks=(1,) + patch_size_tuple,  # 1 for channel dimension, then patch dimensions
                        dtype='float16',  # Changed from 'float32' to reduce memory usage
                        compressor=compressor,
                        fill_value=0,
                        write_empty_chunks=False  # Skip writing empty chunks to save space
                    )
                    
                    # Create count array in the temp zarr store with chunk size matching patch size
                    count_array = self.temp_storage.temp_zarr.create_dataset(
                        f"count_{tgt_name}",
                        shape=(z_max, y_max, x_max),
                        chunks=patch_size_tuple,  # Direct patch dimensions
                        dtype='uint8',  # Changed from 'float32' to reduce memory usage (handles up to 255 overlapping patches)
                        compressor=compressor,
                        fill_value=0,
                        write_empty_chunks=False  # Skip writing empty chunks to save space
                    )
                    
                    # Store in our dictionaries for further use
                    output_arrays[tgt_name] = sum_array
                    count_arrays[tgt_name] = count_array
                    
                    # Create final output array with chunking matched to patch size
                    # Always include a channel dimension in the chunks
                    chunks = (1,) + patch_size_tuple
                    
                    # For segmentation target, create it directly at the root
                    if tgt_name == "segmentation":
                        # For root-level array, use zarr.open() directly with shape and chunks
                        final_arrays[tgt_name] = zarr.open(
                            self.output_path,
                            mode='w',
                            shape=final_shape,
                            chunks=chunks,
                            dtype=final_dtype,
                            compressor=compressor,
                            write_empty_chunks=False  # Skip writing empty chunks to save space
                        )
                        # Also store a reference as output_array for later use
                        output_store = final_arrays[tgt_name]
                    else:
                        # For any other targets, create within an existing group
                        # First ensure the output_store is open
                        if "output_store" not in locals():
                            output_store = zarr.open(self.output_path, mode='a')
                        
                        # Then create the dataset within that store
                        final_arrays[tgt_name] = output_store.create_dataset(
                            tgt_name,
                            shape=final_shape,
                            chunks=chunks,
                            dtype=final_dtype,
                            compressor=compressor,
                            write_empty_chunks=False  # Skip writing empty chunks to save space
                        )
                    
                    if self.verbose:
                        print(f"Created output array for {tgt_name}")
                        print(f"  Shape: {final_shape}, dtype: {final_dtype}")
                        print(f"  Chunking: {chunks}")
            
            # Wait for rank 0 to create output arrays
            if dist.is_initialized():
                dist.barrier()
            
            # -------------------------------------------------------------
            # 2. Set up worker threads for zarr patch writing
            # -------------------------------------------------------------
            # Create writer threads using zarr_writer_worker
            writer_threads = []
            for worker_id in range(self.num_write_workers):
                thread = threading.Thread(
                    target=zarr_writer_worker,
                    args=(self.temp_storage, self.writer_queue, worker_id, self.verbose)
                )
                thread.daemon = True
                thread.start()
                writer_threads.append(thread)
            
            # -------------------------------------------------------------
            # 3. Create dataset and dataloader for this rank
            # -------------------------------------------------------------
            # Create inference dataset
            dataset = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose,
                cache_size=self.cache_size,
                max_cache_bytes=self.max_cache_bytes
            )
            
            # Get the number of patches and update zarr temp storage size
            total_patches = len(dataset)
            if self.verbose:
                print(f"Rank {self.rank}: Created dataset with {total_patches} patches")
                
            # Get volume dimensions for spatial hashing
            volume_shape = dataset.get_input_shape()
            max_z, max_y, max_x = volume_shape
            
            # With our simplified implementation, we just need to know how many patches each rank will process
            # Divide patches evenly among ranks to avoid complex spatial hashing
            patches_per_rank = total_patches // world_size
            remainder = total_patches % world_size
            
            # Calculate how many patches this rank will process
            if self.rank < remainder:
                # First 'remainder' ranks get one extra patch
                my_patches = patches_per_rank + 1
            else:
                my_patches = patches_per_rank
                
            # Add a small safety margin
            safety_factor = 1.1  # 10% extra space should be enough
            expected_patches = int(my_patches * safety_factor) + 1  # +1 for safety margin
            
            # Update temp storage size for each target
            for target in self.targets:
                self.temp_storage.set_expected_patch_count(target.get("name"), expected_patches)
                if self.verbose:
                    print(f"Rank {self.rank}: Setting expected patch count for {target.get('name')} to {expected_patches}")
                    print(f"Rank {self.rank}: Out of {total_patches} total patches")
                    print(f"Rank {self.rank}: Volume dimensions: {max_z}x{max_y}x{max_x}")
            
            # Split the dataset for distributed processing
            if dist.is_initialized():
                dataset.set_distributed(self.rank, world_size)
            
            # Define a custom collate function to ensure positions stay as tuples
            def custom_collate(batch):
                data = torch.stack([item['data'] for item in batch])
                positions = [item['pos'] for item in batch]  # Keep positions as a list of tuples
                indices = [item['index'] for item in batch]
                return {'data': data, 'pos': positions, 'index': indices}
                
            # Create the dataloader with specified batch size and custom collate function
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_dataloader_workers,
                pin_memory=True,
                collate_fn=custom_collate  # Use our custom collate function
            )
            
            # Log dataset and dataloader information
            if self.verbose:
                print(f"Rank {self.rank}: Dataset has {len(dataset)} patches")
                print(f"Rank {self.rank}: Using batch size {self.batch_size}")
                print(f"Rank {self.rank}: Using {self.num_dataloader_workers} dataloader workers")
            
            # -------------------------------------------------------------
            # 4. Process patches with nnUNet model
            # -------------------------------------------------------------
            # Process patches
            start_time = time.time()
            total_patches = len(dataset)
            processed_patches = 0
            
            # Enable evaluation mode
            network.eval()
            
            # Move network to device
            network = network.to(self.device_str)
            
            # Create progress bar (only for rank 0)
            progress_bar = None
            if self.rank == 0:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_patches, desc="Processing patches")
            
            # Inference loop with no_grad and autocast for mixed precision
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for batch_idx, batch in enumerate(dataloader):
                    # Extract input data and positions
                    positions = batch['pos']
                    
                    # Debug position information in much more detail
                    if self.verbose and batch_idx < 2:
                        print(f"Batch #{batch_idx} positions type: {type(positions)}")
                        if hasattr(positions, '__len__'):
                            print(f"Positions length: {len(positions)}")
                            if len(positions) > 0:
                                print(f"First position: {positions[0]}, type: {type(positions[0])}")
                                if hasattr(positions[0], 'shape'):
                                    print(f"First position shape: {positions[0].shape}")
                                    
                        # Print each individual position in the batch
                        if isinstance(positions, list) and len(positions) < 10:
                            print("All positions in batch:", positions)
                        elif hasattr(positions, 'shape') and positions.shape[0] < 10:
                            print("All positions in batch (tensor):")
                            for i in range(positions.shape[0]):
                                print(f"  Position {i}: {positions[i]}")
                                
                        # Check if shape attributes exist
                        if hasattr(positions, 'shape'):
                            print(f"Positions tensor shape: {positions.shape}")
                        if hasattr(batch['data'], 'shape'):
                            print(f"Data batch shape: {batch['data'].shape}")
                    
                    # Get input data tensor and move to device - make contiguous for better performance
                    inputs = batch['data'].to(self.device_str, non_blocking=True).contiguous()
                    
                    # Run inference using the run_inference function that properly handles TTA
                    # Always use 3 for max_tta_combinations to get the 3 single-axis flips
                    outputs = run_inference(
                        model_info=self.model_info,
                        input_tensor=inputs,
                        max_tta_combinations=self.max_tta_combinations,  # Use only the 3 single-axis flips
                        rank=self.rank
                    )
                    
                    # Process model outputs (queue them for writing)
                    self._process_model_outputs(outputs, positions)
                    
                    # Update progress
                    processed_patches += len(positions)
                    if progress_bar is not None:
                        progress_bar.update(len(positions))
                    
                    # Log progress periodically
                    if self.verbose and (batch_idx + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        patches_per_sec = processed_patches / elapsed if elapsed > 0 else 0
                        print(f"Rank {self.rank}: Processed {processed_patches}/{total_patches} patches "
                              f"({patches_per_sec:.2f} patches/sec)")
            
            # Close progress bar if it was created
            if progress_bar is not None:
                progress_bar.close()
            
            # -------------------------------------------------------------
            # 5. Finalize writer threads
            # -------------------------------------------------------------
            # Wait for all writer tasks to complete
            if self.verbose:
                print(f"Rank {self.rank}: Waiting for writer tasks to complete...")
            
            self.writer_queue.join()
            
            # Send sentinel values to stop writer threads
            for _ in range(self.num_write_workers):
                self.writer_queue.put(None)
            
            # Wait for all writer threads to finish
            for thread in writer_threads:
                thread.join()
            
            # Finalize target counts in zarr storage
            for target in self.targets:
                tgt_name = target.get("name")
                self.temp_storage.finalize_target(tgt_name)
            
            # Make sure all ranks have finalized their targets before proceeding
            if dist.is_initialized():
                if self.verbose:
                    print(f"Rank {self.rank}: Waiting at barrier for all ranks to finalize targets")
                dist.barrier()
                if self.verbose:
                    print(f"Rank {self.rank}: All ranks have finalized targets, proceeding")
            
            # -------------------------------------------------------------
            # 6. Blend patches and finalize output
            # -------------------------------------------------------------
            # Only rank 0 needs to blend patches from all ranks
            if self.rank == 0:
                # Wait a moment to ensure all ranks have finalized their data
                time.sleep(1)
                
                # For segmentation array (our root), open it directly
                for target in self.targets:
                    tgt_name = target.get("name")
                    
                    if tgt_name == "segmentation":
                        # For the segmentation target (root array), open it directly
                        final_arrays[tgt_name] = zarr.open(self.output_path, mode='a')
                    else:
                        # For other targets (if any), open the store first then access them
                        if "output_store" not in locals():
                            output_store = zarr.open(self.output_path, mode='a')
                        final_arrays[tgt_name] = output_store[tgt_name]
                
                # Blend all patches from all ranks
                start_time = time.time()
                
                # Dictionary to store target arrays
                target_arrays = {tgt.get("name"): tgt.get("type", "segmentation") for tgt in self.targets}
                
                # Blend patches using zarr-based approach
                print("Starting _blend_patches method...")
                try:
                    self._blend_patches(target_arrays, output_arrays, count_arrays)
                    print("_blend_patches completed successfully")
                except Exception as e:
                    print(f"Error in _blend_patches: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Finalize the arrays without passing tensor_dict (since we're not using it)
                self._finalize_arrays(output_arrays, count_arrays, final_arrays)
                
                # Log blending time
                blend_time = time.time() - start_time
                print(f"Blending completed in {blend_time:.2f} seconds")
            
            # -------------------------------------------------------------
            # 7. Clean up temporary files - only after rank 0 is done blending
            # -------------------------------------------------------------
            # Wait for rank 0 to complete blending before proceeding with cleanup
            if dist.is_initialized():
                if self.verbose and self.rank != 0:
                    print(f"Rank {self.rank}: Waiting for rank 0 to complete blending...")
                dist.barrier()
                if self.verbose and self.rank != 0:
                    print(f"Rank {self.rank}: Rank 0 has completed blending, proceeding with cleanup")
            
            # Only rank 0 should perform cleanup
            if self.rank == 0:
                print(f"Rank {self.rank}: Cleaning up temporary files...")
                
                # Clear references to zarr arrays from the dictionaries
                if 'output_arrays' in locals():
                    output_arrays.clear()
                if 'count_arrays' in locals():
                    count_arrays.clear()
                
                # Clean up ZarrTempStorage - only rank 0 does this
                if self.temp_storage is not None:
                    if self.verbose:
                        print(f"Rank {self.rank}: Cleaning up zarr temporary storage...")
                    self.temp_storage.cleanup()
                    
                # Only rank 0 cleans up the temp directory
                if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                    try:
                        # Try to remove the directory - will only succeed if empty
                        os.rmdir(self.temp_dir)
                        if self.verbose:
                            print(f"Rank {self.rank}: Removed temporary directory {self.temp_dir}")
                    except OSError as e:
                        # Directory not empty - this is expected if other ranks are still using it
                        if self.verbose:
                            print(f"Note: Could not remove temp directory (may not be empty): {e}")
            else:
                # Non-rank-0 processes just clear their references but don't delete anything
                if self.verbose:
                    print(f"Rank {self.rank}: Clearing references but skipping cleanup (only rank 0 performs cleanup)")
                
                # Just clear references to help with memory usage
                if 'output_arrays' in locals():
                    output_arrays.clear()
                if 'count_arrays' in locals():
                    count_arrays.clear()
                
                # Clear reference to temp_storage but don't perform cleanup
                self.temp_storage = None
            
            # Make sure all ranks finish reference clearing before proceeding
            if dist.is_initialized():
                dist.barrier()
            
            # -------------------------------------------------------------
            # 8. Report statistics and completion
            # -------------------------------------------------------------
            # Report cache statistics if available
            if hasattr(dataset, 'zarr_cache') and dataset.zarr_cache is not None:
                stats = dataset.zarr_cache.get_stats()
                print(f"Rank {self.rank}: Zarr cache statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            # Log completion
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Rank {self.rank}: Inference completed in {total_time:.2f} seconds")
        
        except Exception as e:
            # Only rank 0 does any cleanup
            if self.rank == 0:
                # Clean up temporary files in case of error
                if self.temp_storage is not None:
                    print(f"Rank {self.rank}: Cleaning up temp storage after error")
                    self.temp_storage.cleanup()
                
                # Also try to clean up the temp directory
                if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                    try:
                        # Try to remove the directory - will only succeed if empty
                        os.rmdir(self.temp_dir)
                    except OSError:
                        # Ignore errors - cleanup as best effort
                        pass
            
            # All ranks log the error
            print(f"Rank {self.rank}: Error in inference: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Re-raise the exception
            raise


# Main function for standalone execution
def main():
    import argparse
    import torch.distributed as dist

    # Initialize process group for DDP if running with torch.distributed
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        if local_rank == 0:
            print(f"Initializing distributed process group (local_rank={local_rank})")

        # Initialize process group without explicit device_id
        # torchrun already sets the necessary environment variables
        print(f"Rank {local_rank}: Initializing process group with NCCL backend")
        dist.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(description='Run sequential nnUNet inference on a zarr array')
    parser.add_argument('--input', type=str, required=True, help='Path to input zarr array')
    parser.add_argument('--output', type=str, required=True, help='Path to output zarr file (must end with .zarr)')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to nnUNet model folder')
    parser.add_argument('--fold', type=str, default='0', help='Model fold to use (default: 0)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint name (default: checkpoint_final.pth)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference (default: 4)')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size as fraction of patch size (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_write_workers', type=int, default=4, help='Number of writer threads (default: 4)')
    parser.add_argument('--max_tta_combinations', type=int, default=3, help='Number of TTA combinations (default: 3, or the primary axis flips only)")')
    parser.add_argument('--threshold', type=float, help='Optional threshold (0-100) for binarizing the output')
    parser.add_argument('--no_mirroring', action='store_true', help='Disable test time augmentation via mirroring')
    parser.add_argument('--no_probabilities', action='store_true',
                        help='Do not save probability maps, save argmax for multiclass segmentation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--edge_weight_boost', type=float, default=0,
                        help='Factor to boost weights at patch edges (0.0-1.0) to reduce boundary artifacts (default: 0)')
    parser.add_argument('--cache_size', type=int, default=256,
                        help='Number of zarr chunks to cache in memory (default: 256)')
    parser.add_argument('--max_cache_bytes', type=float, default=4.0,
                        help='Maximum memory in GB to use for zarr cache (default: 4.0)')

    args = parser.parse_args()
    
    # Ensure output path ends with .zarr
    if not args.output.endswith('.zarr'):
        raise ValueError(f"Output path must end with '.zarr', got: {args.output}")

    # Convert fold to int if numeric
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold  # Keep as string if not numeric (e.g., "all")

    # Set the device explicitly if using DDP
    if local_rank != -1:
        # When using torchrun, each process should use the GPU assigned by LOCAL_RANK
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # Use modulo in case there are more ranks than GPUs
            device = f"cuda:{local_rank % device_count}"
            print(f"Rank {local_rank}: Assigned to device {device} (out of {device_count} GPUs)")
        else:
            # Fallback to CPU if no CUDA devices
            device = "cpu"
            print(f"Rank {local_rank}: No CUDA devices available, using CPU")
    else:
        device = args.device

    # Adjust workers based on world size for DDP
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dataloader_workers = max(1, args.num_workers // world_size)
        write_workers = max(1, args.num_write_workers // world_size)
        if local_rank == 0:
            print(f"Adjusting workers for {world_size} processes:")
            print(f"  - Dataloader workers: {args.num_workers} -> {dataloader_workers} per process")
            print(f"  - Write workers: {args.num_write_workers} -> {write_workers} per process")
    else:
        dataloader_workers = args.num_workers
        write_workers = args.num_write_workers

    # Create inference handler
    inference = SequentialZarrNNUNetInference(
        input_path=args.input,
        output_path=args.output,
        model_folder=args.model_folder,
        fold=fold,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        step_size=args.step_size,
        num_dataloader_workers=dataloader_workers,
        num_write_workers=write_workers,
        device=device,
        threshold=args.threshold,
        use_mirroring=not args.no_mirroring,
        save_probability_maps=not args.no_probabilities,
        verbose=args.verbose and (local_rank == -1 or local_rank == 0),  # Only be verbose on rank 0
        edge_weight_boost=args.edge_weight_boost,  # Pass the edge weight boost parameter
        cache_size=args.cache_size,  # Pass the cache size parameter
        max_cache_bytes=args.max_cache_bytes  # Pass the max cache bytes parameter
    )

    # Run inference
    inference.infer()

    # Cleanup distributed process group
    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()