# Import os early
import os
import json

import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
import torch.nn
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import torch.distributed as dist
from torch.utils.data import DataLoader
import queue
import time
from data.vc_dataset import VCDataset
from utils.models.blending import (
    create_gaussian_weights_torch,
    blend_patch_torch,
    blend_patch_weighted,
    intersects_chunk
)
from data.io.zarrio.zarr_temp_storage import ZarrTempStorage
from data.io.zarrio.zarr_writer_worker import zarr_writer_worker
from utils.models.load_nnunet_model import load_model_for_inference
from data.vc_dataset import VCDataset
from utils.models.tta import get_tta_augmented_inputs
from utils.models.helpers import merge_tensors

class ZarrInferer:
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 model_info: Dict[str, Any],
                 dataset: VCDataset,
                 dataloader: DataLoader,
                 patch_size: Optional[Tuple[int, int, int]] = None,
                 batch_size: int = 4,
                 step_size: float = 0.5,
                 num_write_workers: int = 4,
                 threshold: Optional[float] = None,
                 use_mirroring: bool = False,
                 max_tta_combinations: Optional[int] = 3,
                 use_rotation_tta: bool = True,
                 rotation_weights: List[float] = None,
                 verbose: bool = False,
                 save_probability_maps: bool = True,
                 output_targets: Optional[List[Dict[str, Any]]] = None,
                 rank: int = 0,
                 edge_weight_boost: float = 0):
        """
        A sequential approach to nnUNet inference on zarr arrays.

        This implementation differs from the original ZarrNNUNetInferenceHandler by:
        1. Using a sequential writing approach to avoid chunk conflicts
        2. Writing each patch to a separate chunk in a temporary storage
        3. Only performing the Gaussian blending at the end during a final reduction pass

        Args:
            input_path: Path to the input zarr store
            output_path: Path to save the output zarr store
            model_info: Dictionary with model information (from load_model_for_inference)
            dataset: Dataset instance for getting patches
            dataloader: DataLoader instance for batch loading
            patch_size: Optional override for the patch size
            batch_size: Batch size for inference
            step_size: Step size for sliding window prediction as a fraction of patch_size (default: 0.5, nnUNet default)
            num_write_workers: Number of worker threads for asynchronous disk writes
            threshold: Optional threshold value (0-100) for binarizing the probability map
            use_mirroring: Enable test time augmentation via mirroring (default: True, matches nnUNet default)
            max_tta_combinations: Maximum number of TTA combinations to use (default: None = auto-detect based on GPU memory)
            verbose: Enable detailed output messages during inference (default: False)
            save_probability_maps: Save full probability maps for multiclass segmentation (default: True, set to False to save space)
            output_targets: Optional list of output target configurations, each a dictionary with 'name' and other parameters
            rank: Process rank for distributed processing (default: 0, only process 0 will print verbose outputs)
            edge_weight_boost: Factor to boost Gaussian weights at patch edges (default: 0.5). Higher values reduce artifacts
                               at volume boundaries but may affect blending quality elsewhere. Set to 0 for original behavior.
        """
        self.input_path = input_path
        
        # Ensure output path ends with .zarr
        if not output_path.endswith('.zarr'):
            raise ValueError(f"Output path must end with '.zarr', got: {output_path}")
        self.output_path = output_path
        
        # Store model info directly
        self.model_info = model_info
        
        # Store the dataset and dataloader
        self.dataset = dataset
        self.dataloader = dataloader
            
        self.batch_size = batch_size
        self.tile_step_size = step_size  # Using nnUNet's naming convention
        # Get device from model_info or default to 'cuda'
        self.device_str = str(next(self.model_info['network'].parameters()).device) if 'network' in self.model_info else 'cuda'
        self.threshold = threshold
        self.use_mirroring = use_mirroring
        self.max_tta_combinations = max_tta_combinations
        self.use_rotation_tta = use_rotation_tta
        self.rotation_weights = rotation_weights
        self.verbose = verbose
        self.save_probability_maps = save_probability_maps
        self.rank = rank  # Store rank for distributed training awareness
        self.edge_weight_boost = edge_weight_boost  # Store edge weight boost factor
        self.total_time = 0.0  # Initialize total_time attribute

        if max_tta_combinations == 0:
            self.use_mirroring = False

        # Convert patch_size to tuple if it's a list or another sequence
        # Use patch_size from model_info if not provided
        if patch_size is None and 'patch_size' in self.model_info:
            self.patch_size = tuple(self.model_info['patch_size'])
            if self.verbose:
                print(f"Using model's patch size: {self.patch_size}")
        elif patch_size is not None:
            self.patch_size = tuple(patch_size)
        else:
            raise ValueError("Patch size must be provided either directly or through model_info")

        # Get number of classes from model_info
        num_classes = self.model_info.get('num_seg_heads', 2)
        
        # Use the provided output_targets or default to a standard configuration
        # Always use the list format expected by InferenceDataset
        if output_targets is not None:
            self.targets = output_targets
        else:
            # Default configuration with actual channel count from model
            self.targets = [
                {
                    "name": "segmentation",  # Internal name for tracking
                    "channels": num_classes,  # Use number from model
                    "activation": "sigmoid" if num_classes <= 2 else "softmax",
                    "nnunet_output_channels": num_classes  # Use number from model
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

        # Number of writer threads
        self.num_write_workers = num_write_workers

        # For TTA, we'll use only the 3 primary axis for mirroring
        self.tta_directions = [(0,), (1,), (2,)]  # Only axis-aligned flips
        # Always use exactly 3 TTA combinations (one for each axis) for efficient inference
        self.max_tta_combinations = max_tta_combinations
        
        # Zarr temp storage - will be initialized in infer()
        self.temp_storage = None

# Model loading logic has been moved out of this class to load_model_for_inference function
# which should be called before creating the ZarrInferer instance


    def _precompute_chunk_weight_map(self, patch_positions, chunk_shape, z_start, z_end, 
                               blend_weights, device):
        """
        Pre-compute the weight map for a specific z-chunk based on all intersecting patches.
        
        Args:
            patch_positions: List of (rank_idx, idx, (z, y, x)) patch positions
            chunk_shape: Shape of the chunk (z_chunk, y_max, x_max)
            z_start, z_end: Z-range of the current chunk
            blend_weights: Gaussian weights tensor for a full patch
            device: Device to create the tensor on
            
        Returns:
            Tensor containing the accumulated weights for each voxel in the chunk
        """
        z_chunk, y_max, x_max = chunk_shape
        weight_map = torch.zeros((z_chunk, y_max, x_max), device=device, dtype=torch.float32)
        
        # Process each patch to see if it intersects with this chunk
        for rank_idx, idx, (z, y, x) in patch_positions:
            # Calculate patch bounds in volume space
            patch_z_end = min(z + self.patch_size[0], z_chunk + z_start)
            patch_y_end = min(y + self.patch_size[1], y_max)
            patch_x_end = min(x + self.patch_size[2], x_max)
            
            # Check if this patch intersects with the current chunk
            if not intersects_chunk(z, y, x, self.patch_size, z_start, z_end):
                continue
                
            # Calculate intersection in global coordinates
            global_target_z_start = max(z_start, z)
            global_target_z_end = min(z_end, patch_z_end)
            
            # Calculate chunk-relative coordinates
            target_z_start = global_target_z_start - z_start
            target_z_end = global_target_z_end - z_start
            
            # Y and X coordinates (not chunked)
            target_y_start = y
            target_y_end = patch_y_end
            
            target_x_start = x
            target_x_end = patch_x_end
            
            # Calculate patch-relative coordinates
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
                
            # Extract the weight slice for this intersection
            weight_slice = blend_weights[
                patch_z_start_rel:patch_z_end_rel,
                patch_y_start_rel:patch_y_end_rel,
                patch_x_start_rel:patch_x_end_rel
            ]
            
            # Add to weight map
            weight_map[
                target_z_start:target_z_end,
                target_y_start:target_y_end,
                target_x_start:target_x_end
            ] += weight_slice
            
        # Ensure no zeros in weight map (avoid division by zero)
        weight_map = torch.clamp(weight_map, min=1e-8)
        
        return weight_map
        
    def _blend_patches(self, patch_arrays, output_arrays, count_arrays):
        """
        Blend patches using a simplified z-chunking approach with pre-computed weight maps.
        The weight maps are a gaussian map that assigns a weight to each voxel in the chunk.
        Higher weights are given to voxels that are closer to the center of the patch.
        Each patch is processed exactly once in the first chunk it intersects with.
        
        Implementation uses a weight map approach:
        1. Pre-compute a weight map for each chunk based on all intersecting patches
        2. Add weighted patches to the output tensor
        3. Normalize the output tensor by dividing by the weight map
        4. Store the normalized result
        """
        if self.verbose:
            print("Starting patch blending phase...")

        # Ensure patch_size is a tuple
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # Create Gaussian blend weights - use device from model
        device = self.device_str

        # Adjusted nnUNet parameters for smoother blending
        sigma_scale = 1/8
        value_scaling_factor = 10  # Standard nnUNet value

        if self.verbose:
            print(f"Using sigma_scale={sigma_scale}, value_scaling_factor={value_scaling_factor}")

        # Use previously imported intersects_chunk

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
                # Use float32 for output tensor to avoid numerical issues
                output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float32).contiguous()
                
                # Find ALL patches that intersect with this chunk
                chunk_patches = []
                for patch_idx, patch_info in enumerate(all_patch_info):
                    rank_idx, idx, (z, y, x) = patch_info

                    # Use the simpler intersection check
                    if intersects_chunk(z, y, x, patch_size_tuple, z_start, z_end):
                        chunk_patches.append((rank_idx, idx, (z, y, x)))
                
                # Pre-compute weight map for this chunk
                chunk_shape = (z_end - z_start, max_y, max_x)
                weight_map = torch.zeros(chunk_shape, device=device, dtype=torch.float32)
                
                # Add weights from each patch to the weight map
                print(f"Pre-computing weight map for chunk [{z_start}:{z_end}]...")
                for weight_rank_idx, weight_idx, (weight_z, weight_y, weight_x) in chunk_patches:
                    # Calculate patch bounds in global coordinates (clamped to volume size)
                    weight_patch_z_end = min(weight_z + patch_size_tuple[0], max_z)
                    weight_patch_y_end = min(weight_y + patch_size_tuple[1], max_y)
                    weight_patch_x_end = min(weight_x + patch_size_tuple[2], max_x)

                    # Calculate intersection with current chunk
                    weight_global_target_z_start = max(z_start, weight_z)
                    weight_global_target_z_end = min(z_end, weight_patch_z_end)

                    # Skip if no actual intersection
                    if weight_global_target_z_end <= weight_global_target_z_start:
                        continue

                    # Calculate chunk-relative coordinates for the intersection
                    weight_target_z_start = weight_global_target_z_start - z_start
                    weight_target_z_end = weight_global_target_z_end - z_start

                    # Y and X coordinates
                    weight_target_y_start = weight_y
                    weight_target_y_end = weight_patch_y_end

                    weight_target_x_start = weight_x
                    weight_target_x_end = weight_patch_x_end

                    # Calculate patch-relative coordinates for the intersection
                    weight_patch_z_start_rel = weight_global_target_z_start - weight_z
                    weight_patch_z_end_rel = weight_global_target_z_end - weight_z

                    weight_patch_y_start_rel = 0
                    weight_patch_y_end_rel = weight_patch_y_end - weight_y

                    weight_patch_x_start_rel = 0
                    weight_patch_x_end_rel = weight_patch_x_end - weight_x

                    # Get the weight slice from the pre-computed Gaussian weights
                    weight_slice = blend_weights[
                        weight_patch_z_start_rel:weight_patch_z_end_rel,
                        weight_patch_y_start_rel:weight_patch_y_end_rel, 
                        weight_patch_x_start_rel:weight_patch_x_end_rel
                    ]
                    
                    # Add to the weight map
                    weight_map[
                        weight_target_z_start:weight_target_z_end,
                        weight_target_y_start:weight_target_y_end,
                        weight_target_x_start:weight_target_x_end
                    ] += weight_slice
                
                # Ensure no zeros in weight map for safe division later
                weight_map = torch.clamp(weight_map, min=1e-8)

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
                    if self.verbose and patch_counter <= 5:
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
                        
                        # Check if this patch has embedded position information
                        # Look for attributes in the zarr array
                        has_position_channels = False
                        try:
                            has_position_channels = self.patch_arrays_by_rank[rank_idx].attrs.get('has_position_channels', False)
                        except:
                            pass
                        
                        # Convert patch to PyTorch tensor and blend
                        # Make contiguous for better GPU memory layout and performance
                        if has_position_channels:
                            # Extract only the actual data channels (skip the first 3 position channels)
                            patch_tensor = torch.as_tensor(patch_data[3:], device=device).contiguous()
                        else:
                            # Use the full patch as-is
                            patch_tensor = torch.as_tensor(patch_data, device=device).contiguous()
                        
                        # Blend the patch using the weighted blending function
                        blend_patch_weighted(
                            output_tensor, 
                            patch_tensor,
                            blend_weights,
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

                # Normalize the output by dividing by the weight map
                print(f"Normalizing output for chunk [{z_start}:{z_end}]...")
                for c_idx in range(output_tensor.shape[0]):
                    output_tensor[c_idx] /= weight_map
                
                # Copy normalized data back to memory-mapped arrays - single CPU transfer
                output_arrays[tgt_name][:, z_start:z_end] = output_tensor.contiguous().cpu().numpy()
                
                # Clean up GPU memory
                del output_tensor
                del weight_map
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

        for target in self.targets:
            tgt_name = target.get("name")
            if self.verbose:
                print(f"Processing {tgt_name}...")

            # Define threshold_val at the beginning for each target
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

            # num z slices for blending , we load this many at once
            chunk_size = 256

            # Only use tqdm progress bar on rank 0
            z_range = range(0, z_max, chunk_size)
            if self.rank == 0:
                z_range = tqdm(z_range, desc=f"Finalizing {tgt_name}")

            for z_start in z_range:
                z_end = min(z_start + chunk_size, z_max)

                output_chunk = output_arrays[tgt_name][:, z_start:z_end]
                count_chunk = count_arrays[tgt_name][z_start:z_end]

                # Convert to PyTorch tensors directly from the zarr array views
                output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float16).contiguous()
                count_tensor = torch.as_tensor(count_chunk, device=device, dtype=torch.float16).contiguous()

                # Determine processing approach based on configuration
                if compute_argmax:
                    # Multiclass case with argmax - compute argmax over channels
                    if self.verbose:
                        print(f"Computing argmax over {output_tensor.shape[0]} channels using torch.argmax")

                    # Ensure count_tensor is float32 for stable division
                    if count_tensor.dtype != torch.float32:
                        count_tensor = count_tensor.float()

                        # With pre-computed weight maps, normalization is already done
                    # No need to divide by count tensor anymore
                    if self.verbose:
                        print(f"Using pre-normalized output (weight maps approach)")
                    
                    # Output is already normalized
                    normalized_tensor = output_tensor

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

                    # With pre-computed weight maps, normalization is already done
                    # No need to divide by count tensor anymore
                    if self.verbose:
                        print(f"Using pre-normalized output (weight maps approach)")
                    
                    # Output is already normalized
                    normalized_tensor = output_tensor

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
                        # Convert to float32 first as histc doesn't support float16
                        foreground_prob_float = foreground_prob.to(torch.float32)
                        hist = torch.histc(foreground_prob_float, bins=10, min=0.0, max=1.0)
                        bin_edges = torch.linspace(0, 1, 11)
                        print(f"Foreground probability histogram:")
                        for i in range(10):
                            print(f"  - {bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i].item():.0f} pixels")

                    if self.save_probability_maps:
                        # When save_probability_maps is True, handle probability map and binary mask separately
                        # 1. Create and handle probability map (foreground probability only)
                        # Get the raw foreground probabilities from the softmax
                        foreground_prob = softmax_tensor[1].clone()  # Clone to avoid potential issues
                        
                        # Debug the raw probability values
                        if self.verbose and self.rank == 0:
                            print(f"Raw foreground probability range: {foreground_prob.min().item():.4f}-{foreground_prob.max().item():.4f}")
                            
                        # Scale probabilities from [0,1] to [0,255] range for uint8 storage
                        # Make sure we're using float32 for the multiplication to avoid precision issues
                        prob_tensor = (foreground_prob.to(torch.float32) * 255).to(torch.uint8)
                        
                        # CRITICAL CHECK: Verify we're not just getting binary values in our probability map
                        if self.verbose and self.rank == 0:
                            unique_probs = torch.unique(prob_tensor)
                            print(f"Unique probability values: {unique_probs.cpu().numpy()}")
                            if len(unique_probs) <= 2:
                                print("WARNING: Probability map appears to be binarized! Should have many values.")
                            else:
                                print(f"Good: Found {len(unique_probs)} unique probability values")
                        
                        # Create a separate tensor for probabilities (single channel)
                        prob_dest_tensor = torch.zeros((1,) + prob_tensor.shape,
                                                     dtype=torch.uint8,
                                                     device=device)
                        prob_dest_tensor[0] = prob_tensor
                        
                        # 2. Create and handle binary mask separately
                        # Generate binary mask with argmax for maximum consistency
                        # This is how nnUNet creates binary segmentations - argmax across channels
                        binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                        
                        # The binary mask is 0 or 1, where 1 means foreground
                        # Scale to 0 or 255 for clearer visualization in 8-bit image viewers
                        binary_tensor = binary_mask * 255
                        
                        # Create a separate tensor for binary mask (single channel)
                        binary_dest_tensor = torch.zeros((1,) + binary_tensor.shape,
                                                      dtype=torch.uint8,
                                                      device=device)
                        binary_dest_tensor[0] = binary_tensor
                        
                        # 3. Combine them into the final destination tensor
                        # IMPORTANT: We need to make sure we're creating a clean, new tensor with the right shape
                        shape_tuple = binary_tensor.shape
                        
                        # Create a fresh tensor with exactly 2 channels
                        dest_tensor = torch.zeros((2,) + shape_tuple,
                                               dtype=torch.uint8,
                                               device=device)
                        
                        # Copy probability values carefully - this is the critical part
                        # Make sure the tensors are the right shape and values
                        if self.verbose and self.rank == 0:
                            print(f"Probability tensor shape: {prob_tensor.shape}")
                            print(f"Binary tensor shape: {binary_tensor.shape}")
                            
                        # Use direct assignment with explicit cloning to ensure complete separation
                        dest_tensor[0] = prob_tensor.clone()  # Channel 0: Probability map [0-255]
                        dest_tensor[1] = binary_tensor.clone()  # Channel 1: Binary mask [0/255]

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
                    safe_count_tensor = torch.clamp(count_tensor, min=1)

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
                try:
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


    def infer(self, skip_blending=False):
        """
        Main inference method
        1. Loading the model
        2. Setting up zarr arrays for output
        3. Processing input data in patches
        4. Blending patches together for final output (unless skip_blending=True)
        5. Handling distributed processing across multiple GPUs
        
        Args:
            skip_blending: If True, skip the blending phase and just save patch data.
                           Use blend_patches.py later to blend the saved patches.
        """
        # Check if we're using the VCDataset with Volume
        if hasattr(self.dataset, 'use_volume') and self.dataset.use_volume:
            # Skip the zarr validation for Volume input
            if self.verbose:
                print(f"Using Volume class for input, skipping zarr validation")
        else:
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
        # If we're using dataset partitioning, include part ID in the temp directory name
        try:
            # Check if dataset has part_id and num_parts attributes
            part_id = getattr(self.dataset, 'part_id', None)
            num_parts = getattr(self.dataset, 'num_parts', 1)
            
            # Print debug info
            if self.verbose:
                print(f"Temp directory decision:")
                print(f"  - Dataset has part_id attribute: {hasattr(self.dataset, 'part_id')}")
                print(f"  - Dataset has num_parts attribute: {hasattr(self.dataset, 'num_parts')}")
                print(f"  - part_id value: {part_id}")
                print(f"  - num_parts value: {num_parts}")
            
            # Only use part-specific directory if explicitly doing partitioning
            if hasattr(self.dataset, 'part_id') and hasattr(self.dataset, 'num_parts') and num_parts > 1:
                # When using partitioning, create a part-specific temp directory
                self.temp_dir = os.path.join(os.path.dirname(self.output_path), f"temp_part{part_id}")
                if self.verbose:
                    print(f"Using part-specific temp directory for part_id={part_id}: {self.temp_dir}")
            else:
                # Standard temp directory
                self.temp_dir = os.path.join(os.path.dirname(self.output_path), "temp")
                if self.verbose:
                    print(f"Using standard temp directory: {self.temp_dir}")
        except AttributeError as e:
            # If there's any issue accessing the attributes, use the standard directory
            self.temp_dir = os.path.join(os.path.dirname(self.output_path), "temp")
            if self.verbose:
                print(f"AttributeError when checking for part_id: {e}")
                print(f"Using standard temp directory: {self.temp_dir}")
        
        # Create the temp directory
        if self.verbose:
            print(f"Creating temp directory at: {self.temp_dir}")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Get the dataset size for proper array allocation
        total_patches = len(self.dataset)
        if self.verbose:
            print(f"Rank {self.rank}: Dataset has {total_patches} patches to process")
        
        # Add a safety margin for potential miscounting
        expected_patches = int(total_patches * 1.1) + 5  # Add 10% + 5 for safety
        
        # Initialize ZarrTempStorage for temporary patch storage
        # Use num_write_workers as the number of parallel I/O workers
        self.temp_storage = ZarrTempStorage(
            output_path=self.temp_dir,
            rank=self.rank,
            world_size=world_size,
            verbose=self.verbose,
            num_io_workers=self.num_write_workers
        )
        self.temp_storage.initialize(expected_patch_count=expected_patches)
        
        # Save blending configuration for later use by blend_patches.py
        # Only rank 0 needs to save this configuration
        if self.rank == 0:
            # Create a dictionary with all the parameters needed for blending
            blend_config = {
                "patch_size": list(self.patch_size),
                "step_size": self.tile_step_size,
                "threshold": self.threshold,
                "save_probability_maps": self.save_probability_maps,
                "edge_weight_boost": self.edge_weight_boost,
                "output_path": self.output_path
            }
            
            # Save the configuration as a JSON file in the temp directory
            config_path = os.path.join(self.temp_dir, "blend_config.json")
            try:
                with open(config_path, 'w') as f:
                    json.dump(blend_config, f, indent=2)
                if self.verbose:
                    print(f"Saved blending configuration to {config_path}")
            except Exception as e:
                print(f"Warning: Failed to save blending configuration: {e}")

        network = self.model_info['network']
        
        # If using distributed mode and network is not already wrapped with DDP, wrap it
        if dist.is_initialized() and not isinstance(network, torch.nn.parallel.DistributedDataParallel):
            from torch.nn.parallel import DistributedDataParallel as DDP
            # Wrap with DDP - use the device we're already on
            network = DDP(network)
            self.model_info['network'] = network
        
        try:
            # -------------------------------------------------------------
            # 1. Create output zarr arrays (only rank 0)
            # -------------------------------------------------------------
            final_arrays = {}  # Dictionary to store final output arrays
            
            if self.rank == 0:
                if os.path.isdir(self.output_path):
                    raise FileExistsError(f"Zarr store '{self.output_path}' already exists.")
                
                # Use the provided dataset to determine the full output shape
                dataset_temp = self.dataset
                
                # Get full input shape
                input_shape = dataset_temp.input_shape
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
                        # Use the value already in the target configuration, which was updated when loading the model
                        num_classes = target.get("channels", 1)
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

                    compressor = Blosc(cname='zstd', clevel=3)
                    
                    # Ensure patch_size_tuple is defined and has correct dimensions
                    patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size
                    
                    # Create sum array in the temp zarr store with chunk size matching patch size
                    sum_array = self.temp_storage.temp_zarr.create_dataset(
                        f"sum_{tgt_name}", 
                        shape=out_shape,
                        chunks=(1,) + patch_size_tuple,  # 1 for channel dimension, then patch dimensions
                        dtype='float32',  # Using float32 instead of float16 to avoid numerical issues
                        compressor=compressor,
                        fill_value=0,
                        write_empty_chunks=False
                    )
                    
                    # Create count array in the temp zarr store with chunk size matching patch size
                    count_array = self.temp_storage.temp_zarr.create_dataset(
                        f"count_{tgt_name}",
                        shape=(z_max, y_max, x_max),
                        chunks=patch_size_tuple,
                        dtype='uint8',
                        compressor=compressor,
                        fill_value=0,
                        write_empty_chunks=False
                    )
                    
                    # Store in our dictionaries for further use
                    output_arrays[tgt_name] = sum_array
                    count_arrays[tgt_name] = count_array
                    
                    # Create final output array with chunking matched to patch size
                    # Always include a channel dimension in the chunks because nnunet always outputs channels
                    # for single channel output models we can rip the channel off later
                    chunks = (1,) + patch_size_tuple
                    
                    # If skip_blending is true, we don't create the final output arrays
                    # These will be created later by the blend_patches script
                    if not skip_blending:
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
                                write_empty_chunks=False
                            )
    
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
                                write_empty_chunks=False
                            )
                    else:
                        # If skip_blending is true, we just create a placeholder in the final_arrays dict
                        # to avoid errors in the rest of the code, but don't actually create the zarr array
                        final_arrays[tgt_name] = None
                    
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
            # 3. Use the provided dataset and dataloader
            # -------------------------------------------------------------
            # Get the dataset and dataloader from initialization
            dataset = self.dataset
            dataloader = self.dataloader
            
            # Get the number of patches - already set in initialization but reconfirm
            total_patches = len(dataset)
            if self.verbose:
                print(f"Rank {self.rank}: Dataset has {total_patches} patches")

            volume_shape = dataset.input_shape
            max_z, max_y, max_x = volume_shape
            
            # Update temp storage size for each target
            for target in self.targets:
                self.temp_storage.set_expected_patch_count(target.get("name"), expected_patches)
                if self.verbose:
                    print(f"Rank {self.rank}: Setting expected patch count for {target.get('name')} to {expected_patches}")
                    print(f"Rank {self.rank}: Out of {total_patches} total patches")
                    print(f"Rank {self.rank}: Volume dimensions: {max_z}x{max_y}x{max_x}")
            
            # Log dataloader information
            if self.verbose:
                print(f"Rank {self.rank}: Using batch size {self.batch_size}")
            
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
                # Calculate total iterations based on batch size
                total_iterations = (total_patches + self.batch_size - 1) // self.batch_size
                progress_bar = tqdm(total=total_iterations, desc="Processing batches")
            
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
                    # Ensure data is float32 before sending to model
                    inputs = batch['data'].float().to(self.device_str, non_blocking=True).contiguous()

                    network = self.model_info['network']
                    
                    # Use TTA if enabled
                    if self.use_mirroring:
                        # Debug input shape when verbose is on
                        if self.verbose and self.rank == 0:
                            print(f"Processing input of shape {inputs.shape} with use_mirroring={self.use_mirroring}, use_rotation_tta={self.use_rotation_tta}")
                            
                        # Get all TTA-transformed inputs and their transformation info
                        # In our config: if rotation TTA is enabled, it takes precedence over mirroring
                        # If rotation TTA is disabled, use mirroring TTA
                        use_mirroring_tta = not self.use_rotation_tta

                        augmented_inputs, transform_info = get_tta_augmented_inputs(
                            input_tensor=inputs,
                            model_info=self.model_info,
                            max_tta_combinations=self.max_tta_combinations,
                            use_rotation_tta=self.use_rotation_tta,
                            use_mirroring=use_mirroring_tta,  # Only use mirroring if rotation is disabled
                            rotation_weights=self.rotation_weights,
                            verbose=self.verbose,
                            rank=self.rank
                        )
                        
                        # Run inference on each augmented input
                        tta_outputs = []
                        for idx, (aug_input, transform) in enumerate(zip(augmented_inputs, transform_info)):
                            # Run inference
                            with torch.no_grad(), torch.amp.autocast('cuda'):
                                output = network(aug_input)
                                tta_outputs.append((output, transform))
                        
                        # Combine the outputs from all TTA variants
                        outputs = merge_tensors(tta_outputs)
                    else:
                        # No TTA - just run standard inference
                        with torch.no_grad(), torch.amp.autocast('cuda'):
                            outputs = network(inputs)
                    
                    # Process model outputs (queue them for writing)
                    self._process_model_outputs(outputs, positions)
                    
                    # Update progress
                    processed_patches += len(positions)
                    if progress_bar is not None:
                        # Update by 1 batch, not by number of positions
                        progress_bar.update(1)
                    
                    # Log progress periodically
                    if self.verbose and (batch_idx + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        patches_per_sec = processed_patches / elapsed if elapsed > 0 else 0
                        print(f"Rank {self.rank}: Processed {processed_patches}/{total_patches} patches "
                              f"({patches_per_sec:.2f} patches/sec)")

            if progress_bar is not None:
                progress_bar.close()
            
            # -------------------------------------------------------------
            # 5. Finalize writer threads
            # -------------------------------------------------------------
            # Wait for all writer tasks to complete
            if self.verbose:
                print(f"Rank {self.rank}: Waiting for writer tasks to complete...")
            
            # Use a timeout for the join operation to avoid hanging indefinitely
            try:
                # Set a reasonable timeout for the join operation (60 seconds)
                start_time = time.time()
                
                # Create a timeout-based approach to avoid hanging
                while not self.writer_queue.empty():
                    # Check if we've waited too long
                    if time.time() - start_time > 60:
                        print(f"Rank {self.rank}: Warning - Queue join taking too long, proceeding with shutdown")
                        break
                    
                    # Small sleep to avoid high CPU usage while waiting
                    time.sleep(0.1)
                
                # After the queue is empty or timeout reached, try to join with a short timeout
                # This ensures all task_done() calls have been processed
                if not self.writer_queue.empty():
                    print(f"Rank {self.rank}: Queue not empty after timeout, proceeding with shutdown")
                else:
                    # Try a final join with a short timeout
                    if self.verbose:
                        print(f"Rank {self.rank}: Queue empty, doing final join check")
                    # Use unfinished_tasks check to determine if we need to wait more
                    if hasattr(self.writer_queue, 'unfinished_tasks') and self.writer_queue.unfinished_tasks > 0:
                        if self.verbose:
                            print(f"Rank {self.rank}: Waiting for {self.writer_queue.unfinished_tasks} unfinished tasks")
                        # Wait a bit more for task_done() calls
                        time.sleep(1)
            except Exception as e:
                print(f"Rank {self.rank}: Error waiting for writer queue to empty: {e}")
            
            # Send sentinel values to stop writer threads
            for _ in range(self.num_write_workers):
                try:
                    self.writer_queue.put(None, block=False)  # Non-blocking put
                except Exception as e:
                    print(f"Rank {self.rank}: Error sending shutdown signal to worker: {e}")
            
            # Wait for all writer threads to finish with timeouts
            for thread_idx, thread in enumerate(writer_threads):
                try:
                    # Use a reasonable timeout for thread joining (5 seconds)
                    thread.join(timeout=5)
                    
                    # Check if thread is still alive
                    if thread.is_alive():
                        print(f"Rank {self.rank}: Warning - Writer thread {thread_idx} did not terminate within timeout")
                        # There's no safe way to force-terminate a thread in Python,
                        # so we'll just have to proceed and leave this thread dangling
                except Exception as e:
                    print(f"Rank {self.rank}: Error joining writer thread {thread_idx}: {e}")
            
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
            # 6. Blend patches and finalize output (if not skipped)
            # -------------------------------------------------------------
            if skip_blending:
                if self.rank == 0:
                    print("Skipping blending phase as requested. To blend patches later, use one of:")
                    print(f"1. python -m models.run.blend_patches --temp_storage {self.temp_dir}/temp.zarr")
                    print(f"   (This will automatically read blend configuration from {self.temp_dir}/blend_config.json)")
                    print(f"2. python -m models.run.blend_patches --temp_storage {self.temp_dir}/temp.zarr --output {self.output_path} --patch_size {self.patch_size[0]} {self.patch_size[1]} {self.patch_size[2]} --step_size {self.tile_step_size}")
                    print(f"   (This uses command line arguments which override the saved configuration)")
                    print(f"Optional flags:")
                    print(f"  --force                Force overwrite of existing output without prompting")
                    print(f"  --no-cleanup           Keep temporary files after blending (default is to clean up)")
                    print(f"  --no-include-parts     Don't automatically include patches from all part directories")
                    print(f"  --verbose              Show detailed progress information")
                
                # We're done here if blending is skipped
                self.temp_dir_preserved = True  # Mark temp dir to not clean it up
            else:
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
            # 7. Clean up temporary files - only after rank 0 is done (unless preserving temp dir)
            # -------------------------------------------------------------
            # Wait for rank 0 to complete processing before proceeding with cleanup
            if dist.is_initialized():
                if self.verbose and self.rank != 0:
                    print(f"Rank {self.rank}: Waiting for rank 0 to complete processing...")
                dist.barrier()
                if self.verbose and self.rank != 0:
                    print(f"Rank {self.rank}: Rank 0 has completed processing, proceeding with cleanup")

            # Only rank 0 should perform cleanup
            if self.rank == 0:
                # Skip cleanup if we're preserving the temp directory for later blending
                if hasattr(self, 'temp_dir_preserved') and self.temp_dir_preserved:
                    print(f"Rank {self.rank}: Preserving temporary directory for later blending: {self.temp_dir}/temp.zarr")
                    
                    # Still clear references to arrays
                    if 'output_arrays' in locals():
                        output_arrays.clear()
                    if 'count_arrays' in locals():
                        count_arrays.clear()
                        
                    # We should NOT clean up the ZarrTempStorage, but we should close it
                    if self.temp_storage is not None:
                        # Just close the store without deleting anything
                        self.temp_storage.temp_zarr = None
                else:
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

            # Make sure all ranks finish reference clearing before proceeding
            if dist.is_initialized():
                dist.barrier()
            
            # -------------------------------------------------------------
            # 8. Report statistics and completion
            # -------------------------------------------------------------
            
            # Log completion
            end_time = time.time()
            self.total_time = end_time - start_time
            print(f"Rank {self.rank}: Inference completed in {self.total_time:.2f} seconds")
            
            # Print distributed status
            if dist.is_initialized():
                print(f"Rank {self.rank}: Process group will be destroyed at the end of main()")
        
        except Exception as e:
            # Only rank 0 does any cleanup
            if self.rank == 0:
                # Skip cleanup if we're preserving the temp directory for later blending
                if hasattr(self, 'temp_dir_preserved') and self.temp_dir_preserved:
                    print(f"Rank {self.rank}: Error occurred, but preserving temporary directory for later blending: {self.temp_dir}/temp.zarr")
                    
                    # We should NOT clean up the ZarrTempStorage, but we should close it
                    if self.temp_storage is not None:
                        # Just close the store without deleting anything
                        self.temp_storage.temp_zarr = None
                else:
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
# Define a custom collate function to ensure positions stay as tuples
def custom_collate(batch):
    data = torch.stack([item['data'] for item in batch])
    positions = [item['pos'] for item in batch]  # Keep positions as a list of tuples
    indices = [item['index'] for item in batch]
    return {'data': data, 'pos': positions, 'index': indices}

def find_free_port(start_port=12000, max_tries=100):
    """Find a free port starting from start_port."""
    import socket
    from contextlib import closing
    
    for port in range(start_port, start_port + max_tries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.2)  # Quick timeout
            try:
                # Try to bind to the port
                sock.bind(('localhost', port))
                return port
            except (socket.error, OSError):
                # Port is in use, try the next one
                continue
    
    # If we get here, we couldn't find a free port
    raise RuntimeError(f"Could not find a free port in range [{start_port}-{start_port+max_tries-1}]")

def run_worker(rank, world_size, 
              input_path, 
              output_path, 
              model_folder,
              hf_model_path,
              hf_token,
              fold,
              checkpoint,
              dist_port,
              batch_size,
              step_size,
              input_format,
              num_workers,
              num_write_workers,
              verbose,
              use_mirroring,
              max_tta_combinations,
              use_rotation_tta,
              rotation_weights,
              save_probability_maps,
              skip_blending,
              threshold,
              dist_backend,
              num_parts,
              part_id,
              use_fsspec=False,
              scroll_id=None,
              energy=None,
              resolution=None,
              segment_id=None):
    """Worker function that runs on each GPU with explicit arguments."""
    import torch.distributed as dist
    import torch
    import os
    import time
    
    # Print out the directly passed values to confirm
    print(f"Rank {rank}: Input path = '{input_path}'")
    print(f"Rank {rank}: Output path = '{output_path}'")
    
    # Ensure input path is not empty or None
    if not input_path:
        print(f"Rank {rank}: ERROR - Empty input path detected! This should not happen.")
        raise ValueError(f"Input path is empty. Cannot proceed with inference.")

    # Configure NumExpr and NumPy thread settings for this worker
    os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())  # Use all available cores
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())      # For NumPy operations
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())      # For NumPy with MKL backend

    # Set environment variables for this process (should already be set in main)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # The port should have been pre-selected by the parent process and passed via args
    # Just use the port directly from args - no file coordination needed
    port = dist_port
    print(f"Rank {rank}: Using port {port} for distributed communication")
    os.environ['MASTER_PORT'] = str(port)

    # Choose appropriate backend based on available hardware
    backend = dist_backend
    if not torch.cuda.is_available() and backend == 'nccl':
        backend = 'gloo'  # Fall back to gloo if NCCL requested but no GPUs available
        print(f"Rank {rank}: Falling back to gloo backend since NCCL requires CUDA devices")

    # Set DataLoader multiprocessing settings to prevent pickling issues with lambda functions
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Initialize process group with timeout and retry logic (following nnUNetv2's approach)
    max_retries = 5
    retry_interval = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Rank {rank}: Initializing process group with {backend} backend (attempt {attempt+1}/{max_retries})")
            # Simple init_process_group call (similar to nnUNetv2's approach)
            dist.init_process_group(backend=backend)
            print(f"Rank {rank}: Process group initialized successfully, world_size={world_size}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Rank {rank}: Process group initialization failed: {e}. Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
                # Increase retry interval with each attempt (exponential backoff)
                retry_interval *= 1.5
            else:
                print(f"Rank {rank}: Failed to initialize process group after {max_retries} attempts: {e}")
                raise

    # Set the device for this process
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        # Use modulo in case there are more ranks than GPUs
        device_idx = rank % device_count
        device = f"cuda:{device_idx}"
        torch.cuda.set_device(device_idx)
        # Print CUDA device properties for debugging
        device_props = torch.cuda.get_device_properties(device_idx)
        print(f"Rank {rank}: Assigned to device {device} ({device_props.name} with {device_props.total_memory/1e9:.1f} GB memory)")
    else:
        # Fallback to CPU if no CUDA devices
        device = "cpu"
        print(f"Rank {rank}: No CUDA devices available, using CPU")

    # Convert fold to int if numeric
    try:
        fold_int = int(fold)
        fold = fold_int
    except ValueError:
        pass  # Keep as string if not numeric (e.g., "all")

    # Adjust workers based on world size for DDP
    dataloader_workers = max(1, num_workers // world_size)
    write_workers = max(1, num_write_workers // world_size)
    if rank == 0:
        print(f"Adjusting workers for {world_size} processes:")
        print(f"  - Dataloader workers: {num_workers} -> {dataloader_workers} per process")
        print(f"  - Write workers: {num_write_workers} -> {write_workers} per process")

    # Load the model first - each process loads its own copy
    print(f"Rank {rank}: Loading model...")
    # TTA is already calculated above
    
    model_info = load_model_for_inference(
        model_folder=model_folder,
        hf_model_path=hf_model_path,
        hf_token=hf_token,
        fold=fold,
        checkpoint_name=checkpoint,
        device_str=device,
        use_mirroring=use_mirroring,  # Pass the explicit parameter
        verbose=verbose and (rank == 0),
        rank=rank
    )
    print(f"Rank {rank}: Model loaded successfully")
    
    # Get patch size from model info
    patch_size = model_info['patch_size']
    
    # Get number of input channels from model info
    num_input_channels = model_info['num_input_channels']
    
    # Determine number of output channels
    num_seg_heads = model_info.get('num_seg_heads', 2)
    
    # Create output targets configuration
    output_targets = [{
        "name": "segmentation",
        "channels": num_seg_heads,
        "activation": "sigmoid" if num_seg_heads <= 2 else "softmax",
        "nnunet_output_channels": num_seg_heads
    }]
    
    # Create dataset
    print(f"Rank {rank}: Creating dataset...")
    
    # Validate partitioning parameters if specified
    if num_parts > 1:
        if part_id < 0 or part_id >= num_parts:
            raise ValueError(f"part_id must be between 0 and {num_parts-1}, got {part_id}")
        print(f"Rank {rank}: Processing part {part_id} of {num_parts} parts along Z-axis")

    # Make sure we use the correct input_path variable that we verified above
    # Force input_path to be a string, in case it was converted to something else
    input_path = str(input_path) if input_path is not None else None
    print(f"Rank {rank}: Creating dataset with input_path='{input_path}' (type: {type(input_path)})")
    
    dataset = VCDataset(
        input_path=input_path,
        targets=output_targets,
        patch_size=patch_size,
        num_input_channels=num_input_channels,
        input_format=input_format,  # Use the explicit parameter
        step_size=step_size,
        load_all=False,
        verbose=verbose and (rank == 0),
        num_parts=num_parts,
        part_id=part_id,
        # Volume-specific parameters passed through worker function arguments
        # These should be included in the worker function parameters if needed
        scroll_id=scroll_id,  # Pass scroll_id from args
        energy=energy,         # Pass energy from args
        resolution=resolution, # Pass resolution from args
        segment_id=segment_id, # Pass segment_id from args
        cache=True,  # Default to True
        normalize=False,  # Default to False
        domain=None,
        use_fsspec=use_fsspec  # Use the args.use_fsspec flag
    )
    
    # Split the dataset for distributed processing
    dataset.set_distributed(rank, world_size)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=True,
        collate_fn=custom_collate,
        prefetch_factor=8
    )
    
    print(f"Rank {rank}: Created dataset with {len(dataset)} patches")
    
    # Print parameters for debug when verbose is on
    if verbose and rank == 0:
        print(f"Worker TTA parameters:")
        print(f"  - use_mirroring: {use_mirroring}")
        print(f"  - max_tta_combinations: {max_tta_combinations}")
        print(f"  - use_rotation_tta: {use_rotation_tta}")
    
    # Create inference handler with the loaded model and dataset
    # TTA is always enabled by default, with rotation taking precedence
    # Only disable TTA if use_mirroring is False (explicitly disabled)
    use_tta = use_mirroring
    
    # Force always using TTA unless explicitly disabled
    if not use_tta and verbose and rank == 0:
        print(f"Test time augmentation is DISABLED: use_tta={use_tta}")
    elif verbose and rank == 0:
        print(f"Test time augmentation is ENABLED: use_tta={use_tta}, use_rotation_tta={use_rotation_tta}")
    
    # Use rotation TTA by default, unless explicitly disabled
    use_rotation_tta = use_rotation_tta and use_tta
    
    inference = ZarrInferer(
        input_path=input_path,  # Use the validated input_path variable
        output_path=output_path,
        model_info=model_info,
        dataset=dataset,
        dataloader=dataloader,
        patch_size=patch_size,
        batch_size=batch_size,
        step_size=step_size,
        num_write_workers=write_workers,
        threshold=threshold,
        use_mirroring=use_tta,
        max_tta_combinations=max_tta_combinations,
        use_rotation_tta=use_rotation_tta,
        rotation_weights=rotation_weights,
        save_probability_maps=save_probability_maps,
        verbose=verbose and (rank == 0),
        rank=rank,
        edge_weight_boost=0,
        output_targets=output_targets
    )

    # Run inference with guaranteed cleanup
    inference_start_time = time.time()
    try:
        # Run inference - no timeout, let it take as long as needed
        inference.infer(skip_blending=skip_blending)
        
        # Print completion message
        print(f"Rank {rank}: Inference completed in {inference.total_time:.2f} seconds")
            
    except Exception as e:
        # Calculate elapsed time in case of error
        inference_elapsed = time.time() - inference_start_time
        inference.total_time = inference_elapsed  # Set the total_time even on error
        
        # Print error information
        print(f"Rank {rank}: Error after {inference_elapsed:.2f} seconds: {e}")
            
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup distributed process group - following nnUNetv2's simpler approach
        if dist.is_initialized():
            try:
                # Clean up CUDA memory first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Simple barrier call to ensure all processes reach this point
                print(f"Rank {rank}: Synchronizing for cleanup...")
                dist.barrier()
                
                # Destroy the process group (simple approach like nnUNetv2)
                print(f"Rank {rank}: Destroying process group...")
                dist.destroy_process_group()
                print(f"Rank {rank}: Process group successfully destroyed")
                
            except Exception as e:
                print(f"Rank {rank}: Error during cleanup: {e}")
                try:
                    # Simple fallback cleanup attempt
                    dist.destroy_process_group()
                except:
                    pass
        else:
            print(f"Rank {rank}: Process group was not initialized, no cleanup needed")

def main():
    """
    Main entry point that parses arguments and handles multi-GPU setup.
    """
    import argparse
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch
    import os
    import signal
    import sys
    
    # Set the multiprocessing start method to 'spawn' to avoid fork issues with TensorStore
    # Force=True to make sure it's set even if another method was already set
    mp.set_start_method('spawn', force=True)
    
    # Configure NumExpr and NumPy thread settings
    os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())  # Use all available cores
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())      # For NumPy operations
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())      # For NumPy with MKL backend
    
    if os.cpu_count() is not None:
        print(f"Setting numerical libraries to use {os.cpu_count()} threads")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run nnUNet inference on a zarr array')
    # Input/output options
    parser.add_argument('--input', type=str, required=False, 
                        help='Path to input zarr array, or identifier for Volume (e.g. "scroll1", "scroll2", or segment ID)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to output zarr file (must end with .zarr)')
    
    # Volume-specific options
    parser.add_argument('--input_format', type=str, default='zarr', choices=['zarr', 'volume'],
                        help='Format of the input data (zarr or volume)')
    parser.add_argument('--scroll_id', type=str, 
                        help='ID of the scroll (for Volume, e.g. "1", "2", "1b")')
    parser.add_argument('--energy', type=int, 
                        help='Energy value for Volume (e.g. 54, 53, 88)')
    parser.add_argument('--resolution', type=float, 
                        help='Resolution value for Volume (e.g. 7.91, 3.24)')
    parser.add_argument('--segment_id', type=int, 
                        help='ID of the segment for Volume')
    parser.add_argument('--volume_cache', action='store_true', default=True,
                        help='Enable caching for Volume')
    parser.add_argument('--no_volume_cache', action='store_true',
                        help='Disable caching for Volume')
    parser.add_argument('--volume_normalize', action='store_true', default=False,
                        help='Normalize Volume data')
    parser.add_argument('--volume_domain', type=str, choices=['dl.ash2txt', 'local'],
                        help='Domain for Volume (dl.ash2txt or local)')
    parser.add_argument('--use_fsspec', action='store_true', default=False,
                        help='Use fsspec instead of TensorStore for faster data access with Volume')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for distributed inference')
    parser.add_argument('--model_folder', type=str, help='Path to nnUNet model folder')
    parser.add_argument('--hf_model_path', type=str, help='Hugging Face model repository path (e.g., "username/model-name")')
    parser.add_argument('--hf_token', type=str, help='Optional Hugging Face token for private repositories')
    parser.add_argument('--fold', type=str, default='0', help='Model fold to use (default: 0)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint name (default: checkpoint_final.pth)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference (default: 4)')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size as fraction of patch size (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_write_workers', type=int, default=4, help='Number of writer threads (default: 4)')
    parser.add_argument('--max_tta_combinations', type=int, default=3, help='Number of TTA combinations (default: 3); set to 0 to disable all TTA')
    # We want rotation TTA to be on by default, but argparse store_true implies default=False
    # So we instead use a store_false flag for "--no_rotation_tta" which is clearer
    parser.add_argument('--no_rotation_tta', action='store_true', help='Disable rotation-based TTA (by default rotation TTA is enabled)')
    parser.add_argument('--rotation_weights', type=float, nargs=3, default=None,
                        help='Weights for each rotation axis in rotation TTA. Three values for [z, x, y] axes. Default is equal weights.')
    parser.add_argument('--threshold', type=float, help='Optional threshold (0-100) for binarizing the output')
    parser.add_argument('--no_mirroring', action='store_true', help='Disable test time augmentation completely (only when combined with --max_tta_combinations=0)')
    parser.add_argument('--no_probabilities', action='store_true',
                        help='Do not save probability maps, save argmax for multiclass segmentation')
    parser.add_argument('--skip_blending', action='store_true',
                        help='Skip the blending phase and only save inference patches for later blending')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--dist_backend', type=str, default='nccl', 
                        help='PyTorch distributed backend (default: nccl, alternatives: gloo, mpi)')
    parser.add_argument('--dist_port', type=int, default=12355,
                        help='Port to use for distributed training (default: 12355)')
    parser.add_argument('--num_parts', type=int, default=1,
                        help='Number of parts to divide the volume into along Z-axis (default: 1, no partitioning)')
    parser.add_argument('--part_id', type=int, default=0,
                        help='Part ID to process (0-indexed, bottom of volume, must be < num_parts, default: 0)')

    args = parser.parse_args()

    # Ensure output path ends with .zarr
    if not args.output.endswith('.zarr'):
        raise ValueError(f"Output path must end with '.zarr', got: {args.output}")
    
    # Validate input requirements
    if args.input_format == 'volume':
        # For Volume class, we need either input or scroll_id
        if args.input is None and args.scroll_id is None:
            raise ValueError("When using --input_format=volume, either --input or --scroll_id must be provided")
        # If input is not provided but scroll_id is, set a default input
        if args.input is None and args.scroll_id is not None:
            args.input = f"scroll{args.scroll_id}"
    else:
        # For zarr format, input is required
        if args.input is None:
            raise ValueError("--input is required when using --input_format=zarr")

    # Ensure either local model folder or HF model path is provided
    if args.model_folder is None and args.hf_model_path is None:
        raise ValueError("Either --model_folder or --hf_model_path must be provided")

    # Check for available GPUs
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        print("No CUDA devices available! Falling back to CPU.")
        # Run in single process mode on CPU
        args.device = "cpu"
        args.num_gpus = 1
        args.dist_backend = 'gloo'  # Use gloo backend for CPU
        
        # Create inference handler for CPU
        single_process_inference(args)
    else:
        # Limit num_gpus to available GPUs
        num_gpus = min(args.num_gpus, available_gpus)
        if num_gpus < args.num_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {num_gpus} GPUs.")

        if num_gpus == 1:
            # Single GPU mode - no need for distributed processing
            print(f"Running in single GPU mode on {args.device}")
            
            # Create inference handler for single GPU
            single_process_inference(args)
        else:
            # Multi-GPU mode
            print(f"Running in multi-GPU mode with {num_gpus} GPUs using {args.dist_backend} backend")
            
            # Add signal handlers to properly clean up on interruption
            def signal_handler(sig, frame):
                print(f"Received signal {sig}, terminating workers...")
                # In a real scenario, you might want to terminate child processes here
                sys.exit(0)
                
            # Register signal handlers
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # kill or system shutdown

            # Set multiprocessing method (following nnUNetv2)
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # It's already set, which is fine
                pass

            # Set port automatically or use provided (like nnUNetv2)
            if 'MASTER_PORT' not in os.environ:
                # Import socket to find free port (like nnUNetv2)
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", 0))
                port = s.getsockname()[1]
                s.close()
                os.environ['MASTER_PORT'] = str(port)
                print(f"Using automatically selected port: {port}")
            else:
                print(f"Using existing MASTER_PORT: {os.environ['MASTER_PORT']}")
                
            os.environ['MASTER_ADDR'] = 'localhost'
            
            # Launch processes using spawn (following nnUNetv2's approach)
            print(f"Starting {num_gpus} worker processes...")
            
            try:
                # Select a free port before spawning processes and share it
                port = args.dist_port
                try:
                    # Quick check if the port is available
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(('localhost', port))
                    sock.close()
                except (socket.error, OSError):
                    print(f"Port {port} is in use. Finding a free port...")
                    port = find_free_port(start_port=port)
                    print(f"Using automatically selected port: {port}")
                
                # Update args with the selected port
                args.dist_port = port
                
                # Create and write to the port file before spawning workers
                parent_pid = os.getpid()
                port_file = os.path.join('/tmp', f'ddp_port_{parent_pid}.txt')
                with open(port_file, 'w') as f:
                    f.write(str(port))
                print(f"Wrote port {port} to {port_file}")
                
                # Use mp.spawn like nnUNetv2 (simpler approach)
                print(f"Starting {num_gpus} worker processes...")
                
                # Get specific arguments needed instead of passing the entire args object
                input_path = args.input
                output_path = args.output
                model_folder = args.model_folder
                hf_model_path = args.hf_model_path
                hf_token = args.hf_token
                fold = args.fold
                checkpoint = args.checkpoint
                dist_port = args.dist_port
                batch_size = args.batch_size
                step_size = args.step_size 
                input_format = args.input_format
                num_workers = args.num_workers
                num_write_workers = args.num_write_workers
                verbose = args.verbose
                # If no_mirroring is True OR max_tta_combinations is 0, disable TTA
                use_mirroring = not (args.no_mirroring or args.max_tta_combinations == 0)
                # Override: Always enable TTA by default unless explicitly disabled
                use_mirroring = True
                max_tta_combinations = args.max_tta_combinations
                use_rotation_tta = not args.no_rotation_tta and use_mirroring
                rotation_weights = args.rotation_weights
                save_probability_maps = not args.no_probabilities
                skip_blending = args.skip_blending
                threshold = args.threshold
                dist_backend = args.dist_backend
                num_parts = args.num_parts
                part_id = args.part_id
                
                print(f"Starting worker processes with essential args:")
                print(f"  - input_path: {input_path}")
                print(f"  - output_path: {output_path}")
                print(f"  - num_parts: {num_parts}")
                print(f"  - part_id: {part_id}")
                print(f"  - batch_size: {batch_size}")
                
                # Display TTA parameters
                print(f"  - use_mirroring: {use_mirroring}")
                print(f"  - max_tta_combinations: {max_tta_combinations}")
                print(f"  - use_rotation_tta: {use_rotation_tta}")
                print(f"  - no_mirroring flag: {args.no_mirroring}")
                print(f"  - skip_blending: {skip_blending}")
                
                # Extract additional Volume-specific parameters
                use_fsspec = args.use_fsspec if hasattr(args, 'use_fsspec') else False
                scroll_id = args.scroll_id if hasattr(args, 'scroll_id') else None
                energy = args.energy if hasattr(args, 'energy') else None
                resolution = args.resolution if hasattr(args, 'resolution') else None
                segment_id = args.segment_id if hasattr(args, 'segment_id') else None
                
                print(f"  - use_fsspec: {use_fsspec}")
                if scroll_id:
                    print(f"  - scroll_id: {scroll_id}")
                if energy:
                    print(f"  - energy: {energy}")
                if resolution:
                    print(f"  - resolution: {resolution}")
                if segment_id:
                    print(f"  - segment_id: {segment_id}")
                
                mp.spawn(
                    run_worker,
                    args=(num_gpus, 
                          input_path, 
                          output_path, 
                          model_folder,
                          hf_model_path,
                          hf_token,
                          fold,
                          checkpoint,
                          dist_port,
                          batch_size,
                          step_size,
                          input_format,
                          num_workers,
                          num_write_workers,
                          verbose,
                          use_mirroring,
                          max_tta_combinations,
                          use_rotation_tta,
                          rotation_weights,
                          save_probability_maps,
                          skip_blending,
                          threshold,
                          dist_backend,
                          num_parts,
                          part_id,
                          use_fsspec,
                          scroll_id,
                          energy,
                          resolution,
                          segment_id),
                    nprocs=num_gpus,
                    join=True
                )
                print("All worker processes have completed")
                
                # Clean up port file after all processes complete
                if os.path.exists(port_file):
                    os.remove(port_file)
                    print(f"Removed port file {port_file}")
                
            except KeyboardInterrupt:
                print("\nCaught keyboard interrupt, terminating workers...")
                sys.exit(1)
            except Exception as e:
                print(f"Error in main process: {e}")
                raise

def single_process_inference(args):
    """Run inference in single process mode"""
    import torch
    
    # Make input_path available for the function
    input_path = args.input
    print(f"Single process: Input path = '{input_path}' (type: {type(input_path)})")
    
    # Ensure input path is not empty or None
    if not input_path:
        print(f"Single process: ERROR - Empty input path detected! This should not happen.")
        raise ValueError("Input path is empty or None. Cannot proceed with inference.")
        
    # Ensure input_path is a string
    input_path = str(input_path)
    
    # Convert fold to int if numeric
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold  # Keep as string if not numeric (e.g., "all")
    
    # Load the model first
    print("Loading model...")
    # TTA is always enabled by default, with rotation taking precedence
    # Only disable TTA if args.no_mirroring is True AND args.max_tta_combinations is 0
    use_tta = not (args.no_mirroring and args.max_tta_combinations == 0)
    
    model_info = load_model_for_inference(
        model_folder=args.model_folder,
        hf_model_path=args.hf_model_path,
        hf_token=args.hf_token,
        fold=fold,
        checkpoint_name=args.checkpoint,
        device_str=args.device,
        use_mirroring=use_tta,  # Use TTA by default
        verbose=args.verbose,
        rank=0
    )
    print("Model loaded successfully")
    
    # Get patch size from model info
    patch_size = model_info['patch_size']
    
    # Get number of input channels from model info
    num_input_channels = model_info['num_input_channels']
    
    # Determine number of output channels
    num_seg_heads = model_info.get('num_seg_heads', 2)
    
    # Create output targets configuration
    output_targets = [{
        "name": "segmentation",
        "channels": num_seg_heads,
        "activation": "sigmoid" if num_seg_heads <= 2 else "softmax",
        "nnunet_output_channels": num_seg_heads
    }]
    
    # Create dataset
    print("Creating dataset...")
    # Validate partitioning parameters if specified
    if args.num_parts > 1:
        if args.part_id < 0 or args.part_id >= args.num_parts:
            raise ValueError(f"part_id must be between 0 and {args.num_parts-1}, got {args.part_id}")
        print(f"Processing part {args.part_id} of {args.num_parts} parts along Z-axis")
    
    # Determine input_path based on format
    input_path = args.input
    
    # Create dataset
    dataset = VCDataset(
        input_path=input_path,
        targets=output_targets,
        patch_size=patch_size,
        num_input_channels=num_input_channels,
        input_format=args.input_format,  # Use the command-line argument
        step_size=args.step_size,
        load_all=False,
        verbose=args.verbose,
        num_parts=args.num_parts,
        part_id=args.part_id,
        # Volume-specific parameters
        scroll_id=args.scroll_id if hasattr(args, 'scroll_id') else None,
        energy=args.energy if hasattr(args, 'energy') else None,
        resolution=args.resolution if hasattr(args, 'resolution') else None,
        segment_id=args.segment_id if hasattr(args, 'segment_id') else None,
        cache=not args.no_volume_cache if hasattr(args, 'no_volume_cache') else True,
        normalize=args.volume_normalize if hasattr(args, 'volume_normalize') else False,
        domain=args.volume_domain if hasattr(args, 'volume_domain') else None,
        use_fsspec=args.use_fsspec if hasattr(args, 'use_fsspec') else False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
        prefetch_factor=8
    )
    
    print(f"Created dataset with {len(dataset)} patches")
    
    # Create inference handler with the loaded model and dataset
    # TTA is always enabled by default, with rotation taking precedence
    # Only disable TTA if args.no_mirroring is True AND args.max_tta_combinations is 0
    use_tta = not (args.no_mirroring and args.max_tta_combinations == 0)
    
    # Use rotation TTA by default, unless explicitly disabled with --no_rotation_tta
    use_rotation_tta = not args.no_rotation_tta and use_tta
    
    inference = ZarrInferer(
        input_path=input_path,  # Use the validated input_path variable
        output_path=args.output,
        model_info=model_info,
        dataset=dataset,
        dataloader=dataloader,
        patch_size=patch_size,
        batch_size=args.batch_size,
        step_size=args.step_size,
        num_write_workers=args.num_write_workers,
        threshold=args.threshold,
        use_mirroring=use_tta,
        max_tta_combinations=args.max_tta_combinations,
        use_rotation_tta=use_rotation_tta,
        rotation_weights=args.rotation_weights,
        save_probability_maps=not args.no_probabilities,
        verbose=args.verbose,
        rank=0,
        edge_weight_boost=0,
        output_targets=output_targets
    )

    # Run inference
    inference_start_time = time.time()
    try:
        # Run inference - no timeout, let it take as long as needed
        inference.infer(skip_blending=args.skip_blending)
        
        # Print completion message
        print(f"Inference completed in {inference.total_time:.2f} seconds")
            
    except Exception as e:
        # Calculate elapsed time in case of error
        inference_elapsed = time.time() - inference_start_time
        inference.total_time = inference_elapsed  # Set the total_time even on error
        
        # Print error information
        print(f"Error after {inference_elapsed:.2f} seconds: {e}")
            
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()