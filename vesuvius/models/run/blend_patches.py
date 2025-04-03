import os
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
import torch.nn
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import time
import json
import argparse
from pathlib import Path
from utils.models.blending import (
    create_gaussian_weights_torch,
    blend_patch_torch,
    blend_patch_weighted,
    intersects_chunk
)
from data.io.zarrio.zarr_temp_storage import ZarrTempStorage


def find_all_temp_storage_paths(base_temp_storage_path: str, verbose: bool = False) -> List[str]:
    """
    Find all related temp storage paths that may exist for different parts.
    
    Args:
        base_temp_storage_path: The path to the main temp storage directory provided by the user
        verbose: Whether to print verbose output
        
    Returns:
        List of all temp storage paths found
    """
    # Always include the base path
    temp_paths = [base_temp_storage_path]
    
    # Get base directory path and check for part-specific directories
    base_dir = os.path.dirname(base_temp_storage_path)
    parent_dir = os.path.dirname(base_dir)
    
    # Track part indices to avoid duplicates
    found_parts = set()
    
    # Check if we're in a part-specific directory
    if "temp_part" in base_dir:
        base_part = os.path.basename(base_dir)
        found_parts.add(base_part)
        if verbose:
            print(f"Detected part directory: {base_part}")
        
        # Look for other parts
        for entry in os.listdir(parent_dir):
            if entry.startswith("temp_part") and entry != base_part:
                # Found another part directory
                other_part_path = os.path.join(parent_dir, entry, os.path.basename(base_temp_storage_path))
                
                # Check if it exists and is valid
                if os.path.exists(other_part_path):
                    try:
                        # Try to open it to verify it's valid
                        test_store = zarr.open(other_part_path, mode='r')
                        if not list(test_store.keys()):
                            if verbose:
                                print(f"Skipping empty zarr store at: {other_part_path}")
                            continue
                        
                        if verbose:
                            print(f"Found additional part path: {other_part_path}")
                        temp_paths.append(other_part_path)
                        found_parts.add(entry)
                    except Exception as e:
                        if verbose:
                            print(f"Error opening zarr at {other_part_path}: {e}")
    
    # If we're in a regular temp directory, also look for part-specific directories
    elif os.path.basename(base_dir) == "temp":
        if verbose:
            print(f"In non-part-specific directory, checking for part directories")
        
        for entry in os.listdir(parent_dir):
            if entry.startswith("temp_part"):
                part_path = os.path.join(parent_dir, entry, os.path.basename(base_temp_storage_path))
                if os.path.exists(part_path):
                    try:
                        test_store = zarr.open(part_path, mode='r')
                        if not list(test_store.keys()):
                            if verbose:
                                print(f"Skipping empty zarr store at: {part_path}")
                            continue
                            
                        if verbose:
                            print(f"Found part-specific path: {part_path}")
                        temp_paths.append(part_path)
                    except Exception as e:
                        if verbose:
                            print(f"Error opening zarr at {part_path}: {e}")
    
    if verbose:
        print(f"Found {len(temp_paths)} valid temp storage paths to blend")
        for i, path in enumerate(temp_paths):
            print(f"  {i+1}: {path}")
    
    return temp_paths


def blend_saved_patches(
    temp_storage_path: str,
    output_path: str,
    patch_size: Tuple[int, int, int],
    step_size: float = 0.5,
    threshold: Optional[float] = None,
    save_probability_maps: bool = True,
    verbose: bool = False,
    edge_weight_boost: float = 0,
    force_overwrite: bool = False,
    cleanup_temp: bool = True,
    include_all_parts: bool = True,
    chunk_memory_limit_gb: float = 16.0,
):
    """
    Blend patches from a temporary storage location into a final segmentation output.
    
    This function loads patches saved during the inference phase and blends them
    together using Gaussian weighting to create a smooth final output.
    
    Args:
        temp_storage_path: Path to the temporary storage location containing inference patches
        output_path: Path to save the final blended output (zarr file)
        patch_size: Patch size used during inference as (z, y, x) tuple
        step_size: Step size used during inference (as fraction of patch size)
        threshold: Optional threshold value (0-100) for binarizing the probability map
        save_probability_maps: Save full probability maps for multiclass segmentation
        verbose: Enable detailed output messages during blending
        edge_weight_boost: Factor to boost Gaussian weights at patch edges
        force_overwrite: If True, overwrite existing output file without prompting
        cleanup_temp: If True, clean up temporary storage files after blending
        include_all_parts: If True, automatically find and include patches from all part-specific
                           temp directories (when using num_parts/part_id in dataset)
        chunk_memory_limit_gb: Maximum GPU memory to use per chunk in GB (adjust based on GPU VRAM)
    
    Returns:
        Path to the output zarr file, or None if blending was cancelled
    """
    if verbose:
        print(f"Starting patch blending from {temp_storage_path} to {output_path}")
    
    # Ensure output path ends with .zarr
    if not output_path.endswith('.zarr'):
        output_path = output_path + '.zarr'
        if verbose:
            print(f"Added .zarr extension to output path: {output_path}")
    
    # Check if output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if output file already exists
    if os.path.exists(output_path):
        if force_overwrite:
            print(f"Output file {output_path} already exists - overwriting as requested with --force")
            import shutil
            shutil.rmtree(output_path)
        else:
            print(f"Warning: Output file {output_path} already exists")
            user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Aborting blend operation")
                return None
            else:
                print(f"Removing existing output file at {output_path}")
                import shutil
                shutil.rmtree(output_path)
    
    # Find all temp storage paths to blend
    temp_storage_paths = [temp_storage_path]
    if include_all_parts:
        temp_storage_paths = find_all_temp_storage_paths(temp_storage_path, verbose)
    
    if verbose:
        print(f"Will blend patches from {len(temp_storage_paths)} temp storage locations")
    
    # Initialize a list to store all temp_storage objects
    temp_storages = []
    
    # First, get available targets from all temp storage paths
    available_targets = []
    
    for i, curr_temp_path in enumerate(temp_storage_paths):
        # Initialize ZarrTempStorage with rank 0 to access the saved patches
        # Detect the actual number of ranks by checking for rank_* directories in the zarr file
        temp_dir = zarr.open(curr_temp_path, mode='r')
        available_ranks = [key for key in temp_dir.keys() if key.startswith('rank_')]
        world_size = len(available_ranks)
        
        if verbose:
            print(f"Detected {world_size} rank directories in {curr_temp_path}: {available_ranks}")
            
        # Ensure world_size is at least 1
        world_size = max(1, world_size)
        
        temp_storage = ZarrTempStorage(
            output_path=os.path.dirname(curr_temp_path),
            rank=0,
            world_size=world_size,  # Use the detected number of ranks
            verbose=verbose,
            num_io_workers=1  # Only need 1 worker for reading
        )
        
        # Open the temp zarr store instead of creating a new one
        temp_storage.temp_zarr_path = curr_temp_path
        temp_storage.temp_zarr = zarr.open(curr_temp_path, mode='r')
        
        # Store the temp_storage object for later use
        temp_storages.append(temp_storage)
        
        # Look for patches in all rank directories
        try:
            for group_name in temp_storage.temp_zarr.group_keys():
                if group_name.startswith('rank_'):
                    rank_group = temp_storage.temp_zarr[group_name]
                    if 'patches' in rank_group:
                        patches_group = rank_group['patches']
                        for target_name in patches_group.array_keys():
                            if target_name not in available_targets:
                                available_targets.append(target_name)
        except Exception as e:
            print(f"Error getting available targets from {curr_temp_path}: {e}")
            # Continue to the next temp storage path instead of raising
            continue
    
    if not available_targets:
        raise ValueError(f"No target patches found in any of the {len(temp_storage_paths)} temp storage locations")
    
    if verbose:
        print(f"Found targets across all temp storage paths: {available_targets}")
    
    # For each target, create output arrays and blend patches
    for target_name in available_targets:
        if verbose:
            print(f"Processing target: {target_name}")
        
        # Direct approach: find all unique patches with their positions
        all_patches = []  # Will store tuples of (path, rank, idx, position)
        seen_positions = set()
        position_count = 0
        duplicate_count = 0
        
        # Scan all temp stores for patches with positions
        for temp_idx, temp_storage in enumerate(temp_storages):
            storage_path = temp_storage.temp_zarr_path
            
            if verbose:
                print(f"Scanning patches in {storage_path}")
            
            try:
                # Open the zarr store
                store = zarr.open(storage_path, mode='r')
                
                # Find rank directories
                rank_groups = [k for k in store.keys() if k.startswith('rank_')]
                if verbose:
                    print(f"  Found {len(rank_groups)} ranks: {rank_groups}")
                
                # Process each rank
                for rank_name in rank_groups:
                    rank_idx = int(rank_name.replace('rank_', ''))
                    rank_group = store[rank_name]
                    
                    if 'patches' not in rank_group:
                        continue
                    
                    patches_group = rank_group['patches']
                    
                    # Check if target exists
                    if target_name not in patches_group:
                        continue
                    
                    # Get the patches array
                    patches_array = patches_group[target_name]
                    
                    # Check if has position channels
                    has_position = patches_array.attrs.get('has_position_channels', False)
                    if not has_position:
                        if verbose:
                            print(f"  Warning: Target {target_name} in {rank_name} does not have embedded positions")
                        continue
                    
                    # Get actual patch count 
                    count = patches_array.attrs.get('actual_count', patches_array.shape[0])
                    if verbose:
                        print(f"  Target {target_name} in {rank_name} has {count} patches")
                    
                    # Extract positions from all patches
                    for idx in range(count):
                        try:
                            # Get just the position channels' first pixel
                            pos_data = patches_array[idx, :3, 0, 0, 0]
                            position = (int(pos_data[0]), int(pos_data[1]), int(pos_data[2]))
                            
                            # Check if this position is already seen
                            if position in seen_positions:
                                duplicate_count += 1
                                if verbose and duplicate_count <= 5:
                                    print(f"  Skipping duplicate position: {position}")
                                continue
                            
                            # If not a duplicate, add to our collections
                            seen_positions.add(position)
                            position_count += 1
                            
                            # Store patch info with source path
                            all_patches.append((storage_path, rank_idx, idx, position))
                            
                            # Print first few for debugging
                            if verbose and position_count <= 5:
                                print(f"  Added position {position} from {rank_name}, idx={idx}")
                        except Exception as e:
                            if verbose:
                                print(f"  Error extracting position for patch {idx}: {e}")
            except Exception as e:
                print(f"Error processing temp storage {storage_path}: {e}")
                continue
        
        if not all_patches:
            print(f"No valid patches found for {target_name}")
            continue
        
        if verbose:
            print(f"Found {len(all_patches)} unique patch positions (skipped {duplicate_count} duplicates)")
        
        # Calculate volume dimensions
        min_z, min_y, min_x = float('inf'), float('inf'), float('inf')
        max_z, max_y, max_x = 0, 0, 0
        
        for _, _, _, (z, y, x) in all_patches:
            min_z = min(min_z, z)
            min_y = min(min_y, y)
            min_x = min(min_x, x)
            max_z = max(max_z, z + patch_size[0])
            max_y = max(max_y, y + patch_size[1])
            max_x = max(max_x, x + patch_size[2])
        
        # Ensure min values are valid
        min_z = 0 if min_z == float('inf') else min_z
        min_y = 0 if min_y == float('inf') else min_y
        min_x = 0 if min_x == float('inf') else min_x
        
        if verbose:
            print(f"\nCoordinate ranges:")
            print(f"  Z range: {min_z} to {max_z}")
            print(f"  Y range: {min_y} to {max_y}")
            print(f"  X range: {min_x} to {max_x}")
        
        # Determine volume shape
        volume_shape = (max_z, max_y, max_x)
        if verbose:
            print(f"Determined volume shape: {volume_shape}")
        
        # Add diagnostics about positions
        if verbose:
            print("\nAnalyzing patch position distribution:")
            # Group by z position
            z_positions = {}
            for _, _, _, (z, y, x) in all_patches:
                z_bin = z // 50  # Group by 50-unit bins
                if z_bin not in z_positions:
                    z_positions[z_bin] = 0
                z_positions[z_bin] += 1
            
            # Print distribution
            print("Z-position distribution (binned by 50 units):")
            for z_bin in sorted(z_positions.keys()):
                z_min = z_bin * 50
                z_max = (z_bin + 1) * 50 - 1
                print(f"  z={z_min}-{z_max}: {z_positions[z_bin]} patches")
            
            # Count patches by source
            source_counts = {}
            for path, _, _, _ in all_patches:
                source_counts[path] = source_counts.get(path, 0) + 1
            
            print("\nPatches by storage source:")
            for path, count in source_counts.items():
                print(f"  {path}: {count} patches")
            print()
        
        # Store path to zarr arrays for efficient access
        zarr_arrays = {}  # Will store {path: {rank: {target: array}}}
        
        # Load references to all patch arrays in a path->rank->target->array structure
        for path, rank_idx, idx, position in all_patches:
            if path not in zarr_arrays:
                # First time seeing this path, create entry and load references
                zarr_arrays[path] = {}
                try:
                    store = zarr.open(path, mode='r')
                    for rank_name in [k for k in store.keys() if k.startswith('rank_')]:
                        rank = int(rank_name.replace('rank_', ''))
                        if 'patches' in store[rank_name]:
                            zarr_arrays[path][rank] = {}
                            for target_key in store[rank_name]['patches'].keys():
                                zarr_arrays[path][rank][target_key] = store[rank_name]['patches'][target_key]
                except Exception as e:
                    print(f"Error loading zarr arrays from {path}: {e}")
            
        if verbose:
            print("\nLoaded zarr array references:")
            for path in zarr_arrays:
                print(f"  {path}: {len(zarr_arrays[path])} ranks")
            print()
        
        # Get sample patch to determine output shape
        if not all_patches:
            raise ValueError(f"No valid patches found to blend")
            
        # Get first patch in the list for reference
        sample_path, sample_rank, sample_idx, _ = all_patches[0]
        try:
            # Get the sample patch
            sample_patch = zarr_arrays[sample_path][sample_rank][target_name][sample_idx]
            
            # Check if patches have embedded position channels
            has_position_channels = zarr_arrays[sample_path][sample_rank][target_name].attrs.get('has_position_channels', False)
            if has_position_channels and verbose:
                print(f"Detected patches with embedded position channels (first 3 channels)")
            
            # Determine number of output channels
            if has_position_channels:
                # Skip the first 3 position channels
                num_channels = sample_patch.shape[0] - 3
                if verbose:
                    print(f"Using {num_channels} data channels (skipping 3 position channels)")
            else:
                num_channels = sample_patch.shape[0]
                if verbose:
                    print(f"Using all {num_channels} channels (no embedded position channels)")
        except Exception as e:
            print(f"Error determining output shape: {e}")
            raise ValueError(f"Could not determine output shape from sample patch")
        
        # Determine if this is binary or multiclass segmentation
        is_multiclass = num_channels > 2
        is_binary = num_channels == 2
        
        # Determine final output shape based on segmentation type and flags
        if not save_probability_maps and is_multiclass:
            # Multiclass segmentation with argmax (single channel)
            final_shape = (1,) + volume_shape
        elif not save_probability_maps and is_binary:
            # Binary segmentation (single channel)
            final_shape = (1,) + volume_shape
        elif save_probability_maps and is_binary:
            # Binary segmentation with probabilities (two channels)
            final_shape = (2,) + volume_shape
        else:
            # Standard case - all probability maps
            final_shape = (num_channels,) + volume_shape
        
        # Create output arrays for accumulation
        if verbose:
            print(f"Creating output arrays for blending with shape {(num_channels,) + volume_shape}")
        
        # Determine output format
        final_dtype = 'uint8'
        
        # Create compressor
        compressor = Blosc(cname='zstd', clevel=3)
        
        # Create final output array
        if verbose:
            print(f"Creating final output array with shape {final_shape}, dtype {final_dtype}")
        
        final_arrays = {}
        final_arrays[target_name] = zarr.open(
            output_path,
            mode='w',
            shape=final_shape,
            chunks=(1,) + patch_size,
            dtype=final_dtype,
            compressor=compressor,
            write_empty_chunks=False
        )
        
        # Create temporary arrays for blending
        # Sum array for accumulating weighted patches
        output_arrays = {}
        output_arrays[target_name] = zarr.open(
            temp_storage_path.replace('.zarr', '_blend_sum.zarr'),
            mode='w',
            shape=(num_channels,) + volume_shape,
            chunks=(1,) + patch_size,
            dtype='float16',
            compressor=compressor,
            fill_value=0,
            write_empty_chunks=False
        )
        
        # Count array for tracking weights
        count_arrays = {}
        count_arrays[target_name] = zarr.open(
            temp_storage_path.replace('.zarr', '_blend_count.zarr'),
            mode='w',
            shape=volume_shape,
            chunks=patch_size,
            dtype='uint8',
            compressor=compressor,
            fill_value=0,
            write_empty_chunks=False
        )
        
        # Create Gaussian weights for blending
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if verbose:
            print(f"Creating Gaussian weights on {device}")
        
        # Create Gaussian weights with standard nnUNet parameters
        sigma_scale = 1/8
        value_scaling_factor = 10
        
        blend_weights = create_gaussian_weights_torch(
            patch_size,
            sigma_scale=sigma_scale,
            value_scaling_factor=value_scaling_factor,
            device=device,
            edge_weight_boost=edge_weight_boost
        )
        
        # Sort patches by z position for better memory locality
        all_patches.sort(key=lambda x: x[3][0])  # Sort by Z position
        
        # Log patch count by storage directory for debugging
        if verbose:
            print("\nFinal patch counts by source after deduplication:")
            patches_by_source = {}
            for path, _, _, _ in all_patches:
                if path not in patches_by_source:
                    patches_by_source[path] = 0
                patches_by_source[path] += 1
            
            for source, count in patches_by_source.items():
                print(f"  {source}: {count} patches")
        
        # Process in z-chunks to manage memory, with overlap to handle boundaries
        # Calculate approximate memory requirements for each slice
        # Output tensor is (channels, z, y, x) in float16: 2 bytes per value
        # Weight map is (z, y, x) in float32: 4 bytes per value
        mem_per_z_slice_mb = ((num_channels * 2 + 4) * max_y * max_x) / (1024 * 1024)
        
        # Convert chunk_memory_limit_gb to MB
        max_chunk_size_mb = chunk_memory_limit_gb * 1024
        
        # Calculate optimal chunk size based on available memory (in slices)
        optimal_chunk_size = min(512, max(16, int(max_chunk_size_mb / mem_per_z_slice_mb)))
        # Round down to multiple of 16 for better memory alignment
        chunk_size = max(16, (optimal_chunk_size // 16) * 16)
        
        if verbose:
            print(f"Memory per Z-slice: {mem_per_z_slice_mb:.2f} MB")
            print(f"Memory limit per chunk: {max_chunk_size_mb:.2f} MB")
            print(f"Using chunk size of {chunk_size} slices")
        else:
            print(f"Memory per Z-slice: {mem_per_z_slice_mb:.2f} MB, using chunk size of {chunk_size} slices")
        
        # Use adequate overlap for patch size
        overlap = min(patch_size[0], max(4, patch_size[0] // 2))  # Half patch size overlap with min/max bounds
        
        # Create overlapping chunks for processing
        z_chunks = []
        for i in range(0, max_z, chunk_size):
            chunk_start = max(0, i - overlap)  # Add overlap buffer at start
            chunk_end = min(max_z, i + chunk_size + overlap)  # Add overlap buffer at end
            # Track valid range (non-overlap) for final output
            valid_start = i
            valid_end = min(i + chunk_size, max_z)
            z_chunks.append((chunk_start, chunk_end, valid_start, valid_end))
        
        # Calculate total chunks
        total_chunks = len(z_chunks)
        print(f"Processing {total_chunks} z-chunks with {overlap}vx overlap, {len(all_patches)} unique patches")

        for chunk_idx, (z_start, z_end, valid_start, valid_end) in enumerate(z_chunks):
            print(f"Processing z-chunk {chunk_idx+1}/{total_chunks}: [{z_start}:{z_end}] (valid: [{valid_start}:{valid_end}])...")
            chunk_process_start = time.time()
            
            # Load chunk data
            output_chunk = output_arrays[target_name][:, z_start:z_end]
            count_chunk = count_arrays[target_name][z_start:z_end]

            # Create tensors from the zarr array views
            output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float16).contiguous()
            
            # Find patches that intersect with this chunk
            chunk_patches = []
            for patch_idx, patch_info in enumerate(all_patches):
                path, rank_idx, idx, (z, y, x) = patch_info
                
                # Check if patch intersects with this chunk
                patch_z_end = z + patch_size[0]
                if z < z_end and patch_z_end > z_start:
                    chunk_patches.append(patch_info)
            
            # Pre-compute weight map for this chunk
            chunk_shape = (z_end - z_start, max_y, max_x)
            weight_map = torch.zeros(chunk_shape, device=device, dtype=torch.float32)
            
            # Add weights from each patch to the weight map
            print(f"Pre-computing weight map for chunk [{z_start}:{z_end}]...")
            for path, weight_rank_idx, weight_idx, (weight_z, weight_y, weight_x) in chunk_patches:
                # Calculate patch bounds in global coordinates
                weight_patch_z_end = min(weight_z + patch_size[0], max_z)
                weight_patch_y_end = min(weight_y + patch_size[1], max_y)
                weight_patch_x_end = min(weight_x + patch_size[2], max_x)
                
                # Calculate intersection with current chunk
                weight_global_target_z_start = max(z_start, weight_z)
                weight_global_target_z_end = min(z_end, weight_patch_z_end)
                
                # Skip if no actual intersection
                if weight_global_target_z_end <= weight_global_target_z_start:
                    continue
                
                # Calculate chunk-relative coordinates for the intersection
                weight_target_z_start = weight_global_target_z_start - z_start
                weight_target_z_end = weight_global_target_z_end - z_start
                
                # We need to handle global Y and X coordinates differently
                # The weight map has the shape (z_chunk, full_y, full_x)
                # We should keep positions as global but make sure they don't exceed bounds
                
                # Y coordinates - already global, but need bounds checking
                weight_target_y_start = max(0, weight_y)  # Don't go below 0
                weight_target_y_end = min(max_y, weight_patch_y_end)  # Don't exceed max_y
                
                # X coordinates - already global, but need bounds checking
                weight_target_x_start = max(0, weight_x)  # Don't go below 0
                weight_target_x_end = min(max_x, weight_patch_x_end)  # Don't exceed max_x
                
                # Calculate patch-relative coordinates for the intersection
                weight_patch_z_start_rel = weight_global_target_z_start - weight_z
                weight_patch_z_end_rel = weight_global_target_z_end - weight_z
                
                # Calculate patch-relative Y coordinates for weight
                # If weight_y is negative (patch starts before volume), adjust weight_patch_y_start_rel
                weight_patch_y_start_rel = max(0, -weight_y) if weight_y < 0 else 0
                weight_patch_y_end_rel = weight_patch_y_end - weight_y
                
                # Calculate patch-relative X coordinates for weight
                # If weight_x is negative (patch starts before volume), adjust weight_patch_x_start_rel
                weight_patch_x_start_rel = max(0, -weight_x) if weight_x < 0 else 0
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
            
            # Process each patch that intersects with this chunk
            print(f"Processing chunk [{z_start}:{z_end}] - {len(chunk_patches)} patches...")
            
            # Count patches by source in this chunk
            if verbose:
                chunk_patches_by_source = {}
                for path, _, _, _ in chunk_patches:
                    if path not in chunk_patches_by_source:
                        chunk_patches_by_source[path] = 0
                    chunk_patches_by_source[path] += 1
                
                print(f"Chunk patches by source:")
                for path, count in chunk_patches_by_source.items():
                    print(f"  {path}: {count} patches")
            
            patch_counter = 0
            successful_blends = 0
            successful_by_source = {path: 0 for path in zarr_arrays.keys()}
            
            for path, rank_idx, idx, (z, y, x) in chunk_patches:
                # Debug progress periodically
                patch_counter += 1
                if patch_counter % 50 == 0 or patch_counter == 1 or patch_counter == len(chunk_patches):
                    print(f"Processing patch {patch_counter}/{len(chunk_patches)}")
                
                # Get patch data directly from zarr_arrays
                if path not in zarr_arrays or rank_idx not in zarr_arrays[path] or target_name not in zarr_arrays[path][rank_idx]:
                    print(f"WARNING: Missing patch array for path={path}, rank={rank_idx}, target={target_name}")
                    continue
                
                # Calculate patch bounds in global coordinates
                patch_z_end = min(z + patch_size[0], max_z)
                patch_y_end = min(y + patch_size[1], max_y)
                patch_x_end = min(x + patch_size[2], max_x)
                
                # Calculate intersection with current chunk
                global_target_z_start = max(z_start, z)
                global_target_z_end = min(z_end, patch_z_end)
                
                # Skip if no actual intersection
                if global_target_z_end <= global_target_z_start:
                    continue
                
                # Calculate chunk-relative coordinates for the intersection
                target_z_start = global_target_z_start - z_start
                target_z_end = global_target_z_end - z_start
                
                # We need to handle global Y and X coordinates differently
                # The output tensor and weight map have the shape (z_chunk, full_y, full_x)
                # We should keep positions as global but make sure they don't exceed bounds
                
                # Y coordinates - already global, but need bounds checking
                target_y_start = max(0, y)  # Don't go below 0
                target_y_end = min(max_y, patch_y_end)  # Don't exceed max_y
                
                # X coordinates - already global, but need bounds checking
                target_x_start = max(0, x)  # Don't go below 0
                target_x_end = min(max_x, patch_x_end)  # Don't exceed max_x
                
                # Calculate patch-relative coordinates for the intersection
                patch_z_start_rel = global_target_z_start - z
                patch_z_end_rel = global_target_z_end - z
                
                # Calculate patch-relative Y coordinates
                # If y is negative (patch starts before volume), adjust patch_y_start_rel
                patch_y_start_rel = max(0, -y) if y < 0 else 0
                patch_y_end_rel = patch_y_end - y
                
                # Calculate patch-relative X coordinates
                # If x is negative (patch starts before volume), adjust patch_x_start_rel
                patch_x_start_rel = max(0, -x) if x < 0 else 0
                patch_x_end_rel = patch_x_end - x
                
                # Skip if patch has invalid dimensions
                if (patch_y_end_rel <= patch_y_start_rel or
                        patch_x_end_rel <= patch_x_start_rel or
                        patch_z_end_rel <= patch_z_start_rel):
                    continue
                
                try:
                    # Load patch data directly
                    patch_array = zarr_arrays[path][rank_idx][target_name]
                    patch_data = patch_array[idx]
                    
                    # Check if this patch has embedded position channels
                    has_position_channels = patch_array.attrs.get('has_position_channels', False)
                    
                    # If position channels are embedded, skip them for blending
                    if has_position_channels:
                        # Extract only data channels (skip first 3 position channels)
                        data_tensor = torch.as_tensor(patch_data[3:], device=device).contiguous()
                    else:
                        # Use full patch
                        data_tensor = torch.as_tensor(patch_data, device=device).contiguous()
                    
                    # Blend the patch using the weighted blending function
                    blend_patch_weighted(
                        output_tensor, 
                        data_tensor,
                        blend_weights,
                        target_z_start, target_z_end,
                        target_y_start, target_y_end,
                        target_x_start, target_x_end,
                        patch_z_start_rel, patch_z_end_rel,
                        patch_y_start_rel, patch_y_end_rel,
                        patch_x_start_rel, patch_x_end_rel
                    )
                    
                    # Track successful blends
                    successful_blends += 1
                    successful_by_source[path] += 1
                    
                    # Help reduce memory pressure
                    del patch_data
                    del data_tensor
                except Exception as e:
                    print(f"Error blending patch (rank={rank_idx}, idx={idx}): {str(e)}")
                    if patch_counter <= 5:
                        import traceback
                        traceback.print_exc()
            
            # Normalize the output by dividing by the weight map
            print(f"Normalizing output for chunk [{z_start}:{z_end}]...")
            for c_idx in range(output_tensor.shape[0]):
                output_tensor[c_idx] /= weight_map

            # Only copy the valid part of the chunk (not the overlap regions) back to the output
            # Calculate indices for valid region within this chunk
            valid_offset = valid_start - z_start
            valid_length = valid_end - valid_start
            
            if verbose:
                print(f"  Saving valid region [{valid_start}:{valid_end}] (offset {valid_offset} in chunk)")
            
            # Extract the valid region from our processed chunk
            valid_data = output_tensor[:, valid_offset:valid_offset+valid_length].contiguous().cpu().numpy()
            
            # Copy normalized data back to memory-mapped arrays (only the valid part)
            output_arrays[target_name][:, valid_start:valid_end] = valid_data
            
            # Clean up GPU memory
            del output_tensor
            del weight_map
            
            # Only empty cache every few chunks to reduce overhead
            if chunk_idx % 4 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Report chunk processing time
            chunk_process_time = time.time() - chunk_process_start
            
            # Print summary of successful blends for this chunk
            print(f"Successful blends in this chunk: {successful_blends}/{len(chunk_patches)}")
            if verbose:
                print("Successful blends by source:")
                for temp_path, count in successful_by_source.items():
                    if count > 0:
                        print(f"  {temp_path}: {count} successful blends")
            
            print(f"Completed z-chunk {chunk_idx+1}/{total_chunks} in {chunk_process_time:.2f} seconds")
        
        # Finalize - convert to appropriate output format
        print(f"Finalizing output arrays for {target_name}...")
        finalize_arrays(
            output_arrays,
            count_arrays,
            final_arrays,
            target_name,
            threshold,
            save_probability_maps,
            verbose,
            chunk_memory_limit_gb
        )
    
    # Clean up temporary blending arrays
    cleanup_temp_files(temp_storage_path, verbose)
    
    # Clean up temp storage directories if requested
    if cleanup_temp:
        # Clean up all temp storage paths
        for temp_path in temp_storage_paths:
            cleanup_temp_storage(temp_path, verbose)
    
    return output_path


def finalize_arrays(output_arrays, count_arrays, final_arrays, target_name, threshold, save_probability_maps, verbose, chunk_memory_limit_gb=16.0):
    """
    Finalize the arrays by converting the accumulated sums to appropriate output format.
    
    Args:
        output_arrays: Dict of output arrays from blending
        count_arrays: Dict of count arrays for tracking weights
        final_arrays: Dict for final output arrays
        target_name: Name of the target being processed
        threshold: Optional threshold value (0-100) for binarizing
        save_probability_maps: Whether to save full probability maps
        verbose: Enable verbose output
        chunk_memory_limit_gb: Maximum GPU memory to use per chunk in GB
    """
    if verbose:
        print("Finalizing arrays...")
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define threshold value if provided
    threshold_val = None
    if threshold is not None:
        threshold_val = threshold / 100.0
    
    # Get array dimensions
    num_channels, z_max, y_max, x_max = output_arrays[target_name].shape
    
    # Identify segmentation types based on channel count
    is_multiclass = num_channels > 2
    is_binary = num_channels == 2
    
    # Determine processing modes
    compute_argmax = not save_probability_maps and is_multiclass
    extract_binary = is_binary
    
    # Calculate memory per slice for optimal chunk sizing
    mem_per_slice_mb = (num_channels * 2 * y_max * x_max) / (1024 * 1024)  # float16 = 2 bytes per element
    max_chunk_size_mb = chunk_memory_limit_gb * 1024 / 2  # Use only half the memory limit for finalization
    
    # Calculate optimal chunk size (in z slices)
    optimal_chunk_size = max(16, min(512, int(max_chunk_size_mb / mem_per_slice_mb)))
    chunk_size = (optimal_chunk_size // 16) * 16  # Round to multiple of 16
    
    if verbose:
        print(f"Finalization memory per slice: {mem_per_slice_mb:.2f} MB")
        print(f"Using finalization chunk size of {chunk_size} slices")
    
    z_range = tqdm(range(0, z_max, chunk_size), desc=f"Finalizing {target_name}")
    
    for z_start in z_range:
        z_end = min(z_start + chunk_size, z_max)
        
        output_chunk = output_arrays[target_name][:, z_start:z_end]
        count_chunk = count_arrays[target_name][z_start:z_end]

        # Convert to PyTorch tensors
        output_tensor = torch.as_tensor(output_chunk, device=device, dtype=torch.float16).contiguous()
        count_tensor = torch.as_tensor(count_chunk, device=device, dtype=torch.float16).contiguous()

        # With pre-computed weight maps, normalization is already done
        normalized_tensor = output_tensor
        
        if compute_argmax:
            # Multiclass case with argmax - compute argmax over channels
            argmax_tensor = torch.argmax(normalized_tensor, dim=0)
            
            # Reshape to add channel dimension and convert to uint8
            dest_tensor = torch.zeros((1,) + argmax_tensor.shape,
                                    dtype=torch.uint8,
                                    device=device)
            dest_tensor[0] = argmax_tensor
            
        elif extract_binary:
            # Binary segmentation case - use both channels and apply softmax
            
            # Apply softmax across channel dimension (0)
            softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)
            
            # Extract foreground probability (class 1)
            foreground_prob = softmax_tensor[1]
            
            if save_probability_maps:
                # When save_probability_maps is True, handle probability map and binary mask separately
                # Scale probabilities from [0,1] to [0,255] range for uint8 storage
                prob_tensor = (foreground_prob.to(torch.float32) * 255).to(torch.uint8)
                
                # Generate binary mask with argmax for maximum consistency
                binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                
                # Scale to 0 or 255 for clearer visualization
                binary_tensor = binary_mask * 255
                
                # Create a fresh tensor with exactly 2 channels
                dest_tensor = torch.zeros((2,) + binary_tensor.shape,
                                        dtype=torch.uint8,
                                        device=device)
                
                # Assign channels
                dest_tensor[0] = prob_tensor.clone()  # Channel 0: Probability map [0-255]
                dest_tensor[1] = binary_tensor.clone()  # Channel 1: Binary mask [0/255]
            else:
                # When save_probability_maps is False: save only binary mask
                binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                binary_tensor = binary_mask * 255  # Scale to 0/255
                dest_tensor = binary_tensor.unsqueeze(0)  # Add channel dimension
        
        else:
            # Standard case for multiclass segmentation (saving full probability maps)
            
            # Apply softmax to get proper probabilities
            softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)
            
            # Convert to uint8 (0-255 range)
            dest_tensor = (softmax_tensor * 255).to(torch.uint8)
            
            # Apply threshold if provided
            if threshold_val is not None:
                # Convert threshold to uint8 scale
                threshold_uint8 = int(threshold_val * 255)
                dest_tensor = (dest_tensor >= threshold_uint8).to(torch.uint8) * 255
        
        # Move to CPU for transfer to zarr
        dest_cpu = dest_tensor.contiguous().cpu().numpy()
        
        # Copy the results to the final array
        final_arrays[target_name][:, z_start:z_end] = dest_cpu
        
        # Cleanup
        del normalized_tensor
        if (z_start // chunk_size) % 4 == 0 and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_temp_files(temp_storage_path, verbose):
    """Clean up temporary files created during blending."""
    try:
        # Remove the temporary sum and count arrays
        sum_path = temp_storage_path.replace('.zarr', '_blend_sum.zarr')
        count_path = temp_storage_path.replace('.zarr', '_blend_count.zarr')
        
        if os.path.exists(sum_path):
            if verbose:
                print(f"Removing temporary sum array at {sum_path}")
            import shutil
            shutil.rmtree(sum_path)
        
        if os.path.exists(count_path):
            if verbose:
                print(f"Removing temporary count array at {count_path}")
            import shutil
            shutil.rmtree(count_path)
    except Exception as e:
        print(f"Warning: Error during cleanup of temporary files: {e}")


def cleanup_temp_storage(temp_storage_path, verbose):
    """
    Clean up the temporary storage directory and its parent directory after blending.
    
    Args:
        temp_storage_path: Path to the temporary storage zarr file
        verbose: Whether to print verbose output
    """
    try:
        # First, remove the temp.zarr directory
        if os.path.exists(temp_storage_path):
            if verbose:
                print(f"Removing temporary storage at {temp_storage_path}")
            import shutil
            shutil.rmtree(temp_storage_path)
        
        # Then, check if we should remove the parent temp directory
        temp_dir = os.path.dirname(temp_storage_path)
        
        # Look for blend_config.json in the parent directory
        config_path = os.path.join(temp_dir, "blend_config.json")
        if os.path.exists(config_path):
            if verbose:
                print(f"Removing blend configuration file at {config_path}")
            os.remove(config_path)
        
        # Check if the temp directory is empty and remove it if it is
        if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
            if verbose:
                print(f"Removing empty temporary directory at {temp_dir}")
            os.rmdir(temp_dir)
        elif os.path.exists(temp_dir) and verbose:
            print(f"Not removing temporary directory {temp_dir} as it still contains files")
            
    except Exception as e:
        print(f"Warning: Error during cleanup of temporary storage: {e}")


def load_blend_config(temp_storage_path: str) -> dict:
    """
    Load blending configuration from a JSON file in the temp storage directory.
    
    Args:
        temp_storage_path: Path to the temporary storage location (.zarr file)
        
    Returns:
        Dictionary with blending configuration parameters
    """
    # Get the parent directory of the temp zarr file
    # The temp_storage_path should point to temp.zarr, and blend_config.json
    # is stored in the parent directory (temp/)
    temp_dir = os.path.dirname(temp_storage_path)
    
    # If temp_dir ends with 'temp.zarr', strip the .zarr to get the directory
    if temp_dir.endswith('temp.zarr'):
        temp_dir = temp_dir[:-5]  # Remove '.zarr'
    
    # Construct the path to the config file
    config_path = os.path.join(temp_dir, "blend_config.json")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Warning: No blend configuration file found at {config_path}")
        # Try alternative path - maybe the temp_storage_path already includes 'temp'
        alt_config_path = os.path.join(os.path.dirname(temp_dir), "blend_config.json")
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
            print(f"Found blend configuration at alternative path: {config_path}")
        else:
            return {}
    
    try:
        # Load the config from the JSON file
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded blending configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading blend configuration: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blend saved patches into a final output')
    parser.add_argument('--temp_storage', type=str, required=True,
                        help='Path to the temporary storage location containing inference patches')
    parser.add_argument('--output', type=str,
                        help='Path to save the final blended output (zarr file). If not provided, output will be determined from config or temp storage path.')
    parser.add_argument('--patch_size', type=int, nargs=3,
                        help='Patch size used during inference as Z Y X. Only needed if not in config file.')
    parser.add_argument('--step_size', type=float,
                        help='Step size used during inference (as fraction of patch size)')
    parser.add_argument('--threshold', type=float, 
                        help='Optional threshold value (0-100) for binarizing the probability map')
    parser.add_argument('--no_probabilities', action='store_true',
                        help='Do not save probability maps, save argmax for multiclass segmentation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed output messages during blending')
    parser.add_argument('--edge_weight_boost', type=float,
                        help='Factor to boost Gaussian weights at patch edges')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite of existing output file without prompting')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Do not clean up temporary directories after blending')
    parser.add_argument('--no-include-parts', action='store_true',
                        help='Do not automatically include patches from all part-specific temp directories')
    parser.add_argument('--chunk-memory-limit', type=float, default=16.0,
                        help='Maximum GPU memory to use per chunk in GB (adjust based on available GPU VRAM)')
    
    args = parser.parse_args()
    
    # Try to load configuration from temp storage
    config = load_blend_config(args.temp_storage)
    
    # Determine parameters with precedence: CLI args > config file > defaults
    temp_storage_path = args.temp_storage
    
    # Output path: CLI arg > config > default
    output_path = args.output
    if output_path is None:
        output_path = config.get('output_path')
    if output_path is None:
        # Default to temp storage path but replace .zarr with _blended.zarr
        output_path = args.temp_storage.replace('.zarr', '_blended.zarr')
    
    # Patch size: CLI arg > config > required
    patch_size = args.patch_size
    if patch_size is None:
        patch_size = config.get('patch_size')
    if patch_size is None:
        raise ValueError("Patch size must be provided either via command line or in config file")
    else:
        patch_size = tuple(patch_size)
    
    # Step size: CLI arg > config > default
    step_size = args.step_size
    if step_size is None:
        step_size = config.get('step_size', 0.5)
    
    # Threshold: CLI arg > config > default (None)
    threshold = args.threshold
    if threshold is None:
        threshold = config.get('threshold')
    
    # Save probability maps: CLI arg overrides config > default (True)
    save_probability_maps = not args.no_probabilities
    if args.no_probabilities is False and 'save_probability_maps' in config:
        save_probability_maps = config.get('save_probability_maps', True)
    
    # Verbose: CLI arg > config > default (False)
    verbose = args.verbose
    if verbose is False and 'verbose' in config:
        verbose = config.get('verbose', False)
    
    # Edge weight boost: CLI arg > config > default (0)
    edge_weight_boost = args.edge_weight_boost
    if edge_weight_boost is None:
        edge_weight_boost = config.get('edge_weight_boost', 0)
        
    # Force overwrite flag is taken directly from CLI args
    force_overwrite = args.force
    
    # Cleanup flag is the inverse of no-cleanup
    cleanup_temp = not args.no_cleanup
    
    # Include all parts flag is the inverse of no-include-parts
    include_all_parts = not args.no_include_parts
    
    # Get chunk memory limit
    chunk_memory_limit_gb = args.chunk_memory_limit
    
    # Show the parameters being used
    print(f"Blending parameters:")
    print(f"  Temp storage path: {temp_storage_path}")
    print(f"  Output path: {output_path}")
    print(f"  Patch size: {patch_size}")
    print(f"  Step size: {step_size}")
    print(f"  Threshold: {threshold}")
    print(f"  Save probability maps: {save_probability_maps}")
    print(f"  Verbose: {verbose}")
    print(f"  Edge weight boost: {edge_weight_boost}")
    print(f"  Force overwrite: {force_overwrite}")
    print(f"  Clean up temp storage: {cleanup_temp}")
    print(f"  Include all part directories: {include_all_parts}")
    print(f"  Chunk memory limit: {chunk_memory_limit_gb} GB")
    
    # Execute blending
    blend_saved_patches(
        temp_storage_path=temp_storage_path,
        output_path=output_path,
        patch_size=patch_size,
        step_size=step_size,
        threshold=threshold,
        save_probability_maps=save_probability_maps,
        verbose=verbose,
        edge_weight_boost=edge_weight_boost,
        force_overwrite=force_overwrite,
        cleanup_temp=cleanup_temp,
        include_all_parts=include_all_parts,
        chunk_memory_limit_gb=chunk_memory_limit_gb
    )