import numpy as np
import os
import re
import json
import zarr
import multiprocessing as mp
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
import torch
from functools import partial
import numcodecs
from concurrent.futures import ProcessPoolExecutor
import math


# --- Gaussian Map Generation ---
def generate_gaussian_map(patch_size: tuple, sigma_scale: float = 8.0, dtype=torch.float32) -> torch.Tensor:
    """
    Generates a Gaussian importance map for a given patch size.
    Weights decay from the center towards the edges.
    Shape: (1, pZ, pY, pX) for easy broadcasting.
    """
    pZ, pY, pX = patch_size
    tmp = torch.zeros(patch_size, dtype=dtype)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i / sigma_scale for i in patch_size]

    tmp[tuple(center_coords)] = 1

    tmp_np = tmp.cpu().numpy()
    gaussian_map_np = gaussian_filter(tmp_np, sigmas, 0, mode='constant', cval=0)
    gaussian_map = torch.from_numpy(gaussian_map_np)
    # Safeguard against division by zero
    gaussian_map /= max(gaussian_map.max().item(), 1e-12)
    gaussian_map = gaussian_map.reshape(1, pZ, pY, pX)
    gaussian_map = torch.clamp(gaussian_map, min=0)
    
    print(
        f"Generated Gaussian map with shape {gaussian_map.shape}, min: {gaussian_map.min().item():.4f}, max: {gaussian_map.max().item():.4f}")
    return gaussian_map


# --- Tile Processing Worker Function ---
def process_tile(tile_info, parent_dir, output_path, weights_path, gaussian_map, 
                patch_size, part_files):
    """
    Process a spatial tile of the volume, handling all patches that intersect with this tile.
    
    Args:
        tile_info: Dictionary with tile boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        parent_dir: Directory containing part files
        output_path: Path to output zarr
        weights_path: Path to weights zarr
        gaussian_map: Pre-computed Gaussian map
        patch_size: Size of patches (pZ, pY, pX)
        part_files: Dictionary of part files
    """
    # Extract tile boundaries
    z_start, z_end = tile_info['z_start'], tile_info['z_end']
    y_start, y_end = tile_info['y_start'], tile_info['y_end']
    x_start, x_end = tile_info['x_start'], tile_info['x_end']
    
    pZ, pY, pX = patch_size
    
    # Convert gaussian map to numpy for efficient processing
    gaussian_map_np = gaussian_map.numpy()
    gaussian_map_spatial_np = gaussian_map_np[0]  # Shape (pZ, pY, pX)
    
    # Open output and weights zarr arrays for this tile
    output_store = zarr.open(output_path, mode='r+')
    weights_store = zarr.open(weights_path, mode='r+')
    
    # Create local accumulators for this tile - initialize with zeros
    # Shape: (C, tile_z, tile_y, tile_x)
    num_classes = output_store.shape[0]
    tile_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
    
    # Initialize accumulators
    tile_logits = np.zeros(tile_shape, dtype=np.float32)
    tile_weights = np.zeros(weights_shape, dtype=np.float32)
    
    # Track which patches intersect with this tile
    patches_processed = 0
    
    # Process each part file sequentially
    for part_id in part_files:
        logits_path = part_files[part_id]['logits']
        coords_path = part_files[part_id]['coordinates']
        
        # Open part zarr stores
        coords_store = zarr.open(coords_path, mode='r')
        logits_store = zarr.open(logits_path, mode='r')
        
        # Read all coordinates for this part
        coords_np = coords_store[:]
        num_patches_in_part = coords_np.shape[0]
        
        # Process patches that intersect with this tile
        for patch_idx in range(num_patches_in_part):
            z, y, x = coords_np[patch_idx].tolist()
            
            # Check if this patch intersects with our tile
            if (z + pZ <= z_start or z >= z_end or
                y + pY <= y_start or y >= y_end or
                x + pX <= x_start or x >= x_end):
                continue  # Skip patches that don't intersect with this tile
                
            # Calculate intersection between patch and tile
            iz_start = max(z, z_start) - z_start
            iz_end = min(z + pZ, z_end) - z_start
            iy_start = max(y, y_start) - y_start
            iy_end = min(y + pY, y_end) - y_start
            ix_start = max(x, x_start) - x_start
            ix_end = min(x + pX, x_end) - x_start
            
            # Patch internal coordinates (for reading from logits)
            pz_start = max(z_start - z, 0)
            pz_end = pZ - max(z + pZ - z_end, 0)
            py_start = max(y_start - y, 0)
            py_end = pY - max(y + pY - y_end, 0)
            px_start = max(x_start - x, 0)
            px_end = pX - max(x + pX - x_end, 0)
            
            # Read patch logits (only the portion that intersects with our tile)
            patch_slice = (
                slice(None),  # All classes
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            )
            
            # Read the specific portion of the logits for this patch
            logit_patch = logits_store[patch_idx][patch_slice]
            
            # Get corresponding weights
            weight_patch = gaussian_map_spatial_np[
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            ]
            
            # Apply weights to logits (broadcasting along class dimension)
            weighted_patch = logit_patch * weight_patch[np.newaxis, :, :, :]
            
            # Accumulate into local arrays
            tile_logits[
                :,  # All classes
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ] += weighted_patch
            
            tile_weights[
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ] += weight_patch
            
            patches_processed += 1
    
    # Write accumulated data back to main arrays
    if patches_processed > 0:
        output_slice = (
            slice(None),  # All classes
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        weight_slice = (
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        # Lock for potentially concurrent writes
        output_store[output_slice] = tile_logits
        weights_store[weight_slice] = tile_weights
    
    return {
        'tile': tile_info,
        'patches_processed': patches_processed
    }


# --- Normalization Worker Function ---
def normalize_tile(tile_info, output_path, weights_path, epsilon=1e-8):
    """
    Normalize a spatial tile by dividing accumulated logits by weights.
    
    Args:
        tile_info: Dictionary with tile boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        output_path: Path to output zarr
        weights_path: Path to weights zarr
        epsilon: Small value to avoid division by zero
    """
    # Extract tile boundaries
    z_start, z_end = tile_info['z_start'], tile_info['z_end']
    y_start, y_end = tile_info['y_start'], tile_info['y_end']
    x_start, x_end = tile_info['x_start'], tile_info['x_end']
    
    # Open zarr arrays
    output_store = zarr.open(output_path, mode='r+')
    weights_store = zarr.open(weights_path, mode='r')
    
    # Define slices for reading data
    output_slice = (
        slice(None),  # All classes
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    weight_slice = (
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    # Read data
    logits = output_store[output_slice]
    weights = weights_store[weight_slice]
    
    # Use in-place division to save memory (no duplication)
    # Initialize result array
    normalized = np.zeros_like(logits)
    
    # Efficient, consistent division: divide only where weights > 0
    np.divide(logits, weights[np.newaxis, :, :, :] + epsilon, 
              out=normalized, where=weights[np.newaxis, :, :, :] > 0)
    
    # Write normalized data back
    output_store[output_slice] = normalized
    
    return {
        'tile': tile_info,
        'normalized_voxels': np.prod(normalized.shape)
    }


# --- Utility Functions ---
def calculate_tiles(volume_shape, patch_size, num_classes=1, output_chunks=None):
    """
    Calculate optimal tile sizes that align with zarr chunk boundaries.
    
    Args:
        volume_shape: Shape of the volume (Z, Y, X)
        patch_size: Size of patches (pZ, pY, pX)
        num_classes: Number of output classes
        output_chunks: Spatial chunk size for the output zarr (z_chunk, y_chunk, x_chunk)
        
    Returns:
        List of tile dictionaries with boundaries
    """
    # Get volume dimensions
    Z, Y, X = volume_shape
    
    # If no chunks specified, use patch size as base
    if output_chunks is None:
        # Default chunk sizes based on patch size
        pZ, pY, pX = patch_size
        z_chunk, y_chunk, x_chunk = pZ, pY, pX
    else:
        # Use the provided chunks (these should be the spatial dimensions only)
        z_chunk, y_chunk, x_chunk = output_chunks
    
    # Determine tile sizes that are multiples of chunk sizes
    # We'll use reasonable defaults rather than complex memory calculations
    tile_z = z_chunk * max(1, min(32, Z // z_chunk))  # Use up to 32 chunks per tile
    tile_y = y_chunk * max(1, min(32, Y // y_chunk))
    tile_x = x_chunk * max(1, min(32, X // x_chunk))
    
    # Generate tiles aligned to chunk boundaries
    tiles = []
    for z_start in range(0, Z, tile_z):
        for y_start in range(0, Y, tile_y):
            for x_start in range(0, X, tile_x):
                z_end = min(z_start + tile_z, Z)
                y_end = min(y_start + tile_y, Y)
                x_end = min(x_start + tile_x, X)
                
                tiles.append({
                    'z_start': z_start, 'z_end': z_end,
                    'y_start': y_start, 'y_end': y_end,
                    'x_start': x_start, 'x_end': x_end
                })
    
    return tiles


# --- Main Merging Function ---
def merge_inference_outputs(
        parent_dir: str,
        output_path: str,
        weight_accumulator_path: str = None,  # Optional: Path for weights, default is temp
        sigma_scale: float = 8.0,
        chunk_size: tuple = None,  # Spatial chunk size (Z, Y, X) for output
        num_workers: int = None,  # Number of worker processes to use
        compression_level: int = 1,  # Compression level (0-9, 0=none)
        delete_weights: bool = True,  # Delete weight accumulator after merge
        verbose: bool = True):
    """
    Merges partial inference results with Gaussian blending using parallel processing.

    Args:
        parent_dir: Directory containing logits_part_X.zarr and coordinates_part_X.zarr.
        output_path: Path for the final merged Zarr store.
        weight_accumulator_path: Path for the temporary weight accumulator Zarr.
                                  If None, defaults to output_path + "_weights.zarr".
        sigma_scale: Determines the sigma for the Gaussian map (patch_size / sigma_scale).
        chunk_size: Spatial chunk size (Z, Y, X) for output Zarr stores.
                    If None, will use patch_size as a starting point.
        num_workers: Number of worker processes to use.
                     If None, defaults to CPU_COUNT - 1.
        compression_level: Zarr compression level (0-9, 0=none)
        delete_weights: Whether to delete the weight accumulator Zarr after completion.
        verbose: Print progress messages.
    """
    # Disable Blosc threading to avoid deadlocks when used with multiprocessing
    numcodecs.blosc.use_threads = False
    if weight_accumulator_path is None:
        base, _ = os.path.splitext(output_path)
        weight_accumulator_path = f"{base}_weights.zarr"
    
    # Configure process pool size
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} worker processes")
        
    # --- 1. Discover Parts ---
    part_files = {}
    part_pattern = re.compile(r"(logits|coordinates)_part_(\d+)\.zarr")
    print(f"Scanning for parts in: {parent_dir}")
    for filename in os.listdir(parent_dir):
        match = part_pattern.match(filename)
        if match:
            file_type, part_id_str = match.groups()
            part_id = int(part_id_str)
            if part_id not in part_files:
                part_files[part_id] = {}
            part_files[part_id][file_type] = os.path.join(parent_dir, filename)

    part_ids = sorted(part_files.keys())
    if not part_ids:
        raise FileNotFoundError(f"No inference parts found in {parent_dir}")
    print(f"Found parts: {part_ids}")

    # Validate that all parts have both files
    for part_id in part_ids:
        if 'logits' not in part_files[part_id] or 'coordinates' not in part_files[part_id]:
            raise FileNotFoundError(f"Part {part_id} is missing logits or coordinates Zarr.")

    # --- 2. Read Metadata (from first available part) ---
    first_part_id = part_ids[0]  # Use the first available part_id 
    print(f"Reading metadata from part {first_part_id}...")
    part0_logits_path = part_files[first_part_id]['logits']
    try:
        # Use zarr.open instead of tensorstore
        part0_logits_store = zarr.open(part0_logits_path, mode='r')

        # Read .zattrs file directly for metadata
        zattrs_path = os.path.join(part0_logits_path, '.zattrs')
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                meta_attrs = json.load(f)

            patch_size = tuple(meta_attrs['patch_size'])  # Already a list in the file
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])  # MUST exist
            num_classes = part0_logits_store.shape[1]  # (N, C, pZ, pY, pX) -> C
        else:
            # Try to infer from the array shape if .zattrs is missing
            part0_coords_path = part_files[first_part_id]['coordinates']
            coords_store = zarr.open(part0_coords_path, mode='r')
            # First patch's logits shape should be (C, pZ, pY, pX)
            first_patch_shape = part0_logits_store[0].shape
            num_classes = first_patch_shape[0]
            patch_size = first_patch_shape[1:]
            
            # Get first and last patch centers to estimate volume size
            coords_data = coords_store[:]
            min_coords = np.min(coords_data, axis=0)
            max_coords = np.max(coords_data, axis=0)
            estimated_shape = tuple((max_coords + np.array(patch_size) - min_coords).astype(int))
            
            original_volume_shape = estimated_shape
            print("WARNING: No .zattrs file found. Using estimated volume shape from coordinates.")
            
        print(f"  Patch Size: {patch_size}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Original Volume Shape (Z,Y,X): {original_volume_shape}")
    except Exception as e:
        print("\nERROR: Failed to read metadata from part 0 logits attributes.")
        print("Ensure 'patch_size' and 'original_volume_shape' were saved during inference.")
        raise e

    # --- 3. Prepare Output Stores ---
    output_shape = (num_classes, *original_volume_shape)  # (C, D, H, W)
    weights_shape = original_volume_shape  # (D, H, W)

    # Use patch_size as the basis for chunking if not specified
    if chunk_size is None or any(c == 0 for c in (chunk_size if chunk_size else [0, 0, 0])):
        # Default to patch size but ensure chunks aren't too large
        max_chunk_size = 256  # Keep below this size for more efficient I/O
        output_chunks = (
            1,  # One class at a time
            min(patch_size[0], max_chunk_size),
            min(patch_size[1], max_chunk_size),
            min(patch_size[2], max_chunk_size)
        )
        weights_chunks = output_chunks[1:]
        if verbose:
            print(f"  Using optimized chunk_size {output_chunks[1:]} based on patch_size")
    else:
        output_chunks = (1, *chunk_size)  # One class at a time, user-specified spatial chunks
        weights_chunks = chunk_size
        if verbose:
            print(f"  Using specified chunk_size {chunk_size}")

    # Setup compression
    if compression_level > 0:
        compressor = numcodecs.Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=numcodecs.blosc.SHUFFLE
        )
    else:
        compressor = None

    print(f"Creating final output store: {output_path}")
    print(f"  Shape: {output_shape}, Chunks: {output_chunks}")
    
    # Create zarr stores with compression (use open instead of save_array to avoid materializing arrays)
    zarr.open(
        output_path,
        mode='w',
        shape=output_shape,
        chunks=output_chunks,
        compressor=compressor,
        dtype=np.float32,
        fill_value=0
    )
    
    print(f"Creating weight accumulator store: {weight_accumulator_path}")
    print(f"  Shape: {weights_shape}, Chunks: {weights_chunks}")
    
    zarr.open(
        weight_accumulator_path,
        mode='w',
        shape=weights_shape,
        chunks=weights_chunks,
        compressor=compressor,
        dtype=np.float32,
        fill_value=0
    )

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)
    # Make sure it's on CPU and convert to numpy
    gaussian_map = gaussian_map.cpu()

    # --- 5. Calculate Processing Tiles ---
    tiles = calculate_tiles(
        original_volume_shape,
        patch_size,
        num_classes,
        output_chunks=output_chunks[1:]  # Skip the class dimension from output_chunks
    )
    
    print(f"Divided volume into {len(tiles)} tiles for parallel processing")
    
    # --- 6. Process Tiles in Parallel ---
    print("\n--- Accumulating Weighted Patches ---")
    
    # Create a partial function with fixed arguments
    process_tile_partial = partial(
        process_tile,
        parent_dir=parent_dir,
        output_path=output_path,
        weights_path=weight_accumulator_path,
        gaussian_map=gaussian_map,
        patch_size=patch_size,
        part_files=part_files
    )
    
    # Process tiles in parallel
    total_patches_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_tile = {executor.submit(process_tile_partial, tile): tile for tile in tiles}
        
        # Use as_completed for better progress tracking and early error detection
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(future_to_tile),
            total=len(tiles),
            desc="Processing Tiles",
            disable=not verbose
        ):
            try:
                result = future.result()
                total_patches_processed += result['patches_processed']
            except Exception as e:
                print(f"Error processing tile: {e}")
                raise e
    
    print(f"\nAccumulation complete. Processed {total_patches_processed} patches total.")
    
    # --- 7. Normalize in Parallel ---
    print("\n--- Normalizing Output ---")
    
    # Create a partial function with fixed arguments
    normalize_tile_partial = partial(
        normalize_tile,
        output_path=output_path,
        weights_path=weight_accumulator_path
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(normalize_tile_partial, tile): tile for tile in tiles}
        
        # Use as_completed for better progress tracking and early error detection
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(futures),
            total=len(tiles),
            desc="Normalizing Tiles",
            disable=not verbose
        ):
            try:
                future.result()  # Check for exceptions
            except Exception as e:
                print(f"Error normalizing tile: {e}")
                raise e
    
    print("\nNormalization complete.")
    
    # --- 8. Save Metadata ---
    # Add metadata to the output zarr
    output_zarr = zarr.open(output_path, mode='r+')
    if hasattr(output_zarr, 'attrs'):
        output_zarr.attrs['patch_size'] = patch_size
        output_zarr.attrs['original_volume_shape'] = original_volume_shape
        output_zarr.attrs['sigma_scale'] = sigma_scale
    
    # --- 9. Cleanup ---
    if delete_weights:
        print(f"Deleting weight accumulator: {weight_accumulator_path}")
        try:
            import shutil
            if os.path.exists(weight_accumulator_path):
                shutil.rmtree(weight_accumulator_path)
                print(f"Successfully deleted weight accumulator")
        except Exception as e:
            print(f"Warning: Failed to delete weight accumulator: {e}")
            print(f"You may need to delete it manually: {weight_accumulator_path}")

    print(f"\n--- Merging Finished ---")
    print(f"Final merged output saved to: {output_path}")


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.blend command line tool."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Merge partial inference outputs with Gaussian blending (optimized parallel version).')
    parser.add_argument('parent_dir', type=str,
                        help='Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr)')
    parser.add_argument('output_path', type=str,
                        help='Path for the final merged Zarr output file.')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Optional path for the temporary weight accumulator Zarr. Defaults to <output_path>_weights.zarr')
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, optimized size will be used.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes. Default: CPU_COUNT - 1')
    parser.add_argument('--compression_level', type=int, default=1, choices=range(10),
                        help='Compression level (0-9, 0=none). Default: 1')
    parser.add_argument('--keep_weights', action='store_true',
                        help='Do not delete the weight accumulator Zarr after merging.')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose progress messages (tqdm bars still show).')

    args = parser.parse_args()

    # Parse chunk_size if provided
    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")

    try:
        merge_inference_outputs(
            parent_dir=args.parent_dir,
            output_path=args.output_path,
            weight_accumulator_path=args.weights_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            delete_weights=not args.keep_weights,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"\n--- Blending Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
