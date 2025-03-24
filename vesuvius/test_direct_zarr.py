#!/usr/bin/env python

import time
import numpy as np
import zarr
import fsspec

def main():
    # URL for the zarr store
    zarr_url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/"
    
    print(f"Opening zarr store directly from {zarr_url}...")
    start_time = time.time()
    
    # Open the zarr store with fsspec
    fs = fsspec.filesystem("http")
    
    # Use zarr directly without consolidated metadata
    zarr_map = fsspec.mapping.FSMap(zarr_url, fs)
    store = zarr.open(zarr_map, mode='r')
    
    # Try accessing the highest resolution (group '0')
    try:
        # Try to access as a multi-resolution zarr
        url_with_group = f"{zarr_url}/0"
        group_map = fsspec.mapping.FSMap(url_with_group, fs)
        zarr_array = zarr.open(group_map, mode='r')
        print("Using highest resolution (group '0')")
    except Exception as e:
        print(f"Error accessing group '0': {e}")
        print("Trying to use the root array directly")
        zarr_array = store

    init_time = time.time() - start_time
    print(f"Zarr store opened in {init_time:.2f} seconds")
    print(f"Shape: {zarr_array.shape}")
    
    # Define patch size similar to what's used in inference
    patch_size = (64, 64, 64)  # Smaller than the 192x192x192 to test faster
    
    # Locations to extract patches
    z_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    y_pos = 0
    x_pos = 0
    
    print(f"\nTesting extraction of {len(z_steps)} patches of size {patch_size}...\n")
    
    # Extract patches at different z positions
    total_extraction_time = 0
    for i, z in enumerate(z_steps):
        print(f"Extracting patch {i+1}/{len(z_steps)} at position ({z}, {y_pos}, {x_pos})...")
        
        patch_start = time.time()
        
        # Extract the patch directly from zarr
        if len(zarr_array.shape) == 3:  # ZYX format
            patch = zarr_array[z:z+patch_size[0], y_pos:y_pos+patch_size[1], x_pos:x_pos+patch_size[2]]
        elif len(zarr_array.shape) == 4:  # CZYX format
            patch = zarr_array[0, z:z+patch_size[0], y_pos:y_pos+patch_size[1], x_pos:x_pos+patch_size[2]]
        
        patch_time = time.time() - patch_start
        total_extraction_time += patch_time
        
        print(f"  Patch extracted in {patch_time:.2f} seconds")
        print(f"  Patch shape: {patch.shape}, min={patch.min():.4f}, max={patch.max():.4f}")
        print()
    
    print(f"\nSummary:")
    print(f"Zarr initialization: {init_time:.2f} seconds")
    print(f"Total extraction time for {len(z_steps)} patches: {total_extraction_time:.2f} seconds")
    print(f"Average extraction time per patch: {total_extraction_time/len(z_steps):.2f} seconds")

if __name__ == "__main__":
    main()