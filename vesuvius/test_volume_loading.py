#!/usr/bin/env python

import time
import numpy as np
from data.volume import Volume

def main():
    print("Creating Volume instance for scroll 1...")
    start_time = time.time()
    
    # Create Volume instance
    volume = Volume(
        type="scroll",
        scroll_id=1,
        energy=54,  # Default energy for scroll 1
        resolution=7.91,  # Default resolution for scroll 1
        cache=True,
        verbose=True
    )
    
    init_time = time.time() - start_time
    print(f"Volume initialized in {init_time:.2f} seconds")
    print(f"Shape: {volume.shape(0)}")
    
    # Define patch size similar to what's used in inference
    patch_size = (64, 64, 64)  # Smaller than the 192x192x192 to test faster
    
    # Use the same locations as the direct zarr test for comparison
    z_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    y_pos = 0
    x_pos = 0
    
    print(f"\nTesting extraction of {len(z_steps)} patches of size {patch_size}...\n")
    
    # Extract patches at different z positions
    total_extraction_time = 0
    for i, z in enumerate(z_steps):
        print(f"Extracting patch {i+1}/{len(z_steps)} at position ({z}, {y_pos}, {x_pos})...")
        
        patch_start = time.time()
        
        # Extract the patch
        patch = np.zeros(patch_size, dtype=np.float32)
        for dz in range(patch_size[0]):
            if z + dz < volume.shape(0)[0]:
                for dy in range(patch_size[1]):
                    if y_pos + dy < volume.shape(0)[1]:
                        try:
                            slice_data = volume[z + dz, y_pos + dy, x_pos:x_pos + patch_size[2]]
                            patch[dz, dy, :min(patch_size[2], len(slice_data))] = slice_data[:min(patch_size[2], len(slice_data))]
                        except Exception as e:
                            print(f"Error extracting slice: {e}")
        
        patch_time = time.time() - patch_start
        total_extraction_time += patch_time
        
        print(f"  Patch extracted in {patch_time:.2f} seconds")
        print(f"  Patch shape: {patch.shape}, min={patch.min():.4f}, max={patch.max():.4f}")
        print()
    
    print(f"\nSummary:")
    print(f"Volume initialization: {init_time:.2f} seconds")
    print(f"Total extraction time for {len(z_steps)} patches: {total_extraction_time:.2f} seconds")
    print(f"Average extraction time per patch: {total_extraction_time/len(z_steps):.2f} seconds")

if __name__ == "__main__":
    main()