#!/usr/bin/env python

import time
import numpy as np
from data.volume import Volume

def main():
    print("Testing Volume class with fsspec")
    print("="*60)
    
    start_time = time.time()
    
    # Create Volume with fsspec
    print("Creating Volume with fsspec...")
    volume = Volume(
        type="scroll1",
        scroll_id=1,
        energy=54,
        resolution=7.91,
        cache=True,
        normalize=False,
        verbose=True,
        use_fsspec=True
    )
    
    init_time = time.time() - start_time
    print(f"Volume initialized in {init_time:.2f} seconds")
    
    # Print metadata
    print("\nVolume metadata:")
    volume.meta()
    
    # Try a small extraction
    print("\nTesting extraction of data...")
    patch_size = (64, 64, 64)
    
    # Test a single extraction
    z, y, x = 0, 0, 0
    print(f"Extracting data at position ({z}, {y}, {x}) with size {patch_size}...")
    
    extraction_start = time.time()
    
    # Extract slice by slice for each dimension
    patch_data = np.zeros(patch_size, dtype=np.float32)
    print("Extracting by slices...")
    
    try:
        # Extract one slice at a time and print progress
        for i in range(patch_size[0]):
            print(f"Extracting slice {i+1}/{patch_size[0]}...")
            slice_start = time.time()
            
            # Get a single Z slice
            try:
                slice_data = volume[z + i, y:y+patch_size[1], x:x+patch_size[2]]
                patch_data[i, :, :] = slice_data
                print(f"  Slice shape: {slice_data.shape}, min={slice_data.min():.4f}, max={slice_data.max():.4f}")
            except Exception as e:
                print(f"ERROR extracting slice: {e}")
            
            slice_time = time.time() - slice_start
            print(f"  Slice extracted in {slice_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error during extraction: {e}")
    
    extraction_time = time.time() - extraction_start
    print(f"\nExtraction completed in {extraction_time:.2f} seconds")
    
    print("\nVolume with fsspec test completed!")

if __name__ == "__main__":
    main()