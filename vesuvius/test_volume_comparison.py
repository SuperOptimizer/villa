#!/usr/bin/env python

import time
import numpy as np
from data.volume import Volume

def test_volume(use_fsspec=False):
    print(f"\n{'='*60}")
    print(f"Testing Volume with {'fsspec' if use_fsspec else 'TensorStore'}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create Volume instance
    volume = Volume(
        type="scroll",
        scroll_id=1,
        energy=54,  # Default energy for scroll 1
        resolution=7.91,  # Default resolution for scroll 1
        cache=True,
        verbose=True,
        use_fsspec=use_fsspec
    )
    
    init_time = time.time() - start_time
    print(f"Volume initialized in {init_time:.2f} seconds")
    print(f"Shape: {volume.shape(0)}")
    
    # Define patch size
    patch_size = (64, 64, 64)
    
    # Locations to extract patches
    z_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    y_pos = 0
    x_pos = 0
    
    print(f"\nTesting extraction of {len(z_steps)} patches of size {patch_size}...")
    print(f"Using {'fsspec' if use_fsspec else 'TensorStore'}\n")
    
    # Extract patches at different z positions
    total_extraction_time = 0
    for i, z in enumerate(z_steps):
        print(f"Extracting patch {i+1}/{len(z_steps)} at position ({z}, {y_pos}, {x_pos})...")
        
        patch_start = time.time()
        
        # Extract the patch directly using __getitem__
        patch = volume[z:z+patch_size[0], y_pos:y_pos+patch_size[1], x_pos:x_pos+patch_size[2]]
        
        patch_time = time.time() - patch_start
        total_extraction_time += patch_time
        
        print(f"  Patch extracted in {patch_time:.2f} seconds")
        print(f"  Patch shape: {patch.shape}, min={patch.min():.4f}, max={patch.max():.4f}")
        print()
    
    print(f"\nSummary for {'fsspec' if use_fsspec else 'TensorStore'}:")
    print(f"Volume initialization: {init_time:.2f} seconds")
    print(f"Total extraction time for {len(z_steps)} patches: {total_extraction_time:.2f} seconds")
    print(f"Average extraction time per patch: {total_extraction_time/len(z_steps):.2f} seconds")
    
    return {
        "implementation": "fsspec" if use_fsspec else "TensorStore",
        "init_time": init_time,
        "total_extraction_time": total_extraction_time,
        "avg_patch_time": total_extraction_time/len(z_steps)
    }

def main():
    print("Running comparative test between TensorStore and fsspec implementations")
    print("Testing 10 patches of size (64, 64, 64) from Scroll 1 volume")
    print("This will help evaluate which implementation is faster for your use case")
    
    # First test with TensorStore (default)
    ts_results = test_volume(use_fsspec=False)
    
    # Then test with fsspec
    fsspec_results = test_volume(use_fsspec=True)
    
    # Compare results
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Initialization time:  TensorStore: {ts_results['init_time']:.2f}s | fsspec: {fsspec_results['init_time']:.2f}s")
    print(f"Total extraction:     TensorStore: {ts_results['total_extraction_time']:.2f}s | fsspec: {fsspec_results['total_extraction_time']:.2f}s")
    print(f"Average patch time:   TensorStore: {ts_results['avg_patch_time']:.2f}s | fsspec: {fsspec_results['avg_patch_time']:.2f}s")
    
    # Determine which is faster for extraction
    if fsspec_results['avg_patch_time'] < ts_results['avg_patch_time']:
        speedup = ts_results['avg_patch_time'] / fsspec_results['avg_patch_time']
        print(f"\nfsspec is {speedup:.2f}x faster for patch extraction!")
        print("Recommendation: Use --use_fsspec for faster inference")
    else:
        speedup = fsspec_results['avg_patch_time'] / ts_results['avg_patch_time']
        print(f"\nTensorStore is {speedup:.2f}x faster for patch extraction!")
        print("Recommendation: Don't use --use_fsspec for this data")

if __name__ == "__main__":
    main()