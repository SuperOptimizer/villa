#!/usr/bin/env python

import time
import numpy as np
from data.vc_dataset import VCDataset

def main():
    print("Testing VCDataset with fsspec")
    print("="*60)
    
    # Define output targets (similar to what inference.py uses)
    output_targets = [{
        "name": "segmentation",
        "channels": 2,
        "activation": "sigmoid",
        "nnunet_output_channels": 2
    }]
    
    patch_size = (64, 64, 64)
    
    start_time = time.time()
    
    # Create VCDataset with fsspec
    print("Creating VCDataset with fsspec...")
    dataset = VCDataset(
        input_path="scroll1",  # This will be interpreted as scroll_id
        targets=output_targets,
        patch_size=patch_size,
        input_format='volume',
        scroll_id=1,
        energy=54,
        resolution=7.91,
        verbose=True,
        use_fsspec=True
    )
    
    init_time = time.time() - start_time
    print(f"VCDataset initialized in {init_time:.2f} seconds")
    print(f"Dataset has {len(dataset)} patches")
    print(f"Input shape: {dataset.input_shape}")
    
    # Extract a few patches to test
    num_patches_to_test = 3
    print(f"\nExtracting {num_patches_to_test} patches to test...")
    
    total_extraction_time = 0
    for i in range(min(num_patches_to_test, len(dataset))):
        print(f"Extracting patch {i+1}/{num_patches_to_test}...")
        
        patch_start = time.time()
        
        # Get patch from dataset
        sample = dataset[i]
        patch = sample["data"]
        
        patch_time = time.time() - patch_start
        total_extraction_time += patch_time
        
        print(f"  Patch extracted in {patch_time:.2f} seconds")
        print(f"  Patch position: {sample['pos']}")
        print(f"  Patch shape: {patch.shape}, min={patch.min().item():.4f}, max={patch.max().item():.4f}")
        print()
    
    if num_patches_to_test > 0:
        print(f"\nExtraction Summary:")
        print(f"Total extraction time for {num_patches_to_test} patches: {total_extraction_time:.2f} seconds")
        print(f"Average extraction time per patch: {total_extraction_time/num_patches_to_test:.2f} seconds")
    
    print("\nVCDataset with fsspec test completed successfully!")

if __name__ == "__main__":
    main()