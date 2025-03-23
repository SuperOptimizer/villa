import sys
import os
import numpy as np
import shutil
import threading
import queue
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zarr_temp_storage import ZarrTempStorage, zarr_writer_worker

def test_origin_position():
    """Test that positions at the origin (0,0,0) are properly handled"""
    # Clean up any existing test directory
    if os.path.exists("./temp_test"):
        shutil.rmtree("./temp_test")
        
    # Create temp directory
    os.makedirs("./temp_test", exist_ok=True)
    
    # Define volume shape for the test
    volume_shape = (5, 5, 5)  # (z, y, x)
    
    # Initialize storage with volume shape for spatial hashing
    storage = ZarrTempStorage(output_path="./temp_test", rank=0, world_size=1, 
                             volume_shape=volume_shape, verbose=True)
    storage.initialize()
    
    # Set expected patch counts
    target_name = "segmentation"
    patch_count = 5  # We'll store 5 patches
    expected_patches = patch_count + 1  # Add 1 for safety
    storage.set_expected_patch_count(target_name, expected_patches)
    
    # Create sample patches
    patch_shape = (2, 2, 2, 2)  # (channels, z, y, x) - small patches for test
    
    # Test positions, including the origin (0,0,0)
    positions = [
        (0, 0, 0),  # The origin - this is the critical test case
        (0, 0, 1),  # Adjacent to origin
        (0, 1, 0),  # Another position
        (1, 1, 1),  # Diagonal from origin
        (2, 2, 2)   # Another position
    ]
    
    # Store patches
    for i, position in enumerate(positions):
        # Create a patch with a unique value
        patch = np.ones(patch_shape, dtype=np.float32) * (i + 1)
        
        # Store the patch
        idx = storage.store_patch(patch, position, target_name)
        print(f"Stored patch {i+1} at position {position}, got index {idx}")
    
    # Finalize target
    storage.finalize_target(target_name)
    
    # Get patches back 
    patches_array, positions_array = storage.get_all_patches(0, target_name)
    
    # Verify data
    print(f"\nRetrieved {patches_array.shape[0]} patches")
    print(f"Retrieved positions: {positions_array}")
    
    # Make sure all our original positions are in the retrieved positions
    retrieved_positions = [tuple(pos) for pos in positions_array]
    for pos in positions:
        assert pos in retrieved_positions, f"Position {pos} not found in retrieved positions"
    
    # Make sure our origin position (0,0,0) is included
    assert (0, 0, 0) in retrieved_positions, "Origin position (0,0,0) was not correctly retrieved"
    
    # Clean up
    storage.cleanup()
    print("\nTest successful: Origin position (0,0,0) is properly handled")
    
if __name__ == "__main__":
    print("Testing that positions at (0,0,0) are properly handled")
    test_origin_position()