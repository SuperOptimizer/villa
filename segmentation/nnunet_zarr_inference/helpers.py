import numpy as np


def generate_positions(min_val, max_val, patch_size, step_size):
    """
    Generate positions for patch extraction with step size.
    
    Args:
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        patch_size: Size of the patch in this dimension
        step_size: Step size for moving the patch window
        
    Returns:
        List of positions (starts of patches)
    """
    # Calculate the total range
    range_size = max_val - min_val
    
    # If the range is smaller than a patch, return just the minimum value
    if range_size <= patch_size:
        return [min_val]
    
    # Generate positions with the specified step size
    positions = list(range(min_val, max_val - patch_size + 1, step_size))
    
    # Always include the last position to ensure we cover the full range
    if positions[-1] + patch_size < max_val:
        positions.append(max_val - patch_size)
    
    return positions


def compute_steps_for_sliding_window(image_size, patch_size, step_size_factor):
    """
    Compute the positions for sliding window patches with specified step size, based on nnUNet implementation.
    
    Args:
        image_size: size of the dimension (e.g., Z, Y, or X)
        patch_size: size of the patch in this dimension
        step_size_factor: step size as a fraction of patch_size (0 < step_size_factor <= 1)
        
    Returns:
        List of step positions for this dimension
    """
    assert image_size >= patch_size, "image size must be larger than patch_size"
    assert 0 < step_size_factor <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
    
    # Calculate step size in voxels
    target_step_size = int(patch_size * step_size_factor)
    
    # Calculate number of steps
    num_steps = int(np.ceil((image_size - patch_size) / target_step_size)) + 1
    
    # Calculate actual steps
    max_step_value = image_size - patch_size
    if num_steps > 1:
        actual_step_size = max_step_value / (num_steps - 1)
    else:
        actual_step_size = 99999999999  # Only one step at position 0
        
    steps = [int(np.round(actual_step_size * i)) for i in range(num_steps)]
    
    return steps