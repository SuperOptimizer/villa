import numpy as np


def generate_positions(min_val, max_val, patch_size, step_size):
    """
    Generate positions for patch extraction with overlap.
    
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