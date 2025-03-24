import torch
import numpy as np

def create_gaussian_weights_torch(patch_size, sigma_scale=0.125, value_scaling_factor=10, device='cuda', 
                             edge_weight_boost=0.5):
    """
    Create a 3D Gaussian window using PyTorch, matching the approach from nnUNet.
    
    This implementation closely mirrors nnUNet's original approach, which:
    1. Creates a single-point impulse at the center
    2. Applies a true Gaussian filter to it
    3. Optionally adds edge handling

    Args:
        patch_size: Tuple of (depth, height, width)
        sigma_scale: Scale factor for sigma calculation (default: 1/8)
        value_scaling_factor: Scaling factor for final weights
        device: PyTorch device to create the tensor on
        edge_weight_boost: Factor to boost edge weights (0.0 = no boost, 1.0 = full plateau)

    Returns:
        3D Gaussian weight tensor
    """
    # Get dimensions
    depth, height, width = patch_size
    
    # Create a tensor with a single 1 at the center, matching nnUNet's approach
    # First on CPU with numpy (scipy.ndimage.gaussian_filter is more reliable than PyTorch equiv)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    # Create single point at center
    tmp = np.zeros(patch_size, dtype=np.float32)  # Use float32 explicitly
    center_coords = [i // 2 for i in patch_size]
    tmp[tuple(center_coords)] = 1.0
    
    # Calculate sigmas exactly as in nnUNet
    sigmas = [i * sigma_scale for i in patch_size]
    
    # Apply Gaussian filter to spread the center point
    # Use float32 for all calculations to ensure numerical stability
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    
    # Convert to PyTorch tensor with explicit float32 precision
    gaussian = torch.from_numpy(gaussian_importance_map).float().to(device)
    
    # Normalize and scale with extra care for numerical precision
    max_val = gaussian.max()
    if max_val > 0:
        gaussian = (gaussian / max_val) * value_scaling_factor
    else:
        # Handle the edge case where max_val might be zero
        # This shouldn't happen with a properly constructed Gaussian
        gaussian = torch.ones_like(gaussian) * value_scaling_factor
    
    # Edge handling - boost weights near the edges to reduce artifacts
    if edge_weight_boost > 0:
        # Create coordinate arrays
        z_coords = torch.arange(depth, dtype=torch.float32, device=device)
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        
        # Create distance maps from each edge
        z_edge_dist = torch.min(z_coords, depth - 1 - z_coords) / (depth / 4)  # Normalize by quarter size
        y_edge_dist = torch.min(y_coords, height - 1 - y_coords) / (height / 4)
        x_edge_dist = torch.min(x_coords, width - 1 - x_coords) / (width / 4)
        
        # Create edge factor maps (high near edges, low in center)
        z_edge_factor = torch.exp(-z_edge_dist)
        y_edge_factor = torch.exp(-y_edge_dist)
        x_edge_factor = torch.exp(-x_edge_dist)
        
        # Combine edge factors using max to boost any edge
        z_grid, y_grid, x_grid = torch.meshgrid(z_edge_factor, y_edge_factor, x_edge_factor, indexing='ij')
        edge_factor = torch.max(torch.max(z_grid, y_grid), x_grid)
        
        # Scale edge factor by boost amount
        edge_factor = edge_factor * edge_weight_boost
        
        # Apply edge boost to gaussian weights
        gaussian = gaussian * (1.0 + edge_factor)
        
        # Re-normalize to keep the max value the same
        gaussian = gaussian * (value_scaling_factor / gaussian.max())

    # Handle possible zeros - match nnUNet's approach exactly
    mask = gaussian == 0
    if torch.any(mask):
        gaussian[mask] = torch.min(gaussian[~mask])
    
    return gaussian

def blend_patch_torch(output_array, count_array, patch, weights,
                      z_start, z_end, y_start, y_end, x_start, x_end,
                      patch_z_start, patch_z_end, patch_y_start, patch_y_end, patch_x_start, patch_x_end):
    """
    Torch implementation to blend a patch into an output tensor.
    
    Args:
        output_array (torch.Tensor): Output tensor to add weighted patch to (C, Z, Y, X), can be float16
        count_array (torch.Tensor): Count tensor to track weights (Z, Y, X), can be uint8/uint16/float16
        patch (torch.Tensor): Patch tensor (C, Z, Y, X), can be uint8 or float32/float16
        weights (torch.Tensor): Gaussian weights (Z, Y, X)
        z_start, z_end, y_start, y_end, x_start, x_end: Target coordinates in output.
        patch_z_start, patch_z_end, patch_y_start, patch_y_end, patch_x_start, patch_x_end: Source coordinates in patch.
    """
    # Slice the patch and weights for the corresponding region.
    patch_slice = patch[:, patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
    weight_slice = weights[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
    
    # Handle different patch dtypes
    # If patch is uint8, convert to float32 and normalize to [0,1] range for better precision
    if patch_slice.dtype == torch.uint8:
        patch_slice = patch_slice.to(torch.float32) / 255.0
    # If it's already float16 or float32, keep as is for efficiency
    elif patch_slice.dtype == torch.float16:
        # Upgrade to float32 for more accurate blending
        patch_slice = patch_slice.to(torch.float32)
    
    # Weight the patch using float32 for better precision during blending
    # This is a critical step where we need the full precision
    weight_slice_float = weight_slice.to(torch.float32)
    weighted_patch = patch_slice * weight_slice_float.unsqueeze(0)
    
    # Add to output using full precision
    # Get the target region and ensure it matches our computation dtype
    region = output_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Convert region to float32 for accurate blending
    region_float = region.to(torch.float32)
    
    # Add weighted patch directly - operations in float32 for accuracy
    region_float += weighted_patch
    
    # Store the result back with appropriate dtype conversion if needed
    if output_array.dtype == torch.float16:
        # Convert back to float16 for storage
        output_array[:, z_start:z_end, y_start:y_end, x_start:x_end] = region_float.to(torch.float16)
    else:
        # Keep in float32 or whatever the output array dtype is
        output_array[:, z_start:z_end, y_start:y_end, x_start:x_end] = region_float
    
    # Update the count array (only once per spatial location)
    count_region = count_array[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Process count updates with proper precision
    if count_region.dtype == torch.uint8:
        # For uint8 counts, convert weight_slice to same type with appropriate scaling
        # (max Gaussian weight is usually 10, so scale by 25 to use full uint8 range)
        weight_slice_typed = (weight_slice * 25).to(torch.uint8)
        
        # Add with saturation to prevent overflow
        # First convert to a higher precision type
        count_sum = count_region.to(torch.int16) + weight_slice_typed.to(torch.int16)
        # Then clamp to uint8 range and convert back
        count_sum = torch.clamp(count_sum, 0, 255).to(torch.uint8)
        
        # Store back to count array
        count_array[z_start:z_end, y_start:y_end, x_start:x_end] = count_sum
    elif count_region.dtype == torch.float16:
        # For float16, use float32 for the addition, then back to float16
        count_sum = count_region.to(torch.float32) + weight_slice.to(torch.float32)
        count_array[z_start:z_end, y_start:y_end, x_start:x_end] = count_sum.to(torch.float16)
    else:
        # For other types (float32, etc.), add directly with type matching
        weight_slice_typed = weight_slice.to(count_region.dtype)
        count_array[z_start:z_end, y_start:y_end, x_start:x_end] = count_region + weight_slice_typed

def intersects_chunk(z, y, x, patch_size_tuple, z_chunk_start, z_chunk_end):
    """
    Simple utility: Does a patch at (z,y,x) with shape patch_size_tuple
    intersect a chunk in the z-dimension range [z_chunk_start, z_chunk_end)?
    We only check Z here, but you could also chunk in Y or X similarly.
    
    Args:
        z, y, x: Patch coordinates
        patch_size_tuple: Tuple of patch dimensions (z, y, x)
        z_chunk_start, z_chunk_end: Z-range to check
        
    Returns:
        Boolean indicating if the patch intersects with the chunk
    """
    pz = patch_size_tuple[0]
    # The patch covers z in [z, z + pz)
    if z >= z_chunk_end:
        return False
    if (z + pz) <= z_chunk_start:
        return False
    return True
