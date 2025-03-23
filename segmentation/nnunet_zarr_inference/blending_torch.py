import torch
import numpy as np


def ensure_tensor(array, device='cuda'):
    """Convert a numpy array to a torch tensor on the specified device."""
    if torch.is_tensor(array):
        return array.to(device)
    else:
        return torch.from_numpy(array).to(device)

def ensure_float_tensor(array, device='cuda'):
    """Convert a numpy array to a float tensor on the specified device."""
    tensor = ensure_tensor(array, device)
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        return tensor
    elif tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    elif tensor.dtype == torch.uint16:
        return tensor.float() / 65535.0
    else:
        return tensor.float()

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
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    tmp[tuple(center_coords)] = 1
    
    # Calculate sigmas exactly as in nnUNet
    sigmas = [i * sigma_scale for i in patch_size]
    
    # Apply Gaussian filter to spread the center point
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    
    # Convert to PyTorch tensor
    gaussian = torch.from_numpy(gaussian_importance_map).float().to(device)
    
    # Normalize and scale
    gaussian = gaussian / gaussian.max() * value_scaling_factor
    
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
    # If patch is uint8, convert to float16 and normalize to [0,1] range
    if patch_slice.dtype == torch.uint8:
        patch_slice = patch_slice.to(torch.float16) / 255.0
    # If it's already float16 or float32, keep as is for efficiency
    
    # Weight the patch directly using float16 computation for memory efficiency
    weighted_patch = patch_slice * weight_slice.unsqueeze(0).to(torch.float16)
    
    # Add to output using native float16 operations
    # Get the target region and ensure it matches our computation dtype
    region = output_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
    if region.dtype != weighted_patch.dtype:
        region = region.to(weighted_patch.dtype)
    
    # Add weighted patch directly - all operations in float16
    region += weighted_patch
    output_array[:, z_start:z_end, y_start:y_end, x_start:x_end] = region
    
    # Update the count array (only once per spatial location)
    count_region = count_array[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Convert weight_slice to match count_array dtype
    weight_slice_typed = weight_slice
    
    # For uint8 counts, we need to convert weights to same type
    if count_region.dtype == torch.uint8:
        # Scale weights by a factor to maintain precision when converting to uint8
        # (max Gaussian weight is usually 10, so scale by 25 to use full uint8 range)
        weight_slice_typed = (weight_slice * 25).to(torch.uint8)
    elif count_region.dtype == torch.float16:
        weight_slice_typed = weight_slice.to(torch.float16)
    
    # Add the weight_slice directly to the count_region
    count_array[z_start:z_end, y_start:y_end, x_start:x_end] = count_region + weight_slice_typed

def normalize_chunk_torch(output_chunk, count_chunk, dest_chunk, threshold=None, device='cpu', verbose=False):
    """
    Normalize a chunk by dividing output by count, with optional thresholding.
    
    Args:
        output_chunk (torch.Tensor): Output chunk (C, Z, Y, X) to normalize, works with float16
        count_chunk (torch.Tensor): Count chunk (Z, Y, X) to divide by, works with uint16/float16
        dest_chunk (torch.Tensor): Destination tensor to write results to
        threshold: Optional threshold value between 0 and 1
        device: PyTorch device to use
    """
    # Get the original dtype to perform operations in that precision if possible
    orig_dtype = output_chunk.dtype
    count_orig_dtype = count_chunk.dtype
    
    # For uint8 counts, need to convert to float
    if count_chunk.dtype == torch.uint8:
        # Convert to float16, adjusting for any scaling factor applied during accumulation
        # If we scaled by 25 during accumulation (see blend_patch_torch), we divide by 25 here
        count_chunk = count_chunk.to(torch.float16) / 25.0
    
    # Keep everything in float16 for memory efficiency if the source is already float16
    compute_dtype = torch.float16 if orig_dtype == torch.float16 else torch.float32
    
    # Convert tensors to computation dtype
    output_comp = output_chunk.to(compute_dtype)
    count_comp = count_chunk.to(compute_dtype)
    
    # Print debug info about count values if verbose
    if verbose:
        print(f"Count chunk stats: min={torch.min(count_comp).item():.4f}, max={torch.max(count_comp).item():.4f}, mean={torch.mean(count_comp).item():.4f}")
        print(f"Computing normalization in {compute_dtype} precision")
    
    # Create safe count tensor (avoid division by zero)
    # Use a small epsilon value for stable normalization
    safe_count = torch.clamp(count_comp, min=1e-4)
    
    # Normalize directly in PyTorch using broadcasting
    normalized = output_comp / safe_count.unsqueeze(0)
    
    # Apply threshold if provided
    if threshold is not None:
        normalized = (normalized >= threshold).to(compute_dtype)
    
    # Copy to destination, converting to destination dtype if needed
    if dest_chunk.dtype != normalized.dtype:
        normalized = normalized.to(dest_chunk.dtype)
    
    dest_chunk.copy_(normalized)

def find_intersecting_patches_torch(positions_tensor, n_patches, z_start, z_end, patch_size_z, max_z, device='cuda'):
    """
    Find all patches that intersect with a given z-range using PyTorch.

    Args:
        positions_tensor: Tensor of patch positions (n_patches, 3)
        n_patches: Number of patches
        z_start: Start of z-range to check
        z_end: End of z-range to check
        patch_size_z: Z dimension of patch size
        max_z: Maximum z value
        device: PyTorch device

    Returns:
        Tensor of indices of intersecting patches
    """
    # Extract z positions from the tensor
    z_positions = positions_tensor[:n_patches, 0]
    
    # Calculate end positions for each patch
    z_end_positions = torch.clamp(z_positions + patch_size_z, max=max_z)
    
    # Create masks for intersection
    intersects_start = z_positions < z_end
    intersects_end = z_end_positions > z_start
    
    # Combine masks to find intersecting patches
    intersects = torch.logical_and(intersects_start, intersects_end)
    
    # Get indices of intersecting patches
    intersecting_indices = torch.nonzero(intersects, as_tuple=True)[0]
    
    return intersecting_indices

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

def check_patch_intersection_torch(z, y, x, z_start, z_end, patch_size_tuple, max_z, max_y, max_x, device='cuda'):
    """
    Check if a patch intersects with a given z-range and compute intersection coordinates using PyTorch.

    Args:
        z, y, x: Patch coordinates
        z_start, z_end: Z-range to check
        patch_size_tuple: Tuple of patch dimensions (z, y, x)
        max_z, max_y, max_x: Maximum output dimensions
        device: PyTorch device

    Returns:
        Tuple of (intersects, target_coords, patch_coords) where:
            intersects: Boolean indicating if the patch intersects
            target_coords: (z_start, z_end, y_start, y_end, x_start, x_end) in output coordinates
            patch_coords: (patch_z_start, patch_z_end, patch_y_start, patch_y_end, patch_x_start, patch_x_end) in patch coordinates
    """
    # Convert to tensors if not already
    if not torch.is_tensor(z):
        z = torch.tensor(z, device=device)
        y = torch.tensor(y, device=device)
        x = torch.tensor(x, device=device)
        z_start = torch.tensor(z_start, device=device)
        z_end = torch.tensor(z_end, device=device)
        patch_size_z, patch_size_y, patch_size_x = [torch.tensor(p, device=device) for p in patch_size_tuple]
        max_z = torch.tensor(max_z, device=device)
        max_y = torch.tensor(max_y, device=device)
        max_x = torch.tensor(max_x, device=device)
    else:
        # Extract patch size components
        patch_size_z, patch_size_y, patch_size_x = patch_size_tuple
    
    # Calculate patch extents
    patch_z_end = torch.min(z + patch_size_z, max_z)
    
    # Check if this patch intersects with the z-range
    if z >= z_end or patch_z_end <= z_start:
        return False, None, None
    
    # Calculate intersection with z-chunk
    patch_z_start_rel = torch.max(torch.tensor(0, device=device), z_start - z)
    patch_z_end_rel = torch.min(patch_size_z, z_end - z)
    
    # Calculate valid dimensions for x and y
    patch_y_start_rel = torch.tensor(0, device=device)
    patch_y_end_rel = torch.min(patch_size_y, max_y - y)
    
    patch_x_start_rel = torch.tensor(0, device=device)
    patch_x_end_rel = torch.min(patch_size_x, max_x - x)
    
    # Skip if patch is invalid in any dimension
    if patch_y_end_rel <= patch_y_start_rel or patch_x_end_rel <= patch_x_start_rel:
        return False, None, None
    
    # Target coordinates in output volume
    target_z_start = torch.max(z_start, z)
    target_z_end = torch.min(z_end, z + patch_size_z)
    
    target_y_start = y
    target_y_end = torch.min(max_y, y + patch_size_y)
    
    target_x_start = x
    target_x_end = torch.min(max_x, x + patch_size_x)
    
    target_coords = (target_z_start, target_z_end, target_y_start, target_y_end, target_x_start, target_x_end)
    patch_coords = (
        patch_z_start_rel, patch_z_end_rel, patch_y_start_rel, patch_y_end_rel, patch_x_start_rel, patch_x_end_rel
    )
    
    return True, target_coords, patch_coords