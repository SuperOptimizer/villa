import numpy as np
import torch.nn as nn


def find_mask_patches(mask_array, label_array, patch_size, stride=None, min_mask_coverage=1.0, min_labeled_ratio=0.05):
    """
    Find 3D patches where:
    1. The patch is fully contained within the masked region
    2. The patch contains a minimum ratio of labeled voxels
    
    Parameters:
    -----------
    mask_array : numpy.ndarray
        3D binary mask array indicating valid regions
    label_array : numpy.ndarray
        3D array containing the actual labels/annotations
    patch_size : list or tuple
        [depth, height, width] of patches to extract
    stride : list or tuple, optional
        Stride for patch extraction, defaults to 50% overlap
    min_mask_coverage : float, optional
        Minimum coverage ratio of mask required (1.0 = 100% coverage)
    min_labeled_ratio : float, optional
        Minimum ratio of labeled voxels in the entire patch (0.05 = 5%)
    
    Returns:
    --------
    list
        List of valid patch positions
    """
    d, h, w = patch_size
    D, H, W = mask_array.shape
    
    # Define stride (overlap)
    if stride is None:
        stride = (d//2, h//2, w//2)  # 50% overlap by default
    
    patches = []
    
    # Generate all possible positions
    for z in range(0, D-d+1, stride[0]):
        for y in range(0, H-h+1, stride[1]):
            for x in range(0, W-w+1, stride[2]):
                # Extract the patch from both mask and label arrays
                mask_patch = mask_array[z:z+d, y:y+h, x:x+w]
                label_patch = label_array[z:z+d, y:y+h, x:x+w]
                
                # Skip patches with no mask values
                if not np.any(mask_patch > 0):
                    continue
                
                # MASK REQUIREMENT:
                # Check mask coverage
                mask_coverage = np.count_nonzero(mask_patch) / mask_patch.size
                if mask_coverage < min_mask_coverage:
                    continue
                    
                # LABEL REQUIREMENT:
                # Calculate ratio of labeled voxels in the patch
                labeled_ratio = np.count_nonzero(label_patch) / label_patch.size
                
                # Only include patches with sufficient labeled voxels
                if labeled_ratio >= min_labeled_ratio:
                    patches.append({'start_pos': [z, y, x]})
    
    return patches

def find_mask_patches_2d(mask_array, label_array, patch_size, stride=None, min_mask_coverage=1.0, min_labeled_ratio=0.05):
    """
    Find 2D patches where:
    1. The patch is fully contained within the masked region
    2. The patch contains a minimum ratio of labeled pixels
    
    Parameters:
    -----------
    mask_array : numpy.ndarray
        2D binary mask array indicating valid regions
    label_array : numpy.ndarray
        2D array containing the actual labels/annotations
    patch_size : list or tuple
        [height, width] of patches to extract
    stride : list or tuple, optional
        Stride for patch extraction, defaults to 50% overlap
    min_mask_coverage : float, optional
        Minimum coverage ratio of mask required (1.0 = 100% coverage)
    min_labeled_ratio : float, optional
        Minimum ratio of labeled pixels in the entire patch (0.05 = 5%)
    
    Returns:
    --------
    list
        List of valid patch positions
    """
    h, w = patch_size
    H, W = mask_array.shape
    
    # Define stride (overlap)
    if stride is None:
        stride = (h//2, w//2)  # 50% overlap by default
    
    patches = []
    
    # Generate all possible positions
    for y in range(0, H-h+1, stride[0]):
        for x in range(0, W-w+1, stride[1]):
            # Extract the patch from both mask and label arrays
            mask_patch = mask_array[y:y+h, x:x+w]
            label_patch = label_array[y:y+h, x:x+w]
            
            # Skip patches with no mask values
            if not np.any(mask_patch > 0):
                continue
                
            # MASK REQUIREMENT:
            # Check mask coverage
            mask_coverage = np.count_nonzero(mask_patch) / mask_patch.size
            if mask_coverage < min_mask_coverage:
                continue
                
            # LABEL REQUIREMENT:
            # Calculate ratio of labeled pixels in the patch
            labeled_ratio = np.count_nonzero(label_patch) / label_patch.size
            
            # Only include patches with sufficient labeled pixels
            if labeled_ratio >= min_labeled_ratio:
                patches.append({'start_pos': [0, y, x]})  # [dummy_z, y, x]
    
    return patches


def generate_positions(min_val, max_val, patch_size, step):
    """
    Returns a list of start indices (inclusive) for sliding-window patches,
    ensuring the final patch covers the end of the volume.
    """
    positions = []
    pos = min_val
    while pos + patch_size <= max_val:
        positions.append(pos)
        pos += step

    # Force the last patch if not already covered
    last_start = max_val - patch_size
    if last_start > positions[-1]:
        positions.append(last_start)

    return sorted(set(positions))


def pad_or_crop_3d(arr, desired_shape, pad_value=0):
    """Pad or crop a 3D array (D,H,W) to the desired shape."""
    d, h, w = arr.shape
    dd, hh, ww = desired_shape
    out = np.full((dd, hh, ww), pad_value, dtype=arr.dtype)

    # Compute the region to copy
    dmin = min(d, dd)
    hmin = min(h, hh)
    wmin = min(w, ww)

    out[:dmin, :hmin, :wmin] = arr[:dmin, :hmin, :wmin]
    return out

def pad_or_crop_4d(arr, desired_shape, pad_value=0):
    """Pad or crop a 4D array (C,D,H,W) to the desired shape."""
    c, d, h, w = arr.shape
    cc, dd, hh, ww = desired_shape
    out = np.full((cc, dd, hh, ww), pad_value, dtype=arr.dtype)

    # Compute the region to copy
    cmin = min(c, cc)
    dmin = min(d, dd)
    hmin = min(h, hh)
    wmin = min(w, ww)

    out[:cmin, :dmin, :hmin, :wmin] = arr[:cmin, :dmin, :hmin, :wmin]
    return out

def pad_or_crop_2d(arr, desired_shape, pad_value=0):
    """Pad or crop a 2D array (H,W) to the desired shape."""
    h, w = arr.shape
    hh, ww = desired_shape
    out = np.full((hh, ww), pad_value, dtype=arr.dtype)

    # Compute the region to copy
    hmin = min(h, hh)
    wmin = min(w, ww)

    out[:hmin, :wmin] = arr[:hmin, :wmin]
    return out

def init_weights_he(module, neg_slope=1e-2):
    if isinstance(module, (nn.Conv2d, nn.Conv3d,
                           nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def compute_bounding_box_3d(mask):
    """
    Given a 3D boolean array (True where labeled, False otherwise),
    returns (minz, maxz, miny, maxy, minx, maxx).
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minz, miny, minx = nonzero_coords.min(axis=0)
    maxz, maxy, maxx = nonzero_coords.max(axis=0)
    return (minz, maxz, miny, maxy, minx, maxx)

def compute_bounding_box_2d(mask):
    """
    Given a 2D boolean array (True where labeled, False otherwise),
    returns (miny, maxy, minx, maxx).
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    miny, minx = nonzero_coords.min(axis=0)
    maxy, maxx = nonzero_coords.max(axis=0)
    return (miny, maxy, minx, maxx)

def bounding_box_volume(bbox):
    """
    Given a bounding box (minz, maxz, miny, maxy, minx, maxx),
    returns the volume (number of voxels) inside the box.
    """
    minz, maxz, miny, maxy, minx, maxx = bbox
    return ((maxz - minz + 1) *
            (maxy - miny + 1) *
            (maxx - minx + 1))

def bounding_box_area(bbox):
    """
    Given a 2D bounding box (miny, maxy, minx, maxx),
    returns the area (number of pixels) inside the box.
    """
    miny, maxy, minx, maxx = bbox
    return (maxy - miny + 1) * (maxx - minx + 1)
