import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch.nn as nn
import fsspec
import zarr


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

def check_patch_chunk_xyz(
    chunk,
    sheet_label,     # shape (Xdim, Ydim, Zdim)
    patch_size_xyz,  # (pX, pY, pZ)
    bbox_threshold=0.5,
    label_threshold=0.05
):
    pX, pY, pZ = patch_size_xyz
    valid_positions = []

    for (x, y, z) in chunk:
        # Extract the patch in X,Y,Z order
        patch = sheet_label[x:x + pX, y:y + pY, z:z + pZ]

        bbox = compute_bounding_box_3d_xyz(patch > 0)
        if bbox is None:
            continue

        bb_vol = bounding_box_volume_xyz(bbox)
        patch_vol = patch.size  # pX * pY * pZ

        # 1) bounding box coverage
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) fraction of labeled voxels
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # Passed checks
        valid_positions.append((x, y, z))

    return valid_positions


def compute_bounding_box_3d_xyz(mask):
    """
    For a 3D boolean array in X,Y,Z order,
    returns (minx, maxx, miny, maxy, minz, maxz).
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minx, miny, minz = nonzero_coords.min(axis=0)
    maxx, maxy, maxz = nonzero_coords.max(axis=0)
    return (minx, maxx, miny, maxy, minz, maxz)

def bounding_box_volume_xyz(bbox):
    """
    Given (minx, maxx, miny, maxy, minz, maxz),
    returns total voxels in that bounding box.
    """
    (minx, maxx, miny, maxy, minz, maxz) = bbox
    return ((maxx - minx + 1) *
            (maxy - miny + 1) *
            (maxz - minz + 1))


def chunker(seq, chunk_size):
    """Yield successive 'chunk_size'-sized chunks from 'seq'."""
    for pos in range(0, len(seq), chunk_size):
        yield seq[pos:pos + chunk_size]

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

def check_patch_chunk(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
    """
    Worker function to check each patch in 'chunk' with both:
      - bounding box coverage >= bbox_threshold
      - overall labeled voxel ratio >= label_threshold
    """
    pD, pH, pW = patch_size
    valid_positions = []

    for (z, y, x) in chunk:
        patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
        # Compute bounding box of nonzero pixels in this patch
        bbox = compute_bounding_box_3d(patch > 0)
        if bbox is None:
            # No nonzero voxels at all -> skip
            continue

        # 1) Check bounding box coverage
        bb_vol = bounding_box_volume(bbox)
        patch_vol = patch.size  # pD * pH * pW
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) Check overall labeled fraction
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # If we passed both checks, add to valid positions
        valid_positions.append((z, y, x))

    return valid_positions

def check_patch_chunk_2d(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
    """
    Worker function to check each 2D patch in 'chunk' with both:
      - bounding box coverage >= bbox_threshold
      - overall labeled ratio >= label_threshold
    """
    pH, pW = patch_size
    valid_positions = []

    for (y, x) in chunk:
        patch = sheet_label[y:y + pH, x:x + pW]
        # Compute bounding box of nonzero pixels in this patch
        bbox = compute_bounding_box_2d(patch > 0)
        if bbox is None:
            # No nonzero pixels at all -> skip
            continue

        # 1) Check bounding box coverage
        bb_area = bounding_box_area(bbox)
        patch_area = patch.size  # pH * pW
        if bb_area / patch_area < bbox_threshold:
            continue

        # 2) Check overall labeled fraction
        labeled_ratio = np.count_nonzero(patch) / patch_area
        if labeled_ratio < label_threshold:
            continue

        # If we passed both checks, add to valid positions
        valid_positions.append((y, x))

    return valid_positions

def find_valid_patches_xyz(
    target_array,
    patch_size_xyz,         # (pX, pY, pZ)
    bbox_threshold=0.97,
    label_threshold=0.10,
    num_workers=4,
    overlap_fraction=0.50
):
    """
    Like find_valid_patches, but for data in X,Y,Z order.
    Returns valid patches in *zyx* positions by default,
    so it stays consistent with your final usage.
    """

    if target_array.ndim == 4:
        # let's assume shape is (C, X, Y, Z)
        target_array = target_array[0]

    pX, pY, pZ = patch_size_xyz
    Xdim, Ydim, Zdim = target_array.shape

    # Convert overlap fraction -> step (stride) along each axis
    x_stride = max(1, int(pX * (1 - overlap_fraction)))
    y_stride = max(1, int(pY * (1 - overlap_fraction)))
    z_stride = max(1, int(pZ * (1 - overlap_fraction)))

    # Generate all possible (x, y, z) starting positions
    all_positions = []
    for x in range(0, Xdim - pX + 1, x_stride):
        for y in range(0, Ydim - pY + 1, y_stride):
            for z in range(0, Zdim - pZ + 1, z_stride):
                all_positions.append((x, y, z))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"[XYZ] Finding valid patches of size: {patch_size_xyz} "
        f"(X,Y,Z) with bounding box >= {bbox_threshold} and labeled fraction >= {label_threshold}"
    )

    valid_positions_xyz = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk_xyz,
                (
                    chunk,
                    target_array,
                    patch_size_xyz,
                    bbox_threshold,
                    label_threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking XYZ patches", total=len(results)):
            valid_positions_xyz.extend(r.get())

    # Convert valid (x,y,z) to a final list storing (z,y,x)
    valid_patches = []
    for (x, y, z) in valid_positions_xyz:
        valid_patches.append({
            'volume_idx': 0,
            'start_pos': [z, y, x]  # reorder to Z,Y,X if that's your standard
        })

    print(f"[XYZ] Found {len(valid_positions_xyz)} valid patches (converted to Z,Y,X).")
    return valid_patches

def find_valid_patches_2d(
        target_array,
        patch_size,          # (pH, pW)
        bbox_threshold=0.97, # bounding-box coverage fraction
        label_threshold=0.10, # minimum % of pixels labeled
        num_workers=4,
        overlap_fraction=0.25,
        bounding_box=None   # optional: (y_min, x_min, y_max, x_max)
):
    """
    Finds 2D patches that have:
      - A bounding box of labeled pixels >= bbox_threshold fraction of the patch area
      - Overall labeled pixel fraction >= label_threshold
    within an optional bounding_box region.
    """
    # Patch dimensions
    pH, pW = patch_size
    Ydim, Xdim = target_array.shape

    # Determine bounding region limits
    if bounding_box is not None:
        y_min, x_min, y_max, x_max = bounding_box

        # Clamp bounding box to valid array range
        y_min = max(0, min(y_min, Ydim))
        x_min = max(0, min(x_min, Xdim))
        y_max = max(0, min(y_max, Ydim))
        x_max = max(0, min(x_max, Xdim))
    else:
        # Use entire image if no bounding box is given
        y_min, x_min = 0, 0
        y_max, x_max = Ydim, Xdim

    # Convert overlap fraction -> step (stride)
    y_stride = max(1, int(pH * (1 - overlap_fraction)))
    x_stride = max(1, int(pW * (1 - overlap_fraction)))

    # Generate all possible (y, x) starting positions within bounding box
    all_positions = []
    for y in range(y_min, y_max - pH + 1, y_stride):
        for x in range(x_min, x_max - pW + 1, x_stride):
            all_positions.append((y, x))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"Finding valid 2D patches of size: {patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk_2d,
                (
                    chunk,
                    target_array,
                    patch_size,
                    bbox_threshold,
                    label_threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking 2D patches", total=len(results)):
            valid_positions.extend(r.get())

    valid_patches = []
    for (y, x) in valid_positions:
        # For 2D data, we'll use a dummy z=0 in the start_pos
        valid_patches.append({'volume_idx': 0, 'start_pos': [0, y, x]})

    print(f"Found {len(valid_positions)} valid 2D patches.")
    return valid_patches

def find_valid_patches(
        target_array,
        patch_size,
        bbox_threshold=0.97,  # bounding-box coverage fraction
        label_threshold=0.10,  # minimum % of voxels labeled
        num_workers=4,
        overlap_fraction=0.25,
        bounding_box=None  # new param: (z_min, y_min, x_min, z_max, y_max, x_max)
):
    """
    Finds patches that have:
      - A bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
      - Overall labeled voxel fraction >= label_threshold
    within an optional bounding_box region.
    """
    # Handle 2D data (special case)
    if len(target_array.shape) == 2:
        # For 2D data, we'll use only the y, x components of patch_size
        return find_valid_patches_2d(
            target_array,
            patch_size=patch_size[1:3],  # Use only [y, x] dimensions
            bbox_threshold=bbox_threshold,
            label_threshold=label_threshold,
            num_workers=num_workers,
            overlap_fraction=overlap_fraction
        )

    # Patch dimensions
    pZ, pY, pX = patch_size
    Zdim, Ydim, Xdim = target_array.shape

    # Determine bounding region limits
    if bounding_box is not None:
        z_min, y_min, x_min, z_max, y_max, x_max = bounding_box

        # Clamp bounding box to valid array range (if desired):
        z_min = max(0, min(z_min, Zdim))
        y_min = max(0, min(y_min, Ydim))
        x_min = max(0, min(x_min, Xdim))
        z_max = max(0, min(z_max, Zdim))
        y_max = max(0, min(y_max, Ydim))
        x_max = max(0, min(x_max, Xdim))

    else:
        # Use entire volume if no bounding box is given
        z_min, y_min, x_min = 0, 0, 0
        z_max, y_max, x_max = Zdim, Ydim, Xdim

    # Convert overlap fraction -> step (stride)
    z_stride = max(1, int(pZ * (1 - overlap_fraction)))
    y_stride = max(1, int(pY * (1 - overlap_fraction)))
    x_stride = max(1, int(pX * (1 - overlap_fraction)))

    # Generate all possible (z, y, x) starting positions within bounding box
    all_positions = []
    # Ensure we don't exceed bounding box bounds
    for z in range(z_min, z_max - pZ + 1, z_stride):
        for y in range(y_min, y_max - pY + 1, y_stride):
            for x in range(x_min, x_max - pX + 1, x_stride):
                all_positions.append((z, y, x))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"Finding valid patches of size: {patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions_ref = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk,
                (
                    chunk,
                    target_array,
                    patch_size,
                    bbox_threshold,
                    label_threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking patches", total=len(results)):
            valid_positions_ref.extend(r.get())

    valid_patches = []
    for (z, y, x) in valid_positions_ref:
        valid_patches.append({'volume_idx': 0, 'start_pos': [z, y, x]})

    print(
        f"Found {len(valid_positions_ref)} valid patches in reference volume. "
        f"Total {len(valid_patches)} across all volumes."
    )

    return valid_patches
