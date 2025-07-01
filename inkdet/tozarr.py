import os
import cv2
import numpy as np
import zarr
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

VESUVIUS_ROOT = "/vesuvius"

def load_and_convert_layer(path, target_shape=None):
    """Load layer and convert to uint8"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Convert to uint8 if needed
    if img.dtype == np.uint16:
        # Scale from 16-bit to 8-bit
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Clip and convert other dtypes
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Pad to target shape if specified
    if target_shape is not None:
        pad_y = target_shape[0] - img.shape[0]
        pad_x = target_shape[1] - img.shape[1]
        if pad_y > 0 or pad_x > 0:
            img = np.pad(img, [(0, max(0, pad_y)), (0, max(0, pad_x))], constant_values=0)

    return img & 0xf0


def get_padded_shape(height, width, chunk_size=64):
    """Calculate shape padded to next multiple of chunk_size"""
    padded_h = ((height + chunk_size - 1) // chunk_size) * chunk_size
    padded_w = ((width + chunk_size - 1) // chunk_size) * chunk_size
    return padded_h, padded_w


def process_fragment(fragment_path, output_path, batch_size=8):
    """Process a single fragment - callable by worker processes"""
    fragment_id = os.path.basename(fragment_path)

    try:
        # Open zarr store in append mode for this worker
        store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=store, overwrite=False)

        export_fragment_to_zarr(fragment_path, root, batch_size)
        return fragment_id, True, None
    except Exception as e:
        return fragment_id, False, str(e)


def export_fragment_to_zarr(fragment_path, zarr_group, batch_size=8):
    """Export a single fragment to zarr"""
    fragment_id = os.path.basename(fragment_path)

    # Find all layer files
    tif_files = sorted(glob.glob(os.path.join(fragment_path, "layers", "*.tif")))
    jpg_files = sorted(glob.glob(os.path.join(fragment_path, "layers", "*.jpg")))

    # Use tif if available, otherwise jpg
    layer_files = tif_files if tif_files else jpg_files

    if not layer_files:
        print(f"No layer files found for {fragment_id}")
        return

    # Only use first 64 layers (0-63)
    layer_files = layer_files[:64]

    if len(layer_files) < 64:
        print(f"Warning: {fragment_id} has only {len(layer_files)} layers")

    # Load first layer to get dimensions
    first_img = load_and_convert_layer(layer_files[0])
    h, w = first_img.shape
    padded_h, padded_w = get_padded_shape(h, w)

    # Create zarr array
    z_array = zarr_group.create_dataset(
        fragment_id,
        shape=(64, padded_h, padded_w),
        chunks=(64, 64, 64),
        dtype='uint8',
        compressor=zarr.Blosc(cname='blosclz', clevel=9),
        overwrite=True
    )

    # Process in batches
    for z_start in range(0, 64, batch_size):
        z_end = min(z_start + batch_size, 64)
        batch_size_actual = z_end - z_start

        # Allocate batch array
        batch = np.zeros((batch_size_actual, padded_h, padded_w), dtype=np.uint8)

        # Load layers in this batch
        for i, z_idx in enumerate(range(z_start, z_end)):
            if z_idx < len(layer_files):
                batch[i] = load_and_convert_layer(layer_files[z_idx], target_shape=(padded_h, padded_w))
            # else: leaves zeros for missing layers

        # Write batch to zarr
        z_array[z_start:z_end] = batch

    print(f"Exported {fragment_id}: shape={z_array.shape}, chunks={z_array.chunks}")


def main():
    train_scrolls_dir = f"{VESUVIUS_ROOT}/train_scrolls"
    output_path = f"/home/forrest/fragments.zarr"

    # Create zarr store with initial empty group
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    # Get all fragment directories
    fragment_dirs = [d for d in glob.glob(os.path.join(train_scrolls_dir, "*"))
                     if os.path.isdir(d) and os.path.exists(os.path.join(d, "layers"))]

    print(f"Found {len(fragment_dirs)} fragments to export")

    # Set up multiprocessing
    num_workers = 8
    print(f"Using {num_workers} workers")

    # Create partial function with fixed output_path
    process_func = partial(process_fragment, output_path=output_path)

    # Process fragments in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, fragment_dirs),
            total=len(fragment_dirs),
            desc="Exporting fragments"
        ))

    # Report results
    successful = sum(1 for _, success, _ in results if success)
    failed = [(fid, err) for fid, success, err in results if not success]

    print(f"\nExport complete. {successful}/{len(fragment_dirs)} fragments exported successfully.")
    if failed:
        print("\nFailed fragments:")
        for fid, err in failed:
            print(f"  {fid}: {err}")

    print(f"\nZarr saved to: {output_path}")
    print("\nTo verify:")
    print(">>> import zarr")
    print(f">>> z = zarr.open('{output_path}', mode='r')")
    print(">>> list(z.keys())")


if __name__ == "__main__":
    main()