import os
import re
import numpy as np
import zarr
import tifffile
import numcodecs
from multiprocessing import Pool
from tqdm import tqdm


# ---------------------------
# Helper function to read one TIFF file.
# ---------------------------
def read_tiff(args):
    """
    Read a TIFF file and return its z index and image array.

    Parameters:
      args: tuple (filepath, z_index)

    Returns:
      (z_index, image_array)
    """
    filepath, z_index = args
    # Read the image from file (adjust options as needed)
    data = tifffile.imread(filepath)
    return z_index, data


# ---------------------------
# Downsampling helper (nearest-neighbor)
# ---------------------------
def downsample_slice(slice_data, factor=2):
    """
    Downsample a 2D slice by taking every 'factor'-th pixel.
    (This is a simple nearest–neighbor downsampling.)
    """
    return slice_data[::factor, ::factor]


# ---------------------------
# Main function to create the OME-Zarr volume
# ---------------------------
def main():
    # ------------- SETTINGS -------------
    # Folder containing your TIFF slices
    tif_folder = "/mnt/raid_nvme/merged_s4"
    # Where to write the output OME-Zarr volume (a directory path)
    output_zarr = "/mnt/raid_nvme/merged_s4_obj.zarr"
    # Number of resolution levels (level 0 is full resolution)
    n_levels = 5
    # Chunk size for all dimensions (z, y, x)
    chunk_size = 128
    # Define a compressor – here using Blosc with zstd (adjust parameters as needed)
    compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    # ------------------------------------

    # Get list of TIFF files in the folder (case-insensitive extensions)
    tif_files = [f for f in os.listdir(tif_folder)
                 if f.lower().endswith(('.tif', '.tiff'))]

    # Use a regex to extract the first group of digits from each filename.
    # (Assumes filenames like "4.tif" or "slice_4.tif".)
    pattern = re.compile(r'(\d+)')
    file_info = []  # list of tuples: (full_path, z_index)
    z_indices_found = []

    for f in tif_files:
        m = pattern.search(f)
        if m:
            z_idx = int(m.group(1))
            full_path = os.path.join(tif_folder, f)
            file_info.append((full_path, z_idx))
            z_indices_found.append(z_idx)

    if not file_info:
        raise ValueError("No TIFF files found with a numeric slice index in the filename.")

    # Determine the overall z range.
    min_z = min(z_indices_found)
    max_z = max(z_indices_found)
    total_slices = max_z + 1  # Assuming slices are numbered from 0 to max_z.

    print(f"Found slices with z indices in range [{min_z}, {max_z}].")
    print(f"Total slices in the full volume will be {total_slices} "
          f"(missing slices will be filled with zeros).")

    # Read one sample file to get the (y, x) shape.
    sample_z, sample_img = file_info[0][1], tifffile.imread(file_info[0][0])
    slice_shape = sample_img.shape  # assume 2D (y, x)
    volume_shape = (total_slices,) + slice_shape  # (z, y, x)

    # ---------------------------
    # Create the root zarr group (as a directory store)
    # ---------------------------
    root = zarr.open(output_zarr, mode='w')

    # Create level 0 (full resolution) dataset using the given chunk size.
    ds0 = root.create_dataset(
        "0", shape=volume_shape, chunks=(chunk_size, chunk_size, chunk_size),
        compressor=compressor, dtype=sample_img.dtype, write_empty_chunks=False
    )

    # Fill the full volume with zeros so that missing slices (if any) are blank.
    print("Initializing full-resolution dataset with zeros...")
    ds0[:] = 0

    # ---------------------------
    # Use multiprocessing to read the TIFF files and insert them into the volume.
    # ---------------------------
    print("Reading TIFF slices and writing into the full-resolution volume (level 0)...")
    with Pool() as pool:
        # pool.imap_unordered will yield (z_index, data) tuples
        for z_idx, data in tqdm(pool.imap_unordered(read_tiff, file_info), total=len(file_info)):
            # Write the slice data to the correct z index.
            ds0[z_idx, :, :] = data

    # ---------------------------
    # Generate downsampled resolution levels (levels 1 .. n_levels-1)
    # ---------------------------
    # For each new level we downsample the previous level by a factor of 2 in y and x.
    previous_ds = ds0
    previous_shape = volume_shape  # (z, y, x)

    for level in range(1, n_levels):
        # Compute new (y, x) dimensions (using nearest–neighbor: simply take every 2nd pixel)
        new_y = (previous_shape[1] + 1) // 2
        new_x = (previous_shape[2] + 1) // 2
        new_shape = (previous_shape[0], new_y, new_x)

        print(f"Creating resolution level {level} with shape {new_shape} ...")
        ds_level = root.create_dataset(
            str(level), shape=new_shape, chunks=(chunk_size, chunk_size, chunk_size),
            compressor=compressor, dtype=sample_img.dtype
        )

        # Downsample each slice in z.
        for z in tqdm(range(new_shape[0]), desc=f"Downsampling level {level}"):
            # Read one slice from the previous level and downsample it.
            slice_prev = previous_ds[z, :, :]
            ds_level[z, :, :] = downsample_slice(slice_prev, factor=2)

        # Prepare for the next iteration.
        previous_ds = ds_level
        previous_shape = new_shape

    # ---------------------------
    # Write multiscale metadata for OME-Zarr (NGFF) compliance.
    # ---------------------------
    root.attrs["multiscales"] = [{
        "version": "0.1",
        "datasets": [{"path": str(l)} for l in range(n_levels)],
        "type": "ngff"
    }]
    print("OME-Zarr multiscale volume created successfully.")


if __name__ == "__main__":
    main()
