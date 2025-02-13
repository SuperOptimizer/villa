import os
import numpy as np
import napari
import zarr
import fsspec
import tifffile
from magicgui import magicgui

# Try to import config defaults; if not found, use empty defaults.
try:
    from config import config
except ImportError:
    config = {}

# Global state – for production code you might encapsulate this in a class.
state = {
    # Volumes for display (lower res) and saving (high res)
    'display_image_volume': None,
    'display_label_volume': None,
    'highres_image_volume': None,
    'highres_label_volume': None,
    'patch_size': None,  # (in display resolution units)
    'patch_coords': None,  # Coordinates (tuples) in display resolution
    'current_index': 0,  # Index into patch_coords; the currently displayed patch is at current_index - 1
    'dataset_out_path': None,
    'images_out_dir': None,
    'labels_out_dir': None,
    'current_patch': None,  # Info about current patch (from display volume)
    'scale_factor': None,  # Downsampling factor: highres = display * scale_factor
    'save_progress': False,  # Whether to save progress to a file
    'progress_file': "",  # Path to the progress file
}


def generate_patch_coords(vol_shape, patch_size, sampling):
    """
    Generate a list of top-left (or front-top-left) coordinates for patches.
    Works for 2D (shape (H, W)) or 3D (shape (D, H, W)) volumes.
    """
    coords = []
    if len(vol_shape) == 2:
        H, W = vol_shape
        for i in range(0, H - patch_size + 1, patch_size):
            for j in range(0, W - patch_size + 1, patch_size):
                coords.append((i, j))
    elif len(vol_shape) >= 3:
        # Assume first three dimensions are spatial.
        D, H, W = vol_shape[:3]
        for z in range(0, D - patch_size + 1, patch_size):
            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    coords.append((z, y, x))
    else:
        raise ValueError("Volume must be at least 2D")
    if sampling == 'random':
        np.random.shuffle(coords)
    return coords


def extract_patch(volume, coord, patch_size):
    """
    Extract a patch from a volume starting at the given coordinate.
    For spatial dimensions, a slice is created from coord to coord+patch_size.
    Any extra dimensions (e.g. channels) are included in full.
    """
    slices = tuple(slice(c, c + patch_size) for c in coord)
    if volume.ndim > len(coord):
        slices = slices + (slice(None),) * (volume.ndim - len(coord))
    return volume[slices]


def load_next_patch():
    """
    Load the next valid patch (one whose label patch is not all zeros)
    from the display volumes and show it in napari.
    """
    global state, viewer
    if state.get('patch_coords') is None:
        print("Volume not initialized.")
        return

    patch_size = state['patch_size']
    display_image_volume = state['display_image_volume']
    display_label_volume = state['display_label_volume']
    coords = state['patch_coords']

    while state['current_index'] < len(coords):
        coord = coords[state['current_index']]
        # Extract patch from the display volumes.
        image_patch = extract_patch(display_image_volume, coord, patch_size)
        label_patch = extract_patch(display_label_volume, coord, patch_size)

        state['current_patch'] = {
            'coords': coord,  # in display resolution
            'image': image_patch,
            'label': label_patch,
            'index': state['current_index']
        }
        state['current_index'] += 1

        # Skip patch if the label patch is completely zero.
        if np.any(label_patch != 0):
            # Update or add napari layers.
            if "patch_image" in viewer.layers:
                viewer.layers["patch_image"].data = image_patch
            else:
                viewer.add_image(image_patch, name="patch_image", colormap='gray')
            if "patch_label" in viewer.layers:
                viewer.layers["patch_label"].data = label_patch
            else:
                viewer.add_labels(label_patch, name="patch_label")
            print(f"Loaded patch at display coordinate {coord}.")
            return
        else:
            print(f"Skipping patch at {coord} (label patch is all zeros).")
    print("No more patches available.")


def save_current_patch():
    """
    Save the current patch extracted from the high resolution volumes.
    The display patch coordinates are scaled up to map to the high resolution.
    File names are constructed using the high-res zyx (or yx) coordinates:
      - Image file gets a '_0000' suffix (e.g. img_z{z}_y{y}_x{x}_0000.tif).
      - Label file does not (e.g. lbl_z{z}_y{y}_x{x}.tif).
    """
    global state
    if state.get('current_patch') is None:
        print("No patch available to save.")
        return

    patch = state['current_patch']
    display_coord = patch['coords']
    patch_size = state['patch_size']
    scale_factor = state['scale_factor']

    # Map display coordinate to high resolution coordinate.
    hr_coord = tuple(c * scale_factor for c in display_coord)
    hr_patch_size = patch_size * scale_factor

    # Extract high-resolution patches.
    highres_image_patch = extract_patch(state['highres_image_volume'], hr_coord, hr_patch_size)
    highres_label_patch = extract_patch(state['highres_label_volume'], hr_coord, hr_patch_size)

    # Construct coordinate string.
    # For 3D, assume hr_coord is (z, y, x); for 2D, assume (y, x).
    if len(hr_coord) == 3:
        coord_str = f"z{hr_coord[0]}_y{hr_coord[1]}_x{hr_coord[2]}"
    elif len(hr_coord) == 2:
        coord_str = f"y{hr_coord[0]}_x{hr_coord[1]}"
    else:
        coord_str = "_".join(str(c) for c in hr_coord)

    # Build filenames using the coordinate string.
    image_filename = f"img_{coord_str}_0000.tif"
    label_filename = f"lbl_{coord_str}.tif"
    image_path = os.path.join(state['images_out_dir'], image_filename)
    label_path = os.path.join(state['labels_out_dir'], label_filename)

    tifffile.imwrite(image_path, highres_image_patch)
    tifffile.imwrite(label_path, highres_label_patch)
    print(f"Saved high-res image patch to {image_path} and label patch to {label_path}")


def update_progress():
    """
    If the save_progress option is enabled, write the current patch index to the progress file.
    """
    if state.get('save_progress') and state.get('progress_file'):
        try:
            with open(state['progress_file'], "w") as f:
                f.write(str(state['current_index']))
            print(f"Progress saved to {state['progress_file']}.")
        except Exception as e:
            print("Error saving progress:", e)


def load_progress():
    """
    If a progress file exists and save_progress is enabled, load the patch index.
    """
    if state.get('save_progress') and state.get('progress_file'):
        if os.path.exists(state['progress_file']):
            try:
                with open(state['progress_file'], "r") as f:
                    idx = int(f.read().strip())
                state['current_index'] = idx
                print(f"Resuming from patch index {idx} as per progress file.")
            except Exception as e:
                print("Error loading progress:", e)


@magicgui(
    sampling={"choices": ["random", "sequence"]}
)
def init_volume(
        image_zarr: str = config.get("image_zarr", ""),
        label_zarr: str = config.get("label_zarr", ""),
        dataset_out_path: str = config.get("dataset_out_path", ""),
        patch_size: int = config.get("patch_size", 64),
        sampling: str = config.get("sampling", "sequence"),
        save_progress: bool = config.get("save_progress", False),
        progress_file: str = config.get("progress_file", ""),
):
    """
    Load image and label volumes from Zarr (local or HTTP).
    For multiresolution Zarrs, load resolution '1' for display and resolution '0' for saving.
    Also creates output directories and computes patch coordinates (based on display volume).
    Optionally loads a progress file to resume patch iteration.
    """
    global state, viewer

    # --- Load image volume ---
    try:
        image_mapper = fsspec.get_mapper(image_zarr)
        image_group = zarr.open_group(image_mapper, mode='r')
        if "1" in image_group:
            display_image_volume = image_group["1"]
            print("Using resolution '1' for display of image volume.")
        else:
            display_image_volume = image_group
            print("No resolution '1' found for image volume; using provided array for display.")
        if "0" in image_group:
            highres_image_volume = image_group["0"]
            print("Using resolution '0' for saving patches of image volume.")
        else:
            highres_image_volume = image_group
            print("No resolution '0' found for image volume; using provided array for saving patches.")
    except Exception as e:
        print(f"Error loading image zarr: {e}")
        return

    # --- Load label volume ---
    try:
        label_mapper = fsspec.get_mapper(label_zarr)
        label_group = zarr.open_group(label_mapper, mode='r')
        if "1" in label_group:
            display_label_volume = label_group["1"]
            print("Using resolution '1' for display of label volume.")
        else:
            display_label_volume = label_group
            print("No resolution '1' found for label volume; using provided array for display.")
        if "0" in label_group:
            highres_label_volume = label_group["0"]
            print("Using resolution '0' for saving patches of label volume.")
        else:
            highres_label_volume = label_group
            print("No resolution '0' found for label volume; using provided array for saving patches.")
    except Exception as e:
        print(f"Error loading label zarr: {e}")
        return

    # Save the loaded volumes.
    state['display_image_volume'] = display_image_volume
    state['highres_image_volume'] = highres_image_volume
    state['display_label_volume'] = display_label_volume
    state['highres_label_volume'] = highres_label_volume
    state['patch_size'] = patch_size  # in display resolution units
    state['dataset_out_path'] = dataset_out_path

    # Save progress options.
    state['save_progress'] = save_progress
    state['progress_file'] = progress_file

    # Create output directories.
    images_dir = os.path.join(dataset_out_path, 'imagesTr')
    labels_dir = os.path.join(dataset_out_path, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    state['images_out_dir'] = images_dir
    state['labels_out_dir'] = labels_dir

    # Compute the scale factor between high-res and display volumes.
    # We assume the spatial dimensions are the first 2 (for 2D) or 3 (for 3D).
    num_spatial = 2 if len(display_image_volume.shape) == 2 else 3
    display_shape = display_image_volume.shape[:num_spatial]
    highres_shape = highres_image_volume.shape[:num_spatial]
    scale_factors = [high / disp for disp, high in zip(display_shape, highres_shape)]
    if not all(np.isclose(sf, scale_factors[0]) for sf in scale_factors):
        print("Warning: Non-isotropic scale factors detected. Using first dimension's scale factor.")
    scale_factor = int(round(scale_factors[0]))
    state['scale_factor'] = scale_factor
    print(f"Computed scale factor: {scale_factor}")

    # Compute patch coordinates on the display volume.
    vol_shape = display_shape
    state['patch_coords'] = generate_patch_coords(vol_shape, patch_size, sampling)

    # If progress saving is enabled and a progress file exists, load the saved progress.
    load_progress()

    print(f"Loaded display volumes with shape {vol_shape}.")
    print(f"Found {len(state['patch_coords'])} patch positions using '{sampling}' sampling.")
    load_next_patch()


@magicgui(call_button="next pair")
def iter_pair(approved: bool):
    """
    When "next pair" is pressed, if "approved" is checked, save the current patch
    from the high-resolution volumes, then load the next patch (skipping all-zero labels).
    After iterating, update the progress file if enabled.
    Also, the "approved" checkbox is reset to False.
    """
    if approved:
        save_current_patch()
    load_next_patch()
    update_progress()
    # Reset the approved checkbox:
    iter_pair.approved.value = False


@magicgui(call_button="previous pair")
def prev_pair():
    """
    When "previous pair" is pressed, go back one patch.
    This is done by decrementing the current index by 2 (so that the previously
    displayed patch becomes the next one to load) and then loading that patch.
    """
    global state
    if state['current_index'] > 1:
        state['current_index'] -= 2
        load_next_patch()
        update_progress()
    else:
        print("No previous patch available.")


if __name__ == '__main__':
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(init_volume, name="Initialize Volumes", area="right")
    viewer.window.add_dock_widget(iter_pair, name="Iterate Patches", area="right")
    viewer.window.add_dock_widget(prev_pair, name="Previous Patch", area="right")
    napari.run()
