# Giorgio Angelotti - 2024
import argparse
import os
import numpy as np
from math import sqrt
import tifffile
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import webknossos as wk
from webknossos import webknossos_context
from webknossos import Annotation

from tools import detect_vesselness

# import vesuvius
# from vesuvius import Volume


############################################################
# 1) PCA on voxel point cloud
############################################################
def classify_fiber_pca_on_voxels(voxel_coords, z_threshold=1./sqrt(2)):
    """
    Classify a fiber as 'vertical' or 'horizontal' by PCA on the *voxel* coordinates.

    Args:
        voxel_coords (np.ndarray): N x 3 array of (z, y, x)
        z_threshold (float)      : Dot product threshold with z-axis to label 'vertical'.

    Returns:
        "vertical" if principal axis is aligned with Z-axis,
        "horizontal" otherwise.
    """
    if voxel_coords.shape[0] < 2:
        # Degenerate: single voxel => pick a default
        return "horizontal"

    # Convert to float
    coords = voxel_coords.astype(np.float32)

    # Center the data
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid

    # Covariance
    cov = np.cov(coords_centered.T)
    # Check for NaNs or Infs
    if np.isnan(cov).any() or np.isinf(cov).any():
        return "horizontal"

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Principal axis
    principal_axis = eigvecs[:, 0]
    principal_axis /= np.linalg.norm(principal_axis)

    # Compare with the Z-axis
    z_axis = np.array([1, 0, 0], dtype=float)

    cos_angle = abs(np.dot(principal_axis, z_axis))
    return "vertical" if cos_angle > z_threshold else "horizontal"


############################################################
# 2) Adaptive interpolation
############################################################
def interpolate_adaptive(start_pos, end_pos, curvature_threshold=0.1, max_recursion=100):
    """Adaptive interpolation between two nodes based on curvature and resolution."""
    segment_vector = end_pos - start_pos
    segment_length = np.linalg.norm(segment_vector)

    if max_recursion == 0 or segment_length < curvature_threshold:
        return [start_pos, end_pos]

    mid_pos = (start_pos + end_pos) / 2.0
    left_segment = interpolate_adaptive(start_pos, mid_pos, curvature_threshold, max_recursion - 1)
    right_segment = interpolate_adaptive(mid_pos, end_pos, curvature_threshold, max_recursion - 1)
    return left_segment[:-1] + right_segment


############################################################
# 3) Fill a TEMP volume for one Tree (binary: 0/1)
############################################################
def fill_volume_for_tree(tree, output_shape, origins=(0, 0, 0)):
    """
    For a single 'tree', use adaptive interpolation to fill a binary volume 
    of shape `output_shape`, offset by `origins`. 
    Returns a np.uint8 array with 1= fiber, 0= background.
    """
    temp_fiber = np.zeros(output_shape, dtype=np.uint8)
    origins = np.array(origins)

    for node1, node2 in tree.edges:
        node1_pos = np.array([node1.position.x, node1.position.y, node1.position.z])
        node2_pos = np.array([node2.position.x, node2.position.y, node2.position.z])
        interpolated_points = interpolate_adaptive(node1_pos, node2_pos)

        for p in interpolated_points:
            voxel_coords = (p - origins).astype(int)  # (x, y, z) => our array might be (z, y, x)
            if np.all((0 <= voxel_coords) & (voxel_coords < output_shape)):
                temp_fiber[voxel_coords[2], voxel_coords[1], voxel_coords[0]] = 1

    return temp_fiber


############################################################
# 4) (Optional) Expand + Vesselness
############################################################
def expand_and_vesselness(binary_volume):
    # binary_volume: shape (Z, Y, X)
    binary_inverted = 1 - binary_volume
    edt = distance_transform_edt(binary_inverted)
    expanded_structure = (edt <= 3).astype(np.uint8)
    vessel = detect_vesselness(expanded_structure.astype(np.float32))
    combined = np.maximum(binary_volume, vessel)
    # Binning => threshold everything above 0 to 1
    bin_edges = np.histogram_bin_edges(combined, bins=2)
    binned_data = np.digitize(combined, bin_edges[1:]).astype(np.uint8)
    binned_data[binned_data > 0] = 1
    return binned_data


############################################################
# 5) Main function: for each tree => fill => PCA => label
############################################################
def voxelize_skeleton(annotation, output_shape=(100, 100, 100), origins=(0, 0, 0)):
    """
    - For each 'tree', we fill a TEMP volume, then gather its voxel coords => PCA => orientation.
    - We store it into separate volumes: vertical vs. horizontal.
    - Optionally do expansions/vesselness on each.
    - Finally merge them: vertical=1, horizontal=2, overlap=3.

    Returns a final volume in shape (Z, Y, X).
    """
    # We'll keep two volumes for final labeling
    voxel_grid_vertical = np.zeros(output_shape, dtype=np.uint8)
    voxel_grid_horizontal = np.zeros(output_shape, dtype=np.uint8)

    # Helper function: process a single tree
    def process_tree(tree):
        # 1) Fill a temporary volume for this tree
        temp_fiber = fill_volume_for_tree(tree, output_shape, origins)  # shape (Z, Y, X)
        # 2) Extract all voxel coords => (z,y,x) for PCA
        fiber_voxels = np.argwhere(temp_fiber > 0)
        # 3) Classify
        orientation = classify_fiber_pca_on_voxels(fiber_voxels)  # "vertical"/"horizontal"
        # 4) Merge into the appropriate volume
        if orientation == "vertical":
            np.maximum(voxel_grid_vertical, temp_fiber, out=voxel_grid_vertical)
        else:
            np.maximum(voxel_grid_horizontal, temp_fiber, out=voxel_grid_horizontal)

    # Process all groups and trees
    for group in tqdm(annotation.skeleton.groups, desc="Groups"):
        for tree in tqdm(group.trees, desc="Trees in groups"):
            process_tree(tree)

    # Process root-level trees
    for tree in tqdm(annotation.skeleton.trees, desc="Trees in root"):
        process_tree(tree)

    # Optionally do expansions + vesselness
    vert_final = expand_and_vesselness(voxel_grid_vertical)
    horz_final = expand_and_vesselness(voxel_grid_horizontal)

    # Merge with sum => overlap = 3
    vert_labeled = (vert_final > 0).astype(np.uint8) * 1
    horz_labeled = (horz_final > 0).astype(np.uint8) * 2
    merged = vert_labeled + horz_labeled

    # Final shape: (Z, Y, X)
    return merged


############################################################
# 6) The driver code
############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voxelize a skeleton .nml, compute PCA on voxelized point clouds, label vertical/horizontal with overlap=3."
    )
    parser.add_argument("--nml_path", required=True, help="Path to the .nml file.")
    parser.add_argument("--output_folder", required=True, help="Output folder path.")
    args = parser.parse_args()

    # Create folder structure if needed
    labels_folder = os.path.join(args.output_folder, "labelsTr")
    images_folder = os.path.join(args.output_folder, "imagesTr")
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # 1) Load the .nml annotation
    annotation = Annotation.load(args.nml_path)
    print(f"Annotation Name: {annotation.name}")

    # 2) Parse name for scroll & bounding box
    parts = annotation.name.split('_')
    scroll_id = parts[1]
    if scroll_id == "s1a":
        scroll_id = "s1"
    z_start = int(parts[2][:-1])
    y_start = int(parts[3][:-1])
    x_start = int(parts[4][:-1])
    size = int(parts[5])

    print(f"Scroll: {scroll_id}")
    print(f"Starting coordinates: {z_start},{y_start},{x_start}. Chunk size: {size}")

    # 3) Voxelize with point-cloud-based PCA
    print("Voxelizing skeleton fibers into volumes and computing orientation from the voxel cloud...")
    voxel_grid = voxelize_skeleton(
        annotation,
        output_shape=(size, size, size),  # (Z, Y, X)
        origins=(x_start, y_start, z_start)  # (X, Y, Z)
    )
    print(f"Voxel grid complete. Shape: {voxel_grid.shape}")

    # 4) Write label volume
    labels_filename_std = os.path.join(
        labels_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}_std.tif"
    )
    labels_filename = os.path.join(
        labels_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}.tif"
    )
    print(f"Writing annotation to {labels_filename} and {labels_filename_std}...")
    tifffile.imwrite(labels_filename, voxel_grid)
    tifffile.imwrite(labels_filename_std, voxel_grid)
    print("Annotation wrote.")

    # 5) Load and write image chunk from WebKnossos
    images_filename = os.path.join(
        images_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}_0000.tif"
    )
    images_filename_std = os.path.join(
        images_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}_std_0000.tif"
    )
    print(f"Writing image chunk to {images_filename} and {images_filename_std}...")

    WK_URL = "http://dl.ash2txt.org:8080"
    with open("token.txt", "r") as file:
        TOKEN = file.read().strip()
    print(f"Loaded TOKEN: {TOKEN}")

    ORGANIZATION_ID = "Scroll_Prize"
    bb = wk.NDBoundingBox(
        topleft=(x_start, y_start, z_start),
        size=(size, size, size),
        index=(0,1,2),
        axes=('x','y','z')
    )

    if int(scroll_id[1:]) == 1:
        dataset_name = "scroll1a"
    elif int(scroll_id[1:]) == 5:
        dataset_name = "scroll5-full"
    else:
        raise ValueError(f"Unrecognized scroll_id: {scroll_id}")

    with webknossos_context(url=WK_URL, token=TOKEN):
        ds = wk.Dataset.open_remote(dataset_name, ORGANIZATION_ID)

        volume = ds.get_layer("volume")
        view = volume.get_mag("1").get_view(absolute_bounding_box=bb)
        data = np.clip(view.read()[0].astype(np.float64)/257,0,255).astype(np.uint8)
        data = np.transpose(data, (2, 1, 0))
        tifffile.imwrite(images_filename, data)

    print("Done")
    '''
    # 6) Load standardized volume from Vesuvius
    print("Loading standardized volume (using Vesuvius library)...")
    scroll_volume = Volume(f"Scroll{int(scroll_id[1:])}")
    # Slicing: volume[z_start:z_end, y_start:y_end, x_start:x_end]
    # shape => (size, size, size)
    data_std = scroll_volume[z_start:z_start+size, y_start:y_start+size, x_start:x_start+size]
    tifffile.imwrite(images_filename_std, data_std)
    print("Done 2/2")'''
