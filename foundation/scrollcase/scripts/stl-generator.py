# Giorgio Angelotti, 2025
"""
stl-generator.py

A script to generate and export final STL models for all scroll cases found in a given input root directory.
Each subfolder in the input directory should be named as the scroll number and contain a file named
   <scroll_number>-registered.obj

For each scroll, the script creates a subfolder (named using the padded scroll name) under the output root and saves:
   - <padded_scroll>_scroll.stl
   - <padded_scroll>_right.stl
   - <padded_scroll>_left.stl

Additionally, a CSV summary (scroll_summary.csv) is created in the output root with columns:
   "Scroll ID", "Height (mm)", "Diameter (mm)"

Usage:
    python stl-generator.py --input /path/to/scrolls --output /path/to/output [--config /path/to/config.yml]
"""

import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml  # Requires PyYAML
from tqdm import tqdm

from scrollcase import mesh, case
from meshlib import mrmeshpy as mm  # for saving STL files

def pad_scroll_name(scroll_number: str) -> str:
    """
    Pads the numeric portion of the scroll number to 4 digits.
    For example, "800" becomes "0800" and "800A" becomes "0800A".
    """
    if scroll_number and scroll_number[-1].isalpha():
        number_part = scroll_number[:-1]
        letter_part = scroll_number[-1]
        padded = f"{int(number_part):04d}{letter_part}"
    else:
        padded = f"{int(scroll_number):04d}"
    return padded

def process_scroll(padded_scroll: str, mesh_file: str, output_dir: str, config=None):
    """
    Process a single scroll:
      1. Exports the original OBJ as an STL file.
      2. Processes the mesh to build the lining.
      3. Builds the scroll case.
      4. Combines the case halves with the mesh lining.
      5. Exports the combined STL files.
    
    The configuration from the YAML file (if provided) is used to override default parameters for the scroll case.
    
    Returns a tuple: (padded_scroll, height, diameter)
    """
    logger = logging.getLogger("stl_generator")

    # 1. Export the original OBJ as an STL file.
    original_mesh = mesh.load_mesh(mesh_file)
    original_stl_path = os.path.join(output_dir, f"{padded_scroll}_scroll.stl")
    mm.saveMesh(original_mesh, original_stl_path)

    # 2. Process the mesh to build the lining.
    scroll_mesh_params = mesh.ScrollMesh(mesh_file)
    (lining_mesh_pos,
     lining_mesh_neg,
     cavity_mesh_pos,
     cavity_mesh_neg,
     mesh_scroll,
     radius,
     height,
     ) = mesh.build_lining(scroll_mesh_params)

    # 3. Build the scroll case.
    # Start with some default parameters.
    scroll_case_defaults = {
        "scroll_height_mm": height,
        "scroll_radius_mm": radius,
        "label_line_1": f"PHerc{padded_scroll}",
        "alignment_ring_spacing_mm": height,
    }
    # If a YAML config was provided and contains "scroll_case", update the defaults.
    if config is not None and "scroll_case" in config:
        scroll_case_defaults.update(config["scroll_case"])

    scroll_case = case.ScrollCase(**scroll_case_defaults)
    case_left, case_right = case.build_case(scroll_case)

    # 4. Combine the BRep case halves with the mesh lining.
    combined_mesh_right = mesh.combine_brep_case_lining(case_right, cavity_mesh_pos, lining_mesh_pos)
    combined_mesh_left  = mesh.combine_brep_case_lining(case_left,  cavity_mesh_neg, lining_mesh_neg)

    # 5. Export the combined STL files.
    right_stl_path = os.path.join(output_dir, f"{padded_scroll}_right.stl")
    left_stl_path  = os.path.join(output_dir, f"{padded_scroll}_left.stl")
    mm.saveMesh(combined_mesh_right, right_stl_path)
    mm.saveMesh(combined_mesh_left, left_stl_path)

    # Return padded_scroll, height, and diameter (2 * radius)
    return padded_scroll, height, 2 * radius

def main():
    # Set up logging to only show errors
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("stl_generator")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate final STL models for all scroll cases in a root directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing subfolders for each scroll case.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory where generated STL files and CSV summary will be saved.",
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Optional YAML configuration file containing scroll case parameters.",
    )
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output

    if not os.path.isdir(input_root):
        logger.error(f"Input root directory does not exist: {input_root}")
        return

    # Load YAML configuration if provided.
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return

    # List subdirectories (each representing a scroll number)
    subdirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not subdirs:
        logger.error("No subdirectories found in the input root directory.")
        return

    # Prepare a list of tasks: each task is a tuple (padded_scroll, mesh_file, scroll_output_dir, config)
    tasks = []
    for scroll_number in subdirs:
        padded_scroll = pad_scroll_name(scroll_number)
        subdir_path = os.path.join(input_root, scroll_number)
        mesh_file = os.path.join(subdir_path, f"{scroll_number}-registered.obj")
        if not os.path.exists(mesh_file):
            logger.error(f"Mesh file not found for scroll '{scroll_number}': {mesh_file}. Skipping.")
            continue

        # Create output folder for this scroll case using the padded scroll name
        scroll_output_dir = os.path.join(output_root, padded_scroll)
        os.makedirs(scroll_output_dir, exist_ok=True)
        tasks.append((padded_scroll, mesh_file, scroll_output_dir, config))

    # List to accumulate CSV summary data as tuples: (Scroll ID, Height, Diameter)
    scroll_summary = []

    # Process scrolls concurrently using ProcessPoolExecutor with a tqdm progress bar
    with ProcessPoolExecutor() as executor:
        future_to_scroll = {
            executor.submit(process_scroll, padded, mesh_file, out_dir, config): padded
            for padded, mesh_file, out_dir, config in tasks
        }
        for future in tqdm(as_completed(future_to_scroll), total=len(future_to_scroll), desc="Processing scrolls"):
            padded_scroll = future_to_scroll[future]
            try:
                result = future.result()  # (padded_scroll, height, diameter)
                scroll_summary.append(result)
            except Exception as exc:
                logger.error(f"Error processing scroll '{padded_scroll}': {exc}")

    # Write CSV summary to the output root folder.
    csv_file = os.path.join(output_root, "scroll_summary.csv")
    try:
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Scroll ID", "Height (mm)", "Diameter (mm)"])
            for scroll_id, height, diameter in scroll_summary:
                writer.writerow([scroll_id, f"{height:.2f}", f"{diameter:.2f}"])
    except Exception as e:
        logger.error(f"Error writing CSV summary: {e}")

if __name__ == "__main__":
    main()