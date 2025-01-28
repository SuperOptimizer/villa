# Giorgio Angelotti - 2024

import os
import argparse
import rawpy
import imageio
from tqdm import tqdm

VALID_RAW_EXTENSIONS = {'.cr2', '.nef', '.arw', '.dng', '.rw2', '.orf', '.raf'}

def convert_raw_to_jpg(input_path, output_path, quality=100):
    """
    Convert a single RAW image to JPG.
    
    :param input_path:  Path to the input RAW file
    :param output_path: Path to the output JPG file
    :param quality:     JPEG quality (1–100)
    """
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,   
        )
    imageio.imsave(output_path, rgb, quality=quality)

def process_directory_recursively(root_dir, quality=100):
    """
    Traverse the directory tree starting from `root_dir`. Whenever RAW files
    are found in a directory, create a `JPGEnhanced` folder in that same
    directory and convert all RAWs to JPEG there.
    
    :param root_dir:   The root directory to start searching from
    :param quality:    JPEG quality (1–100)
    """
    for current_dir, subdirs, files in os.walk(root_dir):
        if "corrupted" in current_dir.lower().split(os.sep):
            continue

        # Collect all RAW files in this directory
        raw_files = [
            f for f in files
            if os.path.splitext(f.lower())[1] in VALID_RAW_EXTENSIONS
        ]
        
        # Skip if no RAW files in this directory
        if not raw_files:
            continue
        
        # Create the JPGEnhanced folder in this directory
        enhanced_dir = os.path.join(current_dir, "JPGEnhanced")
        os.makedirs(enhanced_dir, exist_ok=True)
        
        # Convert each RAW file and store in JPGEnhanced
        for raw_file in tqdm(raw_files, desc=f"Converting files in folder {current_dir}"):
            input_path = os.path.join(current_dir, raw_file)
            base_name = os.path.splitext(raw_file)[0]
            output_file = base_name + ".jpg"
            output_path = os.path.join(enhanced_dir, output_file)
            convert_raw_to_jpg(input_path, output_path, quality=quality)

def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert RAW images in a directory tree to JPEG."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory to search for RAW files."
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="JPEG quality (1–100). Default=100"
    )
    
    args = parser.parse_args()
    process_directory_recursively(args.root_dir, args.quality)

if __name__ == "__main__":
    main()
