#!/usr/bin/env python3
"""
Vesuvius Pipeline - Run the complete inference, blending, and finalization process.
Can perform structure tensor analysis as well.
Uses multiple GPUs by assigning different devices to different parts (not DDP).
"""

import argparse
import os
import sys
import json
import shutil
import torch
from tqdm.auto import tqdm
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def get_available_gpus():
    """Returns a list of available GPU IDs."""
    try:
        num_gpus = torch.cuda.device_count()
        return list(range(num_gpus))
    except:
        return []


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Vesuvius complete inference pipeline (predict, blend, finalize)"
    )
    # Input/output arguments
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to input volume (zarr or folder with TIFFs)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path for the final output zarr')
    parser.add_argument('--workdir', type=str,
                        help='Working directory for intermediate files. Defaults to output_path + "_work"')

    # Model vs. Structure-Tensor switch
    parser.add_argument('--structure-tensor',
                        action='store_true',
                        help='Compute 6-channel 3D structure tensor instead of nnU-Net inference')
    parser.add_argument('--sigma',
                        type=float,
                        default=1.0,
                        help='Gaussian σ for structure-tensor smoothing')
    parser.add_argument(
        '--smooth-components',
        action='store_true',
        help='After computing Jxx…Jzz, apply a second Gaussian smoothing to each channel'
    )

    # Model arguments (only needed if not doing structure-tensor)
    parser.add_argument('--model',
                        type=str,
                        required=False,
                        help='Path to the model directory or checkpoint (required unless --structure-tensor)')

    parser.add_argument('--model-type', type=str,
                        choices=['nnunet', 'custom'],
                        default='nnunet',
                        help='Model type. "nnunet" (default) or "custom".')

    # Processing parameters
    parser.add_argument('--mode', type=str,
        choices=['binary', 'multiclass', 'structure-tensor'],
        default='binary',
        help='Processing mode: "binary", "multiclass", or "structure-tensor" (compute eigenvectors).')
    parser.add_argument('--threshold', dest='threshold',
                        action='store_true',
                        help='Apply thresholding to get binary/class masks instead of probability maps')
    parser.add_argument('--patch-size', dest='patch_size', type=str,
                        help='Patch size (z, y, x) separated by commas')

    # GPU settings
    parser.add_argument('--gpus', type=str, default='all',
                      help='GPU IDs to use, comma-separated (e.g., "0,1,2") or "all" for all available GPUs. Default: all')
    parser.add_argument('--parts-per-gpu', dest='parts_per_gpu', type=int, default=1,
                      help='Number of parts to process per GPU. Higher values use less GPU memory but take longer. Default: 1')
    
    # Performance settings
    parser.add_argument('--tta-type', dest='tta_type',
                        choices=['mirroring', 'rotation'],
                        default='rotation',
                        help='Test time augmentation type. Default: rotation')
    parser.add_argument('--disable-tta', dest='disable_tta',
                        action='store_true',
                        help='Disable test time augmentation')
    parser.add_argument('--single-part', dest='single_part',
                        action='store_true',
                        help='Process as a single part (no splitting for multi-GPU)')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=4,
                        help='Batch size for inference. Default: 4')
    parser.add_argument('--num-workers', dest='num_workers',
                        type=int, default=6,
                        help='Number of data loader workers. Default: 6')

    # Cleanup
    parser.add_argument('--keep-intermediates', dest='keep_intermediates',
                        action='store_true',
                        help='Keep intermediate files after processing')

    # Control flow
    parser.add_argument('--skip-predict', dest='skip_predict',
                        action='store_true',
                        help='Skip the prediction step (use existing prediction outputs)')
    parser.add_argument('--skip-blend', dest='skip_blend',
                        action='store_true',
                        help='Skip the blending step (use existing blended outputs)')
    parser.add_argument('--skip-finalize', dest='skip_finalize',
                        action='store_true',
                        help='Skip the finalization step (only generate blended logits)')

    # Verbosity
    parser.add_argument('--quiet', dest='quiet',
                        action='store_true',
                        help='Reduce verbosity')

    # --- fibers support (only in structure-tensor mode) ---
    parser.add_argument('--fibers',
                        action='store_true',
                        help='Run structure‐tensor twice: volume=1 (vertical) & volume=2 (horizontal)')
    # hidden, internal switch for each sub‐run
    parser.add_argument('--volume',
                        type=int,
                        choices=[1,2],
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # enforce model presence when not doing structure-tensor
    if not args.structure_tensor and not args.model:
        parser.error("--model is required unless --structure-tensor is set")
    # fibers only valid with structure‐tensor
    if args.fibers and not args.structure_tensor:
        parser.error("--fibers only valid when --structure-tensor is set")

    return args


def prepare_directories(args):
    """Prepare directory structure for the pipeline."""
    # Determine the working directory
    if args.workdir is None:
        args.workdir = f"{args.output}_work"

    # Define paths for intermediate outputs
    if args.workdir.startswith('s3://'):
        # For S3 paths, use proper path join
        args.parts_dir   = f"{args.workdir.rstrip('/')}/parts"
        args.blended_path = f"{args.workdir.rstrip('/')}/blended.zarr"

        # Create S3 directories
        import fsspec
        fs = fsspec.filesystem('s3', anon=False)
        # Remove s3:// prefix for fs operations
        workdir_no_prefix = args.workdir.replace('s3://', '')
        parts_dir_no_prefix = args.parts_dir.replace('s3://', '')
        fs.makedirs(workdir_no_prefix, exist_ok=True)
        fs.makedirs(parts_dir_no_prefix, exist_ok=True)
    else:
        # For local paths, use os.path.join
        # Create needed directories
        os.makedirs(args.workdir, exist_ok=True)

        # Define paths for intermediate outputs
        args.parts_dir    = os.path.join(args.workdir, "parts")
        args.blended_path = os.path.join(args.workdir, "blended.zarr")

        # Create parts directory
        os.makedirs(args.parts_dir, exist_ok=True)

    return args


def select_gpus(args):
    """Select GPUs to use based on arguments."""
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("No GPUs available. Running on CPU.")
        return []

    if args.gpus.lower() == 'all':
        selected = available_gpus
    else:
        requested = [int(x.strip()) for x in args.gpus.split(',')]
        selected  = [g for g in requested if g in available_gpus]
    if not selected:
        print("WARNING: No valid GPUs selected. Running on CPU.")
    else:
        print(f"Using GPUs: {selected}")
    return selected


def split_data_for_gpus(args, gpu_ids):
    """Determine how to split the data across GPUs."""
    if getattr(args, 'single_part', False) or not gpu_ids:
        # If single part or no GPUs, don't split
        num_parts = 1
    else:
        # Calculate number of parts based on GPUs and parts per GPU
        num_parts = len(gpu_ids) * getattr(args, 'parts_per_gpu', 1)

    return num_parts


def run_predict(args, part_id, gpu_id):
    """Run the prediction step for a single part."""
    cmd = ['vesuvius.predict']

    # Mode switch: structure-tensor vs nnU-Net
    if args.structure_tensor:
        cmd.append('--structure-tensor')
        cmd.extend(['--sigma', str(args.sigma)])
        if args.smooth_components:
            cmd.append('--smooth-components')
    else:
        cmd.extend(['--model_path', args.model])

    # Common args
    cmd.extend(['--input_dir',  args.input])
    cmd.extend(['--output_dir', args.parts_dir])

    # Add model type argument
    if args.model_type == 'custom':
        # not yet implemented, will be for ink / other models that are not nnunet based
        pass

    # Add device argument
    if gpu_id is not None:
        cmd.extend(['--device', f'cuda:{gpu_id}'])
    else:
        cmd.extend(['--device', 'cpu'])

    # Add part-specific arguments for multi-GPU
    cmd.extend(['--num_parts', str(args.num_parts)])
    cmd.extend(['--part_id',   str(part_id)])

    # TTA (Test Time Augmentation) settings
    if getattr(args, 'tta_type', None):
        cmd.extend(['--tta_type', args.tta_type])
    elif getattr(args, 'disable_tta', False):
        cmd.append('--disable_tta')
    else:
        # Default to rotation TTA
        cmd.extend(['--tta_type', 'rotation'])
    # Default behavior now uses rotation TTA if neither is specified

    # Add other optional arguments
    if args.patch_size:
        cmd.extend(['--patch_size', args.patch_size])

    # Performance settings
    cmd.extend(['--batch_size', str(args.batch_size)])
    if args.quiet:
        cmd.append('--quiet')

    print(f"Running Part {part_id} on GPU {gpu_id if gpu_id is not None else 'CPU'}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=(subprocess.PIPE if args.quiet else None),
        stderr=(subprocess.PIPE if args.quiet else None),
        universal_newlines=True,
        bufsize=1 # Line buffered
    )

    # Wait for the process to complete
    rc = proc.wait()
    if rc != 0 and args.quiet:
        err = proc.stderr.read() if proc.stderr else "No error output available"
        print(f"[Part {part_id}] ERROR:\n{err}")
    return rc == 0


def run_blend(args):
    """Run the blending step to merge all parts."""
    cmd = ['vesuvius.blend_logits', args.parts_dir, args.blended_path]
    if args.quiet:
        cmd.append('--quiet')

    # Run the command
    print(f"Blending parts: {' '.join(cmd)}")

    # Run with live stdout/stderr streaming for progress bars
    proc = subprocess.Popen(
        cmd,
        stdout=(subprocess.PIPE if args.quiet else None),
        stderr=(subprocess.PIPE if args.quiet else None),
        universal_newlines=True,
        bufsize=1 # Line buffered
    )

    # Wait for the process to complete
    rc = proc.wait()
    if rc != 0 and args.quiet:
        err = proc.stderr.read() if proc.stderr else "No error output"
        print(f"[Blend] ERROR:\n{err}")
    return rc == 0


def run_finalize(args):
    """Run the finalization step."""
    # DEBUG: Print threshold flag value before command construction
    print(f"DEBUG - threshold flag value: {args.threshold}")
    print(f"DEBUG - threshold flag type: {type(args.threshold)}")
    print(f"DEBUG - All available args: {vars(args)}")

    cmd = ['vesuvius.finalize_outputs', args.blended_path, args.output]

    # Add mode and threshold arguments
    cmd.extend(['--mode', args.mode])
    if args.threshold:
        cmd.append('--threshold')
    if getattr(args, 'swap_eigenvectors', False):
        cmd.append('--swap-eigenvectors')

    # Delete intermediates if not keeping them
    if not args.keep_intermediates:
        cmd.append('--delete-intermediates')

    # Add num_workers argument - use all available CPU cores
    import multiprocessing as mp
    num_workers = mp.cpu_count()
    cmd.extend(['--num-workers', str(num_workers)])
    print(f"Using {num_workers} worker processes for finalization")

    if args.quiet:
        cmd.append('--quiet')

    # Run the command
    print(f"Finalizing output: {' '.join(cmd)}")

    # Run with live stdout/stderr streaming for progress bars
    proc = subprocess.Popen(
        cmd,
        stdout=(subprocess.PIPE if args.quiet else None),
        stderr=(subprocess.PIPE if args.quiet else None),
        universal_newlines=True,
        bufsize=1  # Line buffered
    )

    # Wait for the process to complete
    rc = proc.wait()
    if rc != 0 and args.quiet:
        err = proc.stderr.read() if proc.stderr else "No error output"
        print(f"[Finalize] ERROR:\n{err}")
    return rc == 0


def cleanup(args):
    """Clean up intermediate files."""
    if not args.keep_intermediates:
        print("Cleaning up intermediate files...")

        # Handle cleanup for S3 paths
        if args.parts_dir.startswith('s3://'):
            try:
                import fsspec
                fs = fsspec.filesystem('s3', anon=False)
                if fs.exists(args.parts_dir):
                    # Remove all files in the parts directory
                    for file_path in fs.ls(args.parts_dir, detail=False):
                        fs.rm(file_path, recursive=True)
                    # Remove the parts directory itself
                    fs.rmdir(args.parts_dir)

                # Check if workdir is empty
                if fs.exists(args.workdir):
                    workdir_files = fs.ls(args.workdir, detail=False)
                    if len(workdir_files) == 0:
                        fs.rmdir(args.workdir)
            except Exception as e:
                print(f"Warning: Failed to clean up S3 intermediates: {e}")
        else:
            # Local file cleanup
            if os.path.exists(args.parts_dir):
                shutil.rmtree(args.parts_dir)

            # Only remove the work directory if it's empty
            try:
                os.rmdir(args.workdir)
            except OSError:
                # Directory not empty, so keep it
                pass


def setup_multipart(args, num_parts):
    """Setup for multi-part processing."""
    # No need to calculate Z-ranges since vesuvius.predict handles
    # the data splitting internally based on num_parts and part_id

    # Just validate and return the number of parts
    if num_parts < 1:
        print("Warning: Number of parts must be at least 1. Setting to 1.")
        num_parts = 1

    print(f"Setting up for processing in {num_parts} parts...")

    # Store the num_parts as an attribute on args for use in run_predict
    args.num_parts = num_parts

    # We're not returning Z-ranges anymore since vesuvius.predict
    # will handle partitioning internally
    return num_parts

def _run_single_pipeline(args):
    """
    The core of run_pipeline(), but isolated to one `args.volume` pass.
    """
    # 1) Prepare dirs, GPUs, parts
    args = prepare_directories(args)
    gpu_ids = select_gpus(args)
    num_parts = setup_multipart(args, split_data_for_gpus(args, gpu_ids))

    # 2) If we're filtering by volume, stash it in an env var for the inference script
    #    (we'll pick it up in inference.py below).
    if args.volume is not None:
        os.environ['VESUVIUS_FIBER_VOLUME'] = str(args.volume)

    # 3) Mode tweaks
    if args.structure_tensor:
        args.mode = "structure-tensor"
        args.disable_tta = True

    # 4) Predict
    if not args.skip_predict:
        print(f"\n--- Step 1: Prediction ({num_parts} parts) ---")
        if num_parts > 1:
            with ThreadPoolExecutor(max_workers=min(num_parts, os.cpu_count())) as ex:
                futures = [
                    ex.submit(run_predict, args, pid, gpu_ids[pid % len(gpu_ids)] if gpu_ids else None)
                    for pid in range(num_parts)
                ]
                if not all(f.result() for f in futures):
                    print("One or more prediction tasks failed. Aborting.")
                    return 1
        else:
            if not run_predict(args, 0, gpu_ids[0] if gpu_ids else None):
                print("Prediction failed. Aborting.")
                return 1

    # 5) Blend
    if not args.skip_blend:
        print("\n--- Step 2: Blending ---")
        
        # Check if parts directory exists and has contents (handle S3 paths)
        if args.parts_dir.startswith('s3://'):
            import fsspec
            fs = fsspec.filesystem('s3', anon=False)
            # Remove s3:// prefix for fs operations
            parts_dir_no_prefix = args.parts_dir.replace('s3://', '')
            parts_exist = fs.exists(parts_dir_no_prefix)
            parts_has_files = len(fs.ls(parts_dir_no_prefix)) > 0 if parts_exist else False
        else:
            if not run_predict(args, 0, gpu_ids[0] if gpu_ids else None):
                print("Prediction failed. Aborting.")
                return 1

    # Step 2: Blending
    if not args.skip_blend:
        print("\n--- Step 2: Blending ---")
        parts_exist = (args.parts_dir.startswith('s3://') 
                       and __import__('fsspec').filesystem('s3').exists(args.parts_dir)) \
            or os.path.exists(args.parts_dir)
        if not parts_exist or not os.listdir(args.parts_dir):
            print("No parts found; please run predictions first.")
            return 1
        if not run_blend(args):
            print("Blending failed. Aborting.")
            return 1

    # Step 3: Finalization
    if not args.skip_finalize:
        print("\n--- Step 3: Finalization ---")
        
        # Check if blended path exists (handle S3 paths)
        if args.blended_path.startswith('s3://'):
            import fsspec
            fs = fsspec.filesystem('s3', anon=False)
            # Remove s3:// prefix for fs operations
            blended_path_no_prefix = args.blended_path.replace('s3://', '')
            blended_exists = fs.exists(blended_path_no_prefix)
        else:
            blended_exists = os.path.exists(args.blended_path)
            
        if not blended_exists:
            print("No blended data found. Please run the blending step first.")
            return 1
        # pass --mode structure-tensor (or binary/multiclass) down to vesuvius.finalize
        if not run_finalize(args):
            print("Finalization failed.")
            return 1

    # Final cleanup
    cleanup(args)

    print(f"\n--- Pipeline Complete ---")
    print(f"Final output saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
