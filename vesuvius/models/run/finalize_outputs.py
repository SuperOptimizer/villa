import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse
import zarr
import fsspec
import numcodecs
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from data.utils import open_zarr


# Worker function to process a single chunk
def process_chunk(chunk_info, input_path, output_path, mode, threshold, num_classes, spatial_shape, output_chunks):
    """
    Process a single chunk of the volume in parallel.
    
    Args:
        chunk_info: Dictionary with chunk boundaries and indices
        input_path: Path to input zarr
        output_path: Path to output zarr
        mode: Processing mode ("binary" or "multiclass")
        threshold: Whether to apply threshold/argmax
        num_classes: Number of classes in input
        spatial_shape: Spatial dimensions of the volume (Z, Y, X)
        output_chunks: Chunk size for output
    """
    # Extract chunk indices
    chunk_idx = chunk_info['indices']
    
    # Calculate slice for this chunk
    spatial_slices = tuple(
        slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
        for idx, chunk, shape_dim in zip(chunk_idx, output_chunks[1:], spatial_shape)
    )
    
    # Open input and output stores
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None
    )
    
    output_store = open_zarr(
        path=output_path,
        mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None
    )
    
    # Read all classes for this spatial region
    input_slice = (slice(None),) + spatial_slices  # All classes, specific spatial region
    logits_np = input_store[input_slice]
    
    # Convert to torch tensor for processing
    logits = torch.from_numpy(logits_np)
    
    # Process based on mode
    if mode == "binary":
        # For binary case, we just need a softmax over dim 0 (channels)
        softmax = F.softmax(logits, dim=0)
        
        if threshold:  # Now a boolean flag
            # Create binary mask using argmax (class 1 is foreground)
            # Simply check if foreground probability > background probability
            binary_mask = (softmax[1] > softmax[0]).float().unsqueeze(0)
            output_data = binary_mask
        else:
            # Extract foreground probability (channel 1)
            fg_prob = softmax[1].unsqueeze(0)  # Add channel dim back
            output_data = fg_prob
            
    else:  # multiclass
        # Apply softmax over channel dimension
        softmax = F.softmax(logits, dim=0)
        
        # Compute argmax
        argmax = torch.argmax(logits, dim=0).float().unsqueeze(0)  # Add channel dim
        
        if threshold:  # Now a boolean flag
            # If threshold is provided for multiclass, only save the argmax
            output_data = argmax
        else:
            # Concatenate softmax and argmax
            output_data = torch.cat([softmax, argmax], dim=0)
    
    # Convert to numpy
    output_np = output_data.numpy()
    
    # Check if this is an empty patch with 0.5 values
    # This handles the specific case where empty patches got 0.5 values
    is_empty = np.isclose(output_np, 0.5, rtol=1e-3).all()
    
    if is_empty:
        # For empty patches, don't write anything to the output store
        # This ensures write_empty_chunks=False works correctly
        return {'chunk_idx': chunk_idx, 'processed_voxels': 0, 'empty': True}
    
    # Scale to uint8 range [0, 255]
    min_val = output_np.min()
    max_val = output_np.max()
    if min_val < max_val:  # Avoid division by zero
        # Scale to [0, 255] range
        output_np = ((output_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        # Handle case where all values are the same but we still want to write it
        # (shouldn't reach here due to empty patch check above)
        output_np = np.zeros_like(output_np, dtype=np.uint8)
    
    # Create output slice (all output channels for this spatial region)
    output_slice = (slice(None),) + spatial_slices
    
    # Write to output store
    output_store[output_slice] = output_np
    
    return {'chunk_idx': chunk_idx, 'processed_voxels': np.prod(output_data.shape)}


def finalize_logits(
    input_path: str,
    output_path: str,
    mode: str = "binary",  # "binary" or "multiclass"
    threshold: bool = False,  # If True, will apply argmax and only save class predictions
    delete_intermediates: bool = False,  # If True, will delete the input logits after processing
    chunk_size: tuple = None,  # Optional custom chunk size for output
    num_workers: int = None,  # Number of worker processes to use
    verbose: bool = True
):
    """
    Process merged logits and apply softmax/argmax to produce final outputs.
    
    Args:
        input_path: Path to the merged logits Zarr store
        output_path: Path for the finalized output Zarr store
        mode: "binary" (2 channels) or "multiclass" (>2 channels)
        threshold: If True, applies argmax and only saves class predictions
        delete_intermediates: Whether to delete input logits after processing
        chunk_size: Optional custom chunk size for output (Z,Y,X)
        num_workers: Number of worker processes to use for parallel processing
        verbose: Print progress messages
    """
    # Disable Blosc threading to avoid deadlocks when used with multiprocessing
    numcodecs.blosc.use_threads = False
    
    # Configure process pool size
    if num_workers is None:
        # Use half of CPU count (rounded up) to balance performance and memory usage
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes for parallel processing")
    
    # Setup compressor
    compressor = numcodecs.Blosc(
        cname='zstd',
        clevel=1,  # Light compression for performance
        shuffle=numcodecs.blosc.SHUFFLE
    )
    
    # Debug info
    print(f"Opening input logits: {input_path}")
    print(f"Mode: {mode}, Threshold flag: {threshold}")
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )
    
    # Get input shape and properties
    input_shape = input_store.shape
    num_classes = input_shape[0]
    spatial_shape = input_shape[1:]  # (Z, Y, X)
    
    # Verify we have the expected number of channels based on mode
    print(f"Input shape: {input_shape}, Num classes: {num_classes}")
    
    if mode == "binary" and num_classes != 2:
        raise ValueError(f"Binary mode expects 2 channels, but input has {num_classes} channels.")
    elif mode == "multiclass" and num_classes < 2:
        raise ValueError(f"Multiclass mode expects at least 2 channels, but input has {num_classes} channels.")
    
    # Use chunks from input if not specified
    if chunk_size is None:
        # Get chunks from input store if available
        try:
            # Zarr chunks are directly accessible as a property
            src_chunks = input_store.chunks
            # Input chunks include class dimension - extract spatial dimensions
            output_chunks = src_chunks[1:]
            if verbose:
                print(f"Using input chunk size: {output_chunks}")
        except:
            # Default to reasonable chunk size if not available
            output_chunks = (64, 64, 64)
            print(f"Could not determine input chunks, using default: {output_chunks}")
    else:
        output_chunks = chunk_size
        if verbose:
            print(f"Using specified chunk size: {output_chunks}")
    
    # Determine output shape based on mode and threshold
    if mode == "binary":
        if threshold:  # Now a boolean flag
            # If thresholding, only output argmax channel for binary
            output_shape = (1, *spatial_shape)  # Just the binary mask (argmax)
            print("Output will have 1 channel: [binary_mask]")
        else:
            # Just the softmax values
            output_shape = (1, *spatial_shape)  # Just softmax of FG class
            print("Output will have 1 channel: [softmax_fg]")
    else:  # multiclass
        if threshold:  # Now a boolean flag
            # If threshold is provided for multiclass, only save the argmax
            output_shape = (1, *spatial_shape)  # Just the argmax
            print("Output will have 1 channel: [argmax]")
        else:
            # For multiclass, we'll output num_classes channels (all softmax values)
            # Plus 1 channel for the argmax
            output_shape = (num_classes + 1, *spatial_shape)
            print(f"Output will have {num_classes + 1} channels: [softmax_c0...softmax_cN, argmax]")
    
    # Create output store
    print(f"Creating output store: {output_path}")
    output_chunks = (1, *output_chunks)  # Chunk each channel separately
    
    # Create output zarr array using our helper function
    output_store = open_zarr(
        path=output_path,
        mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose,
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.uint8,  # Use uint8 for the final outputs
        compressor=compressor,
        write_empty_chunks=False,  # Skip empty chunks for efficiency
        overwrite=True
    )
    
    # Function to calculate chunk indices
    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        # Skip first dimension (channels) as we'll handle all channels at once
        spatial_shape = shape[1:]  # Skip channel dimension
        spatial_chunks = chunks[1:]  # These are the spatial chunks (skip channel dimension)
        
        # Generate all combinations of chunk indices for spatial dimensions
        from itertools import product
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(spatial_shape, spatial_chunks)]
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        
        # Convert to list of dictionaries for parallel processing
        chunks_info = []
        for idx in chunk_indices:
            chunks_info.append({'indices': idx})
        
        return chunks_info
    
    # Get spatial chunk indices
    chunk_infos = get_chunk_indices(input_shape, output_chunks)
    total_chunks = len(chunk_infos)
    print(f"Processing data in {total_chunks} chunks using {num_workers} worker processes...")
    
    # Create a partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk,
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        threshold=threshold,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
        output_chunks=output_chunks
    )
    
    # Process chunks in parallel
    total_processed = 0
    empty_chunks = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunk_infos}
        
        # Use as_completed for better progress tracking
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(future_to_chunk),
            total=total_chunks,
            desc="Processing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result()
                if result.get('empty', False):
                    # Count empty chunks that were skipped
                    empty_chunks += 1
                else:
                    total_processed += result['processed_voxels']
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e
    
    print(f"\nOutput processing complete. Processed {total_chunks - empty_chunks} chunks, skipped {empty_chunks} empty chunks ({empty_chunks/total_chunks:.2%}).")
    
    # Copy metadata/attributes from input to output if they exist
    try:
        if hasattr(input_store, 'attrs') and hasattr(output_store, 'attrs'):
            for key in input_store.attrs:
                output_store.attrs[key] = input_store.attrs[key]
            # Add processing info to attributes
            output_store.attrs['processing_mode'] = mode
            output_store.attrs['threshold_applied'] = threshold
            output_store.attrs['empty_chunks_skipped'] = empty_chunks
            output_store.attrs['total_chunks'] = total_chunks
            output_store.attrs['empty_chunk_percentage'] = float(empty_chunks/total_chunks) if total_chunks > 0 else 0.0
    except Exception as e:
        print(f"Warning: Failed to copy metadata: {e}")
    
    # Clean up intermediate files if requested
    if delete_intermediates:
        print(f"Deleting intermediate logits: {input_path}")
        try:
            # Handle both local and remote paths (S3, etc.) using fsspec
            if input_path.startswith(('s3://', 'gs://', 'azure://')):
                # For remote storage, use fsspec's filesystem
                fs_protocol = input_path.split('://', 1)[0]
                fs = fsspec.filesystem(fs_protocol, anon=False if fs_protocol == 's3' else None)
                
                # Remove protocol prefix for fs operations
                path_no_prefix = input_path.split('://', 1)[1]
                
                if fs.exists(path_no_prefix):
                    fs.rm(path_no_prefix, recursive=True)
                    print(f"Successfully deleted intermediate logits (remote path)")
            elif os.path.exists(input_path):
                shutil.rmtree(input_path)
                print(f"Successfully deleted intermediate logits (local path)")
        except Exception as e:
            print(f"Warning: Failed to delete intermediate logits: {e}")
            print(f"You may need to delete them manually: {input_path}")
    
    print(f"Final output saved to: {output_path}")


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.finalize command."""
    parser = argparse.ArgumentParser(description='Process merged logits to produce final outputs.')
    parser.add_argument('input_path', type=str,
                      help='Path to the merged logits Zarr store')
    parser.add_argument('output_path', type=str,
                      help='Path for the finalized output Zarr store')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                      help='Processing mode. "binary" for 2-class segmentation, "multiclass" for >2 classes. Default: binary')
    parser.add_argument('--threshold', dest='threshold', action='store_true',
                      help='If set, applies argmax and only saves the class predictions (no probabilities). Works for both binary and multiclass.')
    parser.add_argument('--delete-intermediates', dest='delete_intermediates', action='store_true',
                      help='Delete intermediate logits after processing')
    parser.add_argument('--chunk-size', dest='chunk_size', type=str, default=None,
                      help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, input chunks will be used.')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                      help='Number of worker processes for parallel processing. Default: CPU_COUNT // 2')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                      help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Parse chunk_size if provided
    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")
    
    try:
        finalize_logits(
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            threshold=args.threshold,
            delete_intermediates=args.delete_intermediates,
            chunk_size=chunks,
            num_workers=args.num_workers,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"\n--- Finalization Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
