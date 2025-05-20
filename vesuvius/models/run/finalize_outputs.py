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


def finalize_logits(
    input_path: str,
    output_path: str,
    mode: str = "binary",  # "binary" or "multiclass"
    threshold: bool = False,  # If True, will apply argmax and only save class predictions
    delete_intermediates: bool = False,  # If True, will delete the input logits after processing
    chunk_size: tuple = None,  # Optional custom chunk size for output
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
        verbose: Print progress messages
    """
    # Setup compressor (similar to what's in blending.py)
    compressor = numcodecs.Blosc(
        cname='zstd',
        clevel=1,  # Light compression for performance
        shuffle=numcodecs.blosc.SHUFFLE
    )
    
    # Debug info
    print(f"Opening input logits: {input_path}")
    print(f"Mode: {mode}, Threshold flag: {threshold}")
    input_mapper = fsspec.get_mapper(input_path)
    input_store = zarr.open(input_mapper, mode='r')
    
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
    
    # Create output zarr array using fsspec mapper
    # Use proper authentication for S3 paths
    if output_path.startswith('s3://'):
        print(f"Detected S3 output path, using anon=False to use AWS credentials from environment")
        output_mapper = fsspec.get_mapper(output_path, anon=False)
    else:
        output_mapper = fsspec.get_mapper(output_path)
    output_store = zarr.create(
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.float16,  # Use float16 to match <f2 from tensorstore
        compressor=compressor,
        store=output_mapper,
        overwrite=True,
        write_empty_chunks=False  # Skip empty chunks for efficiency
    )
    
    # Process data in chunks to avoid memory issues
    # We'll process one spatial chunk at a time
    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        # Skip first dimension (channels) as we'll handle all channels at once
        spatial_shape = shape[1:]  # Skip channel dimension
        spatial_chunks = chunks[1:]  # These are the spatial chunks (skip channel dimension)
        
        # Generate all combinations of chunk indices for spatial dimensions
        from itertools import product
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(spatial_shape, spatial_chunks)]
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        return chunk_indices
    
    # Get spatial chunk indices
    spatial_chunk_indices = get_chunk_indices(input_shape, output_chunks)
    total_chunks = len(spatial_chunk_indices)
    print(f"Processing data in {total_chunks} chunks...")
    
    # Process each spatial chunk
    chunk_iterator = tqdm(spatial_chunk_indices, total=total_chunks, desc="Processing chunks")
    
    for chunk_idx in chunk_iterator:
        # Calculate slice for this chunk
        spatial_slices = tuple(
            slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
            for idx, chunk, shape_dim in zip(chunk_idx, output_chunks[1:], spatial_shape)
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
        
        # Convert back to numpy for writing with specific dtype to match output store
        output_np = output_data.numpy().astype(np.float16)
        
        # Create output slice (all output channels for this spatial region)
        output_slice = (slice(None),) + spatial_slices
        
        # Write to output store
        output_store[output_slice] = output_np
    
    print("\nOutput processing complete.")
    
    # Copy metadata/attributes from input to output if they exist
    try:
        if hasattr(input_store, 'attrs') and hasattr(output_store, 'attrs'):
            for key in input_store.attrs:
                output_store.attrs[key] = input_store.attrs[key]
            # Add processing info to attributes
            output_store.attrs['processing_mode'] = mode
            output_store.attrs['threshold_applied'] = threshold
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
                fs = fsspec.filesystem(fs_protocol)
                
                if fs.exists(input_path):
                    fs.rm(input_path, recursive=True)
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
