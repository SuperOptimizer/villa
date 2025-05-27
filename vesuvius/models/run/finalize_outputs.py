#!/usr/bin/env python3
"""
vesuvius.finalize_outputs: produce final outputs from merged logits or structure‐tensor field.
Supports binary, multiclass, and structure‐tensor eigenvector modes.
"""
import os
import shutil
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numcodecs
import fsspec

#from typing import Optional

from data.utils import open_zarr


def process_chunk(chunk_info, input_path, output_path,
                  mode, threshold, num_classes,
                  spatial_shape, output_chunks):
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

    out_np = output_data.numpy()

    # empty patch check
    if np.isclose(out_np, 0.5, rtol=1e-3).all():
        return {'empty': True}

    # scale to uint8
    mn, mx = out_np.min(), out_np.max()
    if mn < mx:
        out_np = ((out_np - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        out_np = np.zeros_like(out_np, dtype=np.uint8)

    # write back
    out_slice = (slice(None),) + spatial_slices
    output_store[out_slice] = out_np
    return {'processed': True}


class ChunkDataset(Dataset):
    """Dataset of spatial chunk bounds for structure‐tensor eigen decomposition."""
    def __init__(self, input_path, chunks, device):
        self.input_path = input_path
        self.chunks = chunks
        self.device = device

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # read block
        z0, z1, y0, y1, x0, x1 = self.chunks[idx]
        src = open_zarr(
            path=self.input_path, mode='r',
            storage_options={'anon': False} if self.input_path.startswith('s3://') else None
        )
        block_np = src[:, z0:z1, y0:y1, x0:x1].astype('float32')
        block = torch.from_numpy(block_np).to(self.device)  # [6, dz, dy, dx]
        return idx, (z0, z1, y0, y1, x0, x1), block

def _eigh_and_sanitize(M: torch.Tensor):
    w, v = torch.linalg.eigh(M) 
    # sanitize once
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return w, v

def _compute_eigenvectors(
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, dz, dy, dx = block.shape
    N = dz * dy * dx

    # build + sanitize
    x = block.view(6, N)
    M = torch.empty((N,3,3), dtype=block.dtype, device=block.device)
    M[:, 0, 0] = x[0]; M[:, 0, 1] = x[1]; M[:, 0, 2] = x[2]
    M[:, 1, 0] = x[1]; M[:, 1, 1] = x[3]; M[:, 1, 2] = x[4]
    M[:, 2, 0] = x[2]; M[:, 2, 1] = x[4]; M[:, 2, 2] = x[5]
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    zero_mask = M.abs().sum(dim=(1,2)) == 0

    batch_size = 1048576
    # eigen-decomp (either whole or in chunks)
    if batch_size is None or N <= batch_size:
        w, v = _eigh_and_sanitize(M)
    else:
        ws = []; vs = []
        for chunk in M.split(batch_size, dim=0):
            wi, vi = _eigh_and_sanitize(chunk)
            ws.append(wi); vs.append(vi)
        w = torch.cat(ws, 0)
        v = torch.cat(vs, 0)

    # zero out truly‐empty voxels without branching
    # zero_mask: [N], w: [N,3], v: [N,3,3]
    mask_w = zero_mask.unsqueeze(-1)             # [N,1]
    w = w.masked_fill(mask_w, 0.0)               # [N,3]
    mask_v = mask_w.unsqueeze(-1)                # [N,1,1]
    v = v.masked_fill(mask_v, 0.0)               # [N,3,3]

    # reshape back
    eigvals = w.transpose(0,1).view(3, dz, dy, dx)
    eigvecs = (
        v
        .permute(0,2,1)
        .reshape(N,9)
        .transpose(0,1)
        .view(9, dz, dy, dx)
    )
    return eigvals, eigvecs

_compute_eigenvectors = torch.compile(
    _compute_eigenvectors,
    mode="max-autotune-no-cudagraphs",
    fullgraph=True,
)

def _finalize_structure_tensor_torch(
    input_path, output_path, chunk_size, num_workers, compressor, verbose, swap_eigenvectors=False
):
    # open input
    src = open_zarr(
        path=input_path, mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )
    C, Z, Y, X = src.shape
    assert C == 6, f"Expect 6 channels, got {C}"

    # chunk dims
    if chunk_size is None:
        # src.chunks == (1, cz, cy, cx)
        cz, cy, cx = src.chunks[1:]
    else:
        cz, cy, cx = chunk_size
    if verbose:
        print(f"[Eigen] using chunks (dz,dy,dx)=({cz},{cy},{cx})")

    # prepare eigenvectors output zarr
    out_chunks = (1, cz, cy, cx)
    open_zarr(
        path=output_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose,
        shape=(9, Z, Y, X),
        chunks=out_chunks,
        compressor=compressor,
        dtype=np.float32,
        write_empty_chunks=False,
        overwrite=True
    )

    # prepare eigenvalue output zarr (3 channels)
    eigval_path = output_path.replace('.zarr', '_eigenvalues.zarr')
    open_zarr(
        path=eigval_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        shape=(3, Z, Y, X),
        chunks=out_chunks,
        compressor=compressor,
        dtype=np.float32,
        write_empty_chunks=False,
        overwrite=True
    )
    # build chunk list
    def gen_bounds():
        for z0 in range(0, Z, cz):
            for y0 in range(0, Y, cy):
                for x0 in range(0, X, cx):
                    yield (z0, min(z0+cz,Z),
                           y0, min(y0+cy,Y),
                           x0, min(x0+cx,X))
    chunks = list(gen_bounds())
    if verbose:
        print(f"[Eigen] {len(chunks)} chunks to solve the eigenvalue problem on")

    # 5) Dataset & DataLoader (with a no‐op collate so bounds come through untouched)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ChunkDataset(input_path, chunks, device)
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0]  # return the single sample directly
    )

    # 6) Process each chunk
    for idx, bounds, block in tqdm(loader, desc="[Eigen] Chunks", disable=not verbose):
        # idx: int, bounds: (z0,z1,y0,y1,x0,x1), block: tensor [1,6,dz,dy,dx]
        z0, z1, y0, y1, x0, x1 = bounds
        # squeeze off the dummy batch dim
        with torch.no_grad():
            eigvals_block_gpu, eigvecs_block_gpu = _compute_eigenvectors(block) #TODO: fix batch size intelligently
        eigvals_block = eigvals_block_gpu.cpu().numpy()
        eigvecs_block = eigvecs_block_gpu.cpu().numpy()

        del block, eigvals_block_gpu, eigvecs_block_gpu
        torch.cuda.empty_cache()

        if swap_eigenvectors:
            # 1) reshape eigenvectors into [3 eigenvectors, 3 components, dz,dy,dx]
            v = eigvecs_block.reshape(3, 3, *eigvecs_block.shape[1:])  # → [3,3,dz,dy,dx]
            # eigenvalues are already [3, dz,dy,dx]
            w = eigvals_block                                        # → [3,dz,dy,dx]

            # 2) swap eigenvector #0 <-> #1 and their eigenvalues
            #    this exchanges the *first* and *second* principal directions
            v[[0, 1], :, ...] = v[[1, 0], :, ...]
            w[[0, 1],    ...] = w[[1, 0],    ...]

            # 3) flatten back to your Zarr layout
            eigvecs_block = v.reshape(9, *eigvecs_block.shape[1:])
            eigvals_block  = w

        # write
        dst_vec = open_zarr(
            path=output_path, mode='r+',
            storage_options={'anon': False} if output_path.startswith('s3://') else None
        )
        dst_val = open_zarr(
            path=eigval_path, mode='r+',
            storage_options={'anon': False} if output_path.startswith('s3://') else None
        )
        dst_vec[:, z0:z1, y0:y1, x0:x1] = eigvecs_block
        dst_val[:, z0:z1, y0:y1, x0:x1] = eigvals_block

    if verbose:
        print(f"[Eigen] eigenvectors → {output_path}")
        print(f"[Eigen] eigenvalues  → {eigval_path}")


def finalize_logits(
    input_path, output_path,
    mode="binary", threshold=False,
    delete_intermediates=False,
    chunk_size=None, num_workers=None,
    swap_eigenvectors=False, verbose=True
):
    """
    Process merged logits and apply softmax/argmax to produce final outputs.
    If mode is 'structure-tensor', the appropriate finalization logic will be applied.
    
    Args:
        input_path: Path to the merged logits Zarr store
        output_path: Path for the finalized output Zarr store
        mode: "binary" (2 channels), "multiclass" (>2 channels) or "structure-tensor"
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
        num_workers = max(1, mp.cpu_count()//2)
    if verbose:
        print(f"Using {num_workers} worker(s)")

    compressor = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.blosc.SHUFFLE)

    # structure‐tensor branch
    if mode == 'structure-tensor':
        _finalize_structure_tensor_torch(
            input_path=input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            num_workers=num_workers,
            compressor=compressor,
            verbose=verbose,
            swap_eigenvectors=swap_eigenvectors
        )
        # now delete the intermediate tensor if requested
        if delete_intermediates:
            if input_path.startswith('s3://'):
                fs = fsspec.filesystem(input_path.split('://')[0], anon=False)
                fs.rm(input_path, recursive=True)
            else:
                shutil.rmtree(input_path, ignore_errors=True)
        return

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
    parser = argparse.ArgumentParser(description='Finalize merged logits or structure-tensor')
    parser.add_argument('input_path', help='Merged logits or 6-channel tensor Zarr')
    parser.add_argument('output_path', help='Final output Zarr path')
    parser.add_argument('--mode', choices=['binary','multiclass','structure-tensor'],
                        default='binary', help='Processing mode')
    parser.add_argument('--threshold', action='store_true',
                        help='Apply argmax only (no probabilities)')
    parser.add_argument('--delete-intermediates', action='store_true',
                        help='Remove input after processing')
    parser.add_argument('--chunk-size', default=None,
                        help='Spatial chunk size Z,Y,X (comma-separated)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes (default=CPU//2)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--swap-eigenvectors',
                        action='store_true',
                        help='(structure-tensor horizontal) swap first↔second eigenvectors before writing')

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
            swap_eigenvectors=args.swap_eigenvectors,
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
