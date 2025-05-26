#!/usr/bin/env python3
"""
vesuvius.finalize: produce final outputs from merged logits or structure‐tensor field.
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

from data.utils import open_zarr


def process_chunk(chunk_info, input_path, output_path,
                  mode, threshold, num_classes,
                  spatial_shape, output_chunks):
    """
    Process one spatial chunk for binary or multiclass modes.
    """
    idx_tuple = chunk_info['indices']
    # spatial slices
    spatial_slices = tuple(
        slice(i * c, min((i+1)*c, dim))
        for i, c, dim in zip(idx_tuple, output_chunks[1:], spatial_shape)
    )

    # open stores
    inp = open_zarr(
        path=input_path, mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None
    )
    out = open_zarr(
        path=output_path, mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None
    )

    # read logits
    inp_slice = (slice(None),) + spatial_slices
    logits_np = inp[inp_slice]
    logits = torch.from_numpy(logits_np)

    # compute
    if mode == 'binary':
        sm = F.softmax(logits, dim=0)
        if threshold:
            out_t = (sm[1] > sm[0]).float().unsqueeze(0)
        else:
            out_t = sm[1].unsqueeze(0)
    else:  # multiclass
        sm = F.softmax(logits, dim=0)
        arg = torch.argmax(logits, dim=0).float().unsqueeze(0)
        if threshold:
            out_t = arg
        else:
            out_t = torch.cat([sm, arg], dim=0)

    out_np = out_t.numpy()
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
    out[out_slice] = out_np
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

def _compute_eigenvectors(block: torch.Tensor) -> torch.Tensor:
    """
    block: [6, dz, dy, dx]
    returns: [9, dz, dy, dx] float32
    """
    # 1) Flatten channels
    _, dz, dy, dx = block.shape
    N = dz * dy * dx
    x = block.view(6, N)                  # [6, N]
    jzz, jzy, jzx, jyy, jyx, jxx = x

    # 2) Vectorized build of M: [N,3,3]
    M = torch.stack([
        torch.stack([jzz, jzy, jzx], dim=1),  # row z
        torch.stack([jzy, jyy, jyx], dim=1),  # row y
        torch.stack([jzx, jyx, jxx], dim=1),  # row x
    ], dim=1)                              # [N,3,3]

    # 3) Sanitize any NaN/Inf
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) Micro-batch the eigen-decomposition to avoid CUSOLVER limits
    batch_size = 1024000  # TODO: fix this more intelligently
    w_chunks = [] # EIGENVALUES
    v_chunks = []
    for i in range(0, N, batch_size):
        wi, vi = torch.linalg.eigh(M[i:i+batch_size])  # wi:[B,3], vi:[B,3,3]
        w_chunks.append(wi) # EIGENVALUES
        v_chunks.append(vi)
    w = torch.cat(w_chunks, dim=0)      # [N,3] EIGENVALUES
    v = torch.cat(v_chunks, dim=0)      # [N,3,3]

    # 5) Make sure we are eigenvector-major: [N, eigenvector, component]
    v = v.permute(0, 2, 1).reshape(N, 9).permute(1, 0)  # was [N, comp, ev] → [N, ev, comp]
    eigvecs = v.view(9, dz, dy, dx)      # [9, dz, dy, dx]
    eigvals = w.permute(1, 0).view(3, dz, dy, dx)

    return eigvals, eigvecs

# Compile it once at import time:
_compute_eigenvectors = torch.compile(
    _compute_eigenvectors,
    mode="max-autotune-no-cudagraphs",
    fullgraph=True
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
        print(f"[Eigen] {len(chunks)} chunks to process")

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
        eigvals_block, eigvecs_block = _compute_eigenvectors(block)
        eigvals_block = eigvals_block.cpu().numpy()
        eigvecs_block = eigvecs_block.cpu().numpy()

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
    # disable internal Blosc threads
    numcodecs.blosc.use_threads = False

    if num_workers is None:
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

    # otherwise binary / multiclass
    inp = open_zarr(
        path=input_path, mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )
    shape = inp.shape
    C = shape[0]
    spatial = shape[1:]
    if verbose:
        print(f"Input shape: {shape}")

    if mode == 'binary' and C != 2:
        raise ValueError("Binary mode expects 2 channels")
    if mode == 'multiclass' and C < 2:
        raise ValueError("Multiclass expects >=2 channels")

    # determine chunks
    if chunk_size is None:
        sp_chunks = inp.chunks[1:]
        if verbose:
            print(f"Using input chunks: {sp_chunks}")
    else:
        sp_chunks = chunk_size
        if verbose:
            print(f"Using user chunks: {sp_chunks}")
    out_chunks = (1, *sp_chunks)

    # prepare output Zarr
    if mode == 'binary':
        out_C = 1
        desc = "binary"
    else:
        out_C = C + (0 if threshold else 1)
        desc = "multiclass"
    if verbose:
        print(f"Preparing {desc} output with {out_C} channel(s)")
    open_zarr(
        path=output_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose,
        shape=(out_C, *spatial),
        chunks=out_chunks,
        dtype=np.uint8,
        compressor=compressor,
        write_empty_chunks=False,
        overwrite=True
    )

    # chunk indices
    counts = [int(np.ceil(s/c)) for s,c in zip(spatial, sp_chunks)]
    from itertools import product
    infos = [{'indices': idx} for idx in product(*[range(n) for n in counts])]
    if verbose:
        print(f"{len(infos)} chunks to process")

    # parallel
    fn = partial(process_chunk, input_path=input_path, output_path=output_path,
                 mode=mode, threshold=threshold, num_classes=C,
                 spatial_shape=spatial, output_chunks=out_chunks)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(fn, info) for info in infos]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Finalizing", disable=not verbose):
            pass

    if delete_intermediates:
        if input_path.startswith('s3://'):
            fs = fsspec.filesystem(input_path.split('://')[0])
            fs.rm(input_path, recursive=True)
        else:
            shutil.rmtree(input_path, ignore_errors=True)

    if verbose:
        print(f"Final output saved to: {output_path}")


def main():
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

    chunks = None
    if args.chunk_size:
        try:
            cs = tuple(map(int, args.chunk_size.split(',')))
            if len(cs) != 3:
                raise ValueError()
            chunks = cs
        except:
            parser.error("Invalid --chunk-size; expected Z,Y,X")

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


if __name__ == '__main__':
    main()
